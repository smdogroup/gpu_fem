#pragma once

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

#include "asw.cuh"
#include "cuda_utils.h"
#include "linalg/vec.h"
#include "multigrid/solvers/solve_utils.h"

template <typename T, class Assembler, bool TEMP_ELEM3x3 = false>
class UnstructuredQuadSupportAdditiveSchwarzSmoother : public BaseSolver {
    // additive schwarz smoother for unstructured 1st order elements (any mesh)
    // uses 3x3 node-support blocks (up to 9 nodes per subdomain) as local smoothing
   public:
    // static constexpr int nodes_per_elem = 4;  // num nodes per element

    // Constructor: fill specifies how many fill iterations to perform.
    // This implementation assumes T is double.
    UnstructuredQuadSupportAdditiveSchwarzSmoother(cublasHandle_t &cublasHandle_,
                                                   cusparseHandle_t &cusparseHandle_,
                                                   Assembler &assembler_,
                                                   BsrMat<DeviceVec<T>> kmat_, T omega_ = 0.25,
                                                   int iters_ = 5)
        : cublasHandle(cublasHandle_),
          cusparseHandle(cusparseHandle_),
          assembler(assembler_),
          kmat(kmat_) {
        // Retrieve problem dimensions from the assembler.
        block_dim = assembler_.getBsrData().block_dim;
        N = assembler_.get_num_vars();
        nnodes = N / block_dim;
        // get data out of kmat
        auto d_kmat_bsr_data = kmat.getBsrData();
        d_kmat_vals = kmat.getVec().getPtr();
        d_kmat_rowp = d_kmat_bsr_data.rowp;
        d_kmat_rows = d_kmat_bsr_data.rows;
        d_kmat_cols = d_kmat_bsr_data.cols;
        kmat_nnzb = d_kmat_bsr_data.nnzb;
        d_elem_conn = assembler.getConn();
        h_elem_conn = d_elem_conn.createHostVec();

        // printf("omega = %.4e\n", omega);
        omega = omega_;
        iters = iters_;

        size = 3;               // 3 nodes per direc
        size2 = size * size;    // 3x3 = 9 nodes in subdomain
        size4 = size2 * size2;  // 9 x 9 = 81 blocks in subdomain kmat

        // nodes_per_elem2 = nodes_per_elem * nodes_per_elem;
        block_dim2 = block_dim * block_dim;

        n = size2 * block_dim;  // Block dimension (default leads to 24x24 matrices)
        int num_nodes = assembler.get_num_nodes();
        batchSize = num_nodes;
        // printf("batchSize = %d\n", batchSize);

        static_assert(std::is_same<T, double>::value, "ASW smoother currently requires T=double");

        // Allocate batched device memory for pointers and individual blocks.
        // printf("1 - allocate batched memory\n");
        _allocateBatchedMemory();
        // debugCheckPointerListOnly("after allocation (before fill kernel)");
        // printf("2 - compute Schwarz NZ patterns\n");
        _computeNZPatterns();

        // Compute the Schwarz factorization during construction.
        // printf("3 - initCuda\n");
        _initCuda();
    }

    void factor() {
        // printf("4 - copy matrix values to batched memory\n");
        _copyMatrixValuesToBatched();
        // printf("5 - compute the local Schwarz matrix inverses\n");
        _schwarzFactorization();
    }

    void update_after_assembly(DeviceVec<T> &vars) { factor(); }
    void set_abs_tol(T atol) {}
    void set_rel_tol(T atol) {}
    int get_num_iterations() { return 0; }
    void set_print(bool print) {}
    void free() {}  // TBD on this one
    void set_cycle_type(std::string cycle_) {}

    void set_temp_elem3x3(int nxe_) { nxe = nxe_; }

    T precond_complexity() {
        // get [nnzb(precond) + nnzb(A)] / nnzb(A)
        int precond_nnzb = batchSize * size4;
        return (precond_nnzb + kmat_nnzb) * 1.0 / kmat_nnzb;
    }

    // Applies the Schwarz smoother.
    // First, each block's right-hand side should be collected into d_Xarray.
    // Then, the solution is computed via a batched GEMM.
    bool solve(DeviceVec<T> rhs, DeviceVec<T> soln, bool check_conv = false) {
        // setup rhs and soln with init guess of 0
        cudaMemcpy(d_rhs, rhs.getPtr(), N * sizeof(T), cudaMemcpyDeviceToDevice);
        cudaMemset(d_inner_soln, 0, N * sizeof(T));  // re-zero the solution

        // call smoother on the defect=rhs and solution pair
        this->smoothDefect(d_rhs_vec, d_inner_soln_vec, iters);

        // copy internal soln to external solution of the solve method
        cudaMemcpy(soln.getPtr(), d_inner_soln, N * sizeof(T), cudaMemcpyDeviceToDevice);
        return false;  // fail = False
    }

    void smoothDefect(DeviceVec<T> d_defect, DeviceVec<T> d_soln, int __n_iters, bool print = false,
                      int print_freq = 10) {
        const int n_rhs_blocks = batchSize * size2;
        const int n_rhs_vals = n_rhs_blocks * block_dim;  // includes nz subdomain
        const int n_rhs_nz_vals = nnz_rhs_batch * block_dim;

        for (int iter = 0; iter < iters; iter++) {
            // (0) zero the rhs and soln vecs
            dim3 grid1((n_rhs_vals + 31) / 32);
            k_zeroSubdomainVecs_support<T><<<grid1, 32>>>(n_rhs_vals, block_dim, size, d_Xarray);
            k_zeroSubdomainVecs_support<T><<<grid1, 32>>>(n_rhs_vals, block_dim, size, d_Yarray);

            // (1) Collect the defect RHS vectors for each block into d_Xarray.
            dim3 grid((n_rhs_nz_vals + 31) / 32);
            k_copyRHSIntoBatched_support<T><<<grid, 32>>>(n_rhs_nz_vals, block_dim, size,
                                                          d_rhsDenseMap, d_rhsSDMap,
                                                          d_defect.getPtr(), d_Xarray);

            // (2) Batched matrix–vector multiplication: for each block, compute Y_i = invA_i * X_i.
            const double alpha = 1.0;
            const double beta = 0.0;
            CHECK_CUBLAS(cublasDgemmBatched(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, 1, n, &alpha,
                                            (const double **)d_invAarray, n,
                                            (const double **)d_Xarray, n, &beta, d_Yarray, n,
                                            batchSize));

            // temporarily run element 3x3 for structured mesh problem (temporarily in this class)
            // cause it has better routines for 3x3 subdomains (when 3x3 is not originally dense in
            // nofill pattern)
            if constexpr (TEMP_ELEM3x3) {
                // hack to make equiv to 3x3-elem ASW on a structured domain
                k_zeroBatchedSolnOnElemBoundaries<T><<<grid, 32>>>(
                    nxe, n_rhs_nz_vals, block_dim, size, d_rhsDenseMap, d_rhsSDMap, d_Yarray);
            }

            // (3) Scatter the batched solution stored in d_Yarray into the global 'temp' vector.
            cudaMemset(d_temp, 0.0, N * sizeof(T));
            k_copyBatchedIntoSoln_additiveSupport<T><<<grid, 32>>>(
                n_rhs_nz_vals, block_dim, size, d_rhsDenseMap, d_rhsSDMap, d_Yarray, d_temp);

            // 4) compute defect update after new solution term..
            //     ..(with soln change stored in d_temp)
            T a = -omega, b = 1.0;  // with omega * d_temp update
            CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE, nnodes, nnodes,
                                          kmat_nnzb, &a, descrKmat, d_kmat_vals, d_kmat_rowp,
                                          d_kmat_cols, block_dim, d_temp, &b, d_defect.getPtr()));
            // also update d_soln += omega * d_temp
            a = omega;
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_temp, 1, d_soln.getPtr(), 1));
        }
    }

   private:
    // References to CUDA library handles.
    cublasHandle_t &cublasHandle;
    cusparseHandle_t &cusparseHandle;
    cusparseMatDescr_t descrKmat = 0;

    // Problem data from assembler.
    Assembler assembler;
    int N, block_dim, nnodes;
    int block_dim2;
    int size, size2, size4;
    BsrMat<DeviceVec<T>> kmat;
    T *d_kmat_vals;
    T omega;  // scalar update for additive schwarz smoother

    // updated vectors
    DeviceVec<T> d_temp_vec, d_rhs_vec, d_inner_soln_vec;
    T *d_temp, *d_temp2, *d_resid;
    T *d_rhs, *d_inner_soln;

    // Block and batch sizes for batched operations.
    int n;          // Block dimension (e.g., 24 for 24x24 blocks)
    int batchSize;  // Number of block matrices in the batch
    int ncx, ncy;   // Number of coupling groups / batches in each direction

    int nx;  // for elem3x3 ASW hack

    // Device pointer arrays for batched routines.
    T **h_Aarray, **d_Aarray;        // Pointers to LU-factorized 24x24 matrices.
    T **h_invAarray, **d_invAarray;  // Pointers to computed inverses of the 24x24 blocks.
    T **h_Xarray, **d_Xarray;        // Pointers to 24x1 input vectors (RHS for local solves).
    T **h_Yarray, **d_Yarray;        // Pointers to 24x1 output vectors (local solutions).

    // Device arrays for pivoting and info.
    int *d_PivotArray;
    int *d_InfoArray;

    void _initCuda() {
        // init some util vecs
        d_temp_vec = DeviceVec<T>(N);
        d_temp = d_temp_vec.getPtr();
        d_temp2 = DeviceVec<T>(N).getPtr();
        d_resid = DeviceVec<T>(N).getPtr();

        // for linear solver / precond use
        d_rhs_vec = DeviceVec<T>(N);
        d_rhs = d_rhs_vec.getPtr();
        d_inner_soln_vec = DeviceVec<T>(N);
        d_inner_soln = d_inner_soln_vec.getPtr();

        // make mat handles for SpMV
        CHECK_CUSPARSE(cusparseCreateMatDescr(&descrKmat));
        CHECK_CUSPARSE(cusparseSetMatType(descrKmat, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descrKmat, CUSPARSE_INDEX_BASE_ZERO));
    }

    void _copyMatrixValuesToBatched() {
        // zero then set all subdomains to be identity
        block_dim2 = block_dim * block_dim;
        int n_subdomain_vals = batchSize * size4 * block_dim2;
        dim3 grid1((n_subdomain_vals + 31) / 32);
        k_setSubdomainMatricesToIdentity<T>
            <<<grid1, 32>>>(n_subdomain_vals, block_dim, size, d_Aarray);

        // call kernel to copy assembled kmat values to batched locations
        int n_batch_vals = nnz_batch_blocks * block_dim2;
        dim3 grid2((n_batch_vals + 31) / 32);
        k_copyMatValuesToBatched_support<T><<<grid2, 32>>>(
            n_batch_vals, block_dim, size, d_kmatBlockInds, d_sdBlockInds, d_kmat_vals, d_Aarray);
    }

    // Performs batched LU factorization followed by explicit matrix inversion.
    void _schwarzFactorization() {
        // DEBUG: check matrices before factorization
        // if (nx <= 5) debugPrintBatchedMatrices("Before getrfBatched");
        // debugPrintBatchedMatrices("Before getrfBatched");

        CHECK_CUBLAS(cublasDgetrfBatched(cublasHandle, n, d_Aarray, n, d_PivotArray, d_InfoArray,
                                         batchSize));

        // DEBUG: check LU result
        // if (nx <= 5) debugPrintBatchedMatrices("After getrfBatched");
        // debugPrintBatchedMatrices("After getrfBatched");

        CHECK_CUBLAS(cublasDgetriBatched(cublasHandle, n, (const double **)d_Aarray, n,
                                         d_PivotArray, d_invAarray, n, d_InfoArray, batchSize));

        // DEBUG: check inverse
        // if (nx <= 5) debugPrintBatchedMatrices("After getriBatched", batchSize);
        // debugPrintBatchedMatrices("After getriBatched", batchSize);
    }

    void debugPrintBatchedMatrices(const char *tag, int maxBlocks = 4) {
        printf("\n=== DEBUG: %s ===\n", tag);

        // 1) Copy device pointer list -> host
        std::vector<double *> h_Aptr(batchSize);
        CHECK_CUDA(cudaMemcpy(h_Aptr.data(), d_Aarray, batchSize * sizeof(double *),
                              cudaMemcpyDeviceToHost));

        // Sanity print pointer values
        for (int b = 0; b < std::min(batchSize, maxBlocks); b++) {
            printf("Block %d device ptr = %p\n", b, (void *)h_Aptr[b]);
        }

        // 2) Copy and print each small matrix
        std::vector<double> h_mat(n * n);

        for (int b = 0; b < std::min(batchSize, maxBlocks); b++) {
            CHECK_CUDA(cudaMemcpy(h_mat.data(), h_Aptr[b], n * n * sizeof(double),
                                  cudaMemcpyDeviceToHost));

            printf("A[%d] =\n", b);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    printf("%10.3e ", h_mat[i + j * n]);  // column-major
                }
                printf("\n");
            }
            printf("\n");
        }

        fflush(stdout);
    }

    void debugCheckPointerListOnly(const char *tag, int maxBlocks = 8) {
        printf("\n=== PTR DEBUG: %s ===\n", tag);
        CHECK_CUDA(cudaDeviceSynchronize());  // should be clean here

        std::vector<double *> h_Aptr(batchSize, nullptr);
        CHECK_CUDA(cudaMemcpy(h_Aptr.data(), d_Aarray, (size_t)batchSize * sizeof(double *),
                              cudaMemcpyDeviceToHost));

        for (int b = 0; b < std::min(batchSize, maxBlocks); b++) {
            printf("d_Aarray[%d] = %p\n", b, (void *)h_Aptr[b]);
        }
    }

    // Allocates memory for batched pointers and the corresponding individual blocks.
    void _allocateBatchedMemory() {
        // printf("\tallocate batched memory with %d batches of small (%d,%d) matrices\n",
        // batchSize,
        //        n, n);

        const size_t matrixBytes = (size_t)n * (size_t)n * sizeof(double);
        const size_t vectorBytes = (size_t)n * sizeof(double);

        // --- 1) Allocate HOST arrays-of-pointers (CPU memory) ---
        // (these hold device pointers, but the arrays themselves live on the host)
        h_Aarray = (double **)malloc((size_t)batchSize * sizeof(double *));
        h_invAarray = (double **)malloc((size_t)batchSize * sizeof(double *));
        h_Xarray = (double **)malloc((size_t)batchSize * sizeof(double *));
        h_Yarray = (double **)malloc((size_t)batchSize * sizeof(double *));

        if (!h_Aarray || !h_invAarray || !h_Xarray || !h_Yarray) {
            fprintf(stderr, "malloc failed for host pointer arrays\n");
            abort();
        }

        // --- 2) Allocate each matrix/vector on DEVICE, store pointers in host arrays ---
        for (int i = 0; i < batchSize; i++) {
            CHECK_CUDA(cudaMalloc((void **)&h_Aarray[i], matrixBytes));
            CHECK_CUDA(cudaMalloc((void **)&h_invAarray[i], matrixBytes));
            CHECK_CUDA(cudaMalloc((void **)&h_Xarray[i], vectorBytes));
            CHECK_CUDA(cudaMalloc((void **)&h_Yarray[i], vectorBytes));
        }

        // --- 3) Allocate DEVICE arrays-of-pointers (GPU memory) ---
        CHECK_CUDA(cudaMalloc((void **)&d_Aarray, (size_t)batchSize * sizeof(double *)));
        CHECK_CUDA(cudaMalloc((void **)&d_invAarray, (size_t)batchSize * sizeof(double *)));
        CHECK_CUDA(cudaMalloc((void **)&d_Xarray, (size_t)batchSize * sizeof(double *)));
        CHECK_CUDA(cudaMalloc((void **)&d_Yarray, (size_t)batchSize * sizeof(double *)));

        // --- 4) Copy pointer lists HOST -> DEVICE (tiny copy: batchSize*8 bytes each) ---
        CHECK_CUDA(cudaMemcpy(d_Aarray, h_Aarray, (size_t)batchSize * sizeof(double *),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_invAarray, h_invAarray, (size_t)batchSize * sizeof(double *),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_Xarray, h_Xarray, (size_t)batchSize * sizeof(double *),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_Yarray, h_Yarray, (size_t)batchSize * sizeof(double *),
                              cudaMemcpyHostToDevice));

        // --- 5) Pivot/info arrays (DEVICE) ---
        CHECK_CUDA(cudaMalloc((void **)&d_PivotArray, (size_t)batchSize * (size_t)n * sizeof(int)));
        CHECK_CUDA(cudaMalloc((void **)&d_InfoArray, (size_t)batchSize * sizeof(int)));
    }

    int iters;  // num smoothing iterations
    DeviceVec<int> d_elem_conn;
    HostVec<int> h_elem_conn;
    int nodes_per_elem2;
    int nnz_rhs_batch;
    int nnz_batch_blocks, kmat_nnzb;
    int *h_kmat_rowp, *h_kmat_cols;
    int *d_kmat_rowp, *d_kmat_rows, *d_kmat_cols;
    int *h_sdBlockInds, *h_kmatBlockInds;
    int *d_sdBlockInds, *d_kmatBlockInds;
    int *h_rhsSDMap, *h_rhsDenseMap;
    int *d_rhsSDMap, *d_rhsDenseMap;

    int nxe;

    void _computeNZPatterns() {
        // compute nonzero patterns for the copying of the matrix kmat into batched form
        // uses node support and nofill sparsity therefore

        h_kmat_rowp = DeviceVec<int>(nnodes + 1, d_kmat_rowp).createHostVec().getPtr();
        h_kmat_cols = DeviceVec<int>(kmat_nnzb, d_kmat_cols).createHostVec().getPtr();

        // first compute the number of batch blocks needed (total across all subdomains)
        // loop over each batch / coupled group
        nnz_batch_blocks = 0;
        int *n_supp_nodes = new int[batchSize];
        memset(n_supp_nodes, 0, batchSize * sizeof(int));
        for (int ibatch = 0; ibatch < batchSize; ibatch++) {
            int _node = ibatch;  // equivalent for this unstructured schwarz smoother

            std::vector<int> supp_nodes;
            for (int jp = h_kmat_rowp[_node]; jp < h_kmat_rowp[_node + 1]; jp++) {
                int loc_jp = jp - h_kmat_rowp[_node];
                if (loc_jp >= size2) break;  // cap at 9 support nodes
                int jnode = h_kmat_cols[jp];
                supp_nodes.push_back(jnode);
                n_supp_nodes[_node]++;
            }

            // now compute the block inds of support
            for (int inode : supp_nodes) {
                for (int jnode : supp_nodes) {
                    // check if (inode, jnode) in kmat sparsity (if so increment total number of
                    // batch blocks across all subdomains
                    for (int jp = h_kmat_rowp[inode]; jp < h_kmat_rowp[inode + 1]; jp++) {
                        int knode = h_kmat_cols[jp];
                        if (jnode == knode) {
                            // then the (inode, jnode) pair is in the nofill sparsity
                            nnz_batch_blocks++;
                            break;
                        }
                    }
                }
            }
        }

        // then store the support nodes

        // loop over each batch block and now store the batch blocks
        h_sdBlockInds = new int[nnz_batch_blocks];
        memset(h_sdBlockInds, 0, nnz_batch_blocks * sizeof(int));
        h_kmatBlockInds = new int[nnz_batch_blocks];
        memset(h_kmatBlockInds, 0, nnz_batch_blocks * sizeof(int));
        int inz_batch = 0;
        for (int ibatch = 0; ibatch < batchSize; ibatch++) {
            int _node = ibatch;  // equivalent for this unstructured schwarz smoother

            std::vector<int> supp_nodes;
            for (int jp = h_kmat_rowp[_node]; jp < h_kmat_rowp[_node + 1]; jp++) {
                int loc_jp = jp - h_kmat_rowp[_node];
                if (loc_jp >= size2) break;  // cap at 9 support nodes
                int jnode = h_kmat_cols[jp];
                supp_nodes.push_back(jnode);
                // n_supp_nodes[_node]++;
            }
            int n_supp = n_supp_nodes[_node];
            // printf("subdomain %d with nodes\n", _node);
            // printVec<int>(supp_nodes.size(), supp_nodes.data());

            // now compute the block inds of support
            for (int i = 0; i < n_supp; i++) {
                int inode = supp_nodes[i];
                for (int j = 0; j < n_supp; j++) {
                    int jnode = supp_nodes[j];
                    // check if (inode, jnode) in kmat sparsity (if so increment total number of
                    // batch blocks across all subdomains
                    for (int jp = h_kmat_rowp[inode]; jp < h_kmat_rowp[inode + 1]; jp++) {
                        int knode = h_kmat_cols[jp];
                        if (jnode == knode) {
                            // then the (inode, jnode) pair is in the nofill sparsity
                            // compute block ind in 9x9 max block subdomain storage (not same as
                            // unique NZ blocks)
                            int subdomain_block = size4 * ibatch + size2 * j + i;
                            h_sdBlockInds[inz_batch] = subdomain_block;
                            h_kmatBlockInds[inz_batch] = jp;
                            // printf(
                            //     "subdomain %d with node (%d,%d) and sd block ind %d + global
                            //     block " "ind %d\n", _node, inode, jnode, subdomain_block, jp);
                            inz_batch++;
                        }
                    }
                }
            }
        }

        // now copy host to device pointers
        // printf("h_sdBlockInds: ");
        // printVec<int>(nnz_batch_blocks, h_sdBlockInds);
        // printf("h_sdBlockInds: ");
        // printVec<int>(nnz_batch_blocks, h_kmatBlockInds);
        d_sdBlockInds = HostVec<int>(nnz_batch_blocks, h_sdBlockInds).createDeviceVec().getPtr();
        d_kmatBlockInds =
            HostVec<int>(nnz_batch_blocks, h_kmatBlockInds).createDeviceVec().getPtr();

        // ==================================================
        /* now also compute the RHS block map */
        // ==================================================

        // num supp nodes total is the nnz rhs
        nnz_rhs_batch = 0;
        for (int inode = 0; inode < batchSize; inode++) {
            nnz_rhs_batch += n_supp_nodes[inode];
        }

        h_rhsSDMap = new int[nnz_rhs_batch];
        h_rhsDenseMap = new int[nnz_rhs_batch];
        // * n * n values
        memset(h_rhsSDMap, 0, nnz_rhs_batch * sizeof(int));
        memset(h_rhsDenseMap, 0, nnz_rhs_batch * sizeof(int));
        int inz_rhs = 0;

        for (int ibatch = 0; ibatch < batchSize; ibatch++) {
            int _node = ibatch;  // equivalent for this unstructured schwarz smoother

            std::vector<int> supp_nodes;
            for (int jp = h_kmat_rowp[_node]; jp < h_kmat_rowp[_node + 1]; jp++) {
                int loc_jp = jp - h_kmat_rowp[_node];
                if (loc_jp >= size2) break;  // cap at 9 support nodes
                int jnode = h_kmat_cols[jp];
                supp_nodes.push_back(jnode);
                // n_supp_nodes[_node]++;
            }
            int n_supp = n_supp_nodes[_node];

            // printf("subdomain %d with supp nodes: ", ibatch);
            // printVec<int>(n_supp, supp_nodes.data());

            // now assign subdomain and dense locations for each nz in subdomain batch
            for (int i = 0; i < n_supp; i++) {
                int inode = supp_nodes[i];
                h_rhsSDMap[inz_rhs] = size2 * ibatch + i;
                h_rhsDenseMap[inz_rhs] = inode;
                // printf("subdomain-rhs %d with glob-node %d and loc-sd node %d\n", ibatch, inode,
                //        size2 * ibatch + i);
                inz_rhs++;
            }
        }

        // now copy host to device pointers
        d_rhsSDMap = HostVec<int>(nnz_rhs_batch, h_rhsSDMap).createDeviceVec().getPtr();
        d_rhsDenseMap = HostVec<int>(nnz_rhs_batch, h_rhsDenseMap).createDeviceVec().getPtr();
    }
};