#pragma once

#include <algorithm>
#include <cstdio>
#include <vector>

#include "asw.cuh"
#include "cuda_utils.h"
#include "linalg/vec.h"
#include "multigrid/solvers/solve_utils.h"

template <typename T, class Assembler>
class UnstructuredQuadElementAdditiveSchwarzSmoother : public BaseSolver {
    // additive schwarz smoother for unstructured 1st order elements (any mesh)
    // performs local smoothing on each 4-node element (if tris need to write new version of this)
   public:
    static constexpr int nodes_per_elem = 4;  // num nodes per element

    // Constructor: fill specifies how many fill iterations to perform.
    // This implementation assumes T is double.
    UnstructuredQuadElementAdditiveSchwarzSmoother(cublasHandle_t &cublasHandle_,
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

        size = (int)sqrt(nodes_per_elem);
        size2 = size * size;
        size4 = size2 * size2;

        nodes_per_elem2 = nodes_per_elem * nodes_per_elem;
        block_dim2 = block_dim * block_dim;

        n = nodes_per_elem * block_dim;  // Block dimension (default leads to 24x24 matrices)
        int num_elements = assembler.get_num_elements();
        batchSize = num_elements;
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
        const int n_rhs_vals = n_rhs_blocks * block_dim;

        for (int iter = 0; iter < iters; iter++) {
            // (1) Collect the defect RHS vectors for each block into d_Xarray.
            dim3 grid((n_rhs_vals + 31) / 32);
            k_copyRHSIntoBatched<T><<<grid, 32>>>(n_rhs_vals, block_dim, size, d_RHSblockMap,
                                                  d_defect.getPtr(), d_Xarray);

            // (2) Batched matrix–vector multiplication: for each block, compute Y_i = invA_i * X_i.
            const double alpha = 1.0;
            const double beta = 0.0;
            CHECK_CUBLAS(cublasDgemmBatched(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, 1, n, &alpha,
                                            (const double **)d_invAarray, n,
                                            (const double **)d_Xarray, n, &beta, d_Yarray, n,
                                            batchSize));

            // (3) Scatter the batched solution stored in d_Yarray into the global 'temp' vector.
            cudaMemset(d_temp, 0.0, N * sizeof(T));
            k_copyBatchedIntoSoln_additive<T>
                <<<grid, 32>>>(n_rhs_vals, block_dim, size, d_RHSblockMap, d_Yarray, d_temp);

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
        // call kernel to copy assembled kmat values to batched locations
        block_dim2 = block_dim * block_dim;
        int n_batch_vals = n_batch_blocks * block_dim2;
        dim3 grid((n_batch_vals + 31) / 32);
        k_copyMatValuesToBatched<T>
            <<<grid, 32>>>(n_batch_vals, block_dim, size, d_blockInds, d_kmat_vals, d_Aarray);
    }

    // Performs batched LU factorization followed by explicit matrix inversion.
    void _schwarzFactorization() {
        // DEBUG: check matrices before factorization
        // if (nx <= 5) debugPrintBatchedMatrices("Before getrfBatched");

        CHECK_CUBLAS(cublasDgetrfBatched(cublasHandle, n, d_Aarray, n, d_PivotArray, d_InfoArray,
                                         batchSize));

        // DEBUG: check LU result
        // if (nx <= 5) debugPrintBatchedMatrices("After getrfBatched");

        CHECK_CUBLAS(cublasDgetriBatched(cublasHandle, n, (const double **)d_Aarray, n,
                                         d_PivotArray, d_invAarray, n, d_InfoArray, batchSize));

        // DEBUG: check inverse
        // if (nx <= 5) debugPrintBatchedMatrices("After getriBatched", batchSize);
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
    int n_rhs_batch_blocks;
    int n_batch_blocks, kmat_nnzb;
    int *h_kmat_rowp, *h_kmat_cols;
    int *d_kmat_rowp, *d_kmat_rows, *d_kmat_cols;
    int *h_blockInds;  //, *h_rowInds, *h_colInds;
    int *d_blockInds;  //, *d_rowInds, *d_colInds;
    int *h_RHSblockMap, *d_RHSblockMap;

    void _computeNZPatterns() {
        // compute nonzero patterns for the copying of the matrix kmat into batched form
        // for a structured plate or cylinder grid in lexigraphic order

        h_kmat_rowp = DeviceVec<int>(nnodes + 1, d_kmat_rowp).createHostVec().getPtr();
        h_kmat_cols = DeviceVec<int>(kmat_nnzb, d_kmat_cols).createHostVec().getPtr();

        // copying batchSize * size2 blocks from original matrix into batched matrix
        // need to compute nz pattern + map to facilitate the copy process
        n_batch_blocks = batchSize * nodes_per_elem2;  // number of mat blocks to handle
        h_blockInds =
            new int[n_batch_blocks];  // block ind of kmat for each of batchSize * n * n values
        memset(h_blockInds, 0, n_batch_blocks * sizeof(int));
        // h_rowInds = new int[n_batch_blocks];
        // memset(h_rowInds, 0, n_batch_blocks * sizeof(int));
        // h_colInds = new int[n_batch_blocks];
        // memset(h_colInds, 0, n_batch_blocks * sizeof(int));

        // loop over each batch / coupled group
        int *elem_conn = h_elem_conn.getPtr();
        for (int ibatch = 0; ibatch < batchSize; ibatch++) {
            int ielem = ibatch;  // equivalent for this unstructured schwarz smoother

            for (int ij = 0; ij < nodes_per_elem2; ij++) {
                int i = ij % nodes_per_elem;
                int j = ij / nodes_per_elem;

                int row_node = elem_conn[nodes_per_elem * ielem + i];
                int col_node = elem_conn[nodes_per_elem * ielem + j];
                int _jp = -1;
                for (int jp = h_kmat_rowp[row_node]; jp < h_kmat_rowp[row_node + 1]; jp++) {
                    if (h_kmat_cols[jp] == col_node) {
                        _jp = jp;
                        break;
                    }
                }

                if (_jp != -1) {
                    // flattened three tensor
                    int batch_block_ind = nodes_per_elem2 * ibatch + ij;
                    h_blockInds[batch_block_ind] = _jp;
                }
            }
        }

        // now copy host to device pointers
        d_blockInds = HostVec<int>(n_batch_blocks, h_blockInds).createDeviceVec().getPtr();

        // ==================================================
        /* now also compute the RHS block map */
        // ==================================================

        n_rhs_batch_blocks = batchSize * nodes_per_elem;  // number of rhs blocks to handle
        h_RHSblockMap = new int[n_rhs_batch_blocks];      // block ind of kmat for each of batchSize
                                                          // * n * n values
        memset(h_RHSblockMap, 0, n_rhs_batch_blocks * sizeof(int));

        for (int ibatch = 0; ibatch < batchSize; ibatch++) {
            int ielem = ibatch;

            // loop over batch nodes for each-node
            for (int i = 0; i < nodes_per_elem; i++) {
                int inode = elem_conn[nodes_per_elem * ielem + i];
                int batch_block_ind = nodes_per_elem * ibatch + i;
                h_RHSblockMap[batch_block_ind] = inode;
            }
        }

        // now copy host to device pointers
        d_RHSblockMap = HostVec<int>(n_rhs_batch_blocks, h_RHSblockMap).createDeviceVec().getPtr();
    }
};