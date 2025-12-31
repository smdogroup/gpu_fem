// gpu_fem utils
#include <cusparse_v2.h>
#include <thrust/device_vector.h>

#include "cublas_v2.h"
#include "cuda_utils.h"
#include "linalg/vec.h"
#include "pde.cuh"

template <typename T>
class PoissonSolver {
    /* a CUDA GPU solver for the 2D poisson problem with multigrid capabilities as well.. */
    using I = unsigned long long int;  // large int datatype
    static constexpr int nodes_per_elem = 4;

   public:
    PoissonSolver() = default;  // so you can make pointers..

    PoissonSolver(int nxe_, bool red_black_order_ = false)
        : nxe(nxe_), red_black_order(red_black_order_) {
        nx = nxe + 1;
        N = nx * nx;
        nelems = nxe * nxe;
        dx = 1.0 / (nx - 1);  // element dx or mesh size metric

        assert(nxe % 2 == 0);

        // compute the nnz pattern of the LHS
        buildCSRPattern();

        N_half = (N + 1) / 2;  // for red-black GS

        // then assemble LHS and RHS also..
        assembleLHS();
        assembleRHS();

        // init soln and defects
        d_soln = DeviceVec<T>(N);
        resetSoln();
        getTrueSoln();  // get true soln for reference (method of manafactured soln)

        // other init steps
        init_cuda();
        computeDefectFromRHS();
    }

    void init_cuda() {
        // init cusparse and cublas handleshandles
        CHECK_CUBLAS(cublasCreate(&cublasHandle));
        CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

        // init dense vecs
        d_temp = DeviceVec<T>(N).getPtr();
        d_temp_half = DeviceVec<T>(N_half).getPtr();
        d_resid = DeviceVec<T>(N).getPtr();
        d_defect = DeviceVec<T>(N);
        cusparseCreateDnVec(&vecB, N, d_rhs.getPtr(), CUDA_R_32F);
        cusparseCreateDnVec(&vecTMP, N, d_temp, CUDA_R_32F);
        cusparseCreateDnVec(&vecHalf, N_half, d_temp_half, CUDA_R_32F);
        cusparseCreateDnVec(&vecX, N, d_soln.getPtr(), CUDA_R_32F);
        cusparseCreateDnVec(&vecR, N, d_resid, CUDA_R_32F);
        cusparseCreateDnVec(&vecD, N, d_defect.getPtr(), CUDA_R_32F);

        CHECK_CUSPARSE(cusparseCreateDnVec(&vecTMP, N, d_temp, CUDA_R_32F));
        CHECK_CUSPARSE(cusparseCreateDnVec(&vecHalf, N_half, d_temp_half, CUDA_R_32F));
        CHECK_CUSPARSE(cusparseCreateDnVec(&vecR, N, d_resid, CUDA_R_32F));

        // init A matrix
        CHECK_CUSPARSE(cusparseCreateCsr(&matA, N, N, csr_nnz, d_csr_rowp.getPtr(),
                                         d_csr_cols.getPtr(), d_lhs.getPtr(), CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

        // init spMV buffer for CSR
        T floatone = 1.0, floatzero = 0.0;
        CHECK_CUSPARSE(cusparseSpMV_bufferSize(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone, matA, vecTMP, &floatzero,
            vecR, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSizeMV));
        CHECK_CUDA(cudaMalloc(&buffer_MV, bufferSizeMV));
        // this buffer size should also be sufficient for the red and black matrices I believe (they
        // are smaller..)

        // if (red_black_order) {
        //     int N_half = N / 2;
        //     int csr_nnz_half = csr_nnz / 2;

        //     // also initialize A_red and A_black
        //     CHECK_CUSPARSE(cusparseCreateCsr(&matA_red, N_half, N, csr_nnz_half,
        //                                      d_csr_rowp.getPtr(), d_csr_cols.getPtr(),
        //                                      d_lhs.getPtr(), CUSPARSE_INDEX_32I,
        //                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
        //                                      CUDA_R_32F));

        //     // A_black
        //     CHECK_CUSPARSE(cusparseCreateCsr(&matA_black, N_half, N, csr_nnz_half,
        //                                      &d_csr_rowp.getPtr()[N_half], d_csr_cols.getPtr(),
        //                                      d_lhs.getPtr(), CUSPARSE_INDEX_32I,
        //                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
        //                                      CUDA_R_32F));
        // }
    }

    void free() {
        // TBD all of these steps
        cusparseDestroyDnVec(vecB);
        cusparseDestroyDnVec(vecTMP);
        cusparseDestroyDnVec(vecX);
        cusparseDestroyDnVec(vecR);
        cusparseDestroyDnVec(vecD);

        cusparseDestroy(cusparseHandle);
        cublasDestroy(cublasHandle);
    }

    void assembleLHS() {
        // build the lhs or CSR matrix
        int nblocks = (csr_nnz + 31) / 32;
        dim3 block(32);
        dim3 grid(nblocks, 16);

        d_lhs = DeviceVec<T>(csr_nnz);
        k_assembleLHS<T><<<grid, block>>>(nxe, csr_nnz, nelems, d_csr_rows.getPtr(),
                                          d_csr_cols.getPtr(), dx, d_lhs.getPtr());

        k_applyBCsLHS<T><<<grid, block>>>(nx, N, csr_nnz, d_iperm.getPtr(), d_csr_rows.getPtr(),
                                          d_csr_cols.getPtr(), d_lhs.getPtr());

        // get diag inv for jacobi
        d_diag_inv = DeviceVec<T>(N);
        dim3 grid2((N + 31) / 32);
        k_get_diag_inv<T><<<grid2, block>>>(N, d_csr_rowp.getPtr(), d_csr_cols.getPtr(),
                                            d_lhs.getPtr(), d_diag_inv.getPtr());
    }

    void assembleRHS() {
        // build the rhs for exponential load
        int nblocks = (N + 31) / 32;
        dim3 block(32);
        dim3 grid(nblocks, 16);

        d_rhs = DeviceVec<T>(N);
        k_assembleRHS<T><<<grid, block>>>(nx, N, dx, d_iperm.getPtr(), d_rhs.getPtr());
    }

    void computeDefectFromRHS() {
        // compute the current defect from soln
        getResidNorm();  // copy resid => defect
        cudaMemcpy(d_defect.getPtr(), d_resid, N * sizeof(T), cudaMemcpyDeviceToDevice);
    }

    void buildCSRPattern() {
        /* compute the CSR matrix nnz pattern on the GPU (with nofill) */
        // TODO : can lattern extend some of these methods to compute nnz patterns on GPU with TACS

        // init cuda grid and perm pointers
        dim3 block(32), grid0((N + 31) / 32);
        d_perm = DeviceVec<int>(N), d_iperm = DeviceVec<int>(N);

        // get perm vector
        if (red_black_order) {
            k_getRedBlackPerms<<<grid0, block>>>(N, d_perm.getPtr(), d_iperm.getPtr());
        } else {
            k_getNaturalPerm<<<grid0, block>>>(N, d_perm.getPtr(), d_iperm.getPtr());
        }

        // printf("d_perm : ");
        // int *h_perm = d_perm.createHostVec().getPtr();
        // printVec<int>(min(N, 50), h_perm);
        // printf("d_iperm : ");
        // int *h_iperm = d_iperm.createHostVec().getPtr();
        // printVec<int>(min(N, 50), h_iperm);
        // return;  // temp

        // 1) get total row counts (non-unique)
        auto d_csr_NUNIQ_row_counts = DeviceVec<int>(N);
        int nblocks = (nelems + 31) / 32;
        dim3 grid(nblocks, 16);
        k_csrNUNIQRowCounts<I><<<grid, block>>>(red_black_order, nxe, nelems, d_perm.getPtr(),
                                                d_csr_NUNIQ_row_counts.getPtr());
        // NOTE : realized after the fact => some of this CSR pattern stuff maybe shouldn't have
        // used elements, should have just gone by nodes, but whatever..

        // DEBUG
        // printf("h_rowCounts (not unique): ");
        // int *h_rowCounts1 = d_csr_NUNIQ_row_counts.createHostVec().getPtr();
        // printVec<int>(min(N, 50), h_rowCounts1);
        // return;  // temp

        // 2) get max num row counts (non-unique)
        auto d_maxRowCount1 = DeviceVec<int>(1);
        dim3 grid3((N + 31) / 32);
        k_csrMaxRowCount<<<grid3, block>>>(N, d_csr_NUNIQ_row_counts.getPtr(),
                                           d_maxRowCount1.getPtr());
        int maxRowCount1 = d_maxRowCount1.createHostVec().getPtr()[0];
        d_maxRowCount1.free();
        d_csr_NUNIQ_row_counts.free();

        // printf("max row count (non-unique) = %d\n", maxRowCount1);
        // return;

        // 3) get row counts (unique)
        auto d_csr_row_counts = DeviceVec<int>(N);
        auto d_minConnectedNode_in = DeviceVec<int>(N);
        auto d_minConnectedNode_out = DeviceVec<int>(N);
        CHECK_CUDA(cudaMemset(d_minConnectedNode_in.getPtr(), -1, N * sizeof(int)));
        // printf("N = %d\n", N);

        auto d_csr_nnz = DeviceVec<I>(1);

        for (int i_minNode = 0; i_minNode < maxRowCount1; i_minNode++) {
            // reset to N so min actually works (cudaMemset doesn't work with large #s)
            k_vecset<<<grid3, block>>>(N, N, d_minConnectedNode_out.getPtr());

            // get the ith min node connected to this nodal row (loops over each element..)
            k_getMinConnectedNode<<<grid, block>>>(red_black_order, nxe, N, nelems, d_perm.getPtr(),
                                                   d_minConnectedNode_in.getPtr(),
                                                   d_minConnectedNode_out.getPtr());

            // add this ith min connected node to the row (until no more to add)
            k_uniqueRowCount<I><<<grid3, block>>>(nx, N, d_minConnectedNode_out.getPtr(),
                                                  d_csr_row_counts.getPtr(), d_csr_nnz.getPtr());

            // eliminates race conditiions with two arrays
            cudaMemcpy(d_minConnectedNode_in.getPtr(), d_minConnectedNode_out.getPtr(),
                       N * sizeof(int), cudaMemcpyDeviceToDevice);
        }
        csr_nnz = d_csr_nnz.createHostVec().getPtr()[0];
        d_csr_nnz.free();

        // DEBUG
        // printf("h_rowCounts (unique): ");
        // int *h_rowCounts2 = d_csr_row_counts.createHostVec().getPtr();
        // printVec<int>(min(N, 50), h_rowCounts2);
        // return;

        // 4) go from row counts to row ptr
        d_csr_rowp = DeviceVec<int>(N + 1);
        thrustPrefixScan(d_csr_row_counts.getPtr());

        // // DEBUG
        // printf("h_rowp: ");
        // int *h_rowp = d_csr_rowp.createHostVec().getPtr();
        // printVec<int>(min(N + 1, 100), h_rowp);
        // return;

        // 5) get max row count (unique row counts)
        auto d_maxRowCount2 = DeviceVec<int>(1);
        k_csrMaxRowCount<<<grid3, block>>>(N, d_csr_row_counts.getPtr(), d_maxRowCount2.getPtr());
        int maxRowCount2 = d_maxRowCount2.createHostVec().getPtr()[0];
        d_maxRowCount2.free();
        d_csr_row_counts.free();

        // // DEBUG
        // printf("max row count unique = %d\n", maxRowCount2);
        // printf("csr nnz = %d\n", (int)csr_nnz);
        // return;

        // 6) get colptr
        d_csr_cols = DeviceVec<int>(csr_nnz);
        d_csr_rows = DeviceVec<int>(csr_nnz);  // this rows vector is useful also for bcs later..
        cudaMemset(d_minConnectedNode_in.getPtr(), -1, N * sizeof(int));  // reset
        for (int i_minNode = 0; i_minNode < maxRowCount2; i_minNode++) {
            // reset to N so min actually works
            k_vecset<<<grid3, block>>>(N, N, d_minConnectedNode_out.getPtr());

            // get the ith min node connected to this nodal row (loops over each element..)
            k_getMinConnectedNode<<<grid, block>>>(red_black_order, nxe, N, nelems, d_perm.getPtr(),
                                                   d_minConnectedNode_in.getPtr(),
                                                   d_minConnectedNode_out.getPtr());

            // DEBUG
            // printf("h_minConnNodeOut step %d: ", i_minNode);
            // int *h_minConnNodeOut = d_minConnectedNode_out.createHostVec().getPtr();
            // printVec<int>(100, h_minConnNodeOut);

            // add this ith min connected node to the row (until no more to add)
            k_addMinConnectedNodeRows<<<grid3, block>>>(i_minNode, N, d_csr_rowp.getPtr(),
                                                        d_csr_rows.getPtr());

            k_addMinConnectedNodeCols<<<grid3, block>>>(i_minNode, N,
                                                        d_minConnectedNode_out.getPtr(),
                                                        d_csr_rowp.getPtr(), d_csr_cols.getPtr());

            // DEBUG
            // printf("h_csr_rows step %d: ", i_minNode);
            // int *h_csr_rows = d_csr_rows.createHostVec().getPtr();
            // printVec<int>(100, h_csr_rows);

            // eliminates race conditiions with two arrays
            cudaMemcpy(d_minConnectedNode_in.getPtr(), d_minConnectedNode_out.getPtr(),
                       N * sizeof(int), cudaMemcpyDeviceToDevice);
        }
        d_minConnectedNode_in.free();
        d_minConnectedNode_out.free();

        // DEBUG
        // printf("h_cols: ");
        // int *h_cols = d_csr_cols.createHostVec().getPtr();
        // // printVec<int>(csr_nnz, h_cols);
        // printVec<int>(min((int)csr_nnz, 100), h_cols);
    }

    void getTrueSoln() {
        // get the true soln exp(x * y)
        d_true_soln = DeviceVec<T>(N);

        dim3 block(32);
        int nblocks = (N + block.x - 1) / block.x;
        dim3 grid(nblocks);
        k_getTrueSoln<T><<<grid, block>>>(nx, N, dx, d_iperm.getPtr(), d_true_soln.getPtr());
    }

    T getSolnError() {
        // get the soln error of the discrete soln (for mesh conv and verification)
        // d_temp = d_soln - d_true_soln
        CHECK_CUDA(cudaMemcpy(d_temp, d_soln.getPtr(), N * sizeof(T), cudaMemcpyDeviceToDevice));
        T a = -1.0;
        CHECK_CUBLAS(cublasSaxpy(cublasHandle, N, &a, d_true_soln.getPtr(), 1, d_temp, 1));

        // then get error norm
        T err_nrm;
        CHECK_CUBLAS(cublasSnrm2(cublasHandle, N, d_temp, 1, &err_nrm));

        // NOTE : the different treatment of bcs is leading to slight errors near bndry
        // may have to remove bcs from the unknowns to get same good conv results as python, we'll
        // see (but I did fix the bcs, no column bcs for NZ dirichlet..)

        // could also be float vs double?

        return err_nrm;
    }

    T getDefectNorm() {
        T def_nrm;
        CHECK_CUBLAS(cublasSnrm2(cublasHandle, N, d_defect.getPtr(), 1, &def_nrm));
        return def_nrm;
    }

    T getResidNorm() {
        // get the residual nrm of the linear system R = b - Ax
        // reset resid to zero
        cudaMemset(d_resid, 0.0, N * sizeof(T));

        // d_temp = -Ax
        T float_mone = -1.0, floatzero = 0.0;
        CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &float_mone,
                                    matA, vecX, &floatzero, vecR, CUDA_R_32F,
                                    CUSPARSE_SPMV_ALG_DEFAULT, buffer_MV));

        // d_temp += b
        T a = 1.0;
        CHECK_CUBLAS(cublasSaxpy(cublasHandle, N, &a, d_rhs.getPtr(), 1, d_resid, 1));

        // then get resid norm
        T res_nrm;
        CHECK_CUBLAS(cublasSnrm2(cublasHandle, N, d_resid, 1, &res_nrm));

        return res_nrm;
    }

    void resetSoln() {
        // init soln to zero but nz on bndry
        dim3 block(32);
        dim3 grid((N + 31) / 32);
        k_initSoln<T><<<grid, block>>>(nx, N, dx, d_iperm.getPtr(), d_soln.getPtr());
    }

    void resetDefect() {
        // zero the defect
        cudaMemset(d_defect.getPtr(), 0.0, N * sizeof(T));
    }

    void dampedJacobiSolve(int n_iters, T omega = 2.0 / 3.0, bool print = false,
                           int print_freq = 10) {
        // do damped jacobi iteration (for multigrid smoothing or solve)
        for (int iter = 0; iter < n_iters; iter++) {
            // compute dx = Dinv * (b - A * x_prev)

            // first d_resid = -A * x
            T floatnegone = -1.0, floatzero = 0.0;
            CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &floatnegone, matA, vecX, &floatzero, vecR, CUDA_R_32F,
                                        CUSPARSE_SPMV_ALG_DEFAULT, buffer_MV));

            // then d_resid = b + d_resid = b - A * x
            T a = 1.0;
            CHECK_CUBLAS(cublasSaxpy(cublasHandle, N, &a, d_rhs.getPtr(), 1, d_resid, 1));

            // compute resid nrm
            T res_nrm;
            CHECK_CUBLAS(cublasSnrm2(cublasHandle, N, d_resid, 1, &res_nrm));
            if (print && iter % print_freq == 0)
                printf("\tDampJac %d/%d : ||resid|| = %.4e\n", iter + 1, n_iters, res_nrm);

            // then Dinv * d_resid => d_temp
            dim3 block(32), grid((N + 31) / 32);
            k_diag_inv_vec<T><<<grid, block>>>(N, d_diag_inv.getPtr(), d_resid, d_temp);

            // then x += omega * d_temp
            a = omega;
            CHECK_CUBLAS(cublasSaxpy(cublasHandle, N, &a, d_temp, 1, d_soln.getPtr(), 1));
        }
    }

    void dampedJacobiDefect(int n_iters, T omega = 2.0 / 3.0, bool print = false,
                            int print_freq = 10) {
        // do damped jacobi iteration (for multigrid smoothing or solve)
        // this time on defect though
        for (int iter = 0; iter < n_iters; iter++) {
            // dx = Dinv * defect
            dim3 block(32), grid((N + 31) / 32);
            k_diag_inv_vec<T><<<grid, block>>>(N, d_diag_inv.getPtr(), d_defect.getPtr(), d_temp);

            // x += omega * dx
            T a = omega;
            CHECK_CUBLAS(cublasSaxpy(cublasHandle, N, &a, d_temp, 1, d_soln.getPtr(), 1));

            // defect -= A * omega * dx
            T nomega = -omega, floatone = 1.0;
            CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &nomega,
                                        matA, vecTMP, &floatone, vecD, CUDA_R_32F,
                                        CUSPARSE_SPMV_ALG_DEFAULT, buffer_MV));

            // compute defect nrm
            T defect_nrm;
            CHECK_CUBLAS(cublasSnrm2(cublasHandle, N, d_defect.getPtr(), 1, &defect_nrm));
            if (print && iter % print_freq == 0)
                printf("\tDampJac %d/%d : ||defect|| = %.4e\n", iter + 1, n_iters, defect_nrm);
        }
    }

    void redBlackGaussSeidelDefect(int n_iters, bool print = false, int print_freq = 10) {
        // do red black gauss-seidel on the defect
        // here red are even nodes and black are odd nodes from lexigraphic (or natural order)
        // uses 2 full-size matrix-vec products on each iter (twice cost of damped jacobi, but much
        // more powerful smoother often about 3x less ovr cost and less tuning needed)

        T init_defect_nrm;
        CHECK_CUBLAS(cublasSnrm2(cublasHandle, N, d_defect.getPtr(), 1, &init_defect_nrm));
        if (print) printf("Red-Black GS init defect nrm = %.4e\n", init_defect_nrm);

        for (int iter = 0; iter < n_iters; iter++) {
            /* 1) D_red * dx_red = defect_red */

            // defect_red => d_temp_half (half size)
            cudaMemcpy(d_temp_half, d_defect.getPtr(), N_half * sizeof(T),
                       cudaMemcpyDeviceToDevice);

            // Dinv * d_temp_half => d_temp, so that d_temp [dx_red, 0] now
            cudaMemset(d_temp, 0.0, N * sizeof(T));
            dim3 block(32), grid((N_half + 31) / 32);
            k_diag_inv_vec<T><<<grid, block>>>(N_half, d_diag_inv.getPtr(), d_temp_half, d_temp);

            /* 2) now update soln and defect from dx = [dx_red, 0] red update */
            T a = 1.0;
            CHECK_CUBLAS(cublasSaxpy(cublasHandle, N, &a, d_temp, 1, d_soln.getPtr(), 1));
            T float_mone = -1.0, float_one = 1.0;
            CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &float_mone, matA, vecTMP, &float_one, vecD, CUDA_R_32F,
                                        CUSPARSE_SPMV_ALG_DEFAULT, buffer_MV));

            /* 3) D_black * dx_black = defect_black */

            // defect_black => d_temp_half (half size) // since lengths N_half and N_half-1..
            cudaMemcpy(d_temp_half, &d_defect.getPtr()[N_half], (N_half - 1) * sizeof(T),
                       cudaMemcpyDeviceToDevice);

            // Dinv * d_temp_half = Dinv * [0, defect_black] => d_temp, so that d_temp [0, dx_black]
            cudaMemset(d_temp, 0.0, N * sizeof(T));
            k_diag_inv_vec<T><<<grid, block>>>(N_half - 1, &d_diag_inv.getPtr()[N_half],
                                               d_temp_half, &d_temp[N_half]);

            /* 4) now update soln and defect from dx = [0, dx_black] black update */
            a = 1.0;
            CHECK_CUBLAS(cublasSaxpy(cublasHandle, N, &a, d_temp, 1, d_soln.getPtr(), 1));
            CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &float_mone, matA, vecTMP, &float_one, vecD, CUDA_R_32F,
                                        CUSPARSE_SPMV_ALG_DEFAULT, buffer_MV));

            /* 5) report progress of defect nrm if printing.. */
            T defect_nrm;
            CHECK_CUBLAS(cublasSnrm2(cublasHandle, N, d_defect.getPtr(), 1, &defect_nrm));
            if (print && iter % print_freq == 0)
                printf("\tRB-GaussSeidel %d/%d : ||defect|| = %.4e\n", iter + 1, n_iters,
                       defect_nrm);
        }
    }

    void prolongate(DeviceVec<int> d_coarse_iperm, DeviceVec<T> coarse_soln_in) {
        // transfer a coarser mesh to this fine mesh

        // zero temp so we can store dx in it
        cudaMemset(d_temp, 0.0, N * sizeof(T));

        // launch kernel so coalesced with every group of N_coarse threads covering whole domain
        // then repeats with extra group of grids (9x for 9x adjacent fine nodes of FD stencil)
        dim3 block(32);
        int Nc = N / 4;
        int nblocks_x = (Nc + 31) / 32;
        dim3 grid(nblocks_x, 9);

        k_coarse_fine<T><<<grid, block>>>(nxe, N, d_perm.getPtr(), d_coarse_iperm.getPtr(),
                                          coarse_soln_in.getPtr(), d_temp);

        // add coarse-fine dx into soln
        T a = 1.0;
        CHECK_CUBLAS(cublasSaxpy(cublasHandle, N, &a, d_temp, 1, d_soln.getPtr(), 1));
        // defect -= A * dx
        T floatnegone = -1.0, floatone = 1.0;
        CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatnegone,
                                    matA, vecTMP, &floatone, vecD, CUDA_R_32F,
                                    CUSPARSE_SPMV_ALG_DEFAULT, buffer_MV));
    }

    void restrict_defect(DeviceVec<int> d_fine_perm, DeviceVec<T> fine_defect_in) {
        // transfer from finer mesh to this coarse mesh (on defect)
        // first reset defect on this coarser grid and soln
        resetDefect();
        resetSoln();

        // launch kernel so coalesced with every group of N_coarse threads covering whole domain
        // then repeats with extra group of grids (9x for 9x adjacent fine nodes of FD stencil)
        dim3 block(32);
        int nblocks_x = (N + 31) / 32;
        dim3 grid(nblocks_x, 9);

        k_fine_coarse<T><<<grid, block>>>(nxe, N, d_fine_perm.getPtr(), d_iperm.getPtr(),
                                          fine_defect_in.getPtr(), d_defect.getPtr());

        // temp debug defect
        // T grid1_def_nrm;
        // CHECK_CUBLAS(cublasSnrm2(cublasHandle, N * 4, fine_defect_in.getPtr(), 1,
        // &grid1_def_nrm)); T grid2_def_nrm = getDefectNorm(); printf("\t\trestrict defect |fine|
        // %.2e => |coarse| %.2e\n", grid1_def_nrm, grid2_def_nrm);
    }

    // public data
    int nxe, nx, N, nelems, N_half;
    I csr_nnz;
    DeviceVec<int> d_csr_rowp, d_csr_cols, d_node_min_elems, d_csr_rows;
    DeviceVec<int> d_perm, d_iperm;  // perm gives old to new node, iperm is flipped
    T dx;
    DeviceVec<T> d_lhs, d_rhs, d_soln, d_true_soln, d_diag_inv, d_defect;
    T *d_temp, *d_temp_half, *d_resid;
    bool red_black_order;

   private:
    /* other helper methods (non device) */
    void thrustPrefixScan(int *_d_csr_row_counts) {
        // use thrust here to do prefix scan, there's also some interesting steps to get efficient
        // Bleloch Scan kernel on GPU (with shared mem + bank conflicts)
        // https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
        // see this on how to handle CUDA shared mem banks,
        // https://leimao.github.io/blog/CUDA-Shared-Memory-Bank/

        thrust::device_vector<int> d_counts(N);
        thrust::copy_n(thrust::device_pointer_cast(_d_csr_row_counts), N, d_counts.begin());
        thrust::device_vector<int> d_rowp2(N + 1);
        d_rowp2[0] = 0;
        thrust::inclusive_scan(d_counts.begin(), d_counts.end(), d_rowp2.begin() + 1);

        cudaMemcpy(d_csr_rowp.getPtr(), thrust::raw_pointer_cast(d_rowp2.data()),
                   (N + 1) * sizeof(int), cudaMemcpyDeviceToDevice);
    }

    // private data
    cublasHandle_t cublasHandle = NULL;
    cusparseHandle_t cusparseHandle = NULL;
    cusparseMatDescr_t descrA = 0;
    cusparseDnVecDescr_t vecX, vecB, vecR, vecTMP, vecD, vecHalf;
    cusparseSpMatDescr_t matA = NULL;  //, matA_red = NULL, matA_black = NULL;
    size_t bufferSizeMV;
    void *buffer_MV = nullptr;
};