#pragma once
#include "linalg/vec.h"

template <class Assembler>
class LexigraphicGaussSeidelSmoother {
    /* a lexigraphic gauss seidel smoother */
    using T = Assembler::T;

    LexigraphicGaussSeidelSmoother(Assembler &assembler_, BsrMat<DeviceVec<T>> Kmat_) {
        Kmat = Kmat_;
        d_rhs = d_rhs_;
        h_color_rowp = h_color_rowp_;
        block_dim = 6;
        N = assembler_.get_num_vars();
        nnodes = N / 6;
        assembler = assembler_;

        // get data out of kmat
        auto d_kmat_bsr_data = Kmat.getBsrData();
        d_kmat_vals = Kmat.getVec().getPtr();
        d_kmat_rowp = d_kmat_bsr_data.rowp;
        d_kmat_cols = d_kmat_bsr_data.cols;
        kmat_nnzb = d_kmat_bsr_data.nnzb;

        initCuda();
        const bool startup = true;
        initLowerMatForGaussSeidel<startup>();
    }

    void update_assembly() {
        const bool startup = false;
        initLowerMatForGaussSeidel<startup>();
    }
    void factor() {}

    void initCuda() {
        // init handles
        CHECK_CUBLAS(cublasCreate(&cublasHandle));
        CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

        // init some util vecs
        d_temp_vec = DeviceVec<T>(N);
        d_temp = d_temp_vec.getPtr();

        // make mat handles for SpMV
        CHECK_CUSPARSE(cusparseCreateMatDescr(&descrKmat));
        CHECK_CUSPARSE(cusparseSetMatType(descrKmat, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descrKmat, CUSPARSE_INDEX_BASE_ZERO));

        CHECK_CUSPARSE(cusparseCreateMatDescr(&descrDinvMat));
        CHECK_CUSPARSE(cusparseSetMatType(descrDinvMat, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descrDinvMat, CUSPARSE_INDEX_BASE_ZERO));
    }

    template <bool startup = false>
    void initLowerMatForGaussSeidel() { /* init L+D matrix for lexigraphic or RCM Gauss-seidel */

        if constexpr (startup) {
            // init kmat descriptor for L+D matrix (no ilu0 factor, this is just the matrix itself
            // nofill)
            cusparseCreateMatDescr(&descr_kmat_L);
            cusparseSetMatIndexBase(descr_kmat_L, CUSPARSE_INDEX_BASE_ZERO);
            cusparseSetMatType(descr_kmat_L, CUSPARSE_MATRIX_TYPE_GENERAL);
            cusparseSetMatFillMode(descr_kmat_L, CUSPARSE_FILL_MODE_LOWER);
            cusparseSetMatDiagType(descr_kmat_L,
                                   CUSPARSE_DIAG_TYPE_NON_UNIT);  // includes diag here..
            cusparseCreateBsrsv2Info(&info_kmat_L);

            // get buffer size
            int pbufferSize;
            CHECK_CUSPARSE(cusparseDbsrsv2_bufferSize(
                cusparseHandle, dir, trans_L, nnodes, kmat_nnzb, descr_kmat_L, d_kmat_vals,
                d_kmat_rowp, d_kmat_cols, block_dim, info_kmat_L, &pbufferSize));
            cudaMalloc(&kmat_pBuffer, pbufferSize);
        }  // end of startup part

        // compute symbolic analysis for efficient triangular solves
        CHECK_CUSPARSE(cusparseDbsrsv2_analysis(cusparseHandle, dir, trans_L, nnodes, kmat_nnzb,
                                                descr_kmat_L, d_kmat_vals, d_kmat_rowp, d_kmat_cols,
                                                block_dim, info_kmat_L, policy_L, kmat_pBuffer));
        // CHECK_CUDA(cudaDeviceSynchronize());
    }

    void smoothDefect(DeviceVec<T> d_defect, DeviceVec<T> d_soln, int n_iters, bool print = false,
                      int print_freq = 10) {
        // this is lexigraphic or RCM GS (RCM if more general mesh..)
        T a, b;

        for (int iter = 0; iter < n_iters; iter++) {
            // 1) (L+D)*dx = defect with triang solve
            const double alpha = 1.0;
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                cusparseHandle, dir, trans_L, nnodes, kmat_nnzb, &alpha, descr_kmat_L, d_kmat_vals,
                d_kmat_rowp, d_kmat_cols, block_dim, info_kmat_L, d_defect.getPtr(), d_temp,
                policy_L, kmat_pBuffer));  // prob only need U^-1 part for block diag.. TBD

            // 2) update d_soln += d_temp (aka dx)
            a = 1.0;
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_temp, 1, d_soln.getPtr(), 1));

            // 3) compute new defect = prev_defect - A * dx
            a = -1.0,
            b = 1.0;  // so that defect := defect - mat*vec
            CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE, nnodes, nnodes,
                                          kmat_nnzb, &a, descrKmat, d_kmat_vals, d_kmat_rowp,
                                          d_kmat_cols, block_dim, d_temp, &b, d_defect.getPtr()));

            /* report progress of defect nrm if printing.. */
            T defect_nrm;
            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_defect.getPtr(), 1, &defect_nrm));
            if (print && iter % print_freq == 0)
                printf("\tLX-BGS %d/%d : ||defect|| = %.4e\n", iter + 1, n_iters, defect_nrm);

        }  // next block-GS iteration
    }

    // standard matrix data
    Assembler assembler;
    int N, nelems, block_dim, nnodes;
    BsrMat<DeviceVec<T>> Kmat;  // can't get Dinv_mat directly at moment
    DeviceVec<T> d_temp_vec;
    T *d_temp;

    // CUSPARSE and cublas data
    cusparseHandle_t cusparseHandle = NULL;
    cublasHandle_t cublasHandle = NULL;
    cusparseMatDescr_t descr_L = 0;
    bsrsv2Info_t info_L = 0;
    void *pBuffer = 0;
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;

    // for kmat
    int kmat_nnzb, *d_kmat_rowp, *d_kmat_cols;
    T *d_kmat_vals, *d_kmat_lu_vals;
    cusparseMatDescr_t descrKmat = 0;
    size_t bufferSizeMV;
    void *buffer_MV = nullptr;

    cusparseMatDescr_t descr_kmat_L = 0;
    bsrsv2Info_t info_kmat_L = 0;
    void *kmat_pBuffer = 0;
};