#pragma once
#include "../solve_utils.h"

// cusparse directLU solves with multigrid (V-cycle style preconditioner for two levels)
// only works from coarsest grid to next level

template <typename T, class Assembler, bool MULTI_SMOOTH = false>
class CusparseMGDirectLU : public BaseSolver {
   public:
    CusparseMGDirectLU(cublasHandle_t &cublasHandle_, cusparseHandle_t &cusparseHandle_,
                       Assembler &assembler_, BsrMat<DeviceVec<T>> &kmat_, T omega_ = 1.0,
                       int n_iters_ = 1)
        : cublasHandle(cublasHandle_), cusparseHandle(cusparseHandle_) {
        /* create the cusparse direct solver (for repeated solves) */

        assembler = assembler_;
        kmat = kmat_;
        n_iters = n_iters_;
        omega = omega_;

        BsrData bsr_data = kmat.getBsrData();
        N = assembler.get_num_vars();
        mb = bsr_data.nnodes;
        nnzb = bsr_data.nnzb;
        nnz = kmat.get_nnz();
        block_dim = bsr_data.block_dim;
        d_rowp = bsr_data.rowp;
        d_cols = bsr_data.cols;
        d_vals = kmat.getVec().getPtr();
        temp_vec = DeviceVec<T>(N);
        d_temp = temp_vec.getPtr();
        d_vals = kmat.getVec().getPtr();
        d_vals_ILU0 = DeviceVec<T>(nnz).getPtr();

        // for smoothing
        d_rhs_vec = DeviceVec<T>(N);
        d_rhs = d_rhs_vec.getPtr();
        d_inner_soln_vec = DeviceVec<T>(N);
        d_inner_soln = d_inner_soln_vec.getPtr();
        d_temp2 = DeviceVec<T>(N).getPtr();

        cusparseHandle = cusparseHandle_;

        // for checking residual with SpMV
        descrK = 0;
        CHECK_CUSPARSE(cusparseCreateMatDescr(&descrK));
        CHECK_CUSPARSE(cusparseSetMatType(descrK, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descrK, CUSPARSE_INDEX_BASE_ZERO));

        // startup factorization steps
        // -----------------------------------

        // create M matrix object (for full numeric factorization)
        cusparseCreateMatDescr(&descr_M);
        cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ZERO);
        cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseCreateBsrilu02Info(&info_M);

        // init L matrix objects (for triangular solve)
        cusparseCreateMatDescr(&descr_L);
        cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO);
        cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
        cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT);
        cusparseCreateBsrsv2Info(&info_L);

        // init U matrix objects (for triangular solve)
        cusparseCreateMatDescr(&descr_U);
        cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO);
        cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER);
        cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT);
        cusparseCreateBsrsv2Info(&info_U);

        // symbolic and numeric factorizations
        CHECK_CUSPARSE(cusparseDbsrilu02_bufferSize(cusparseHandle, dir, mb, nnzb, descr_M, d_vals,
                                                    d_rowp, d_cols, block_dim, info_M,
                                                    &pBufferSize_M));
        CHECK_CUSPARSE(cusparseDbsrsv2_bufferSize(cusparseHandle, dir, trans_L, mb, nnzb, descr_L,
                                                  d_vals, d_rowp, d_cols, block_dim, info_L,
                                                  &pBufferSize_L));
        CHECK_CUSPARSE(cusparseDbsrsv2_bufferSize(cusparseHandle, dir, trans_U, mb, nnzb, descr_U,
                                                  d_vals, d_rowp, d_cols, block_dim, info_U,
                                                  &pBufferSize_U));
        pBufferSize = std::max({pBufferSize_M, pBufferSize_L, pBufferSize_U});
        // cudaMalloc((void **)&pBuffer, pBufferSize);
        cudaMalloc((void **)&pBuffer, pBufferSize);

        // perform ILU symbolic factorization on L
        CHECK_CUSPARSE(cusparseDbsrilu02_analysis(cusparseHandle, dir, mb, nnzb, descr_M, d_vals,
                                                  d_rowp, d_cols, block_dim, info_M, policy_M,
                                                  pBuffer));
        status = cusparseXbsrilu02_zeroPivot(cusparseHandle, info_M, &structural_zero);
        if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
            printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
        }

        // analyze sparsity patern of L for efficient triangular solves
        CHECK_CUSPARSE(cusparseDbsrsv2_analysis(cusparseHandle, dir, trans_L, mb, nnzb, descr_L,
                                                d_vals, d_rowp, d_cols, block_dim, info_L, policy_L,
                                                pBuffer));
        CHECK_CUDA(cudaDeviceSynchronize());

        // analyze sparsity pattern of U for efficient triangular solves
        CHECK_CUSPARSE(cusparseDbsrsv2_analysis(cusparseHandle, dir, trans_U, mb, nnzb, descr_U,
                                                d_vals, d_rowp, d_cols, block_dim, info_U, policy_U,
                                                pBuffer));
        CHECK_CUDA(cudaDeviceSynchronize());

        // first time, then factor the matrix
        // factor();
    }

    void update_after_assembly(DeviceVec<T> &vars) {
        // do a new LU factorization
        factor();
    }

    // does nothing cause it's a directLU solve
    void set_print(bool print) {}
    void set_rel_tol(T rtol) {}
    void set_abs_tol(T atol) {}
    int get_num_iterations() { return 1; }
    void set_cycle_type(std::string cycle_) {}

    void assemble_matrix(DeviceVec<T> &vars) {
        assembler.set_variables(vars);
        assembler.add_jacobian_fast(kmat);
        assembler.apply_bcs(kmat);
    }

    T precond_complexity(int kmat_orig_nnzb) {
        // get [nnzb(precond) + nnzb(A)] / nnzb(A)
        int kmat_new_nnzb = nnzb;
        int precond_nnzb = nnzb;  // because kmat and ILU now stored with new
        // for larger fillin ends up being double fillin factor is new mem storage
        return (precond_nnzb + kmat_new_nnzb) * 1.0 / kmat_orig_nnzb;
    }

    void factor() {
        // copy the data from the original matrix to new place for factor
        CHECK_CUDA(cudaMemcpy(d_vals_ILU0, d_vals, nnz * sizeof(T), cudaMemcpyDeviceToDevice));

        // do factor (without object recreation here)

        // temp objects for the factorization

        // perform ILU numeric factorization (with M policy)
        CHECK_CUSPARSE(cusparseDbsrilu02(cusparseHandle, dir, mb, nnzb, descr_M, d_vals_ILU0,
                                         d_rowp, d_cols, block_dim, info_M, policy_M, pBuffer));
        status = cusparseXbsrilu02_zeroPivot(cusparseHandle, info_M, &numerical_zero);
        if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
            printf("block U(%d,%d) is not invertible\n", numerical_zero, numerical_zero);
        }

        CHECK_CUDA(cudaDeviceSynchronize());
    }

    void smoothDefect(DeviceVec<T> d_defect, DeviceVec<T> d_soln, int _n_iters = -1,
                      bool print = false, int print_freq = 0) {
        for (int iter = 0; iter < n_iters; iter++) {
            // get the solution from the defect rhs
            // =====================================

            cudaMemset(d_temp2, 0, N * sizeof(T));  // re-zero the solution

            // triangular solve L*z = x
            const double alpha = 1.0;
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                cusparseHandle, dir, trans_L, mb, nnzb, &alpha, descr_L, d_vals_ILU0, d_rowp,
                d_cols, block_dim, info_L, d_defect.getPtr(), d_temp, policy_L, pBuffer));

            // triangular solve U*y = z
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, mb, nnzb, &alpha,
                                                 descr_U, d_vals_ILU0, d_rowp, d_cols, block_dim,
                                                 info_U, d_temp, d_temp2, policy_U, pBuffer));

            // 4) compute defect update after new solution term..
            //     ..(with soln change stored in d_temp2)
            T a = -omega, b = 1.0;  // with omega * d_temp2 update
            CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a,
                                          descrK, d_vals, d_rowp, d_cols, block_dim, d_temp2, &b,
                                          d_defect.getPtr()));
            // also update d_soln += omega * d_temp2
            a = omega;
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_temp2, 1, d_soln.getPtr(), 1));
        }
    }

    bool solve(DeviceVec<T> rhs, DeviceVec<T> soln, bool check_conv = false) {
        /* assume here the rhs and soln are in solver permutations / orderings */

        if constexpr (MULTI_SMOOTH) {
            // setup rhs and soln with init guess of 0
            cudaMemcpy(d_rhs, rhs.getPtr(), N * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemset(d_inner_soln, 0, N * sizeof(T));  // re-zero the solution

            // call smoother on the defect=rhs and solution pair
            this->smoothDefect(d_rhs_vec, d_inner_soln_vec);

            // copy internal soln to external solution of the solve method
            cudaMemcpy(soln.getPtr(), d_inner_soln, N * sizeof(T), cudaMemcpyDeviceToDevice);

        } else {
            // not as smoother, just does full-solve (like direct solve here)

            // coarse grid directLU solve
            // triangular solve L*z = x
            const double alpha = 1.0;
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_L, mb, nnzb, &alpha,
                                                 descr_L, d_vals_ILU0, d_rowp, d_cols, block_dim,
                                                 info_L, rhs.getPtr(), d_temp, policy_L, pBuffer));

            // triangular solve U*y = z
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, mb, nnzb, &alpha,
                                                 descr_U, d_vals_ILU0, d_rowp, d_cols, block_dim,
                                                 info_U, d_temp, soln.getPtr(), policy_U, pBuffer));
        }

        // TEMP debug
        // bool coarse_fail = computeResidual(rhs, soln);
        // return coarse_fail;

        return false;
    }

    bool computeResidual(DeviceVec<T> &rhs, DeviceVec<T> &soln) {
        /* compute the residual of the direct solve */

        // maybe just for debugging here

        cudaMemcpy(d_temp, rhs.getPtr(), N * sizeof(T), cudaMemcpyDeviceToDevice);
        T init_norm;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_temp, 1, &init_norm));

        // subtract A*soln into temp which holds res
        T a = -1.0, b = 1.0;
        CHECK_CUSPARSE(cusparseDbsrmv(
            cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb,
            &a, descrK, d_vals, d_rowp, d_cols, block_dim, soln.getPtr(), &b, d_temp));

        T fin_norm;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_temp, 1, &fin_norm));

        T rel_conv = fin_norm / init_norm;
        printf("coarse solver rel conv %.8e\n", rel_conv);
        return rel_conv >= 1e-6;
    }

    void free() {
        if (is_free) return;
        is_free = true;  // now it's freed

        cudaFree(pBuffer);
        if (d_vals_ILU0) cudaFree(d_vals_ILU0);
        if (d_temp) cudaFree(d_temp);
        cusparseDestroyMatDescr(descr_L);
        cusparseDestroyMatDescr(descr_U);
        cusparseDestroyBsrsv2Info(info_L);
        cusparseDestroyBsrsv2Info(info_U);
        cusparseDestroyBsrilu02Info(info_M);
        cusparseDestroyMatDescr(descr_M);

        assembler.free();
        kmat.free();
    }

   private:
    // solver data
    int mb, N, nnzb, nnz, block_dim, *d_rowp, *d_cols;
    DeviceVec<T> temp_vec;
    T *d_temp, *d_vals, *d_vals_ILU0;

    bool is_free = false;
    Assembler assembler;
    BsrMat<DeviceVec<T>> kmat;

    // cusparse and cublas data
    cublasHandle_t &cublasHandle;
    cusparseHandle_t &cusparseHandle;
    cusparseMatDescr_t descr_L = 0, descr_U = 0;
    bsrsv2Info_t info_L = 0, info_U = 0;
    void *pBuffer = 0;
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE,
                              trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;

    cusparseMatDescr_t descrK;

    // factor utilities
    cusparseMatDescr_t descr_M = 0;
    bsrilu02Info_t info_M = 0;
    int pBufferSize_M, pBufferSize_L, pBufferSize_U, pBufferSize;
    int structural_zero, numerical_zero;
    const cusparseSolvePolicy_t policy_M =
        CUSPARSE_SOLVE_POLICY_USE_LEVEL;  // CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    cusparseStatus_t status;

    // FOR SMOOTHING
    // updated vectors
    DeviceVec<T> d_rhs_vec, d_inner_soln_vec;
    T *d_temp2, *d_resid;
    T *d_rhs, *d_inner_soln;
    int n_iters;
    T omega;
};