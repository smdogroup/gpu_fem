#pragma once
#include "../solve_utils.h"

// cusparse directLU solves with multigrid (V-cycle style preconditioner for two levels)
// only works from coarsest grid to next level

template <typename T, class Assembler>
class CusparseMGDirectLU : public BaseSolver {
public:
    CusparseMGDirectLU(Assembler &assembler, BsrMat<DeviceVec<T>> &kmat) {
        /* create the cusparse direct solver (for repeated solves) */

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

        // create cusparse handle
        cusparseCreate(&handle);

        // first time, then factor the matrix
        factor_matrix();
    }

    void update_after_assembly(DeviceVec<T> &vars) {
        // do a new LU factorization
        factor_matrix();
    }

    // does nothing cause it's a directLU solve
    void set_rel_tol(T rtol) {}
    void set_abs_tol(T atol) {}

    void factor_matrix() {
        // copy the data from the original matrix to new place for factor
        CHECK_CUDA(
        cudaMemcpy(d_vals_ILU0, d_vals, nnz * sizeof(T), cudaMemcpyDeviceToDevice));

        CUSPARSE::perform_ilu0_factorization(handle, descr_L, descr_U, info_L, info_U, &pBuffer, mb,
                                         nnzb, block_dim, d_vals_ILU0, d_rowp, d_cols, trans_L,
                                         trans_U, policy_L, policy_U, dir);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    void solve(DeviceVec<T> rhs, DeviceVec<T> soln, bool check_conv = false) {
        /* assume here the rhs and soln are in solver permutations / orderings */

        // coarse grid directLU solve
        // triangular solve L*z = x
        const double alpha = 1.0;
        CHECK_CUSPARSE(cusparseDbsrsv2_solve(handle, dir, trans_L, mb, nnzb, &alpha, descr_L,
                                            d_vals_ILU0, d_rowp, d_cols, block_dim, info_L, rhs.getPtr(),
                                            d_temp, policy_L, pBuffer));

        // triangular solve U*y = z
        CHECK_CUSPARSE(cusparseDbsrsv2_solve(handle, dir, trans_U, mb, nnzb, &alpha, descr_U,
                                            d_vals_ILU0, d_rowp, d_cols, block_dim, info_U, d_temp,
                                            soln.getPtr(), policy_U, pBuffer));
    }

    

    void free() {
        cudaFree(pBuffer);
        cudaFree(d_vals_ILU0);
        cudaFree(d_temp);
        cusparseDestroyMatDescr(descr_L);
        cusparseDestroyMatDescr(descr_U);
        cusparseDestroyBsrsv2Info(info_L);
        cusparseDestroyBsrsv2Info(info_U);
        cusparseDestroy(handle);
    }

private:
    // solver data
    int mb, N, nnzb, nnz, block_dim, *d_rowp, *d_cols;
    DeviceVec<T> temp_vec;
    T *d_temp, *d_vals, *d_vals_ILU0;

    // cusparse and cublas data
    cusparseHandle_t handle;
    cusparseMatDescr_t descr_L = 0, descr_U = 0;
    bsrsv2Info_t info_L = 0, info_U = 0;
    void *pBuffer = 0;
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE,
                              trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;
};