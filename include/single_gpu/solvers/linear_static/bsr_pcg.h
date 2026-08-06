#pragma once

#include <assert.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include <chrono>
#include <iostream>

#include "_cusparse_utils.h"
#include "_utils.h"
#include "cublas_v2.h"
#include "cuda_utils.h"

namespace CUSPARSE {

template <typename T>
void PCG_solve(BsrMat<DeviceVec<T>> &mat, DeviceVec<T> &rhs, DeviceVec<T> &soln, int _n_iter = 100,
               T abs_tol = 1e-8, T rel_tol = 1e-8, bool can_print = false, bool debug = false,
               int print_freq = 10, int _max_iter = -1) {
    static_assert(std::is_same<T, double>::value,
                  "Only double precision is written in our code for cuSparse PCG");

    /* PCG (Preconditioned Conjugate Gradient) algorithm */

    auto start = std::chrono::high_resolution_clock::now();
    auto rhs_perm = inv_permute_rhs<BsrMat<DeviceVec<T>>, DeviceVec<T>>(mat, rhs);

    BsrData bsr_data = mat.getBsrData();
    int mb = bsr_data.nnodes;
    int nnzb = bsr_data.nnzb;
    int block_dim = bsr_data.block_dim;
    index_t *d_rowp = bsr_data.rowp;
    index_t *d_cols = bsr_data.cols;
    int *iperm = bsr_data.iperm;
    int N = soln.getSize();
    int n_iter = min(_n_iter, bsr_data.nnodes);
    int max_iter = _max_iter == -1 ? n_iter : _max_iter;  // no restarts if -1 input
    T *d_rhs = rhs_perm.getPtr();
    T *d_x = soln.getPtr();

    // permute data in soln if guess is not zero
    soln.permuteData(block_dim, iperm);

    // note this changes the mat data to be LU (but that's the whole point
    // of LU solve is for repeated linear solves we now just do triangular
    // solves)
    T *d_vals = mat.getPtr();

    // also make a temporary array for the preconditioner values
    T *d_vals_ILU0 = DeviceVec<T>(mat.get_nnz()).getPtr();
    // ILU0 equiv to ILU(k) if sparsity pattern has ILU(k)
    CHECK_CUDA(
        cudaMemcpy(d_vals_ILU0, d_vals, mat.get_nnz() * sizeof(T), cudaMemcpyDeviceToDevice));

    // create initial cusparse and cublas handles --------------

    /* Create CUBLAS context */
    cublasHandle_t cublasHandle = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    /* Create CUSPARSE context */
    cusparseHandle_t cusparseHandle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

    // create the matrix BSR object
    // -----------------------------

    /* Description of the A matrix */
    cusparseMatDescr_t descrA = 0;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
    CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

    // create ILU(0) preconditioner
    // -----------------------------
    // [equiv to ILU(k) precondioner if ILU(k) sparsity pattern used in BsrData object]

    // init objects for LU factorization and LU solve
    cusparseMatDescr_t descr_L = 0, descr_U = 0;
    bsrsv2Info_t info_L = 0, info_U = 0;
    void *pBuffer = 0;
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    // tried changing both policy L and U to be USE_LEVEL not really a change
    // policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL,
    // policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE,
                              trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;

    T a = 1.0, b = 0.0;

    // perform the symbolic and numeric factorization of LU on given sparsity pattern
    CUSPARSE::perform_ilu0_factorization(cusparseHandle, descr_L, descr_U, info_L, info_U, &pBuffer,
                                         mb, nnzb, block_dim, d_vals_ILU0, d_rowp, d_cols, trans_L,
                                         trans_U, policy_L, policy_U, dir);

    // prelim vectors and data for BiCGStab
    // ------------------------------------

    // make temp vecs
    T *d_tmp = DeviceVec<T>(N).getPtr();
    T *d_resid = DeviceVec<T>(N).getPtr();
    T *d_p = DeviceVec<T>(N).getPtr();
    T *d_w = DeviceVec<T>(N).getPtr();
    T *d_z = DeviceVec<T>(N).getPtr();

    int nrestarts = max_iter / n_iter;
    int total_iter = 0;
    bool converged = false;
    for (int irestart = 0; irestart < nrestarts; irestart++) {
        // compute r_0 = b - Ax
        CHECK_CUDA(cudaMemcpy(d_resid, d_rhs, N * sizeof(T), cudaMemcpyDeviceToDevice));
        a = 1.0, b = 0.0;
        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a, descrA,
                                      d_vals, d_rowp, d_cols, block_dim, d_x, &b, d_tmp));
        a = -1.0;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_tmp, 1, d_resid, 1));

        // compute |r_0|
        T init_resid_norm;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &init_resid_norm));
        if (can_print) printf("PCG init_resid = %.8e\n", init_resid_norm);

        // compute z = M^-1 r_0
        a = 1.0;
        CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_L, mb, nnzb, &a, descr_L,
                                             d_vals_ILU0, d_rowp, d_cols, block_dim, info_L,
                                             d_resid, d_tmp, policy_L, pBuffer));
        CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, mb, nnzb, &a, descr_U,
                                             d_vals_ILU0, d_rowp, d_cols, block_dim, info_U, d_tmp,
                                             d_z, policy_U, pBuffer));

        // copy z => p
        CHECK_CUDA(cudaMemcpy(d_p, d_z, N * sizeof(T), cudaMemcpyDeviceToDevice));

        // inner loop
        for (int j = 0; j < n_iter; j++, total_iter++) {
            // w = A * p
            a = 1.0, b = 0.0;
            CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a,
                                          descrA, d_vals, d_rowp, d_cols, block_dim, d_p, &b, d_w));

            // alpha = <r,z> / <w,p>, with dot products in rz0, wp0
            T rz0;
            CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_resid, 1, d_z, 1, &rz0));
            T wp0;
            CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_w, 1, d_p, 1, &wp0));
            T alpha = rz0 / wp0;

            // x += alpha * p
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &alpha, d_p, 1, d_x, 1));

            // r -= alpha * w
            a = -alpha;
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_w, 1, d_resid, 1));

            // z = M^-1 * r
            a = 1.0;
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_L, mb, nnzb, &a,
                                                 descr_L, d_vals_ILU0, d_rowp, d_cols, block_dim,
                                                 info_L, d_resid, d_tmp, policy_L, pBuffer));
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, mb, nnzb, &a,
                                                 descr_U, d_vals_ILU0, d_rowp, d_cols, block_dim,
                                                 info_U, d_tmp, d_z, policy_U, pBuffer));

            // beta = <r_new,z_new> / <r_old,z_old> = rz1 / rz0
            T rz1;
            CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_resid, 1, d_z, 1, &rz1));
            T beta = rz1 / rz0;

            // p = z + beta * p
            a = beta;  // p *= beta scalar
            CHECK_CUBLAS(cublasDscal(cublasHandle, N, &a, d_p, 1));
            a = 1.0;  // p += z
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_z, 1, d_p, 1));

            // check for convergence
            T resid_norm;
            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &resid_norm));
            if (can_print && (j % print_freq == 0)) printf("PCG [%d] = %.8e\n", j, resid_norm);

            if (abs(resid_norm) < (abs_tol + init_resid_norm * rel_tol)) {
                converged = true;
                if (can_print)
                    printf("\nPCG converged in %d iterations to %.9e resid\n", j + 1, resid_norm);
                break;
            }
        }                      // end of inner loop
        if (converged) break;  // exit outer loop if converged
    }

    // print timing data
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    double dt = duration.count() / 1e6;
    if (can_print) {
        printf("\tfinished PCG with BSR ILU in %.4e sec\n", dt);
    }
}  // end of PCG method with BSR and ILU
};  // namespace CUSPARSE