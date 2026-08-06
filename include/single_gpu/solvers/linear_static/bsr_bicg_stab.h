#pragma once

#include <assert.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include <iostream>

#include "_cusparse_utils.h"
#include "_utils.h"
#include "chrono"
#include "cublas_v2.h"
#include "cuda_utils.h"

namespace CUSPARSE {

/* source:
 * https://utminers.utep.edu/xzeng/2017spring_math5330/MATH_5330_Computational_Methods_of_Linear_Algebra_files/ln07.pdf
 */

template <typename T, bool use_precond = true>
void BiCGStab_solve(BsrMat<DeviceVec<T>> &mat, DeviceVec<T> &rhs, DeviceVec<T> &soln,
                    int _n_iter = 100, T abs_tol = 1e-8, T rel_tol = 1e-8, bool can_print = false,
                    bool debug = false, int print_freq = 10) {
    static_assert(std::is_same<T, double>::value,
                  "Only double precision is written in our code for cuSparse BiCGStab");

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
    T *d_tmp2 = DeviceVec<T>(N).getPtr();
    T *d_resid = DeviceVec<T>(N).getPtr();
    T *d_resid0 = DeviceVec<T>(N).getPtr();
    T *d_p = DeviceVec<T>(N).getPtr();
    T *d_y = DeviceVec<T>(N).getPtr();
    T *d_v = DeviceVec<T>(N).getPtr();
    T *d_h = DeviceVec<T>(N).getPtr();
    T *d_s = DeviceVec<T>(N).getPtr();
    T *d_z = DeviceVec<T>(N).getPtr();
    T *d_Az = DeviceVec<T>(N).getPtr();

    // TODO : could add outer loops to reset resid0 vector
    // compute initial residual
    CHECK_CUDA(cudaMemcpy(d_resid, d_rhs, N * sizeof(T), cudaMemcpyDeviceToDevice));

    // then subtract Ax from rhs
    a = 1.0, b = 0.0;
    CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a, descrA,
                                  d_vals, d_rowp, d_cols, block_dim, d_x, &b, d_tmp));
    // resid -= A * x
    a = -1.0;
    CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_tmp, 1, d_resid, 1));

    T init_resid_norm;
    CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &init_resid_norm));
    if (can_print) printf("BiCGStab init_resid = %.8e\n", init_resid_norm);

    // now copy from resid to resid0 and p0
    CHECK_CUDA(cudaMemcpy(d_resid0, d_resid, N * sizeof(T), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(d_p, d_resid, N * sizeof(T), cudaMemcpyDeviceToDevice));

    T rho_prev;
    CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_resid0, 1, d_resid, 1, &rho_prev));

    // start main iteration of BiCGStab
    for (int i = 0; i < _n_iter; i++) {
        // ------------------
        // y = U^-1 L^-1 p
        a = 1.0;
        CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_L, mb, nnzb, &a, descr_L,
                                             d_vals_ILU0, d_rowp, d_cols, block_dim, info_L, d_p,
                                             d_tmp, policy_L, pBuffer));
        CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, mb, nnzb, &a, descr_U,
                                             d_vals_ILU0, d_rowp, d_cols, block_dim, info_U, d_tmp,
                                             d_y, policy_U, pBuffer));

        // -------------------
        // v = A y
        a = 1.0, b = 0.0;
        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a, descrA,
                                      d_vals, d_rowp, d_cols, block_dim, d_y, &b, d_v));

        // alpha = rho_i-1 / <resid0, v>
        T rv_dot;
        CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_resid0, 1, d_v, 1, &rv_dot));
        T alpha = rho_prev / rv_dot;

        // h = x + alpha * y
        CHECK_CUDA(cudaMemcpy(d_h, d_x, N * sizeof(T), cudaMemcpyDeviceToDevice));
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &alpha, d_y, 1, d_h, 1));

        // s = r_i-1 - alpha * v
        a = -alpha;
        CHECK_CUDA(cudaMemcpy(d_s, d_resid, N * sizeof(T), cudaMemcpyDeviceToDevice));
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_v, 1, d_s, 1));

        // check convergence at s
        T resid_norm;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_s, 1, &resid_norm));
        if (can_print && (i % print_freq == 0))
            printf("BiCGStab iter %d : resid %.9e\n", i, resid_norm);
        if (abs(resid_norm) < (abs_tol + init_resid_norm * rel_tol)) {
            if (can_print)
                printf("BiCGStab converged in %d iterations to %.9e resid\n", i + 1, resid_norm);
            break;
        }

        // z = U^-1 L^-1 s
        a = 1.0;
        CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_L, mb, nnzb, &a, descr_L,
                                             d_vals_ILU0, d_rowp, d_cols, block_dim, info_L, d_s,
                                             d_tmp2, policy_L, pBuffer));
        CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, mb, nnzb, &a, descr_U,
                                             d_vals_ILU0, d_rowp, d_cols, block_dim, info_U, d_tmp2,
                                             d_z, policy_U, pBuffer));

        // t = A * z, with t stored in d_Az (because d_t is reserved)
        a = 1.0, b = 0.0;
        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a, descrA,
                                      d_vals, d_rowp, d_cols, block_dim, d_z, &b, d_Az));

        // tmp = L^-1 t (recall t stored in d_Az)
        a = 1.0;
        CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_L, mb, nnzb, &a, descr_L,
                                             d_vals_ILU0, d_rowp, d_cols, block_dim, info_L, d_Az,
                                             d_tmp, policy_L, pBuffer));
        // tmp2 = L^-1 s, already stored in d_tmp2 from earlier (aka tmp2)
        // w = <tmp, tmp2> / <tmp, tmp>
        T num, den;
        CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_tmp, 1, d_tmp2, 1, &num));
        CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_tmp, 1, d_tmp, 1, &den));
        T w = num / den;

        // x_i = h + w * z
        CHECK_CUDA(cudaMemcpy(d_x, d_h, N * sizeof(T), cudaMemcpyDeviceToDevice));
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &w, d_z, 1, d_x, 1));

        // r_i = s - w * t (recall t is d_Az)
        a = -w;
        CHECK_CUDA(cudaMemcpy(d_resid, d_s, N * sizeof(T), cudaMemcpyDeviceToDevice));
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_Az, 1, d_resid, 1));

        // check for convergence
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &resid_norm));
        if (abs(resid_norm) < (abs_tol + init_resid_norm * rel_tol)) {
            if (can_print)
                printf("BiCGStab converged in %d iterations to %.9e resid\n", i + 1, resid_norm);
            break;
        }

        // rhoi = <resid0, resid>
        T rho;
        CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_resid0, 1, d_resid, 1, &rho));

        // beta = rhoi / rhoi-1 * alpha / w
        T beta = rho / rho_prev * alpha / w;
        rho_prev = rho;
        // printf("beta = %.4e, rho_prev = %.4e, w = %.4e\n", beta, rho_prev, w);

        // p_i = r_i + beta * (p_{i-1} - w * v)
        CHECK_CUDA(cudaMemcpy(d_tmp, d_p, N * sizeof(T), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(d_p, d_resid, N * sizeof(T), cudaMemcpyDeviceToDevice));
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &beta, d_tmp, 1, d_p, 1));
        a = -w * beta;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_v, 1, d_p, 1));

        // then do another iteration of BiCGStab
    }

    // cleanup and inverse permute the solution for exit
    // -------------------------------------------------

    // now also inverse permute the soln data
    permute_soln<BsrMat<DeviceVec<T>>, DeviceVec<T>>(mat, soln);

    // free resources
    cudaFree(pBuffer);
    cusparseDestroyMatDescr(descr_L);
    cusparseDestroyMatDescr(descr_U);
    cusparseDestroyBsrsv2Info(info_L);
    cusparseDestroyBsrsv2Info(info_U);
    cusparseDestroy(cusparseHandle);

    // TODO : still missing a few free / delete[] statements
    CHECK_CUDA(cudaFree(d_tmp))
    CHECK_CUDA(cudaFree(d_tmp2))
    CHECK_CUDA(cudaFree(d_resid))
    CHECK_CUDA(cudaFree(d_resid0))
    CHECK_CUDA(cudaFree(d_p))
    CHECK_CUDA(cudaFree(d_y))
    CHECK_CUDA(cudaFree(d_v))
    CHECK_CUDA(cudaFree(d_h))
    CHECK_CUDA(cudaFree(d_s))
    CHECK_CUDA(cudaFree(d_z))
    CHECK_CUDA(cudaFree(d_Az))

    // print timing data
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    double dt = duration.count() / 1e6;
    if (can_print) {
        printf("\tfinished BiCGStab in %.4e sec\n", dt);
    }
}

};  // namespace CUSPARSE
