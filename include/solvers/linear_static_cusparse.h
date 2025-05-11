#pragma once

#include <assert.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include <iostream>

#include "../cuda_utils.h"
#include "chrono"
#include "cublas_v2.h"
#include "utils/_cusparse_utils.h"
#include "utils/_utils.h"

namespace CUSPARSE {

template <typename T>
void direct_LU_solve(BsrMat<DeviceVec<T>> &mat, DeviceVec<T> &rhs, DeviceVec<T> &soln,
                     bool can_print = false) {
    /* direct LU solve => performs LU factorization then L and U triangular solves.
        Best for solving with same matrix and multiple rhs vectors such as aeroelastic + linear
       structures Although need to check LU factorization held in place correctly and boolean to not
       refactorize maybe */
    auto rhs_perm = inv_permute_rhs<BsrMat<DeviceVec<T>>, DeviceVec<T>>(mat, rhs);

    if (can_print) {
        printf("begin cusparse direct LU solve\n");
    }
    auto start = std::chrono::high_resolution_clock::now();

    // copy important inputs for Bsr structure out of BsrMat
    // TODO : was trying to make some of these const but didn't accept it in
    // final solve
    BsrData bsr_data = mat.getBsrData();
    int mb = bsr_data.nnodes;
    int nnzb = bsr_data.nnzb;
    int block_dim = bsr_data.block_dim;
    index_t *d_rowp = bsr_data.rowp;
    index_t *d_cols = bsr_data.cols;
    T *d_rhs = rhs_perm.getPtr();
    T *d_soln = soln.getPtr();
    DeviceVec<T> temp = DeviceVec<T>(soln.getSize());
    T *d_temp = temp.getPtr();

    // note this changes the mat data to be LU (but that's the whole point
    // of LU solve is for repeated linear solves we now just do triangular
    // solves)
    T *d_vals = mat.getPtr();

    // Initialize the cuda cusparse handle
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // init objects for LU factorization and LU solve
    cusparseMatDescr_t descr_L = 0, descr_U = 0;
    bsrsv2Info_t info_L = 0, info_U = 0;
    void *pBuffer = 0;
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE,
                              trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;

    // perform the symbolic and numeric factorization of LU on given sparsity pattern
    CUSPARSE::perform_LU_factorization(handle, descr_L, descr_U, info_L, info_U, &pBuffer, mb, nnzb,
                                       block_dim, d_vals, d_rowp, d_cols, trans_L, trans_U,
                                       policy_L, policy_U, dir);

    // triangular solve L*z = x
    const double alpha = 1.0;
    CHECK_CUSPARSE(cusparseDbsrsv2_solve(handle, dir, trans_L, mb, nnzb, &alpha, descr_L, d_vals,
                                         d_rowp, d_cols, block_dim, info_L, d_rhs, d_temp, policy_L,
                                         pBuffer));

    // triangular solve U*y = z
    CHECK_CUSPARSE(cusparseDbsrsv2_solve(handle, dir, trans_U, mb, nnzb, &alpha, descr_U, d_vals,
                                         d_rowp, d_cols, block_dim, info_U, d_temp, d_soln,
                                         policy_U, pBuffer));

    // free resources
    cudaFree(pBuffer);
    cusparseDestroyMatDescr(descr_L);
    cusparseDestroyMatDescr(descr_U);
    cusparseDestroyBsrsv2Info(info_L);
    cusparseDestroyBsrsv2Info(info_U);
    cusparseDestroy(handle);

    // now also inverse permute the soln data
    permute_soln<BsrMat<DeviceVec<T>>, DeviceVec<T>>(mat, soln);

    // print timing data
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    double dt = duration.count() / 1e6;
    if (can_print) {
        printf("\tfinished in %.4e sec\n", dt);
    }
}

template <typename T, bool use_precond = true, bool right = false>
void GMRES_solve(BsrMat<DeviceVec<T>> &mat, DeviceVec<T> &rhs, DeviceVec<T> &soln,
                 int _n_iter = 100, int max_iter = 500, T abs_tol = 1e-8, T rel_tol = 1e-8,
                 bool can_print = false, bool debug = false, int print_freq = 10) {
    /* GMRES iterative solve using a BsrMat on GPU with CUDA / CuSparse
        only supports T = double right now, may add float at some point (but float won't converge as
       deeply the residual, only about 1e-7) */
    auto rhs_perm = inv_permute_rhs<BsrMat<DeviceVec<T>>, DeviceVec<T>>(mat, rhs);

    // which type of preconditioners
    constexpr bool left_precond = use_precond && !right;
    constexpr bool right_precond = use_precond && right;

    // if (can_print) {
    //     printf("begin cusparse GMRES solve\n");
    // }
    auto start = std::chrono::high_resolution_clock::now();

    // copy important inputs for Bsr structure out of BsrMat
    // TODO : was trying to make some of these const but didn't accept it in
    // final solve
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
    CHECK_CUDA(cudaMalloc((void **)&d_vals_ILU0, mat.get_nnz() * sizeof(T)));
    CHECK_CUDA(
        cudaMemcpy(d_vals_ILU0, d_vals, mat.get_nnz() * sizeof(T), cudaMemcpyDeviceToDevice));

    // make temp vecs
    T *d_tmp = DeviceVec<T>(soln.getSize()).getPtr();
    T *d_tmp2 = DeviceVec<T>(soln.getSize()).getPtr();
    T *d_resid = DeviceVec<T>(soln.getSize()).getPtr();
    T *d_w = DeviceVec<T>(soln.getSize()).getPtr();
    T *d_xR = DeviceVec<T>(soln.getSize()).getPtr();  // for right preconditioning

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
    CUSPARSE::perform_LU_factorization(cusparseHandle, descr_L, descr_U, info_L, info_U, &pBuffer,
                                       mb, nnzb, block_dim, d_vals_ILU0, d_rowp, d_cols, trans_L,
                                       trans_U, policy_L, policy_U, dir);

    // main GMRES solve
    // ----------------

    // initialize GMRES data, some on host, some on GPU
    // host GMRES data
    T g[n_iter + 1], cs[n_iter], ss[n_iter];
    T H[(n_iter + 1) * (n_iter)];

    // GMRES device data
    T *d_Vmat, *d_V;
    CHECK_CUDA(cudaMalloc((void **)&d_Vmat, (n_iter + 1) * N * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void **)&d_V, N * sizeof(T)));
    cusparseDnVecDescr_t vec_V;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vec_V, N, d_V, CUDA_R_64F));

    bool converged = false;
    int total_iter = 0;

    for (int iouter = 0; iouter < max_iter / n_iter; iouter++) {
        int jj = n_iter - 1;

        // apply precond to rhs if in use
        // copy b or rhs to resid
        CHECK_CUDA(cudaMemcpy(d_resid, d_rhs, N * sizeof(T), cudaMemcpyDeviceToDevice));
        a = 1.0, b = 0.0;

        if constexpr (right_precond) {
            // compute xR0 = M * x0, so we can update y later
            // first U * x0 => tmp
            CHECK_CUSPARSE(cusparseDbsrmv(
                cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb,
                nnzb, &a, descr_U, d_vals_ILU0, d_rowp, d_cols, block_dim, d_x, &b, d_tmp));
            CHECK_CUDA(cudaDeviceSynchronize());

            // then L * tmp => xR0
            CHECK_CUSPARSE(cusparseDbsrmv(
                cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb,
                nnzb, &a, descr_L, d_vals_ILU0, d_rowp, d_cols, block_dim, d_tmp, &b, d_xR));
            CHECK_CUDA(cudaDeviceSynchronize());

            // then for right precond instead of A*M^-1*xR0
            // this is equal to A*x0, so we leave it at b - A*x0 here
        }

        // then subtract Ax from rhs

        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a, descrA,
                                      d_vals, d_rowp, d_cols, block_dim, d_x, &b, d_tmp));
        CHECK_CUDA(cudaDeviceSynchronize());
        // resid -= A * x
        a = -1.0;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_tmp, 1, d_resid, 1));
        CHECK_CUDA(cudaDeviceSynchronize());

        T init_true_resid;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &init_true_resid));
        CHECK_CUDA(cudaDeviceSynchronize());

        // now apply precond to the resid
        if constexpr (left_precond) {
            // zero vec_tmp
            CHECK_CUDA(cudaMemset(d_tmp, 0.0, N * sizeof(T)));

            // ILU solve U^-1 L^-1 * b
            // L^-1 * b => tmp
            a = 1.0;
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_L, mb, nnzb, &a,
                                                 descr_L, d_vals_ILU0, d_rowp, d_cols, block_dim,
                                                 info_L, d_resid, d_tmp, policy_L, pBuffer));
            CHECK_CUDA(cudaDeviceSynchronize());

            // U^-1 * tmp => into rhs
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, mb, nnzb, &a,
                                                 descr_U, d_vals_ILU0, d_rowp, d_cols, block_dim,
                                                 info_U, d_tmp, d_resid, policy_U, pBuffer));
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        // temp debug
        // if (debug) {
        //     CHECK_CUDA(cudaMemcpy(d_x, d_resid, N * sizeof(T), cudaMemcpyDeviceToHost));
        //     // now also inverse permute the soln data
        //     permute_soln<BsrMat<DeviceVec<T>>, DeviceVec<T>>(mat, soln);
        //     return;
        // }

        // GMRES initial residual
        // assumes here d_X is 0 initially => so r0 = b - Ax
        T beta;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &beta));
        CHECK_CUDA(cudaDeviceSynchronize());
        if (can_print)
            printf("GMRES init resid = true %.9e, precond %.9e\n", init_true_resid, beta);
        g[0] = beta;

        // set v0 = r0 / beta (unit vec)
        a = 1.0 / beta;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_resid, 1, &d_Vmat[0], 1));
        CHECK_CUDA(cudaDeviceSynchronize());

        T *h_v0 = new T[N];
        CHECK_CUDA(cudaMemcpy(h_v0, d_Vmat, N * sizeof(T), cudaMemcpyDeviceToHost));
        // print vec
        if (debug) {
            printf("r0:");
            printVec<T>(4, h_v0);
        }

        // then begin main GMRES iteration loop!
        for (int j = 0; j < n_iter; j++, total_iter++) {
            // zero this vec
            // CHECK_CUDA(cudaMemset(&d_Vmat[j * N], 0.0, N * sizeof(T)));

            // get vj and copy it into the cusparseDnVec_t
            void *vj_col = static_cast<void *>(&d_Vmat[j * N]);
            CHECK_CUSPARSE(cusparseDnVecSetValues(vec_V, vj_col));

            if constexpr (right_precond) {
                // U^-1 L^-1 * vj => vj precond solve here
                a = 1.0;
                CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                    cusparseHandle, dir, trans_L, mb, nnzb, &a, descr_L, d_vals_ILU0, d_rowp,
                    d_cols, block_dim, info_L, &d_Vmat[j * N], d_tmp, policy_L, pBuffer));
                CHECK_CUDA(cudaDeviceSynchronize());
                // U^-1 * tmp => into vj
                CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                    cusparseHandle, dir, trans_U, mb, nnzb, &a, descr_U, d_vals_ILU0, d_rowp,
                    d_cols, block_dim, info_U, d_tmp, d_tmp2, policy_U, pBuffer));
                CHECK_CUDA(cudaDeviceSynchronize());

                // w = A * vj + 0 * w
                // BSR matrix multiply here MV
                a = 1.0, b = 0.0;
                CHECK_CUSPARSE(cusparseDbsrmv(
                    cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, mb,
                    mb, nnzb, &a, descrA, d_vals, d_rowp, d_cols, block_dim, d_tmp2, &b, d_w));
                CHECK_CUDA(cudaDeviceSynchronize());
            }

            if constexpr (left_precond) {
                // w = A * vj + 0 * w
                // BSR matrix multiply here MV
                a = 1.0, b = 0.0;
                CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                              CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a,
                                              descrA, d_vals, d_rowp, d_cols, block_dim,
                                              &d_Vmat[j * N], &b, d_w));
                CHECK_CUDA(cudaDeviceSynchronize());

                // U^-1 L^-1 * w => w precond solve here
                a = 1.0;
                // first L^-1 * w => tmp
                CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                    cusparseHandle, dir, trans_L, mb, nnzb, &a, descr_L, d_vals_ILU0, d_rowp,
                    d_cols, block_dim, info_L, d_w, d_tmp, policy_L, pBuffer));
                CHECK_CUDA(cudaDeviceSynchronize());
                // U^-1 * tmp => into w
                CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                    cusparseHandle, dir, trans_U, mb, nnzb, &a, descr_U, d_vals_ILU0, d_rowp,
                    d_cols, block_dim, info_U, d_tmp, d_w, policy_U, pBuffer));
                CHECK_CUDA(cudaDeviceSynchronize());
            }

            // now update householder matrix
            for (int i = 0; i < j + 1; i++) {
                // get vi column
                void *vi_col = static_cast<void *>(&d_Vmat[i * N]);
                CHECK_CUSPARSE(cusparseDnVecSetValues(vec_V, vi_col));

                T w_vi_dot;
                CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_w, 1, &d_Vmat[i * N], 1, &w_vi_dot));
                CHECK_CUDA(cudaDeviceSynchronize());

                // H_ij = vi dot w
                H[n_iter * i + j] = w_vi_dot;

                if (debug) printf("H[%d,%d] = %.9e\n", i, j, H[n_iter * i + j]);

                // w -= Hij * vi
                a = -H[n_iter * i + j];
                CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, &d_Vmat[i * N], 1, d_w, 1));
                CHECK_CUDA(cudaDeviceSynchronize());
            }

            // double check and print the value of
            if (debug && N <= 16) {
                T *h_w = new T[N];
                // printf("checkpt3\n");
                CHECK_CUDA(cudaMemcpy(h_w, d_w, N * sizeof(T), cudaMemcpyDeviceToHost));
                printf("h_w[%d] post GS:", j);
                printVec<T>(N, h_w);

                // if (j == 0) return 0;
            }

            // norm of w
            T nrm_w;
            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_w, 1, &nrm_w));
            CHECK_CUDA(cudaDeviceSynchronize());

            // H_{j+1,j}
            H[n_iter * (j + 1) + j] = nrm_w;

            // v_{j+1} column unit vec = w / H_{j+1,j}
            a = 1.0 / H[n_iter * (j + 1) + j];
            CHECK_CUBLAS(cublasDcopy(cublasHandle, N, d_w, 1, &d_Vmat[(j + 1) * N], 1));
            CHECK_CUDA(cudaDeviceSynchronize());

            CHECK_CUBLAS(cublasDscal(cublasHandle, N, &a, &d_Vmat[(j + 1) * N], 1));
            CHECK_CUDA(cudaDeviceSynchronize());

            if (debug && N <= 16) {
                T *h_tmp = new T[N];
                // printf("checkpt3\n");
                CHECK_CUDA(
                    cudaMemcpy(h_tmp, &d_Vmat[(j + 1) * N], N * sizeof(T), cudaMemcpyDeviceToHost));
                // printf("next V:");
                // printVec<T>(N, h_tmp);

                // if (j == 0) return 0;
            }

            // then givens rotations to elim householder matrix
            for (int i = 0; i < j; i++) {
                T temp = H[i * n_iter + j];
                H[n_iter * i + j] = cs[i] * H[n_iter * i + j] + ss[i] * H[n_iter * (i + 1) + j];
                H[n_iter * (i + 1) + j] = -ss[i] * temp + cs[i] * H[n_iter * (i + 1) + j];
            }

            T Hjj = H[n_iter * j + j], Hj1j = H[n_iter * (j + 1) + j];
            cs[j] = Hjj / sqrt(Hjj * Hjj + Hj1j * Hj1j);
            ss[j] = cs[j] * Hj1j / Hjj;

            T g_temp = g[j];
            g[j] *= cs[j];
            g[j + 1] = -ss[j] * g_temp;

            // printf("GMRES iter %d : resid %.9e\n", j, nrm_w);
            if (can_print && (j % print_freq == 0))
                printf("GMRES iter %d : resid %.9e\n", j, abs(g[j + 1]));

            if (debug) printf("j=%d, g[j]=%.9e, g[j+1]=%.9e\n", j, g[j], g[j + 1]);

            H[n_iter * j + j] = cs[j] * H[n_iter * j + j] + ss[j] * H[n_iter * (j + 1) + j];
            H[n_iter * (j + 1) + j] = 0.0;

            if (abs(g[j + 1]) < (abs_tol + beta * rel_tol)) {
                if (can_print)
                    printf("GMRES converged in %d iterations to %.9e resid\n", j + 1, g[j + 1]);
                jj = j;
                converged = true;
                break;
            }
        }  // end of inner loop

        // now solve Householder triangular system
        // only up to size jj+1 x jj+1 where we exited on iteration jj
        T *Hred = new T[(jj + 1) * (jj + 1)];
        for (int i = 0; i < jj + 1; i++) {
            for (int j = 0; j < jj + 1; j++) {
                // in-place transpose to be compatible with column-major cublasDtrsv later on
                Hred[(jj + 1) * i + j] = H[n_iter * j + i];

                // Hred[(jj+1) * i + j] = H[n_iter * i + j];
            }
        }

        // now print out Hred
        if (debug) {
            printf("Hred:");
            printVec<T>((jj + 1) * (jj + 1), Hred);
            printf("gred:");
            printVec<T>((jj + 1), g);
        }

        // now copy data from Hred host to device
        T *d_Hred;
        CHECK_CUDA(cudaMalloc(&d_Hred, (jj + 1) * (jj + 1) * sizeof(T)));
        CHECK_CUDA(
            cudaMemcpy(d_Hred, Hred, (jj + 1) * (jj + 1) * sizeof(T), cudaMemcpyHostToDevice));

        // also create gred vector on the device
        T *d_gred;
        CHECK_CUDA(cudaMalloc(&d_gred, (jj + 1) * sizeof(T)));
        CHECK_CUDA(cudaMemcpy(d_gred, g, (jj + 1) * sizeof(T), cudaMemcpyHostToDevice));

        // now solve Householder system H * y = g
        // T *d_y;
        // CHECK_CUDA(cudaMalloc(&d_y, (jj+1) * sizeof(T)));
        CHECK_CUBLAS(cublasDtrsv(cublasHandle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                                 CUBLAS_DIAG_NON_UNIT, jj + 1, d_Hred, jj + 1, d_gred, 1));
        CHECK_CUDA(cudaDeviceSynchronize());
        // writes g => y inplace

        // now copy back to the host
        T *h_y = new T[jj + 1];
        CHECK_CUDA(cudaMemcpy(h_y, d_gred, (jj + 1) * sizeof(T), cudaMemcpyDeviceToHost));

        if (debug && N <= 16) {
            printf("yred:");
            printVec<T>((jj + 1), h_y);
        }

        if constexpr (left_precond) {
            // now compute the matrix product soln = V * y one column at a time
            // zero solution (d_x is already zero)
            for (int j = 0; j < jj + 1; j++) {
                a = h_y[j];
                CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, &d_Vmat[j * N], 1, d_x, 1));
            }
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        if constexpr (right_precond) {
            // now compute matrix product xR = V * y (the preconditioned solution first)
            for (int j = 0; j < jj + 1; j++) {
                a = h_y[j];
                CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, &d_Vmat[j * N], 1, d_xR, 1));
            }
            CHECK_CUDA(cudaDeviceSynchronize());

            // then compute the solution from the preconditioned one x = M^-1 * xR
            // U^-1 L^-1 * xR => x precond solve here
            a = 1.0;
            // first L^-1 * w => tmp
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_L, mb, nnzb, &a,
                                                 descr_L, d_vals_ILU0, d_rowp, d_cols, block_dim,
                                                 info_L, d_xR, d_tmp, policy_L, pBuffer));
            CHECK_CUDA(cudaDeviceSynchronize());
            // U^-1 * tmp => into w
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, mb, nnzb, &a,
                                                 descr_U, d_vals_ILU0, d_rowp, d_cols, block_dim,
                                                 info_U, d_tmp, d_x, policy_U, pBuffer));
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        if (converged) break;

    }  // end of outer iterations

    // check final residual
    // --------------------

    // copy rhs into resid again
    CHECK_CUDA(cudaMemcpy(d_resid, d_rhs, N * sizeof(T), cudaMemcpyDeviceToDevice));

    // then subtract Ax from b
    a = 1.0, b = 0.0;
    CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a, descrA,
                                  d_vals, d_rowp, d_cols, block_dim, d_x, &b, d_tmp));
    CHECK_CUDA(cudaDeviceSynchronize());
    // resid -= A * x
    a = -1.0;
    CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_tmp, 1, d_resid, 1));
    CHECK_CUDA(cudaDeviceSynchronize());

    T final_resid;
    CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &final_resid));

    // debugging
    // int NPRINT = 100;
    // printf("x: ");
    // printVec<T>(NPRINT, soln.createHostVec().getPtr());
    // printf("b: ");
    // printVec<T>(NPRINT, rhs_perm.createHostVec().getPtr());
    // printf("A*x: ");
    // printVec<T>(NPRINT, DeviceVec<T>(N, d_tmp).createHostVec().getPtr());
    // printf("Ax-b: ");
    // printVec<T>(NPRINT, DeviceVec<T>(N, d_resid).createHostVec().getPtr());

    // now apply preconditioner to the residual
    if constexpr (left_precond) {
        // zero vec_tmp
        CHECK_CUDA(cudaMemset(d_tmp, 0.0, N * sizeof(T)));

        // ILU solve U^-1 L^-1 * b
        // L^-1 * b => tmp
        a = 1.0;
        CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_L, mb, nnzb, &a, descr_L,
                                             d_vals_ILU0, d_rowp, d_cols, block_dim, info_L,
                                             d_resid, d_tmp, policy_L, pBuffer));
        CHECK_CUDA(cudaDeviceSynchronize());

        // U^-1 * tmp => into rhs
        CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, mb, nnzb, &a, descr_U,
                                             d_vals_ILU0, d_rowp, d_cols, block_dim, info_U, d_tmp,
                                             d_resid, policy_U, pBuffer));
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    T final_precond_resid;
    CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &final_precond_resid));
    CHECK_CUDA(cudaDeviceSynchronize());

    if (can_print)
        printf("GMRES converged to %.4e resid, %.4e precond resid in %d iterations\n", final_resid,
               final_precond_resid, total_iter);

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

    CHECK_CUDA(cudaFree(d_resid));
    CHECK_CUDA(cudaFree(d_w));
    CHECK_CUDA(cudaFree(d_tmp));

    // print timing data
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    double dt = duration.count() / 1e6;
    if (can_print) {
        printf("\tfinished in %.4e sec\n", dt);
    }
}

};  // namespace CUSPARSE