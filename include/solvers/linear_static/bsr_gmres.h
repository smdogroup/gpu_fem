#pragma once

#include <assert.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include <chrono>
#include <iostream>

#include "../../cuda_utils.h"
#include "_cusparse_utils.h"
#include "_utils.h"
#include "cublas_v2.h"

namespace CUSPARSE {

template <typename T, bool right = true, bool modifiedGS = true, bool use_precond = true>
void GMRES_solve(BsrMat<DeviceVec<T>> &mat, DeviceVec<T> &rhs, DeviceVec<T> &soln,
                 int _n_iter = 100, int max_iter = 500, T abs_tol = 1e-8, T rel_tol = 1e-8,
                 bool can_print = false, bool debug = false, int print_freq = 10,
                 bool permute_inout = true, T eta = 0.0) {
    /* GMRES iterative solve using a BsrMat on GPU with CUDA / CuSparse
        only supports T = double right now, may add float at some point (but float won't
       converge as deeply the residual, only about 1e-7) */
    static_assert(std::is_same<T, double>::value,
                  "Only double precision is written in our code for cuSparse BSR GMRES with "
                  "Modified Gram-Schmidt");

    T *d_rhs;
    if (permute_inout) {
        auto rhs_perm = inv_permute_rhs<BsrMat<DeviceVec<T>>, DeviceVec<T>>(mat, rhs);
        d_rhs = rhs_perm.getPtr();
    } else {
        d_rhs = rhs.getPtr();
    }

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

    // try adding a diagonal nugget here..
    // T eta = 1e-4;
    // T eta = 1e-6;
    // T eta = 1e-8;
    if (eta != 0.0) {
        printf("diag frac eta %.8e in use\n", eta);
    #ifdef USE_GPU
        dim3 block(32);
        int nblocks = (nnzb + block.x - 1) / block.x;
        dim3 grid(nblocks);

        // adds eta * I to the diag where eta > 0 is a scalar
        add_nodal_block_diag_kernel<T><<<grid, block>>>(nnzb, block_dim, d_vals_ILU0, eta);
        CHECK_CUDA(cudaDeviceSynchronize());
    #endif  // USE_GPU
    }
    

    // just add diagonal to precond only?, temp debug
    // int ndiag = mb * block_dim;
    // dim3 block(32);
    // int nblocks = (ndiag + block.x - 1) / block.x;
    // dim3 grid(nblocks);
    // // T eta = 1e7;
    // T eta = 1e-3;
    // // add_mat_diag_kernel<T><<<grid, block>>>(mb, block_dim, d_rowp, d_cols, d_vals_ILU0, eta);
    // mult_diag_kernel<T><<<grid, block>>>(mb, block_dim, d_rowp, d_cols, d_vals_ILU0, eta);

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
    // level scheduling makes it more parallel
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                                policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    // const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_USE_LEVEL,

    // tried changing both policy L and U to be USE_LEVEL not really a change
    // policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL,
    // policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE,
                              trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;

    T a = 1.0, b = 0.0;

    // perform the symbolic and numeric factorization of LU on given sparsity pattern
    if (debug) CHECK_CUDA(cudaDeviceSynchronize());
    auto start_factor = std::chrono::high_resolution_clock::now();

    CUSPARSE::perform_ilu0_factorization(cusparseHandle, descr_L, descr_U, info_L, info_U, &pBuffer,
                                         mb, nnzb, block_dim, d_vals_ILU0, d_rowp, d_cols, trans_L,
                                         trans_U, policy_L, policy_U, dir);

    if (debug) CHECK_CUDA(cudaDeviceSynchronize());
    auto end_factor = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> factor_time = end_factor - start_factor;
    if (can_print) printf("ILU(0) factor time in %.4e sec\n", factor_time.count());

    // main GMRES solve
    // ----------------

    // initialize GMRES data, some on host, some on GPU
    // host GMRES data
    T g[n_iter + 1], cs[n_iter], ss[n_iter];
    T H[(n_iter + 1) * (n_iter)];

    memset(H, 0.0, (n_iter + 1) * (n_iter) * sizeof(T));

    double *d_H;  // buffer for H
    cudaMalloc(&d_H, sizeof(double) * n_iter);
    double *h_H_buf = new double[n_iter];

    // GMRES device data
    T *d_Vmat;  // , *d_V;
    CHECK_CUDA(cudaMalloc((void **)&d_Vmat, (n_iter + 1) * N * sizeof(T)));
    // CHECK_CUDA(cudaMalloc((void **)&d_V, N * sizeof(T)));
    // cusparseDnVecDescr_t vec_V;
    // CHECK_CUSPARSE(cusparseCreateDnVec(&vec_V, N, d_V, CUDA_R_64F));

    bool converged = false;
    int total_iter = 0;

    double triang_time = 0.0, SpMV_time = 0.0, GS_time = 0.0;

    for (int iouter = 0; iouter < max_iter / n_iter; iouter++) {
        int jj = n_iter - 1;

        // apply precond to rhs if in use
        // copy b or rhs to resid
        CHECK_CUDA(cudaMemcpy(d_resid, d_rhs, N * sizeof(T), cudaMemcpyDeviceToDevice));
        a = 1.0, b = 0.0;

        // then subtract Ax from rhs
        if (debug) CHECK_CUDA(cudaDeviceSynchronize());
        auto start_mult = std::chrono::high_resolution_clock::now();
        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a, descrA,
                                      d_vals, d_rowp, d_cols, block_dim, d_x, &b, d_tmp));
        if (debug) CHECK_CUDA(cudaDeviceSynchronize());
        auto end_mult = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> spmv_time = end_mult - start_mult;
        SpMV_time += spmv_time.count();
        if (can_print && debug) {
            printf("\tSpMV on GPU in %.4e sec\n", spmv_time.count());
        }

        // resid -= A * x
        a = -1.0;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_tmp, 1, d_resid, 1));
        T init_true_resid;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &init_true_resid));

        // now apply precond to the resid
        if constexpr (left_precond) {
            if (debug) CHECK_CUDA(cudaDeviceSynchronize());
            auto start_triang = std::chrono::high_resolution_clock::now();

            // compute r0' = U^-1 L^-1 * r0
            a = 1.0;
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_L, mb, nnzb, &a,
                                                 descr_L, d_vals_ILU0, d_rowp, d_cols, block_dim,
                                                 info_L, d_resid, d_tmp, policy_L, pBuffer));
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, mb, nnzb, &a,
                                                 descr_U, d_vals_ILU0, d_rowp, d_cols, block_dim,
                                                 info_U, d_tmp, d_resid, policy_U, pBuffer));

            if (debug) CHECK_CUDA(cudaDeviceSynchronize());  // take these out for speedup
            auto end_triang = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> triang_time_loc = end_triang - start_triang;
            triang_time += triang_time_loc.count();
            if (can_print && debug)
                printf("\ttriang time loc %.4e, total %.4e\n", triang_time_loc.count(),
                       triang_time);
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
        if (debug) CHECK_CUDA(cudaDeviceSynchronize());
        if (can_print)
            printf("GMRES init resid = true %.9e, precond %.9e\n", init_true_resid, beta);
        g[0] = beta;

        // set v0 = r0 / beta (unit vec)
        a = 1.0 / beta;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_resid, 1, &d_Vmat[0], 1));

        // then begin main GMRES iteration loop!
        for (int j = 0; j < n_iter; j++, total_iter++) {
            if constexpr (right_precond) {
                // U^-1 L^-1 * vj => vj precond solve here
                a = 1.0;
                if (debug) CHECK_CUDA(cudaDeviceSynchronize());
                auto start_triang = std::chrono::high_resolution_clock::now();
                CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                    cusparseHandle, dir, trans_L, mb, nnzb, &a, descr_L, d_vals_ILU0, d_rowp,
                    d_cols, block_dim, info_L, &d_Vmat[j * N], d_tmp, policy_L, pBuffer));
                CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                    cusparseHandle, dir, trans_U, mb, nnzb, &a, descr_U, d_vals_ILU0, d_rowp,
                    d_cols, block_dim, info_U, d_tmp, d_tmp2, policy_U, pBuffer));
                if (debug) CHECK_CUDA(cudaDeviceSynchronize());
                auto end_triang = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> triang_time_loc = end_triang - start_triang;
                triang_time += triang_time_loc.count();

                // w = A * vj + 0 * w
                // BSR matrix multiply here MV
                a = 1.0, b = 0.0;
                CHECK_CUSPARSE(cusparseDbsrmv(
                    cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, mb,
                    mb, nnzb, &a, descrA, d_vals, d_rowp, d_cols, block_dim, d_tmp2, &b, d_w));
            }

            if constexpr (left_precond) {
                if (debug) CHECK_CUDA(cudaDeviceSynchronize());
                auto start_mult = std::chrono::high_resolution_clock::now();

                // w = A * vj + 0 * w
                // BSR matrix multiply here MV
                a = 1.0, b = 0.0;
                CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                              CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a,
                                              descrA, d_vals, d_rowp, d_cols, block_dim,
                                              &d_Vmat[j * N], &b, d_w));
                if (debug) CHECK_CUDA(cudaDeviceSynchronize());

                auto end_mult = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> spmv_time = end_mult - start_mult;
                SpMV_time += spmv_time.count();
                if (can_print && debug && j % print_freq == 0) {
                    printf("\tSpMV on GPU in %.4e sec, total %.4e\n", spmv_time.count(), SpMV_time);
                }

                if (debug) CHECK_CUDA(cudaDeviceSynchronize());
                auto start_triang = std::chrono::high_resolution_clock::now();

                // U^-1 L^-1 * w => w precond solve here
                a = 1.0;
                CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                    cusparseHandle, dir, trans_L, mb, nnzb, &a, descr_L, d_vals_ILU0, d_rowp,
                    d_cols, block_dim, info_L, d_w, d_tmp, policy_L, pBuffer));
                CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                    cusparseHandle, dir, trans_U, mb, nnzb, &a, descr_U, d_vals_ILU0, d_rowp,
                    d_cols, block_dim, info_U, d_tmp, d_w, policy_U, pBuffer));

                if (debug) CHECK_CUDA(cudaDeviceSynchronize());
                auto end_triang = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> triang_time_loc = end_triang - start_triang;
                triang_time += triang_time_loc.count();
                if (can_print && debug && j % print_freq == 0) {
                    printf("\ttriang time loc %.4e, total %.4e\n", triang_time_loc.count(),
                           triang_time);
                }
            }

            if (debug) CHECK_CUDA(cudaDeviceSynchronize());
            auto start_GS = std::chrono::high_resolution_clock::now();

            // now update householder matrix
            for (int i = 0; i < j + 1; i++) {
                // compute w -= <w, vi> * vi
                CHECK_CUBLAS(
                    cublasDdot(cublasHandle, N, d_w, 1, &d_Vmat[i * N], 1, &H[n_iter * i + j]));
                // if (debug) printf("H[%d,%d] = %.9e\n", i, j, H[n_iter * i + j]);
                a = -H[n_iter * i + j];
                CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, &d_Vmat[i * N], 1, d_w, 1));
            }

            if (debug) CHECK_CUDA(cudaDeviceSynchronize());
            auto end_GS = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> GS_loc = end_GS - start_GS;
            GS_time += GS_loc.count();
            // if (can_print && debug && j % print_freq == 0) {
            //     printf("\tGS_time_loc %.4e, total %.4e\n", GS_loc.count(), GS_time);
            // }

            // norm of w => H_{j+1,j}
            T nrm_w;
            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_w, 1, &H[n_iter * (j + 1) + j]));

            // v_{j+1} column unit vec = w / H_{j+1,j}
            a = 1.0 / H[n_iter * (j + 1) + j];
            CHECK_CUBLAS(cublasDcopy(cublasHandle, N, d_w, 1, &d_Vmat[(j + 1) * N], 1));
            CHECK_CUBLAS(cublasDscal(cublasHandle, N, &a, &d_Vmat[(j + 1) * N], 1));

            // then givens rotations to elim householder matrix
            for (int i = 0; i < j; i++) {
                T temp = H[i * n_iter + j];
                H[n_iter * i + j] = cs[i] * H[n_iter * i + j] + ss[i] * H[n_iter * (i + 1) + j];
                H[n_iter * (i + 1) + j] = -ss[i] * temp + cs[i] * H[n_iter * (i + 1) + j];
            }

            // T Hjj = H[n_iter * j + j], Hj1j = H[n_iter * (j + 1) + j];
            // cs[j] = Hjj / sqrt(Hjj * Hjj + Hj1j * Hj1j);
            // ss[j] = cs[j] * Hj1j / Hjj;

            T hx = H[n_iter * j + j];        // + 1e-12;
            T hy = H[n_iter * (j + 1) + j];  // + 1e-12;
            // cs[j - 1] = hx / sqrt(hx * hx + hy * hy);
            // ss[j - 1] = cs[j - 1] * hy / hx;
            T r = hypot(hx, hy);  // always non-negative
            cs[j] = hx / r;
            ss[j] = hy / r;

            T g_temp = g[j];
            g[j] *= cs[j];
            g[j + 1] = -ss[j] * g_temp;

            // printf("GMRES iter %d : resid %.9e\n", j, nrm_w);
            if (can_print && (j % print_freq == 0))
                printf("GMRES iter %d : resid %.9e\n", j, abs(g[j + 1]));

            // if (debug) printf("j=%d, g[j]=%.9e, g[j+1]=%.9e\n", j, g[j], g[j + 1]);

            H[n_iter * j + j] = r;
            H[n_iter * (j + 1) + j] = 0.0;

            if (abs(g[j + 1]) < (abs_tol + beta * rel_tol)) {
                if (can_print)
                    printf("GMRES converged in %d iterations to %.9e resid\n", j + 1, g[j + 1]);
                jj = j;
                converged = true;
                break;
            }
        }  // end of inner loop

        // now solve Hessenberg triangular system
        // only up to size jj+1 x jj+1 where we exited on iteration jj
        T *Hred = new T[(jj + 1) * (jj + 1)];
        // TODO : this doesn't transpose, rewrite it like new deflated GMRES
        for (int i = 0; i < jj + 1; i++) {
            for (int j = 0; j < jj + 1; j++) {
                // in-place transpose to be compatible with column-major cublasDtrsv later on
                Hred[(jj + 1) * i + j] = H[n_iter * j + i];

                // Hred[(jj+1) * i + j] = H[n_iter * i + j];
            }
        }

        // now print out Hred
        // if (debug) {
        //     printf("Hred:");
        //     printVec<T>((jj + 1) * (jj + 1), Hred);
        //     printf("gred:");
        //     printVec<T>((jj + 1), g);
        // }

        // TODO : switch to solving this part on the GPU
        // now copy data from Hred host to device
        T *d_Hred;
        CHECK_CUDA(cudaMalloc(&d_Hred, (jj + 1) * (jj + 1) * sizeof(T)));
        CHECK_CUDA(
            cudaMemcpy(d_Hred, Hred, (jj + 1) * (jj + 1) * sizeof(T), cudaMemcpyHostToDevice));

        // also create gred vector on the device
        T *d_gred;
        CHECK_CUDA(cudaMalloc(&d_gred, (jj + 1) * sizeof(T)));
        CHECK_CUDA(cudaMemcpy(d_gred, g, (jj + 1) * sizeof(T), cudaMemcpyHostToDevice));

        // now solve Hessenberg system H * y = g
        // T *d_y;
        // CHECK_CUDA(cudaMalloc(&d_y, (jj+1) * sizeof(T)));
        CHECK_CUBLAS(cublasDtrsv(cublasHandle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                                 CUBLAS_DIAG_NON_UNIT, jj + 1, d_Hred, jj + 1, d_gred, 1));

        // writes g => y inplace

        // now copy back to the host
        T *h_y = new T[jj + 1];
        CHECK_CUDA(cudaMemcpy(h_y, d_gred, (jj + 1) * sizeof(T), cudaMemcpyDeviceToHost));

        if constexpr (left_precond) {
            // now compute the matrix product soln = V * y one column at a time
            // zero solution (d_x is already zero)
            for (int j = 0; j < jj + 1; j++) {
                a = h_y[j];
                CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, &d_Vmat[j * N], 1, d_x, 1));
            }
        }

        if constexpr (right_precond) {
            // zero xR
            CHECK_CUDA(cudaMemset(d_xR, 0.0, N * sizeof(T)));

            // now compute matrix product xR = V * y (the preconditioned solution first)
            for (int j = 0; j < jj + 1; j++) {
                a = h_y[j];
                CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, &d_Vmat[j * N], 1, d_xR, 1));
            }

            // then compute xR = M^-1 xR (un-preconditions it back to x space)
            a = 1.0;
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_L, mb, nnzb, &a,
                                                 descr_L, d_vals_ILU0, d_rowp, d_cols, block_dim,
                                                 info_L, d_xR, d_tmp, policy_L, pBuffer));
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, mb, nnzb, &a,
                                                 descr_U, d_vals_ILU0, d_rowp, d_cols, block_dim,
                                                 info_U, d_tmp, d_xR, policy_U, pBuffer));

            // then update x = x_0 + xR
            a = 1.0;
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_xR, 1, d_x, 1));
        }

        if (converged) break;

    }  // end of outer iterations

    if (debug)
        printf("GMRES: triang time %.4e, SpMV time %.4e, GS_time %.4e\n", triang_time, SpMV_time,
               GS_time);

    // check final residual
    // --------------------

    // copy rhs into resid again
    CHECK_CUDA(cudaMemcpy(d_resid, d_rhs, N * sizeof(T), cudaMemcpyDeviceToDevice));

    // then subtract Ax from b
    a = 1.0, b = 0.0;
    CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a, descrA,
                                  d_vals, d_rowp, d_cols, block_dim, d_x, &b, d_tmp));

    // resid -= A * x
    a = -1.0;
    CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_tmp, 1, d_resid, 1));

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

        // U^-1 * tmp => into rhs
        CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, mb, nnzb, &a, descr_U,
                                             d_vals_ILU0, d_rowp, d_cols, block_dim, info_U, d_tmp,
                                             d_resid, policy_U, pBuffer));
    }

    T final_precond_resid;
    CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &final_precond_resid));

    if (can_print)
        printf("GMRES converged to %.4e resid, %.4e precond resid in %d iterations\n", final_resid,
               final_precond_resid, total_iter);

    // cleanup and inverse permute the solution for exit
    // -------------------------------------------------

    // now also inverse permute the soln data
    if (permute_inout) {
        permute_soln<BsrMat<DeviceVec<T>>, DeviceVec<T>>(mat, soln);
    }

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
    CHECK_CUDA(cudaFree(d_vals_ILU0));

    // print timing data
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    double dt = duration.count() / 1e6;
    if (can_print) {
        printf("\tfinished in %.4e sec\n", dt);
    }
}  // end of GMRES BSR

};  // namespace CUSPARSE