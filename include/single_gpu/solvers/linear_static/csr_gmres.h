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

template <typename T, bool use_precond = true>
void GMRES_chol_solve(CsrMat<DeviceVec<T>> &mat, DeviceVec<T> &rhs, DeviceVec<T> &soln,
                      int _n_iter = 100, int max_iter = 100, T abs_tol = 1e-8, T rel_tol = 1e-8,
                      bool can_print = true, bool debug = false, int print_freq = 10) {
    /* GMRES CSR with Cholesky preconditioner and MGS orthog only written for left preconditioning
     * rn and zero inital guess, TBD can generalize, but then again it may be fairly slow compared
     * to BSR anyways so not worth it
     */

    static_assert(std::is_same<T, double>::value,
                  "Only double precision is written in our code for cuSparse GMRES CSR Cholesky");

    if (can_print) {
        printf("CSR GMRES solve with IC0\n");
    }
    auto start = std::chrono::high_resolution_clock::now();

    auto rhs_perm = inv_permute_rhs<CsrMat<DeviceVec<T>>, DeviceVec<T>>(mat, rhs);

    /* load CSR data of sparse matrix */
    BsrData bsr_data = mat.getBsrData();  // this is really CSRData
    int N = bsr_data.nnodes;
    int nnz = bsr_data.nnzb;
    int *d_rowp = bsr_data.rowp;
    int *d_cols = bsr_data.cols;
    int n_iter = min(_n_iter, bsr_data.nnodes);
    T *d_vals = mat.getPtr();  // matrix values (will be overwritten by IC0 factor)
    T *d_rhs = rhs_perm.getPtr();
    T *d_soln = soln.getPtr();
    T *d_y = DeviceVec<T>(N).getPtr();
    T *d_resid = DeviceVec<T>(N).getPtr();
    T *d_tmp = DeviceVec<T>(N).getPtr();
    T *d_w = DeviceVec<T>(N).getPtr();

    // create separate pointer for IC0 factorization
    T *d_vals_IC0 = DeviceVec<T>(mat.get_nnz()).getPtr();
    CHECK_CUDA(cudaMemcpy(d_vals_IC0, d_vals, mat.get_nnz() * sizeof(T), cudaMemcpyDeviceToDevice));

    /* create cuSPARSE objects */
    cusparseHandle_t cusparseHandle;
    cusparseCreate(&cusparseHandle);

    // make dense vecs in new cuSPARSE API
    cusparseDnVecDescr_t vecB, vecY, vecX, vecR, vecTMP, vecW;
    cusparseCreateDnVec(&vecB, N, d_rhs, CUDA_R_64F);
    cusparseCreateDnVec(&vecY, N, d_y, CUDA_R_64F);
    cusparseCreateDnVec(&vecX, N, d_soln, CUDA_R_64F);
    cusparseCreateDnVec(&vecR, N, d_resid, CUDA_R_64F);
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecTMP, N, d_tmp, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecW, N, d_w, CUDA_R_64F));

    // make prelim matrix and buffer objects
    cusparseSpMatDescr_t matL;
    cusparseSpSVDescr_t SpSV_L, SpSV_LT;
    size_t bufferSizeMV;
    void *buffer_L = nullptr, *buffer_LT = nullptr, *buffer_MV = nullptr;
    const T floatone = 1.0;
    const T floatzero = 0.0;

    /* Create CUBLAS context */
    cublasHandle_t cublasHandle = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    /* Description of the A matrix */
    cusparseSpMatDescr_t matA = NULL;
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, N, N, nnz, d_rowp, d_cols, d_vals, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    // setup A matrix for SpMV
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &floatone, matA, vecTMP, &floatzero, vecW, CUDA_R_64F,
                                           CUSPARSE_SPMV_ALG_DEFAULT, &bufferSizeMV));
    CHECK_CUDA(cudaMalloc(&buffer_MV, bufferSizeMV));

    // do main IC(0) incomplete Cholesky factorization with full IC pattern
    CUSPARSE::perform_ic0_factorization(cusparseHandle, matL, SpSV_L, SpSV_LT, N, nnz, d_rowp,
                                        d_cols, d_vals_IC0, vecB, vecY, vecX, buffer_L, buffer_LT);

    /* prelim for GMRES */
    T g[n_iter + 1], cs[n_iter], ss[n_iter];
    T H[(n_iter + 1) * (n_iter)];

    // GMRES device data
    T *d_Vmat, *d_V;
    CHECK_CUDA(cudaMalloc((void **)&d_Vmat, (n_iter + 1) * N * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void **)&d_V, N * sizeof(T)));
    // use single cusparseDnVecDescr_t of size N and just update it's values occasionally
    cusparseDnVecDescr_t vec_V;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vec_V, N, d_V, CUDA_R_64F));

    for (int iouter = 0; iouter < max_iter / n_iter; iouter++) {
        int jj = n_iter - 1;

        // precond application : L^-T L^-1 (b-Ax) => tmp but Ax = 0 assumed
        T alpha = 1.0;
        CHECK_CUSPARSE(cusparseSpSV_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                                          matL, vecB, vecY, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT,
                                          SpSV_L));
        CHECK_CUSPARSE(cusparseSpSV_solve(cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, &alpha,
                                          matL, vecY, vecR, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT,
                                          SpSV_LT));

        T beta;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &beta));
        printf("GMRES init resid = %.9e\n", beta);
        g[0] = beta;

        // set v0 = r0 / beta (unit vec)
        T a = 1.0 / beta;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_resid, 1, &d_Vmat[0], 1));

        // then begin main GMRES iteration loop!
        for (int j = 0; j < n_iter; j++) {
            // get vj and copy it into the cusparseDnVec_t
            void *vj_col = static_cast<void *>(&d_Vmat[j * N]);
            CHECK_CUSPARSE(cusparseDnVecSetValues(vec_V, vj_col));

            // w = A * vj + 0 * w
            CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone,
                                        matA, vec_V, &floatzero, vecW, CUDA_R_64F,
                                        CUSPARSE_SPMV_ALG_DEFAULT, buffer_MV));

            if constexpr (use_precond) {
                // L^-T L^-1 w => w, apply precond
                CHECK_CUSPARSE(cusparseSpSV_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  &alpha, matL, vecW, vecTMP, CUDA_R_64F,
                                                  CUSPARSE_SPSV_ALG_DEFAULT, SpSV_L));
                CHECK_CUSPARSE(cusparseSpSV_solve(cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE,
                                                  &alpha, matL, vecTMP, vecW, CUDA_R_64F,
                                                  CUSPARSE_SPSV_ALG_DEFAULT, SpSV_LT));
            }

            // now update householder matrix
            for (int i = 0; i < j + 1; i++) {
                // H_{i,j} = <wj,vi>
                CHECK_CUBLAS(
                    cublasDdot(cublasHandle, N, d_w, 1, &d_Vmat[i * N], 1, &H[n_iter * i + j]));
                if (debug) printf("H[%d,%d] = %.9e\n", i, j, H[n_iter * i + j]);
                a = -H[n_iter * i + j];  // wj -= H_{i,j} vi
                CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, &d_Vmat[i * N], 1, d_w, 1));
            }

            // normalize wj => v_{j+1}
            T nrm_w;
            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_w, 1, &nrm_w));
            H[n_iter * (j + 1) + j] = nrm_w;  // H_{j+1,j} = |w|
            a = 1.0 / H[n_iter * (j + 1) + j];
            CHECK_CUBLAS(cublasDscal(cublasHandle, N, &a, d_w, 1));
            CHECK_CUBLAS(cublasDcopy(cublasHandle, N, d_w, 1, &d_Vmat[(j + 1) * N], 1));

            // apply previous Givens rotations to new Hessenberg column
            for (int i = 0; i < j; i++) {
                T temp = H[i * n_iter + j];
                H[n_iter * i + j] = cs[i] * H[n_iter * i + j] + ss[i] * H[n_iter * (i + 1) + j];
                H[n_iter * (i + 1) + j] = -ss[i] * temp + cs[i] * H[n_iter * (i + 1) + j];
            }

            // compute new Givens rotations
            T hx = H[n_iter * j + j];
            T hy = H[n_iter * (j + 1) + j];
            T r = hypot(hx, hy);  // always non-negative
            cs[j] = hx / r;
            ss[j] = hy / r;

            // apply new Givens rotation to RHS
            T g_temp = g[j];
            g[j] *= cs[j];
            g[j + 1] = -ss[j] * g_temp;

            // apply new Givens rotation to upper Hessenberg matrix to make this new column lie in
            // upper triangular part only
            H[n_iter * j + j] = cs[j] * H[n_iter * j + j] + ss[j] * H[n_iter * (j + 1) + j];
            H[n_iter * (j + 1) + j] = 0.0;

            // printf("GMRES iter %d : resid %.9e\n", j, nrm_w);
            if (j % print_freq == 0 and can_print)
                printf("GMRES [%d] : resid %.9e\n", j, abs(g[j + 1]));

            if (debug) printf("j=%d, g[j]=%.9e, g[j+1]=%.9e\n", j, g[j], g[j + 1]);

            if (abs(g[j + 1]) < (abs_tol + beta * rel_tol)) {
                printf("GMRES converged in %d iterations to %.9e resid\n", j + 1, g[j + 1]);
                jj = j;
                break;
            }
        }

        // now solve Hessenberg triangular system
        // only up to size jj+1 x jj+1 where we exited on iteration jj
        T *Hred = new T[(jj + 1) * (jj + 1)];
        for (int i = 0; i < jj + 1; i++) {
            for (int j = 0; j < jj + 1; j++) {
                // in-place transpose to be compatible with column-major cublasDtrsv later on
                Hred[(jj + 1) * i + j] = H[n_iter * j + i];

                // Hred[(jj+1) * i + j] = H[n_iter * i + j];
            }
        }

        // TODO : could make this use cleaner triangular solve on host with double for loop
        // see bsr_gmres.h

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
        // writes g => y inplace
        // now copy back to the host
        T *h_y = new T[jj + 1];
        CHECK_CUDA(cudaMemcpy(h_y, d_gred, (jj + 1) * sizeof(T), cudaMemcpyDeviceToHost));

        // now compute the matrix product soln = V * y one column at a time
        // zero solution (d_x is already zero)
        for (int j = 0; j < jj + 1; j++) {
            a = h_y[j];
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, &d_Vmat[j * N], 1, d_soln, 1));
        }

        // now compute the residual again
        // resid = b - A * x
        // resid = -1 * A * vj + 1 * w
        // T float_neg_one = -1.0;
        // CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        //                             &float_neg_one, matA, vec_rhs, &floatone, vec_tmp,
        //                             CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_bufferMV));
    }

    /* Cholesky triangular solves x = L^-T L^-1 b */

    // Cleanup
    cusparseDestroySpMat(matL);
    cusparseDestroyDnVec(vecB);
    cusparseDestroyDnVec(vecY);
    cusparseDestroyDnVec(vecX);
    cusparseSpSV_destroyDescr(SpSV_L);
    cusparseSpSV_destroyDescr(SpSV_LT);
    cudaFree(buffer_L);
    cudaFree(buffer_LT);
    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);
    cudaFree(d_vals_IC0);

    // permute soln again
    permute_soln<CsrMat<DeviceVec<T>>, DeviceVec<T>>(mat, soln);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    if (can_print) {
        printf("\tfinished in %.4e sec\n", duration.count() / 1e6);
    }
}

};  // namespace CUSPARSE