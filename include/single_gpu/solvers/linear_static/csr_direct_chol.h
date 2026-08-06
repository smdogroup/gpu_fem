#pragma once

#include <assert.h>
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>

#include <iostream>

#include "_utils.h"
#include "chrono"
#include "cuda_utils.h"

namespace CUSPARSE {

template <typename T>
void direct_cholesky_solve(CsrMat<DeviceVec<T>> &mat, DeviceVec<T> &rhs, DeviceVec<T> &soln,
                           bool can_print = true) {
    /* a cuSparse variant of direct Cholesky CSR*/

    static_assert(std::is_same<T, double>::value,
                  "Only double precision is written in our code for cuSparse chol");

    if (can_print) {
        printf("direct Cholesky solve with IC0\n");
    }
    auto start = std::chrono::high_resolution_clock::now();

    auto rhs_perm = inv_permute_rhs<CsrMat<DeviceVec<T>>, DeviceVec<T>>(mat, rhs);

    /* load CSR data of sparse matrix */
    BsrData bsr_data = mat.getBsrData();  // this is really CSRData
    int N = bsr_data.nnodes;
    int nnz = bsr_data.nnzb;
    int *d_rowp = bsr_data.rowp;
    int *d_cols = bsr_data.cols;
    T *d_vals = mat.getPtr();  // matrix values (will be overwritten by IC0 factor)
    T *d_rhs = rhs_perm.getPtr();
    T *d_soln = soln.getPtr();
    DeviceVec<T> temp_vec(N);
    T *d_temp = temp_vec.getPtr();

    /* create cuSPARSE objects */
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // make dense vecs in new cuSPARSE API
    cusparseDnVecDescr_t vecB, vecY, vecX;
    cusparseCreateDnVec(&vecB, N, d_rhs, CUDA_R_64F);
    cusparseCreateDnVec(&vecY, N, d_temp, CUDA_R_64F);
    cusparseCreateDnVec(&vecX, N, d_soln, CUDA_R_64F);

    // make prelim matrix and buffer objects
    cusparseSpMatDescr_t matL;
    cusparseSpSVDescr_t SpSV_L, SpSV_LT;
    void *buffer_L, *buffer_LT;

    // do main IC(0) incomplete Cholesky factorization with full IC pattern
    CUSPARSE::perform_ic0_factorization(handle, matL, SpSV_L, SpSV_LT, N, nnz, d_rowp, d_cols,
                                        d_vals, vecB, vecY, vecX, buffer_L, buffer_LT);

    /* Cholesky triangular solves x = L^-T L^-1 b */
    T alpha = 1.0;
    cusparseSpSV_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matL, vecB, vecY,
                       CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, SpSV_L);
    cusparseSpSV_solve(handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, matL, vecY, vecX, CUDA_R_64F,
                       CUSPARSE_SPSV_ALG_DEFAULT, SpSV_LT);

    // Cleanup
    cusparseDestroySpMat(matL);
    cusparseDestroyDnVec(vecB);
    cusparseDestroyDnVec(vecY);
    cusparseDestroyDnVec(vecX);
    cusparseSpSV_destroyDescr(SpSV_L);
    cusparseSpSV_destroyDescr(SpSV_LT);
    cudaFree(buffer_L);
    cudaFree(buffer_LT);

    cusparseDestroy(handle);

    permute_soln<CsrMat<DeviceVec<T>>, DeviceVec<T>>(mat, soln);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    if (can_print) {
        printf("\tfinished in %.4e sec\n", duration.count() / 1e6);
    }
}  // end of direct CSR Cholesky solve

}  // namespace CUSPARSE