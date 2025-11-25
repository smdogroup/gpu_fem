#pragma once

#include <assert.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include <iostream>

#include "../../cuda_utils.h"
#include "_cusparse_utils.h"
#include "_utils.h"
#include "chrono"
#include "cublas_v2.h"

namespace CUSPARSE {

template <typename T>
void direct_LU_solve(BsrMat<DeviceVec<T>> &mat, DeviceVec<T> &rhs, DeviceVec<T> &soln,
                     bool can_print = true, bool permute_inout = true) {
    /* direct LU solve => performs LU factorization then L and U triangular solves.
        Best for solving with same matrix and multiple rhs vectors such as aeroelastic + linear
       structures Although need to check LU factorization held in place correctly and boolean to not
       refactorize maybe */

    static_assert(std::is_same<T, double>::value,
                  "Only double precision is written in our code for cuSparse direct LU solve");

    T *d_rhs;
    if (permute_inout) {
        auto rhs_perm = inv_permute_rhs<BsrMat<DeviceVec<T>>, DeviceVec<T>>(mat, rhs);
        d_rhs = rhs_perm.getPtr();
    } else {
        d_rhs = rhs.getPtr();
    }

    if (can_print) {
        printf("direct LU cusparse solve\n");
    }
    // auto start = std::chrono::high_resolution_clock::now();

    // copy important inputs for Bsr structure out of BsrMat
    // TODO : was trying to make some of these const but didn't accept it in
    // final solve
    BsrData bsr_data = mat.getBsrData();
    int mb = bsr_data.nnodes;
    int nnzb = bsr_data.nnzb;
    int block_dim = bsr_data.block_dim;
    index_t *d_rowp = bsr_data.rowp;
    index_t *d_cols = bsr_data.cols;
    T *d_soln = soln.getPtr();
    DeviceVec<T> temp = DeviceVec<T>(soln.getSize());
    T *d_temp = temp.getPtr();

    // note this changes the mat data to be LU (but that's the whole point
    // of LU solve is for repeated linear solves we now just do triangular
    // solves)
    T *d_vals = mat.getPtr();
    // T *d_vals_ILU0 = d_vals;

    T *d_vals_ILU0 = DeviceVec<T>(mat.get_nnz()).getPtr();
    // ILU0 equiv to ILU(k) if sparsity pattern has ILU(k)
    CHECK_CUDA(
        cudaMemcpy(d_vals_ILU0, d_vals, mat.get_nnz() * sizeof(T), cudaMemcpyDeviceToDevice));

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
    CUSPARSE::perform_ilu0_factorization(handle, descr_L, descr_U, info_L, info_U, &pBuffer, mb,
                                         nnzb, block_dim, d_vals_ILU0, d_rowp, d_cols, trans_L,
                                         trans_U, policy_L, policy_U, dir);

    // // cehck LU factor (TEMP DEBUG)
    // int *h_rowp = DeviceVec<int>(mb + 1, d_rowp).createHostVec().getPtr();
    // int *h_cols = DeviceVec<int>(nnzb, d_cols).createHostVec().getPtr();
    // int *h_perm = DeviceVec<int>(mb, bsr_data.perm).createHostVec().getPtr();
    // T *h_vals0 = DeviceVec<T>(121 * nnzb, d_vals).createHostVec().getPtr();
    // T *h_vals_ILU0 = DeviceVec<T>(121 * nnzb, d_vals_ILU0).createHostVec().getPtr();
    // printf("block_dim %d with %d nodes\n", block_dim, mb);
    // for (int _row = 0; _row < mb; _row++) {
    //     for (int jp = h_rowp[_row]; jp < h_rowp[_row + 1]; jp++) {
    //         int _col = h_cols[jp];
    //         // unpermute from solve to vis order
    //         int row = h_perm[_row], col = h_perm[_col];

    //         printf("row %d from _row %d\n", row, _row);

    //         printf("h_vals0 of nodes (%d,%d): \n", row, col);
    //         for (int i = 0; i < 11; i++) {
    //             printVec<T>(11, &h_vals0[121 * jp + 11 * i]);
    //         }
    //         printf("\n--------\n");

    //         printf("h_vals_ILU0 of nodes (%d,%d): \n", row, col);
    //         for (int i = 0; i < 11; i++) {
    //             printVec<T>(11, &h_vals_ILU0[121 * jp + 11 * i]);
    //         }
    //         printf("\n--------\n");
    //     }
    // }

    // temp debug, time the triang solves only
    // cudaDeviceSynchronize();
    // auto start = std::chrono::high_resolution_clock::now();

    // triangular solve L*z = x
    const double alpha = 1.0;
    CHECK_CUSPARSE(cusparseDbsrsv2_solve(handle, dir, trans_L, mb, nnzb, &alpha, descr_L,
                                         d_vals_ILU0, d_rowp, d_cols, block_dim, info_L, d_rhs,
                                         d_temp, policy_L, pBuffer));

    // T *h_temp = DeviceVec<T>(11 * mb, d_temp).createHostVec().getPtr();
    // printf("h_temp1: ");
    // for (int _inode = 0; _inode < mb; _inode++) {
    //     int inode = h_perm[_inode];
    //     printf("h_temp1 of node %d\n", inode);
    //     printVec<T>(11, &h_temp[11 * _inode]);
    // }

    // triangular solve U*y = z
    CHECK_CUSPARSE(cusparseDbsrsv2_solve(handle, dir, trans_U, mb, nnzb, &alpha, descr_U,
                                         d_vals_ILU0, d_rowp, d_cols, block_dim, info_U, d_temp,
                                         d_soln, policy_U, pBuffer));

    // T *h_temp2 = DeviceVec<T>(11 * mb, d_soln).createHostVec().getPtr();
    // for (int _inode = 0; _inode < mb; _inode++) {
    //     int inode = h_perm[_inode];
    //     printf("h_temp2 of node %d\n", inode);
    //     printVec<T>(11, &h_temp2[11 * _inode]);
    // }

    // print timing data
    // cudaDeviceSynchronize();
    // auto stop = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    // double dt = duration.count() / 1e6;
    // if (can_print) {
    //     printf("\tfinished in %.4e sec\n", dt);
    // }

    // free resources
    cudaFree(pBuffer);
    cudaFree(d_vals_ILU0);
    cudaFree(d_temp);
    cudaFree(d_rhs);
    cusparseDestroyMatDescr(descr_L);
    cusparseDestroyMatDescr(descr_U);
    cusparseDestroyBsrsv2Info(info_L);
    cusparseDestroyBsrsv2Info(info_U);
    cusparseDestroy(handle);

    // now also inverse permute the soln data
    if (permute_inout) {
        permute_soln<BsrMat<DeviceVec<T>>, DeviceVec<T>>(mat, soln);
    }
}
}  // namespace CUSPARSE