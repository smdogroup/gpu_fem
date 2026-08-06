#pragma once

#include <assert.h>
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>

#include <iostream>

#include "_utils.h"
#include "chrono"
#include "cuda_utils.h"

namespace CUSOLVER {

/* a cuSolver variant of direct Cholesky CSR*/
template <typename T>
void direct_cholesky_solve(CsrMat<DeviceVec<T>> &mat, DeviceVec<T> &rhs, DeviceVec<T> &soln,
                           bool can_print = true) {
    static_assert(std::is_same<T, double>::value,
                  "Only double precision is written in our code for cuSolver chol");

    if (can_print) {
        printf("begin cuSolver direct Cholesky solve\n");
    }
    auto start = std::chrono::high_resolution_clock::now();

    auto rhs_perm = inv_permute_rhs<CsrMat<DeviceVec<T>>, DeviceVec<T>>(mat, rhs);

    // Get CSR data
    BsrData bsr_data = mat.getBsrData();
    int N = bsr_data.nnodes;
    int nnz = bsr_data.nnzb;
    int *d_rowp = bsr_data.rowp;
    int *d_cols = bsr_data.cols;
    T *d_vals = mat.getPtr();
    T *d_rhs = rhs_perm.getPtr();
    T *d_soln = soln.getPtr();

    // cuSolver handle
    cusolverSpHandle_t solver_handle;
    cusolverSpCreate(&solver_handle);

    // Matrix descriptor
    cusparseMatDescr_t descrA;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_SYMMETRIC);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatFillMode(descrA, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_NON_UNIT);

    // Solve Ax = b using Cholesky
    int singularity = -1;
    double tol = 1e-14;

    cusolverSpDcsrlsvchol(solver_handle, N, nnz, descrA, d_vals, d_rowp, d_cols, d_rhs, tol, 0,
                          d_soln, &singularity);

    if (singularity != -1 && can_print) {
        printf("WARNING: Matrix is singular at row %d\n", singularity);
    }

    // Cleanup
    cusparseDestroyMatDescr(descrA);
    cusolverSpDestroy(solver_handle);

    permute_soln<CsrMat<DeviceVec<T>>, DeviceVec<T>>(mat, soln);

    auto stop = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration<double>(stop - start).count();

    if (can_print) {
        printf("\tfinished in %.4e sec\n", dt);
    }
}

}  // namespace CUSOLVER