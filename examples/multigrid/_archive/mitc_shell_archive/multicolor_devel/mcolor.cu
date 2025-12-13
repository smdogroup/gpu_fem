/* develop fast block-GS multicolor using small matrix for testing purposes.. */
// 7 node test matrix with 2x2 block dim for multicoloring (aka 14 DOF).. that I've made up

#include <cusparse_v2.h>
#include "cublas_v2.h"
#include "cuda_utils.h"
#include "linalg/vec.h"
#include "solvers/linear_static/_cusparse_utils.h"

template <typename T>
void baseline_solve_single_color(const int icolor, T *h_defect, T *h_new_defect) {
    /* baseline to solve update for a single color with full LU pattern */
    // we'll want to match each color to the new fast version (but we'll only call one at a time)

    // start by setting up cusparse objects..
    cublasHandle_t cublasHandle = NULL;
    cusparseHandle_t cusparseHandle = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

    // printf("h_defect\n");
    // printVec<T>(14, h_defect);

    // copy the defect to the device
    T *d_defect = HostVec<T>(14, h_defect).createDeviceVec().getPtr();

    // make a new temp color defect array, and only set this color part NZ
    int start, stop;
    if (icolor == 0) {
        start = 0, stop = 6;
    } else if (icolor == 1) {
        start = 6, stop = 10;
    } else if (icolor == 2) {
        start = 10, stop = 14; 
    }
    int ncolor_vals = stop - start;

    // printf("start %d, stop %d\n", start, stop);

    // compare this result with python also..
    T *d_defect_color = DeviceVec<T>(14).getPtr();
    cudaMemcpy(&d_defect_color[start], &d_defect[start], ncolor_vals * sizeof(T), cudaMemcpyDeviceToDevice);

    // now make temp for soln change
    T *d_dsoln = DeviceVec<T>(14).getPtr();
    // get pointer for the color part of soln change
    T *d_dsoln_color = &d_dsoln[start];

    // printf("here1\n");

    // define the matrix..
    const int nnodes = 7;
    int h_rowp[nnodes + 1] = {0, 3, 4, 5, 8, 11, 14, 16};
    const int nnzb = 16;
    int h_colors[nnzb] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2};
    int h_rows[nnzb] =   {0, 0, 0, 1, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6};
    int h_cols[nnzb] =   {0, 3, 4, 1, 2, 0, 3, 6, 0, 4, 5, 0, 4, 5, 3, 6};
    const int block_dim = 2;
    const int nnz = nnzb * 4;

    T *h_vals = new T[nnz];
    memset(h_vals, 0.0, nnz * sizeof(T));
    int nnzb_diag = 7;
    int nnz_diag = nnzb_diag * 4;
    T *h_diag_vals = new T[nnz_diag];
    for (int inz = 0; inz < nnz; inz++) {
        int inzb = inz / 4; // block index (like on CSR pattern)
        int inner = inz % 4;
        int ix = inner / 2, iy = inner % 2; // inside each 2x2 block

        int inner_diag = ix == iy;
        T inner_val = 1.0 + 1.0 * inner_diag;

        int color = h_colors[inzb];
        int off_diag = h_rows[inzb] != h_cols[inzb];
        T csr_val = 3.0 - color - 0.4 * off_diag;

        h_vals[inz] = csr_val * inner_val;

        int block_row = h_rows[inzb], block_col = h_cols[inzb];
        if (block_row == block_col) {
            int diag_nnz = 4 * block_row + inner;
            h_diag_vals[diag_nnz] = h_vals[inz];
        }
    }

    // printf("h_vals: ");
    // printVec<T>(nnz, h_vals);

    // printf("here2\n");

    // the diag matrix only..
    int h_diag_rowp[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    int h_diag_cols[7] = {0, 1, 2, 3, 4, 5, 6};
    int diag_nnzb = 7;

    cusparseMatDescr_t descrKmat = 0; //, descrDinvMat = 0;
    void *pBuffer = 0;
    // CUSPARSE triang solve for Dinv as diag LU
    cusparseMatDescr_t descr_L = 0, descr_U = 0;
    bsrsv2Info_t info_L = 0, info_U = 0;
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE,
                              trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;

    // printf("here3\n");

    // pass onto device
    int *d_diag_rowp = HostVec<int>(8, h_diag_rowp).createDeviceVec().getPtr();
    int *d_diag_cols = HostVec<int>(7, h_diag_cols).createDeviceVec().getPtr();
    T *d_temp = DeviceVec<T>(14).getPtr();
    T *d_temp2 = DeviceVec<T>(14).getPtr();
    T *d_temp3 = DeviceVec<T>(14).getPtr();
    T *d_kmat_vals = HostVec<T>(nnz, h_vals).createDeviceVec().getPtr();
    T *d_diag_LU_vals = HostVec<T>(4 * diag_nnzb, h_diag_vals).createDeviceVec().getPtr();
    // printf("h_vals:");
    // printVec<T>(nnz, h_vals);
    int *d_rowp = HostVec<int>(8, h_rowp).createDeviceVec().getPtr();
    int *d_cols = HostVec<int>(nnzb, h_cols).createDeviceVec().getPtr();
    T *d_new_defect = DeviceVec<T>(14).getPtr();

    // printf("here4\n");

    // ILU(0) factor of the diag matrix..
    CUSPARSE::perform_ilu0_factorization(cusparseHandle, descr_L, descr_U, info_L, info_U,
                                                 &pBuffer, nnodes, diag_nnzb, block_dim,
                                                 d_diag_LU_vals, d_diag_rowp, d_diag_cols, trans_L,
                                                 trans_U, policy_L, policy_U, dir);

    CHECK_CUDA(cudaDeviceSynchronize());
    // printf("here5\n");

    // T *h_diag_LU_vals = new T[4 * diag_nnzb];
    // cudaMemcpy(h_diag_LU_vals, d_diag_LU_vals, 4 * diag_nnzb * sizeof(T), cudaMemcpyDeviceToHost);
    // printf("h_diag_LU_vals: ");
    // printVec<T>(4 * diag_nnzb, h_diag_LU_vals);


    // now do LU solves
    T a = 1.0, b = 0.0;
    const double alpha = 1.0;
    CHECK_CUSPARSE(cusparseDbsrsv2_solve(
        cusparseHandle, dir, trans_L, nnodes, diag_nnzb, &alpha, descr_L,
        d_diag_LU_vals, d_diag_rowp, d_diag_cols, block_dim, info_L, d_defect_color, d_temp,
        policy_L, pBuffer));  
    CHECK_CUDA(cudaDeviceSynchronize());

    // T *h_temp = new T[14];
    // cudaMemcpy(h_temp, d_temp, 14 * sizeof(T), cudaMemcpyDeviceToHost);
    // printf("h_temp: ");
    // printVec<T>(14, h_temp);

    // printf("here6\n");
    CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, nnodes,
                                    diag_nnzb, &alpha, descr_U, d_diag_LU_vals,
                                        d_diag_rowp, d_diag_cols, block_dim, info_U,
                                        d_temp, d_dsoln, policy_U, pBuffer));
    CHECK_CUDA(cudaDeviceSynchronize());
    // printf("here7\n");

    // T *h_dsoln = new T[14];
    // cudaMemcpy(h_dsoln, d_dsoln, 14 * sizeof(T), cudaMemcpyDeviceToHost);
    // printf("\th_dsoln: ");
    // printVec<T>(14, h_dsoln);

    // 3) updated defect with soln
    cusparseCreateMatDescr(&descrKmat);
    cusparseSetMatIndexBase(descrKmat, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descrKmat, CUSPARSE_MATRIX_TYPE_GENERAL);

    a = -1.0,
    b = 1.0;  // so that defect := defect - mat*vec
    CHECK_CUSPARSE(cusparseDbsrmv(
        cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
        nnodes, nnodes, nnzb, &a, descrKmat, d_kmat_vals, d_rowp, d_cols,
        block_dim, d_dsoln, &b, d_new_defect));
    // CHECK_CUDA(cudaDeviceSynchronize());
    // printf("here8\n");

    // now copy new defect to output..
    cudaMemcpy(h_new_defect, d_new_defect, 14 * sizeof(T), cudaMemcpyDeviceToHost);
    // CHECK_CUDA(cudaDeviceSynchronize());
    // printf("here9\n");
}

template <typename T>
void fast_solve_single_color(const int icolor, T *h_defect, T *h_new_defect) {
    /* faster version to solve update of a single color (of multicolor block-GS) */

    // start by setting up cusparse objects..
    cublasHandle_t cublasHandle = NULL;
    cusparseHandle_t cusparseHandle = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

    // printf("h_defect\n");
    // printVec<T>(14, h_defect);

    // copy the defect to the device
    T *d_defect = HostVec<T>(14, h_defect).createDeviceVec().getPtr();

    // make a new temp color defect array, and only set this color part NZ
    int start, stop;
    if (icolor == 0) {
        start = 0, stop = 6;
    } else if (icolor == 1) {
        start = 6, stop = 10;
    } else if (icolor == 2) {
        start = 10, stop = 14; 
    }
    int ncolor_vals = stop - start;

    // printf("start %d, stop %d\n", start, stop);

    // compare this result with python also..
    T *d_defect_color = DeviceVec<T>(14).getPtr();
    cudaMemcpy(&d_defect_color[start], &d_defect[start], ncolor_vals * sizeof(T), cudaMemcpyDeviceToDevice);

    // now make temp for soln change
    T *d_dsoln = DeviceVec<T>(14).getPtr();
    // get pointer for the color part of soln change
    T *d_dsoln_color = &d_dsoln[start];

    // printf("here1\n");

    // define the matrix..
    const int nnodes = 7;
    int h_rowp[nnodes + 1] = {0, 3, 4, 5, 8, 11, 14, 16};
    const int nnzb = 16;
    int h_colors[nnzb] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2};
    int h_rows[nnzb] =   {0, 0, 0, 1, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6};
    int h_cols[nnzb] =   {0, 3, 4, 1, 2, 0, 3, 6, 0, 4, 5, 0, 4, 5, 3, 6};
    const int block_dim = 2;
    const int nnz = nnzb * 4;

    int h_kmat_nnzb_color_starts[4] = {0, 5, 11, 16};

    T *h_vals = new T[nnz];
    memset(h_vals, 0.0, nnz * sizeof(T));
    int nnzb_diag = 7;
    int nnz_diag = nnzb_diag * 4;
    T *h_diag_vals = new T[nnz_diag];
    for (int inz = 0; inz < nnz; inz++) {
        int inzb = inz / 4; // block index (like on CSR pattern)
        int inner = inz % 4;
        int ix = inner / 2, iy = inner % 2; // inside each 2x2 block

        int inner_diag = ix == iy;
        T inner_val = 1.0 + 1.0 * inner_diag;

        int color = h_colors[inzb];
        int off_diag = h_rows[inzb] != h_cols[inzb];
        T csr_val = 3.0 - color - 0.4 * off_diag;

        h_vals[inz] = csr_val * inner_val;

        int block_row = h_rows[inzb], block_col = h_cols[inzb];
        if (block_row == block_col) {
            int diag_nnz = 4 * block_row + inner;
            h_diag_vals[diag_nnz] = h_vals[inz];
        }
    }

    // printf("h_vals: ");
    // printVec<T>(nnz, h_vals);

    // printf("here2\n");

    // the diag matrix only..
    int h_diag_rowp[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    int h_diag_cols[7] = {0, 1, 2, 3, 4, 5, 6};
    int diag_nnzb = 7;

    cusparseMatDescr_t descrKmat = 0; //, descrDinvMat = 0;
    void *pBuffer = 0;
    // CUSPARSE triang solve for Dinv as diag LU
    cusparseMatDescr_t descr_L = 0, descr_U = 0;
    bsrsv2Info_t info_L = 0, info_U = 0;
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE,
                              trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;

    // printf("here3\n");

    // pass onto device
    int *d_diag_rowp = HostVec<int>(8, h_diag_rowp).createDeviceVec().getPtr();
    int *d_diag_cols = HostVec<int>(7, h_diag_cols).createDeviceVec().getPtr();
    T *d_temp = DeviceVec<T>(14).getPtr();
    T *d_temp2 = DeviceVec<T>(14).getPtr();
    T *d_temp3 = DeviceVec<T>(14).getPtr();
    T *d_kmat_vals = HostVec<T>(nnz, h_vals).createDeviceVec().getPtr();
    T *d_diag_LU_vals = HostVec<T>(4 * diag_nnzb, h_diag_vals).createDeviceVec().getPtr();
    // printf("h_vals:");
    // printVec<T>(nnz, h_vals);
    int *d_rowp = HostVec<int>(8, h_rowp).createDeviceVec().getPtr();
    int *d_cols = HostVec<int>(nnzb, h_cols).createDeviceVec().getPtr();
    T *d_new_defect = DeviceVec<T>(14).getPtr();

    // printf("here4\n");

    // ILU(0) factor of the diag matrix..
    CUSPARSE::perform_ilu0_factorization(cusparseHandle, descr_L, descr_U, info_L, info_U,
                                                 &pBuffer, nnodes, diag_nnzb, block_dim,
                                                 d_diag_LU_vals, d_diag_rowp, d_diag_cols, trans_L,
                                                 trans_U, policy_L, policy_U, dir);

    CHECK_CUDA(cudaDeviceSynchronize());
    // printf("here5\n");

    // T *h_diag_LU_vals = new T[4 * diag_nnzb];
    // cudaMemcpy(h_diag_LU_vals, d_diag_LU_vals, 4 * diag_nnzb * sizeof(T), cudaMemcpyDeviceToHost);
    // printf("h_diag_LU_vals: ");
    // printVec<T>(4 * diag_nnzb, h_diag_LU_vals);


    // now do LU solves
    T a = 1.0, b = 0.0;
    const double alpha = 1.0;

    // can't skip L or U solve even if only block diag or BSR it seems, D_C^-1 @ d_C
    int nnodes_color = ncolor_vals / block_dim;
    int diag_nnzb_color = nnodes_color;
    int nnz_start = start * block_dim;

    CHECK_CUSPARSE(cusparseDbsrsv2_solve(
        cusparseHandle, dir, trans_L, nnodes_color, diag_nnzb_color, &alpha, descr_L,
        &d_diag_LU_vals[nnz_start], d_diag_rowp, d_diag_cols, block_dim, info_L, &d_defect_color[start], &d_temp[start],
        policy_L, pBuffer));  
    CHECK_CUDA(cudaDeviceSynchronize());

    // T *h_temp = new T[14];
    // cudaMemcpy(h_temp, d_temp, 14 * sizeof(T), cudaMemcpyDeviceToHost);
    // printf("h_temp: ");
    // printVec<T>(14, h_temp);

    // printf("here6\n");
    CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, nnodes_color,
        diag_nnzb_color, &alpha, descr_U, &d_diag_LU_vals[nnz_start],
                                        d_diag_rowp, d_diag_cols, block_dim, info_U,
                                        &d_temp[start], &d_dsoln[start], policy_U, pBuffer));
    CHECK_CUDA(cudaDeviceSynchronize());
    // printf("here7\n");

    // T *h_dsoln = new T[14];
    // cudaMemcpy(h_dsoln, d_dsoln, 14 * sizeof(T), cudaMemcpyDeviceToHost);
    // printf("h_dsoln: ");
    // printVec<T>(14, h_dsoln);

    // 3) updated defect with soln
    cusparseCreateMatDescr(&descrKmat);
    cusparseSetMatIndexBase(descrKmat, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descrKmat, CUSPARSE_MATRIX_TYPE_GENERAL);

    a = -1.0,
    b = 1.0;  // so that defect := defect - mat*vec

    // old full matrix-mult
    // CHECK_CUSPARSE(cusparseDbsrmv(
    //     cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
    //     nnodes, nnodes, nnzb, &a, descrKmat, d_kmat_vals, d_rowp, d_cols,
    //     block_dim, d_dsoln, &b, d_new_defect));
    // CHECK_CUDA(cudaDeviceSynchronize());
    // printf("here8\n");

    // new matrix-mult use K^T and row-slicing, only those rows..
    int nz_start = h_kmat_nnzb_color_starts[icolor] * block_dim * block_dim;
    // get new rowp for kmat in these rows.. cols the same..
    int *h_rowp_color = new int[nnodes_color + 1];
    h_rowp_color[0] = 0;
    // int rowp_start = h_rowp[start];
    for (int inode = 0; inode < nnodes_color + 1; inode++) {
        h_rowp_color[inode] = h_rowp[inode] - h_rowp[start];
    }
    int *d_rowp_color = HostVec<int>(nnodes_color + 1, h_rowp_color).createDeviceVec().getPtr();
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("h_rowp_color for color %d: ", icolor);
    printVec<int>(nnodes_color + 1, h_rowp_color);
    int nnzb_color = h_kmat_nnzb_color_starts[icolor + 1] - h_kmat_nnzb_color_starts[icolor];
    printf("nnzb color %d, nnodes_color %d\n", nnzb_color, nnodes_color);
    printf("nz start %d\n", nz_start);

    // // K[color_slice, :]^T * dx_color => defect update (full size defect), CUSPARSE_DIRECTION_COLUMN to make it do transpose?
    // CUSPARSE_NON_TRANSPOSE (only is supported)
    CHECK_CUSPARSE(cusparseDbsrmv(
        cusparseHandle, CUSPARSE_DIRECTION_COLUMN, CUSPARSE_OPERATION_NON_TRANSPOSE,
        nnodes_color, nnodes, nnzb_color, &a, descrKmat, &d_kmat_vals[nz_start], d_rowp_color, d_cols,
        block_dim, &d_dsoln[start], &b, d_new_defect));

    // now copy new defect to output..
    cudaMemcpy(h_new_defect, d_new_defect, 14 * sizeof(T), cudaMemcpyDeviceToHost);
    // CHECK_CUDA(cudaDeviceSynchronize());
    // printf("here9\n");
}

int main() {
    using T = double;
    T *h_defect = new T[14];
    T *h_new_defect = new T[14];
    T *h_new_defect_fast = new T[14];
    for (int i = 0; i < 14; i++) {
        h_defect[i] = 1.234 - 2.189 * i;
    }

    /* maybe compare to python solves here also */
    // T h_defect_python[14] = {...};
    
    for (int icolor = 0; icolor < 3; icolor++) {
        baseline_solve_single_color<T>(icolor, h_defect, h_new_defect);
        fast_solve_single_color<T>(icolor, h_defect, h_new_defect_fast);

        printf("color %d, host defect baseline\n\t", icolor);
        printVec<T>(14, h_new_defect);
        printf("color %d, host defect fast\n\t", icolor);
        printVec<T>(14, h_new_defect_fast);
    }
    
};