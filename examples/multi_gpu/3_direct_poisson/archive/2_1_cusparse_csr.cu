// poisson_csr_lu_pattern_ic0_direct.cu
//
// Standalone performance test:
//   CSR Laplace -> host BSR -> AMD full symbolic LU BSR pattern -> CSR
//   -> cuSPARSE CSR csric02 factorization
//   -> direct-style Cholesky solve using SpSV: L y = b, L^T x = y.
//
// Pattern utilities come from include/poisson.h.
// No host pattern methods are redefined here.

#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "include/poisson.h"

#define CHECK_CUDA(x) do {                                                     \
    cudaError_t e = (x);                                                       \
    if (e != cudaSuccess) {                                                    \
        printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
        exit(1);                                                               \
    }                                                                          \
} while (0)

#define CHECK_CUSPARSE(x) do {                                                 \
    cusparseStatus_t s = (x);                                                  \
    if (s != CUSPARSE_STATUS_SUCCESS) {                                        \
        printf("cuSPARSE error %d at %s:%d\n", (int)s, __FILE__, __LINE__);   \
        exit(1);                                                               \
    }                                                                          \
} while (0)

static double computeResidualCSR(int N, const int *rowp, const int *cols,
                                 const double *vals, const double *x,
                                 const double *rhs) {
    double r2 = 0.0, b2 = 0.0;
    for (int i = 0; i < N; i++) {
        double Ax = 0.0;
        for (int jp = rowp[i]; jp < rowp[i + 1]; jp++) {
            Ax += vals[jp] * x[cols[jp]];
        }
        const double r = Ax - rhs[i];
        r2 += r * r;
        b2 += rhs[i] * rhs[i];
    }
    return std::sqrt(r2 / std::max(b2, 1e-300));
}

static void checkZeroPivotIC0(cusparseHandle_t handle, csric02Info_t info,
                              const char *label) {
    int pivot = -1;
    cusparseStatus_t status = cusparseXcsric02_zeroPivot(handle, info, &pivot);
    if (status == CUSPARSE_STATUS_ZERO_PIVOT) {
        printf("%s zero pivot at row %d\n", label, pivot);
    } else if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("cuSPARSE zero-pivot check error %d at %s\n", (int)status, label);
        exit(1);
    }
}


// Build an AMD-permuted full Cholesky lower pattern on the BSR block graph.
// This is derived from the full LU pattern computed by BsrData, then filtered to
// the lower triangular block part. The numerical values are copied from the
// permuted original matrix where present; fill entries are explicit zeros.
template <typename T>
void convertBSRtoCholeskyPattern(
    int block_dim,
    int N,
    int nnzb,
    const int *bsr_rowp,
    const int *bsr_cols,
    const T *bsr_vals,
    int **perm,
    int **iperm,
    int **rowp,
    int **cols,
    T **vals,
    int *chol_nnzb
) {
    assert(N % block_dim == 0);

    const int nnodes = N / block_dim;
    const int block_dim2 = block_dim * block_dim;

    BsrData bsr_data(
        nnodes,
        block_dim,
        nnzb,
        const_cast<int *>(bsr_rowp),
        const_cast<int *>(bsr_cols),
        nullptr,
        nullptr,
        true
    );

    bsr_data.AMD_reordering();
    bsr_data.compute_full_LU_pattern();

    *perm = new int[nnodes];
    *iperm = new int[nnodes];

    for (int i = 0; i < nnodes; i++) {
        (*perm)[i] = bsr_data.perm[i];    // permuted row -> original row
        (*iperm)[i] = bsr_data.iperm[i];  // original row -> permuted row
    }

    std::vector<int> h_rowp(nnodes + 1, 0);
    std::vector<int> h_cols;
    std::vector<T> h_vals;

    // The full LU pattern should contain the Cholesky fill graph.  For SPD
    // Cholesky, keep only pcol <= prow.
    for (int prow = 0; prow < nnodes; prow++) {
        h_rowp[prow] = (int)h_cols.size();

        const int orig_row = (*perm)[prow];

        for (int jp = bsr_data.rowp[prow]; jp < bsr_data.rowp[prow + 1]; jp++) {
            const int pcol = bsr_data.cols[jp];

            if (pcol > prow) {
                continue;
            }

            const int orig_col = (*perm)[pcol];

            h_cols.push_back(pcol);
            const int new_jp = (int)h_cols.size() - 1;

            for (int ii = 0; ii < block_dim2; ii++) {
                h_vals.push_back(T(0));
            }

            int old_jp = -1;
            for (int jp2 = bsr_rowp[orig_row]; jp2 < bsr_rowp[orig_row + 1]; jp2++) {
                if (bsr_cols[jp2] == orig_col) {
                    old_jp = jp2;
                    break;
                }
            }

            if (old_jp >= 0) {
                for (int ii = 0; ii < block_dim2; ii++) {
                    h_vals[block_dim2 * new_jp + ii] =
                        bsr_vals[block_dim2 * old_jp + ii];
                }
            }
        }
    }
    h_rowp[nnodes] = (int)h_cols.size();

    *chol_nnzb = (int)h_cols.size();

    *rowp = new int[nnodes + 1];
    *cols = new int[*chol_nnzb];
    *vals = new T[(size_t)block_dim2 * (*chol_nnzb)];

    std::copy(h_rowp.begin(), h_rowp.end(), *rowp);
    std::copy(h_cols.begin(), h_cols.end(), *cols);
    std::copy(h_vals.begin(), h_vals.end(), *vals);
}

// Convert a lower-block BSR Cholesky pattern to lower scalar CSR.
// For diagonal blocks, only scalar entries with col <= row are emitted.
// For strict lower off-diagonal blocks, the full dense block is emitted.
template <typename T>
void BSRLowerToCSR(
    int block_dim,
    int N,
    int nnzb,
    const int *bsr_rowp,
    const int *bsr_cols,
    const T *bsr_vals,
    int **csr_rowp,
    int **csr_cols,
    T **csr_vals,
    int *nz
) {
    assert(N % block_dim == 0);

    const int nbrows = N / block_dim;
    const int block_dim2 = block_dim * block_dim;

    std::vector<int> h_rowp;
    std::vector<int> h_cols;
    std::vector<T> h_vals;

    h_rowp.reserve(N + 1);
    h_rowp.push_back(0);

    for (int row = 0; row < N; row++) {
        const int brow = row / block_dim;
        const int inner_row = row % block_dim;

        for (int jp = bsr_rowp[brow]; jp < bsr_rowp[brow + 1]; jp++) {
            const int bcol = bsr_cols[jp];

            for (int inner_col = 0; inner_col < block_dim; inner_col++) {
                const int col = block_dim * bcol + inner_col;

                if (col > row) {
                    continue;
                }

                const int val_ind =
                    block_dim2 * jp + block_dim * inner_row + inner_col;

                h_cols.push_back(col);
                h_vals.push_back(bsr_vals[val_ind]);
            }
        }

        h_rowp.push_back((int)h_cols.size());
    }

    *nz = (int)h_cols.size();

    *csr_rowp = new int[N + 1];
    *csr_cols = new int[*nz];
    *csr_vals = new T[*nz];

    std::copy(h_rowp.begin(), h_rowp.end(), *csr_rowp);
    std::copy(h_cols.begin(), h_cols.end(), *csr_cols);
    std::copy(h_vals.begin(), h_vals.end(), *csr_vals);
}


int main(int argc, char **argv) {
    using T = double;

    int nx = 128;   // nodes per direction
    int nrepeat = 1;

    if (argc > 1) nx = std::atoi(argv[1]);
    if (argc > 2) nrepeat = std::atoi(argv[2]);

    const int block_dim = 6;
    const int block_dim2 = block_dim * block_dim;

    const int nnode = nx * nx;
    const int N = block_dim * nnode;

    const int nz_node = 5 * nnode - 4 * nx;
    const int nz = block_dim * nz_node;

    printf("nx = %d\n", nx);
    // printf("nnode = %d\n", nnode);
    printf("scalar N = %d\n", N);
    // printf("block_dim = %d\n", block_dim);

    int *csr_rowp = (int *)malloc((N + 1) * sizeof(int));
    int *csr_cols = (int *)malloc(nz * sizeof(int));
    T *csr_vals = (T *)malloc(nz * sizeof(T));
    T *rhs = (T *)malloc(N * sizeof(T));
    T *x = (T *)malloc(N * sizeof(T));

    std::fill(rhs, rhs + N, T(0));
    std::fill(x, x + N, T(0));

    genBlockDiagPoissonCSR<T>(
        csr_rowp,
        csr_cols,
        csr_vals,
        nnode,
        block_dim,
        rhs
    );

    // genLaplaceCSR gives negative definite Laplacian; flip sign so SPD for IC/Cholesky.
    for (int k = 0; k < nz; k++) csr_vals[k] *= T(-1);
    for (int i = 0; i < N; i++) rhs[i] *= T(-1);

    int *bsr_rowp = nullptr, *bsr_cols = nullptr, bsr_nnzb = 0;
    T *bsr_vals = nullptr;
    int *perm = nullptr, *iperm = nullptr;
    int *chol_bsr_rowp = nullptr, *chol_bsr_cols = nullptr, chol_bsr_nnzb = 0;
    T *chol_bsr_vals = nullptr;
    int *chol_csr_rowp = nullptr, *chol_csr_cols = nullptr, chol_csr_nnz = 0;
    T *chol_csr_vals = nullptr;

    const int mb = N / block_dim;

    auto tpat0 = std::chrono::high_resolution_clock::now();
    CSRtoBSR<T>(block_dim, N, csr_rowp, csr_cols, csr_vals,
                &bsr_rowp, &bsr_cols, &bsr_vals, &bsr_nnzb);
    auto tpat1 = std::chrono::high_resolution_clock::now();

    convertBSRtoCholeskyPattern<T>(block_dim, N, bsr_nnzb, bsr_rowp, bsr_cols, bsr_vals,
                                &perm, &iperm, &chol_bsr_rowp, &chol_bsr_cols,
                                &chol_bsr_vals, &chol_bsr_nnzb);
    auto tpat2 = std::chrono::high_resolution_clock::now();

    BSRLowerToCSR<T>(block_dim, N, chol_bsr_nnzb, chol_bsr_rowp, chol_bsr_cols, chol_bsr_vals,
                  &chol_csr_rowp, &chol_csr_cols, &chol_csr_vals, &chol_csr_nnz);
    auto tpat3 = std::chrono::high_resolution_clock::now();

    const double csr_to_bsr_ms = std::chrono::duration<double, std::milli>(tpat1 - tpat0).count();
    const double bsr_to_chol_ms = std::chrono::duration<double, std::milli>(tpat2 - tpat1).count();
    const double chol_bsr_to_csr_ms = std::chrono::duration<double, std::milli>(tpat3 - tpat2).count();
    const double pattern_ms = std::chrono::duration<double, std::milli>(tpat3 - tpat0).count();

    // printf("Original CSR: N=%d nnz=%d\n", N, nz);
    // printf("Original BSR: mb=%d block_dim=%d nnzb=%d scalar_nnz=%d\n",
    //        mb, block_dim, bsr_nnzb, bsr_nnzb * block_dim2);
    // printf("LU-pattern BSR: mb=%d block_dim=%d nnzb=%d scalar_nnz=%d fill_ratio=%.6f\n",
    //        mb, block_dim, lu_bsr_nnzb, lu_bsr_nnzb * block_dim2,
    //        (double)lu_bsr_nnzb / std::max(1, bsr_nnzb));
    // printf("LU-pattern CSR: N=%d nnz=%d\n", N, lu_csr_nnz);

    // printf("\nHost conversion timing:\n");
    // printf("  CSR -> BSR              : %.6f ms\n", csr_to_bsr_ms);
    // printf("  BSR -> LU pattern BSR   : %.6f ms\n", bsr_to_lu_ms);
    // printf("  LU pattern BSR -> CSR   : %.6f ms\n", lu_bsr_to_csr_ms);
    // printf("  host pattern total      : %.6f ms\n", pattern_ms);
    printf("\nHost conversion timing: CSR->BSR = %.6f ms, BSR->Cholesky-pattern BSR = %.6f ms, Cholesky-pattern BSR->lower CSR = %.6f ms, total = %.6f ms\n",
       csr_to_bsr_ms, bsr_to_chol_ms, chol_bsr_to_csr_ms, pattern_ms);

    T *rhs_perm = (T *)malloc(N * sizeof(T));
    T *x_perm = (T *)malloc(N * sizeof(T));
    permute_block_vec<T>(N, block_dim, perm, rhs, rhs_perm);
    std::fill(x_perm, x_perm + N, T(0));

    int *d_rowp = nullptr, *d_cols = nullptr;
    T *d_vals = nullptr, *d_vals_IC0 = nullptr, *d_rhs = nullptr, *d_y = nullptr, *d_x = nullptr;
    CHECK_CUDA(cudaMalloc((void **)&d_rowp, (N + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&d_cols, chol_csr_nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&d_vals, (size_t)chol_csr_nnz * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void **)&d_vals_IC0, (size_t)chol_csr_nnz * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void **)&d_rhs, N * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void **)&d_y, N * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void **)&d_x, N * sizeof(T)));

    CHECK_CUDA(cudaMemcpy(d_rowp, chol_csr_rowp, (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_cols, chol_csr_cols, chol_csr_nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vals, chol_csr_vals, (size_t)chol_csr_nnz * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_rhs, rhs_perm, N * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_y, 0, N * sizeof(T)));
    CHECK_CUDA(cudaMemset(d_x, 0, N * sizeof(T)));
    CHECK_CUDA(cudaMemcpy(d_vals_IC0, d_vals, (size_t)chol_csr_nnz * sizeof(T), cudaMemcpyDeviceToDevice));

    cusparseHandle_t handle = nullptr;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    cusparseMatDescr_t descrA = nullptr;
    csric02Info_t info_IC = nullptr;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
    CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
    CHECK_CUSPARSE(cusparseSetMatFillMode(descrA, CUSPARSE_FILL_MODE_LOWER));
    CHECK_CUSPARSE(cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_NON_UNIT));
    CHECK_CUSPARSE(cusparseCreateCsric02Info(&info_IC));

    int bufferSize_IC = 0;
    CHECK_CUSPARSE(cusparseDcsric02_bufferSize(handle, N, chol_csr_nnz, descrA,
                                               d_vals_IC0, d_rowp, d_cols,
                                               info_IC, &bufferSize_IC));
    void *buffer_IC = nullptr;
    CHECK_CUDA(cudaMalloc((void **)&buffer_IC, bufferSize_IC));

    CHECK_CUSPARSE(cusparseDcsric02_analysis(handle, N, chol_csr_nnz, descrA,
                                             d_vals_IC0, d_rowp, d_cols, info_IC,
                                             CUSPARSE_SOLVE_POLICY_NO_LEVEL, buffer_IC));
    checkZeroPivotIC0(handle, info_IC, "structural IC0");

    cusparseSpMatDescr_t matL = nullptr;
    cusparseDnVecDescr_t vecB = nullptr, vecY = nullptr, vecX = nullptr;
    cusparseSpSVDescr_t SpSV_L = nullptr, SpSV_LT = nullptr;

    CHECK_CUSPARSE(cusparseCreateCsr(&matL, N, N, chol_csr_nnz, d_rowp, d_cols, d_vals_IC0,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    cusparseFillMode_t fill_mode = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t diag_type = CUSPARSE_DIAG_TYPE_NON_UNIT;
    CHECK_CUSPARSE(cusparseSpMatSetAttribute(matL, CUSPARSE_SPMAT_FILL_MODE,
                                             &fill_mode, sizeof(fill_mode)));
    CHECK_CUSPARSE(cusparseSpMatSetAttribute(matL, CUSPARSE_SPMAT_DIAG_TYPE,
                                             &diag_type, sizeof(diag_type)));

    CHECK_CUSPARSE(cusparseCreateDnVec(&vecB, N, d_rhs, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, N, d_y, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, N, d_x, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseSpSV_createDescr(&SpSV_L));
    CHECK_CUSPARSE(cusparseSpSV_createDescr(&SpSV_LT));

    const double alpha = 1.0;
    size_t bufferSize_L = 0, bufferSize_LT = 0;
    CHECK_CUSPARSE(cusparseSpSV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, matL, vecB, vecY, CUDA_R_64F,
                                           CUSPARSE_SPSV_ALG_DEFAULT, SpSV_L,
                                           &bufferSize_L));
    CHECK_CUSPARSE(cusparseSpSV_bufferSize(handle, CUSPARSE_OPERATION_TRANSPOSE,
                                           &alpha, matL, vecY, vecX, CUDA_R_64F,
                                           CUSPARSE_SPSV_ALG_DEFAULT, SpSV_LT,
                                           &bufferSize_LT));
    void *buffer_L = nullptr, *buffer_LT = nullptr;
    CHECK_CUDA(cudaMalloc((void **)&buffer_L, bufferSize_L));
    CHECK_CUDA(cudaMalloc((void **)&buffer_LT, bufferSize_LT));

    CHECK_CUSPARSE(cusparseSpSV_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &alpha, matL, vecB, vecY, CUDA_R_64F,
                                         CUSPARSE_SPSV_ALG_DEFAULT, SpSV_L, buffer_L));
    CHECK_CUSPARSE(cusparseSpSV_analysis(handle, CUSPARSE_OPERATION_TRANSPOSE,
                                         &alpha, matL, vecY, vecX, CUDA_R_64F,
                                         CUSPARSE_SPSV_ALG_DEFAULT, SpSV_LT, buffer_LT));
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(d_vals_IC0, d_vals, (size_t)chol_csr_nnz * sizeof(T), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());
    auto tf0 = std::chrono::high_resolution_clock::now();
    CHECK_CUSPARSE(cusparseDcsric02(handle, N, chol_csr_nnz, descrA, d_vals_IC0,
                                    d_rowp, d_cols, info_IC,
                                    CUSPARSE_SOLVE_POLICY_NO_LEVEL, buffer_IC));
    CHECK_CUDA(cudaDeviceSynchronize());
    auto tf1 = std::chrono::high_resolution_clock::now();
    const double factor_ms = std::chrono::duration<double, std::milli>(tf1 - tf0).count();
    checkZeroPivotIC0(handle, info_IC, "numerical IC0");

    CHECK_CUSPARSE(cusparseSpSV_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      &alpha, matL, vecB, vecY, CUDA_R_64F,
                                      CUSPARSE_SPSV_ALG_DEFAULT, SpSV_L));
    CHECK_CUSPARSE(cusparseSpSV_solve(handle, CUSPARSE_OPERATION_TRANSPOSE,
                                      &alpha, matL, vecY, vecX, CUDA_R_64F,
                                      CUSPARSE_SPSV_ALG_DEFAULT, SpSV_LT));
    CHECK_CUDA(cudaDeviceSynchronize());

    double triL_total = 0.0, triLT_total = 0.0;
    for (int k = 0; k < nrepeat; k++) {
        CHECK_CUDA(cudaDeviceSynchronize());
        auto ts0 = std::chrono::high_resolution_clock::now();
        CHECK_CUSPARSE(cusparseSpSV_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          &alpha, matL, vecB, vecY, CUDA_R_64F,
                                          CUSPARSE_SPSV_ALG_DEFAULT, SpSV_L));
        CHECK_CUDA(cudaDeviceSynchronize());
        auto ts1 = std::chrono::high_resolution_clock::now();
        triL_total += std::chrono::duration<double, std::milli>(ts1 - ts0).count();

        CHECK_CUDA(cudaDeviceSynchronize());
        ts0 = std::chrono::high_resolution_clock::now();
        CHECK_CUSPARSE(cusparseSpSV_solve(handle, CUSPARSE_OPERATION_TRANSPOSE,
                                          &alpha, matL, vecY, vecX, CUDA_R_64F,
                                          CUSPARSE_SPSV_ALG_DEFAULT, SpSV_LT));
        CHECK_CUDA(cudaDeviceSynchronize());
        ts1 = std::chrono::high_resolution_clock::now();
        triLT_total += std::chrono::duration<double, std::milli>(ts1 - ts0).count();
    }

    CHECK_CUDA(cudaMemcpy(x_perm, d_x, N * sizeof(T), cudaMemcpyDeviceToHost));
    unpermute_block_vec<T>(N, block_dim, perm, x_perm, x);
    const double rel_res = computeResidualCSR(N, csr_rowp, csr_cols, csr_vals, x, rhs);

    // printf("\nTiming breakdown (host sync):\n");
    // printf("  host LU pattern build    : %.6f ms\n", pattern_ms);
    // printf("  IC numeric factor        : %.6f ms\n", factor_ms);
    // printf("  forward triangular (L)   : %.6f ms\n", triL_total / nrepeat);
    // printf("  backward triangular (LT) : %.6f ms\n", triLT_total / nrepeat);
    // printf("  total triangular solve   : %.6f ms\n", (triL_total + triLT_total) / nrepeat);
    // printf("  total solve (fact+tri)   : %.6f ms\n",
    //        factor_ms + (triL_total + triLT_total) / nrepeat);
    // printf("  repeats                  : %d\n", nrepeat);
    // printf("\nrelative residual = %.15e\n", rel_res);

    printf("\nTimings: pattern=%.3f ms, factor=%.3f ms, L=%.3f ms, LT=%.3f ms, tri=%.3f ms, repeats=%d\n",
       pattern_ms,
       factor_ms,
       triL_total / nrepeat,
       triLT_total / nrepeat,
       (triL_total + triLT_total) / nrepeat,
       nrepeat);

    printf("Total solve (fact+tri)=%.3f ms, rel_res=%.3e\n",
        factor_ms + (triL_total + triLT_total) / nrepeat,
        rel_res);

    CHECK_CUSPARSE(cusparseSpSV_destroyDescr(SpSV_LT));
    CHECK_CUSPARSE(cusparseSpSV_destroyDescr(SpSV_L));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecB));
    CHECK_CUSPARSE(cusparseDestroySpMat(matL));
    CHECK_CUSPARSE(cusparseDestroyCsric02Info(info_IC));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descrA));
    CHECK_CUSPARSE(cusparseDestroy(handle));

    CHECK_CUDA(cudaFree(buffer_LT));
    CHECK_CUDA(cudaFree(buffer_L));
    CHECK_CUDA(cudaFree(buffer_IC));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUDA(cudaFree(d_rhs));
    CHECK_CUDA(cudaFree(d_vals_IC0));
    CHECK_CUDA(cudaFree(d_vals));
    CHECK_CUDA(cudaFree(d_cols));
    CHECK_CUDA(cudaFree(d_rowp));

    free(x_perm);
    free(rhs_perm);
    delete[] chol_csr_vals;
    delete[] chol_csr_cols;
    delete[] chol_csr_rowp;
    delete[] chol_bsr_vals;
    delete[] chol_bsr_cols;
    delete[] chol_bsr_rowp;
    delete[] iperm;
    delete[] perm;
    delete[] bsr_vals;
    delete[] bsr_cols;
    delete[] bsr_rowp;
    free(x);
    free(rhs);
    free(csr_vals);
    free(csr_cols);
    free(csr_rowp);

    return 0;
}