// poisson_bsr_lu_pattern_ilu_direct.cu
// Standalone performance test:
//   CSR Laplace -> host BSR -> AMD full symbolic LU BSR pattern -> cuSPARSE bsrilu02
//   -> one direct-style solve using L/U triangular solves. Pattern utilities come from include/poisson.h.

#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
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

#define CHECK_CUBLAS(x) do {                                                   \
    cublasStatus_t s = (x);                                                    \
    if (s != CUBLAS_STATUS_SUCCESS) {                                          \
        printf("cuBLAS error %d at %s:%d\n", (int)s, __FILE__, __LINE__);     \
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
        double r = Ax - rhs[i];
        r2 += r * r;
        b2 += rhs[i] * rhs[i];
    }
    return std::sqrt(r2 / std::max(b2, 1e-300));
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

    // printf("nx = %d\n", nx);
    // printf("nnode = %d\n", nnode);
    // printf("scalar N = %d\n", N);
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

    int *bsr_rowp = nullptr, *bsr_cols = nullptr, bsr_nnzb = 0;
    T *bsr_vals = nullptr;
    CSRtoBSR<T>(block_dim, N, csr_rowp, csr_cols, csr_vals,
                &bsr_rowp, &bsr_cols, &bsr_vals, &bsr_nnzb);

    int mb = N / block_dim;
    // printf("Original BSR: mb=%d block_dim=%d nnzb=%d scalar_nnz=%d\n",
    //        mb, block_dim, bsr_nnzb, bsr_nnzb * block_dim2);

    auto tpat0 = std::chrono::high_resolution_clock::now();
    int *perm = nullptr, *iperm = nullptr;
    int *lu_rowp = nullptr, *lu_cols = nullptr, lu_nnzb = 0;
    T *lu_vals = nullptr;

    // Uses include/poisson.h AMD-based BSR -> full LU-pattern conversion.
    convertBSRtoLUpattern<T>(block_dim, N, bsr_nnzb, bsr_rowp, bsr_cols, bsr_vals,
                             &perm, &iperm, &lu_rowp, &lu_cols, &lu_vals, &lu_nnzb);

    // Optional conversion through include/poisson.h, mainly to sanity-check the expanded pattern.
    int *lu_csr_rowp = nullptr, *lu_csr_cols = nullptr, lu_csr_nnz = 0;
    T *lu_csr_vals = nullptr;
    BSRtoCSR<T>(block_dim, N, lu_nnzb, lu_rowp, lu_cols, lu_vals,
                &lu_csr_rowp, &lu_csr_cols, &lu_csr_vals, &lu_csr_nnz);

    auto tpat1 = std::chrono::high_resolution_clock::now();
    double pattern_ms = std::chrono::duration<double, std::milli>(tpat1 - tpat0).count();

    // printf("LU-pattern BSR: mb=%d block_dim=%d nnzb=%d scalar_nnz=%d fill_ratio=%.6f\n",
    //        mb, block_dim, lu_nnzb, lu_nnzb * block_dim2,
    //        (double)lu_nnzb / (double)bsr_nnzb);
    // printf("LU-pattern CSR: N=%d nnz=%d\n", N, lu_csr_nnz);
    printf("Host BSR->LU-pattern + BSR->CSR time: %.6f ms\n", pattern_ms);

    // The LU-pattern matrix is in AMD-permuted block ordering. Permute RHS into that
    // ordering for the solve; later unpermute x before residual checking.
    T *rhs_perm = (T *)malloc(N * sizeof(T));
    T *x_perm = (T *)malloc(N * sizeof(T));
    for (int pnode = 0; pnode < mb; pnode++) {
        const int onode = perm[pnode];
        for (int k = 0; k < block_dim; k++) {
            rhs_perm[block_dim * pnode + k] = rhs[block_dim * onode + k];
            x_perm[block_dim * pnode + k] = 0.0;
        }
    }

    int *d_rowp = nullptr, *d_cols = nullptr;
    T *d_vals = nullptr, *d_vals_ILU0 = nullptr, *d_rhs = nullptr, *d_x = nullptr, *d_temp = nullptr;
    CHECK_CUDA(cudaMalloc((void **)&d_rowp, (mb + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&d_cols, lu_nnzb * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&d_vals, (size_t)lu_nnzb * block_dim2 * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void **)&d_vals_ILU0, (size_t)lu_nnzb * block_dim2 * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void **)&d_rhs, N * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void **)&d_x, N * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void **)&d_temp, N * sizeof(T)));

    CHECK_CUDA(cudaMemcpy(d_rowp, lu_rowp, (mb + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_cols, lu_cols, lu_nnzb * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vals, lu_vals, (size_t)lu_nnzb * block_dim2 * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_rhs, rhs_perm, N * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_x, 0, N * sizeof(T)));
    CHECK_CUDA(cudaMemcpy(d_vals_ILU0, d_vals, (size_t)lu_nnzb * block_dim2 * sizeof(T), cudaMemcpyDeviceToDevice));

    cublasHandle_t cublasHandle = nullptr;
    cusparseHandle_t cusparseHandle = nullptr;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

    const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;
    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseOperation_t trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseSolvePolicy_t policy_M = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    const cusparseSolvePolicy_t policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;

    cusparseMatDescr_t descr_M = 0, descr_L = 0, descr_U = 0;
    bsrilu02Info_t info_M = 0;
    bsrsv2Info_t info_L = 0, info_U = 0;

    CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_M));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ZERO));
    CHECK_CUSPARSE(cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseCreateBsrilu02Info(&info_M));

    CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_L));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO));
    CHECK_CUSPARSE(cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER));
    CHECK_CUSPARSE(cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT));
    CHECK_CUSPARSE(cusparseCreateBsrsv2Info(&info_L));

    CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_U));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO));
    CHECK_CUSPARSE(cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER));
    CHECK_CUSPARSE(cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT));
    CHECK_CUSPARSE(cusparseCreateBsrsv2Info(&info_U));

    int pBufferSize_M = 0, pBufferSize_L = 0, pBufferSize_U = 0;
    CHECK_CUSPARSE(cusparseDbsrilu02_bufferSize(cusparseHandle, dir, mb, lu_nnzb, descr_M,
                                                d_vals_ILU0, d_rowp, d_cols, block_dim,
                                                info_M, &pBufferSize_M));
    CHECK_CUSPARSE(cusparseDbsrsv2_bufferSize(cusparseHandle, dir, trans_L, mb, lu_nnzb, descr_L,
                                              d_vals_ILU0, d_rowp, d_cols, block_dim,
                                              info_L, &pBufferSize_L));
    CHECK_CUSPARSE(cusparseDbsrsv2_bufferSize(cusparseHandle, dir, trans_U, mb, lu_nnzb, descr_U,
                                              d_vals_ILU0, d_rowp, d_cols, block_dim,
                                              info_U, &pBufferSize_U));
    int pBufferSize = std::max({pBufferSize_M, pBufferSize_L, pBufferSize_U});
    void *pBuffer = nullptr;
    CHECK_CUDA(cudaMalloc((void **)&pBuffer, pBufferSize));

    int structural_zero = -1, numerical_zero = -1;
    cusparseStatus_t status;

    CHECK_CUSPARSE(cusparseDbsrilu02_analysis(cusparseHandle, dir, mb, lu_nnzb, descr_M,
                                              d_vals_ILU0, d_rowp, d_cols, block_dim,
                                              info_M, policy_M, pBuffer));
    status = cusparseXbsrilu02_zeroPivot(cusparseHandle, info_M, &structural_zero);
    if (status == CUSPARSE_STATUS_ZERO_PIVOT) {
        printf("structural zero pivot: missing block A(%d,%d)\n", structural_zero, structural_zero);
    }

    CHECK_CUSPARSE(cusparseDbsrsv2_analysis(cusparseHandle, dir, trans_L, mb, lu_nnzb, descr_L,
                                            d_vals_ILU0, d_rowp, d_cols, block_dim,
                                            info_L, policy_L, pBuffer));
    CHECK_CUSPARSE(cusparseDbsrsv2_analysis(cusparseHandle, dir, trans_U, mb, lu_nnzb, descr_U,
                                            d_vals_ILU0, d_rowp, d_cols, block_dim,
                                            info_U, policy_U, pBuffer));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Numeric factorization.
    CHECK_CUDA(cudaMemcpy(d_vals_ILU0, d_vals, (size_t)lu_nnzb * block_dim2 * sizeof(T), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());
    auto tf0 = std::chrono::high_resolution_clock::now();
    CHECK_CUSPARSE(cusparseDbsrilu02(cusparseHandle, dir, mb, lu_nnzb, descr_M,
                                     d_vals_ILU0, d_rowp, d_cols, block_dim,
                                     info_M, policy_M, pBuffer));
    CHECK_CUDA(cudaDeviceSynchronize());
    auto tf1 = std::chrono::high_resolution_clock::now();
    double factor_ms = std::chrono::duration<double, std::milli>(tf1 - tf0).count();

    status = cusparseXbsrilu02_zeroPivot(cusparseHandle, info_M, &numerical_zero);
    if (status == CUSPARSE_STATUS_ZERO_PIVOT) {
        printf("numerical zero pivot: block U(%d,%d) is not invertible\n", numerical_zero, numerical_zero);
    }

    const double alpha = 1.0;

    // Warm-up direct solve path: L z = rhs, U x = z.
    CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_L, mb, lu_nnzb, &alpha,
                                         descr_L, d_vals_ILU0, d_rowp, d_cols, block_dim,
                                         info_L, d_rhs, d_temp, policy_L, pBuffer));
    CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, mb, lu_nnzb, &alpha,
                                         descr_U, d_vals_ILU0, d_rowp, d_cols, block_dim,
                                         info_U, d_temp, d_x, policy_U, pBuffer));
    CHECK_CUDA(cudaDeviceSynchronize());

    double triL_total = 0.0, triU_total = 0.0;
    for (int k = 0; k < nrepeat; k++) {
        CHECK_CUDA(cudaDeviceSynchronize());
        auto t0 = std::chrono::high_resolution_clock::now();
        CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_L, mb, lu_nnzb, &alpha,
                                             descr_L, d_vals_ILU0, d_rowp, d_cols, block_dim,
                                             info_L, d_rhs, d_temp, policy_L, pBuffer));
        CHECK_CUDA(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        triL_total += std::chrono::duration<double, std::milli>(t1 - t0).count();

        CHECK_CUDA(cudaDeviceSynchronize());
        t0 = std::chrono::high_resolution_clock::now();
        CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, mb, lu_nnzb, &alpha,
                                             descr_U, d_vals_ILU0, d_rowp, d_cols, block_dim,
                                             info_U, d_temp, d_x, policy_U, pBuffer));
        CHECK_CUDA(cudaDeviceSynchronize());
        t1 = std::chrono::high_resolution_clock::now();
        triU_total += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    CHECK_CUDA(cudaMemcpy(x_perm, d_x, N * sizeof(T), cudaMemcpyDeviceToHost));
    for (int pnode = 0; pnode < mb; pnode++) {
        const int onode = perm[pnode];
        for (int k = 0; k < block_dim; k++) {
            x[block_dim * onode + k] = x_perm[block_dim * pnode + k];
        }
    }
    double rel_res = computeResidualCSR(N, csr_rowp, csr_cols, csr_vals, x, rhs);

    // printf("\nTiming breakdown (host sync):\n");
    // printf("  host LU pattern build    : %.6f ms\n", pattern_ms);
    // printf("  ILU numeric factor       : %.6f ms\n", factor_ms);
    // printf("  forward triangular (L)   : %.6f ms\n", triL_total / nrepeat);
    // printf("  backward triangular (U)  : %.6f ms\n", triU_total / nrepeat);
    // printf("  total triangular solve   : %.6f ms\n", (triL_total + triU_total) / nrepeat);
    // printf("  total solve (fact+tri)   : %.6f ms\n",
    //     factor_ms + (triL_total + triU_total) / nrepeat);
    // printf("  repeats                  : %d\n", nrepeat);
    // printf("\nrelative residual = %.15e\n", rel_res);

    printf("Timings: pattern=%.3f ms, factor=%.3f ms, L=%.3f ms, U=%.3f ms, tri=%.3f ms, repeats=%d\n",
        pattern_ms, factor_ms,
        triL_total / nrepeat,
        triU_total / nrepeat,
        (triL_total + triU_total) / nrepeat,
        nrepeat);

    // printf("Total solve (fact+tri)=%.3f ms, rel_res=%.3e\n",
    //     factor_ms + (triL_total + triU_total) / nrepeat,
    //     rel_res);
    printf("Total solve (fact+tri)=%.3f ms\n",
    factor_ms + (triL_total + triU_total) / nrepeat);
    printf("relative residual = %.9e\n",
    rel_res);

    // printf("x[0]   = %.15e\n", x[0]);
    // printf("x[N/2] = %.15e\n", x[N / 2]);
    // printf("x[N-1] = %.15e\n", x[N - 1]);

    CHECK_CUDA(cudaFree(pBuffer));
    CHECK_CUDA(cudaFree(d_temp));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_rhs));
    CHECK_CUDA(cudaFree(d_vals_ILU0));
    CHECK_CUDA(cudaFree(d_vals));
    CHECK_CUDA(cudaFree(d_cols));
    CHECK_CUDA(cudaFree(d_rowp));

    CHECK_CUSPARSE(cusparseDestroyBsrsv2Info(info_U));
    CHECK_CUSPARSE(cusparseDestroyBsrsv2Info(info_L));
    CHECK_CUSPARSE(cusparseDestroyBsrilu02Info(info_M));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_U));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_L));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_M));
    CHECK_CUSPARSE(cusparseDestroy(cusparseHandle));
    CHECK_CUBLAS(cublasDestroy(cublasHandle));

    delete[] lu_csr_vals;
    delete[] lu_csr_cols;
    delete[] lu_csr_rowp;
    delete[] lu_vals;
    delete[] lu_cols;
    delete[] lu_rowp;
    delete[] iperm;
    delete[] perm;
    free(x_perm);
    free(rhs_perm);
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