// poisson_cudss_mg_lu_pattern.cu
//
// cuDSS test on an explicitly expanded host LU sparsity pattern:
//   CSR -> BSR -> AMD/full-LU-pattern BSR -> CSR -> cuDSS
//
// This is intended for performance comparison against the cuSPARSE BSR ILU/direct-LU-pattern path.

#include <cuda_runtime.h>
#include <cudss.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// #include "linalg/bsr_data.h"
#include "include/poisson.h"

#define CHECK_CUDA(x) do {                                      \
    cudaError_t e = (x);                                        \
    if (e != cudaSuccess) {                                     \
        printf("CUDA error %s at %s:%d\n",                     \
               cudaGetErrorString(e), __FILE__, __LINE__);      \
        exit(1);                                                \
    }                                                           \
} while (0)

#define CHECK_CUDSS(x) do {                                     \
    cudssStatus_t s = (x);                                      \
    if (s != CUDSS_STATUS_SUCCESS) {                            \
        printf("cuDSS error %d at %s:%d\n",                    \
               (int)s, __FILE__, __LINE__);                     \
        exit(1);                                                \
    }                                                           \
} while (0)

void solve_cudss_mg_csr(
    int N,
    int nnz,
    const int *h_rowp,
    const int *h_cols,
    const double *h_vals,
    const double *h_b,
    double *h_x,
    int requested_gpus
) {
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));

    if (device_count <= 0) {
        printf("ERROR: no CUDA GPUs found\n");
        exit(1);
    }

    int ngpu = std::min(requested_gpus, device_count);
    if (ngpu <= 0) {
        ngpu = device_count;
    }

    std::vector<int> devices(ngpu);
    for (int g = 0; g < ngpu; g++) {
        devices[g] = g;
    }

    printf("Using cuDSS MG with %d GPU(s)\n", ngpu);

    CHECK_CUDA(cudaSetDevice(devices[0]));

    int *d_rowp = nullptr;
    int *d_cols = nullptr;
    double *d_vals = nullptr;
    double *d_b = nullptr;
    double *d_x = nullptr;

    CHECK_CUDA(cudaMalloc((void **)&d_rowp, (N + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&d_cols, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&d_vals, nnz * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void **)&d_b, N * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void **)&d_x, N * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_rowp, h_rowp, (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_cols, h_cols, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vals, h_vals, nnz * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_x, 0, N * sizeof(double)));

    cudssHandle_t handle = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t data = nullptr;
    cudssMatrix_t A = nullptr;
    cudssMatrix_t B = nullptr;
    cudssMatrix_t X = nullptr;

    CHECK_CUDSS(cudssCreateMg(&handle, ngpu, devices.data()));
    CHECK_CUDSS(cudssConfigCreate(&config));
    CHECK_CUDSS(cudssDataCreate(handle, &data));

    CHECK_CUDSS(cudssMatrixCreateCsr(
        &A,
        N,
        N,
        nnz,
        d_rowp,
        NULL,
        d_cols,
        d_vals,
        CUDA_R_32I,
        CUDA_R_64F,
        CUDSS_MTYPE_SPD,
        CUDSS_MVIEW_FULL,
        CUDSS_BASE_ZERO
    ));

    const int nrhs = 1;

    CHECK_CUDSS(cudssMatrixCreateDn(
        &B,
        N,
        nrhs,
        N,
        d_b,
        CUDA_R_64F,
        CUDSS_LAYOUT_COL_MAJOR
    ));

    CHECK_CUDSS(cudssMatrixCreateDn(
        &X,
        N,
        nrhs,
        N,
        d_x,
        CUDA_R_64F,
        CUDSS_LAYOUT_COL_MAJOR
    ));

    cudaEvent_t ev_start, ev_stop;
    CHECK_CUDA(cudaEventCreate(&ev_start));
    CHECK_CUDA(cudaEventCreate(&ev_stop));

    float analysis_ms = 0.0f;
    float factor_ms = 0.0f;
    float solve_ms = 0.0f;

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(ev_start, 0));
    CHECK_CUDSS(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, config, data, A, X, B));
    CHECK_CUDA(cudaEventRecord(ev_stop, 0));
    CHECK_CUDA(cudaEventSynchronize(ev_stop));
    CHECK_CUDA(cudaEventElapsedTime(&analysis_ms, ev_start, ev_stop));

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(ev_start, 0));
    CHECK_CUDSS(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, config, data, A, X, B));
    CHECK_CUDA(cudaEventRecord(ev_stop, 0));
    CHECK_CUDA(cudaEventSynchronize(ev_stop));
    CHECK_CUDA(cudaEventElapsedTime(&factor_ms, ev_start, ev_stop));

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(ev_start, 0));
    CHECK_CUDSS(cudssExecute(handle, CUDSS_PHASE_SOLVE, config, data, A, X, B));
    CHECK_CUDA(cudaEventRecord(ev_stop, 0));
    CHECK_CUDA(cudaEventSynchronize(ev_stop));
    CHECK_CUDA(cudaEventElapsedTime(&solve_ms, ev_start, ev_stop));

    const float fact_solve_ms = factor_ms + solve_ms;
    const float total_ms = analysis_ms + factor_ms + solve_ms;

    // printf("\ncuDSS MG timing breakdown:\n");
    // printf("  analysis phase          : %.6f ms\n", analysis_ms);
    // printf("  factorization phase     : %.6f ms\n", factor_ms);
    // printf("  solve phase             : %.6f ms\n", solve_ms);
    // printf("  factor + solve          : %.6f ms\n", fact_solve_ms);
    // printf("  analysis + fact + solve : %.6f ms\n", total_ms);

    printf("\ncuDSS MG timing breakdown:\n");
    printf("  analysis phase          : %.6f ms  [setup only; reuse if pattern fixed]\n",
           analysis_ms);
    printf("  factorization phase     : %.6f ms\n", factor_ms);
    printf("  solve phase             : %.6f ms\n", solve_ms);
    printf("  factor + solve          : %.6f ms  [repeated solve metric]\n",
           fact_solve_ms);

    CHECK_CUDA(cudaEventDestroy(ev_start));
    CHECK_CUDA(cudaEventDestroy(ev_stop));

    CHECK_CUDA(cudaMemcpy(h_x, d_x, N * sizeof(double), cudaMemcpyDeviceToHost));

    CHECK_CUDSS(cudssMatrixDestroy(A));
    CHECK_CUDSS(cudssMatrixDestroy(B));
    CHECK_CUDSS(cudssMatrixDestroy(X));
    CHECK_CUDSS(cudssDataDestroy(handle, data));
    CHECK_CUDSS(cudssConfigDestroy(config));
    CHECK_CUDSS(cudssDestroy(handle));

    cudaFree(d_rowp);
    cudaFree(d_cols);
    cudaFree(d_vals);
    cudaFree(d_b);
    cudaFree(d_x);
}

double compute_residual_norm(
    int N,
    const int *rowp,
    const int *cols,
    const double *vals,
    const double *x,
    const double *b
) {
    double r2 = 0.0;
    double b2 = 0.0;

    for (int i = 0; i < N; i++) {
        double Ax = 0.0;
        for (int jp = rowp[i]; jp < rowp[i + 1]; jp++) {
            Ax += vals[jp] * x[cols[jp]];
        }

        const double r = Ax - b[i];
        r2 += r * r;
        b2 += b[i] * b[i];
    }

    return std::sqrt(r2 / std::max(b2, 1e-300));
}

int main(int argc, char **argv) {
    int N = 16384;
    if (argc > 1) {
        N = std::atoi(argv[1]);
    }

    const int block_dim = 2;
    const int block_dim2 = block_dim * block_dim;
    const int requested_gpus = 1;  // set <=0 to use all visible GPUs

    const int n = (int)std::sqrt((double)N);
    if (n * n != N) {
        printf("ERROR: N must be square\n");
        return 1;
    }
    if (N % block_dim != 0) {
        printf("ERROR: N must be divisible by block_dim\n");
        return 1;
    }

    const int nnz = 5 * N - 4 * n;

    int *csr_rowp = (int *)malloc((N + 1) * sizeof(int));
    int *csr_cols = (int *)malloc(nnz * sizeof(int));
    double *csr_vals = (double *)malloc(nnz * sizeof(double));
    double *rhs = (double *)malloc(N * sizeof(double));
    double *x = (double *)malloc(N * sizeof(double));

    for (int i = 0; i < N; i++) {
        rhs[i] = 0.0;
        x[i] = 0.0;
    }

    genLaplaceCSR<double>(csr_rowp, csr_cols, csr_vals, N, nnz, rhs);

    // genLaplaceCSR gives negative definite Laplacian:
    // diag = -4, offdiag = +1. Flip sign so matrix is SPD.
    for (int k = 0; k < nnz; k++) {
        csr_vals[k] *= -1.0;
    }
    for (int i = 0; i < N; i++) {
        rhs[i] *= -1.0;
    }

    int *bsr_rowp = nullptr;
    int *bsr_cols = nullptr;
    double *bsr_vals = nullptr;
    int bsr_nnzb = 0;

    int *perm = nullptr;
    int *iperm = nullptr;
    int *lu_bsr_rowp = nullptr;
    int *lu_bsr_cols = nullptr;
    double *lu_bsr_vals = nullptr;
    int lu_bsr_nnzb = 0;

    int *lu_csr_rowp = nullptr;
    int *lu_csr_cols = nullptr;
    double *lu_csr_vals = nullptr;
    int lu_csr_nnz = 0;

    double *rhs_perm = (double *)malloc(N * sizeof(double));
    double *x_perm = (double *)malloc(N * sizeof(double));

    auto t0 = std::chrono::high_resolution_clock::now();

    CSRtoBSR<double>(
        block_dim,
        N,
        csr_rowp,
        csr_cols,
        csr_vals,
        &bsr_rowp,
        &bsr_cols,
        &bsr_vals,
        &bsr_nnzb
    );

    auto t1 = std::chrono::high_resolution_clock::now();

    convertBSRtoLUpattern<double>(
        block_dim,
        N,
        bsr_nnzb,
        bsr_rowp,
        bsr_cols,
        bsr_vals,
        &perm,
        &iperm,
        &lu_bsr_rowp,
        &lu_bsr_cols,
        &lu_bsr_vals,
        &lu_bsr_nnzb
    );

    auto t2 = std::chrono::high_resolution_clock::now();

    BSRtoCSR<double>(
        block_dim,
        N,
        lu_bsr_nnzb,
        lu_bsr_rowp,
        lu_bsr_cols,
        lu_bsr_vals,
        &lu_csr_rowp,
        &lu_csr_cols,
        &lu_csr_vals,
        &lu_csr_nnz
    );

    auto t3 = std::chrono::high_resolution_clock::now();

    const double csr_to_bsr_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double bsr_to_lu_ms =
        std::chrono::duration<double, std::milli>(t2 - t1).count();
    const double lu_bsr_to_csr_ms =
        std::chrono::duration<double, std::milli>(t3 - t2).count();
    const double host_pattern_total_ms =
        std::chrono::duration<double, std::milli>(t3 - t0).count();

    const int mb = N / block_dim;

    printf("Original CSR: N=%d nnz=%d\n", N, nnz);
    printf("Original BSR: mb=%d block_dim=%d nnzb=%d scalar_nnz=%d\n",
           mb, block_dim, bsr_nnzb, bsr_nnzb * block_dim2);
    printf("LU-pattern BSR: mb=%d block_dim=%d nnzb=%d scalar_nnz=%d fill_ratio=%.6f\n",
           mb, block_dim, lu_bsr_nnzb, lu_bsr_nnzb * block_dim2,
           (double)lu_bsr_nnzb / std::max(1, bsr_nnzb));
    printf("LU-pattern CSR: N=%d nnz=%d\n", N, lu_csr_nnz);

    printf("\nHost conversion timing:\n");
    printf("  CSR -> BSR              : %.6f ms\n", csr_to_bsr_ms);
    printf("  BSR -> LU pattern BSR   : %.6f ms\n", bsr_to_lu_ms);
    printf("  LU pattern BSR -> CSR   : %.6f ms\n", lu_bsr_to_csr_ms);
    printf("  host pattern total      : %.6f ms\n", host_pattern_total_ms);

    // The LU-pattern matrix is AMD-permuted by block. Permute RHS, solve permuted system,
    // then unpermute x before checking the residual in the original CSR ordering.
    permute_block_vec<double>(N, block_dim, perm, rhs, rhs_perm);
    std::fill(x_perm, x_perm + N, 0.0);

    solve_cudss_mg_csr(
        N,
        lu_csr_nnz,
        lu_csr_rowp,
        lu_csr_cols,
        lu_csr_vals,
        rhs_perm,
        x_perm,
        requested_gpus
    );

    unpermute_block_vec<double>(N, block_dim, perm, x_perm, x);

    const double rel_res = compute_residual_norm(N, csr_rowp, csr_cols, csr_vals, x, rhs);

    printf("cuDSS MG LU-pattern solve completed.\n");
    printf("relative residual = %.15e\n", rel_res);

    free(csr_rowp);
    free(csr_cols);
    free(csr_vals);
    free(rhs);
    free(x);
    free(rhs_perm);
    free(x_perm);

    delete[] bsr_rowp;
    delete[] bsr_cols;
    delete[] bsr_vals;

    delete[] perm;
    delete[] iperm;
    delete[] lu_bsr_rowp;
    delete[] lu_bsr_cols;
    delete[] lu_bsr_vals;

    delete[] lu_csr_rowp;
    delete[] lu_csr_cols;
    delete[] lu_csr_vals;

    return 0;
}