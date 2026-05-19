// poisson_cudss_mg_native_csr.cu
//
// Native cuDSS test on the original CSR matrix only:
//   CSR Laplace -> cuDSS analysis/factorization/solve
//
// No host BSR conversion, no host AMD reorder, no explicit host LU-pattern expansion.
// cuDSS does its own ordering/symbolic analysis during CUDSS_PHASE_ANALYSIS.

#include <cuda_runtime.h>
#include <cudss.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

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

    // cuDSS does ordering + symbolic factorization internally here.
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

    // printf("\ncuDSS MG timing breakdown, native CSR input:\n");
    // printf("  analysis phase          : %.6f ms  [cuDSS ordering + symbolic setup]\n", analysis_ms);
    // printf("  factorization phase     : %.6f ms\n", factor_ms);
    // printf("  solve phase             : %.6f ms\n", solve_ms);
    // printf("  factor + solve          : %.6f ms  [repeated solve metric if pattern fixed]\n", fact_solve_ms);
    printf("cuDSS timings: analysis=%.3f ms, factor=%.3f ms, solve=%.3f ms\n",
        analysis_ms, factor_ms, solve_ms);
    printf("Total solve (fact+solve)=%.3f ms\n", fact_solve_ms);

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
    using T = double;

    int nx = 128;
    int nrepeat = 1;

    if (argc > 1) nx = std::atoi(argv[1]);
    if (argc > 2) nrepeat = std::atoi(argv[2]);

    const int block_dim = 6;
    const int nnode = nx * nx;
    const int N = block_dim * nnode;

    const int nz_node = 5 * nnode - 4 * nx;
    const int nz = block_dim * nz_node;

    // printf("nx = %d\n", nx);
    // printf("scalar N = %d\n", N);

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

    const int requested_gpus = -1; // so uses all avail GPUs

    for (int k = 0; k < nz; k++) {
        csr_vals[k] *= -1.0;
    }
    for (int i = 0; i < N; i++) {
        rhs[i] *= -1.0;
    }

    // printf("Original CSR: N=%d nnz=%d\n", N, nz);

    solve_cudss_mg_csr(
        N,
        nz,
        csr_rowp,
        csr_cols,
        csr_vals,
        rhs,
        x,
        requested_gpus
    );

    const double rel_res = compute_residual_norm(N, csr_rowp, csr_cols, csr_vals, x, rhs);

    // printf("cuDSS MG native-CSR solve completed.\n");
    printf("relative residual = %.15e\n", rel_res);

    free(csr_rowp);
    free(csr_cols);
    free(csr_vals);
    free(rhs);
    free(x);

    return 0;
}