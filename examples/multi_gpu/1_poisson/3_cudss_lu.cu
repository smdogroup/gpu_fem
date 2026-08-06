// poisson_cudss_mg.cu

#include "include/poisson.h"

#include <cuda_runtime.h>
#include <cudss.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK_CUDA(x) do {                                      \
    cudaError_t e = (x);                                        \
    if (e != cudaSuccess) {                                     \
        printf("CUDA error %s at %s:%d\n",                     \
               cudaGetErrorString(e), __FILE__, __LINE__);      \
        exit(1);                                                \
    }                                                          \
} while (0)

#define CHECK_CUDSS(x) do {                                     \
    cudssStatus_t s = (x);                                      \
    if (s != CUDSS_STATUS_SUCCESS) {                            \
        printf("cuDSS error %d at %s:%d\n",                    \
               (int)s, __FILE__, __LINE__);                     \
        exit(1);                                                \
    }                                                          \
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

    int ngpu = std::min(requested_gpus, device_count);
    if (ngpu <= 0) {
        printf("ERROR: no CUDA GPUs found\n");
        exit(1);
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

    CHECK_CUDA(cudaMemcpy(d_rowp, h_rowp,
                          (N + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_cols, h_cols,
                          nnz * sizeof(int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vals, h_vals,
                          nnz * sizeof(double),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b,
                          N * sizeof(double),
                          cudaMemcpyHostToDevice));
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
        NULL,              // rowEnd must be NULL for standard CSR
        d_cols,
        d_vals,
        CUDA_R_32I,
        CUDA_R_64F,
        CUDSS_MTYPE_SPD,
        CUDSS_MVIEW_FULL,
        CUDSS_BASE_ZERO
    ));

    int nrhs = 1;

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

    CHECK_CUDSS(cudssExecute(
        handle,
        CUDSS_PHASE_ANALYSIS,
        config,
        data,
        A,
        X,
        B
    ));

    CHECK_CUDSS(cudssExecute(
        handle,
        CUDSS_PHASE_FACTORIZATION,
        config,
        data,
        A,
        X,
        B
    ));

    // CHECK_CUDSS(cudssExecute(
    //     handle,
    //     CUDSS_PHASE_SOLVE,
    //     config,
    //     data,
    //     A,
    //     X,
    //     B
    // ));

        cudaEvent_t solve_start, solve_stop;
    CHECK_CUDA(cudaEventCreate(&solve_start));
    CHECK_CUDA(cudaEventCreate(&solve_stop));

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(solve_start, 0));

    CHECK_CUDSS(cudssExecute(
        handle,
        CUDSS_PHASE_SOLVE,
        config,
        data,
        A,
        X,
        B
    ));

    CHECK_CUDA(cudaEventRecord(solve_stop, 0));
    CHECK_CUDA(cudaEventSynchronize(solve_stop));

    float solve_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&solve_ms, solve_start, solve_stop));

    // printf("cuDSS MG solve phase runtime = %.6f ms\n", solve_ms);
    printf("cuDSS MG solve phase runtime = %.9f s\n", 1.0e-3 * solve_ms);

    CHECK_CUDA(cudaEventDestroy(solve_start));
    CHECK_CUDA(cudaEventDestroy(solve_stop));

    CHECK_CUDA(cudaMemcpy(h_x, d_x,
                          N * sizeof(double),
                          cudaMemcpyDeviceToHost));

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

        double r = Ax - b[i];
        r2 += r * r;
        b2 += b[i] * b[i];
    }

    return std::sqrt(r2 / std::max(b2, 1e-300));
}

int main() {
    int N = 16384;
    int n = (int)std::sqrt((double)N);

    if (n * n != N) {
        printf("ERROR: N must be square\n");
        return 1;
    }

    int nnz = 5 * N - 4 * n;

    int *rowp = (int *)malloc((N + 1) * sizeof(int));
    int *cols = (int *)malloc(nnz * sizeof(int));
    double *vals = (double *)malloc(nnz * sizeof(double));
    double *rhs = (double *)malloc(N * sizeof(double));
    double *x = (double *)malloc(N * sizeof(double));

    for (int i = 0; i < N; i++) {
        rhs[i] = 0.0;
        x[i] = 0.0;
    }

    genLaplaceCSR<double>(rowp, cols, vals, N, nnz, rhs);

    // genLaplaceCSR gives negative definite Laplacian:
    // diag = -4, offdiag = +1.
    // Flip sign so matrix is SPD.
    for (int k = 0; k < nnz; k++) {
        vals[k] *= -1.0;
    }
    for (int i = 0; i < N; i++) {
        rhs[i] *= -1.0;
    }

    // If debugging, first try 1, then 2, then 4.
    int requested_gpus = 4;

    solve_cudss_mg_csr(
        N,
        nnz,
        rowp,
        cols,
        vals,
        rhs,
        x,
        requested_gpus
    );

    double rel_res = compute_residual_norm(N, rowp, cols, vals, x, rhs);

    printf("cuDSS MG solve completed.\n");
    printf("relative residual = %.15e\n", rel_res);
    printf("x[0]   = %.15e\n", x[0]);
    printf("x[N/2] = %.15e\n", x[N / 2]);
    printf("x[N-1] = %.15e\n", x[N - 1]);

    free(rowp);
    free(cols);
    free(vals);
    free(rhs);
    free(x);

    return 0;
}