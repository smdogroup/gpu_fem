// poisson_cudss_blockdiag_gpu_local_matrices_threaded.cu

#include <cuda_runtime.h>
#include <cudss.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#include "include/poisson.h"

#define CHECK_CUDA(x) do {                                      \
    cudaError_t e = (x);                                        \
    if (e != cudaSuccess) {                                     \
        printf("CUDA error %s at %s:%d\n",                     \
               cudaGetErrorString(e), __FILE__, __LINE__);      \
        std::exit(1);                                           \
    }                                                           \
} while (0)

#define CHECK_CUDSS(x) do {                                     \
    cudssStatus_t s = (x);                                      \
    if (s != CUDSS_STATUS_SUCCESS) {                            \
        printf("cuDSS error %d at %s:%d\n",                    \
               (int)s, __FILE__, __LINE__);                     \
        std::exit(1);                                           \
    }                                                           \
} while (0)

struct GpuLocalSolve {
    int device = -1;

    int sub_start = 0;
    int sub_end = 0;
    int nsubs_local = 0;

    int nx = 0;
    int block_dim = 0;
    int nnode_sub = 0;
    int Nsub = 0;

    int N = 0;
    int nnz = 0;

    int *d_rowp = nullptr;
    int *d_cols = nullptr;
    double *d_vals = nullptr;
    double *d_rhs = nullptr;
    double *d_x = nullptr;

    cudaStream_t stream = nullptr;

    cudssHandle_t handle = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t data = nullptr;
    cudssMatrix_t A = nullptr;
    cudssMatrix_t B = nullptr;
    cudssMatrix_t X = nullptr;

    float analysis_ms = 0.0f;
    float factor_ms = 0.0f;
    float solve_ms = 0.0f;
};

__device__ __forceinline__ int poisson_row_nnz_local(
    int row_in_sub,
    int nx,
    int block_dim
) {
    const int inode = row_in_sub / block_dim;
    const int ix = inode % nx;
    const int iy = inode / nx;

    int ct = 1;
    if (iy > 0) ct++;
    if (ix > 0) ct++;
    if (ix < nx - 1) ct++;
    if (iy < nx - 1) ct++;
    return ct;
}

__global__ void count_gpu_blockdiag_nnz_kernel(
    int nx,
    int block_dim,
    int Nsub,
    int Nlocal,
    int *row_counts
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= Nlocal) return;

    const int row_in_sub = row % Nsub;
    row_counts[row] = poisson_row_nnz_local(row_in_sub, nx, block_dim);
}

__global__ void assemble_gpu_blockdiag_csr_kernel(
    int nx,
    int block_dim,
    int Nsub,
    int Nlocal,
    const int *rowp,
    int *cols,
    double *vals,
    double *rhs
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= Nlocal) return;

    const int local_sub = row / Nsub;
    const int row_in_sub = row - local_sub * Nsub;

    const int inode = row_in_sub / block_dim;
    const int d = row_in_sub % block_dim;
    const int ix = inode % nx;
    const int iy = inode / nx;

    const int col_base = local_sub * Nsub;

    int p = rowp[row];
    double b = 0.0;

    if (iy > 0) {
        cols[p] = col_base + block_dim * (inode - nx) + d;
        vals[p] = -1.0;
        p++;
    } else {
        b += 1.0;
    }

    if (ix > 0) {
        cols[p] = col_base + block_dim * (inode - 1) + d;
        vals[p] = -1.0;
        p++;
    }

    cols[p] = row;
    vals[p] = 4.0;
    p++;

    if (ix < nx - 1) {
        cols[p] = col_base + block_dim * (inode + 1) + d;
        vals[p] = -1.0;
        p++;
    }

    if (iy < nx - 1) {
        cols[p] = col_base + block_dim * (inode + nx) + d;
        vals[p] = -1.0;
        p++;
    }

    rhs[row] = b;
}

static void build_rowp_on_host(
    int device,
    int N,
    int *d_counts,
    int **d_rowp_out,
    int *nnz_out
) {
    CHECK_CUDA(cudaSetDevice(device));

    std::vector<int> h_counts(N);
    CHECK_CUDA(cudaMemcpy(h_counts.data(), d_counts,
                          N * sizeof(int),
                          cudaMemcpyDeviceToHost));

    std::vector<int> h_rowp(N + 1);
    h_rowp[0] = 0;

    for (int i = 0; i < N; i++) {
        h_rowp[i + 1] = h_rowp[i] + h_counts[i];
    }

    *nnz_out = h_rowp[N];

    CHECK_CUDA(cudaMalloc((void **)d_rowp_out, (N + 1) * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(*d_rowp_out, h_rowp.data(),
                          (N + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
}

static void assemble_gpu_local_matrix(GpuLocalSolve &g) {
    CHECK_CUDA(cudaSetDevice(g.device));
    CHECK_CUDA(cudaStreamCreate(&g.stream));

    g.nsubs_local = g.sub_end - g.sub_start;
    g.N = g.nsubs_local * g.Nsub;

    int *d_counts = nullptr;
    CHECK_CUDA(cudaMalloc((void **)&d_counts, g.N * sizeof(int)));

    const int block = 256;
    const int grid = (g.N + block - 1) / block;

    count_gpu_blockdiag_nnz_kernel<<<grid, block, 0, g.stream>>>(
        g.nx, g.block_dim, g.Nsub, g.N, d_counts
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(g.stream));

    build_rowp_on_host(g.device, g.N, d_counts, &g.d_rowp, &g.nnz);
    CHECK_CUDA(cudaFree(d_counts));

    CHECK_CUDA(cudaMalloc((void **)&g.d_cols, g.nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&g.d_vals, g.nnz * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void **)&g.d_rhs, g.N * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void **)&g.d_x, g.N * sizeof(double)));

    CHECK_CUDA(cudaMemsetAsync(g.d_x, 0, g.N * sizeof(double), g.stream));

    assemble_gpu_blockdiag_csr_kernel<<<grid, block, 0, g.stream>>>(
        g.nx, g.block_dim, g.Nsub, g.N,
        g.d_rowp, g.d_cols, g.d_vals, g.d_rhs
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(g.stream));

    printf("GPU %d local matrix: subdomains [%d, %d), nsubs=%d, N=%d, nnz=%d\n",
           g.device, g.sub_start, g.sub_end, g.nsubs_local, g.N, g.nnz);
}

static void init_gpu_local_cudss(GpuLocalSolve &g) {
    CHECK_CUDA(cudaSetDevice(g.device));

    CHECK_CUDSS(cudssCreate(&g.handle));
    CHECK_CUDSS(cudssConfigCreate(&g.config));
    CHECK_CUDSS(cudssDataCreate(g.handle, &g.data));
    CHECK_CUDSS(cudssSetStream(g.handle, g.stream));

    CHECK_CUDSS(cudssMatrixCreateCsr(
        &g.A,
        g.N,
        g.N,
        g.nnz,
        g.d_rowp,
        NULL,
        g.d_cols,
        g.d_vals,
        CUDA_R_32I,
        CUDA_R_64F,
        CUDSS_MTYPE_SPD,
        CUDSS_MVIEW_FULL,
        CUDSS_BASE_ZERO
    ));

    const int nrhs = 1;

    CHECK_CUDSS(cudssMatrixCreateDn(
        &g.B, g.N, nrhs, g.N, g.d_rhs,
        CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR
    ));

    CHECK_CUDSS(cudssMatrixCreateDn(
        &g.X, g.N, nrhs, g.N, g.d_x,
        CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR
    ));
}

static void execute_cudss_phase(
    GpuLocalSolve &g,
    cudssPhase_t phase,
    float &phase_ms
) {
    CHECK_CUDA(cudaSetDevice(g.device));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, g.stream));

    CHECK_CUDSS(cudssExecute(
        g.handle,
        phase,
        g.config,
        g.data,
        g.A,
        g.X,
        g.B
    ));

    CHECK_CUDA(cudaEventRecord(stop, g.stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&phase_ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

static double run_parallel_phase(
    std::vector<GpuLocalSolve> &gpu_solves,
    cudssPhase_t phase,
    const char *name
) {
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;
    threads.reserve(gpu_solves.size());

    for (int ig = 0; ig < (int)gpu_solves.size(); ig++) {
        threads.emplace_back([&, ig]() {
            GpuLocalSolve &g = gpu_solves[ig];

            if (phase == CUDSS_PHASE_ANALYSIS) {
                execute_cudss_phase(g, phase, g.analysis_ms);
            } else if (phase == CUDSS_PHASE_FACTORIZATION) {
                execute_cudss_phase(g, phase, g.factor_ms);
            } else if (phase == CUDSS_PHASE_SOLVE) {
                execute_cudss_phase(g, phase, g.solve_ms);
            }
        });
    }

    for (auto &t : threads) {
        t.join();
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    const double wall_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    double max_gpu_ms = 0.0;
    for (const auto &g : gpu_solves) {
        double gpu_ms = 0.0;
        if (phase == CUDSS_PHASE_ANALYSIS) {
            gpu_ms = g.analysis_ms;
        } else if (phase == CUDSS_PHASE_FACTORIZATION) {
            gpu_ms = g.factor_ms;
        } else if (phase == CUDSS_PHASE_SOLVE) {
            gpu_ms = g.solve_ms;
        }
        max_gpu_ms = std::max(max_gpu_ms, gpu_ms);
    }

    printf("%s wall time = %.3f ms, max per-GPU event time = %.3f ms\n",
           name, wall_ms, max_gpu_ms);

    return wall_ms;
}

static void destroy_gpu_local_cudss(GpuLocalSolve &g) {
    CHECK_CUDA(cudaSetDevice(g.device));

    if (g.A) CHECK_CUDSS(cudssMatrixDestroy(g.A));
    if (g.B) CHECK_CUDSS(cudssMatrixDestroy(g.B));
    if (g.X) CHECK_CUDSS(cudssMatrixDestroy(g.X));
    if (g.data) CHECK_CUDSS(cudssDataDestroy(g.handle, g.data));
    if (g.config) CHECK_CUDSS(cudssConfigDestroy(g.config));
    if (g.handle) CHECK_CUDSS(cudssDestroy(g.handle));

    g.A = nullptr;
    g.B = nullptr;
    g.X = nullptr;
    g.data = nullptr;
    g.config = nullptr;
    g.handle = nullptr;
}

static double compute_residual_norm(
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

static void free_gpu_local(GpuLocalSolve &g) {
    CHECK_CUDA(cudaSetDevice(g.device));

    if (g.d_rowp) CHECK_CUDA(cudaFree(g.d_rowp));
    if (g.d_cols) CHECK_CUDA(cudaFree(g.d_cols));
    if (g.d_vals) CHECK_CUDA(cudaFree(g.d_vals));
    if (g.d_rhs) CHECK_CUDA(cudaFree(g.d_rhs));
    if (g.d_x) CHECK_CUDA(cudaFree(g.d_x));
    if (g.stream) CHECK_CUDA(cudaStreamDestroy(g.stream));

    g.d_rowp = nullptr;
    g.d_cols = nullptr;
    g.d_vals = nullptr;
    g.d_rhs = nullptr;
    g.d_x = nullptr;
    g.stream = nullptr;
}

int main(int argc, char **argv) {
    int nx = 128;
    int requested_gpus = 0;
    int nsubdomains = 4;

    if (argc > 1) nx = std::atoi(argv[1]);
    if (argc > 2) requested_gpus = std::atoi(argv[2]);
    if (argc > 3) nsubdomains = std::atoi(argv[3]);

    const int block_dim = 6;
    const int nnode_sub = nx * nx;
    const int Nsub = block_dim * nnode_sub;
    const int Nglobal = nsubdomains * Nsub;

    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        printf("ERROR: no CUDA GPUs found\n");
        return 1;
    }

    int ngpu = requested_gpus;
    if (ngpu <= 0) ngpu = device_count;
    ngpu = std::min(ngpu, device_count);
    ngpu = std::min(ngpu, nsubdomains);

    std::vector<int> devices(ngpu);
    for (int g = 0; g < ngpu; g++) {
        devices[g] = g;
    }

    printf("nx=%d, block_dim=%d, Nsub=%d, nsubdomains=%d, Nglobal=%d, ngpu=%d\n",
           nx, block_dim, Nsub, nsubdomains, Nglobal, ngpu);

    std::vector<GpuLocalSolve> gpu_solves(ngpu);

    for (int ig = 0; ig < ngpu; ig++) {
        const int sub_start = (long long)nsubdomains * ig / ngpu;
        const int sub_end = (long long)nsubdomains * (ig + 1) / ngpu;

        GpuLocalSolve &g = gpu_solves[ig];

        g.device = devices[ig];
        g.sub_start = sub_start;
        g.sub_end = sub_end;
        g.nsubs_local = sub_end - sub_start;

        g.nx = nx;
        g.block_dim = block_dim;
        g.nnode_sub = nnode_sub;
        g.Nsub = Nsub;
    }

    auto assembly_t0 = std::chrono::high_resolution_clock::now();

    {
        std::vector<std::thread> assembly_threads;
        assembly_threads.reserve(ngpu);

        for (int ig = 0; ig < ngpu; ig++) {
            assembly_threads.emplace_back([&, ig]() {
                assemble_gpu_local_matrix(gpu_solves[ig]);
            });
        }

        for (auto &t : assembly_threads) {
            t.join();
        }
    }

    auto assembly_t1 = std::chrono::high_resolution_clock::now();
    const double assembly_wall_ms =
        std::chrono::duration<double, std::milli>(assembly_t1 - assembly_t0).count();

    printf("assembly wall time = %.3f ms\n", assembly_wall_ms);

    for (int ig = 0; ig < ngpu; ig++) {
        init_gpu_local_cudss(gpu_solves[ig]);
    }

    const double analysis_wall_ms = run_parallel_phase(
        gpu_solves,
        CUDSS_PHASE_ANALYSIS,
        "analysis"
    );

    const double factor_wall_ms = run_parallel_phase(
        gpu_solves,
        CUDSS_PHASE_FACTORIZATION,
        "factorization"
    );

    const double solve_wall_ms = run_parallel_phase(
        gpu_solves,
        CUDSS_PHASE_SOLVE,
        "solve"
    );

    double max_gpu_analysis = 0.0;
    double max_gpu_factor = 0.0;
    double max_gpu_solve = 0.0;
    double max_gpu_fact_solve = 0.0;

    for (const auto &g : gpu_solves) {
        max_gpu_analysis = std::max(max_gpu_analysis, (double)g.analysis_ms);
        max_gpu_factor = std::max(max_gpu_factor, (double)g.factor_ms);
        max_gpu_solve = std::max(max_gpu_solve, (double)g.solve_ms);
        max_gpu_fact_solve = std::max(max_gpu_fact_solve,
                                      (double)(g.factor_ms + g.solve_ms));

        printf("GPU %d cuDSS local solve: analysis %.3f ms, factor %.3f ms, solve %.3f ms, fact+solve %.3f ms\n",
               g.device,
               g.analysis_ms,
               g.factor_ms,
               g.solve_ms,
               g.factor_ms + g.solve_ms);
    }

    printf("\nGPU-local block-diagonal solve summary:\n");
    printf("  wall analysis             = %.3f ms\n", analysis_wall_ms);
    printf("  wall factorization        = %.3f ms\n", factor_wall_ms);
    printf("  wall solve                = %.3f ms\n", solve_wall_ms);
    printf("  wall factorization+solve  = %.3f ms\n", factor_wall_ms + solve_wall_ms);
    printf("  max GPU analysis event    = %.3f ms\n", max_gpu_analysis);
    printf("  max GPU factor event      = %.3f ms\n", max_gpu_factor);
    printf("  max GPU solve event       = %.3f ms\n", max_gpu_solve);
    printf("  max GPU fact+solve event  = %.3f ms\n", max_gpu_fact_solve);

    const int nz_node = 5 * nnode_sub - 4 * nx;
    const int nz_ref = block_dim * nz_node;

    std::vector<int> h_rowp(Nsub + 1), h_cols(nz_ref);
    std::vector<double> h_vals(nz_ref), h_rhs(Nsub, 0.0);

    genBlockDiagPoissonCSR<double>(
        h_rowp.data(),
        h_cols.data(),
        h_vals.data(),
        nnode_sub,
        block_dim,
        h_rhs.data()
    );

    for (int k = 0; k < nz_ref; k++) h_vals[k] *= -1.0;
    for (int i = 0; i < Nsub; i++) h_rhs[i] *= -1.0;

    double max_rel_res = 0.0;

    for (const auto &g : gpu_solves) {
        CHECK_CUDA(cudaSetDevice(g.device));

        std::vector<double> h_x_local(g.N);
        CHECK_CUDA(cudaMemcpy(h_x_local.data(),
                              g.d_x,
                              g.N * sizeof(double),
                              cudaMemcpyDeviceToHost));

        for (int ls = 0; ls < g.nsubs_local; ls++) {
            const int global_sub = g.sub_start + ls;

            const double rel_res = compute_residual_norm(
                Nsub,
                h_rowp.data(),
                h_cols.data(),
                h_vals.data(),
                h_x_local.data() + ls * Nsub,
                h_rhs.data()
            );

            max_rel_res = std::max(max_rel_res, rel_res);

            // printf("subdomain %d on GPU %d relative residual = %.15e\n",
            //        global_sub, g.device, rel_res);
        }
    }

    printf("max relative residual = %.15e\n", max_rel_res);

    for (auto &g : gpu_solves) {
        destroy_gpu_local_cudss(g);
    }

    for (auto &g : gpu_solves) {
        free_gpu_local(g);
    }

    return 0;
}