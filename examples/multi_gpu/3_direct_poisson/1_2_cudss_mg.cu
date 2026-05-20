// poisson_cudss_mg_partitioned_root_gather.cu
//
// Multi-GPU partitioned assembly, gather-to-root, then cuDSS MG solve.
//
// This follows the native cuDSS timing/solve path, but the CSR matrix/RHS are
// assembled in scalar-row partitions on the visible/requested GPUs. Each GPU owns
// a contiguous scalar-row range. The local CSR uses GLOBAL column ids. The local
// partitions are then copied/gathered to root GPU 0 before cudssMatrixCreateCsr().
//
// Test matrix: block_dim independent Poisson fields interleaved per node.
// For block_dim=6, this is 6 uncoupled Poisson problems on the same nx-by-nx grid.

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

struct GpuCsrPartition {
    int device = -1;
    int row_start = 0;       // global first owned scalar row
    int row_end = 0;         // global one-past-last owned scalar row
    int nrows = 0;
    int nnz = 0;
    int *d_rowp = nullptr;   // local row pointer, length nrows+1, starts at 0
    int *d_cols = nullptr;   // global scalar column ids
    double *d_vals = nullptr;
    double *d_rhs = nullptr;
};

__device__ __forceinline__ int blockdiag_poisson_row_nnz(
    int row,
    int nx,
    int block_dim
) {
    const int inode = row / block_dim;
    const int ix = inode % nx;
    const int iy = inode / nx;

    int ct = 1;              // center
    if (iy > 0) ct++;        // up
    if (ix > 0) ct++;        // left
    if (ix < nx - 1) ct++;   // right
    if (iy < nx - 1) ct++;   // down
    return ct;
}

__global__ void count_blockdiag_poisson_nnz_kernel(
    int nx,
    int block_dim,
    int row_start,
    int nrows,
    int *row_counts
) {
    const int lr = blockIdx.x * blockDim.x + threadIdx.x;
    if (lr >= nrows) return;

    const int row = row_start + lr;
    row_counts[lr] = blockdiag_poisson_row_nnz(row, nx, block_dim);
}

__global__ void assemble_blockdiag_poisson_csr_kernel(
    int nx,
    int block_dim,
    int row_start,
    int nrows,
    const int *rowp,
    int *cols,
    double *vals,
    double *rhs
) {
    const int lr = blockIdx.x * blockDim.x + threadIdx.x;
    if (lr >= nrows) return;

    const int row = row_start + lr;
    const int inode = row / block_dim;
    const int d = row % block_dim;
    const int ix = inode % nx;
    const int iy = inode / nx;

    int p = rowp[lr];
    double b = 0.0;

    // Directly assemble SPD version of genBlockDiagPoissonCSR after sign flip:
    // diagonal +4, off-diagonal -1, top boundary rhs +1.

    // up
    if (iy > 0) {
        const int col_node = inode - nx;
        cols[p] = block_dim * col_node + d;
        vals[p] = -1.0;
        p++;
    } else {
        b += 1.0;
    }

    // left
    if (ix > 0) {
        const int col_node = inode - 1;
        cols[p] = block_dim * col_node + d;
        vals[p] = -1.0;
        p++;
    }

    // center
    cols[p] = row;
    vals[p] = 4.0;
    p++;

    // right
    if (ix < nx - 1) {
        const int col_node = inode + 1;
        cols[p] = block_dim * col_node + d;
        vals[p] = -1.0;
        p++;
    }

    // down
    if (iy < nx - 1) {
        const int col_node = inode + nx;
        cols[p] = block_dim * col_node + d;
        vals[p] = -1.0;
        p++;
    }

    rhs[lr] = b;
}

static void exclusive_scan_counts_to_rowp_on_host(
    int device,
    int nrows,
    int *d_counts,
    int **d_rowp_out,
    int *nnz_out
) {
    std::vector<int> h_counts(nrows);
    CHECK_CUDA(cudaSetDevice(device));
    CHECK_CUDA(cudaMemcpy(h_counts.data(), d_counts, nrows * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<int> h_rowp(nrows + 1);
    h_rowp[0] = 0;
    for (int i = 0; i < nrows; i++) {
        h_rowp[i + 1] = h_rowp[i] + h_counts[i];
    }

    *nnz_out = h_rowp[nrows];
    CHECK_CUDA(cudaMalloc((void **)d_rowp_out, (nrows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(*d_rowp_out, h_rowp.data(), (nrows + 1) * sizeof(int), cudaMemcpyHostToDevice));
}

static std::vector<GpuCsrPartition> assemble_blockdiag_poisson_partitions_on_gpus(
    int nx,
    int block_dim,
    int N,
    const std::vector<int> &devices
) {
    const int ngpu = (int)devices.size();
    std::vector<GpuCsrPartition> parts(ngpu);

    for (int ig = 0; ig < ngpu; ig++) {
        const int dev = devices[ig];
        const int row_start = (long long)N * ig / ngpu;
        const int row_end = (long long)N * (ig + 1) / ngpu;
        const int nrows = row_end - row_start;

        parts[ig].device = dev;
        parts[ig].row_start = row_start;
        parts[ig].row_end = row_end;
        parts[ig].nrows = nrows;

        CHECK_CUDA(cudaSetDevice(dev));

        int *d_counts = nullptr;
        CHECK_CUDA(cudaMalloc((void **)&d_counts, nrows * sizeof(int)));

        const int block = 256;
        const int grid = (nrows + block - 1) / block;

        count_blockdiag_poisson_nnz_kernel<<<grid, block>>>(
            nx, block_dim, row_start, nrows, d_counts
        );
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        exclusive_scan_counts_to_rowp_on_host(dev, nrows, d_counts,
                                              &parts[ig].d_rowp, &parts[ig].nnz);
        CHECK_CUDA(cudaFree(d_counts));

        CHECK_CUDA(cudaMalloc((void **)&parts[ig].d_cols, parts[ig].nnz * sizeof(int)));
        CHECK_CUDA(cudaMalloc((void **)&parts[ig].d_vals, parts[ig].nnz * sizeof(double)));
        CHECK_CUDA(cudaMalloc((void **)&parts[ig].d_rhs, nrows * sizeof(double)));

        assemble_blockdiag_poisson_csr_kernel<<<grid, block>>>(
            nx,
            block_dim,
            row_start,
            nrows,
            parts[ig].d_rowp,
            parts[ig].d_cols,
            parts[ig].d_vals,
            parts[ig].d_rhs
        );
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        printf("GPU %d assembled scalar rows [%d, %d), nrows=%d, local nnz=%d\n",
               dev, row_start, row_end, nrows, parts[ig].nnz);
    }

    return parts;
}

static void enable_peer_access_best_effort(const std::vector<int> &devices) {
    for (int src : devices) {
        CHECK_CUDA(cudaSetDevice(src));
        for (int dst : devices) {
            if (src == dst) continue;
            int can_access = 0;
            CHECK_CUDA(cudaDeviceCanAccessPeer(&can_access, src, dst));
            if (!can_access) continue;

            cudaError_t e = cudaDeviceEnablePeerAccess(dst, 0);
            if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
                printf("WARNING: cudaDeviceEnablePeerAccess(%d -> %d) failed: %s\n",
                       src, dst, cudaGetErrorString(e));
                cudaGetLastError();
            } else if (e == cudaErrorPeerAccessAlreadyEnabled) {
                cudaGetLastError();
            }
        }
    }
}

// static void gather_partitions_to_root_gpu(
//     const std::vector<GpuCsrPartition> &parts,
//     int N,
//     int root_device,
//     int **d_rowp_root,
//     int **d_cols_root,
//     double **d_vals_root,
//     double **d_rhs_root,
//     int *nnz_global_out
// ) {
//     const int ngpu = (int)parts.size();

//     std::vector<int> nnz_offsets(ngpu + 1, 0);
//     for (int ig = 0; ig < ngpu; ig++) {
//         nnz_offsets[ig + 1] = nnz_offsets[ig] + parts[ig].nnz;
//     }

//     const int nnz_global = nnz_offsets[ngpu];
//     *nnz_global_out = nnz_global;

//     CHECK_CUDA(cudaSetDevice(root_device));
//     CHECK_CUDA(cudaMalloc((void **)d_rowp_root, (N + 1) * sizeof(int)));
//     CHECK_CUDA(cudaMalloc((void **)d_cols_root, nnz_global * sizeof(int)));
//     CHECK_CUDA(cudaMalloc((void **)d_vals_root, nnz_global * sizeof(double)));
//     CHECK_CUDA(cudaMalloc((void **)d_rhs_root, N * sizeof(double)));

//     std::vector<int> h_global_rowp(N + 1);

//     for (int ig = 0; ig < ngpu; ig++) {
//         const GpuCsrPartition &p = parts[ig];

//         std::vector<int> h_lrowp(p.nrows + 1);
//         CHECK_CUDA(cudaSetDevice(p.device));
//         CHECK_CUDA(cudaMemcpy(h_lrowp.data(), p.d_rowp,
//                               (p.nrows + 1) * sizeof(int),
//                               cudaMemcpyDeviceToHost));

//         for (int lr = 0; lr < p.nrows; lr++) {
//             h_global_rowp[p.row_start + lr] = nnz_offsets[ig] + h_lrowp[lr];
//         }
//         h_global_rowp[p.row_end] = nnz_offsets[ig] + h_lrowp[p.nrows];

//         CHECK_CUDA(cudaMemcpyPeer(*d_cols_root + nnz_offsets[ig], root_device,
//                                   p.d_cols, p.device,
//                                   p.nnz * sizeof(int)));
//         CHECK_CUDA(cudaMemcpyPeer(*d_vals_root + nnz_offsets[ig], root_device,
//                                   p.d_vals, p.device,
//                                   p.nnz * sizeof(double)));
//         CHECK_CUDA(cudaMemcpyPeer(*d_rhs_root + p.row_start, root_device,
//                                   p.d_rhs, p.device,
//                                   p.nrows * sizeof(double)));
//     }

//     CHECK_CUDA(cudaSetDevice(root_device));
//     CHECK_CUDA(cudaMemcpy(*d_rowp_root, h_global_rowp.data(),
//                           (N + 1) * sizeof(int),
//                           cudaMemcpyHostToDevice));
//     CHECK_CUDA(cudaDeviceSynchronize());

//     printf("Gathered global CSR on root GPU %d: N=%d, nnz=%d\n", root_device, N, nnz_global);
// }

static void gather_partitions_to_root_gpu(
    const std::vector<GpuCsrPartition> &parts,
    int N,
    int root_device,
    int **d_rowp_root,
    int **d_cols_root,
    double **d_vals_root,
    double **d_rhs_root,
    int *nnz_global_out
) {
    const int ngpu = (int)parts.size();

    std::vector<int> nnz_offsets(ngpu + 1, 0);
    for (int ig = 0; ig < ngpu; ig++) {
        nnz_offsets[ig + 1] = nnz_offsets[ig] + parts[ig].nnz;
    }

    const int nnz_global = nnz_offsets[ngpu];
    *nnz_global_out = nnz_global;

    CHECK_CUDA(cudaSetDevice(root_device));
    CHECK_CUDA(cudaMalloc((void **)d_rowp_root, (N + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)d_cols_root, nnz_global * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)d_vals_root, nnz_global * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void **)d_rhs_root, N * sizeof(double)));

    std::vector<int> h_global_rowp(N + 1);

    double total_cols_ms = 0.0;
    double total_vals_ms = 0.0;
    double total_rhs_ms  = 0.0;
    double total_peer_ms = 0.0;

    printf("\nGather-to-root timing, root GPU %d:\n", root_device);

    for (int ig = 0; ig < ngpu; ig++) {
        const GpuCsrPartition &p = parts[ig];

        std::vector<int> h_lrowp(p.nrows + 1);
        CHECK_CUDA(cudaSetDevice(p.device));
        CHECK_CUDA(cudaMemcpy(h_lrowp.data(), p.d_rowp,
                              (p.nrows + 1) * sizeof(int),
                              cudaMemcpyDeviceToHost));

        for (int lr = 0; lr < p.nrows; lr++) {
            h_global_rowp[p.row_start + lr] = nnz_offsets[ig] + h_lrowp[lr];
        }
        h_global_rowp[p.row_end] = nnz_offsets[ig] + h_lrowp[p.nrows];

        double cols_ms = 0.0;
        double vals_ms = 0.0;
        double rhs_ms  = 0.0;

        if (p.device != root_device) {
            cudaEvent_t start, stop;
            CHECK_CUDA(cudaSetDevice(root_device));
            CHECK_CUDA(cudaEventCreate(&start));
            CHECK_CUDA(cudaEventCreate(&stop));

            CHECK_CUDA(cudaEventRecord(start, 0));
            CHECK_CUDA(cudaMemcpyPeer(*d_cols_root + nnz_offsets[ig], root_device,
                                      p.d_cols, p.device,
                                      p.nnz * sizeof(int)));
            CHECK_CUDA(cudaEventRecord(stop, 0));
            CHECK_CUDA(cudaEventSynchronize(stop));
            float ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
            cols_ms = ms;

            CHECK_CUDA(cudaEventRecord(start, 0));
            CHECK_CUDA(cudaMemcpyPeer(*d_vals_root + nnz_offsets[ig], root_device,
                                      p.d_vals, p.device,
                                      p.nnz * sizeof(double)));
            CHECK_CUDA(cudaEventRecord(stop, 0));
            CHECK_CUDA(cudaEventSynchronize(stop));
            CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
            vals_ms = ms;

            CHECK_CUDA(cudaEventRecord(start, 0));
            CHECK_CUDA(cudaMemcpyPeer(*d_rhs_root + p.row_start, root_device,
                                      p.d_rhs, p.device,
                                      p.nrows * sizeof(double)));
            CHECK_CUDA(cudaEventRecord(stop, 0));
            CHECK_CUDA(cudaEventSynchronize(stop));
            CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
            rhs_ms = ms;

            CHECK_CUDA(cudaEventDestroy(start));
            CHECK_CUDA(cudaEventDestroy(stop));
        } else {
            // Already on root GPU: no inter-GPU transfer cost.
            CHECK_CUDA(cudaSetDevice(root_device));
            CHECK_CUDA(cudaMemcpy(*d_cols_root + nnz_offsets[ig],
                                  p.d_cols,
                                  p.nnz * sizeof(int),
                                  cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpy(*d_vals_root + nnz_offsets[ig],
                                  p.d_vals,
                                  p.nnz * sizeof(double),
                                  cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpy(*d_rhs_root + p.row_start,
                                  p.d_rhs,
                                  p.nrows * sizeof(double),
                                  cudaMemcpyDeviceToDevice));
        }

        const double part_ms = cols_ms + vals_ms + rhs_ms;
        total_cols_ms += cols_ms;
        total_vals_ms += vals_ms;
        total_rhs_ms  += rhs_ms;
        total_peer_ms += part_ms;

        const double mb = 1.0e-6 * (
            p.nnz * (sizeof(int) + sizeof(double)) +
            p.nrows * sizeof(double)
        );

        printf("  GPU %d -> root: cols=%.3f ms, vals=%.3f ms, rhs=%.3f ms, total=%.3f ms, moved=%.3f MB\n",
               p.device, cols_ms, vals_ms, rhs_ms, part_ms, mb);
    }

    CHECK_CUDA(cudaSetDevice(root_device));
    CHECK_CUDA(cudaMemcpy(*d_rowp_root, h_global_rowp.data(),
                          (N + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("Gather totals: cols=%.3f ms, vals=%.3f ms, rhs=%.3f ms, interGPU_total=%.3f ms\n",
           total_cols_ms, total_vals_ms, total_rhs_ms, total_peer_ms);

    printf("Gathered global CSR on root GPU %d: N=%d, nnz=%d\n",
           root_device, N, nnz_global);
}

static void free_partitions(std::vector<GpuCsrPartition> &parts) {
    for (auto &p : parts) {
        CHECK_CUDA(cudaSetDevice(p.device));
        if (p.d_rowp) CHECK_CUDA(cudaFree(p.d_rowp));
        if (p.d_cols) CHECK_CUDA(cudaFree(p.d_cols));
        if (p.d_vals) CHECK_CUDA(cudaFree(p.d_vals));
        if (p.d_rhs) CHECK_CUDA(cudaFree(p.d_rhs));
        p.d_rowp = nullptr;
        p.d_cols = nullptr;
        p.d_vals = nullptr;
        p.d_rhs = nullptr;
    }
}

static void solve_cudss_mg_root_device_csr(
    int N,
    int nnz,
    int *d_rowp,
    int *d_cols,
    double *d_vals,
    double *d_b,
    double *h_x,
    const std::vector<int> &devices,
    int root_device
) {
    const int ngpu = (int)devices.size();

    CHECK_CUDA(cudaSetDevice(root_device));

    double *d_x = nullptr;
    CHECK_CUDA(cudaMalloc((void **)&d_x, N * sizeof(double)));
    CHECK_CUDA(cudaMemset(d_x, 0, N * sizeof(double)));

    cudssHandle_t handle = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t data = nullptr;
    cudssMatrix_t A = nullptr;
    cudssMatrix_t B = nullptr;
    cudssMatrix_t X = nullptr;

    CHECK_CUDSS(cudssCreateMg(&handle, ngpu, const_cast<int *>(devices.data())));
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
    CHECK_CUDSS(cudssMatrixCreateDn(&B, N, nrhs, N, d_b, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));
    CHECK_CUDSS(cudssMatrixCreateDn(&X, N, nrhs, N, d_x, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));

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
    printf("cuDSS timings: analysis=%.3f ms, factor=%.3f ms, solve=%.3f ms\n",
           analysis_ms, factor_ms, solve_ms);
    printf("Total solve (fact+solve)=%.3f ms\n", fact_solve_ms);

    CHECK_CUDA(cudaMemcpy(h_x, d_x, N * sizeof(double), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventDestroy(ev_start));
    CHECK_CUDA(cudaEventDestroy(ev_stop));

    CHECK_CUDSS(cudssMatrixDestroy(A));
    CHECK_CUDSS(cudssMatrixDestroy(B));
    CHECK_CUDSS(cudssMatrixDestroy(X));
    CHECK_CUDSS(cudssDataDestroy(handle, data));
    CHECK_CUDSS(cudssConfigDestroy(config));
    CHECK_CUDSS(cudssDestroy(handle));

    CHECK_CUDA(cudaFree(d_x));
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

int main(int argc, char **argv) {
    using T = double;

    int nx = 128;
    int requested_gpus = 0;  // <=0 means use all visible CUDA GPUs

    if (argc > 1) nx = std::atoi(argv[1]);
    if (argc > 2) requested_gpus = std::atoi(argv[2]);

    const int block_dim = 6;
    const int nnode = nx * nx;
    const int N = block_dim * nnode;

    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        printf("ERROR: no CUDA GPUs found\n");
        return 1;
    }

    int ngpu = std::min(requested_gpus, device_count);
    if (ngpu <= 0) ngpu = device_count;

    std::vector<int> devices(ngpu);
    for (int g = 0; g < ngpu; g++) devices[g] = g;
    const int root = devices[0];

    printf("nx=%d, block_dim=%d, scalar N=%d, ngpu=%d\n", nx, block_dim, N, ngpu);

    enable_peer_access_best_effort(devices);

    std::vector<GpuCsrPartition> parts = assemble_blockdiag_poisson_partitions_on_gpus(
        nx, block_dim, N, devices
    );

    int *d_rowp_root = nullptr;
    int *d_cols_root = nullptr;
    double *d_vals_root = nullptr;
    double *d_rhs_root = nullptr;
    int nnz = 0;

    gather_partitions_to_root_gpu(parts, N, root,
                                  &d_rowp_root, &d_cols_root,
                                  &d_vals_root, &d_rhs_root,
                                  &nnz);

    std::vector<double> x(N, 0.0);
    solve_cudss_mg_root_device_csr(N, nnz,
                                   d_rowp_root, d_cols_root, d_vals_root, d_rhs_root,
                                   x.data(), devices, root);

    // Host reference using include/poisson.h, then same sign flip as native CSR tests.
    const int nz_node = 5 * nnode - 4 * nx;
    const int nz_ref = block_dim * nz_node;
    std::vector<int> h_rowp(N + 1), h_cols(nz_ref);
    std::vector<double> h_vals(nz_ref), h_rhs(N, 0.0);

    genBlockDiagPoissonCSR<double>(h_rowp.data(), h_cols.data(), h_vals.data(),
                                   nnode, block_dim, h_rhs.data());
    for (int k = 0; k < nz_ref; k++) h_vals[k] *= -1.0;
    for (int i = 0; i < N; i++) h_rhs[i] *= -1.0;

    const double rel_res = compute_residual_norm(N, h_rowp.data(), h_cols.data(),
                                                 h_vals.data(), x.data(), h_rhs.data());

    printf("relative residual = %.15e\n", rel_res);

    CHECK_CUDA(cudaSetDevice(root));
    CHECK_CUDA(cudaFree(d_rowp_root));
    CHECK_CUDA(cudaFree(d_cols_root));
    CHECK_CUDA(cudaFree(d_vals_root));
    CHECK_CUDA(cudaFree(d_rhs_root));

    free_partitions(parts);

    return 0;
}