// cudss distributed (but don't copy back to root, try to keep distributed)a// poisson_cudss_mgmn_distributed_csr.cu
//
// True distributed-input cuDSS MGMN testbed:
//   - one MPI rank per GPU is recommended
//   - each rank assembles only its owned CSR rows on its local GPU
//   - NO gather/copy of the matrix to a root GPU
//   - cudssMatrixSetDistributionRow1d() tells cuDSS the row ownership
//   - cuDSS MGMN uses its communication layer internally during analysis/factor/solve
//
// Usage example:
//   export CUDSS_COMM_LIB=$CUDSS_ROOT/lib/libcudss_commlayer_openmpi.so.0
//   mpicxx -std=c++17 -O3 -I$CUDA_HOME/include -I$CUDSS_ROOT/include \
//     poisson_cudss_mgmn_distributed_csr.cu \
//     -L$CUDA_HOME/lib64 -lcudart \
//     -L$CUDSS_ROOT/lib -l:libcudss.so.0 -lmpi -o 6_cudss_mgmn_dist.out
//
//   mpirun -np 4 ./6_cudss_mgmn_dist.out 128 (for a 4 GPU and 4 CPU case)
//
// Notes:
//   * There is no explicit inter-GPU matrix gather in this code.
//   * cuDSS will still communicate internally during MGMN analysis/factor/solve.
//   * Residual checking gathers only the final solution vector on rank 0.

#include <mpi.h>

#include <cuda_runtime.h>
#include <cudss.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#include "include/poisson.h"

#define CHECK_CUDA(x) do {                                      \
    cudaError_t e = (x);                                        \
    if (e != cudaSuccess) {                                     \
        printf("CUDA error %s at %s:%d\n",                    \
               cudaGetErrorString(e), __FILE__, __LINE__);      \
        MPI_Abort(MPI_COMM_WORLD, 1);                           \
    }                                                           \
} while (0)

#define CHECK_CUDSS(x) do {                                     \
    cudssStatus_t s = (x);                                      \
    if (s != CUDSS_STATUS_SUCCESS) {                            \
        printf("cuDSS error %d at %s:%d\n",                   \
               (int)s, __FILE__, __LINE__);                     \
        MPI_Abort(MPI_COMM_WORLD, 2);                           \
    }                                                           \
} while (0)

struct LocalCsr {
    int first_row = 0;       // global first owned scalar row
    int last_row = -1;       // global last owned scalar row, inclusive
    int nrows = 0;
    int local_nnz = 0;
    int global_nnz = 0;
    int *d_rowp = nullptr;   // local CSR row ptr, length nrows+1, starts at 0
    int *d_cols = nullptr;   // global scalar column ids
    double *d_vals = nullptr;
    double *d_rhs = nullptr;
    double *d_x = nullptr;
};

__device__ __forceinline__ int blockdiag_poisson_row_nnz(
    int row,
    int nx,
    int block_dim
) {
    const int inode = row / block_dim;
    const int ix = inode % nx;
    const int iy = inode / nx;

    int ct = 1;
    if (iy > 0) ct++;
    if (ix > 0) ct++;
    if (ix < nx - 1) ct++;
    if (iy < nx - 1) ct++;
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
    row_counts[lr] = blockdiag_poisson_row_nnz(row_start + lr, nx, block_dim);
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

    // SPD version: diag +4, offdiag -1, top boundary rhs +1.
    if (iy > 0) {
        cols[p] = block_dim * (inode - nx) + d;
        vals[p] = -1.0;
        p++;
    } else {
        b += 1.0;
    }

    if (ix > 0) {
        cols[p] = block_dim * (inode - 1) + d;
        vals[p] = -1.0;
        p++;
    }

    cols[p] = row;
    vals[p] = 4.0;
    p++;

    if (ix < nx - 1) {
        cols[p] = block_dim * (inode + 1) + d;
        vals[p] = -1.0;
        p++;
    }

    if (iy < nx - 1) {
        cols[p] = block_dim * (inode + nx) + d;
        vals[p] = -1.0;
        p++;
    }

    rhs[lr] = b;
}

static void assemble_local_csr_on_rank_gpu(
    int nx,
    int block_dim,
    int N,
    int global_nnz,
    int rank,
    int nranks,
    LocalCsr &A
) {
    A.first_row = (long long)N * rank / nranks;
    const int row_end_excl = (long long)N * (rank + 1) / nranks;
    A.last_row = row_end_excl - 1;
    A.nrows = row_end_excl - A.first_row;
    A.global_nnz = global_nnz;

    int *d_counts = nullptr;
    CHECK_CUDA(cudaMalloc((void **)&d_counts, A.nrows * sizeof(int)));

    const int block = 256;
    const int grid = (A.nrows + block - 1) / block;
    count_blockdiag_poisson_nnz_kernel<<<grid, block>>>(
        nx, block_dim, A.first_row, A.nrows, d_counts
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<int> h_counts(A.nrows);
    CHECK_CUDA(cudaMemcpy(h_counts.data(), d_counts, A.nrows * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_counts));

    std::vector<int> h_rowp(A.nrows + 1, 0);
    for (int i = 0; i < A.nrows; i++) h_rowp[i + 1] = h_rowp[i] + h_counts[i];
    A.local_nnz = h_rowp[A.nrows];

    CHECK_CUDA(cudaMalloc((void **)&A.d_rowp, (A.nrows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&A.d_cols, A.local_nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&A.d_vals, A.local_nnz * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void **)&A.d_rhs, A.nrows * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void **)&A.d_x, A.nrows * sizeof(double)));
    CHECK_CUDA(cudaMemset(A.d_x, 0, A.nrows * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(A.d_rowp, h_rowp.data(), (A.nrows + 1) * sizeof(int), cudaMemcpyHostToDevice));

    assemble_blockdiag_poisson_csr_kernel<<<grid, block>>>(
        nx, block_dim, A.first_row, A.nrows, A.d_rowp, A.d_cols, A.d_vals, A.d_rhs
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("rank %d assembled local rows [%d, %d], nrows=%d, local_nnz=%d\n",
           rank, A.first_row, A.last_row, A.nrows, A.local_nnz);
}

static void free_local_csr(LocalCsr &A) {
    if (A.d_rowp) CHECK_CUDA(cudaFree(A.d_rowp));
    if (A.d_cols) CHECK_CUDA(cudaFree(A.d_cols));
    if (A.d_vals) CHECK_CUDA(cudaFree(A.d_vals));
    if (A.d_rhs) CHECK_CUDA(cudaFree(A.d_rhs));
    if (A.d_x) CHECK_CUDA(cudaFree(A.d_x));
    A = LocalCsr{};
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
    MPI_Init(&argc, &argv);

    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    int nx = 128;
    if (argc > 1) nx = std::atoi(argv[1]);

    const int block_dim = 6;
    const int nnode = nx * nx;
    const int N = block_dim * nnode;
    const int nz_node = 5 * nnode - 4 * nx;
    const int global_nnz = block_dim * nz_node;

    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        if (rank == 0) printf("ERROR: no CUDA GPUs found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (device_count != nranks) {
        printf("WARNING: want exactly one GPU per MPI rank, but you have %d MPI ranks and %d GPUs\n", nranks, device_count);
    }

    const int device = rank % device_count;
    CHECK_CUDA(cudaSetDevice(device));

    if (rank == 0) {
        printf("cuDSS MGMN distributed CSR input: nx=%d, block_dim=%d, scalar N=%d, global_nnz=%d, nranks=%d\n",
               nx, block_dim, N, global_nnz, nranks);
        printf("No explicit matrix gather/copy-to-root is performed in this code.\n");
    }
    printf("rank %d using CUDA device %d\n", rank, device);

    LocalCsr local;
    assemble_local_csr_on_rank_gpu(nx, block_dim, N, global_nnz, rank, nranks, local);

    cudssHandle_t handle = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t data = nullptr;
    cudssMatrix_t A = nullptr;
    cudssMatrix_t B = nullptr;
    cudssMatrix_t X = nullptr;

    CHECK_CUDSS(cudssCreate(&handle));

    // NULL means cuDSS reads the communication-layer path from CUDSS_COMM_LIB.
    // For OpenMPI, this is usually libcudss_commlayer_openmpi.so.0.
    CHECK_CUDSS(cudssSetCommLayer(handle, NULL));

    CHECK_CUDSS(cudssConfigCreate(&config));
    CHECK_CUDSS(cudssDataCreate(handle, &data));

    MPI_Comm comm = MPI_COMM_WORLD;
    CHECK_CUDSS(cudssDataSet(handle, data, CUDSS_DATA_COMM, &comm, sizeof(MPI_Comm)));

    CHECK_CUDSS(cudssMatrixCreateCsr(
        &A,
        N,
        N,
        global_nnz,
        local.d_rowp,
        NULL,
        local.d_cols,
        local.d_vals,
        CUDA_R_32I,
        CUDA_R_64F,
        CUDSS_MTYPE_SPD,
        CUDSS_MVIEW_FULL,
        CUDSS_BASE_ZERO
    ));

    // In distributed row mode, first/last are 0-based and inclusive.
    int64_t first_row = local.first_row;
    int64_t last_row = local.last_row;
    // CHECK_CUDSS(cudssMatrixSetDistributionRow1d(A, &first_row, &last_row));
    CHECK_CUDSS(cudssMatrixSetDistributionRow1d(A, first_row, last_row));

    const int nrhs = 1;
    CHECK_CUDSS(cudssMatrixCreateDn(
        &B,
        N,
        nrhs,
        local.nrows,
        local.d_rhs,
        CUDA_R_64F,
        CUDSS_LAYOUT_COL_MAJOR
    ));
    CHECK_CUDSS(cudssMatrixCreateDn(
        &X,
        N,
        nrhs,
        local.nrows,
        local.d_x,
        CUDA_R_64F,
        CUDSS_LAYOUT_COL_MAJOR
    ));
    // CHECK_CUDSS(cudssMatrixSetDistributionRow1d(B, &first_row, &last_row));
    // CHECK_CUDSS(cudssMatrixSetDistributionRow1d(X, &first_row, &last_row));
    CHECK_CUDSS(cudssMatrixSetDistributionRow1d(B, first_row, last_row));
    CHECK_CUDSS(cudssMatrixSetDistributionRow1d(X, first_row, last_row));

    cudaEvent_t ev_start, ev_stop;
    CHECK_CUDA(cudaEventCreate(&ev_start));
    CHECK_CUDA(cudaEventCreate(&ev_stop));

    float analysis_ms = 0.0f, factor_ms = 0.0f, solve_ms = 0.0f;

    MPI_Barrier(MPI_COMM_WORLD);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(ev_start, 0));
    CHECK_CUDSS(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, config, data, A, X, B));
    CHECK_CUDA(cudaEventRecord(ev_stop, 0));
    CHECK_CUDA(cudaEventSynchronize(ev_stop));
    CHECK_CUDA(cudaEventElapsedTime(&analysis_ms, ev_start, ev_stop));

    MPI_Barrier(MPI_COMM_WORLD);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(ev_start, 0));
    CHECK_CUDSS(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, config, data, A, X, B));
    CHECK_CUDA(cudaEventRecord(ev_stop, 0));
    CHECK_CUDA(cudaEventSynchronize(ev_stop));
    CHECK_CUDA(cudaEventElapsedTime(&factor_ms, ev_start, ev_stop));

    MPI_Barrier(MPI_COMM_WORLD);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(ev_start, 0));
    CHECK_CUDSS(cudssExecute(handle, CUDSS_PHASE_SOLVE, config, data, A, X, B));
    CHECK_CUDA(cudaEventRecord(ev_stop, 0));
    CHECK_CUDA(cudaEventSynchronize(ev_stop));
    CHECK_CUDA(cudaEventElapsedTime(&solve_ms, ev_start, ev_stop));

    float max_analysis = 0.0f, max_factor = 0.0f, max_solve = 0.0f;
    MPI_Reduce(&analysis_ms, &max_analysis, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&factor_ms, &max_factor, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&solve_ms, &max_solve, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("cuDSS MGMN timings: analysis=%.3f ms, factor=%.3f ms, solve=%.3f ms\n",
               max_analysis, max_factor, max_solve);
        printf("Total solve (fact+solve)=%.3f ms\n", max_factor + max_solve);
    }

    // Gather the solution vector only for residual checking. This is not a matrix gather.
    std::vector<int> recvcounts, displs;
    std::vector<double> x_local(local.nrows);
    CHECK_CUDA(cudaMemcpy(x_local.data(), local.d_x, local.nrows * sizeof(double), cudaMemcpyDeviceToHost));

    if (rank == 0) {
        recvcounts.resize(nranks);
        displs.resize(nranks);
        for (int r = 0; r < nranks; r++) {
            const int rs = (long long)N * r / nranks;
            const int re = (long long)N * (r + 1) / nranks;
            recvcounts[r] = re - rs;
            displs[r] = rs;
        }
    }

    std::vector<double> x_global;
    if (rank == 0) x_global.resize(N);

    MPI_Gatherv(x_local.data(), local.nrows, MPI_DOUBLE,
                rank == 0 ? x_global.data() : nullptr,
                rank == 0 ? recvcounts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::vector<int> h_rowp(N + 1), h_cols(global_nnz);
        std::vector<double> h_vals(global_nnz), h_rhs(N, 0.0);
        genBlockDiagPoissonCSR<double>(h_rowp.data(), h_cols.data(), h_vals.data(),
                                       nnode, block_dim, h_rhs.data());
        for (int k = 0; k < global_nnz; k++) h_vals[k] *= -1.0;
        for (int i = 0; i < N; i++) h_rhs[i] *= -1.0;

        const double rel_res = compute_residual_norm(
            N, h_rowp.data(), h_cols.data(), h_vals.data(), x_global.data(), h_rhs.data()
        );
        printf("relative residual = %.15e\n", rel_res);
    }

    CHECK_CUDA(cudaEventDestroy(ev_start));
    CHECK_CUDA(cudaEventDestroy(ev_stop));

    CHECK_CUDSS(cudssMatrixDestroy(A));
    CHECK_CUDSS(cudssMatrixDestroy(B));
    CHECK_CUDSS(cudssMatrixDestroy(X));
    CHECK_CUDSS(cudssDataDestroy(handle, data));
    CHECK_CUDSS(cudssConfigDestroy(config));
    CHECK_CUDSS(cudssDestroy(handle));

    free_local_csr(local);

    MPI_Finalize();
    return 0;
}