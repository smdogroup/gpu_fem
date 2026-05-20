// poisson_cudss_mgmn_distributed_csr_streamed.cu
//
// cuDSS MGMN distributed CSR testbed, rewritten for one MPI rank per GPU.
//
// Important:
//   * This is NOT the same as cudssCreateMg() single-process MG.
//   * MGMN is distributed multi-process mode.
//   * Use one MPI rank per GPU, e.g. mpirun -np 4 ./a.out 128.
//   * No explicit matrix gather/copy-to-root is performed.
//   * Each rank owns a contiguous row slab: local CSR is local_nrows x global_N.
//   * cuDSS communicates internally during analysis/factor/solve.
//   * Timings use MPI_Wtime + MPI_MAX, which is more meaningful for synchronous
//     distributed phases than per-rank CUDA events alone.
//
// Build notes:
//   link with -l:libcudss.so.0 if installed from pip wheel.
//   export CUDSS_COMM_LIB=/path/to/libcudss_commlayer_openmpi.so.0

#include <mpi.h>

#include <cuda_runtime.h>
#include <cudss.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "include/poisson.h"

#define CHECK_CUDA(call) do {                                             \
    cudaError_t e = (call);                                               \
    if (e != cudaSuccess) {                                               \
        int _rank = -1;                                                   \
        MPI_Comm_rank(MPI_COMM_WORLD, &_rank);                            \
        printf("[rank %d] CUDA error %s at %s:%d\n",                     \
               _rank, cudaGetErrorString(e), __FILE__, __LINE__);         \
        MPI_Abort(MPI_COMM_WORLD, 1);                                     \
    }                                                                     \
} while (0)

#define CHECK_CUDSS(call) do {                                            \
    cudssStatus_t s = (call);                                             \
    if (s != CUDSS_STATUS_SUCCESS) {                                      \
        int _rank = -1;                                                   \
        MPI_Comm_rank(MPI_COMM_WORLD, &_rank);                            \
        printf("[rank %d] cuDSS error %d at %s:%d\n",                    \
               _rank, (int)s, __FILE__, __LINE__);                        \
        MPI_Abort(MPI_COMM_WORLD, 2);                                     \
    }                                                                     \
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

static int get_local_rank_fallback(int global_rank) {
    const char *vars[] = {
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "SLURM_LOCALID",
        "MV2_COMM_WORLD_LOCAL_RANK",
        "PMI_LOCAL_RANK"
    };

    for (const char *v : vars) {
        const char *s = std::getenv(v);
        if (s) return std::atoi(s);
    }

    // Fallback only. Correct on single-node runs, not robust on multi-node.
    return global_rank;
}

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

    // SPD version of block-diagonal Poisson:
    // diag +4, offdiag -1, top boundary rhs +1.

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
    cudaStream_t stream,
    LocalCsr &A
) {
    A.first_row = (long long)N * rank / nranks;
    const int row_end_excl = (long long)N * (rank + 1) / nranks;
    A.last_row = row_end_excl - 1;
    A.nrows = row_end_excl - A.first_row;
    A.global_nnz = global_nnz;

    const int block = 256;
    const int grid = (A.nrows + block - 1) / block;

    int *d_counts = nullptr;
    CHECK_CUDA(cudaMalloc((void **)&d_counts, A.nrows * sizeof(int)));

    count_blockdiag_poisson_nnz_kernel<<<grid, block, 0, stream>>>(
        nx, block_dim, A.first_row, A.nrows, d_counts
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::vector<int> h_counts(A.nrows);
    CHECK_CUDA(cudaMemcpyAsync(h_counts.data(), d_counts,
                               A.nrows * sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaFree(d_counts));

    std::vector<int> h_rowp(A.nrows + 1, 0);
    for (int i = 0; i < A.nrows; i++) {
        h_rowp[i + 1] = h_rowp[i] + h_counts[i];
    }
    A.local_nnz = h_rowp[A.nrows];

    CHECK_CUDA(cudaMalloc((void **)&A.d_rowp, (A.nrows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&A.d_cols, A.local_nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&A.d_vals, A.local_nnz * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void **)&A.d_rhs, A.nrows * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void **)&A.d_x, A.nrows * sizeof(double)));

    CHECK_CUDA(cudaMemcpyAsync(A.d_rowp, h_rowp.data(),
                               (A.nrows + 1) * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemsetAsync(A.d_x, 0, A.nrows * sizeof(double), stream));

    assemble_blockdiag_poisson_csr_kernel<<<grid, block, 0, stream>>>(
        nx, block_dim, A.first_row, A.nrows,
        A.d_rowp, A.d_cols, A.d_vals, A.d_rhs
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream));

    printf("[rank %d] rows=[%d,%d], nrows=%d, local_nnz=%d\n",
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

static double time_cudss_phase_max(
    cudssHandle_t handle,
    cudssPhase_t phase,
    cudssConfig_t config,
    cudssData_t data,
    cudssMatrix_t A,
    cudssMatrix_t X,
    cudssMatrix_t B,
    cudaStream_t stream,
    MPI_Comm comm
) {
    CHECK_CUDA(cudaStreamSynchronize(stream));
    MPI_Barrier(comm);

    const double t0 = MPI_Wtime();
    CHECK_CUDSS(cudssExecute(handle, phase, config, data, A, X, B));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    const double t1 = MPI_Wtime();

    const double local_ms = 1000.0 * (t1 - t0);
    double max_ms = 0.0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    return max_ms;
}

int main(int argc, char **argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, nranks = 1;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);

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
        MPI_Abort(comm, 1);
    }

    const int local_rank = get_local_rank_fallback(rank);
    const int device = local_rank % device_count;
    CHECK_CUDA(cudaSetDevice(device));

    if (local_rank >= device_count) {
        printf("[rank %d] WARNING: local_rank=%d but only %d visible GPU(s); multiple ranks may share GPU %d\n",
               rank, local_rank, device_count, device);
    }

    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    if (rank == 0) {
        printf("cuDSS MGMN distributed CSR input: nx=%d, block_dim=%d, scalar N=%d, global_nnz=%d, nranks=%d\n",
               nx, block_dim, N, global_nnz, nranks);
        printf("No explicit matrix gather/copy-to-root is performed. Timings use MPI_Wtime max across ranks.\n");
        printf("NOTE: MGMN is distributed MPI mode; it is not equivalent to cudssCreateMg single-process MG.\n");
    }

    printf("[rank %d] local_rank=%d using CUDA device %d\n", rank, local_rank, device);

    LocalCsr local;
    double t_asm0 = MPI_Wtime();
    assemble_local_csr_on_rank_gpu(nx, block_dim, N, global_nnz, rank, nranks, stream, local);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    double t_asm1 = MPI_Wtime();

    const double asm_local_ms = 1000.0 * (t_asm1 - t_asm0);
    double asm_max_ms = 0.0;
    MPI_Reduce(&asm_local_ms, &asm_max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    cudssHandle_t handle = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t data = nullptr;
    cudssMatrix_t A = nullptr;
    cudssMatrix_t B = nullptr;
    cudssMatrix_t X = nullptr;

    CHECK_CUDSS(cudssCreate(&handle));
    CHECK_CUDSS(cudssSetStream(handle, stream));

    const char *comm_lib = std::getenv("CUDSS_COMM_LIB");
    if (!comm_lib) {
        if (rank == 0) {
            printf("ERROR: CUDSS_COMM_LIB is not set. Example:\n");
            printf("  export CUDSS_COMM_LIB=$CUDSS_ROOT/lib/libcudss_commlayer_openmpi.so.0\n");
        }
        MPI_Abort(comm, 2);
    }
    CHECK_CUDSS(cudssSetCommLayer(handle, comm_lib));

    CHECK_CUDSS(cudssConfigCreate(&config));
    CHECK_CUDSS(cudssDataCreate(handle, &data));

    MPI_Comm cudss_comm = comm;
    CHECK_CUDSS(cudssDataSet(handle, data, CUDSS_DATA_COMM,
                             &cudss_comm, sizeof(MPI_Comm)));

    CHECK_CUDSS(cudssMatrixCreateCsr(
        &A,
        N,
        N,
        global_nnz,       // global nnz required in MGMN
        local.d_rowp,     // local row pointer
        NULL,
        local.d_cols,     // global column ids for local rows
        local.d_vals,
        CUDA_R_32I,
        CUDA_R_64F,
        CUDSS_MTYPE_SPD,
        CUDSS_MVIEW_FULL,
        CUDSS_BASE_ZERO
    ));

    const int64_t first_row = (int64_t)local.first_row;
    const int64_t last_row  = (int64_t)local.last_row;  // inclusive
    CHECK_CUDSS(cudssMatrixSetDistributionRow1d(A, first_row, last_row));

    const int nrhs = 1;
    CHECK_CUDSS(cudssMatrixCreateDn(
        &B,
        N,              // global rows
        nrhs,
        local.nrows,    // local leading dimension
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

    CHECK_CUDSS(cudssMatrixSetDistributionRow1d(B, first_row, last_row));
    CHECK_CUDSS(cudssMatrixSetDistributionRow1d(X, first_row, last_row));

    const double analysis_ms = time_cudss_phase_max(
        handle, CUDSS_PHASE_ANALYSIS, config, data, A, X, B, stream, comm
    );
    const double factor_ms = time_cudss_phase_max(
        handle, CUDSS_PHASE_FACTORIZATION, config, data, A, X, B, stream, comm
    );
    const double solve_ms = time_cudss_phase_max(
        handle, CUDSS_PHASE_SOLVE, config, data, A, X, B, stream, comm
    );

    if (rank == 0) {
        printf("Assembly max=%.3f ms\n", asm_max_ms);
        printf("cuDSS MGMN timings: analysis=%.3f ms, factor=%.3f ms, solve=%.3f ms\n",
               analysis_ms, factor_ms, solve_ms);
        printf("Total solve (fact+solve)=%.3f ms\n", factor_ms + solve_ms);
    }

    // Gather solution only for residual checking. This is not a matrix gather.
    std::vector<double> x_local(local.nrows);
    CHECK_CUDA(cudaMemcpyAsync(x_local.data(), local.d_x,
                               local.nrows * sizeof(double),
                               cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::vector<int> recvcounts;
    std::vector<int> displs;

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
                MPI_DOUBLE, 0, comm);

    if (rank == 0) {
        std::vector<int> h_rowp(N + 1), h_cols(global_nnz);
        std::vector<double> h_vals(global_nnz), h_rhs(N, 0.0);

        genBlockDiagPoissonCSR<double>(h_rowp.data(), h_cols.data(), h_vals.data(),
                                       nnode, block_dim, h_rhs.data());
        for (int k = 0; k < global_nnz; k++) h_vals[k] *= -1.0;
        for (int i = 0; i < N; i++) h_rhs[i] *= -1.0;

        const double rel_res = compute_residual_norm(
            N, h_rowp.data(), h_cols.data(), h_vals.data(),
            x_global.data(), h_rhs.data()
        );

        printf("relative residual = %.15e\n", rel_res);
    }

    CHECK_CUDSS(cudssMatrixDestroy(A));
    CHECK_CUDSS(cudssMatrixDestroy(B));
    CHECK_CUDSS(cudssMatrixDestroy(X));
    CHECK_CUDSS(cudssDataDestroy(handle, data));
    CHECK_CUDSS(cudssConfigDestroy(config));
    CHECK_CUDSS(cudssDestroy(handle));

    free_local_csr(local);

    CHECK_CUDA(cudaStreamDestroy(stream));

    MPI_Finalize();
    return 0;
}