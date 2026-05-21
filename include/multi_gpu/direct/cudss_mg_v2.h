#pragma once
#include <cuda_runtime.h>
#include <cudss.h>

#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#include "MultiGPUContext.h"
#include "cuda_utils.h"

#define CHECK_CUDSS(x)                                                       \
    do {                                                                     \
        cudssStatus_t s = (x);                                               \
        if (s != CUDSS_STATUS_SUCCESS) {                                     \
            printf("cuDSS error %d at %s:%d\n", (int)s, __FILE__, __LINE__); \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

template <typename T>
struct CudssMgType;
template <>
struct CudssMgType<double> {
    static constexpr cudaDataType_t type = CUDA_R_64F;
};
template <>
struct CudssMgType<float> {
    static constexpr cudaDataType_t type = CUDA_R_32F;
};

template <typename T>
__global__ void bsr_to_local_csr_kernel(int mb, int block_dim, const int *bsr_rowp,
                                        const int *bsr_cols, const T *bsr_vals,
                                        const int *local_csr_rowp, int *local_csr_cols,
                                        T *local_csr_vals) {
    int scalar_row = blockIdx.x * blockDim.x + threadIdx.x;
    int nscalar_rows = mb * block_dim;
    if (scalar_row >= nscalar_rows) return;

    int brow = scalar_row / block_dim;
    int bi = scalar_row % block_dim;

    int p = local_csr_rowp[scalar_row];

    for (int jp = bsr_rowp[brow]; jp < bsr_rowp[brow + 1]; jp++) {
        int bcol = bsr_cols[jp];

        for (int bj = 0; bj < block_dim; bj++) {
            local_csr_cols[p] = bcol * block_dim + bj;
            local_csr_vals[p] = bsr_vals[jp * block_dim * block_dim + bi * block_dim + bj];
            p++;
        }
    }
}

template <typename T>
__global__ void scatter_local_csr_to_root_kernel(int mb, int block_dim, const int *Vc_nodes,
                                                 const int *local_csr_rowp,
                                                 const int *local_csr_cols, const T *local_csr_vals,
                                                 const int *root_csr_rowp, int *root_csr_cols,
                                                 T *root_csr_vals) {
    int scalar_row = blockIdx.x * blockDim.x + threadIdx.x;
    int nscalar_rows = mb * block_dim;
    if (scalar_row >= nscalar_rows) return;

    int brow = scalar_row / block_dim;
    int bi = scalar_row % block_dim;

    int root_brow = Vc_nodes[brow];
    int root_scalar_row = root_brow * block_dim + bi;

    int src0 = local_csr_rowp[scalar_row];
    int src1 = local_csr_rowp[scalar_row + 1];
    int dst0 = root_csr_rowp[root_scalar_row];

    for (int k = src0; k < src1; k++) {
        int dst = dst0 + (k - src0);
        root_csr_cols[dst] = local_csr_cols[k];
        root_csr_vals[dst] = local_csr_vals[k];
    }
}

template <typename T>
__global__ void pack_rhs_to_root_kernel(int mb, int block_dim, const int *Vc_nodes,
                                        const T *rhs_local, T *rhs_root) {
    int scalar_row = blockIdx.x * blockDim.x + threadIdx.x;
    int nscalar_rows = mb * block_dim;
    if (scalar_row >= nscalar_rows) return;

    int brow = scalar_row / block_dim;
    int bi = scalar_row % block_dim;

    int root_brow = Vc_nodes[brow];
    rhs_root[root_brow * block_dim + bi] = rhs_local[scalar_row];
}

template <typename T>
__global__ void unpack_sol_from_root_kernel(int mb, int block_dim, const int *Vc_nodes,
                                            const T *sol_root, T *sol_local) {
    int scalar_row = blockIdx.x * blockDim.x + threadIdx.x;
    int nscalar_rows = mb * block_dim;
    if (scalar_row >= nscalar_rows) return;

    int brow = scalar_row / block_dim;
    int bi = scalar_row % block_dim;

    int root_brow = Vc_nodes[brow];
    sol_local[scalar_row] = sol_root[root_brow * block_dim + bi];
}

template <typename T>
class CudssMgBSRSolverV2 {
   public:
    CudssMgBSRSolverV2(MultiGPUContext *ctx_, int sgpu_Vc_nnodes_, int *Vc_nnodes_, int **Vc_nodes_,
                       int block_dim_, int **Svv_rowp_, int **Svv_cols_, int *Svv_nnzb_,
                       int *Svv_rows_, T **d_Svv_vals_)
        : ctx(ctx_),
          sgpu_Vc_nnodes(sgpu_Vc_nnodes_),
          Vc_nnodes(Vc_nnodes_),
          Vc_nodes(Vc_nodes_),
          block_dim(block_dim_),
          Svv_rowp(Svv_rowp_),
          Svv_cols(Svv_cols_),
          Svv_nnzb(Svv_nnzb_),
          Svv_rows(Svv_rows_),
          d_Svv_vals(d_Svv_vals_) {
        ngpus = ctx->ngpus;
        root = ctx->device(0);
        N = sgpu_Vc_nnodes * block_dim;

        devices.resize(ngpus);
        for (int g = 0; g < ngpus; g++) {
            devices[g] = ctx->device(g);
        }
    }

    void factor() {
        build_root_csr();
        init_cudss();
        execute(CUDSS_PHASE_ANALYSIS, analysis_ms);
        execute(CUDSS_PHASE_FACTORIZATION, factor_ms);
    }

    void solve(T **d_rhs_local, T **d_sol_local) {
        gather_rhs_to_root(d_rhs_local);

        if (B) {
            CHECK_CUDSS(cudssMatrixDestroy(B));
            B = nullptr;
        }
        if (X) {
            CHECK_CUDSS(cudssMatrixDestroy(X));
            X = nullptr;
        }

        CHECK_CUDA(cudaSetDevice(root));

        CHECK_CUDSS(cudssMatrixCreateDn(&B, N, 1, N, d_b_root, CudssMgType<T>::type,
                                        CUDSS_LAYOUT_COL_MAJOR));

        CHECK_CUDSS(cudssMatrixCreateDn(&X, N, 1, N, d_x_root, CudssMgType<T>::type,
                                        CUDSS_LAYOUT_COL_MAJOR));

        execute(CUDSS_PHASE_SOLVE, solve_ms);
        scatter_solution_from_root(d_sol_local);
    }

    void free() {
        CHECK_CUDA(cudaSetDevice(root));

        if (A_cudss) CHECK_CUDSS(cudssMatrixDestroy(A_cudss));
        if (B) CHECK_CUDSS(cudssMatrixDestroy(B));
        if (X) CHECK_CUDSS(cudssMatrixDestroy(X));
        if (data) CHECK_CUDSS(cudssDataDestroy(handle, data));
        if (config) CHECK_CUDSS(cudssConfigDestroy(config));
        if (handle) CHECK_CUDSS(cudssDestroy(handle));

        if (d_rowp_root) CHECK_CUDA(cudaFree(d_rowp_root));
        if (d_cols_root) CHECK_CUDA(cudaFree(d_cols_root));
        if (d_vals_root) CHECK_CUDA(cudaFree(d_vals_root));
        if (d_b_root) CHECK_CUDA(cudaFree(d_b_root));
        if (d_x_root) CHECK_CUDA(cudaFree(d_x_root));

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(ctx->device(g)));

            if (d_Vc_nodes[g]) CHECK_CUDA(cudaFree(d_Vc_nodes[g]));
            if (d_bsr_rowp[g]) CHECK_CUDA(cudaFree(d_bsr_rowp[g]));
            if (d_bsr_cols[g]) CHECK_CUDA(cudaFree(d_bsr_cols[g]));
            if (d_local_csr_rowp[g]) CHECK_CUDA(cudaFree(d_local_csr_rowp[g]));
            if (d_local_csr_cols[g]) CHECK_CUDA(cudaFree(d_local_csr_cols[g]));
            if (d_local_csr_vals[g]) CHECK_CUDA(cudaFree(d_local_csr_vals[g]));
        }

        A_cudss = nullptr;
        B = nullptr;
        X = nullptr;
        data = nullptr;
        config = nullptr;
        handle = nullptr;
    }

    float getAnalysisMs() const { return analysis_ms; }
    float getFactorMs() const { return factor_ms; }
    float getSolveMs() const { return solve_ms; }
    int getN() const { return N; }
    int getNnz() const { return nnz; }

   private:
    MultiGPUContext *ctx = nullptr;

    int sgpu_Vc_nnodes = 0;
    int *Vc_nnodes = nullptr;
    int **Vc_nodes = nullptr;

    int block_dim = 0;
    int ngpus = 0;
    int root = 0;
    int N = 0;
    int nnz = 0;

    int **Svv_rowp = nullptr;
    int **Svv_cols = nullptr;
    int *Svv_nnzb = nullptr;
    int *Svv_rows = nullptr;
    T **d_Svv_vals = nullptr;

    std::vector<int> devices;
    std::vector<std::vector<int>> h_local_csr_rowp;

    std::vector<int *> d_Vc_nodes;
    std::vector<int *> d_bsr_rowp;
    std::vector<int *> d_bsr_cols;
    std::vector<int *> d_local_csr_rowp;
    std::vector<int *> d_local_csr_cols;
    std::vector<T *> d_local_csr_vals;

    int *d_rowp_root = nullptr;
    int *d_cols_root = nullptr;
    T *d_vals_root = nullptr;
    T *d_b_root = nullptr;
    T *d_x_root = nullptr;

    cudssHandle_t handle = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t data = nullptr;
    cudssMatrix_t A_cudss = nullptr;
    cudssMatrix_t B = nullptr;
    cudssMatrix_t X = nullptr;

    float analysis_ms = 0.0f;
    float factor_ms = 0.0f;
    float solve_ms = 0.0f;

    void build_root_csr() {
        std::vector<int> h_row_counts(N, 0);
        h_local_csr_rowp.resize(ngpus);

        for (int g = 0; g < ngpus; g++) {
            int mb = Vc_nnodes[g];
            int local_N = mb * block_dim;

            h_local_csr_rowp[g].resize(local_N + 1);
            h_local_csr_rowp[g][0] = 0;

            for (int brow = 0; brow < mb; brow++) {
                int bnnz = Svv_rowp[g][brow + 1] - Svv_rowp[g][brow];
                int root_brow = Vc_nodes[g][brow];

                for (int bi = 0; bi < block_dim; bi++) {
                    int local_scalar_row = brow * block_dim + bi;
                    int root_scalar_row = root_brow * block_dim + bi;
                    int scalar_nnz = bnnz * block_dim;

                    h_local_csr_rowp[g][local_scalar_row + 1] =
                        h_local_csr_rowp[g][local_scalar_row] + scalar_nnz;

                    h_row_counts[root_scalar_row] += scalar_nnz;
                }
            }
        }

        std::vector<int> h_rowp_root(N + 1);
        h_rowp_root[0] = 0;

        for (int i = 0; i < N; i++) {
            h_rowp_root[i + 1] = h_rowp_root[i] + h_row_counts[i];
        }

        nnz = h_rowp_root[N];

        CHECK_CUDA(cudaSetDevice(root));

        CHECK_CUDA(cudaMalloc((void **)&d_rowp_root, (N + 1) * sizeof(int)));
        CHECK_CUDA(cudaMalloc((void **)&d_cols_root, nnz * sizeof(int)));
        CHECK_CUDA(cudaMalloc((void **)&d_vals_root, nnz * sizeof(T)));
        CHECK_CUDA(cudaMalloc((void **)&d_b_root, N * sizeof(T)));
        CHECK_CUDA(cudaMalloc((void **)&d_x_root, N * sizeof(T)));

        CHECK_CUDA(cudaMemcpy(d_rowp_root, h_rowp_root.data(), (N + 1) * sizeof(int),
                              cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaMemset(d_cols_root, 0, nnz * sizeof(int)));
        CHECK_CUDA(cudaMemset(d_vals_root, 0, nnz * sizeof(T)));
        CHECK_CUDA(cudaMemset(d_b_root, 0, N * sizeof(T)));
        CHECK_CUDA(cudaMemset(d_x_root, 0, N * sizeof(T)));

        d_Vc_nodes.resize(ngpus, nullptr);
        d_bsr_rowp.resize(ngpus, nullptr);
        d_bsr_cols.resize(ngpus, nullptr);
        d_local_csr_rowp.resize(ngpus, nullptr);
        d_local_csr_cols.resize(ngpus, nullptr);
        d_local_csr_vals.resize(ngpus, nullptr);

        std::vector<std::thread> threads;
        threads.reserve(ngpus);

        for (int g = 0; g < ngpus; g++) {
            threads.emplace_back([&, g]() { build_and_scatter_gpu_csr(g); });
        }

        for (auto &t : threads) {
            t.join();
        }

        CHECK_CUDA(cudaSetDevice(root));
        CHECK_CUDA(cudaDeviceSynchronize());

        printf("CudssMgBSRSolver root CSR: N=%d nnz=%d root=%d ngpus=%d\n", N, nnz, root, ngpus);
    }

    void build_and_scatter_gpu_csr(int g) {
        int dev = ctx->device(g);
        int mb = Vc_nnodes[g];
        int local_N = mb * block_dim;
        int local_nnz = h_local_csr_rowp[g][local_N];

        CHECK_CUDA(cudaSetDevice(dev));

        CHECK_CUDA(cudaMalloc((void **)&d_Vc_nodes[g], mb * sizeof(int)));
        CHECK_CUDA(cudaMalloc((void **)&d_bsr_rowp[g], (mb + 1) * sizeof(int)));
        CHECK_CUDA(cudaMalloc((void **)&d_bsr_cols[g], Svv_nnzb[g] * sizeof(int)));
        CHECK_CUDA(cudaMalloc((void **)&d_local_csr_rowp[g], (local_N + 1) * sizeof(int)));
        CHECK_CUDA(cudaMalloc((void **)&d_local_csr_cols[g], local_nnz * sizeof(int)));
        CHECK_CUDA(cudaMalloc((void **)&d_local_csr_vals[g], local_nnz * sizeof(T)));

        CHECK_CUDA(cudaMemcpyAsync(d_Vc_nodes[g], Vc_nodes[g], mb * sizeof(int),
                                   cudaMemcpyHostToDevice, ctx->streams[g]));

        CHECK_CUDA(cudaMemcpyAsync(d_bsr_rowp[g], Svv_rowp[g], (mb + 1) * sizeof(int),
                                   cudaMemcpyHostToDevice, ctx->streams[g]));

        CHECK_CUDA(cudaMemcpyAsync(d_bsr_cols[g], Svv_cols[g], Svv_nnzb[g] * sizeof(int),
                                   cudaMemcpyHostToDevice, ctx->streams[g]));

        CHECK_CUDA(cudaMemcpyAsync(d_local_csr_rowp[g], h_local_csr_rowp[g].data(),
                                   (local_N + 1) * sizeof(int), cudaMemcpyHostToDevice,
                                   ctx->streams[g]));

        int block = 256;
        int grid = (local_N + block - 1) / block;

        bsr_to_local_csr_kernel<T><<<grid, block, 0, ctx->streams[g]>>>(
            mb, block_dim, d_bsr_rowp[g], d_bsr_cols[g], d_Svv_vals[g], d_local_csr_rowp[g],
            d_local_csr_cols[g], d_local_csr_vals[g]);

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaStreamSynchronize(ctx->streams[g]));

        if (dev == root) {
            scatter_local_csr_to_root_kernel<T><<<grid, block, 0, ctx->streams[g]>>>(
                mb, block_dim, d_Vc_nodes[g], d_local_csr_rowp[g], d_local_csr_cols[g],
                d_local_csr_vals[g], d_rowp_root, d_cols_root, d_vals_root);

            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaStreamSynchronize(ctx->streams[g]));
        } else {
            int *d_rows_root = nullptr;
            int *d_lrowp_root = nullptr;
            int *d_lcols_root = nullptr;
            T *d_lvals_root = nullptr;

            CHECK_CUDA(cudaSetDevice(root));

            CHECK_CUDA(cudaMalloc((void **)&d_rows_root, mb * sizeof(int)));
            CHECK_CUDA(cudaMalloc((void **)&d_lrowp_root, (local_N + 1) * sizeof(int)));
            CHECK_CUDA(cudaMalloc((void **)&d_lcols_root, local_nnz * sizeof(int)));
            CHECK_CUDA(cudaMalloc((void **)&d_lvals_root, local_nnz * sizeof(T)));

            CHECK_CUDA(cudaMemcpyPeer(d_rows_root, root, d_Vc_nodes[g], dev, mb * sizeof(int)));

            CHECK_CUDA(cudaMemcpyPeer(d_lrowp_root, root, d_local_csr_rowp[g], dev,
                                      (local_N + 1) * sizeof(int)));

            CHECK_CUDA(cudaMemcpyPeer(d_lcols_root, root, d_local_csr_cols[g], dev,
                                      local_nnz * sizeof(int)));

            CHECK_CUDA(cudaMemcpyPeer(d_lvals_root, root, d_local_csr_vals[g], dev,
                                      local_nnz * sizeof(T)));

            scatter_local_csr_to_root_kernel<T>
                <<<grid, block>>>(mb, block_dim, d_rows_root, d_lrowp_root, d_lcols_root,
                                  d_lvals_root, d_rowp_root, d_cols_root, d_vals_root);

            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());

            CHECK_CUDA(cudaFree(d_rows_root));
            CHECK_CUDA(cudaFree(d_lrowp_root));
            CHECK_CUDA(cudaFree(d_lcols_root));
            CHECK_CUDA(cudaFree(d_lvals_root));
        }

        printf("GPU %d Svv BSR->CSR: mb=%d local_N=%d nnzb=%d local_nnz=%d\n", dev, mb, local_N,
               Svv_nnzb[g], local_nnz);
    }

    void init_cudss() {
        CHECK_CUDA(cudaSetDevice(root));

        CHECK_CUDSS(cudssCreateMg(&handle, ngpus, devices.data()));
        CHECK_CUDSS(cudssConfigCreate(&config));
        CHECK_CUDSS(cudssDataCreate(handle, &data));

        CHECK_CUDSS(cudssMatrixCreateCsr(&A_cudss, N, N, nnz, d_rowp_root, nullptr, d_cols_root,
                                         d_vals_root, CUDA_R_32I, CudssMgType<T>::type,
                                         CUDSS_MTYPE_SPD, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));

        CHECK_CUDSS(cudssMatrixCreateDn(&B, N, 1, N, d_b_root, CudssMgType<T>::type,
                                        CUDSS_LAYOUT_COL_MAJOR));

        CHECK_CUDSS(cudssMatrixCreateDn(&X, N, 1, N, d_x_root, CudssMgType<T>::type,
                                        CUDSS_LAYOUT_COL_MAJOR));
    }

    void execute(cudssPhase_t phase, float &ms_out) {
        CHECK_CUDA(cudaSetDevice(root));

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(start, 0));

        CHECK_CUDSS(cudssExecute(handle, phase, config, data, A_cudss, X, B));

        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&ms_out, start, stop));

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    void gather_rhs_to_root(T **d_rhs_local) {
        CHECK_CUDA(cudaSetDevice(root));
        CHECK_CUDA(cudaMemset(d_b_root, 0, N * sizeof(T)));

        std::vector<std::thread> threads;
        threads.reserve(ngpus);

        for (int g = 0; g < ngpus; g++) {
            threads.emplace_back([&, g]() {
                int dev = ctx->device(g);
                int mb = Vc_nnodes[g];
                int local_N = mb * block_dim;

                int block = 256;
                int grid = (local_N + block - 1) / block;

                if (dev == root) {
                    CHECK_CUDA(cudaSetDevice(root));

                    pack_rhs_to_root_kernel<T><<<grid, block, 0, ctx->streams[g]>>>(
                        mb, block_dim, d_Vc_nodes[g], d_rhs_local[g], d_b_root);

                    CHECK_CUDA(cudaGetLastError());
                    CHECK_CUDA(cudaStreamSynchronize(ctx->streams[g]));
                } else {
                    T *d_rhs_root_tmp = nullptr;
                    int *d_rows_root = nullptr;

                    CHECK_CUDA(cudaSetDevice(root));

                    CHECK_CUDA(cudaMalloc((void **)&d_rhs_root_tmp, local_N * sizeof(T)));
                    CHECK_CUDA(cudaMalloc((void **)&d_rows_root, mb * sizeof(int)));

                    CHECK_CUDA(cudaMemcpyPeer(d_rhs_root_tmp, root, d_rhs_local[g], dev,
                                              local_N * sizeof(T)));

                    CHECK_CUDA(
                        cudaMemcpyPeer(d_rows_root, root, d_Vc_nodes[g], dev, mb * sizeof(int)));

                    pack_rhs_to_root_kernel<T>
                        <<<grid, block>>>(mb, block_dim, d_rows_root, d_rhs_root_tmp, d_b_root);

                    CHECK_CUDA(cudaGetLastError());
                    CHECK_CUDA(cudaDeviceSynchronize());

                    CHECK_CUDA(cudaFree(d_rhs_root_tmp));
                    CHECK_CUDA(cudaFree(d_rows_root));
                }
            });
        }

        for (auto &t : threads) {
            t.join();
        }

        CHECK_CUDA(cudaSetDevice(root));
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    void scatter_solution_from_root(T **d_sol_local) {
        std::vector<std::thread> threads;
        threads.reserve(ngpus);

        for (int g = 0; g < ngpus; g++) {
            threads.emplace_back([&, g]() {
                int dev = ctx->device(g);
                int mb = Vc_nnodes[g];
                int local_N = mb * block_dim;

                int block = 256;
                int grid = (local_N + block - 1) / block;

                if (dev == root) {
                    CHECK_CUDA(cudaSetDevice(root));

                    unpack_sol_from_root_kernel<T><<<grid, block, 0, ctx->streams[g]>>>(
                        mb, block_dim, d_Vc_nodes[g], d_x_root, d_sol_local[g]);

                    CHECK_CUDA(cudaGetLastError());
                    CHECK_CUDA(cudaStreamSynchronize(ctx->streams[g]));
                } else {
                    T *d_x_dev = nullptr;

                    CHECK_CUDA(cudaSetDevice(dev));
                    CHECK_CUDA(cudaMalloc((void **)&d_x_dev, N * sizeof(T)));

                    CHECK_CUDA(cudaMemcpyPeer(d_x_dev, dev, d_x_root, root, N * sizeof(T)));

                    unpack_sol_from_root_kernel<T><<<grid, block, 0, ctx->streams[g]>>>(
                        mb, block_dim, d_Vc_nodes[g], d_x_dev, d_sol_local[g]);

                    CHECK_CUDA(cudaGetLastError());
                    CHECK_CUDA(cudaStreamSynchronize(ctx->streams[g]));

                    CHECK_CUDA(cudaFree(d_x_dev));
                }
            });
        }

        for (auto &t : threads) {
            t.join();
        }

        ctx->sync();
    }
};