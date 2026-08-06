#pragma once
#include <cuda_runtime.h>
#include <cudss.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#include "cuda_utils.h"
#include "utils/multigpu_context.h"

#define CHECK_CUDSS(x)                                                       \
    do {                                                                     \
        cudssStatus_t s = (x);                                               \
        if (s != CUDSS_STATUS_SUCCESS) {                                     \
            printf("cuDSS error %d at %s:%d\n", (int)s, __FILE__, __LINE__); \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

template <typename T>
struct CudssType;

template <>
struct CudssType<double> {
    static constexpr cudaDataType_t type = CUDA_R_64F;
};

template <>
struct CudssType<float> {
    static constexpr cudaDataType_t type = CUDA_R_32F;
};

template <typename T>
__global__ void bsr_to_csr_values_cols_kernel(int n_block_rows, int block_dim, const int *bsr_rowp,
                                              const int *bsr_cols, const T *bsr_vals,
                                              const int *csr_rowp, int *csr_cols, T *csr_vals) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int nrows = n_block_rows * block_dim;
    if (row >= nrows) return;

    int brow = row / block_dim;
    int bi = row % block_dim;

    int p = csr_rowp[row];

    for (int jp = bsr_rowp[brow]; jp < bsr_rowp[brow + 1]; jp++) {
        int bcol = bsr_cols[jp];

        for (int bj = 0; bj < block_dim; bj++) {
            csr_cols[p] = bcol * block_dim + bj;
            csr_vals[p] = bsr_vals[jp * block_dim * block_dim + bi * block_dim + bj];
            p++;
        }
    }
}

template <typename T>
class CudssSubdomainBsrSolve {
   public:
    struct GpuSolve {
        int logical_gpu = -1;
        int device = -1;

        int n_block_rows = 0;
        int block_dim = 0;
        int N = 0;

        int nnzb = 0;
        int nnz = 0;

        int *d_csr_rowp = nullptr;
        int *d_csr_cols = nullptr;
        T *d_csr_vals = nullptr;

        T *d_rhs = nullptr;
        T *d_sol = nullptr;

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

    CudssSubdomainBsrSolve(MultiGPUContext *ctx_, int *n_block_rows_, int block_dim_,
                           int **h_bsr_rowp_, int **h_bsr_cols_, int *nnzb_, T **d_bsr_vals_)
        : ctx(ctx_),
          n_block_rows(n_block_rows_),
          block_dim(block_dim_),
          h_bsr_rowp(h_bsr_rowp_),
          h_bsr_cols(h_bsr_cols_),
          nnzb(nnzb_),
          d_bsr_vals(d_bsr_vals_) {
        ngpu = ctx->ngpus;
        gpu.resize(ngpu);

        for (int g = 0; g < ngpu; g++) {
            GpuSolve &gs = gpu[g];

            gs.logical_gpu = g;
            gs.device = ctx->device(g);

            gs.n_block_rows = n_block_rows[g];
            gs.block_dim = block_dim;
            gs.N = gs.n_block_rows * block_dim;

            gs.nnzb = nnzb[g];
            gs.nnz = gs.nnzb * block_dim * block_dim;
        }
    }

    void factor() {
        assemble_all();
        init_all();

        run_phase(CUDSS_PHASE_ANALYSIS);
        run_phase(CUDSS_PHASE_FACTORIZATION);
    }

    void solve(T **d_rhs_gpu, T **d_sol_gpu) {
        copy_rhs(d_rhs_gpu);
        update_dn_matrices();

        run_phase(CUDSS_PHASE_SOLVE);

        copy_sol(d_sol_gpu);
    }

    void free() {
        for (auto &gs : gpu) {
            CHECK_CUDA(cudaSetDevice(gs.device));

            if (gs.A) CHECK_CUDSS(cudssMatrixDestroy(gs.A));
            if (gs.B) CHECK_CUDSS(cudssMatrixDestroy(gs.B));
            if (gs.X) CHECK_CUDSS(cudssMatrixDestroy(gs.X));

            if (gs.data) CHECK_CUDSS(cudssDataDestroy(gs.handle, gs.data));
            if (gs.config) CHECK_CUDSS(cudssConfigDestroy(gs.config));
            if (gs.handle) CHECK_CUDSS(cudssDestroy(gs.handle));

            if (gs.d_csr_rowp) CHECK_CUDA(cudaFree(gs.d_csr_rowp));
            if (gs.d_csr_cols) CHECK_CUDA(cudaFree(gs.d_csr_cols));
            if (gs.d_csr_vals) CHECK_CUDA(cudaFree(gs.d_csr_vals));

            if (gs.d_rhs) CHECK_CUDA(cudaFree(gs.d_rhs));
            if (gs.d_sol) CHECK_CUDA(cudaFree(gs.d_sol));

            gs = GpuSolve{};
        }
    }

    int getNumGpus() const { return ngpu; }

    int getLocalN(int g) const { return gpu[g].N; }

    int getLocalNnz(int g) const { return gpu[g].nnz; }

    float getAnalysisMs(int g) const { return gpu[g].analysis_ms; }

    float getFactorMs(int g) const { return gpu[g].factor_ms; }

    float getSolveMs(int g) const { return gpu[g].solve_ms; }

    T *getSolutionPtr(int g) const { return gpu[g].d_sol; }

    T *getRhsPtr(int g) const { return gpu[g].d_rhs; }

    int *getCsrRowpPtr(int g) const { return gpu[g].d_csr_rowp; }

    int *getCsrColsPtr(int g) const { return gpu[g].d_csr_cols; }

    T *getCsrValsPtr(int g) const { return gpu[g].d_csr_vals; }

   private:
    MultiGPUContext *ctx = nullptr;

    int *n_block_rows = nullptr;
    int block_dim = 0;
    int ngpu = 0;

    int **h_bsr_rowp = nullptr;
    int **h_bsr_cols = nullptr;
    int *nnzb = nullptr;
    T **d_bsr_vals = nullptr;

    std::vector<GpuSolve> gpu;

    void assemble_all() {
        std::vector<std::thread> threads;
        threads.reserve(ngpu);

        for (int g = 0; g < ngpu; g++) {
            threads.emplace_back([&, g]() { assemble_gpu(gpu[g]); });
        }

        for (auto &t : threads) {
            t.join();
        }
    }

    void assemble_gpu(GpuSolve &gs) {
        CHECK_CUDA(cudaSetDevice(gs.device));

        std::vector<int> h_csr_rowp(gs.N + 1);
        h_csr_rowp[0] = 0;

        for (int br = 0; br < gs.n_block_rows; br++) {
            int bnnz_row = h_bsr_rowp[gs.logical_gpu][br + 1] - h_bsr_rowp[gs.logical_gpu][br];

            for (int bi = 0; bi < block_dim; bi++) {
                int row = br * block_dim + bi;
                h_csr_rowp[row + 1] = h_csr_rowp[row] + bnnz_row * block_dim;
            }
        }

        int nnz_from_rowp = h_csr_rowp[gs.N];

        if (nnz_from_rowp != gs.nnz) {
            printf("WARNING GPU %d: nnz from rowp = %d but nnzb*bdim^2 = %d\n", gs.device,
                   nnz_from_rowp, gs.nnz);
            gs.nnz = nnz_from_rowp;
        }

        CHECK_CUDA(cudaMalloc((void **)&gs.d_csr_rowp, (gs.N + 1) * sizeof(int)));

        CHECK_CUDA(cudaMalloc((void **)&gs.d_csr_cols, gs.nnz * sizeof(int)));

        CHECK_CUDA(cudaMalloc((void **)&gs.d_csr_vals, gs.nnz * sizeof(T)));

        CHECK_CUDA(cudaMalloc((void **)&gs.d_rhs, gs.N * sizeof(T)));

        CHECK_CUDA(cudaMalloc((void **)&gs.d_sol, gs.N * sizeof(T)));

        CHECK_CUDA(cudaMemcpyAsync(gs.d_csr_rowp, h_csr_rowp.data(), (gs.N + 1) * sizeof(int),
                                   cudaMemcpyHostToDevice, ctx->streams[gs.logical_gpu]));

        CHECK_CUDA(cudaMemsetAsync(gs.d_rhs, 0, gs.N * sizeof(T), ctx->streams[gs.logical_gpu]));

        CHECK_CUDA(cudaMemsetAsync(gs.d_sol, 0, gs.N * sizeof(T), ctx->streams[gs.logical_gpu]));

        int *d_bsr_rowp = nullptr;
        int *d_bsr_cols_dev = nullptr;

        CHECK_CUDA(cudaMalloc((void **)&d_bsr_rowp, (gs.n_block_rows + 1) * sizeof(int)));

        CHECK_CUDA(cudaMalloc((void **)&d_bsr_cols_dev, gs.nnzb * sizeof(int)));

        CHECK_CUDA(cudaMemcpyAsync(d_bsr_rowp, h_bsr_rowp[gs.logical_gpu],
                                   (gs.n_block_rows + 1) * sizeof(int), cudaMemcpyHostToDevice,
                                   ctx->streams[gs.logical_gpu]));

        CHECK_CUDA(cudaMemcpyAsync(d_bsr_cols_dev, h_bsr_cols[gs.logical_gpu],
                                   gs.nnzb * sizeof(int), cudaMemcpyHostToDevice,
                                   ctx->streams[gs.logical_gpu]));

        int block = 256;
        int grid = (gs.N + block - 1) / block;

        bsr_to_csr_values_cols_kernel<T><<<grid, block, 0, ctx->streams[gs.logical_gpu]>>>(
            gs.n_block_rows, block_dim, d_bsr_rowp, d_bsr_cols_dev, d_bsr_vals[gs.logical_gpu],
            gs.d_csr_rowp, gs.d_csr_cols, gs.d_csr_vals);

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaStreamSynchronize(ctx->streams[gs.logical_gpu]));

        CHECK_CUDA(cudaFree(d_bsr_rowp));
        CHECK_CUDA(cudaFree(d_bsr_cols_dev));

        // printf("GPU %d local BSR->CSR: n_block_rows=%d, N=%d, nnzb=%d, nnz=%d\n", gs.device,
        //        gs.n_block_rows, gs.N, gs.nnzb, gs.nnz);
    }

    void init_all() {
        for (auto &gs : gpu) {
            CHECK_CUDA(cudaSetDevice(gs.device));

            CHECK_CUDSS(cudssCreate(&gs.handle));
            CHECK_CUDSS(cudssConfigCreate(&gs.config));
            CHECK_CUDSS(cudssDataCreate(gs.handle, &gs.data));
            CHECK_CUDSS(cudssSetStream(gs.handle, ctx->streams[gs.logical_gpu]));

            CHECK_CUDSS(cudssMatrixCreateCsr(&gs.A, gs.N, gs.N, gs.nnz, gs.d_csr_rowp, nullptr,
                                             gs.d_csr_cols, gs.d_csr_vals, CUDA_R_32I,
                                             CudssType<T>::type, CUDSS_MTYPE_SPD, CUDSS_MVIEW_FULL,
                                             CUDSS_BASE_ZERO));

            CHECK_CUDSS(cudssMatrixCreateDn(&gs.B, gs.N, 1, gs.N, gs.d_rhs, CudssType<T>::type,
                                            CUDSS_LAYOUT_COL_MAJOR));

            CHECK_CUDSS(cudssMatrixCreateDn(&gs.X, gs.N, 1, gs.N, gs.d_sol, CudssType<T>::type,
                                            CUDSS_LAYOUT_COL_MAJOR));
        }
    }

    void update_dn_matrices() {
        for (auto &gs : gpu) {
            CHECK_CUDA(cudaSetDevice(gs.device));

            if (gs.B) {
                CHECK_CUDSS(cudssMatrixDestroy(gs.B));
                gs.B = nullptr;
            }

            if (gs.X) {
                CHECK_CUDSS(cudssMatrixDestroy(gs.X));
                gs.X = nullptr;
            }

            CHECK_CUDSS(cudssMatrixCreateDn(&gs.B, gs.N, 1, gs.N, gs.d_rhs, CudssType<T>::type,
                                            CUDSS_LAYOUT_COL_MAJOR));

            CHECK_CUDSS(cudssMatrixCreateDn(&gs.X, gs.N, 1, gs.N, gs.d_sol, CudssType<T>::type,
                                            CUDSS_LAYOUT_COL_MAJOR));
        }
    }

    void run_phase(cudssPhase_t phase) {
        std::vector<std::thread> threads;
        threads.reserve(ngpu);

        for (int g = 0; g < ngpu; g++) {
            threads.emplace_back([&, g]() {
                GpuSolve &gs = gpu[g];

                CHECK_CUDA(cudaSetDevice(gs.device));

                cudaEvent_t start;
                cudaEvent_t stop;

                CHECK_CUDA(cudaEventCreate(&start));
                CHECK_CUDA(cudaEventCreate(&stop));

                CHECK_CUDA(cudaEventRecord(start, ctx->streams[gs.logical_gpu]));

                CHECK_CUDSS(cudssExecute(gs.handle, phase, gs.config, gs.data, gs.A, gs.X, gs.B));

                CHECK_CUDA(cudaEventRecord(stop, ctx->streams[gs.logical_gpu]));
                CHECK_CUDA(cudaEventSynchronize(stop));

                float ms = 0.0f;
                CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

                if (phase == CUDSS_PHASE_ANALYSIS) {
                    gs.analysis_ms = ms;
                } else if (phase == CUDSS_PHASE_FACTORIZATION) {
                    gs.factor_ms = ms;
                } else if (phase == CUDSS_PHASE_SOLVE) {
                    gs.solve_ms = ms;
                }

                CHECK_CUDA(cudaEventDestroy(start));
                CHECK_CUDA(cudaEventDestroy(stop));
            });
        }

        for (auto &t : threads) {
            t.join();
        }
    }

    void copy_rhs(T **d_rhs_gpu) {
        std::vector<std::thread> threads;
        threads.reserve(ngpu);

        for (int g = 0; g < ngpu; g++) {
            threads.emplace_back([&, g]() {
                GpuSolve &gs = gpu[g];

                CHECK_CUDA(cudaSetDevice(gs.device));

                CHECK_CUDA(cudaMemcpyAsync(gs.d_rhs, d_rhs_gpu[g], gs.N * sizeof(T),
                                           cudaMemcpyDeviceToDevice, ctx->streams[gs.logical_gpu]));

                CHECK_CUDA(cudaStreamSynchronize(ctx->streams[gs.logical_gpu]));
            });
        }

        for (auto &t : threads) {
            t.join();
        }
    }

    void copy_sol(T **d_sol_gpu) {
        std::vector<std::thread> threads;
        threads.reserve(ngpu);

        for (int g = 0; g < ngpu; g++) {
            threads.emplace_back([&, g]() {
                GpuSolve &gs = gpu[g];

                CHECK_CUDA(cudaSetDevice(gs.device));

                CHECK_CUDA(cudaMemcpyAsync(d_sol_gpu[g], gs.d_sol, gs.N * sizeof(T),
                                           cudaMemcpyDeviceToDevice, ctx->streams[gs.logical_gpu]));

                CHECK_CUDA(cudaStreamSynchronize(ctx->streams[gs.logical_gpu]));
            });
        }

        for (auto &t : threads) {
            t.join();
        }
    }
};