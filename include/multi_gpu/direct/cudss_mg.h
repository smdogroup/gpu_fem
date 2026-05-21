// cudss_mg_gpu_bsrmat_root_gather.cuh
#pragma once
#include <cuda_runtime.h>
#include <cudss.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "gpubsrmat.h"
#include "gpuvec.h"
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
struct CudssCudaType;
template <>
struct CudssCudaType<double> {
    static constexpr cudaDataType_t value = CUDA_R_64F;
};
template <>
struct CudssCudaType<float> {
    static constexpr cudaDataType_t value = CUDA_R_32F;
};

template <typename T>
__global__ void bsr_local_to_global_csr_kernel(int mb, int block_dim, const int *bsr_rowp,
                                               const int *bsr_cols, const T *bsr_vals,
                                               const int *local_nodes, const int *csr_rowp_global,
                                               int row_start_global, int *csr_cols_global,
                                               T *csr_vals_global) {
    int local_scalar_row = blockIdx.x * blockDim.x + threadIdx.x;
    int nscalar_rows = mb * block_dim;
    if (local_scalar_row >= nscalar_rows) return;

    int brow = local_scalar_row / block_dim;
    int bi = local_scalar_row % block_dim;

    int global_node_row = local_nodes[brow];
    int global_scalar_row = global_node_row * block_dim + bi;

    int p = csr_rowp_global[global_scalar_row] - csr_rowp_global[row_start_global];

    for (int jp = bsr_rowp[brow]; jp < bsr_rowp[brow + 1]; jp++) {
        int bcol = bsr_cols[jp];
        int global_node_col = local_nodes[bcol];

        for (int bj = 0; bj < block_dim; bj++) {
            csr_cols_global[p] = global_node_col * block_dim + bj;
            csr_vals_global[p] = bsr_vals[jp * block_dim * block_dim + bi * block_dim + bj];
            p++;
        }
    }
}

template <typename T>
__global__ void pack_rhs_local_to_global_order_kernel(int mb, int block_dim, const int *local_nodes,
                                                      const T *x_local, T *x_global_local_rows) {
    int local_scalar_row = blockIdx.x * blockDim.x + threadIdx.x;
    int nscalar_rows = mb * block_dim;
    if (local_scalar_row >= nscalar_rows) return;

    int node = local_scalar_row / block_dim;
    int d = local_scalar_row % block_dim;
    int global_node = local_nodes[node];

    x_global_local_rows[global_node * block_dim + d] = x_local[local_scalar_row];
}

template <typename T>
__global__ void unpack_solution_global_to_local_kernel(int mb, int block_dim,
                                                       const int *local_nodes, const T *x_global,
                                                       T *x_local) {
    int local_scalar_row = blockIdx.x * blockDim.x + threadIdx.x;
    int nscalar_rows = mb * block_dim;
    if (local_scalar_row >= nscalar_rows) return;

    int node = local_scalar_row / block_dim;
    int d = local_scalar_row % block_dim;
    int global_node = local_nodes[node];

    x_local[local_scalar_row] = x_global[global_node * block_dim + d];
}

template <typename T, class Partitioner>
class CudssMgBSRSolver {
    // accepts a BSR matrix on multi-GPU
    // then uses CuDSS multi-GPU schur complement solver by gathering from distributed GPU to root
    // GPU and BSR to CSR conversion (CuDSS is very fast solver)

   public:
    CudssMgBSRSolver(MultiGPUContext *ctx_, const Partitioner *part_, GPUbsrmat<T, Partitioner> *A_)
        : ctx(ctx_), part(part_), A(A_) {
        ngpus = ctx->ngpus;
        block_dim = A->getBlockDim();
        N = part->num_nodes * block_dim;
        root = ctx->device(0);
        devices.resize(ngpus);
        for (int g = 0; g < ngpus; g++) devices[g] = ctx->device(g);
    }

    void factor() {
        build_root_csr();
        init_cudss();
        execute(CUDSS_PHASE_ANALYSIS, analysis_ms);
        execute(CUDSS_PHASE_FACTORIZATION, factor_ms);
    }

    void solve(GPUvec<T, Partitioner> *b, GPUvec<T, Partitioner> *x) {
        gather_rhs_to_root(b);
        if (B) CHECK_CUDSS(cudssMatrixDestroy(B));
        if (X) CHECK_CUDSS(cudssMatrixDestroy(X));

        CHECK_CUDA(cudaSetDevice(root));
        CHECK_CUDSS(cudssMatrixCreateDn(&B, N, 1, N, d_b_root, CudssCudaType<T>::value,
                                        CUDSS_LAYOUT_COL_MAJOR));
        CHECK_CUDSS(cudssMatrixCreateDn(&X, N, 1, N, d_x_root, CudssCudaType<T>::value,
                                        CUDSS_LAYOUT_COL_MAJOR));

        execute(CUDSS_PHASE_SOLVE, solve_ms);
        scatter_solution_from_root(x);
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

        for (int g = 0; g < (int)d_local_nodes.size(); g++) {
            CHECK_CUDA(cudaSetDevice(ctx->device(g)));
            if (d_local_nodes[g]) CHECK_CUDA(cudaFree(d_local_nodes[g]));
            if (d_tmp_cols[g]) CHECK_CUDA(cudaFree(d_tmp_cols[g]));
            if (d_tmp_vals[g]) CHECK_CUDA(cudaFree(d_tmp_vals[g]));
        }
    }

    float getAnalysisMs() const { return analysis_ms; }
    float getFactorMs() const { return factor_ms; }
    float getSolveMs() const { return solve_ms; }

   private:
    MultiGPUContext *ctx = nullptr;
    const Partitioner *part = nullptr;
    GPUbsrmat<T, Partitioner> *A = nullptr;

    int ngpus = 0;
    int block_dim = 0;
    int N = 0;
    int nnz = 0;
    int root = 0;
    std::vector<int> devices;

    int *d_rowp_root = nullptr;
    int *d_cols_root = nullptr;
    T *d_vals_root = nullptr;
    T *d_b_root = nullptr;
    T *d_x_root = nullptr;

    std::vector<int *> d_local_nodes;
    std::vector<int *> d_tmp_cols;
    std::vector<T *> d_tmp_vals;

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

        for (int g = 0; g < ngpus; g++) {
            int mb = A->getLocalNumBlockRows(g);
            int *h_rowp = A->getHostLocalRowp(g);
            int *local_nodes = part->h_local_nodes[g];

            for (int brow = 0; brow < mb; brow++) {
                int global_node = local_nodes[brow];
                int bnnz = h_rowp[brow + 1] - h_rowp[brow];
                for (int bi = 0; bi < block_dim; bi++) {
                    h_row_counts[global_node * block_dim + bi] += bnnz * block_dim;
                }
            }
        }

        std::vector<int> h_rowp_root(N + 1);
        h_rowp_root[0] = 0;
        for (int i = 0; i < N; i++) h_rowp_root[i + 1] = h_rowp_root[i] + h_row_counts[i];
        nnz = h_rowp_root[N];

        CHECK_CUDA(cudaSetDevice(root));
        CHECK_CUDA(cudaMalloc((void **)&d_rowp_root, (N + 1) * sizeof(int)));
        CHECK_CUDA(cudaMalloc((void **)&d_cols_root, nnz * sizeof(int)));
        CHECK_CUDA(cudaMalloc((void **)&d_vals_root, nnz * sizeof(T)));
        CHECK_CUDA(cudaMalloc((void **)&d_b_root, N * sizeof(T)));
        CHECK_CUDA(cudaMalloc((void **)&d_x_root, N * sizeof(T)));

        CHECK_CUDA(cudaMemcpy(d_rowp_root, h_rowp_root.data(), (N + 1) * sizeof(int),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_b_root, 0, N * sizeof(T)));
        CHECK_CUDA(cudaMemset(d_x_root, 0, N * sizeof(T)));

        d_local_nodes.resize(ngpus, nullptr);
        d_tmp_cols.resize(ngpus, nullptr);
        d_tmp_vals.resize(ngpus, nullptr);

        for (int g = 0; g < ngpus; g++) {
            int dev = ctx->device(g);
            int mb = A->getLocalNumBlockRows(g);
            int local_nnz_scalar = A->getLocalNumNonzeros(g);

            CHECK_CUDA(cudaSetDevice(dev));
            CHECK_CUDA(cudaMalloc((void **)&d_local_nodes[g], mb * sizeof(int)));
            CHECK_CUDA(cudaMemcpyAsync(d_local_nodes[g], part->h_local_nodes[g], mb * sizeof(int),
                                       cudaMemcpyHostToDevice, ctx->streams[g]));

            CHECK_CUDA(cudaMalloc((void **)&d_tmp_cols[g], local_nnz_scalar * sizeof(int)));
            CHECK_CUDA(cudaMalloc((void **)&d_tmp_vals[g], local_nnz_scalar * sizeof(T)));

            int block = 256;
            int grid = (mb * block_dim + block - 1) / block;

            bsr_local_to_global_csr_kernel<T><<<grid, block, 0, ctx->streams[g]>>>(
                mb, block_dim, A->getLocalRowp(g), A->getLocalCols(g), A->getLocalVals(g),
                d_local_nodes[g], d_rowp_root, 0, d_tmp_cols[g], d_tmp_vals[g]);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaStreamSynchronize(ctx->streams[g]));

            CHECK_CUDA(cudaMemcpyPeer(d_cols_root + local_scalar_nnz_offset(g, h_rowp_root), root,
                                      d_tmp_cols[g], dev, local_nnz_scalar * sizeof(int)));
            CHECK_CUDA(cudaMemcpyPeer(d_vals_root + local_scalar_nnz_offset(g, h_rowp_root), root,
                                      d_tmp_vals[g], dev, local_nnz_scalar * sizeof(T)));
        }

        CHECK_CUDA(cudaSetDevice(root));
        CHECK_CUDA(cudaDeviceSynchronize());

        printf("root gathered GPUbsrmat CSR: N=%d nnz=%d root=%d ngpus=%d\n", N, nnz, root, ngpus);
    }

    int local_scalar_nnz_offset(int g, const std::vector<int> &h_rowp_root) const {
        int min_node = part->h_local_nodes[g][0];
        for (int i = 1; i < part->local_nnodes[g]; i++) {
            min_node = std::min(min_node, part->h_local_nodes[g][i]);
        }
        return h_rowp_root[min_node * block_dim];
    }

    void init_cudss() {
        CHECK_CUDA(cudaSetDevice(root));

        CHECK_CUDSS(cudssCreateMg(&handle, ngpus, devices.data()));
        CHECK_CUDSS(cudssConfigCreate(&config));
        CHECK_CUDSS(cudssDataCreate(handle, &data));

        CHECK_CUDSS(cudssMatrixCreateCsr(&A_cudss, N, N, nnz, d_rowp_root, nullptr, d_cols_root,
                                         d_vals_root, CUDA_R_32I, CudssCudaType<T>::value,
                                         CUDSS_MTYPE_SPD, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));

        CHECK_CUDSS(cudssMatrixCreateDn(&B, N, 1, N, d_b_root, CudssCudaType<T>::value,
                                        CUDSS_LAYOUT_COL_MAJOR));

        CHECK_CUDSS(cudssMatrixCreateDn(&X, N, 1, N, d_x_root, CudssCudaType<T>::value,
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

    void gather_rhs_to_root(GPUvec<T, Partitioner> *b) {
        CHECK_CUDA(cudaSetDevice(root));
        CHECK_CUDA(cudaMemset(d_b_root, 0, N * sizeof(T)));

        for (int g = 0; g < ngpus; g++) {
            int dev = ctx->device(g);
            int mb = A->getLocalNumBlockRows(g);

            CHECK_CUDA(cudaSetDevice(dev));

            T *d_tmp_rhs = nullptr;
            CHECK_CUDA(cudaMalloc((void **)&d_tmp_rhs, N * sizeof(T)));
            CHECK_CUDA(cudaMemsetAsync(d_tmp_rhs, 0, N * sizeof(T), ctx->streams[g]));

            int block = 256;
            int grid = (mb * block_dim + block - 1) / block;

            pack_rhs_local_to_global_order_kernel<T><<<grid, block, 0, ctx->streams[g]>>>(
                mb, block_dim, d_local_nodes[g], b->getLocalPtr(g), d_tmp_rhs);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaStreamSynchronize(ctx->streams[g]));

            CHECK_CUDA(cudaMemcpyPeer(d_b_root, root, d_tmp_rhs, dev, N * sizeof(T)));
            CHECK_CUDA(cudaFree(d_tmp_rhs));
        }

        CHECK_CUDA(cudaSetDevice(root));
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    void scatter_solution_from_root(GPUvec<T, Partitioner> *x) {
        for (int g = 0; g < ngpus; g++) {
            int dev = ctx->device(g);
            int mb = A->getLocalNumBlockRows(g);

            CHECK_CUDA(cudaSetDevice(dev));

            T *d_x_copy = nullptr;
            CHECK_CUDA(cudaMalloc((void **)&d_x_copy, N * sizeof(T)));
            CHECK_CUDA(cudaMemcpyPeer(d_x_copy, dev, d_x_root, root, N * sizeof(T)));

            int block = 256;
            int grid = (mb * block_dim + block - 1) / block;

            unpack_solution_global_to_local_kernel<T><<<grid, block, 0, ctx->streams[g]>>>(
                mb, block_dim, d_local_nodes[g], d_x_copy, x->getLocalPtr(g));
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaStreamSynchronize(ctx->streams[g]));
            CHECK_CUDA(cudaFree(d_x_copy));
        }

        ctx->sync();
    }
};