#pragma once
#include <cstring>
#include <type_traits>
#include <vector>

#include "asw.cuh"
#include "cuda_utils.h"
#include "gpumat.h"
#include "gpuvec.h"

#pragma once
#include "cuda_utils.h"

template <typename T>
__global__ void k_setupBatchedPointers(int batch_size, int n, T *Adata, T *invAdata, T *Xdata,
                                       T *Ydata, T **Aarray, T **invAarray, T **Xarray,
                                       T **Yarray) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    Aarray[b] = &Adata[(size_t)b * n * n];
    invAarray[b] = &invAdata[(size_t)b * n * n];
    Xarray[b] = &Xdata[(size_t)b * n];
    Yarray[b] = &Ydata[(size_t)b * n];
}

template <typename T>
__global__ void k_copyMatValuesToBatchedContiguous(int n_batch_vals, int block_dim, int size,
                                                   const int *__restrict__ block_map,
                                                   const T *__restrict__ vals,
                                                   T *__restrict__ Adata) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_batch_vals) return;

    int block_dim2 = block_dim * block_dim;
    int size2 = size * size;
    int size4 = size2 * size2;
    int n = size2 * block_dim;

    int batch_block_ind = tid / block_dim2;
    int inner = tid % block_dim2;

    int batch = batch_block_ind / size4;
    int inner_block = batch_block_ind % size4;

    int i_node = inner_block % size2;
    int j_node = inner_block / size2;

    int p = inner / block_dim;
    int q = inner % block_dim;

    int jp = block_map[batch_block_ind];
    if (jp < 0) return;

    int row = i_node * block_dim + p;
    int col = j_node * block_dim + q;

    T *A = &Adata[(size_t)batch * n * n];
    A[row + col * n] = vals[(size_t)jp * block_dim2 + inner];
}

template <typename T>
__global__ void k_copyLocalRHSIntoBatched(int n_rhs_vals, int block_dim, int size,
                                          const int *__restrict__ local_node_map,
                                          const T *__restrict__ rhs_local,
                                          T **__restrict__ Xarray) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_rhs_vals) return;

    int size2 = size * size;

    int node_entry = tid / block_dim;
    int idof = tid % block_dim;

    int batch = node_entry / size2;
    int local_slot = node_entry % size2;
    int local_node = local_node_map[node_entry];

    T *x = Xarray[batch];
    x[local_slot * block_dim + idof] = rhs_local[local_node * block_dim + idof];
}

template <typename T>
__global__ void k_copyBatchedIntoOwnedSoln(int n_rhs_vals, int block_dim, int size,
                                           const int *__restrict__ owned_node_map,
                                           T **__restrict__ Yarray, T *__restrict__ soln_owned) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_rhs_vals) return;

    int size2 = size * size;

    int node_entry = tid / block_dim;
    int idof = tid % block_dim;

    int batch = node_entry / size2;
    int local_slot = node_entry % size2;
    int owned_node = owned_node_map[node_entry];

    if (owned_node < 0) return;

    const T *y = Yarray[batch];
    T val = y[local_slot * block_dim + idof];

    atomicAdd(&soln_owned[owned_node * block_dim + idof], val);
}

template <typename T, class Partitioner>
class MultiGPUElementASW {
   public:
    static constexpr int nodes_per_elem = 4;

    MultiGPUElementASW(MultiGPUContext *ctx_, const Partitioner *part_,
                       GPUbsrmat<T, Partitioner> *A_, T omega_ = 0.25, int iters_ = 1)
        : ctx(ctx_),
          part(part_),
          A(A_),
          cublasHandles(ctx_->cublasHandles),
          cusparseHandles(ctx_->cusparseHandles),
          streams(ctx_->streams),
          ngpus(part_->ngpus),
          block_dim(A_->getBlockDim()),
          block_dim2(block_dim * block_dim),
          omega(omega_),
          iters(iters_),
          debug(ctx_->debug) {
        static_assert(std::is_same<T, double>::value,
                      "This ASW implementation currently assumes double.");

        size = (int)sqrt(nodes_per_elem);
        size2 = nodes_per_elem;
        size4 = nodes_per_elem * nodes_per_elem;
        n = nodes_per_elem * block_dim;

        printf("ASW - allocate_arrays\n");
        allocate_arrays();
        printf("ASW - build_maps\n");
        build_maps();
        printf("ASW - allocate_batched_memory\n");
        allocate_batched_memory();
        printf("ASW - move_maps_to_device\n");
        move_maps_to_device();
        printf("ASW - done with constructor\n");

        temp = new GPUvec<T, Partitioner>(ctx, part, block_dim);
        defect = new GPUvec<T, Partitioner>(ctx, part, block_dim);

        ctx->sync();
    }

    ~MultiGPUElementASW() {
        ctx->sync();

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));

            if (d_block_inds[g]) cudaFree(d_block_inds[g]);
            if (d_rhs_local_map[g]) cudaFree(d_rhs_local_map[g]);
            if (d_rhs_owned_map[g]) cudaFree(d_rhs_owned_map[g]);

            if (d_Aarray[g]) cudaFree(d_Aarray[g]);
            if (d_invAarray[g]) cudaFree(d_invAarray[g]);
            if (d_Xarray[g]) cudaFree(d_Xarray[g]);
            if (d_Yarray[g]) cudaFree(d_Yarray[g]);

            if (d_Adata[g]) cudaFree(d_Adata[g]);
            if (d_invAdata[g]) cudaFree(d_invAdata[g]);
            if (d_Xdata[g]) cudaFree(d_Xdata[g]);
            if (d_Ydata[g]) cudaFree(d_Ydata[g]);

            if (d_PivotArray[g]) cudaFree(d_PivotArray[g]);
            if (d_InfoArray[g]) cudaFree(d_InfoArray[g]);

            delete[] h_block_inds[g];
            delete[] h_rhs_local_map[g];
            delete[] h_rhs_owned_map[g];
        }

        delete temp;
        delete defect;

        delete[] batch_size;
        delete[] n_batch_blocks;
        delete[] n_rhs_blocks;

        delete[] h_block_inds;
        delete[] h_rhs_local_map;
        delete[] h_rhs_owned_map;

        delete[] d_block_inds;
        delete[] d_rhs_local_map;
        delete[] d_rhs_owned_map;

        delete[] d_Aarray;
        delete[] d_invAarray;
        delete[] d_Xarray;
        delete[] d_Yarray;

        delete[] d_Adata;
        delete[] d_invAdata;
        delete[] d_Xdata;
        delete[] d_Ydata;

        delete[] d_PivotArray;
        delete[] d_InfoArray;
    }

    void factor() {
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            CHECK_CUBLAS(cublasSetStream(cublasHandles[g], streams[g]));

            CHECK_CUDA(cudaMemsetAsync(d_Adata[g], 0, (size_t)batch_size[g] * n * n * sizeof(T),
                                       streams[g]));

            int nvals = n_batch_blocks[g] * block_dim2;
            dim3 block(128);
            dim3 grid((nvals + block.x - 1) / block.x);

            k_copyMatValuesToBatchedContiguous<T><<<grid, block, 0, streams[g]>>>(
                nvals, block_dim, size, d_block_inds[g], A->getLocalVals(g), d_Adata[g]);

            CHECK_CUDA(cudaGetLastError());

            CHECK_CUBLAS(cublasDgetrfBatched(cublasHandles[g], n, d_Aarray[g], n, d_PivotArray[g],
                                             d_InfoArray[g], batch_size[g]));

            CHECK_CUBLAS(cublasDgetriBatched(cublasHandles[g], n, (const double **)d_Aarray[g], n,
                                             d_PivotArray[g], d_invAarray[g], n, d_InfoArray[g],
                                             batch_size[g]));
        }

        ctx->sync();
    }

    bool solve(GPUvec<T, Partitioner> *rhs, GPUvec<T, Partitioner> *soln, bool check_conv = false) {
        rhs->copyTo(defect);
        soln->zero();

        smoothDefect(defect, soln, iters);
        return false;
    }

    void smoothDefect(GPUvec<T, Partitioner> *def, GPUvec<T, Partitioner> *soln, int n_iters = -1) {
        if (n_iters < 0) n_iters = iters;

        for (int iter = 0; iter < n_iters; iter++) {
            def->expandToLocal();
            temp->zero();

            for (int g = 0; g < ngpus; g++) {
                CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
                CHECK_CUBLAS(cublasSetStream(cublasHandles[g], streams[g]));

                int nrhs_vals = n_rhs_blocks[g] * block_dim;
                dim3 block(128);
                dim3 grid((nrhs_vals + block.x - 1) / block.x);

                k_copyLocalRHSIntoBatched<T>
                    <<<grid, block, 0, streams[g]>>>(nrhs_vals, block_dim, size, d_rhs_local_map[g],
                                                     def->getLocalPtr(g), d_Xarray[g]);

                CHECK_CUDA(cudaGetLastError());

                const double alpha = 1.0;
                const double beta = 0.0;

                CHECK_CUBLAS(cublasDgemmBatched(cublasHandles[g], CUBLAS_OP_N, CUBLAS_OP_N, n, 1, n,
                                                &alpha, (const double **)d_invAarray[g], n,
                                                (const double **)d_Xarray[g], n, &beta, d_Yarray[g],
                                                n, batch_size[g]));

                k_copyBatchedIntoOwnedSoln<T><<<grid, block, 0, streams[g]>>>(
                    nrhs_vals, block_dim, size, d_rhs_owned_map[g], d_Yarray[g], temp->getPtr(g));

                CHECK_CUDA(cudaGetLastError());
            }

            ctx->sync();

            T minus_omega = -omega;
            T one = 1.0;
            A->mult(minus_omega, temp, one, def);

            soln->axpy(omega, temp);
        }
    }

   private:
    MultiGPUContext *ctx = nullptr;
    const Partitioner *part = nullptr;
    GPUbsrmat<T, Partitioner> *A = nullptr;

    cublasHandle_t *cublasHandles = nullptr;
    cusparseHandle_t *cusparseHandles = nullptr;
    cudaStream_t *streams = nullptr;

    int ngpus = 0;
    int block_dim = 0, block_dim2 = 0;
    int size = 2, size2 = 4, size4 = 16, n = 24;
    T omega = 0.25;
    int iters = 1;
    bool debug = false;

    int *batch_size = nullptr;
    int *n_batch_blocks = nullptr;
    int *n_rhs_blocks = nullptr;

    int **h_block_inds = nullptr;
    int **h_rhs_local_map = nullptr;
    int **h_rhs_owned_map = nullptr;

    int **d_block_inds = nullptr;
    int **d_rhs_local_map = nullptr;
    int **d_rhs_owned_map = nullptr;

    T ***d_Aarray = nullptr;
    T ***d_invAarray = nullptr;
    T ***d_Xarray = nullptr;
    T ***d_Yarray = nullptr;

    T **d_Adata = nullptr;
    T **d_invAdata = nullptr;
    T **d_Xdata = nullptr;
    T **d_Ydata = nullptr;

    int **d_PivotArray = nullptr;
    int **d_InfoArray = nullptr;

    GPUvec<T, Partitioner> *temp = nullptr;
    GPUvec<T, Partitioner> *defect = nullptr;

    void allocate_arrays() {
        batch_size = new int[ngpus];
        n_batch_blocks = new int[ngpus];
        n_rhs_blocks = new int[ngpus];

        h_block_inds = new int *[ngpus];
        h_rhs_local_map = new int *[ngpus];
        h_rhs_owned_map = new int *[ngpus];

        d_block_inds = new int *[ngpus];
        d_rhs_local_map = new int *[ngpus];
        d_rhs_owned_map = new int *[ngpus];

        d_Aarray = new T **[ngpus];
        d_invAarray = new T **[ngpus];
        d_Xarray = new T **[ngpus];
        d_Yarray = new T **[ngpus];

        d_Adata = new T *[ngpus];
        d_invAdata = new T *[ngpus];
        d_Xdata = new T *[ngpus];
        d_Ydata = new T *[ngpus];

        d_PivotArray = new int *[ngpus];
        d_InfoArray = new int *[ngpus];

        std::memset(batch_size, 0, ngpus * sizeof(int));
        std::memset(n_batch_blocks, 0, ngpus * sizeof(int));
        std::memset(n_rhs_blocks, 0, ngpus * sizeof(int));

        std::memset(h_block_inds, 0, ngpus * sizeof(int *));
        std::memset(h_rhs_local_map, 0, ngpus * sizeof(int *));
        std::memset(h_rhs_owned_map, 0, ngpus * sizeof(int *));

        std::memset(d_block_inds, 0, ngpus * sizeof(int *));
        std::memset(d_rhs_local_map, 0, ngpus * sizeof(int *));
        std::memset(d_rhs_owned_map, 0, ngpus * sizeof(int *));

        std::memset(d_Aarray, 0, ngpus * sizeof(T **));
        std::memset(d_invAarray, 0, ngpus * sizeof(T **));
        std::memset(d_Xarray, 0, ngpus * sizeof(T **));
        std::memset(d_Yarray, 0, ngpus * sizeof(T **));

        std::memset(d_Adata, 0, ngpus * sizeof(T *));
        std::memset(d_invAdata, 0, ngpus * sizeof(T *));
        std::memset(d_Xdata, 0, ngpus * sizeof(T *));
        std::memset(d_Ydata, 0, ngpus * sizeof(T *));

        std::memset(d_PivotArray, 0, ngpus * sizeof(int *));
        std::memset(d_InfoArray, 0, ngpus * sizeof(int *));
    }

    // void build_maps() {
    //     for (int g = 0; g < ngpus; g++) {
    //         batch_size[g] = part->local_nelems[g];
    //         n_batch_blocks[g] = batch_size[g] * size4;
    //         n_rhs_blocks[g] = batch_size[g] * size2;

    //         h_block_inds[g] = new int[n_batch_blocks[g]];
    //         h_rhs_local_map[g] = new int[n_rhs_blocks[g]];
    //         h_rhs_owned_map[g] = new int[n_rhs_blocks[g]];

    //         std::memset(h_block_inds[g], 0, n_batch_blocks[g] * sizeof(int));
    //         std::memset(h_rhs_local_map[g], 0, n_rhs_blocks[g] * sizeof(int));
    //         std::memset(h_rhs_owned_map[g], -1, n_rhs_blocks[g] * sizeof(int));

    //         int *elem_ind_map = A->getHostLocalElemIndMap(g);  // add this getter
    //         int *row_conn = A->getHostRowRedElemConn(g);       // add this getter
    //         int *col_conn = A->getHostColRedElemConn(g);       // add this getter

    //         for (int e = 0; e < batch_size[g]; e++) {
    //             for (int ij = 0; ij < size4; ij++) {
    //                 int ind = e * size4 + ij;
    //                 h_block_inds[g][ind] = elem_ind_map[ind];
    //             }

    //             for (int a = 0; a < size2; a++) {
    //                 int ind = e * size2 + a;
    //                 h_rhs_local_map[g][ind] = col_conn[e * size2 + a];
    //                 h_rhs_owned_map[g][ind] = row_conn[e * size2 + a];
    //             }
    //         }
    //     }
    // }

    void build_maps() {
        for (int g = 0; g < ngpus; g++) {
            batch_size[g] = part->local_nelems[g];
            n_batch_blocks[g] = batch_size[g] * size4;
            n_rhs_blocks[g] = batch_size[g] * size2;

            h_block_inds[g] = new int[n_batch_blocks[g]];
            h_rhs_local_map[g] = new int[n_rhs_blocks[g]];
            h_rhs_owned_map[g] = new int[n_rhs_blocks[g]];

            std::fill(h_block_inds[g], h_block_inds[g] + n_batch_blocks[g], -1);
            std::fill(h_rhs_local_map[g], h_rhs_local_map[g] + n_rhs_blocks[g], -1);
            std::fill(h_rhs_owned_map[g], h_rhs_owned_map[g] + n_rhs_blocks[g], -1);

            int *row_conn = A->getHostRowRedElemConn(g);  // owned-row ids, -1 for ghost rows
            int *col_conn = A->getHostColRedElemConn(g);  // local ids, includes ghosts
            int *rowp = A->getHostLocalRowp(g);
            int *cols = A->getHostLocalCols(g);

            for (int e = 0; e < batch_size[g]; e++) {
                for (int ij = 0; ij < size4; ij++) {
                    int a = ij % size2;  // local patch row node
                    int b = ij / size2;  // local patch col node

                    int row = row_conn[e * size2 + a];
                    int col = col_conn[e * size2 + b];

                    int jp = -1;
                    if (row >= 0 && col >= 0) {
                        for (int p = rowp[row]; p < rowp[row + 1]; p++) {
                            if (cols[p] == col) {
                                jp = p;
                                break;
                            }
                        }
                    }

                    h_block_inds[g][e * size4 + ij] = jp;
                }

                for (int a = 0; a < size2; a++) {
                    int ind = e * size2 + a;
                    h_rhs_local_map[g][ind] = col_conn[ind];
                    h_rhs_owned_map[g][ind] = row_conn[ind];
                }
            }
        }
    }

    void move_maps_to_device() {
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));

            CHECK_CUDA(cudaMalloc((void **)&d_block_inds[g], n_batch_blocks[g] * sizeof(int)));
            CHECK_CUDA(cudaMalloc((void **)&d_rhs_local_map[g], n_rhs_blocks[g] * sizeof(int)));
            CHECK_CUDA(cudaMalloc((void **)&d_rhs_owned_map[g], n_rhs_blocks[g] * sizeof(int)));

            CHECK_CUDA(cudaMemcpyAsync(d_block_inds[g], h_block_inds[g],
                                       n_batch_blocks[g] * sizeof(int), cudaMemcpyHostToDevice,
                                       streams[g]));
            CHECK_CUDA(cudaMemcpyAsync(d_rhs_local_map[g], h_rhs_local_map[g],
                                       n_rhs_blocks[g] * sizeof(int), cudaMemcpyHostToDevice,
                                       streams[g]));
            CHECK_CUDA(cudaMemcpyAsync(d_rhs_owned_map[g], h_rhs_owned_map[g],
                                       n_rhs_blocks[g] * sizeof(int), cudaMemcpyHostToDevice,
                                       streams[g]));
        }
    }

    void allocate_batched_memory() {
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));

            size_t mat_bytes = (size_t)batch_size[g] * n * n * sizeof(T);
            size_t vec_bytes = (size_t)batch_size[g] * n * sizeof(T);
            size_t ptr_bytes = (size_t)batch_size[g] * sizeof(T *);

            CHECK_CUDA(cudaMalloc((void **)&d_Adata[g], mat_bytes));
            CHECK_CUDA(cudaMalloc((void **)&d_invAdata[g], mat_bytes));
            CHECK_CUDA(cudaMalloc((void **)&d_Xdata[g], vec_bytes));
            CHECK_CUDA(cudaMalloc((void **)&d_Ydata[g], vec_bytes));

            CHECK_CUDA(cudaMalloc((void **)&d_Aarray[g], ptr_bytes));
            CHECK_CUDA(cudaMalloc((void **)&d_invAarray[g], ptr_bytes));
            CHECK_CUDA(cudaMalloc((void **)&d_Xarray[g], ptr_bytes));
            CHECK_CUDA(cudaMalloc((void **)&d_Yarray[g], ptr_bytes));

            CHECK_CUDA(
                cudaMalloc((void **)&d_PivotArray[g], (size_t)batch_size[g] * n * sizeof(int)));
            CHECK_CUDA(cudaMalloc((void **)&d_InfoArray[g], (size_t)batch_size[g] * sizeof(int)));

            dim3 block(128);
            dim3 grid((batch_size[g] + block.x - 1) / block.x);

            k_setupBatchedPointers<T><<<grid, block, 0, streams[g]>>>(
                batch_size[g], n, d_Adata[g], d_invAdata[g], d_Xdata[g], d_Ydata[g], d_Aarray[g],
                d_invAarray[g], d_Xarray[g], d_Yarray[g]);

            CHECK_CUDA(cudaGetLastError());
        }
        ctx->sync();
    }
};