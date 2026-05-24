#pragma once
#include <cmath>
#include <cstring>

#include "_matvec.cuh"
#include "cuda_utils.h"
#include "linalg/vec.cuh"
#include "linalg/vec.h"
#include "utils/multigpu_context.h"

template <typename T, class Partitioner>
class GPUvec {
   public:
    GPUvec(MultiGPUContext *ctx_, const Partitioner *part_, int block_dim_ = 6)
        : ctx(ctx_),
          part(part_),
          cublasHandles(ctx_->cublasHandles),
          streams(ctx_->streams),
          ngpus(part_->ngpus),
          num_nodes(part_->num_nodes),
          block_dim(block_dim_),
          N(part_->num_nodes * block_dim_),
          debug(ctx_->debug) {
        allocate_owned();
        allocate_local();
        allocate_reduced();
        allocate_single_gpu_transfer();
        ctx->sync();
    }

    void free() {
        sync();

        if (d_vals_owned) {
            for (int g = 0; g < ngpus; g++) {
                CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
                if (d_vals_owned[g]) cudaFree(d_vals_owned[g]);
                if (d_vals_local[g]) cudaFree(d_vals_local[g]);
            }
        }

        if (d_vals_red) {
            for (int dst = 0; dst < ngpus; dst++) {
                for (int src = 0; src < ngpus; src++) {
                    int idx = pair_index(dst, src);

                    if (d_vals_red[idx]) {
                        CHECK_CUDA(cudaSetDevice(debug ? 0 : src));
                        cudaFree(d_vals_red[idx]);
                    }

                    if (d_vals_red_dst[idx]) {
                        CHECK_CUDA(cudaSetDevice(debug ? 0 : dst));
                        cudaFree(d_vals_red_dst[idx]);
                    }
                }
            }
        }

        if (d_single_gpu_owned_nodes) {
            CHECK_CUDA(cudaSetDevice(0));
            for (int g = 0; g < ngpus; g++) {
                if (d_single_gpu_owned_nodes[g]) cudaFree(d_single_gpu_owned_nodes[g]);
                if (d_single_gpu_pack[g]) cudaFree(d_single_gpu_pack[g]);
            }
        }

        if (d_single_pack) {
            for (int g = 0; g < ngpus; g++) {
                CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
                if (d_single_pack[g]) cudaFree(d_single_pack[g]);
            }
        }

        delete[] owned_N;
        delete[] local_N;
        delete[] red_N;
        delete[] d_vals_owned;
        delete[] d_vals_local;
        delete[] d_vals_red;
        delete[] d_vals_red_dst;

        delete[] d_single_gpu_owned_nodes;
        delete[] d_single_pack;
        delete[] d_single_gpu_pack;
    }

    int pair_index(int dst, int src) const { return ngpus * dst + src; }

    void printValuesOnHost() {
        // prints owned values only from each GPU
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            T *h_vals_owned = DeviceVec<T>(owned_N[g], d_vals_owned[g]).createHostVec().getPtr();
            int owned_nnodes = owned_N[g] / block_dim;
            printf("h_vec(nnodes=%d) on GPU[%d]\n", owned_nnodes, g);
            for (int i = 0; i < owned_nnodes; i++) {
                int global_node = part->h_owned_nodes[g][i];
                T *h_block = &h_vals_owned[block_dim * i];
                printf("GPU[%d]-node[%d]: ", g, global_node);
                printVec<T>(block_dim, h_block);
            }
        }

        ctx->sync();
    }

    T *getLocalVecOnHost(int g) {
        T *h_vals_local = DeviceVec<T>(local_N[g], d_vals_local[g]).createHostVec().getPtr();
        return h_vals_local;
    }

    void allocate_owned() {
        owned_N = new int[ngpus];
        d_vals_owned = new T *[ngpus];
        std::memset(owned_N, 0, ngpus * sizeof(int));
        std::memset(d_vals_owned, 0, ngpus * sizeof(T *));

        for (int g = 0; g < ngpus; g++) {
            owned_N[g] = part->owned_nnodes[g] * block_dim;
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            CHECK_CUBLAS(cublasSetStream(cublasHandles[g], streams[g]));
            CHECK_CUDA(cudaMalloc((void **)&d_vals_owned[g], owned_N[g] * sizeof(T)));
            CHECK_CUDA(cudaMemsetAsync(d_vals_owned[g], 0, owned_N[g] * sizeof(T), streams[g]));
        }
    }

    void allocate_local() {
        local_N = new int[ngpus];
        d_vals_local = new T *[ngpus];
        std::memset(local_N, 0, ngpus * sizeof(int));
        std::memset(d_vals_local, 0, ngpus * sizeof(T *));

        for (int g = 0; g < ngpus; g++) {
            local_N[g] = part->local_nnodes[g] * block_dim;
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            CHECK_CUDA(cudaMalloc((void **)&d_vals_local[g], local_N[g] * sizeof(T)));
            CHECK_CUDA(cudaMemsetAsync(d_vals_local[g], 0, local_N[g] * sizeof(T), streams[g]));
        }
    }

    void allocate_reduced() {
        int npairs = ngpus * ngpus;
        red_N = new int[npairs];
        d_vals_red = new T *[npairs];
        d_vals_red_dst = new T *[npairs];

        std::memset(red_N, 0, npairs * sizeof(int));
        std::memset(d_vals_red, 0, npairs * sizeof(T *));
        std::memset(d_vals_red_dst, 0, npairs * sizeof(T *));

        for (int dst = 0; dst < ngpus; dst++) {
            for (int src = 0; src < ngpus; src++) {
                if (src == dst) continue;

                int idx = pair_index(dst, src);
                red_N[idx] = part->srcdest_nnodes[idx] * block_dim;

                if (red_N[idx] == 0) continue;

                CHECK_CUDA(cudaSetDevice(debug ? 0 : src));
                CHECK_CUDA(cudaMalloc((void **)&d_vals_red[idx], red_N[idx] * sizeof(T)));

                CHECK_CUDA(cudaSetDevice(debug ? 0 : dst));
                CHECK_CUDA(cudaMalloc((void **)&d_vals_red_dst[idx], red_N[idx] * sizeof(T)));
            }
        }
    }

    void allocate_single_gpu_transfer() {
        d_single_gpu_owned_nodes = new int *[ngpus];
        d_single_pack = new T *[ngpus];
        d_single_gpu_pack = new T *[ngpus];

        std::memset(d_single_gpu_owned_nodes, 0, ngpus * sizeof(int *));
        std::memset(d_single_pack, 0, ngpus * sizeof(T *));
        std::memset(d_single_gpu_pack, 0, ngpus * sizeof(T *));

        for (int g = 0; g < ngpus; g++) {
            int owned_nnodes = part->owned_nnodes[g];
            int Nowned = owned_nnodes * block_dim;
            if (Nowned == 0) continue;

            // GPU-0 copy of owned global-node map for final scatter/gather
            CHECK_CUDA(cudaSetDevice(0));
            CHECK_CUDA(
                cudaMalloc((void **)&d_single_gpu_owned_nodes[g], owned_nnodes * sizeof(int)));
            CHECK_CUDA(cudaMemcpy(d_single_gpu_owned_nodes[g], part->h_owned_nodes[g],
                                  owned_nnodes * sizeof(int), cudaMemcpyHostToDevice));

            CHECK_CUDA(cudaMalloc((void **)&d_single_gpu_pack[g], Nowned * sizeof(T)));

            // Per-GPU contiguous pack buffer
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            CHECK_CUDA(cudaMalloc((void **)&d_single_pack[g], Nowned * sizeof(T)));
        }

        ctx->sync();
    }

    void zeroAll() {
        zero();
        zeroLocal();
    }

    void zero() {
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            CHECK_CUDA(cudaMemsetAsync(d_vals_owned[g], 0, owned_N[g] * sizeof(T), streams[g]));
        }
        sync();
    }

    void zeroLocal() {
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            CHECK_CUDA(cudaMemsetAsync(d_vals_local[g], 0, local_N[g] * sizeof(T), streams[g]));
        }
        sync();
    }

    void copyTo(GPUvec<T, Partitioner> *y) {
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            CHECK_CUDA(cudaMemcpyAsync(y->d_vals_owned[g], d_vals_owned[g], owned_N[g] * sizeof(T),
                                       cudaMemcpyDeviceToDevice, streams[g]));
        }
        sync();
    }

    void copyToSingleGPU(T *d_single_gpu_vals, bool zero_single_gpu_vals = true) {
        // d_single_gpu_vals lives on GPU 0 and has size N = num_nodes * block_dim

        if (zero_single_gpu_vals) {
            CHECK_CUDA(cudaSetDevice(0));
            CHECK_CUDA(cudaMemsetAsync(d_single_gpu_vals, 0, N * sizeof(T), streams[0]));
        }

        // 1) Copy owned values into contiguous per-GPU pack buffers
        for (int g = 0; g < ngpus; g++) {
            int Nowned = owned_N[g];
            if (Nowned == 0) continue;

            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            CHECK_CUDA(cudaMemcpyAsync(d_single_pack[g], d_vals_owned[g], Nowned * sizeof(T),
                                       cudaMemcpyDeviceToDevice, streams[g]));
        }

        sync();

        // 2) Peer-copy per-GPU packs to GPU 0 packs
        for (int g = 0; g < ngpus; g++) {
            int Nowned = owned_N[g];
            if (Nowned == 0) continue;

            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));

            if (debug || g == 0) {
                CHECK_CUDA(cudaMemcpyAsync(d_single_gpu_pack[g], d_single_pack[g],
                                           Nowned * sizeof(T), cudaMemcpyDeviceToDevice,
                                           streams[g]));
            } else {
                CHECK_CUDA(cudaMemcpyPeerAsync(d_single_gpu_pack[g], 0, d_single_pack[g], g,
                                               Nowned * sizeof(T), streams[g]));
            }
        }

        sync();

        // 3) Scatter GPU 0 packs into full single-GPU vector
        CHECK_CUDA(cudaSetDevice(0));

        for (int g = 0; g < ngpus; g++) {
            int owned_nnodes = part->owned_nnodes[g];
            int Nowned = owned_N[g];
            if (Nowned == 0) continue;

            dim3 block(128);
            dim3 grid((Nowned + block.x - 1) / block.x);

            k_scatter_single_pack_to_global<T><<<grid, block, 0, streams[0]>>>(
                owned_nnodes, block_dim, d_single_gpu_owned_nodes[g], d_single_gpu_pack[g],
                d_single_gpu_vals);

            CHECK_CUDA(cudaGetLastError());
        }

        sync();
    }

    void copyFromSingleGPU(const T *d_single_gpu_vals) {
        // d_single_gpu_vals lives on GPU 0 and has size N = num_nodes * block_dim

        CHECK_CUDA(cudaSetDevice(0));

        // 1) Gather full single-GPU vector into GPU 0 packs
        for (int g = 0; g < ngpus; g++) {
            int owned_nnodes = part->owned_nnodes[g];
            int Nowned = owned_N[g];
            if (Nowned == 0) continue;

            dim3 block(128);
            dim3 grid((Nowned + block.x - 1) / block.x);

            k_gather_global_to_single_pack<T><<<grid, block, 0, streams[0]>>>(
                owned_nnodes, block_dim, d_single_gpu_owned_nodes[g], d_single_gpu_vals,
                d_single_gpu_pack[g]);

            CHECK_CUDA(cudaGetLastError());
        }

        sync();

        // 2) Peer-copy GPU 0 packs back to per-GPU packs
        for (int g = 0; g < ngpus; g++) {
            int Nowned = owned_N[g];
            if (Nowned == 0) continue;

            CHECK_CUDA(cudaSetDevice(0));

            if (debug || g == 0) {
                CHECK_CUDA(cudaMemcpyAsync(d_single_pack[g], d_single_gpu_pack[g],
                                           Nowned * sizeof(T), cudaMemcpyDeviceToDevice,
                                           streams[0]));
            } else {
                CHECK_CUDA(cudaMemcpyPeerAsync(d_single_pack[g], g, d_single_gpu_pack[g], 0,
                                               Nowned * sizeof(T), streams[0]));
            }
        }

        sync();

        // 3) Copy contiguous per-GPU packs into owned vectors
        for (int g = 0; g < ngpus; g++) {
            int Nowned = owned_N[g];
            if (Nowned == 0) continue;

            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            CHECK_CUDA(cudaMemcpyAsync(d_vals_owned[g], d_single_pack[g], Nowned * sizeof(T),
                                       cudaMemcpyDeviceToDevice, streams[g]));
        }

        sync();
    }

    void setValuesFromHost(const T *h_vals) {
        for (int g = 0; g < ngpus; g++) {
            if (owned_N[g] == 0) continue;

            T *h_owned_tmp = new T[owned_N[g]];

            for (int i = 0; i < part->owned_nnodes[g]; i++) {
                int global_node = part->h_owned_nodes[g][i];
                for (int idof = 0; idof < block_dim; idof++) {
                    h_owned_tmp[i * block_dim + idof] = h_vals[global_node * block_dim + idof];
                }
            }

            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            CHECK_CUDA(cudaMemcpyAsync(d_vals_owned[g], h_owned_tmp, owned_N[g] * sizeof(T),
                                       cudaMemcpyHostToDevice, streams[g]));

            CHECK_CUDA(cudaStreamSynchronize(streams[g]));
            delete[] h_owned_tmp;
        }
    }

    void getValuesToHost(T *h_vals) const {
        for (int g = 0; g < ngpus; g++) {
            if (owned_N[g] == 0) continue;

            T *h_owned_tmp = new T[owned_N[g]];

            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            CHECK_CUDA(cudaMemcpyAsync(h_owned_tmp, d_vals_owned[g], owned_N[g] * sizeof(T),
                                       cudaMemcpyDeviceToHost, streams[g]));

            CHECK_CUDA(cudaStreamSynchronize(streams[g]));

            for (int i = 0; i < part->owned_nnodes[g]; i++) {
                int global_node = part->h_owned_nodes[g][i];
                for (int idof = 0; idof < block_dim; idof++) {
                    h_vals[global_node * block_dim + idof] = h_owned_tmp[i * block_dim + idof];
                }
            }

            delete[] h_owned_tmp;
        }
    }

    void packGhostReduced() {
        for (int dst = 0; dst < ngpus; dst++) {
            for (int src = 0; src < ngpus; src++) {
                if (src == dst) continue;

                int idx = pair_index(dst, src);
                int Nred = red_N[idx];

                if (Nred == 0) continue;

                CHECK_CUDA(cudaSetDevice(debug ? 0 : src));

                dim3 block(32);
                dim3 grid((Nred + block.x - 1) / block.x);

                k_pack_ghost_red<T><<<grid, block, 0, streams[src]>>>(
                    Nred / block_dim, block_dim, part->d_srcred_map[idx], d_vals_owned[src],
                    d_vals_red[idx]);

                CHECK_CUDA(cudaGetLastError());
            }
        }
    }

    void expandToLocal() {
        zeroLocal();

        for (int g = 0; g < ngpus; g++) {
            if (part->owned_nnodes[g] == 0) continue;

            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));

            dim3 block(32);
            dim3 grid((owned_N[g] + block.x - 1) / block.x);

            k_scatter_owned_to_local<T><<<grid, block, 0, streams[g]>>>(
                part->owned_nnodes[g], block_dim, part->d_owned_to_local_map[g], d_vals_owned[g],
                d_vals_local[g]);

            CHECK_CUDA(cudaGetLastError());
        }

        packGhostReduced();

        for (int dst = 0; dst < ngpus; dst++) {
            for (int src = 0; src < ngpus; src++) {
                if (src == dst) continue;

                int idx = pair_index(dst, src);
                int Nred = red_N[idx];

                if (Nred == 0) continue;

                CHECK_CUDA(cudaSetDevice(debug ? 0 : src));

                if (debug) {
                    CHECK_CUDA(cudaMemcpyAsync(d_vals_red_dst[idx], d_vals_red[idx],
                                               Nred * sizeof(T), cudaMemcpyDeviceToDevice,
                                               streams[src]));
                } else {
                    CHECK_CUDA(cudaMemcpyPeerAsync(d_vals_red_dst[idx], dst, d_vals_red[idx], src,
                                                   Nred * sizeof(T), streams[src]));
                }
            }
        }

        sync();

        for (int dst = 0; dst < ngpus; dst++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : dst));

            for (int src = 0; src < ngpus; src++) {
                if (src == dst) continue;

                int idx = pair_index(dst, src);
                int Nred = red_N[idx];

                if (Nred == 0) continue;

                int nred_nodes = Nred / block_dim;

                dim3 block(32);
                dim3 grid((Nred + block.x - 1) / block.x);

                k_place_ghost_red<T><<<grid, block, 0, streams[dst]>>>(
                    nred_nodes, block_dim, part->d_dstred_map[idx], d_vals_red_dst[idx],
                    d_vals_local[dst]);

                CHECK_CUDA(cudaGetLastError());
            }
        }

        sync();
    }

    void reduceFromLocal() {
        zero();

        for (int g = 0; g < ngpus; g++) {
            int Nowned = owned_N[g];
            if (Nowned == 0) continue;

            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));

            dim3 block(32);
            dim3 grid((Nowned + block.x - 1) / block.x);

            k_add_local_owned_to_owned<T><<<grid, block, 0, streams[g]>>>(
                Nowned, block_dim, part->d_owned_to_local_map[g], d_vals_local[g], d_vals_owned[g]);

            CHECK_CUDA(cudaGetLastError());
        }

        sync();

        for (int dst = 0; dst < ngpus; dst++) {
            for (int src = 0; src < ngpus; src++) {
                if (src == dst) continue;

                int idx = pair_index(dst, src);
                int Nred = red_N[idx];
                if (Nred == 0) continue;

                int red_nnodes = Nred / block_dim;

                CHECK_CUDA(cudaSetDevice(debug ? 0 : dst));

                dim3 block(32);
                dim3 grid((Nred + block.x - 1) / block.x);

                k_pack_local_ghost_red<T><<<grid, block, 0, streams[dst]>>>(
                    red_nnodes, block_dim, part->d_dstred_map[idx], d_vals_local[dst],
                    d_vals_red_dst[idx]);

                CHECK_CUDA(cudaGetLastError());
            }
        }

        for (int dst = 0; dst < ngpus; dst++) {
            for (int src = 0; src < ngpus; src++) {
                if (src == dst) continue;

                int idx = pair_index(dst, src);
                int Nred = red_N[idx];
                if (Nred == 0) continue;

                CHECK_CUDA(cudaSetDevice(debug ? 0 : dst));

                if (debug) {
                    CHECK_CUDA(cudaMemcpyAsync(d_vals_red[idx], d_vals_red_dst[idx],
                                               Nred * sizeof(T), cudaMemcpyDeviceToDevice,
                                               streams[dst]));
                } else {
                    CHECK_CUDA(cudaMemcpyPeerAsync(d_vals_red[idx], src, d_vals_red_dst[idx], dst,
                                                   Nred * sizeof(T), streams[dst]));
                }
            }
        }

        sync();

        for (int dst = 0; dst < ngpus; dst++) {
            for (int src = 0; src < ngpus; src++) {
                if (src == dst) continue;

                int idx = pair_index(dst, src);
                int Nred = red_N[idx];
                if (Nred == 0) continue;
                int red_nnodes = Nred / block_dim;

                CHECK_CUDA(cudaSetDevice(debug ? 0 : src));

                dim3 block(32);
                dim3 grid((Nred + block.x - 1) / block.x);

                k_add_red_to_owned<T><<<grid, block, 0, streams[src]>>>(
                    red_nnodes, block_dim, part->d_srcred_map[idx], d_vals_red[idx],
                    d_vals_owned[src]);

                CHECK_CUDA(cudaGetLastError());
            }
        }

        sync();
    }

    void apply_bcs(int *n_owned_bcs, int **d_owned_bcs, int *n_local_bcs, int **d_local_bcs) {
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            CHECK_CUBLAS(cublasSetStream(cublasHandles[g], streams[g]));

            dim3 block(32);
            dim3 grid1((n_owned_bcs[g] + 31) / 32);
            k_vec_apply_bcs<T>
                <<<grid1, block, 0, streams[g]>>>(n_owned_bcs[g], d_owned_bcs[g], d_vals_owned[g]);

            dim3 grid2((n_local_bcs[g] + 31) / 32);
            k_vec_apply_bcs<T>
                <<<grid2, block, 0, streams[g]>>>(n_local_bcs[g], d_local_bcs[g], d_vals_local[g]);
        }
        ctx->sync();
    }

    T dotProd(GPUvec<T, Partitioner> *y) {
        T total = 0.0;
        T *h_dot = new T[ngpus];
        std::memset(h_dot, 0, ngpus * sizeof(T));

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            CHECK_CUBLAS(cublasSetStream(cublasHandles[g], streams[g]));
            CHECK_CUBLAS(cublasDdot(cublasHandles[g], owned_N[g], d_vals_owned[g], 1,
                                    y->d_vals_owned[g], 1, &h_dot[g]));
        }

        sync();

        for (int g = 0; g < ngpus; g++) total += h_dot[g];

        delete[] h_dot;
        return total;
    }

    T norm() {
        T dot = dotProd(this);
        return std::sqrt(dot);
    }

    void scale(T alpha) {
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            CHECK_CUBLAS(cublasSetStream(cublasHandles[g], streams[g]));
            CHECK_CUBLAS(cublasDscal(cublasHandles[g], owned_N[g], &alpha, d_vals_owned[g], 1));
        }
        sync();
    }

    void axpy(T alpha, GPUvec<T, Partitioner> *x) {
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            CHECK_CUBLAS(cublasSetStream(cublasHandles[g], streams[g]));
            CHECK_CUBLAS(cublasDaxpy(cublasHandles[g], owned_N[g], &alpha, x->d_vals_owned[g], 1,
                                     d_vals_owned[g], 1));
        }
        sync();
    }

    void axpby(T alpha, GPUvec<T, Partitioner> *x, T beta) {
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            CHECK_CUBLAS(cublasSetStream(cublasHandles[g], streams[g]));
            CHECK_CUBLAS(cublasDscal(cublasHandles[g], owned_N[g], &beta, d_vals_owned[g], 1));
            CHECK_CUBLAS(cublasDaxpy(cublasHandles[g], owned_N[g], &alpha, x->d_vals_owned[g], 1,
                                     d_vals_owned[g], 1));
        }
        sync();
    }

    T **getOwnedDoublePtr() { return d_vals_owned; }
    T **getLocalDoublePtr() { return d_vals_local; }

    T *getPtr(int g) { return d_vals_owned[g]; }
    T *getLocalPtr(int g) { return d_vals_local[g]; }
    T *getRedPtr(int dst, int src) { return d_vals_red[pair_index(dst, src)]; }
    T *getRedDstPtr(int dst, int src) { return d_vals_red_dst[pair_index(dst, src)]; }

    // TODO : nomenclature change here (getLocalSize should not be for owned_N, etc..)
    int getLocalSize(int g) const { return owned_N[g]; }
    int getExpandedSize(int g) const { return local_N[g]; }
    int getLocalNodes(int g) const { return part->owned_nnodes[g]; }
    int getExpandedNodes(int g) const { return part->local_nnodes[g]; }
    int getRedSize(int dst, int src) const { return red_N[pair_index(dst, src)]; }

    void sync() const {
        if (ctx) {
            ctx->sync();
            return;
        }

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            CHECK_CUDA(cudaStreamSynchronize(streams[g]));
        }
    }

    MultiGPUContext *ctx = nullptr;
    const Partitioner *part = nullptr;
    cublasHandle_t *cublasHandles = nullptr;
    cudaStream_t *streams = nullptr;

    int ngpus = 0;
    int num_nodes = 0;
    int block_dim = 0;
    int N = 0;
    bool debug = false;

    int *owned_N = nullptr;
    int *local_N = nullptr;
    int *red_N = nullptr;

    T **d_vals_owned = nullptr;
    T **d_vals_local = nullptr;
    T **d_vals_red = nullptr;
    T **d_vals_red_dst = nullptr;

    int **d_single_gpu_owned_nodes = nullptr;
    T **d_single_pack = nullptr;
    T **d_single_gpu_pack = nullptr;
};