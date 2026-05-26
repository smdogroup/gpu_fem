#pragma once
#include <cstring>
#include <type_traits>
#include <vector>

#include "_asw.cuh"
#include "cuda_utils.h"
#include "matvec/gpumat.h"
#include "matvec/gpuvec.h"

template <typename T, class Partitioner>
class MultiGPUElementASW {
   public:
    static constexpr int nodes_per_elem = 4;

    MultiGPUElementASW(MultiGPUContext *ctx_, Partitioner *part_, GPUbsrmat<T, Partitioner> *A_,
                       T omega_ = 0.25, int iters_ = 5)
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
        printf("ASW - build_ghost_maps\n");
        // build_ghost_maps(true);
        build_ghost_maps();
        printf("ASW - allocate_batched_memory\n");
        allocate_batched_memory();
        printf("ASW - allocate_ghost_batched_memory\n");
        allocate_ghost_batched_memory();

        printf("ASW - move_maps_to_device\n");
        move_maps_to_device();
        printf("ASW - move_ghost_maps_to_device\n");
        move_ghost_maps_to_device();
        printf("ASW - done with constructor\n");

        temp = new GPUvec<T, Partitioner>(ctx, part, block_dim);
        defect = new GPUvec<T, Partitioner>(ctx, part, block_dim);

        ctx->sync();
    }

    void free() {
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

        temp->free();
        defect->free();

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

        int npairs = ngpus * ngpus;
        for (int idx = 0; idx < npairs; idx++) {
            int dst = idx / ngpus;
            int src = idx % ngpus;

            if (d_ghost_asw_blocks[idx]) {
                CHECK_CUDA(cudaSetDevice(debug ? 0 : dst));
                cudaFree(d_ghost_asw_blocks[idx]);
            }

            if (d_ghost_kmat_blocks[idx]) {
                CHECK_CUDA(cudaSetDevice(debug ? 0 : src));
                cudaFree(d_ghost_kmat_blocks[idx]);
            }

            if (d_ghost_vals_red[idx]) {
                CHECK_CUDA(cudaSetDevice(debug ? 0 : src));
                cudaFree(d_ghost_vals_red[idx]);
            }

            if (d_ghost_vals_red_dst[idx]) {
                CHECK_CUDA(cudaSetDevice(debug ? 0 : dst));
                cudaFree(d_ghost_vals_red_dst[idx]);
            }
        }

        delete[] ghost_pair_nblocks;
        delete[] h_ghost_asw_blocks;
        delete[] h_ghost_kmat_blocks;
        delete[] d_ghost_asw_blocks;
        delete[] d_ghost_kmat_blocks;
        delete[] d_ghost_vals_red;
        delete[] d_ghost_vals_red_dst;
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
        }

        ctx->sync();

        add_ghost_ghost_blocks_to_batched();

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            CHECK_CUBLAS(cublasSetStream(cublasHandles[g], streams[g]));

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
            temp->zeroLocal();

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

                k_addBatchedIntoLocalSoln<T>
                    <<<grid, block, 0, streams[g]>>>(nrhs_vals, block_dim, size, d_rhs_local_map[g],
                                                     d_Yarray[g], temp->getLocalPtr(g));

                CHECK_CUDA(cudaGetLastError());
            }

            ctx->sync();

            temp->reduceFromLocal();

            T minus_omega = -omega;
            T one = 1.0;
            A->mult(minus_omega, temp, one, def);

            soln->axpy(omega, temp);
        }
    }

    template <bool SET_VALUES = false>
    void printSubdomainMatValues() {
        if constexpr (SET_VALUES) {
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
            }
            ctx->sync();
        }

        T **h_Adata = new T *[ngpus];

        for (int g = 0; g < ngpus; g++) {
            size_t nmat_vals = (size_t)batch_size[g] * n * n;
            h_Adata[g] = new T[nmat_vals];

            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            CHECK_CUDA(
                cudaMemcpy(h_Adata[g], d_Adata[g], nmat_vals * sizeof(T), cudaMemcpyDeviceToHost));
        }

        for (int g = 0; g < ngpus; g++) {
            printf("MultiGPU ASW elem blocks on GPU[%d / %d]\n", g, ngpus);
            printf("---------------------------\n");

            for (int ibatch = 0; ibatch < batch_size[g]; ibatch++) {
                int global_elem = ibatch + part->getStartElem(g);
                printf("ASW subdomain mat elem %d on GPU[%d]\n", ibatch, g);

                T *Aelem = &h_Adata[g][(size_t)ibatch * n * n];

                for (int brow = 0; brow < size2; brow++) {
                    for (int bcol = 0; bcol < size2; bcol++) {
                        printf("\tblock node (%d,%d)\n", brow, bcol);

                        for (int ii = 0; ii < block_dim; ii++) {
                            for (int jj = 0; jj < block_dim; jj++) {
                                int row = brow * block_dim + ii;
                                int col = bcol * block_dim + jj;

                                // column-major dense storage for cuBLAS
                                printf(" % .6e", Aelem[row + n * col]);
                            }
                            printf("\n");
                        }
                    }
                }
            }
        }

        for (int g = 0; g < ngpus; g++) {
            delete[] h_Adata[g];
        }
        delete[] h_Adata;
    }

   private:
    MultiGPUContext *ctx = nullptr;
    Partitioner *part = nullptr;
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

    int *ghost_pair_nblocks = nullptr;

    // ghost data
    std::vector<int> *h_ghost_asw_blocks = nullptr;
    std::vector<int> *h_ghost_kmat_blocks = nullptr;
    int **d_ghost_asw_blocks = nullptr;
    int **d_ghost_kmat_blocks = nullptr;
    T **d_ghost_vals_red = nullptr;      // allocated on src
    T **d_ghost_vals_red_dst = nullptr;  // allocated on dst

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

        int npairs = ngpus * ngpus;

        ghost_pair_nblocks = new int[npairs];
        h_ghost_asw_blocks = new std::vector<int>[npairs];
        h_ghost_kmat_blocks = new std::vector<int>[npairs];

        d_ghost_asw_blocks = new int *[npairs];
        d_ghost_kmat_blocks = new int *[npairs];
        d_ghost_vals_red = new T *[npairs];
        d_ghost_vals_red_dst = new T *[npairs];

        std::memset(ghost_pair_nblocks, 0, npairs * sizeof(int));
        std::memset(d_ghost_asw_blocks, 0, npairs * sizeof(int *));
        std::memset(d_ghost_kmat_blocks, 0, npairs * sizeof(int *));
        std::memset(d_ghost_vals_red, 0, npairs * sizeof(T *));
        std::memset(d_ghost_vals_red_dst, 0, npairs * sizeof(T *));
    }

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

            // Local element connectivity: element-local slot -> local node index
            int *loc_conn = A->getHostLocalElemConn(g);

            // Local BSR sparsity
            int *local_rowp = A->getHostLocalRowp(g);
            int *local_cols = A->getHostLocalCols(g);

            for (int e = 0; e < batch_size[g]; e++) {
                // Build 4x4 ASW local matrix block map in the exact same
                // ordering assumed by k_copyMatValuesToBatchedContiguous:
                //
                // ij = i + size2 * j
                // i = row node slot inside element
                // j = col node slot inside element
                for (int ij = 0; ij < size4; ij++) {
                    int i = ij % size2;
                    int j = ij / size2;

                    int row_node = loc_conn[e * size2 + i];
                    int col_node = loc_conn[e * size2 + j];

                    int jp_found = -1;

                    if (row_node >= 0 && col_node >= 0) {
                        for (int jp = local_rowp[row_node]; jp < local_rowp[row_node + 1]; jp++) {
                            if (local_cols[jp] == col_node) {
                                jp_found = jp;
                                break;
                            }
                        }
                    }

                    h_block_inds[g][e * size4 + ij] = jp_found;
                }

                // RHS/local correction map
                for (int a = 0; a < size2; a++) {
                    int ind = e * size2 + a;
                    h_rhs_local_map[g][ind] = loc_conn[e * size2 + a];

                    // Still unused, but keep allocated if other code expects it.
                    h_rhs_owned_map[g][ind] = -1;
                }
            }
        }
    }

    int pair_index(int dst, int src) const { return ngpus * dst + src; }

    // void build_ghost_maps() {
    //     for (int dst = 0; dst < ngpus; dst++) {
    //         int *dst_conn = A->getHostLocalElemConn(dst);

    //         for (int e = 0; e < batch_size[dst]; e++) {
    //             for (int ij = 0; ij < size4; ij++) {
    //                 int i = ij % size2;
    //                 int j = ij / size2;

    //                 int dst_row_node = dst_conn[e * size2 + i];
    //                 int dst_col_node = dst_conn[e * size2 + j];

    //                 if (dst_row_node < 0 || dst_col_node < 0) continue;

    //                 // Only patch ghost x ghost blocks on dst
    //                 if (dst_row_node < part->owned_nnodes[dst]) continue;
    //                 if (dst_col_node < part->owned_nnodes[dst]) continue;

    //                 int glob_row = part->h_local_nodes[dst][dst_row_node];
    //                 int glob_col = part->h_local_nodes[dst][dst_col_node];

    //                 for (int src = 0; src < ngpus; src++) {
    //                     if (src == dst) continue;

    //                     int *src_rowp = A->getHostLocalRowp(src);
    //                     int *src_cols = A->getHostLocalCols(src);

    //                     int src_row_node = -1;
    //                     int src_col_node = -1;

    //                     for (int a = 0; a < part->local_nnodes[src]; a++) {
    //                         int gnode = part->h_local_nodes[src][a];
    //                         if (gnode == glob_row) src_row_node = a;
    //                         if (gnode == glob_col) src_col_node = a;
    //                     }

    //                     if (src_row_node < 0 || src_col_node < 0) continue;

    //                     int jp_found = -1;
    //                     for (int jp = src_rowp[src_row_node]; jp < src_rowp[src_row_node + 1];
    //                          jp++) {
    //                         if (src_cols[jp] == src_col_node) {
    //                             jp_found = jp;
    //                             break;
    //                         }
    //                     }

    //                     if (jp_found >= 0) {
    //                         int idx = pair_index(dst, src);
    //                         h_ghost_asw_blocks[idx].push_back(e * size4 + ij);
    //                         h_ghost_kmat_blocks[idx].push_back(jp_found);
    //                         break;
    //                     }
    //                 }
    //             }
    //         }
    //     }

    //     for (int dst = 0; dst < ngpus; dst++) {
    //         for (int src = 0; src < ngpus; src++) {
    //             if (src == dst) continue;
    //             int idx = pair_index(dst, src);
    //             ghost_pair_nblocks[idx] = (int)h_ghost_asw_blocks[idx].size();
    //         }
    //     }
    // }

    // void build_ghost_maps(bool debug_print = true) {
    //     int total_candidates = 0;
    //     int total_already_local = 0;
    //     int total_src_nodes_found = 0;
    //     int total_src_block_found = 0;
    //     int total_pushed = 0;

    //     for (int dst = 0; dst < ngpus; dst++) {
    //         int *dst_conn = A->getHostLocalElemConn(dst);

    //         for (int e = 0; e < batch_size[dst]; e++) {
    //             for (int ij = 0; ij < size4; ij++) {
    //                 int i = ij % size2;
    //                 int j = ij / size2;

    //                 int dst_row_node = dst_conn[e * size2 + i];
    //                 int dst_col_node = dst_conn[e * size2 + j];

    //                 if (dst_row_node < 0 || dst_col_node < 0) continue;

    //                 int glob_row = part->h_local_nodes[dst][dst_row_node];
    //                 int glob_col = part->h_local_nodes[dst][dst_col_node];

    //                 // bool dst_row_is_ghost = (part->h_node_gpu_ind[glob_row] != dst);
    //                 // bool dst_col_is_ghost = (part->h_node_gpu_ind[glob_col] != dst);

    //                 // not if it's owned, but if this node also appears on other proc is ghost
    //                 here bool dst_row_is_ghost = part->h_is_local_ghost[dst][dst_row_node]; bool
    //                 dst_col_is_ghost = part->h_is_local_ghost[dst][dst_col_node];

    //                 // Only destination ghost x ghost blocks
    //                 if (!dst_row_is_ghost || !dst_col_is_ghost) continue;

    //                 total_candidates++;

    //                 int dst_batch_block = e * size4 + ij;
    //                 bool already_local = (h_block_inds[dst][dst_batch_block] >= 0);
    //                 if (already_local) total_already_local++;

    //                 bool found_src_nodes = false;
    //                 bool found_src_block = false;

    //                 for (int src = 0; src < ngpus; src++) {
    //                     if (src == dst) continue;

    //                     int *src_rowp = A->getHostLocalRowp(src);
    //                     int *src_cols = A->getHostLocalCols(src);

    //                     int src_row_node = -1;
    //                     int src_col_node = -1;

    //                     for (int a = 0; a < part->local_nnodes[src]; a++) {
    //                         int gnode = part->h_local_nodes[src][a];

    //                         if (gnode == glob_row) src_row_node = a;
    //                         if (gnode == glob_col) src_col_node = a;

    //                         if (src_row_node >= 0 && src_col_node >= 0) break;
    //                     }

    //                     if (src_row_node < 0 || src_col_node < 0) continue;
    //                     found_src_nodes = true;

    //                     int jp_found = -1;
    //                     for (int jp = src_rowp[src_row_node]; jp < src_rowp[src_row_node + 1];
    //                          jp++) {
    //                         if (src_cols[jp] == src_col_node) {
    //                             jp_found = jp;
    //                             break;
    //                         }
    //                     }

    //                     if (jp_found < 0) continue;
    //                     found_src_block = true;

    //                     int idx = pair_index(dst, src);
    //                     h_ghost_asw_blocks[idx].push_back(dst_batch_block);
    //                     h_ghost_kmat_blocks[idx].push_back(jp_found);
    //                     total_pushed++;

    //                     break;
    //                 }

    //                 if (found_src_nodes) total_src_nodes_found++;
    //                 if (found_src_block) total_src_block_found++;
    //             }
    //         }
    //     }

    //     for (int dst = 0; dst < ngpus; dst++) {
    //         for (int src = 0; src < ngpus; src++) {
    //             if (src == dst) continue;

    //             int idx = pair_index(dst, src);
    //             ghost_pair_nblocks[idx] = (int)h_ghost_asw_blocks[idx].size();

    //             if (debug_print && ghost_pair_nblocks[idx] > 0) {
    //                 printf("[ghost-map] dst %d <- src %d : %d blocks\n", dst, src,
    //                        ghost_pair_nblocks[idx]);
    //             }
    //         }
    //     }

    //     if (debug_print) {
    //         printf(
    //             "[ghost-map] candidates=%d already_local=%d "
    //             "src_nodes_found=%d src_block_found=%d pushed=%d\n",
    //             total_candidates, total_already_local, total_src_nodes_found,
    //             total_src_block_found, total_pushed);
    //     }
    // }

    void build_ghost_maps() {
        // global_to_local[g][global_node] = local node index on GPU g, or -1
        int **global_to_local = new int *[ngpus];

        for (int g = 0; g < ngpus; g++) {
            global_to_local[g] = new int[part->num_nodes];
            std::fill(global_to_local[g], global_to_local[g] + part->num_nodes, -1);

            for (int loc = 0; loc < part->local_nnodes[g]; loc++) {
                int gnode = part->h_local_nodes[g][loc];
                global_to_local[g][gnode] = loc;
            }
        }

        for (int dst = 0; dst < ngpus; dst++) {
            int *dst_conn = A->getHostLocalElemConn(dst);

            for (int e = 0; e < batch_size[dst]; e++) {
                for (int ij = 0; ij < size4; ij++) {
                    int i = ij % size2;
                    int j = ij / size2;

                    int dst_row_node = dst_conn[e * size2 + i];
                    int dst_col_node = dst_conn[e * size2 + j];

                    if (dst_row_node < 0 || dst_col_node < 0) continue;

                    int glob_row = part->h_local_nodes[dst][dst_row_node];
                    int glob_col = part->h_local_nodes[dst][dst_col_node];

                    bool dst_row_is_ghost = part->h_is_local_ghost[dst][dst_row_node];
                    bool dst_col_is_ghost = part->h_is_local_ghost[dst][dst_col_node];

                    // Only destination ghost x ghost blocks
                    if (!dst_row_is_ghost || !dst_col_is_ghost) continue;

                    int dst_batch_block = e * size4 + ij;

                    for (int src = 0; src < ngpus; src++) {
                        if (src == dst) continue;

                        int src_row_node = global_to_local[src][glob_row];
                        int src_col_node = global_to_local[src][glob_col];

                        if (src_row_node < 0 || src_col_node < 0) continue;

                        int *src_rowp = A->getHostLocalRowp(src);
                        int *src_cols = A->getHostLocalCols(src);

                        int jp_found = -1;
                        for (int jp = src_rowp[src_row_node]; jp < src_rowp[src_row_node + 1];
                             jp++) {
                            if (src_cols[jp] == src_col_node) {
                                jp_found = jp;
                                break;
                            }
                        }

                        if (jp_found < 0) continue;

                        int idx = pair_index(dst, src);
                        h_ghost_asw_blocks[idx].push_back(dst_batch_block);
                        h_ghost_kmat_blocks[idx].push_back(jp_found);

                        break;
                    }
                }
            }
        }

        for (int dst = 0; dst < ngpus; dst++) {
            for (int src = 0; src < ngpus; src++) {
                if (src == dst) continue;

                int idx = pair_index(dst, src);
                ghost_pair_nblocks[idx] = static_cast<int>(h_ghost_asw_blocks[idx].size());
            }
        }

        for (int g = 0; g < ngpus; g++) {
            delete[] global_to_local[g];
        }
        delete[] global_to_local;
    }

    void add_ghost_ghost_blocks_to_batched() {
        // 1) Pack source K blocks on src GPU
        for (int dst = 0; dst < ngpus; dst++) {
            for (int src = 0; src < ngpus; src++) {
                if (src == dst) continue;

                int idx = pair_index(dst, src);
                int nb = ghost_pair_nblocks[idx];
                if (nb == 0) continue;

                int nvals = nb * block_dim2;

                CHECK_CUDA(cudaSetDevice(debug ? 0 : src));

                dim3 block(128);
                dim3 grid((nvals + block.x - 1) / block.x);

                k_packGhostGhostMatBlocks<T>
                    <<<grid, block, 0, streams[src]>>>(nvals, block_dim, d_ghost_kmat_blocks[idx],
                                                       A->getLocalVals(src), d_ghost_vals_red[idx]);

                CHECK_CUDA(cudaGetLastError());
            }
        }

        // 2) Peer-copy src packed values to dst
        for (int dst = 0; dst < ngpus; dst++) {
            for (int src = 0; src < ngpus; src++) {
                if (src == dst) continue;

                int idx = pair_index(dst, src);
                int nb = ghost_pair_nblocks[idx];
                if (nb == 0) continue;

                size_t bytes = (size_t)nb * block_dim2 * sizeof(T);

                CHECK_CUDA(cudaSetDevice(debug ? 0 : src));

                if (debug) {
                    CHECK_CUDA(cudaMemcpyAsync(d_ghost_vals_red_dst[idx], d_ghost_vals_red[idx],
                                               bytes, cudaMemcpyDeviceToDevice, streams[src]));
                } else {
                    CHECK_CUDA(cudaMemcpyPeerAsync(d_ghost_vals_red_dst[idx], dst,
                                                   d_ghost_vals_red[idx], src, bytes,
                                                   streams[src]));
                }
            }
        }

        ctx->sync();

        // 3) Add received ghost x ghost blocks into dst batched matrices
        for (int dst = 0; dst < ngpus; dst++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : dst));

            for (int src = 0; src < ngpus; src++) {
                if (src == dst) continue;

                int idx = pair_index(dst, src);
                int nb = ghost_pair_nblocks[idx];
                if (nb == 0) continue;

                int nvals = nb * block_dim2;

                dim3 block(128);
                dim3 grid((nvals + block.x - 1) / block.x);

                k_addGhostGhostMatBlocksToBatched<T><<<grid, block, 0, streams[dst]>>>(
                    nvals, block_dim, size, d_ghost_asw_blocks[idx], d_ghost_vals_red_dst[idx],
                    d_Adata[dst]);

                CHECK_CUDA(cudaGetLastError());
            }
        }

        ctx->sync();
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

    void move_ghost_maps_to_device() {
        for (int dst = 0; dst < ngpus; dst++) {
            for (int src = 0; src < ngpus; src++) {
                if (src == dst) continue;

                int idx = pair_index(dst, src);
                int nb = ghost_pair_nblocks[idx];
                if (nb == 0) continue;

                CHECK_CUDA(cudaSetDevice(debug ? 0 : dst));
                CHECK_CUDA(cudaMalloc((void **)&d_ghost_asw_blocks[idx], nb * sizeof(int)));
                CHECK_CUDA(cudaMemcpyAsync(d_ghost_asw_blocks[idx], h_ghost_asw_blocks[idx].data(),
                                           nb * sizeof(int), cudaMemcpyHostToDevice, streams[dst]));

                CHECK_CUDA(cudaSetDevice(debug ? 0 : src));
                CHECK_CUDA(cudaMalloc((void **)&d_ghost_kmat_blocks[idx], nb * sizeof(int)));
                CHECK_CUDA(cudaMemcpyAsync(d_ghost_kmat_blocks[idx],
                                           h_ghost_kmat_blocks[idx].data(), nb * sizeof(int),
                                           cudaMemcpyHostToDevice, streams[src]));
            }
        }

        ctx->sync();
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

    void allocate_ghost_batched_memory() {
        for (int dst = 0; dst < ngpus; dst++) {
            for (int src = 0; src < ngpus; src++) {
                if (src == dst) continue;

                int idx = pair_index(dst, src);
                int nb = ghost_pair_nblocks[idx];
                if (nb == 0) continue;

                size_t bytes = (size_t)nb * block_dim2 * sizeof(T);

                CHECK_CUDA(cudaSetDevice(debug ? 0 : src));
                CHECK_CUDA(cudaMalloc((void **)&d_ghost_vals_red[idx], bytes));

                CHECK_CUDA(cudaSetDevice(debug ? 0 : dst));
                CHECK_CUDA(cudaMalloc((void **)&d_ghost_vals_red_dst[idx], bytes));
            }
        }

        ctx->sync();
    }
};