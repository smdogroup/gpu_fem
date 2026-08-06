#pragma once
#include <cmath>
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

        printf("\n========== MultiGPUElementASW constructor START ==========\n");
        printf(
            "[ASW ctor] ngpus=%d block_dim=%d block_dim2=%d size=%d size2=%d size4=%d n=%d "
            "debug=%d\n",
            ngpus, block_dim, block_dim2, size, size2, size4, n, (int)debug);
        check_all_devices_for_stale_error("constructor entry", true);

        printf("ASW - allocate_arrays\n");
        allocate_arrays();
        check_all_devices_for_stale_error("after allocate_arrays", true);

        printf("ASW - build_maps\n");
        build_maps();
        check_all_devices_for_stale_error("after build_maps", true);

        printf("ASW - build_ghost_maps\n");
        build_ghost_maps();
        check_all_devices_for_stale_error("after build_ghost_maps", true);

        printf("ASW - allocate_batched_memory\n");
        allocate_batched_memory();

        printf("ASW - allocate_ghost_batched_memory\n");
        allocate_ghost_batched_memory();

        printf("ASW - move_maps_to_device\n");
        move_maps_to_device();

        printf("ASW - move_ghost_maps_to_device\n");
        move_ghost_maps_to_device();

        printf("ASW - create temp/defect vectors\n");
        temp = new GPUvec<T, Partitioner>(ctx, part, block_dim);
        defect = new GPUvec<T, Partitioner>(ctx, part, block_dim);

        ctx->sync();
        check_all_devices_for_stale_error("constructor final ctx sync", true);
        printf("========== MultiGPUElementASW constructor DONE ==========\n\n");
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

        if (temp) temp->free();
        if (defect) defect->free();

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
        printf("\n[ASW factor] START\n");
        check_all_devices_for_stale_error("factor entry", true);

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            CHECK_CUBLAS(cublasSetStream(cublasHandles[g], streams[g]));

            printf("[ASW factor] GPU[%d] memset d_Adata batch_size=%d n=%d\n", g, batch_size[g], n);
            CHECK_CUDA(cudaMemsetAsync(d_Adata[g], 0, (size_t)batch_size[g] * n * n * sizeof(T),
                                       streams[g]));

            int nvals = n_batch_blocks[g] * block_dim2;
            dim3 block(128);
            dim3 grid((nvals + block.x - 1) / block.x);

            printf("[ASW factor] GPU[%d] copyMat nvals=%d grid=%u block=%u\n", g, nvals, grid.x,
                   block.x);

            if (nvals > 0) {
                k_copyMatValuesToBatchedContiguous<T><<<grid, block, 0, streams[g]>>>(
                    nvals, block_dim, size, d_block_inds[g], A->getLocalVals(g), d_Adata[g]);
                CHECK_CUDA(cudaGetLastError());
            }
        }

        ctx->sync();
        check_all_devices_for_stale_error("factor after local copy", true);

        add_ghost_ghost_blocks_to_batched();

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            CHECK_CUBLAS(cublasSetStream(cublasHandles[g], streams[g]));

            if (batch_size[g] <= 0) continue;

            printf("[ASW factor] GPU[%d] getrf/getri batch_size=%d n=%d\n", g, batch_size[g], n);

            CHECK_CUBLAS(cublasDgetrfBatched(cublasHandles[g], n, d_Aarray[g], n, d_PivotArray[g],
                                             d_InfoArray[g], batch_size[g]));

            CHECK_CUBLAS(cublasDgetriBatched(cublasHandles[g], n, (const double **)d_Aarray[g], n,
                                             d_PivotArray[g], d_invAarray[g], n, d_InfoArray[g],
                                             batch_size[g]));
        }

        ctx->sync();
        check_all_devices_for_stale_error("factor exit", true);
        printf("[ASW factor] DONE\n\n");
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

                if (nrhs_vals > 0) {
                    k_copyLocalRHSIntoBatched<T><<<grid, block, 0, streams[g]>>>(
                        nrhs_vals, block_dim, size, d_rhs_local_map[g], def->getLocalPtr(g),
                        d_Xarray[g]);
                    CHECK_CUDA(cudaGetLastError());
                }

                const double alpha = 1.0;
                const double beta = 0.0;

                if (batch_size[g] > 0) {
                    CHECK_CUBLAS(cublasDgemmBatched(cublasHandles[g], CUBLAS_OP_N, CUBLAS_OP_N, n,
                                                    1, n, &alpha, (const double **)d_invAarray[g],
                                                    n, (const double **)d_Xarray[g], n, &beta,
                                                    d_Yarray[g], n, batch_size[g]));
                }

                if (nrhs_vals > 0) {
                    k_addBatchedIntoLocalSoln<T><<<grid, block, 0, streams[g]>>>(
                        nrhs_vals, block_dim, size, d_rhs_local_map[g], d_Yarray[g],
                        temp->getLocalPtr(g));
                    CHECK_CUDA(cudaGetLastError());
                }
            }

            ctx->sync();

            temp->reduceFromLocal();

            T minus_omega = -omega;
            T one = 1.0;
            A->mult(minus_omega, temp, one, def);

            soln->axpy(omega, temp);
        }
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

    std::vector<int> *h_ghost_asw_blocks = nullptr;
    std::vector<int> *h_ghost_kmat_blocks = nullptr;
    int **d_ghost_asw_blocks = nullptr;
    int **d_ghost_kmat_blocks = nullptr;
    T **d_ghost_vals_red = nullptr;
    T **d_ghost_vals_red_dst = nullptr;

    GPUvec<T, Partitioner> *temp = nullptr;
    GPUvec<T, Partitioner> *defect = nullptr;

    int pair_index(int dst, int src) const { return ngpus * dst + src; }

    void check_all_devices_for_stale_error(const char *stage, bool clear) {
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));
            cudaError_t err = clear ? cudaGetLastError() : cudaPeekAtLastError();
            if (err != cudaSuccess) {
                printf("[ASW CUDA ERROR] %s GPU[%d] dev=%d error=%s\n", stage, g, debug ? 0 : g,
                       cudaGetErrorString(err));
                exit(1);
            }
        }
    }

    void allocate_arrays() {
        printf("[ASW allocate_arrays] START ngpus=%d\n", ngpus);

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

        printf("[ASW allocate_arrays] allocated host pointer arrays npairs=%d\n", npairs);
        printf("[ASW allocate_arrays] DONE\n");
    }

    void build_maps() {
        printf("[ASW build_maps] START\n");

        int total_missing_blocks = 0;
        int total_bad_conn = 0;

        for (int g = 0; g < ngpus; g++) {
            printf("[ASW build_maps] GPU[%d] start\n", g);

            batch_size[g] = part->local_nelems[g];
            n_batch_blocks[g] = batch_size[g] * size4;
            n_rhs_blocks[g] = batch_size[g] * size2;

            printf(
                "[ASW build_maps] GPU[%d] batch_size=%d n_batch_blocks=%d n_rhs_blocks=%d "
                "part local_nnodes=%d owned_nnodes=%d\n",
                g, batch_size[g], n_batch_blocks[g], n_rhs_blocks[g], part->local_nnodes[g],
                part->owned_nnodes[g]);

            h_block_inds[g] = new int[n_batch_blocks[g]];
            h_rhs_local_map[g] = new int[n_rhs_blocks[g]];
            h_rhs_owned_map[g] = new int[n_rhs_blocks[g]];

            std::fill(h_block_inds[g], h_block_inds[g] + n_batch_blocks[g], -1);
            std::fill(h_rhs_local_map[g], h_rhs_local_map[g] + n_rhs_blocks[g], -1);
            std::fill(h_rhs_owned_map[g], h_rhs_owned_map[g] + n_rhs_blocks[g], -1);

            int *loc_conn = A->getHostLocalElemConn(g);
            int *local_rowp = A->getHostLocalRowp(g);
            int *local_cols = A->getHostLocalCols(g);

            printf("[ASW build_maps] GPU[%d] ptrs loc_conn=%p rowp=%p cols=%p\n", g,
                   (void *)loc_conn, (void *)local_rowp, (void *)local_cols);

            if (!loc_conn || !local_rowp || !local_cols) {
                printf("[ASW build_maps] ERROR GPU[%d] null local arrays\n", g);
                exit(1);
            }

            int missing_blocks_g = 0;
            int bad_conn_g = 0;

            for (int e = 0; e < batch_size[g]; e++) {
                for (int ij = 0; ij < size4; ij++) {
                    int i = ij % size2;
                    int j = ij / size2;

                    int row_node = loc_conn[e * size2 + i];
                    int col_node = loc_conn[e * size2 + j];

                    int jp_found = -1;

                    if (row_node < 0 || row_node >= part->local_nnodes[g] || col_node < 0 ||
                        col_node >= part->local_nnodes[g]) {
                        bad_conn_g++;
                        if (bad_conn_g <= 20) {
                            printf(
                                "[ASW build_maps] BAD CONN GPU[%d] e=%d ij=%d row_node=%d "
                                "col_node=%d local_nnodes=%d\n",
                                g, e, ij, row_node, col_node, part->local_nnodes[g]);
                        }
                    } else {
                        for (int jp = local_rowp[row_node]; jp < local_rowp[row_node + 1]; jp++) {
                            if (local_cols[jp] == col_node) {
                                jp_found = jp;
                                break;
                            }
                        }

                        if (jp_found < 0) {
                            missing_blocks_g++;
                            if (missing_blocks_g <= 20) {
                                int grow = part->h_local_nodes[g][row_node];
                                int gcol = part->h_local_nodes[g][col_node];
                                printf(
                                    "[ASW build_maps] MISSING BLOCK GPU[%d] e=%d ij=%d "
                                    "row_node=%d col_node=%d glob=(%d,%d) rowp=[%d,%d)\n",
                                    g, e, ij, row_node, col_node, grow, gcol, local_rowp[row_node],
                                    local_rowp[row_node + 1]);
                            }
                        }
                    }

                    h_block_inds[g][e * size4 + ij] = jp_found;
                }

                for (int a = 0; a < size2; a++) {
                    int ind = e * size2 + a;
                    int loc = loc_conn[e * size2 + a];

                    if (loc < 0 || loc >= part->local_nnodes[g]) {
                        printf(
                            "[ASW build_maps] BAD RHS MAP GPU[%d] e=%d a=%d loc=%d "
                            "local_nnodes=%d\n",
                            g, e, a, loc, part->local_nnodes[g]);
                    }

                    h_rhs_local_map[g][ind] = loc;
                    h_rhs_owned_map[g][ind] = -1;
                }
            }

            total_missing_blocks += missing_blocks_g;
            total_bad_conn += bad_conn_g;

            printf("[ASW build_maps] GPU[%d] done missing_blocks=%d bad_conn=%d\n", g,
                   missing_blocks_g, bad_conn_g);
        }

        printf("[ASW build_maps] DONE total_missing_blocks=%d total_bad_conn=%d\n",
               total_missing_blocks, total_bad_conn);
    }

    void build_ghost_maps() {
        printf("[ASW build_ghost_maps] START\n");

        int **global_to_local = new int *[ngpus];

        for (int g = 0; g < ngpus; g++) {
            printf(
                "[ASW build_ghost_maps] building global_to_local GPU[%d] num_nodes=%d "
                "local_nnodes=%d\n",
                g, part->num_nodes, part->local_nnodes[g]);

            global_to_local[g] = new int[part->num_nodes];
            std::fill(global_to_local[g], global_to_local[g] + part->num_nodes, -1);

            for (int loc = 0; loc < part->local_nnodes[g]; loc++) {
                int gnode = part->h_local_nodes[g][loc];

                if (gnode < 0 || gnode >= part->num_nodes) {
                    printf(
                        "[ASW build_ghost_maps] BAD local node GPU[%d] loc=%d gnode=%d "
                        "num_nodes=%d\n",
                        g, loc, gnode, part->num_nodes);
                    exit(1);
                }

                global_to_local[g][gnode] = loc;
            }
        }

        int total_candidates = 0;
        int total_src_nodes_found = 0;
        int total_src_block_found = 0;
        int total_pushed = 0;
        int total_bad_dst_conn = 0;

        for (int dst = 0; dst < ngpus; dst++) {
            int *dst_conn = A->getHostLocalElemConn(dst);

            printf("[ASW build_ghost_maps] dst=%d batch_size=%d dst_conn=%p local_nnodes=%d\n", dst,
                   batch_size[dst], (void *)dst_conn, part->local_nnodes[dst]);

            if (!dst_conn) {
                printf("[ASW build_ghost_maps] ERROR dst=%d null dst_conn\n", dst);
                exit(1);
            }

            int dst_candidates = 0;
            int dst_pushed = 0;
            int dst_bad_conn = 0;

            for (int e = 0; e < batch_size[dst]; e++) {
                for (int ij = 0; ij < size4; ij++) {
                    int i = ij % size2;
                    int j = ij / size2;

                    int dst_row_node = dst_conn[e * size2 + i];
                    int dst_col_node = dst_conn[e * size2 + j];

                    if (dst_row_node < 0 || dst_row_node >= part->local_nnodes[dst] ||
                        dst_col_node < 0 || dst_col_node >= part->local_nnodes[dst]) {
                        dst_bad_conn++;
                        total_bad_dst_conn++;
                        if (dst_bad_conn <= 20) {
                            printf(
                                "[ASW build_ghost_maps] BAD DST CONN dst=%d e=%d ij=%d "
                                "row=%d col=%d local_nnodes=%d\n",
                                dst, e, ij, dst_row_node, dst_col_node, part->local_nnodes[dst]);
                        }
                        continue;
                    }

                    int glob_row = part->h_local_nodes[dst][dst_row_node];
                    int glob_col = part->h_local_nodes[dst][dst_col_node];

                    bool dst_row_is_ghost = part->h_is_local_ghost[dst][dst_row_node];
                    bool dst_col_is_ghost = part->h_is_local_ghost[dst][dst_col_node];

                    if (!dst_row_is_ghost || !dst_col_is_ghost) continue;

                    int dst_batch_block = e * size4 + ij;

                    total_candidates++;
                    dst_candidates++;

                    bool found_src_nodes = false;
                    bool found_src_block = false;

                    for (int src = 0; src < ngpus; src++) {
                        if (src == dst) continue;

                        int src_row_node = global_to_local[src][glob_row];
                        int src_col_node = global_to_local[src][glob_col];

                        if (src_row_node < 0 || src_col_node < 0) continue;

                        found_src_nodes = true;

                        int *src_rowp = A->getHostLocalRowp(src);
                        int *src_cols = A->getHostLocalCols(src);

                        if (!src_rowp || !src_cols) {
                            printf("[ASW build_ghost_maps] ERROR src=%d null rowp/cols\n", src);
                            exit(1);
                        }

                        int jp_found = -1;
                        for (int jp = src_rowp[src_row_node]; jp < src_rowp[src_row_node + 1];
                             jp++) {
                            if (src_cols[jp] == src_col_node) {
                                jp_found = jp;
                                break;
                            }
                        }

                        if (jp_found < 0) continue;

                        found_src_block = true;

                        int idx = pair_index(dst, src);
                        h_ghost_asw_blocks[idx].push_back(dst_batch_block);
                        h_ghost_kmat_blocks[idx].push_back(jp_found);

                        total_pushed++;
                        dst_pushed++;
                        break;
                    }

                    if (found_src_nodes) total_src_nodes_found++;
                    if (found_src_block) total_src_block_found++;

                    if (!found_src_block && dst_candidates <= 40) {
                        printf(
                            "[ASW build_ghost_maps] NO SRC BLOCK dst=%d e=%d ij=%d "
                            "glob=(%d,%d) dst_locs=(%d,%d) ghost=(%d,%d)\n",
                            dst, e, ij, glob_row, glob_col, dst_row_node, dst_col_node,
                            (int)dst_row_is_ghost, (int)dst_col_is_ghost);
                    }
                }
            }

            printf("[ASW build_ghost_maps] dst=%d candidates=%d pushed=%d bad_conn=%d\n", dst,
                   dst_candidates, dst_pushed, dst_bad_conn);
        }

        for (int dst = 0; dst < ngpus; dst++) {
            for (int src = 0; src < ngpus; src++) {
                if (src == dst) continue;

                int idx = pair_index(dst, src);
                ghost_pair_nblocks[idx] = static_cast<int>(h_ghost_asw_blocks[idx].size());

                printf("[ASW build_ghost_maps] pair dst=%d src=%d nb=%d\n", dst, src,
                       ghost_pair_nblocks[idx]);
            }
        }

        for (int g = 0; g < ngpus; g++) {
            delete[] global_to_local[g];
        }
        delete[] global_to_local;

        printf(
            "[ASW build_ghost_maps] DONE candidates=%d src_nodes_found=%d "
            "src_block_found=%d pushed=%d bad_dst_conn=%d\n",
            total_candidates, total_src_nodes_found, total_src_block_found, total_pushed,
            total_bad_dst_conn);
    }

    void allocate_batched_memory() {
        printf("\n[ASW] allocate_batched_memory START\n");
        printf("[ASW] ngpus=%d block_dim=%d n=%d debug=%d\n", ngpus, block_dim, n, (int)debug);

        for (int g = 0; g < ngpus; g++) {
            int dev = debug ? 0 : g;

            printf("\n[ASW] GPU[%d] dev=%d entering\n", g, dev);
            printf("[ASW] GPU[%d] batch_size=%d n_batch_blocks=%d n_rhs_blocks=%d\n", g,
                   batch_size[g], n_batch_blocks[g], n_rhs_blocks[g]);

            CHECK_CUDA(cudaSetDevice(dev));

            cudaError_t pre_err = cudaGetLastError();
            if (pre_err != cudaSuccess) {
                printf("[ASW] GPU[%d] pre-existing CUDA error: %s\n", g,
                       cudaGetErrorString(pre_err));
                exit(1);
            }

            size_t mat_bytes = (size_t)batch_size[g] * n * n * sizeof(T);
            size_t vec_bytes = (size_t)batch_size[g] * n * sizeof(T);
            size_t ptr_bytes = (size_t)batch_size[g] * sizeof(T *);

            printf("[ASW] GPU[%d] mat_bytes=%zu vec_bytes=%zu ptr_bytes=%zu\n", g, mat_bytes,
                   vec_bytes, ptr_bytes);

            if (batch_size[g] <= 0) {
                printf("[ASW] GPU[%d] WARNING batch_size <= 0, skipping allocation/kernel\n", g);
                continue;
            }

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

            printf("[ASW] GPU[%d] launch k_setupBatchedPointers grid=%u block=%u stream=%p\n", g,
                   grid.x, block.x, (void *)streams[g]);

            k_setupBatchedPointers<T><<<grid, block, 0, streams[g]>>>(
                batch_size[g], n, d_Adata[g], d_invAdata[g], d_Xdata[g], d_Ydata[g], d_Aarray[g],
                d_invAarray[g], d_Xarray[g], d_Yarray[g]);

            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaStreamSynchronize(streams[g]));

            printf("[ASW] GPU[%d] done\n", g);
        }

        ctx->sync();
        printf("[ASW] allocate_batched_memory DONE\n\n");
    }

    void allocate_ghost_batched_memory() {
        printf("[ASW allocate_ghost_batched_memory] START\n");

        for (int dst = 0; dst < ngpus; dst++) {
            for (int src = 0; src < ngpus; src++) {
                if (src == dst) continue;

                int idx = pair_index(dst, src);
                int nb = ghost_pair_nblocks[idx];

                printf("[ASW allocate_ghost] dst=%d src=%d idx=%d nb=%d\n", dst, src, idx, nb);

                if (nb == 0) continue;

                size_t bytes = (size_t)nb * block_dim2 * sizeof(T);

                CHECK_CUDA(cudaSetDevice(debug ? 0 : src));
                CHECK_CUDA(cudaMalloc((void **)&d_ghost_vals_red[idx], bytes));

                CHECK_CUDA(cudaSetDevice(debug ? 0 : dst));
                CHECK_CUDA(cudaMalloc((void **)&d_ghost_vals_red_dst[idx], bytes));
            }
        }

        ctx->sync();
        printf("[ASW allocate_ghost_batched_memory] DONE\n");
    }

    void move_maps_to_device() {
        printf("[ASW move_maps_to_device] START\n");

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(debug ? 0 : g));

            printf("[ASW move_maps] GPU[%d] n_batch_blocks=%d n_rhs_blocks=%d\n", g,
                   n_batch_blocks[g], n_rhs_blocks[g]);

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

            CHECK_CUDA(cudaStreamSynchronize(streams[g]));
        }

        printf("[ASW move_maps_to_device] DONE\n");
    }

    void move_ghost_maps_to_device() {
        printf("[ASW move_ghost_maps_to_device] START\n");

        for (int dst = 0; dst < ngpus; dst++) {
            for (int src = 0; src < ngpus; src++) {
                if (src == dst) continue;

                int idx = pair_index(dst, src);
                int nb = ghost_pair_nblocks[idx];

                printf("[ASW move_ghost_maps] dst=%d src=%d idx=%d nb=%d\n", dst, src, idx, nb);

                if (nb == 0) continue;

                CHECK_CUDA(cudaSetDevice(debug ? 0 : dst));
                CHECK_CUDA(cudaMalloc((void **)&d_ghost_asw_blocks[idx], nb * sizeof(int)));
                CHECK_CUDA(cudaMemcpyAsync(d_ghost_asw_blocks[idx], h_ghost_asw_blocks[idx].data(),
                                           nb * sizeof(int), cudaMemcpyHostToDevice, streams[dst]));
                CHECK_CUDA(cudaStreamSynchronize(streams[dst]));

                CHECK_CUDA(cudaSetDevice(debug ? 0 : src));
                CHECK_CUDA(cudaMalloc((void **)&d_ghost_kmat_blocks[idx], nb * sizeof(int)));
                CHECK_CUDA(cudaMemcpyAsync(d_ghost_kmat_blocks[idx],
                                           h_ghost_kmat_blocks[idx].data(), nb * sizeof(int),
                                           cudaMemcpyHostToDevice, streams[src]));
                CHECK_CUDA(cudaStreamSynchronize(streams[src]));
            }
        }

        ctx->sync();
        printf("[ASW move_ghost_maps_to_device] DONE\n");
    }

    void add_ghost_ghost_blocks_to_batched() {
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
};