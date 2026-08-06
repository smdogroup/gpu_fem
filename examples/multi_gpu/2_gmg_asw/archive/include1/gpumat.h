#pragma once
#include <algorithm>
#include <cstring>
#include <unordered_set>
#include <vector>

// dependency to smdogroup/sparse-utils repo
#include "cuda_utils.h"
#include "gpuvec.h"
#include "matvec.cuh"
#include "multigpu_context.h"

// in order to work with cusparse index_t has to be int
typedef int index_t;

#include "sparse_utils/sparse_matrix.h"
#include "sparse_utils/sparse_symbolic.h"

template <typename T, class Partitioner>
class GPUbsrmat {
   public:
    GPUbsrmat(MultiGPUContext *ctx_, const Partitioner *part_, int block_dim_ = 6)
        : ctx(ctx_),
          part(part_),
          cublasHandles(ctx_->cublasHandles),
          cusparseHandles(ctx_->cusparseHandles),
          streams(ctx_->streams),
          ngpus(part_->ngpus),
          num_nodes(part_->num_nodes),
          block_dim(block_dim_),
          block_dim2(block_dim_ * block_dim_) {
        build_reduced_element_connectivity();
        build_element_bsr_sparsity();
        build_transpose_pattern();
        build_element_ind_map();
        move_matrix_pattern_to_device();
        sync();
    }

    ~GPUbsrmat() {
        sync();

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));

            if (d_row_red_elem_conn && d_row_red_elem_conn[g]) cudaFree(d_row_red_elem_conn[g]);
            if (d_col_red_elem_conn && d_col_red_elem_conn[g]) cudaFree(d_col_red_elem_conn[g]);
            if (d_loc_elem_ind_map && d_loc_elem_ind_map[g]) cudaFree(d_loc_elem_ind_map[g]);

            if (d_loc_rowp && d_loc_rowp[g]) cudaFree(d_loc_rowp[g]);
            if (d_loc_cols && d_loc_cols[g]) cudaFree(d_loc_cols[g]);
            if (d_loc_vals && d_loc_vals[g]) cudaFree(d_loc_vals[g]);

            if (h_row_red_elem_conn && h_row_red_elem_conn[g]) delete[] h_row_red_elem_conn[g];
            if (h_col_red_elem_conn && h_col_red_elem_conn[g]) delete[] h_col_red_elem_conn[g];
            if (h_loc_elem_ind_map && h_loc_elem_ind_map[g]) delete[] h_loc_elem_ind_map[g];

            if (h_loc_rowp && h_loc_rowp[g]) delete[] h_loc_rowp[g];
            if (h_loc_cols && h_loc_cols[g]) delete[] h_loc_cols[g];
            if (h_loc_vals && h_loc_vals[g]) delete[] h_loc_vals[g];

            if (descrA && descrA[g]) cusparseDestroyMatDescr(descrA[g]);
        }

        delete[] elem_conn_N;
        delete[] elem_ind_map_N;

        delete[] h_row_red_elem_conn;
        delete[] h_col_red_elem_conn;
        delete[] h_loc_elem_ind_map;

        delete[] d_row_red_elem_conn;
        delete[] d_col_red_elem_conn;
        delete[] d_loc_elem_ind_map;

        delete[] loc_mb;
        delete[] loc_nb;
        delete[] loc_nnzb;
        delete[] loc_nnz;

        delete[] h_loc_rowp;
        delete[] h_loc_cols;
        delete[] h_loc_vals;

        delete[] d_loc_rowp;
        delete[] d_loc_cols;
        delete[] d_loc_vals;

        delete[] descrA;
    }

    void apply_bcs(int *n_owned_bcs, int **d_owned_bcs, int *n_local_bcs, int **d_local_bcs) {
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            CHECK_CUBLAS(cublasSetStream(cublasHandles[g], streams[g]));

            // TODO : add perm + iperm into the matrices later..
            printf("GPU[%d] - apply row bcs to matrix\n", g);
            dim3 block(32);
            if (n_owned_bcs[g] > 0) {
                dim3 grid1((n_owned_bcs[g] + 31) / 32);
                k_mat_apply_row_bcs<T><<<grid1, block, 0, streams[g]>>>(
                    block_dim, loc_mb[g], n_owned_bcs[g], d_owned_bcs[g],
                    part->d_owned_to_local_map[g], d_loc_rowp[g], d_loc_cols[g], d_loc_vals[g]);

                CHECK_CUDA(cudaGetLastError());
            }

            printf("GPU[%d] - apply col bcs to matrix\n", g);
            if (n_local_bcs[g] > 0) {
                dim3 grid2((n_local_bcs[g] + 31) / 32);
                k_mat_apply_col_bcs<T><<<grid2, block, 0, streams[g]>>>(
                    block_dim, loc_nb[g], n_local_bcs[g], d_local_bcs[g],
                    part->d_local_to_owned_map[g], d_tr_loc_rowp[g], d_tr_loc_cols[g],
                    d_tr_block_map[g], d_loc_vals[g]);

                CHECK_CUDA(cudaGetLastError());
            }
        }
        ctx->sync();
    }

    void zeroValues() {
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            CHECK_CUDA(cudaMemsetAsync(d_loc_vals[g], 0, loc_nnz[g] * sizeof(T), streams[g]));
        }

        sync();
    }

    void mult(T alpha, GPUvec<T, Partitioner> *x, T beta, GPUvec<T, Partitioner> *y) {
        printf("before expandToLocal in mult\n");
        x->expandToLocal();
        printf("after expandToLocal in mult\n");

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            CHECK_CUSPARSE(cusparseSetStream(cusparseHandles[g], streams[g]));

            CHECK_CUSPARSE(cusparseDbsrmv(
                cusparseHandles[g], CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
                loc_mb[g], loc_nb[g], loc_nnzb[g], &alpha, descrA[g], d_loc_vals[g], d_loc_rowp[g],
                d_loc_cols[g], block_dim, x->getLocalPtr(g), &beta, y->getPtr(g)));
        }

        sync();
    }

    void mult(GPUvec<T, Partitioner> *x, GPUvec<T, Partitioner> *y) {
        T alpha = 1.0;
        T beta = 0.0;
        mult(alpha, x, beta, y);
    }

    int *getHostRowRedElemConn(int g) { return h_row_red_elem_conn[g]; }
    int *getHostColRedElemConn(int g) { return h_col_red_elem_conn[g]; }
    int *getHostLocalElemIndMap(int g) { return h_loc_elem_ind_map[g]; }

    int *getRowRedElemConn(int g) { return d_row_red_elem_conn[g]; }
    int *getColRedElemConn(int g) { return d_col_red_elem_conn[g]; }
    int *getLocalElemIndMap(int g) { return d_loc_elem_ind_map[g]; }

    int *getHostLocalRowp(int g) { return h_loc_rowp[g]; }
    int *getHostLocalCols(int g) { return h_loc_cols[g]; }
    int *getLocalRowp(int g) { return d_loc_rowp[g]; }
    int *getLocalCols(int g) { return d_loc_cols[g]; }
    T *getLocalVals(int g) { return d_loc_vals[g]; }

    int getLocalNumBlockRows(int g) const { return loc_mb[g]; }
    int getLocalNumBlockCols(int g) const { return loc_nb[g]; }
    int getLocalNumNonzeroBlocks(int g) const { return loc_nnzb[g]; }
    int getLocalNumNonzeros(int g) const { return loc_nnz[g]; }

    int getBlockDim() const { return block_dim; }
    int getBlockDim2() const { return block_dim2; }

    void sync() const {
        if (ctx) {
            ctx->sync();
            return;
        }

        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            CHECK_CUDA(cudaStreamSynchronize(streams[g]));
        }
    }

   private:
    void build_reduced_element_connectivity() {
        elem_conn_N = new int[ngpus];

        h_row_red_elem_conn = new int *[ngpus];
        h_col_red_elem_conn = new int *[ngpus];

        d_row_red_elem_conn = new int *[ngpus];
        d_col_red_elem_conn = new int *[ngpus];

        std::memset(elem_conn_N, 0, ngpus * sizeof(int));
        std::memset(h_row_red_elem_conn, 0, ngpus * sizeof(int *));
        std::memset(h_col_red_elem_conn, 0, ngpus * sizeof(int *));
        std::memset(d_row_red_elem_conn, 0, ngpus * sizeof(int *));
        std::memset(d_col_red_elem_conn, 0, ngpus * sizeof(int *));

        for (int g = 0; g < ngpus; g++) {
            int nelems = part->local_nelems[g];
            int npe = part->nodes_per_elem;
            int conn_size = nelems * npe;

            elem_conn_N[g] = conn_size;
            h_row_red_elem_conn[g] = new int[conn_size];
            h_col_red_elem_conn[g] = new int[conn_size];

            std::vector<int> global_to_owned(part->num_nodes, -1);
            std::vector<int> global_to_local(part->num_nodes, -1);

            for (int i = 0; i < part->owned_nnodes[g]; i++) {
                global_to_owned[part->h_owned_nodes[g][i]] = i;
            }

            for (int i = 0; i < part->local_nnodes[g]; i++) {
                global_to_local[part->h_local_nodes[g][i]] = i;
            }

            int *conn = part->h_local_elem_conn[g];

            for (int i = 0; i < conn_size; i++) {
                int node = conn[i];

                h_row_red_elem_conn[g][i] = global_to_owned[node];
                h_col_red_elem_conn[g][i] = global_to_local[node];
            }
        }
    }

    void build_element_bsr_sparsity() {
        loc_mb = new int[ngpus];
        loc_nb = new int[ngpus];
        loc_nnzb = new int[ngpus];
        loc_nnz = new int[ngpus];

        h_loc_rowp = new int *[ngpus];
        h_loc_cols = new int *[ngpus];
        h_loc_vals = new T *[ngpus];

        d_loc_rowp = new int *[ngpus];
        d_loc_cols = new int *[ngpus];
        d_loc_vals = new T *[ngpus];

        descrA = new cusparseMatDescr_t[ngpus];

        std::memset(loc_mb, 0, ngpus * sizeof(int));
        std::memset(loc_nb, 0, ngpus * sizeof(int));
        std::memset(loc_nnzb, 0, ngpus * sizeof(int));
        std::memset(loc_nnz, 0, ngpus * sizeof(int));

        std::memset(h_loc_rowp, 0, ngpus * sizeof(int *));
        std::memset(h_loc_cols, 0, ngpus * sizeof(int *));
        std::memset(h_loc_vals, 0, ngpus * sizeof(T *));

        std::memset(d_loc_rowp, 0, ngpus * sizeof(int *));
        std::memset(d_loc_cols, 0, ngpus * sizeof(int *));
        std::memset(d_loc_vals, 0, ngpus * sizeof(T *));

        for (int g = 0; g < ngpus; g++) {
            loc_mb[g] = part->owned_nnodes[g];
            loc_nb[g] = part->local_nnodes[g];

            std::vector<std::unordered_set<int>> row_cols(loc_mb[g]);

            int nelems = part->local_nelems[g];
            int npe = part->nodes_per_elem;

            int *row_conn = h_row_red_elem_conn[g];
            int *col_conn = h_col_red_elem_conn[g];

            for (int e = 0; e < nelems; e++) {
                for (int a = 0; a < npe; a++) {
                    int row = row_conn[e * npe + a];
                    if (row < 0) continue;

                    for (int b = 0; b < npe; b++) {
                        int col = col_conn[e * npe + b];
                        if (col >= 0) row_cols[row].insert(col);
                    }
                }
            }

            loc_nnzb[g] = 0;
            h_loc_rowp[g] = new int[loc_mb[g] + 1];
            h_loc_rowp[g][0] = 0;

            for (int row = 0; row < loc_mb[g]; row++) {
                loc_nnzb[g] += static_cast<int>(row_cols[row].size());
                h_loc_rowp[g][row + 1] = loc_nnzb[g];
            }

            loc_nnz[g] = loc_nnzb[g] * block_dim2;

            h_loc_cols[g] = new int[loc_nnzb[g]];
            h_loc_vals[g] = new T[loc_nnz[g]];

            std::memset(h_loc_cols[g], 0, loc_nnzb[g] * sizeof(int));
            std::memset(h_loc_vals[g], 0, loc_nnz[g] * sizeof(T));

            int offset = 0;
            for (int row = 0; row < loc_mb[g]; row++) {
                std::vector<int> cols(row_cols[row].begin(), row_cols[row].end());
                std::sort(cols.begin(), cols.end());

                for (int col : cols) {
                    h_loc_cols[g][offset++] = col;
                }
            }
        }
    }

    void build_transpose_pattern() {
        h_tr_loc_rowp = new int *[ngpus];
        h_tr_loc_cols = new int *[ngpus];
        h_tr_block_map = new int *[ngpus];

        d_tr_loc_rowp = new int *[ngpus];
        d_tr_loc_cols = new int *[ngpus];
        d_tr_block_map = new int *[ngpus];

        std::memset(h_tr_loc_rowp, 0, ngpus * sizeof(int *));
        std::memset(h_tr_loc_cols, 0, ngpus * sizeof(int *));
        std::memset(h_tr_block_map, 0, ngpus * sizeof(int *));

        std::memset(d_tr_loc_rowp, 0, ngpus * sizeof(int *));
        std::memset(d_tr_loc_cols, 0, ngpus * sizeof(int *));
        std::memset(d_tr_block_map, 0, ngpus * sizeof(int *));

        for (int g = 0; g < ngpus; g++) {
            int mb = loc_mb[g];
            int nb = loc_nb[g];
            int nnzb = loc_nnzb[g];

            int *rowp = h_loc_rowp[g];
            int *cols = h_loc_cols[g];

            h_tr_loc_rowp[g] = new int[nb + 1];
            h_tr_loc_cols[g] = new int[nnzb];
            h_tr_block_map[g] = new int[nnzb];

            std::memset(h_tr_loc_rowp[g], 0, (nb + 1) * sizeof(int));

            // Count entries in each transpose row = original column counts
            for (int row = 0; row < mb; row++) {
                for (int jp = rowp[row]; jp < rowp[row + 1]; jp++) {
                    int col = cols[jp];
                    h_tr_loc_rowp[g][col + 1]++;
                }
            }

            // Prefix sum
            for (int i = 0; i < nb; i++) {
                h_tr_loc_rowp[g][i + 1] += h_tr_loc_rowp[g][i];
            }

            std::vector<int> offset(nb, 0);

            // Fill transpose cols and map:
            // original block:     A(row, col) at jp
            // transpose pattern:  A^T(col, row) at jp_tr
            // map:                jp_tr -> jp
            for (int row = 0; row < mb; row++) {
                for (int jp = rowp[row]; jp < rowp[row + 1]; jp++) {
                    int col = cols[jp];

                    int jp_tr = h_tr_loc_rowp[g][col] + offset[col]++;
                    h_tr_loc_cols[g][jp_tr] = row;
                    h_tr_block_map[g][jp_tr] = jp;
                }
            }
        }
    }

    void build_element_ind_map() {
        elem_ind_map_N = new int[ngpus];
        h_loc_elem_ind_map = new int *[ngpus];
        d_loc_elem_ind_map = new int *[ngpus];

        std::memset(elem_ind_map_N, 0, ngpus * sizeof(int));
        std::memset(h_loc_elem_ind_map, 0, ngpus * sizeof(int *));
        std::memset(d_loc_elem_ind_map, 0, ngpus * sizeof(int *));

        for (int g = 0; g < ngpus; g++) {
            int nelems = part->local_nelems[g];
            int npe = part->nodes_per_elem;
            int blocks_per_elem = npe * npe;

            elem_ind_map_N[g] = nelems * blocks_per_elem;
            h_loc_elem_ind_map[g] = new int[elem_ind_map_N[g]];

            int *row_conn = h_row_red_elem_conn[g];
            int *col_conn = h_col_red_elem_conn[g];
            int *rowp = h_loc_rowp[g];
            int *cols = h_loc_cols[g];

            for (int e = 0; e < nelems; e++) {
                for (int a = 0; a < npe; a++) {
                    int row = row_conn[e * npe + a];

                    for (int b = 0; b < npe; b++) {
                        int elem_block = a * npe + b;
                        int map_index = e * blocks_per_elem + elem_block;
                        int col = col_conn[e * npe + b];

                        if (row < 0 || col < 0) {
                            h_loc_elem_ind_map[g][map_index] = -1;
                            continue;
                        }

                        int bsr_ind = -1;

                        for (int jp = rowp[row]; jp < rowp[row + 1]; jp++) {
                            if (cols[jp] == col) {
                                bsr_ind = jp;
                                break;
                            }
                        }

                        h_loc_elem_ind_map[g][map_index] = bsr_ind;
                    }
                }
            }
        }
    }

    void move_matrix_pattern_to_device() {
        for (int g = 0; g < ngpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));

            CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA[g]));
            CHECK_CUSPARSE(cusparseSetMatType(descrA[g], CUSPARSE_MATRIX_TYPE_GENERAL));
            CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA[g], CUSPARSE_INDEX_BASE_ZERO));

            CHECK_CUDA(cudaMalloc((void **)&d_row_red_elem_conn[g], elem_conn_N[g] * sizeof(int)));
            CHECK_CUDA(cudaMalloc((void **)&d_col_red_elem_conn[g], elem_conn_N[g] * sizeof(int)));
            CHECK_CUDA(
                cudaMalloc((void **)&d_loc_elem_ind_map[g], elem_ind_map_N[g] * sizeof(int)));

            CHECK_CUDA(cudaMemcpyAsync(d_row_red_elem_conn[g], h_row_red_elem_conn[g],
                                       elem_conn_N[g] * sizeof(int), cudaMemcpyHostToDevice,
                                       streams[g]));

            CHECK_CUDA(cudaMemcpyAsync(d_col_red_elem_conn[g], h_col_red_elem_conn[g],
                                       elem_conn_N[g] * sizeof(int), cudaMemcpyHostToDevice,
                                       streams[g]));

            CHECK_CUDA(cudaMemcpyAsync(d_loc_elem_ind_map[g], h_loc_elem_ind_map[g],
                                       elem_ind_map_N[g] * sizeof(int), cudaMemcpyHostToDevice,
                                       streams[g]));

            CHECK_CUDA(cudaMalloc((void **)&d_loc_rowp[g], (loc_mb[g] + 1) * sizeof(int)));
            CHECK_CUDA(cudaMalloc((void **)&d_loc_cols[g], loc_nnzb[g] * sizeof(int)));
            CHECK_CUDA(cudaMalloc((void **)&d_loc_vals[g], loc_nnz[g] * sizeof(T)));

            CHECK_CUDA(cudaMemcpyAsync(d_loc_rowp[g], h_loc_rowp[g], (loc_mb[g] + 1) * sizeof(int),
                                       cudaMemcpyHostToDevice, streams[g]));

            CHECK_CUDA(cudaMemcpyAsync(d_loc_cols[g], h_loc_cols[g], loc_nnzb[g] * sizeof(int),
                                       cudaMemcpyHostToDevice, streams[g]));

            CHECK_CUDA(cudaMemcpyAsync(d_loc_vals[g], h_loc_vals[g], loc_nnz[g] * sizeof(T),
                                       cudaMemcpyHostToDevice, streams[g]));

            CHECK_CUDA(cudaMalloc((void **)&d_tr_loc_rowp[g], (loc_nb[g] + 1) * sizeof(int)));
            CHECK_CUDA(cudaMalloc((void **)&d_tr_loc_cols[g], loc_nnzb[g] * sizeof(int)));
            CHECK_CUDA(cudaMalloc((void **)&d_tr_block_map[g], loc_nnzb[g] * sizeof(int)));

            CHECK_CUDA(cudaMemcpyAsync(d_tr_loc_rowp[g], h_tr_loc_rowp[g],
                                       (loc_nb[g] + 1) * sizeof(int), cudaMemcpyHostToDevice,
                                       streams[g]));

            CHECK_CUDA(cudaMemcpyAsync(d_tr_loc_cols[g], h_tr_loc_cols[g],
                                       loc_nnzb[g] * sizeof(int), cudaMemcpyHostToDevice,
                                       streams[g]));

            CHECK_CUDA(cudaMemcpyAsync(d_tr_block_map[g], h_tr_block_map[g],
                                       loc_nnzb[g] * sizeof(int), cudaMemcpyHostToDevice,
                                       streams[g]));
        }
    }

   private:
    MultiGPUContext *ctx = nullptr;
    const Partitioner *part = nullptr;

    cublasHandle_t *cublasHandles = nullptr;
    cusparseHandle_t *cusparseHandles = nullptr;
    cudaStream_t *streams = nullptr;
    cusparseMatDescr_t *descrA = nullptr;

    int ngpus = 0;
    int num_nodes = 0;
    int block_dim = 0;
    int block_dim2 = 0;

    int *elem_conn_N = nullptr;
    int *elem_ind_map_N = nullptr;

    int **h_row_red_elem_conn = nullptr;
    int **h_col_red_elem_conn = nullptr;
    int **h_loc_elem_ind_map = nullptr;

    int **d_row_red_elem_conn = nullptr;
    int **d_col_red_elem_conn = nullptr;
    int **d_loc_elem_ind_map = nullptr;

    int *loc_mb = nullptr;
    int *loc_nb = nullptr;
    int *loc_nnzb = nullptr;
    int *loc_nnz = nullptr;

    int **h_loc_rowp = nullptr;
    int **h_loc_cols = nullptr;
    T **h_loc_vals = nullptr;

    int **d_loc_rowp = nullptr;
    int **d_loc_cols = nullptr;
    T **d_loc_vals = nullptr;

    int **h_tr_loc_rowp = nullptr;
    int **h_tr_loc_cols = nullptr;
    int **h_tr_block_map = nullptr;

    int **d_tr_loc_rowp = nullptr;
    int **d_tr_loc_cols = nullptr;
    int **d_tr_block_map = nullptr;
};