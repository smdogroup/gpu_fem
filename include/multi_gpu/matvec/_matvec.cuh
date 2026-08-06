#pragma once
#include "cuda_utils.h"


template <typename T>
__global__ void k_pack_ghost_red(int nnodes, int block_dim, const int *map, const T *x, T *xred) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int size = nnodes * block_dim;
    if (tid < size) {
        int inode = tid / block_dim;
        int idof = tid % block_dim;
        int src_node = map[inode];
        xred[tid] = x[src_node * block_dim + idof];
    }
}

template <typename T>
__global__ void k_place_ghost_red(int nnodes, int block_dim, const int *map, const T *xred,
                                  T *xloc) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int size = nnodes * block_dim;
    if (tid < size) {
        int inode = tid / block_dim;
        int idof = tid % block_dim;
        int dst_node = map[inode];
        xloc[dst_node * block_dim + idof] = xred[tid];
    }
}
template <typename T>
__global__ void k_scatter_owned_to_local(int nnodes, int block_dim,
                                         const int *map,
                                         const T *xowned,
                                         T *xloc) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int N = nnodes * block_dim;

    if (tid < N) {
        int inode = tid / block_dim;
        int idof = tid % block_dim;

        int loc_node = map[inode];

        xloc[loc_node * block_dim + idof] = xowned[tid];
    }
}

template <typename T>
__global__ void k_add_local_owned_to_owned(int nnodes, int block_dim,
                                           const int *owned_to_local_map,
                                           const T *local_vals,
                                           T *owned_vals) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int N = nnodes * block_dim;

    if (tid < N) {
        int inode = tid / block_dim;
        int idof = tid % block_dim;

        int local_node = owned_to_local_map[inode];

        owned_vals[tid] += local_vals[block_dim * local_node + idof];
    }
}

template <typename T>
__global__ void k_vec_apply_bcs(int nbcs, const int *bcs, T *data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < nbcs) {
        int idof = bcs[tid];
        data[idof] = 0.0;
    }
}

template <typename T>
__GLOBAL__ void k_scatter_single_pack_to_global(
    int owned_nnodes,
    int block_dim,
    const int *owned_nodes,
    const T *pack_vals,
    T *global_vals) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int N = owned_nnodes * block_dim;
    if (tid >= N) return;

    int inode = tid / block_dim;
    int idof = tid % block_dim;

    int global_node = owned_nodes[inode];
    global_vals[global_node * block_dim + idof] = pack_vals[tid];
}

template <typename T>
__GLOBAL__ void k_gather_global_to_single_pack(
    int owned_nnodes,
    int block_dim,
    const int *owned_nodes,
    const T *global_vals,
    T *pack_vals) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int N = owned_nnodes * block_dim;
    if (tid >= N) return;

    int inode = tid / block_dim;
    int idof = tid % block_dim;

    int global_node = owned_nodes[inode];
    pack_vals[tid] = global_vals[global_node * block_dim + idof];
}

template <typename T>
__global__ void k_copy_local_owned_to_owned(int nnodes, int block_dim,
                                            const int *owned_to_local_map,
                                            const T *local_vals,
                                            T *owned_vals) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int N = nnodes * block_dim;

    if (tid < N) {
        int inode = tid / block_dim;
        int idof = tid % block_dim;

        int local_node = owned_to_local_map[inode];

        owned_vals[tid] = local_vals[block_dim * local_node + idof];
    }
}

template <typename T>
__global__ void k_pack_local_ghost_red(int nnodes, int block_dim,
                                       const int *dstred_map,
                                       const T *local_vals,
                                       T *red_vals) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int N = nnodes * block_dim;

    if (tid < N) {
        int inode = tid / block_dim;
        int idof = tid % block_dim;

        int local_node = dstred_map[inode];

        red_vals[tid] = local_vals[block_dim * local_node + idof];
    }
}

template <typename T>
__global__ void k_add_red_to_owned(int nnodes, int block_dim,
                                   const int *srcred_map,
                                   const T *red_vals,
                                   T *owned_vals) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int N = nnodes * block_dim;

    if (tid < N) {
        int inode = tid / block_dim;
        int idof = tid % block_dim;

        int owned_node = srcred_map[inode];

        owned_vals[block_dim * owned_node + idof] += red_vals[tid];
    }
}

template <typename T, bool ones_on_diag = true>
__GLOBAL__ void k_mat_apply_row_bcs(
    const int block_dim,
    const int mb,
    const int n_local_bcs,
    const int *d_local_bcs,
    const int *rowp,
    const int *cols,
    T *values) {
    
    int ibc = blockIdx.x * blockDim.x + threadIdx.x;
    if (ibc >= n_local_bcs) return;

    const int block_dim2 = block_dim * block_dim;

    int bc_dof = d_local_bcs[ibc];
    int block_row = bc_dof / block_dim;
    int inner_row = bc_dof % block_dim;

    if (block_row < 0 || block_row >= mb) return;

    for (int jp = rowp[block_row]; jp < rowp[block_row + 1]; jp++) {
        int block_col = cols[jp];

        for (int inner_col = 0; inner_col < block_dim; inner_col++) {
            int inner_block = block_dim * inner_row + inner_col;

            bool is_diag =
                (block_col == block_row) &&
                (inner_col == inner_row);

            T diag_val = ones_on_diag ? T(1.0) : T(0.0);
            values[block_dim2 * jp + inner_block] =
                is_diag ? diag_val : T(0.0);
        }
    }
}

template <typename T, bool ones_on_diag = true>
__GLOBAL__ void k_mat_apply_col_bcs(
    const int block_dim,
    const int nb,
    const int n_local_bcs,
    const int *d_local_bcs,
    const int *tr_rowp,
    const int *tr_cols,
    const int *tr_block_map,
    T *values) {
    
    int ibc = blockIdx.x * blockDim.x + threadIdx.x;
    if (ibc >= n_local_bcs) return;

    const int block_dim2 = block_dim * block_dim;

    int bc_dof = d_local_bcs[ibc];
    int block_col = bc_dof / block_dim;
    int inner_col = bc_dof % block_dim;

    if (block_col < 0 || block_col >= nb) return;

    for (int jp_tr = tr_rowp[block_col]; jp_tr < tr_rowp[block_col + 1]; jp_tr++) {
        int block_row = tr_cols[jp_tr];
        int jp = tr_block_map[jp_tr];

        for (int inner_row = 0; inner_row < block_dim; inner_row++) {
            int inner_block = block_dim * inner_row + inner_col;

            bool is_diag =
                (block_row == block_col) &&
                (inner_row == inner_col);

            T diag_val = ones_on_diag ? T(1.0) : T(0.0);
            values[block_dim2 * jp + inner_block] =
                is_diag ? diag_val : T(0.0);
        }
    }
}


template <typename T>
__GLOBAL__ void k_copy_nofill_vals_to_lu_vals(int nofill_nnzb, int block_dim2,
                                              const int *dest_blocks, const T *src_vals,
                                              T *dst_vals) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int N = nofill_nnzb * block_dim2;
    if (tid >= N) return;

    int src_block = tid / block_dim2;
    int inner = tid % block_dim2;

    int dst_block = dest_blocks[src_block];
    if (dst_block < 0) return;

    dst_vals[dst_block * block_dim2 + inner] = src_vals[src_block * block_dim2 + inner];
}

template <typename T>
__global__ void k_set_owned_from_global_host_order(
    int nnodes,
    int block_dim,
    const int *owned_nodes,
    const T *global_vals,
    T *owned_vals) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int N = nnodes * block_dim;
    if (tid >= N) return;

    int i = tid / block_dim;
    int idof = tid % block_dim;
    int global_node = owned_nodes[i];

    owned_vals[tid] = global_vals[global_node * block_dim + idof];
}

template <typename T>
__global__ void k_get_owned_to_global_host_order(
    int nnodes,
    int block_dim,
    const int *owned_nodes,
    const T *owned_vals,
    T *global_vals) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int N = nnodes * block_dim;
    if (tid >= N) return;

    int i = tid / block_dim;
    int idof = tid % block_dim;
    int global_node = owned_nodes[i];

    global_vals[global_node * block_dim + idof] = owned_vals[tid];
}