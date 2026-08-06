#pragma once
#include "cuda_utils.h"



template <typename T>
__global__ void k_setupBatchedPointers(
    int batch_size,
    int n,
    T *Adata,
    T *invAdata,
    T *Xdata,
    T *Ydata,
    T **Aarray,
    T **invAarray,
    T **Xarray,
    T **Yarray) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    Aarray[b] = &Adata[(size_t)b * n * n];
    invAarray[b] = &invAdata[(size_t)b * n * n];
    Xarray[b] = &Xdata[(size_t)b * n];
    Yarray[b] = &Ydata[(size_t)b * n];
}

template <typename T>
__global__ void k_copyMatValuesToBatchedContiguous(
    int n_batch_vals,
    int block_dim,
    int size,
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
__global__ void k_copyLocalRHSIntoBatched(
    int n_rhs_vals,
    int block_dim,
    int size,
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
    x[local_slot * block_dim + idof] =
        rhs_local[local_node * block_dim + idof];
}

template <typename T>
__global__ void k_addBatchedIntoLocalSoln(
    int n_rhs_vals,
    int block_dim,
    int size,
    const int *__restrict__ local_node_map,
    T **__restrict__ Yarray,
    T *__restrict__ soln_local) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_rhs_vals) return;

    int size2 = size * size;

    int node_entry = tid / block_dim;
    int idof = tid % block_dim;

    int batch = node_entry / size2;
    int local_slot = node_entry % size2;
    int local_node = local_node_map[node_entry];

    if (local_node < 0) return;

    const T *y = Yarray[batch];
    T val = y[local_slot * block_dim + idof];

    atomicAdd(&soln_local[local_node * block_dim + idof], val);
}

template <typename T>
__global__ void k_packGhostGhostMatBlocks(
    int nvals,
    int block_dim,
    const int *__restrict__ kmat_blocks,
    const T *__restrict__ vals,
    T *__restrict__ red_vals) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nvals) return;

    int block_dim2 = block_dim * block_dim;
    int iblock = tid / block_dim2;
    int inner = tid % block_dim2;

    int jp = kmat_blocks[iblock];
    if (jp < 0) {
        red_vals[tid] = 0.0;
    } else {
        red_vals[tid] = vals[(size_t)jp * block_dim2 + inner];
    }
}

template <typename T>
__global__ void k_addGhostGhostMatBlocksToBatched(
    int nvals,
    int block_dim,
    int size,
    const int *__restrict__ asw_blocks,
    const T *__restrict__ red_vals,
    T *__restrict__ Adata) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nvals) return;

    int block_dim2 = block_dim * block_dim;
    int size2 = size * size;
    int size4 = size2 * size2;
    int n = size2 * block_dim;

    int iblock = tid / block_dim2;
    int inner = tid % block_dim2;

    int batch_block = asw_blocks[iblock];
    if (batch_block < 0) return;

    int batch = batch_block / size4;
    int inner_block = batch_block % size4;

    int i_node = inner_block % size2;
    int j_node = inner_block / size2;

    int p = inner / block_dim;
    int q = inner % block_dim;

    int row = i_node * block_dim + p;
    int col = j_node * block_dim + q;

    atomicAdd(&Adata[(size_t)batch * n * n + row + col * n], red_vals[tid]);
}