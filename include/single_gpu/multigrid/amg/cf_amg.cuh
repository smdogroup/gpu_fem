#pragma once
#include "cuda_utils.h"
#include "multigrid/smoothers/_smoothers.cuh"
#include "multigrid/amg/sa_amg.cuh"

template <typename T>
__global__ void k_copy_CF_mat(const int ncopy,  
    const int block_dim, const int *d_P_blocks, const int *d_K_blocks, 
    const T *kmat_vals, T *prolong_vals) {
    // copy rigid body modes into tentative prolongator (with fill-in pattern)
    int ind = threadIdx.x + blockIdx.x * blockDim.x;
    int block_dim2 = block_dim * block_dim;
    int nvals = block_dim2 * ncopy;
    if (ind >= nvals) return;

    int iblock = ind / block_dim2;
    int P_block_ind = d_P_blocks[iblock];
    int K_block_ind = d_K_blocks[iblock];
    const T *K_block = &kmat_vals[block_dim2 * K_block_ind];
    T *P_block = &prolong_vals[block_dim2 * P_block_ind];

    int inn_ind = ind % block_dim2;
    P_block[inn_ind] = K_block[inn_ind];
}

template <typename T>
__global__ void k_set_eye_CF_mat(const int neye,  
    const int block_dim, const int *d_coarse_blocks, T *prolong_vals) {
    // copy rigid body modes into tentative prolongator (with fill-in pattern)
    int ind = threadIdx.x + blockIdx.x * blockDim.x;
    int block_dim2 = block_dim * block_dim;
    int nvals = block_dim2 * neye;
    if (ind >= nvals) return;

    int iblock = ind / block_dim2;
    int P_block_ind = d_coarse_blocks[iblock];
    T *P_block = &prolong_vals[block_dim2 * P_block_ind];

    int inn_ind = ind % block_dim2;
    int row = inn_ind / block_dim, col = inn_ind % block_dim;
    T val = (row == col) ? 1.0 : 0.0;
    P_block[inn_ind] = val;
}

template <typename T>
__global__ void k_add_row_sums(const int nnzb, const int block_dim, const int *d_rows, 
    const T *prolong_vals, T *row_sums) {
    int ind = threadIdx.x + blockIdx.x * blockDim.x;
    int block_dim2 = block_dim * block_dim;
    int nvals = block_dim2 * nnzb;
    if (ind >= nvals) return;

    int iblock = ind / block_dim2;
    int block_row = d_rows[iblock];
    int inn_ind = ind % block_dim2;
    int inn_row = inn_ind / block_dim, inn_col = inn_ind % block_dim;

    int vec_ind = block_dim * block_row + inn_row;
    const T *P_block = &prolong_vals[block_dim2 * iblock];

    atomicAdd(&row_sums[vec_ind], P_block[inn_ind]);
}

template <typename T>
__global__ void k_normalize_with_row_sums(const int nnzb, const int block_dim, const int *d_rows, 
    const T *row_sums, T *prolong_vals) {
    int ind = threadIdx.x + blockIdx.x * blockDim.x;
    int block_dim2 = block_dim * block_dim;
    int nvals = block_dim2 * nnzb;
    if (ind >= nvals) return;

    int iblock = ind / block_dim2;
    int block_row = d_rows[iblock];
    int inn_ind = ind % block_dim2;
    int inn_row = inn_ind / block_dim, inn_col = inn_ind % block_dim;

    int vec_ind = block_dim * block_row + inn_row;
    T *P_block = &prolong_vals[block_dim2 * iblock];

    P_block[inn_ind] /= (1e-14 + row_sums[vec_ind]);
}