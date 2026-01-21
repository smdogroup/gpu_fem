#pragma once
#include "cuda_utils.h"
#include "multigrid/smoothers/_smoothers.cuh"

template <typename T>
__global__ void k_get_diag_norms(const int nnodes, const int *diagp, const int block_dim, 
    const T *mat_vals, T *diag_norms) {
    int inode = blockIdx.x;
    if (inode >= nnodes) return;
    int diag_block_ind = diagp[inode];

    int tid = threadIdx.x;
    int block_dim2 = block_dim * block_dim;
    const T *block_vals = &mat_vals[block_dim2 * diag_block_ind];
    
    T __shared__ shared_nrm2[1];
    if (threadIdx.x == 0) shared_nrm2[0] = 0.0;
    __syncthreads();

    // compute frobenius norm
    for (int i = threadIdx.x; i < block_dim2; i += blockDim.x) {
        shared_nrm2[0] += block_vals[i] * block_vals[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        diag_norms[inode] = sqrt(shared_nrm2[0]);
    }
}

template <typename T>
__global__ void k_compute_strength_bools(const int nnzb, const int block_dim, const T *d_diag_norms, 
    const int *d_rows, const int *d_cols, const T *mat_vals, const T threshold, bool *d_strength_indicator) {
    
    int block_ind = blockIdx.x;
    if (block_ind >= nnzb) return;
    int block_row = d_rows[block_ind];
    int block_col = d_cols[block_ind];
    T diag_row = d_diag_norms[block_row];
    T diag_col = d_diag_norms[block_col];
    T lb = threshold * sqrt(diag_row * diag_col);
    int block_dim2 = block_dim * block_dim;

    const T *block_vals = &mat_vals[block_dim2 * block_ind];

    T __shared__ shared_nrm2[1];
    if (threadIdx.x == 0) shared_nrm2[0] = 0.0;
    __syncthreads();
    
    for (int i = threadIdx.x; i < block_dim2; i += blockDim.x) {
        shared_nrm2[0] += block_vals[i] * block_vals[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        T norm = sqrt(shared_nrm2[0]);
        d_strength_indicator[block_ind] = (norm >= lb ? 1 : 0);
    }
}


template <typename T>
__global__ void k_compute_PTAP_product6(const int PTAP_nnzb_prod, const int block_dim, 
    const int *Kc_blocks, const int *PL_blocks, const int *K_blocks, const int *PR_blocks, 
    const T *prolong_vals, const T *kmat_vals, T *galerkin_vals) {
    
    int prod_block_ind = blockIdx.x;
    if (prod_block_ind >= PTAP_nnzb_prod) return;
    int block_dim2 = block_dim * block_dim;
    int tid = threadIdx.x;
    T __shared__ AP_vals[36];
    memset(AP_vals, 0.0, 36 * sizeof(T));

    const T *PL_vals = &prolong_vals[block_dim2 * PL_blocks[prod_block_ind]];
    const T *K_vals = &kmat_vals[block_dim2 * K_blocks[prod_block_ind]];
    const T *PR_vals = &prolong_vals[block_dim2 * PR_blocks[prod_block_ind]];
    T *Kc_vals = &galerkin_vals[block_dim2 * Kc_blocks[prod_block_ind]];

    // computes one 6x6 * 6x6 * 6x6 matrix into 6x6 out of P^T * A *P
    // intermediate A*P product stored in shared memory (since not storing matrix values for AP)

    // first compute AP product (216 sum-prod terms)
    for (int ip = threadIdx.x; ip < 216; ip += blockDim.x) {
        int ij = ip % 36, k = ip / 36;
        int i = ip / 6, j = ip % 6;

        atomicAdd(&AP_vals[6 * i + k], K_vals[6 * i + j] * PR_vals[6 * j + k]);
    }

    // then compute P^T * AP (216 sum-prod terms) into P^T * A * P => Kc matrix
    for (int ip = threadIdx.x; ip < 216; ip += blockDim.x) {
        int ij = ip % 36, k = ip / 36;
        int i = ip / 6, j = ip % 6;

        // PL_vals we need to read in transposed here (i,j) => (j,i)
        atomicAdd(&galerkin_vals[6 * i + k], PL_vals[6 * j + i] * AP_vals[6 * j + k]);
    }
}

template <typename T>
__global__ void k_compute_aggregate_norms2(const int imode, const int nnodes,  
    const int block_dim, const int *d_aggregate_ind,  
    const T *rigid_modes, T *d_agg_norms2) {
    // compute the aggregate norms of imode
    int ind = threadIdx.x + blockIdx.x * blockDim.x;
    int N = block_dim * nnodes;
    if (ind >= N) return;

    int fine_node = ind / block_dim;
    int coarse_node = d_aggregate_ind[fine_node];
    // get norms of imode column in rigid_modes B [N x 6] matrix
    T val = rigid_modes[ind];
    atomicAdd(&d_agg_norms2[coarse_node], val * val);
}

template <typename T>
__global__ void k_compute_sqrt_norms(const int imode, const int num_aggregates, const int block_dim, 
    const T *d_norms2, T *coarse_rigid_modes) {
    // store sqrt norm in diagonal (imode,imode) spot in coarse rigid modes
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= num_aggregates) return;

    T sqrt_norm = sqrt(d_norms2[tid]);
    int block_dim2 = block_dim * block_dim;
    T *coarse_block = &coarse_rigid_modes[block_dim2 * tid];
    coarse_block[imode * block_dim + imode] = sqrt_norm;
}

template <typename T>
__global__ void k_normalize_tentative_modes(const int imode, const int nnodes,  
    const int block_dim, const int *d_aggregate_ind, const int *d_tentative_block_map, 
    const T *rigid_modes, const T *rigid_coarse_modes, T *prolong_vals) {
    // normalize tentative mode (imode) from B into T tentative prolongator
    int ind = threadIdx.x + blockIdx.x * blockDim.x;
    int N = block_dim * nnodes;
    if (ind >= N) return;

    int fine_node = ind / block_dim;
    int coarse_node = d_aggregate_ind[fine_node];
    int P_block_ind = d_tentative_block_map[fine_node];
    int block_dim2 = block_dim * block_dim;
    T *P_block = &prolong_vals[block_dim2 * P_block_ind];
    const T *B_block = &rigid_modes[block_dim2 * fine_node];
    const T *Bc_block = &rigid_coarse_modes[block_dim2 * coarse_node];
    T sqrt_norm = Bc_block[imode * block_dim + imode]; // read diagonal entry

    int irow = ind % block_dim;
    P_block[block_dim * irow + imode] = B_block[block_dim * irow + imode] / sqrt_norm;
}


template <typename T>
__global__ void k_compute_GS_inner_product(const int imode, const int jmode, const int nnodes,  
    const int block_dim, const int *d_aggregate_ind, const int *d_tentative_block_map, 
    const T *prolong_vals, T *rigid_coarse_modes) {
    // compute inner product (vi, vj) columns in P tentative prolongator already
    // and store in (imode,jmode) of (6,6) sub-block in Bc rigid coarse modes
    int ind = threadIdx.x + blockIdx.x * blockDim.x;
    int N = block_dim * nnodes;
    if (ind >= N) return;

    int fine_node = ind / block_dim;
    int coarse_node = d_aggregate_ind[fine_node];
    int P_block_ind = d_tentative_block_map[fine_node];
    int block_dim2 = block_dim * block_dim;
    const T *P_block = &prolong_vals[block_dim2 * P_block_ind];
    T *Bc_block = &rigid_coarse_modes[block_dim2 * coarse_node];

    // upper-triangular storage in Bc here as it should be
    int inn_row = ind % block_dim;
    T val_i = P_block[block_dim * inn_row + imode];
    T val_j = P_block[block_dim * inn_row + jmode];
    atomicAdd(&Bc_block[block_dim * imode + jmode], val_i * val_j);
}


template <typename T>
__global__ void k_remove_GS_projector_mode(const int imode, const int jmode, const int nnodes,  
    const int block_dim, const int *d_aggregate_ind, const int *d_tentative_block_map, 
    const T *rigid_coarse_modes, T *prolong_vals) {
    // subtraction v_i = v_i - (v_i, e_j) * e_j where i > j
    // and dot product stored in rigid body modes Bc vals
    // v_i and e_j are stored in tentative prolongator with full A*T sparsity pattern though

    int ind = threadIdx.x + blockIdx.x * blockDim.x;
    int N = block_dim * nnodes;
    if (ind >= N) return;

    int fine_node = ind / block_dim;
    int coarse_node = d_aggregate_ind[fine_node];
    int P_block_ind = d_tentative_block_map[fine_node];
    int block_dim2 = block_dim * block_dim;
    T *P_block = &prolong_vals[block_dim2 * P_block_ind];
    const T *Bc_block = &rigid_coarse_modes[block_dim2 * coarse_node];

    
    // upper-triangular storage in Bc here as it should be
    T ij_dot = Bc_block[block_dim * imode + jmode];
    int inn_row = ind % block_dim;
    P_block[block_dim * inn_row + imode] -= ij_dot * P_block[block_dim * inn_row + jmode];
}
