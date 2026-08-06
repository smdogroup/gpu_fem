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
    // if (inode == 0) {
    //     printf("inode 0 : block_val %d = %.4e\n", tid, block_vals[tid]);
    // }
    for (int i = threadIdx.x; i < block_dim2; i += blockDim.x) {
        atomicAdd(&shared_nrm2[0], block_vals[i] * block_vals[i]);
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
        atomicAdd(&shared_nrm2[0], block_vals[i] * block_vals[i]);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        T norm = sqrt(shared_nrm2[0]);
        // if (block_row == 0) {
        //     printf("norm %.4e, lb %.4e\n", norm, lb);
        // }
        d_strength_indicator[block_ind] = (norm >= lb ? 1 : 0);
    }
}


template <typename T, template <typename> class Vec>
__global__ void apply_mat_bcs_P_kernel(Vec<int> bcs, const int32_t block_dim, const int *rowp, const int *cols, 
                                          T *values)
{
    // apply bcs to the rows of the matrix
    int this_thread_bc = blockIdx.x * blockDim.x + threadIdx.x;
    // int stride = blockDim.x;
    int num_bcs = bcs.getSize();
    int block_dim2 = block_dim * block_dim;

    // assume num_bcs is reasonable that you can parallelize over it
    // with one bc per thread
    // if very large, we can change parallelization strategy
    if (this_thread_bc < num_bcs)
    {
        // get data associated with the row of this BC in the BSR matrix
        int bc_dof = bcs[this_thread_bc];
        // int _global_row = bc_dof;
        int _block_row = bc_dof / block_dim;
        // int block_row = iperm[_block_row]; // old to new brow
        int block_row = _block_row; // no perm
        int inner_row = bc_dof % block_dim;
        int global_row = block_dim * block_row + inner_row;
        int istart = rowp[block_row];
        int iend = rowp[block_row + 1];
        T *val = &values[block_dim2 * istart];

        // now loop over all columns, setting this entire row zero
        for (int col_ptr_ind = istart; col_ptr_ind < iend; col_ptr_ind++)
        {
            int block_col = cols[col_ptr_ind];
            // printf("set row %d, block col %d to zero\n", global_row, block_col);
            for (int inner_col = 0; inner_col < block_dim; inner_col++) {
                int inz = block_dim * inner_row + inner_col;
                val[inz] = 0.0;
            }
            val += block_dim2; // go to the next nonzero block (on this row)
        }

        // printf("apply bc %d\n", bc_dof);
    }
}

template <typename T>
__global__ void k_copy_rbm_into_tentative_prolongator(const int nnodes,  
    const int block_dim, const int *d_tentative_block_map, 
    const T *rigid_modes, T *prolong_vals) {
    // copy rigid body modes into tentative prolongator (with fill-in pattern)
    int ind = threadIdx.x + blockIdx.x * blockDim.x;
    int block_dim2 = block_dim * block_dim;
    int nvals = block_dim2 * nnodes;
    if (ind >= nvals) return;

    int fine_node = ind / block_dim2;
    const T *B_block = &rigid_modes[block_dim2 * fine_node];
    int P_block_ind = d_tentative_block_map[fine_node];
    T *P_block = &prolong_vals[block_dim2 * P_block_ind];

    int inn_ind = ind % block_dim2;
    P_block[inn_ind] = B_block[inn_ind];
}



template <typename T>
__global__ void k_compute_aggregate_norms2(const int imode, const int nnodes,  
    const int block_dim, const int *d_aggregate_ind, const int *d_tentative_block_map, 
    const T *prolong_vals, T *d_agg_norms2) {
    // compute the aggregate norms of imode
    int ind = threadIdx.x + blockIdx.x * blockDim.x;
    int N = block_dim * nnodes;
    if (ind >= N) return;

    int fine_node = ind / block_dim;
    int idof = ind % block_dim;
    int coarse_node = d_aggregate_ind[fine_node];
    // get norms of imode column in rigid_modes B [N x 6] matrix
    int block_dim2 = block_dim * block_dim;
    int P_block_ind = d_tentative_block_map[fine_node];
    const T *P_block_vals = &prolong_vals[block_dim2 * P_block_ind];
    T val = P_block_vals[block_dim * idof + imode];
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
    int coarse_agg = tid;
    T *coarse_block = &coarse_rigid_modes[block_dim2 * coarse_agg];
    coarse_block[imode * block_dim + imode] = sqrt_norm;
}

template <typename T>
__global__ void k_normalize_tentative_modes(const int imode, const int nnodes,  
    const int block_dim, const int *d_aggregate_ind, const int *d_tentative_block_map, 
    const T *rigid_coarse_modes, T *prolong_vals) {
    // normalize tentative mode (imode) from B into T tentative prolongator
    int ind = threadIdx.x + blockIdx.x * blockDim.x;
    int N = block_dim * nnodes;
    if (ind >= N) return;

    int fine_node = ind / block_dim;
    int coarse_node = d_aggregate_ind[fine_node];
    int P_block_ind = d_tentative_block_map[fine_node];
    int block_dim2 = block_dim * block_dim;
    T *P_block = &prolong_vals[block_dim2 * P_block_ind];
    const T *Bc_block = &rigid_coarse_modes[block_dim2 * coarse_node];
    T sqrt_norm = Bc_block[imode * block_dim + imode]; // read diagonal entry

    int irow = ind % block_dim;
    P_block[block_dim * irow + imode] /= sqrt_norm;
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
    atomicAdd(&Bc_block[block_dim * jmode + imode], val_i * val_j);
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
    T ij_dot = Bc_block[block_dim * jmode + imode];
    int inn_row = ind % block_dim;
    P_block[block_dim * inn_row + imode] -= ij_dot * P_block[block_dim * inn_row + jmode];
}



// THIS VERSION IS WRONG cause adds into AP first in shared mem (but not all AP blocks are added in, so not correct foil method)
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

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("Kc_vals: ");
        printVec<T>(36, Kc_vals);
        printf("PL_vals: ");
        printVec<T>(36, PL_vals);
        printf("PR_vals: ");
        printVec<T>(36, PR_vals);
        printf("K_vals: ");
        printVec<T>(36, K_vals);
    }

    int Kc_ind = Kc_blocks[prod_block_ind];
    // if (threadIdx.x == 0) {
    //     printf("block %d, Kc_ind %d\n", prod_block_ind, Kc_ind);
    // }

    // computes one 6x6 * 6x6 * 6x6 matrix into 6x6 out of P^T * A *P
    // intermediate A*P product stored in shared memory (since not storing matrix values for AP)

    // first compute AP product (216 sum-prod terms)
    for (int ip = threadIdx.x; ip < 216; ip += blockDim.x) {
        int ij = ip % 36, k = ip / 36;
        int i = ij / 6, j = ij % 6;

        // (AP)_{ik} = K_{ij} * P_{jk} (where second index is col and col is ind % 6 while row is ind / 6)
        atomicAdd(&AP_vals[6 * i + k], K_vals[6 * i + j] * PR_vals[6 * j + k]);
    }
    __syncthreads();

    // then compute P^T * AP (216 sum-prod terms) into P^T * A * P => Kc matrix
    for (int ip = threadIdx.x; ip < 216; ip += blockDim.x) {
        int ij = ip % 36, k = ip / 36;
        int i = ij / 6, j = ij % 6;

        // PL_vals we need to read in transposed here (i,j) => (j,i)
        // (Kc)_{ik} = (P^T)_{ij} * (AP)_{jk} = P_{ji} * (AP)_{jk}
        atomicAdd(&Kc_vals[6 * i + k], PL_vals[6 * j + i] * AP_vals[6 * j + k]);
    }
}

template <typename T>
__global__ void k_compute_PTAP_product6_v2(const int PTAP_nnzb_prod, const int block_dim, 
    const int *Kc_blocks, const int *PL_blocks, const int *K_blocks, const int *PR_blocks, 
    const T *prolong_vals, const T *kmat_vals, T *galerkin_vals) {
    
    int prod_block_ind = blockIdx.x;
    if (prod_block_ind >= PTAP_nnzb_prod) return;
    int block_dim2 = block_dim * block_dim;
    int tid = threadIdx.x;
    // can't do shared memory.. or add into AP partial cause isn't adding all blocks into AP
    // that was wrong last time

    const T *PL_vals = &prolong_vals[block_dim2 * PL_blocks[prod_block_ind]];
    const T *K_vals = &kmat_vals[block_dim2 * K_blocks[prod_block_ind]];
    const T *PR_vals = &prolong_vals[block_dim2 * PR_blocks[prod_block_ind]];
    T *Kc_vals = &galerkin_vals[block_dim2 * Kc_blocks[prod_block_ind]];

    int Kc_ind = Kc_blocks[prod_block_ind];
    // computes one 6x6 * 6x6 * 6x6 matrix into 6x6 out of P^T * A *P

    // compute full P^T * A * P product in-place (1296 product terms = 6^4)
    for (int il = threadIdx.x; il < 1296; il += blockDim.x) {
        int ijk = il % 216, l = il / 216;
        int ij = ijk % 36, k = ijk / 36;
        int i = ij / 6, j = ij % 6;

        // if (blockIdx.x == 0) {
        //     printf("i %d, j %d, k %d, l %d\n", i, j, k, l);
        // }

        // (Kc)_{il} = (P^T)_{ij} * A_{jk} * P_{kl}
        atomicAdd(&Kc_vals[6 * i + l], PL_vals[6 * j + i] * K_vals[6 * j + k] * PR_vals[6 * k + l]);
    }
}