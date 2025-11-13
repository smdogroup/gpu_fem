#pragma once

// helper kernels for the sa plate and potentially other examples

template <typename T>
__global__ void k_compute_P_K_P_mmprod(int nnzb_prod, int block_dim, T scale, 
    int *d_K_blocks, int *d_P_blocks, int *d_PF_blocks,
    const T *d_K_vals, const T *d_P_vals, T *d_PF_vals) {
    
    // computes K * P => new P array as mat-mat product with pre-computed nonzero block pattern
    int iprod = blockIdx.x;
    // int iprod = threadIdx.x + blockIdx.x * blockDim.x;
    if (iprod >= nnzb_prod) return;

    // get block indices in each matrix
    int bind_P = d_P_blocks[iprod];
    int bind_K = d_K_blocks[iprod];
    int bind_PF = d_PF_blocks[iprod];
    __syncthreads();

    // parallelize over the dense product
    int block_dim2 = block_dim * block_dim;
    int block_dim3 = block_dim2 * block_dim;
    // int ndense_prods = block_dim3; // do need blockDim.x > ndense_prods to work correctly 
    // aka need >= 216 block dim here to work well

    // get data at start of block each matrix
    const T *K = &d_K_vals[block_dim2 * bind_K];
    const T *P = &d_P_vals[block_dim2 * bind_P];
    T *PF = &d_PF_vals[block_dim2 * bind_PF];

    // parallelize over the whole ijk dense mat-mat product
    for (int iprod = threadIdx.x; iprod < block_dim3; iprod += blockDim.x) {
        int ix = iprod / block_dim2, iyz = iprod % block_dim2;
        int iy = iyz / block_dim, iz = iyz % block_dim;
        atomicAdd(&PF[block_dim * ix + iz], scale * K[block_dim * ix + iy] * P[block_dim * iy + iz]);  
    }
}


template <typename T>
__global__ void k_compute_Dinv_P_mmprod(int nnzb_prod, int block_dim, 
    const T *d_Dinv_vals, int *d_PF_rows, T *d_PF_vals) {
    
    // computes K * P => new P array as mat-mat product with pre-computed nonzero block pattern
    int iblock = blockIdx.x;
    // int iblock = threadIdx.x + blockIdx.x * blockDim.x;
    if (iblock >= nnzb_prod) return;

    int brow = d_PF_rows[iblock];
    // parallelize over the dense product
    int block_dim2 = block_dim * block_dim;
    int block_dim3 = block_dim2 * block_dim;
    // int ndense_prods = block_dim3; // do need blockDim.x > ndense_prods to work correctly 
    // aka need >= 216 block dim here to work well

    const T *Dinv = &d_Dinv_vals[block_dim2 * brow]; // since diagonal matrix brow = block
    T *PF = &d_PF_vals[block_dim2 * iblock];

    // load values of PF first into shared mem
    __shared__ T PF_vals0[36];
    for (int i = threadIdx.x; i < 36; i += blockDim.x) {
        PF_vals0[i] = PF[i];
    }
    __syncthreads();

    // now zero this block (since we'll be adding into it in a sec)
    for (int i = threadIdx.x; i < block_dim2; i += blockDim.x) {
        PF[i] = 0.0;
    }
    __syncthreads();

    // now do the product in-place into PF output
    for (int iprod = threadIdx.x; iprod < block_dim3; iprod += blockDim.x) {
        int ix = iprod / block_dim2, iyz = iprod % block_dim2;
        int iy = iyz / block_dim, iz = iyz % block_dim;
        atomicAdd(&PF[block_dim * ix + iz], Dinv[block_dim * ix + iy] * PF_vals0[block_dim * iy + iz]);  
    }
}

template <typename T>
__global__ void k_add_colored_submat_PFP(int color_nnzb, int block_dim, T omegaMC, int start_block, 
    const T *d_PF_vals, T *d_P_vals) {
    /* add colored rows of Dinv*PF=>PF previous step into P matrix as color smoother update */
    
    // P and PF both have K*P filled-in sparsity
    int tid = blockIdx.x;
    if (tid >= color_nnzb) return;
    int block_dim2 = block_dim * block_dim;
    int iblock = tid + start_block;
    const T *PF = &d_PF_vals[block_dim2 * iblock];
    T *P = &d_P_vals[block_dim2 * iblock];

    for (int ii = threadIdx.x; ii < block_dim2; ii += blockDim.x) {
        P[ii] += omegaMC * PF[ii]; // no atomic add, all separate
    }
}

template <typename T>
__global__ void k_copy_P_to_fillP(int nnzb, const int block_dim, const int *d_fill_map, const T *d_vals0, T *d_vals) {

    int iblock = blockIdx.x;
    if (iblock >= nnzb) return;

    int iblock_fill = d_fill_map[iblock];
    int block_dim2 = block_dim * block_dim;
    const T *vals0 = &d_vals0[block_dim2 * iblock];
    T *vals = &d_vals[block_dim2 * iblock_fill];

    // now read and copy    
    for (int ii = threadIdx.x; ii < block_dim2; ii += blockDim.x) {
        vals[ii] = vals0[ii];
    }
}

template <typename T>
__global__ void k_normalize_rows(const int nbrows, const int block_dim, const int max_inner_row, const int *d_rowp, 
        T *d_vals) {
    
    int brow = blockIdx.x;
    int thread = threadIdx.x; // local thread ids
    if (brow >= nbrows) return; // num block rows
    // __shared__ T row_norms_sq[6]; // 6 >= block_dim for all stuff in TACS (except hellinger-reissner element)
    // __shared__ T row_norms[6];
    // if (threadIdx.x == 0) {
    //     memset(row_norms, 0.0, 6 * sizeof(T));
    //     memset(row_norms_sq, 0.0, 6 * sizeof(T));
    // }
    // changing to row abs sums instead (so initially satisfied and doesn't mess up initial prolong starting point)
    __shared__ T row_abs_sums[6]; // 6 >= block_dim for all stuff in TACS (except hellinger-reissner element)
    if (threadIdx.x == 0) {
        memset(row_abs_sums, 0.0, 6 * sizeof(T));
    }
    __syncthreads();

    // each thread gets the number of blocks in this block row
    int start_block = d_rowp[brow], end_block = d_rowp[brow+1];
    int nblocks = end_block - start_block;
    int block_dim2 = block_dim * block_dim;
    int nvals = nblocks * block_dim2;

    // get the row norms squared for each dof in the block
    for (int i = threadIdx.x; i < nvals; i += blockDim.x) {
        int iblock = i / block_dim2 + start_block, idof = i % block_dim2;
        int irow = idof / block_dim; //, icol = idof % block_dim;
        T val = d_vals[block_dim2 * iblock + idof];
        // atomicAdd(&row_norms_sq[irow], val * val);
        atomicAdd(&row_abs_sums[irow], abs(val));
    }
    
    __syncthreads();

    // then compute the row norms from the row norms sq
    // for (int i = threadIdx.x; i < block_dim; i += blockDim.x) {
    //     row_norms[i] = sqrt(row_norms_sq[i]);
    // }
    // __syncthreads();

    // now normalize each row of the matrix
    for (int i = threadIdx.x; i < nvals; i += blockDim.x) {
        int iblock = i / block_dim2 + start_block, idof = i % block_dim2;
        int irow = idof / block_dim; //, icol = idof % block_dim;
        // T scale = 1.0 / row_norms[irow];
        T scale = 1.0 / row_abs_sums[irow];
        scale *= (irow < max_inner_row);
        d_vals[block_dim2 * iblock + idof] *= scale;
    }

}