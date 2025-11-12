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
    for (int iprod = threadIdx.x; iprod < block_dim2; iprod += blockDim.x) {
        int ix = iprod / block_dim, iy = iprod % block_dim;
        PF[block_dim * ix + iy] = 0.0;
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
__global__ void k_add_colored_submat_PFP(int color_nnzb, int block_dim, T omegaMC, int start_block, int end_block,
    const T *d_PF_vals, T *d_P_vals) {
    /* add colored rows of Dinv*PF=>PF previous step into P matrix as color smoother update */
    
    // P and PF both have K*P filled-in sparsity
    int iblock = blockIdx.x;
    if (iblock >= color_nnzb) return;
    int block_dim2 = block_dim * block_dim;
    const T *PF = &d_PF_vals[block_dim2 * start_block];
    T *P = &d_P_vals[block_dim2 * end_block];

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