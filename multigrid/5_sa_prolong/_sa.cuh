#pragma once

// helper kernels for the sa plate and potentially other examples

template <typename T>
__global__ void k_compute_P_K_P_mmprod(int nnzb_prod, int block_dim, T scale, 
    int *d_K_blocks, int *d_P_blocks, int *d_PF_blocks,
    const T *d_K_vals, const T *d_P_vals, T *d_PF_vals) {
    
    // computes K * P => new P array as mat-mat product with pre-computed nonzero block pattern
    
    int iprod = threadIdx.x + blockIdx.x * blockDim.x;
    if (iprod > nnzb_prod) return;

    // get block indices in each matrix
    int bind_P = d_P_blocks[iprod];
    int bind_K = d_K_blocks[iprod];
    int bind_PF = d_PF_blocks[iprod];
    __syncthreads();

    // parallelize over the dense product
    int block_dim2 = block_dim * block_dim;
    int block_dim3 = block_dim2 * block_dim;
    int ndense_prods = block_dim3; // do need blockDim.x > ndense_prods to work correctly 
    // aka need >= 216 block dim here to work well

    // get data at start of block each matrix
    const T *K = &d_K_vals[block_dim2 * bind_K];
    const T *P = &d_P_vals[block_dim2 * bind_P];
    T *PF = &d_PF_vals[block_dim2 * bind_PF];


    // parallelize over the whole ijk dense mat-mat product
    for (int iprod = 0; iprod < block_dim3; iprod++) {
        int ix = iprod / block_dim2, iyz = iprod % block_dim2;
        int iy = iyz / block_dim, iz = iyz % block_dim;
        atomicAdd(&PF[block_dim * ix + iz], scale * K[block_dim * ix + iy] * P[block_dim * iy + iz]);  
    }

}

