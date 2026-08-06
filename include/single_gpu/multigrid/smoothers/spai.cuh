#pragma once



template <typename T>
__global__ void k_abs_value_col_sums(int nnzb, int block_dim, int *d_cols, T *d_vals, T *d_colsums) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int block_dim2 = block_dim * block_dim;
    int nvals = nnzb * block_dim2;
    if (tid >= nvals) return;

    int block_ind = tid / block_dim2;
    int inner_col = tid % block_dim;
    int block_col = d_cols[block_ind];
    int col = block_dim * block_col + inner_col;
    T abs_val = abs(d_vals[tid]);
    atomicAdd(&d_colsums[col], abs_val);
}

template <typename T>
__global__ void k_vec_max(int N, T *vals, T *out) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= N) return;
    atomicMax(out, vals[tid]);
}

template <typename T>
__global__ static void k_add_diag_matrix(const int nnodes, const int block_dim, const T value, const int *diagp, T *vals) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int nvals = nnodes * block_dim;
    if (tid >= nvals) return;

    // printf("node %d / nnodes %d\n", tid / block_dim2, nnodes);

    int node_ind = tid / block_dim;
    int block_ind = diagp[node_ind];
    int i = tid % block_dim;
    // int inner_col = inner_dof % block_dim;
    // int inner_row = inner_dof / block_dim;
    // bool is_diag = inner_row == inner_col;
    // vals[block_dim2 * block_ind + inner_dof] += is_diag ? value : 0.0;
    vals[block_dim * block_dim * block_ind + block_dim * i + i] += value;
    // atomicAdd(&vals[block_dim * block_dim * block_ind + block_dim * i + i], value);
}


// copy of method from smoothers.h in multigrid include
template <typename T>
__global__ static void k_compute_mat_mat_prod2(int nnzb_prod, int block_dim, T scale, 
    int *d_K_blocks, int *d_P_blocks, int *d_PF_blocks,
    const T *d_K_vals, const T *d_P_vals, T *d_PF_vals) {
    /* compute -K*P => Z  sparse mat-mat product for matrix smoothing */
    
    // computes K * P => new P array as mat-mat product with pre-computed nonzero block pattern
    int iprod = blockIdx.x;
    // int iprod = threadIdx.x + blockIdx.x * blockDim.x;
    if (iprod >= nnzb_prod) return;

    // if (threadIdx.x == 0) {
    //     printf("iprod %d\n", iprod);
    // }

    // get block indices in each matrix
    int bind_P = d_P_blocks[iprod];
    int bind_K = d_K_blocks[iprod];
    int bind_PF = d_PF_blocks[iprod];
    __syncthreads();

    // if (threadIdx.x == 0) {
    //     printf("iprod %d, P block %d, K block %d, PF block %d\n", iprod, bind_P, bind_K, bind_PF);
    // }

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
__global__ void k_add_colored_submat_PFP2(int color_nnzb, int block_dim, T omegaMC, int start_block, 
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