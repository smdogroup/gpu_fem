#pragma once

template <typename T>
__global__ static void k_copyBlockDiagFromBsrMat(int nnodes, int block_dim, int *kmat_diagp,
                                                       T *kmat_vals, T *diag_vals) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int block_dim2 = block_dim * block_dim;
    int diag_nnz = nnodes * block_dim2;
    if (tid < diag_nnz) {
        int node_ind = tid / block_dim2, inner_block_ind = tid % block_dim2;
        int kmat_nzb_ind = kmat_diagp[node_ind];
        int kmat_nz_ind = block_dim2 * kmat_nzb_ind + inner_block_ind;

        T val = kmat_vals[kmat_nz_ind];
        diag_vals[tid] = val;
    }
}


template <typename T>
__global__
void k_computeBlockL1Norms(int kmat_nnzb, int block_dim,
                           const int *kmat_rows,
                           const int *kmat_cols,
                           const T *kmat_vals,
                           T *block_norms) {

    
    int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= kmat_nnzb) return;

    int row = kmat_rows[bid];
    int col = kmat_cols[bid];
    bool diagonal = (row == col);

    int b = block_dim;
    int block_dim2 = b * b;

    const T* B = kmat_vals + bid * block_dim2;

    // compute block 1-norm = max column sum
    T maxcol = 0;
    for (int j = 0; j < b; j++) {
        T sum = 0;
        for (int i = 0; i < b; i++)
            sum += abs(B[i*b + j]);
        maxcol = max(maxcol, sum);
    }

    block_norms[bid] = maxcol * (!diagonal); // so diagonal will have zero value
}

template <typename T>
__global__
void k_accumulateBlockL1ToDiag(int kmat_nnzb, int block_dim,
                               const int *kmat_rows,
                               const T *block_norms,
                               T *diag_vals)
{
    int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= kmat_nnzb) return;

    int brow = kmat_rows[bid];

    // Add accum * Identity(b)
    int b = block_dim;
    T* D = diag_vals + brow * b * b;

    for (int d = 0; d < b; d++) {
        atomicAdd(&D[d * b + d], block_norms[bid]); // note block norms is zero on diag
    }
}


template <typename T>
__global__ static void k_setBlockUnitVec(int nnodes, int block_dim, int ii, T *vec) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < nnodes) {
        int inode = tid;
        vec[block_dim * inode + ii] = 1.0;
    }
}

template <typename T>
__global__ static void k_setLUinv_operator(int nnodes, int block_dim, int ii, const T *rhs, T *Dinv_vals) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int block_dim2 = block_dim * block_dim;
    int nvec = nnodes * block_dim;
    if (tid < nvec) {
        int inode = tid / block_dim;
        int idof = tid % block_dim;

        // copy values of this rhs result from (LU)^-1 triang solve linear operator into Dinv
        Dinv_vals[block_dim2 * inode + block_dim * idof + ii] = rhs[tid];
    }
}

template <typename T>
__global__ void k_compute_mat_mat_prod(int nnzb_prod, int block_dim, T scale, 
    int *d_K_blocks, int *d_P_blocks, int *d_PF_blocks,
    const T *d_K_vals, const T *d_P_vals, T *d_PF_vals) {
    /* compute -K*P => Z  sparse mat-mat product for matrix smoothing */
    
    
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
__global__ static void k_copy_color_submat(const int nnodes, const int submat_nnzb, const int start_col, const int block_dim, 
    const int *d_submat_rows, const int *d_submat_cols, const int *d_kmat_rowp, const int *d_kmat_cols, const T *d_kmat_vals, T *d_submat_vals) {

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int block_dim2 = block_dim * block_dim;
    int submat_nnz = submat_nnzb * block_dim2;
    int block_ind = blockIdx.x;
    if (tid >= submat_nnz) return; // escape condition

    // get the row and column of this block nodal value
    int brow = d_submat_rows[block_ind];
    int bcol = d_submat_cols[block_ind] + start_col;

    // find the pointer in kmat memory for this block node in Kmat
    int _jp = 0;
    // repeated for every thread currently...
    for (int jp = d_kmat_rowp[brow]; jp < d_kmat_rowp[brow+1]; jp++) {
        int bcol2 = d_kmat_cols[jp];
        _jp += jp * (bcol2 == bcol); // GPU friendly if condition and set value here..
    }
    
    // if (threadIdx.x == 0 && blockIdx.x < 3) {
    //     printf("brow %d, bcol %d, block_ind %d, _jp %d\n", brow, bcol, block_ind, _jp);
    // }

    // now copy values in the nodal block
    int iloc = threadIdx.x;
    d_submat_vals[block_dim2 * block_ind + iloc] = d_kmat_vals[block_dim2 * _jp + iloc];
}