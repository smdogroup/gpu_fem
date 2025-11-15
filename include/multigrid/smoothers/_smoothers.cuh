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
__global__ static void k_computeL1BlockDiags(const int kmat_nnzb, const int block_dim, const int *kmat_rows,
                                                       const T *kmat_vals, T *diag_vals) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int block_dim2 = block_dim * block_dim;
    int kmat_nvals = kmat_nnzb * block_dim2;
    if (tid >= kmat_nvals) return;
    int node_block = tid / block_dim2;
    int node_row = kmat_rows[node_block];
    int ii = tid % block_dim2;
    // int i = ii / block_dim, j = ii % block_dim;

    // compute absolute value row-sums
    T abs_val = abs(kmat_vals[tid]);
    atomicAdd(&diag_vals[block_dim2 * node_row + ii], abs_val);
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