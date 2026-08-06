#pragma once

// TODO : also make rigid body modes for linear structure here

template <typename T>
__global__ void k_compute_linear_rigid_body_modes(const int nnodes, const int block_dim, const T *xpts, T *rbm) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // int N = nnodes * block_dim;
    if (tid >= nnodes) return;

    // also assumes no nodal permutations done on sparsity / kmat (stiffness matrix)
    int block_dim2 = block_dim * block_dim;
    int inode = tid;
    
    // get x,y,z coords for this node
    T x = xpts[3 * inode];
    T y = xpts[3 * inode + 1];
    T z = xpts[3 * inode + 2];

    /* compute each of six rigid body modes in BSR format */

    // that is N x 6 rigid body modes are stored in nnodes x 6 x 6 BSR format (so each node is 6x6 matrix)
    // and nnzb = nnodes (even though I store it in a vector)
    T *block_rbm = &rbm[block_dim2 * inode];

    // x translation (mode 0)
    block_rbm[block_dim * 0 + 0] = 1.0;
    // y translation (mode 1)
    block_rbm[block_dim * 1 + 1] = 1.0; 
    // z translation (mode 2)
    block_rbm[block_dim * 2 + 2] = 1.0;

    // vw or thx rotation (mode 3)
    block_rbm[block_dim * 1 + 3] = -z;
    block_rbm[block_dim * 2 + 3] = +y;
    block_rbm[block_dim * 3 + 3] = 1.0;

    // uw or thy rotation (mode 4)
    block_rbm[block_dim * 0 + 4] = +z;
    block_rbm[block_dim * 2 + 4] = -x;
    block_rbm[block_dim * 4 + 4] = 1.0;

    // uv or thz rotation (mode 5)
    block_rbm[block_dim * 0 + 5] = -y;
    block_rbm[block_dim * 1 + 5] = +x;
    block_rbm[block_dim * 5 + 5] = 1.0;
}
template <typename T>
__global__ void k_orthog_projector_computeUTU(
    const int nnodes_fine, const int block_dim, const T *R_vals,
    const bool *free_dof_ptr, const int *rowp, const int *cols, T *UTU_vals) {

    // part 1 - compute UTU resultant matrices
    // UTU = sum_K U_K^T U_K, where U_K = R_K * F
    // with F a diagonal free-dof mask for the fine node

    int bid = blockIdx.x;
    if (bid >= nnodes_fine) return;
    int inode = bid;

    // NOTE: assumes block_dim == 6
    __shared__ int Fi[6];
    for (int idof = threadIdx.x; idof < 6; idof += blockDim.x) {
        Fi[idof] = (int)free_dof_ptr[block_dim * inode + idof];
    }
    __syncthreads();

    int start_block = rowp[bid];
    int end_block   = rowp[bid + 1];
    int nblocks     = end_block - start_block;

    int block_dim2 = block_dim * block_dim;
    int block_dim3 = block_dim2 * block_dim;
    int nprods     = nblocks * block_dim3;

    T *UTU = &UTU_vals[block_dim2 * inode];

    // safer full init
    for (int i = threadIdx.x; i < block_dim2; i += blockDim.x) {
        UTU[i] = T(0);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < block_dim; i += blockDim.x) {
        UTU[block_dim * i + i] = T(1e-12);
    }
    __syncthreads();

    // accumulate UTU[j,l] += sum_K sum_k U[k,j] * U[k,l]
    for (int iprod = threadIdx.x; iprod < nprods; iprod += blockDim.x) {
        int loc_row_block = iprod / block_dim3;
        int iblock        = start_block + loc_row_block;

        int jkl = iprod % block_dim3;
        int l   = jkl % block_dim;
        int jk  = jkl / block_dim;
        int k   = jk % block_dim;
        int j   = jk / block_dim;

        int ic = cols[iblock];   // FIXED

        T U_kl = R_vals[block_dim2 * ic + block_dim * k + l] * Fi[l];
        T U_kj = R_vals[block_dim2 * ic + block_dim * k + j] * Fi[j];

        atomicAdd(&UTU[block_dim * j + l], U_kj * U_kl);
    }
}


template <typename T>
__global__ void k_orthog_projector_computeSU(
    const int nnodes_fine, const int block_dim, const T *R_vals,
    const bool *free_dof_ptr, const int *rowp, const int *cols,
    const T *S_vals, T *SU_vals) {

    // Each thread block handles one fine-node block row.
    // Computes, for that node:
    //   SU = sum_over_neighbor_blocks ( S_ik * U_k )
    // where U_k = R_k * F, with F diagonal from free_dof_ptr.
    // Entrywise:
    //   SU[j,l] += sum_k S[j,k] * U[k,l]

    int bid = blockIdx.x;
    if (bid >= nnodes_fine) return;
    int inode = bid;

    // NOTE: this assumes block_dim == 6
    __shared__ int Fi[6];
    for (int idof = threadIdx.x; idof < 6; idof += blockDim.x) {
        Fi[idof] = (int)free_dof_ptr[block_dim * inode + idof];
    }
    __syncthreads();

    int start_block = rowp[bid];
    int end_block   = rowp[bid + 1];
    int nblocks     = end_block - start_block;

    int block_dim2 = block_dim * block_dim;
    int block_dim3 = block_dim2 * block_dim;
    int nprods     = nblocks * block_dim3;

    T *SU = &SU_vals[block_dim2 * inode];

    for (int iprod = threadIdx.x; iprod < nprods; iprod += blockDim.x) {
        int loc_row_block = iprod / block_dim3;
        int iblock        = start_block + loc_row_block;

        int jkl = iprod % block_dim3;
        int l   = jkl % block_dim;
        int jk  = jkl / block_dim;
        int k   = jk % block_dim;
        int j   = jk / block_dim;

        // Neighbor coarse/fine node index for this block entry
        int ic = cols[iblock];

        // U[k,l] = R[k,l] * F[l]
        T R_kl = R_vals[block_dim2 * ic + block_dim * k + l];
        R_kl *= Fi[l];

        // S[j,k] for this block
        T S_jk = S_vals[block_dim2 * iblock + block_dim * j + k];

        atomicAdd(&SU[block_dim * j + l], S_jk * R_kl);
    }
}

template <typename T>
__global__ void k_orthog_projector_removeRowSums(
    const int nnodes_fine, const int block_dim, const T *R_vals,
    const bool *free_dof_ptr, const int *rowp, const int *cols,
    const T *SU_vals, const T *UTUinv_vals, T *S_vals) {

    // part 3 - remove row sums with
    // S = S0 - (SU) * (UTUinv) * U^T

    int bid = blockIdx.x;
    if (bid >= nnodes_fine) return;
    int inode = bid;

    // NOTE: assumes block_dim == 6
    __shared__ T V[36];
    __shared__ int Fi[6];

    for (int idof = threadIdx.x; idof < 6; idof += blockDim.x) {
        Fi[idof] = (int)free_dof_ptr[block_dim * inode + idof];
    }

    for (int i = threadIdx.x; i < 36; i += blockDim.x) {
        V[i] = T(0);
    }
    __syncthreads();

    int start_block = rowp[bid];
    int end_block   = rowp[bid + 1];
    int nblocks     = end_block - start_block;

    int block_dim2 = block_dim * block_dim;
    int block_dim3 = block_dim2 * block_dim;
    int nprods     = nblocks * block_dim3;

    const T *SU     = &SU_vals[block_dim2 * bid];
    const T *UTUinv = &UTUinv_vals[block_dim2 * bid];

    // 1) V = SU * UTUinv
    for (int jkl = threadIdx.x; jkl < block_dim3; jkl += blockDim.x) {
        int l  = jkl % block_dim;
        int jk = jkl / block_dim;
        int k  = jk % block_dim;
        int j  = jk / block_dim;

        T SU_jk     = SU[block_dim * j + k];
        T UTUinv_kl = UTUinv[block_dim * k + l];

        atomicAdd(&V[block_dim * j + l], SU_jk * UTUinv_kl);
    }
    __syncthreads();

    // 2) S -= V * U^T
    for (int iprod = threadIdx.x; iprod < nprods; iprod += blockDim.x) {
        int loc_row_block = iprod / block_dim3;
        int iblock        = start_block + loc_row_block;

        int jkl = iprod % block_dim3;
        int l   = jkl % block_dim;
        int jk  = jkl / block_dim;
        int k   = jk % block_dim;
        int j   = jk / block_dim;

        int ic = cols[iblock]; 

        // (U^T)_{kl} = U_{lk} = R_{lk} * F_k
        T UT_kl = R_vals[block_dim2 * ic + block_dim * l + k] * Fi[k];
        T V_jk  = V[block_dim * j + k];

        atomicAdd(&S_vals[block_dim2 * iblock + block_dim * j + l],
                  -(V_jk * UT_kl));
    }
}