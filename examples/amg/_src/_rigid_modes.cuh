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
__global__ void k_orthog_projector_computeUTU(const int nnodes_fine, const int block_dim, const T *R_vals,
    const bool *free_dof_ptr, const int *rowp, const int *cols, T *UTU_vals) {
    /* part 1 - compute UTU resultant matrices (can be done before any smoothing steps)
        * enforces the orthog projection to elim changes to fine rigid body modes 
          of the projector in projected steepest descent method*/

    // each thread block does orthog projection to one block row (or one nodal row)
    int bid = blockIdx.x; // node number or block row number
    if (bid >= nnodes_fine) return;
    int inode = bid; // in solve order
    
    __shared__ int Fi[6]; // fine node free DOF indicator (since only one fine node here, easy to just load into shared)
    for (int idof = threadIdx.x; idof < 6; idof += blockDim.x) {
        Fi[idof] = (int)free_dof_ptr[block_dim * inode + idof]; 
    }
    __syncthreads();

    int start_block = rowp[bid], end_block = rowp[bid+1];
    int nblocks = end_block - start_block;
    // compute num products to compute in U^T * U (each thread will do different product + sum, looping over them together)
    int block_dim2 = block_dim * block_dim;
    int block_dim3 = block_dim2 * block_dim;
    int nprods = nblocks * block_dim3;

    // DEBUG
    // if (inode == 1 && threadIdx.x == 0) {
    //     printf("Fi on node 1: ");
    //     printVec<int>(6, Fi);
    // }


    // first add in 1e-12 to each diagonal entry
    T *UTU = &UTU_vals[block_dim2 * inode];
    for (int i = threadIdx.x; i < block_dim; i += blockDim.x) {
        T eps = 1e-12;
        UTU[block_dim * i + i] += eps; // no need atomic add, no overlap here
    }
    __syncthreads();

    // now compute UTU = sum_k (U_{ik})^T * U_{ik}, same number of products
    for (int iprod = threadIdx.x; iprod < nprods; iprod += blockDim.x) {
        int loc_row_block = iprod / block_dim3;
        int iblock = loc_row_block + start_block; // block num in this row
        int jkl = iprod % block_dim3; // ijkljk product index (jkl < 216)
        // not quite sure the best order and how to make this most efficient (TBD), but gonna do l, then k, then j order
        int l = jkl % block_dim, jk = jkl / block_dim;
        int k = jk % block_dim, j = jk / block_dim; // with each of (j,k,l) in [0,6) integers
        
        // get the coarse node for this U 6x6 matrix in R
        int ic = cols[loc_row_block]; // in solve order since R_vals is also in solve order and so is S and P

        // once U = {[U1,U2,...,]}_{iK} the fine node i and coarse node K have been loaded as 6x6 matrix, we compute the 
        // 6x6 mat-mat product (UTU)_{jl} = U_{jk}^T * U_{kl} = U_{kj} U_{kl}
        // now load the correct part of the rigid body modes to compute each U_{kj} and U_{kl} value
        T U_kl = R_vals[block_dim2 * ic + block_dim * k + l] * Fi[l]; // Fi is 6x6 which then
        T U_kj = R_vals[block_dim2 * ic + block_dim * k + j] * Fi[j]; // zeroes out certain columns in U if not free node
        //Fi itself is 6x6 the l and j above are indexing hte 6x6 matrix of (Fi) => 6x6

        // DEBUG
        // if (inode == 1) {
        //     printf("U[%d,%d] * U[%d,%d]: %.4e * %.4e => (+ %.4e) with ic=%d\n", k, j, k, l, U_kj, U_kl, U_kj * U_kl, ic);
        // }

        // now multiply and add into UTU
        atomicAdd(&UTU[block_dim * j + l], U_kj * U_kl);
        // since includes product Fi[j] * Fi[l], UTU will have row and cols zero for Dirichlet BC dof.
    }
    __syncthreads();

    // also add in 
}


template <typename T>
__global__ void k_orthog_projector_computeSU(const int nnodes_fine, const int block_dim, const T *R_vals,
    const bool *free_dof_ptr, const int *rowp, const int *cols, const T *S_vals, T *SU_vals) {
    /* part 2 - compute SU resultant 6x6 matrices for each fine node.
        * enforces the orthog projection to elim changes to fine rigid body modes 
          of the projector in projected steepest descent method*/

    // each thread block does orthog projection to one block row (or one nodal row)
    int bid = blockIdx.x; // node number or block row number
    if (bid >= nnodes_fine) return;
    int inode = bid; // in solve order
    
    __shared__ int Fi[6]; // fine node free DOF indicator (since only one fine node here, easy to just load into shared)
    for (int idof = threadIdx.x; idof < 6; idof += blockDim.x) {
        Fi[idof] = (int)free_dof_ptr[block_dim * inode + idof]; 
    }
    __syncthreads();

    int start_block = rowp[bid], end_block = rowp[bid+1];
    int nblocks = end_block - start_block;
    // compute num products to compute in U^T * U (each thread will do different product + sum, looping over them together)
    int block_dim2 = block_dim * block_dim;
    int block_dim3 = block_dim2 * block_dim;
    int nprods = nblocks * block_dim3;

    T *SU = &SU_vals[block_dim2 * inode];

    // now compute SU = sum_k S_{ik} * U_{ik}, same number of products
    for (int iprod = threadIdx.x; iprod < nprods; iprod += blockDim.x) {
        int loc_row_block = iprod / block_dim3;
        int iblock = loc_row_block + start_block; // block num in this row
        int jkl = iprod % block_dim3; // ijkljk product index (jkl < 216)
        // not quite sure the best order and how to make this most efficient (TBD), but gonna do l, then k, then j order
        int l = jkl % block_dim, jk = jkl / block_dim;
        int k = jk % block_dim, j = jk / block_dim; // with each of (j,k,l) in [0,6) integers
        
        // load the correct value of R_{jk} the coarse rigid body modes
        int ic = cols[loc_row_block]; // in solve order since R_vals is also in solve order and so is S and P
        T R_kl = R_vals[block_dim2 * ic + block_dim * k + l]; // R_{kl} of coarse node ic
        // multiply R_kl by F_l (zeroes out columns of R aka individual rigid body modes)
        R_kl *= Fi[l]; // equal to R*F 6x6 matrix here (but one entry of it)

        // load the correct value of S_{ij} = dP_{ij} the projection update 
        T S_jk = S_vals[block_dim2 * iblock + block_dim * j + k]; // S_{jk} of fine node bid aka i

        // now multiply and add into SU
        atomicAdd(&SU[block_dim * j + l], S_jk * R_kl);
    }
}

template <typename T>
__global__ void k_orthog_projector_removeRowSums(const int nnodes_fine, const int block_dim, const T *R_vals,
    const bool *free_dof_ptr, const int *rowp, const int *cols, const T *SU_vals, const T *UTUinv_vals, T *S_vals) {
    /* part 3 - remove row sums with S = S_0 - (SU) * (UTUinv) * U^T triple product and subtract
        * final kernel to enforces the orthog projection to elim changes to fine rigid body modes 
          of the projector in projected steepest descent method*/

    // each thread block does orthog projection to one block row (or one nodal row)
    int bid = blockIdx.x; // node number or block row number
    if (bid >= nnodes_fine) return;
    int inode = bid; // in solve order
    
    __shared__ T V[36];
    __shared__ int Fi[6]; // fine node free DOF indicator (since only one fine node here, easy to just load into shared)
    for (int idof = threadIdx.x; idof < 6; idof += blockDim.x) {
        Fi[idof] = (int)free_dof_ptr[block_dim * inode + idof]; 
    }
    if (threadIdx.x == 0) {
        memset(V, 0.0, 36 * sizeof(T));
    }
    __syncthreads();

    int start_block = rowp[bid], end_block = rowp[bid+1];
    int nblocks = end_block - start_block;
    // compute num products to compute in S*U (each thread will do different product + sum, looping over them together)
    int block_dim2 = block_dim * block_dim;
    int block_dim3 = block_dim2 * block_dim;
    int nprods = nblocks * block_dim3;

    // get pointers for this fine node SU and UTUinv
    const T *SU = &SU_vals[block_dim2 * bid];
    const T *UTUinv = &UTUinv_vals[block_dim2 * bid];

    // 1) compute SU * UTUinv => V matrix in shared memory for each fine node resultant
    for (int jkl = threadIdx.x; jkl < block_dim3; jkl += blockDim.x) {
        // not quite sure the best order and how to make this most efficient (TBD), but gonna do l, then k, then j order
        int l = jkl % block_dim, jk = jkl / block_dim;
        int k = jk % block_dim, j = jk / block_dim; // with each of (j,k,l) in [0,6) integers

        // get SU and UTUinv values for mat-mat product
        T SU_jk = SU[block_dim * j + k];
        T UTUinv_kl = UTUinv[block_dim * k + l];

        // then add into V = SU * UTUinv shared mem location
        atomicAdd(&V[block_dim * j + l], SU_jk * UTUinv_kl);
    }
    __syncthreads();

    // 2) update all S values in this nodal row with the projector S -= V * U^T
    for (int iprod = threadIdx.x; iprod < nprods; iprod += blockDim.x) {
        int loc_row_block = iprod / block_dim3;
        int iblock = loc_row_block + start_block; // block num in this row
        int jkl = iprod % block_dim3; // ijkljk product index (jkl < 216)
        // not quite sure the best order and how to make this most efficient (TBD), but gonna do l, then k, then j order
        int l = jkl % block_dim, jk = jkl / block_dim;
        int k = jk % block_dim, j = jk / block_dim; // with each of (j,k,l) in [0,6) integers

        // get the value of U^T in this fine-coarse node pair
        int ic = cols[loc_row_block];
        // UT_{kl} = U_{lk} where kl are just 6x6 block indices not fine or coarse node
        // still the Fi matrix modifies columns of U so k column modified here not l due to transpose
        T UT_kl = R_vals[block_dim2 * ic + block_dim * l + k] * Fi[k];

        // get V_{jk} value for the mat-mat product
        T V_jk = V[block_dim * j + k];

        // final S -= V * U^T remove row sum step
        atomicAdd(&S_vals[block_dim2 * iblock + block_dim * j + l], -1.0 * V_jk * UT_kl);
    }
    // END OF kernel!
}