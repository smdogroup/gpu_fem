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
__global__ void k_compute_Dinv_P_mmprod_hc(int nnzb_prod, int block_dim, 
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
__global__ void k_add_colored_submat_PFP_hc(int color_nnzb, int block_dim, T omegaMC, int start_block, 
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

__global__ void k_get_free_dof(const int n_bcs, int block_dim, int *bcs, 
    int *iperm, bool *free_dof_ptr) {
    // get the free dof (in solve order, so permuted)
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_bcs) return;
    int ibc = tid;
    int bc = bcs[ibc]; // BC dof in visualization order
    int bc_node = bc / block_dim, bc_dof = bc % block_dim;
    int bc_pnode = iperm[bc_node]; // BC node permtued to solve order 
    int pbc = block_dim * bc_pnode + bc_dof;

    free_dof_ptr[pbc] = false; // not a free DOF
}

#define EPS 1e-12
__device__ void cholesky6_shared(double* UTU)
{
    int t = threadIdx.x; // thread 0..5, one per row
    if (t >= 6) return;

    for (int k = 0; k < 6; k++)
    {
        // Thread k computes the diagonal
        double Lkk = 0.0;
        if (t == k)
        {
            double sum = 0.0;
            for (int s = 0; s < k; s++)
                sum += UTU[k*6 + s] * UTU[k*6 + s];
            Lkk = sqrt(max(UTU[k*6 + k] - sum, EPS));
            UTU[k*6 + k] = Lkk;
        }

        __syncthreads(); // make sure Lkk is visible

        // Threads t > k compute column k
        if (t > k && t < 6)
        {
            double sum = 0.0;
            for (int s = 0; s < k; s++)
                sum += UTU[t*6 + s] * UTU[k*6 + s];
            UTU[t*6 + k] = (UTU[t*6 + k] - sum) / UTU[k*6 + k];
        }

        __syncthreads();
    }

    // Zero upper triangle
    if (t < 6)
    {
        for (int j = t+1; j < 6; j++)
            UTU[t*6 + j] = 0.0;
    }
}

// Solve L*L^T x = b using in-place forward/backward substitution
// UTU: lower-triangular from cholesky6_shared
// b: 6-vector, replaced with solution x
__device__ void cholesky6_solve_inplace_shared(double* UTU, double* b)
{
    int t = threadIdx.x; // thread 0..5
    if (t >= 6) return;

    __shared__ double y[6];

    // Forward substitution L * y = b
    for (int i = 0; i < 6; i++)
    {
        if (t == i)
        {
            double sum = 0.0;
            for (int j = 0; j < i; j++)
                sum += UTU[i*6 + j] * y[j];
            y[i] = (b[i] - sum) / UTU[i*6 + i];
        }
        __syncthreads();
    }

    // Backward substitution L^T * x = y
    for (int i = 5; i >= 0; i--)
    {
        if (t == i)
        {
            double sum = 0.0;
            for (int j = i+1; j < 6; j++)
                sum += UTU[j*6 + i] * b[j]; // b[j] will hold solution x[j]
            b[i] = (y[i] - sum) / UTU[i*6 + i];
        }
        __syncthreads();
    }
}

// FIRST VERSION of orthog projector code => ran out of shared memory
// so splitting up the kernel into several steps first compute UTU and UTU_inv
// then SU kernel and then mat-mat-mat product kernel and update S
// THIS METHOD "k_orthog_projector_old" is DEPRECATED cause overloads shared memory
//   and I don't really trust the Cholesky factor code (untested) => using LU factor from cusparse now (so done outside kernel)
template <typename T>
__global__ void k_orthog_projector_old(const int nnodes_fine, const int block_dim, const T *R_vals,
    const bool *free_dof_ptr, const int *rowp, const int *cols, T *S_vals) {
    /* enforces the orthog projection to elim changes to fine rigid body modes 
        of the projector in projected steepest descent method*/

    // each thread block does orthog projection to one block row (or one nodal row)
    int bid = blockIdx.x; // node number or block row number
    if (bid >= nnodes_fine) return;
    int inode = bid; // in solve order
    
    // computing two sum of mat-mat products first (each 6x6), where S is the projection update delta(P) = S
    // 1) SU = sum_k S_{ik} U_{ik} where S_{ik}, U_{ik} in R^{6x6}
    // 2) UTU = sum_k U_{ik}^T U_{ik}

    __shared__ T SU[36]; // could declare less than 36 for smaller block size problems (but don't care now as = 36), + has to be compile time constant here
    __shared__ T UTU[36];
    __shared__ T U[36];
    __shared__ T V[36];
    __shared__ int Fi[6]; // fine node free DOF indicator (since only one fine node here, easy to just load into shared)
    if (threadIdx.x == 0) {
        memset(SU, 0.0, 36 * sizeof(T));
        memset(UTU, 0.0, 36 * sizeof(T));
    }
    for (int idof = threadIdx.x; idof < 6; idof += blockDim.x) {
        Fi[idof] = (int)free_dof_ptr[block_dim * inode + idof]; 
    }
    __syncthreads();

    // 1) first compute SU = sum_k S_{ik} U_{ik}, where U_{ik} = n_{ik} * R_k * F_i where i is fine node, k is coarse node,
    //  n_{ik} is sparsity indicator, R_k is the coarse rigid body mode of node k and F_i is the free DOF indicator 6x6 matrix
    //  F_i is just gonna be 0 or 1s on the diag so no mat-mat product needed for that.
    // compute the number of S blocks in this row (aka P)
    int start_block = rowp[bid], end_block = rowp[bid+1];
    int nblocks = end_block - start_block;
    // compute num products to compute in S*U (each thread will do different product + sum, looping over them together)
    int block_dim2 = block_dim * block_dim;
    int block_dim3 = block_dim2 * block_dim;
    int nprods = nblocks * block_dim3;
    for (int iprod = threadIdx.x; iprod < nprods; iprod += blockDim.x) {
        int loc_row_block = iprod / block_dim3;
        int iblock = loc_row_block + start_block; // block num in this row
        int jkl = iprod % block_dim3; // ijkljk product index (jkl < 216)
        // not quite sure the best order and how to make this most efficient (TBD), but gonna do l, then k, then j order
        int l = jkl % block_dim, jk = jkl / block_dim;
        int k = jk / block_dim, j = jk % block_dim; // with each of (j,k,l) in [0,6) integers
        
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
    __syncthreads(); // splits computation and registers here

    // 2) now compute UTU = sum_k (U_{ik})^T * U_{ik}, same number of products
    for (int iprod = threadIdx.x; iprod < nprods; iprod += blockDim.x) {
        int loc_row_block = iprod / block_dim3;
        int iblock = loc_row_block + start_block; // block num in this row
        int jkl = iprod % block_dim3; // ijkljk product index (jkl < 216)
        // not quite sure the best order and how to make this most efficient (TBD), but gonna do l, then k, then j order
        int l = jkl % block_dim, jk = jkl / block_dim;
        int k = jk / block_dim, j = jk % block_dim; // with each of (j,k,l) in [0,6) integers
        
        // get the coarse node for this U 6x6 matrix in R
        int ic = cols[loc_row_block]; // in solve order since R_vals is also in solve order and so is S and P

        // once U = {[U1,U2,...,]}_{iK} the fine node i and coarse node K have been loaded as 6x6 matrix, we compute the 
        // 6x6 mat-mat product (UTU)_{jl} = U_{jk}^T * U_{kl} = U_{kj} U_{kl}
        // now load the correct part of the rigid body modes to compute each U_{kj} and U_{kl} value
        T U_kl = R_vals[block_dim2 * ic + block_dim * k + l] * Fi[l]; // Fi is 6x6 which then
        T U_kj = R_vals[block_dim2 * ic + block_dim * k + j] * Fi[j]; // zeroes out certain columns in U if not free node
        //Fi itself is 6x6 the l and j above are indexing hte 6x6 matrix of (Fi) => 6x6

        // now multiply and add into SU
        atomicAdd(&UTU[block_dim * j + l], U_kj * U_kl);
        // since includes product Fi[j] * Fi[l], UTU will have row and cols zero for Dirichlet BC dof.
    }
    __syncthreads(); // splits computation and registers here

    // now need to compute the 6x6 pseudo-inverse (UTU)^+ = ((UTU)^2)^{-1} * UTU (to handle the dirichlet bcs correctly)
    // while the pseudo-inv is mathematically correct way to do it.. it may be easier to add small 1e-12 value or something to the diag
    // and do cholesky solve or something.. would be numerically equivalent and stable I think..

    // 3.0) how would you do the cholesky solve then? Need it to be exact..
    //  not great to do (UTU)^2 * M = UTU * U^T since (UTU)^2 destroys condition number.
    //  so instead I'm gonna do (UTU + eps * I)^-1 instead of pinv (shouild work due to fully zero each row and col of constr DOF)
    //  will check the linear system is solved later
    // first add small eps*I to UTU (1e-12 should be well below it given typical mesh sizes and the I in it)
    for (int i = threadIdx.x; i < block_dim; i += blockDim.x) {
        UTU[block_dim * i + i] += 1e-12;
    } 
    __syncthreads();

    // 3) now do a 6x6 cholesky factorization of (UTU+eps*I) approx LL^T in place in UTU storage
    cholesky6_shared(UTU);
    __syncthreads();


    // 4) loop over each block coarse node (with all threads) => updating S values
    int t = threadIdx.x;
    for (int loc_block = 0; loc_block < nblocks; loc_block++) {
        int ic = cols[loc_block]; // coarse node number
        int iblock = loc_block + start_block;

        // 4.1) now we'll do the linear solve of (LL^T)^{-1} * U^T => V (with six right-hand-sides)
        if (t < 6)
        {
            // Forward substitution: L * Y = U^T
            // Y is stored temporarily in V
            for (int col = 0; col < 6; col++)
            {
                T sum = 0.0;
                for (int k = 0; k < t; k++) sum += UTU[t*6 + k] * V[k*6 + col];
                T U_ct = R_vals[block_dim2 * ic + block_dim * col + t] * Fi[col]; // U^T[row,col] = U[col,row]
                V[t*6 + col] = (U_ct - sum) / UTU[t*6 + t]; 
            }
        }
        __syncthreads();

        // 4.2) Backward substitution: L^T * X = Y, overwrite V
        if (t < 6)
        {
            for (int col = 0; col < 6; col++)
            {
                T sum = 0.0;
                for (int k = t+1; k < 6; k++)
                    sum += UTU[k*6 + t] * V[k*6 + col];
                V[t*6 + col] = (V[t*6 + col] - sum) / UTU[t*6 + t];
            }
        }
        __syncthreads();

        // 5) update all S values in this nodal row with the projector S -= SU * V
        //      this just does one block product here (one fine and coarse node pair)
        for (int jkl = threadIdx.x; jkl < block_dim3; jkl += blockDim.x) {
            int l = jkl % block_dim, jk = jkl / block_dim;
            int k = jk / block_dim, j = jk % block_dim; // with each of (j,k,l) in [0,6) integers

            // get V_{kl} value
            T V_kl = V[6 * k + l];
            T SU_jk = SU[6 * j + k];
            atomicAdd(&S_vals[block_dim2 * iblock + block_dim * j + l], -1.0 * SU_jk * V_kl);
        }

    } // end of coarse node block loop
    


    

    // END OF kernel!
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