// ===========================================
// FETI-DP kernels on CUDA GPU
// ===========================================
#pragma once

template <typename T>
__global__ static void k_bsrmv_transpose_ax(const int nnzb, const int block_dim, const int *rows, const int *cols, 
    const T *vals, const T *fine_vec_in, T a, T *coarse_vec_out) {
    /* transpose product like u_c = P^T * u_f (since cusparse doesn't have bsrmv_transpose option) */
    // this way we don't have to store a transposed copy R = P^T
    // assumes vectors are in solve order (so no permutations during product)
    // also the fact we use rows instead of rowp (may be more efficient than cusparse (less reads))

    // parallelizes over each product individually
    // can explore different methods later
    int block_dim2 = block_dim * block_dim;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int nprods = nnzb * block_dim2;
    if (tid >= nprods) return;

    // loops through Pmat in BSR order (during product)
    int block_id = tid / block_dim2;
    int block_row = rows[block_id], block_col = cols[block_id];
    int ii_prod = tid % block_dim2; // which of the block_dim^2 products we do for this thread
    int ii_fine = ii_prod / block_dim, ii_coarse = ii_prod % block_dim; // not sure which order best here yet

    // get the fine vec and mat value for this thread
    T f_val = fine_vec_in[block_dim * block_row + ii_fine];
    T mat_val = vals[block_dim2 * block_id + ii_prod];

    // now add into the output
    T new_val = mat_val * f_val * a;
    atomicAdd(&coarse_vec_out[block_dim * block_col + ii_coarse], new_val);
}

template <typename T>
__global__ static void k_addVecSmallerIn(const int in_nnodes, const int block_dim, const int *d_in_imap, 
    const T *x, T *y, const T a) {
    int N_in = in_nnodes * block_dim;
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= N_in) return;

    int node_in = tid / block_dim;
    int node_out = d_in_imap[node_in];
    int idof = tid % block_dim;
    int dof_in = block_dim * node_in + idof;
    int dof_out = block_dim * node_out + idof;

    atomicAdd(&y[dof_out], a * x[dof_in]);
}

template <typename T>
__global__ static void k_addVecSmallerOut(const int out_nnodes, const int block_dim, const int *d_out_imap, 
    const T *x, T *y, const T a) {
    int N_out = out_nnodes * block_dim;
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= N_out) return;

    int node_out = tid / block_dim;
    int node_in = d_out_imap[node_out];
    int idof = tid % block_dim;
    int dof_in = block_dim * node_in + idof;
    int dof_out = block_dim * node_out + idof;

    atomicAdd(&y[dof_out], a * x[dof_in]);
}

template <typename T>
__global__ static void k_addVec_IEVtoVc(const int V_nnodes, const int block_dim, const int *d_IEVtoV_imap, 
    const int *d_VcToV_imap, const T *x, T *y, const T a) {
    int N_V = V_nnodes * block_dim;
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= N_V) return;

    int V_node = tid / block_dim;
    int Vc_node = d_VcToV_imap[V_node];
    int IEV_node = d_IEVtoV_imap[V_node];
    int idof = tid % block_dim;
    int dof_Vc = block_dim * Vc_node + idof;
    int dof_IEV = block_dim * IEV_node + idof;

    atomicAdd(&y[dof_Vc], a * x[dof_IEV]);
}

template <typename T>
__global__ static void k_addVec_VctoIEV(const int V_nnodes, const int block_dim, const int *d_IEVtoV_imap, 
    const int *d_VcToV_imap, const T *x, T *y, const T a) {
    int N_V = V_nnodes * block_dim;
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= N_V) return;

    int V_node = tid / block_dim;
    int Vc_node = d_VcToV_imap[V_node];
    int IEV_node = d_IEVtoV_imap[V_node];
    int idof = tid % block_dim;
    int dof_Vc = block_dim * Vc_node + idof;
    int dof_IEV = block_dim * IEV_node + idof;

    atomicAdd(&y[dof_IEV], a * x[dof_Vc]);
}

template <typename T>
__global__ static void k_zeroInterior(const int nnodes, const int block_dim, const bool *d_interior, 
    T *x) {
    int N = nnodes * block_dim;
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= N) return;

    int node = tid / block_dim;
    bool is_interior = d_interior[node];
    T val = x[tid];
    T new_val = is_interior ? 0.0 : val;
    x[tid] = new_val;
}

template <typename T>
__global__ static void k_addVec_IEtoGlobal(const int IE_nnodes, const int block_dim, const int *IE_globalMap, 
    const int *d_IE_nsd,  const T *x, T *y, T a) {
    int N_IE = IE_nnodes * block_dim;
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= N_IE) return;

    int IE_node = tid / block_dim;
    int glob_node = IE_globalMap[IE_node];
    // bool is_edge = d_general_edge[IE_node];
    int idof = tid % block_dim;
    int dof_IE = block_dim * IE_node + idof;
    int dof_glob = block_dim * glob_node + idof;
    // T alpha = is_edge ? (0.5) : 1.0; // half-weight for edge-nodes
    T alpha = 1.0 / d_IE_nsd[IE_node];
    // this edge map includes edge + dirichlet edge (which also add 0.5 weight into global, a bit tricky)

    atomicAdd(&y[dof_glob], a * alpha * x[dof_IE]);
}

template <typename T, bool scaled = false>
__global__ static void k_addVec_GlobalToIE(const int IE_nnodes, const int block_dim, const int *IE_globalMap, 
    const int *d_IE_nsd,  const T *x, T *y, T a) {
    int N_IE = IE_nnodes * block_dim;
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= N_IE) return;

    int IE_node = tid / block_dim;
    int glob_node = IE_globalMap[IE_node];
    // bool is_edge = d_general_edge[IE_node];
    int idof = tid % block_dim;
    int dof_IE = block_dim * IE_node + idof;
    int dof_glob = block_dim * glob_node + idof;
    // don't want half-weight for scattering global solution to edge nodes (it's more of a copy)
    T alpha;
    if constexpr (scaled) {
        alpha = 1.0 / d_IE_nsd[IE_node];
        // alpha = is_edge ? (0.5) : 1.0; // half-weight for edge-nodes
    } else {
        alpha = 1.0;
    }
    // this edge map includes edge + dirichlet edge (which also add 0.5 weight into global, a bit tricky)

    atomicAdd(&y[dof_IE], a * alpha * x[dof_glob]);
}

template <typename T>
__global__ static void k_subdomain_normalize_vec(const int nnodes, const int block_dim, const int *d_num_subdomain, const T *vec_in, T *vec_out) {
    int N = nnodes * block_dim;
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= N) return;

    int inode = tid / block_dim;
    // change to float type here for next division
    T nsd = 1.0 * d_num_subdomain[inode];
    vec_out[tid] = vec_in[tid] / nsd;
}

template <typename T>
__global__ static void k_subdomain_normalize_vec_inout(const int nnodes, const int block_dim, const int *d_num_subdomain, T *vec) {
    int N = nnodes * block_dim;
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= N) return;

    int inode = tid / block_dim;
    // change to float type here for next division
    T nsd = 1.0 * d_num_subdomain[inode];
    vec[tid] /= nsd;
}

template <typename T>
__global__ static void k_addVec_VctoGlobal(const int Vc_nnodes, const int block_dim, const int *Vc_globalMap, 
    const T *x, T *y, T a) {
    int N_Vc = Vc_nnodes * block_dim;
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= N_Vc) return;

    int Vc_node = tid / block_dim;
    int glob_node = Vc_globalMap[Vc_node];
    int idof = tid % block_dim;
    int dof_Vc = block_dim * Vc_node + idof;
    int dof_glob = block_dim * glob_node + idof;

    // no rescale cause Vc coarse node DOF are not repeated
    atomicAdd(&y[dof_glob], a * x[dof_Vc]);
}

template <typename T, bool scaled = false>
__global__ static void k_addVec_GlobaltoVc(const int Vc_nnodes, const int block_dim, const int *Vc_globalMap, const int *d_vertex_nsd,
    const T *x, T *y, T a) {
    int N_Vc = Vc_nnodes * block_dim;
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= N_Vc) return;

    int Vc_node = tid / block_dim;
    int glob_node = Vc_globalMap[Vc_node];
    int idof = tid % block_dim;
    int dof_Vc = block_dim * Vc_node + idof;
    int dof_glob = block_dim * glob_node + idof;
    T weight = 1.0;
    T nsd = d_vertex_nsd[Vc_node] * 1.0; // convert from int to float
    if constexpr (scaled) {
        // weight *= 0.25;
        weight /= nsd;
    }
    a *= weight;

    // no rescale cause Vc coarse node DOF are not repeated
    atomicAdd(&y[dof_Vc], a * x[dof_glob]);
}

template <typename T, bool add = false>
__global__ static void k_copyMatToMat_restrict(const int nnzb, const int block_dim, 
    const int *in_map, const int *out_map, const T *d_in_vals, T *d_out_vals) {
    // restrict values from in to out matrix (out matrix is smaller size and thus uses restrBlockMap)
    int block_dim2 = block_dim * block_dim;
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int nvals = block_dim2 * nnzb;
    if (tid >= nvals) return;

    int node_block = tid / block_dim2;
    int inner_dof = tid % block_dim2;
    int in_block = in_map[node_block];
    int out_block = out_map[node_block];
    int inn_ind = block_dim2 * in_block + inner_dof;
    int out_ind = block_dim2 * out_block + inner_dof;
    // simple copy, no add..
    if constexpr (add) {
        atomicAdd(&d_out_vals[out_ind], d_in_vals[inn_ind]);
    } else {
        d_out_vals[out_ind] = d_in_vals[inn_ind];
    }
}

template <typename T>
__global__ static void k_addVecIEtoGam(const int IE_nnodes, const int block_dim, 
    const int *IE_to_lam_map, const T *vec_IE, T *vec_lam, T a) {
    int N = IE_nnodes * block_dim;
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= N) return;

    int IE_node = tid / block_dim;
    int lam_node = IE_to_lam_map[IE_node];
    if (lam_node < 0) return;
    int idof = tid % block_dim;
    int IE_dof = block_dim * IE_node + idof;
    int lam_dof = block_dim * lam_node + idof;
    T s = 1.0; // always positive sign
    T as = a * s;

    atomicAdd(&vec_lam[lam_dof], as * vec_IE[IE_dof]);   
}

template <typename T>
__global__ static void k_addVecIEtoLam(const int IE_nnodes, const int block_dim, 
    const int *IE_to_lam_map, const T *IE_to_lam_vec, const T *vec_IE, T *vec_lam, T a) {
    int N = IE_nnodes * block_dim;
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= N) return;

    int IE_node = tid / block_dim;
    int lam_node = IE_to_lam_map[IE_node];
    if (lam_node < 0) return;
    int idof = tid % block_dim;
    int IE_dof = block_dim * IE_node + idof;
    int lam_dof = block_dim * lam_node + idof;
    T s = IE_to_lam_vec[IE_node];
    T as = a * s;

    atomicAdd(&vec_lam[lam_dof], as * vec_IE[IE_dof]);   
}

template <typename T>
__global__ static void k_addVecLamtoIE(const int IE_nnodes, const int block_dim, 
    const int *IE_to_lam_map, const T *IE_to_lam_vec, const T *vec_lam, T *vec_IE, T a) {
    int N = IE_nnodes * block_dim;
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= N) return;

    int IE_node = tid / block_dim;
    int lam_node = IE_to_lam_map[IE_node];
    if (lam_node < 0) return;
    int idof = tid % block_dim;
    int IE_dof = block_dim * IE_node + idof;
    int lam_dof = block_dim * lam_node + idof;
    T s = IE_to_lam_vec[IE_node];
    T as = a * s;

    atomicAdd(&vec_IE[IE_dof], as * vec_lam[lam_dof]);   
}


template <typename T>
__global__ static void k_addVecGamtoIE(const int IE_nnodes, const int block_dim, 
    const int *IE_to_lam_map, const T *vec_lam, T *vec_IE, T a) {
    int N = IE_nnodes * block_dim;
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= N) return;

    int IE_node = tid / block_dim;
    int lam_node = IE_to_lam_map[IE_node];
    if (lam_node < 0) return;
    int idof = tid % block_dim;
    int IE_dof = block_dim * IE_node + idof;
    int lam_dof = block_dim * lam_node + idof;
    T s = 1.0;
    T as = a * s;

    atomicAdd(&vec_IE[IE_dof], as * vec_lam[lam_dof]);   
}

template <typename T>
__global__ static void k_setVec_IEVtoV_vals(const int set_nnodes, const int block_dim,
                                            const int irow, const int *d_blocks,
                                            T *vec_IEV, const T val) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= set_nnodes) return;

    int idof = irow % block_dim;
    int iev_node = d_blocks[tid];

    vec_IEV[block_dim * iev_node + idof] = val;
}

template <typename T>
__global__ static void k_addMat_IEVtoV_vals(const int set_nnzb, const int block_dim,
                                            const int icol, const int *d_vecBlocks,
                                            const int *d_matBlocks, const T *hvec,
                                            T *mat_vals) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int nvals = set_nnzb * block_dim;
    if (tid >= nvals) return;

    const int block_dim2 = block_dim * block_dim;

    int iblock = tid / block_dim;
    int inn_row = tid % block_dim;
    int inn_col = icol % block_dim;

    int mat_block = d_matBlocks[iblock];
    int vec_block = d_vecBlocks[iblock];

    // atomicAdd(&mat_vals[block_dim2 * mat_block + block_dim * inn_row + inn_col],
    //           hvec[block_dim * vec_block + inn_row]);
    atomicAdd(&mat_vals[block_dim2 * mat_block + block_dim * inn_col + inn_row],
              hvec[block_dim * vec_block + inn_row]); // this was flipped before somehow (the mat row,col)
}