

template <typename T>
__global__ void k_copyRHSIntoBatched(const int n_batch_vals, const int block_dim, const int size,
    const int *d_blockMap, const T *rhs, T **array_rhs) {
    // copies original rhs vector into batched storage for each Schwarz coupling group
    
    // compute the appropriate indices in Aarray in order
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= n_batch_vals) return;
    int block_dim2 = block_dim * block_dim;
    int batch_block_ind = tid / block_dim;
    int size2 = size * size;
    int batch_ind = batch_block_ind / size2;
    int inner_block_ind = batch_block_ind % size2; 

    // get the block ind in the kmat
    int rhs_block_ind = d_blockMap[batch_block_ind];

    // on each thread perform the copy operation
    int inner_ind = tid % block_dim;
    T val = rhs[block_dim * rhs_block_ind + inner_ind];
    T *batch_ptr = array_rhs[batch_ind];
    batch_ptr[block_dim * inner_block_ind + inner_ind] = val;
}

template <typename T>
__global__ void k_copyBatchedIntoSoln(const int n_batch_vals, const int block_dim, const int size,
    const int *d_blockMap, T **array_soln, T *soln) {
    // copies original rhs vector into batched storage for each Schwarz coupling group
    
    // compute the appropriate indices in Aarray in order
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= n_batch_vals) return;
    int block_dim2 = block_dim * block_dim;
    int batch_block_ind = tid / block_dim;
    int size2 = size * size;
    int batch_ind = batch_block_ind / size2;
    int inner_block_ind = batch_block_ind % size2; 

    // get the block ind in the kmat
    int rhs_block_ind = d_blockMap[batch_block_ind];

    // on each thread perform the copy operation
    int inner_ind = tid % block_dim;
    T *batch_ptr = array_soln[batch_ind];
    T val = batch_ptr[block_dim * inner_block_ind + inner_ind];
    soln[block_dim * rhs_block_ind + inner_ind] = val;
}