#pragma once
#include "cuda_utils.h"


template <typename T>
__global__ void k_copyMatValuesToBatched(
    const int n_batch_vals,           // = n_batch_blocks * block_dim^2
    const int block_dim,
    const int size,                   // patch side length in nodes (size2=size*size)
    const int* __restrict__ d_blockMap, // length n_batch_blocks; jp indices into vals blocks
    const T* __restrict__ vals,         // BSR block values, row-major within each block
    T** __restrict__ array_vals         // length batchSize; each points to dense n*n
) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= n_batch_vals) return;

    const int block_dim2 = block_dim * block_dim;
    const int size2 = size * size;
    const int size4 = size2 * size2;
    const int n = size2 * block_dim;                 // e.g. 24

    // Which (patch-block, entry-within-block)?
    const int batch_block_ind = tid / block_dim2;    // 0 .. n_batch_blocks-1
    const int inner_ind       = tid % block_dim2;    // 0 .. block_dim2-1

    const int batch_ind       = batch_block_ind / size4; // 0..batchSize-1
    const int inner_batch_ind = batch_block_ind % size4; // 0..size4-1

    // Which (i,j) node-block inside patch?
    // You built inner_batch_ind = size2 * j + i
    const int i = inner_batch_ind % size2;           // row-node index in patch
    const int j = inner_batch_ind / size2;           // col-node index in patch

    // Row-major inside each BSR block:
    const int p = inner_ind / block_dim;             // row dof inside block
    const int q = inner_ind % block_dim;             // col dof inside block

    const int row = i * block_dim + p;
    const int col = j * block_dim + q;

    const int kmat_block_ind = d_blockMap[batch_block_ind];
    // if (kmat_block_ind < 0) return;                  // optional if you ever store -1

    T* A = array_vals[batch_ind];

    // if (batch_ind == 0 && inner_ind == 0) {
    //     printf("batch ind 0 : local nodes (%d,%d) to kmat block %d\n", i, j, kmat_block_ind);
    // }

    // Dense local matrix stored column-major for cuBLAS
    A[row + col * n] = vals[kmat_block_ind * block_dim2 + inner_ind];
}

template <typename T>
__global__ void k_copyRHSIntoBatched(
    const int n_batch_vals,           // = (batchSize * size2) * block_dim
    const int block_dim,
    const int size,
    const int* __restrict__ d_blockMap, // length batchSize*size2, maps local node -> global node
    const T* __restrict__ rhs,          // length N (global dofs)
    T** __restrict__ array_rhs          // length batchSize, each points to local rhs (n = size2*block_dim)
) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= n_batch_vals) return;

    const int size2 = size * size;

    const int batch_block_ind = tid / block_dim;      // 0 .. batchSize*size2 - 1
    const int inner_ind       = tid % block_dim;      // dof within node (0..block_dim-1)

    const int batch_ind       = batch_block_ind / size2;
    const int inner_node_ind  = batch_block_ind % size2;

    // Optional safety
    // if (batch_ind >= batchSize) return;

    const int global_node = d_blockMap[batch_block_ind];

    T* b = array_rhs[batch_ind];
    b[inner_node_ind * block_dim + inner_ind] = rhs[global_node * block_dim + inner_ind];
}

template <typename T>
__global__ void k_copyBatchedIntoSoln_additive(
    const int n_batch_vals,            // = (batchSize * size2) * block_dim
    const int block_dim,
    const int size,
    const int* __restrict__ d_blockMap, // length batchSize*size2, maps local node -> global node
    T** __restrict__ array_soln,        // length batchSize, local solution vectors (n=size2*block_dim)
    T* __restrict__ soln               // global vector to accumulate into
) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= n_batch_vals) return;

    const int size2 = size * size;

    const int batch_block_ind = tid / block_dim;
    const int inner_ind       = tid % block_dim;

    const int batch_ind       = batch_block_ind / size2;
    const int inner_node_ind  = batch_block_ind % size2;

    const int global_node = d_blockMap[batch_block_ind];

    const T* xloc = array_soln[batch_ind];
    const T val = xloc[inner_node_ind * block_dim + inner_ind];

    // Accumulate because overlap causes collisions
    atomicAdd(&soln[global_node * block_dim + inner_ind], val);
}


// =====================================
// Support ASW version
// =====================================

template <typename T>
__global__ void k_setSubdomainMatricesToIdentity(
    const int n_subdomain_vals,           // = n_batch * size4 * block_dim^2
    const int block_dim,
    const int size,                   // patch side length in nodes (size2=size*size)
    T** __restrict__ array_vals         // length batchSize; each points to dense n*n
) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= n_subdomain_vals) return;

    const int block_dim2 = block_dim * block_dim;
    const int size2 = size * size;
    const int size4 = size2 * size2;
    const int n = size2 * block_dim;                 // e.g. 24

    // get the subdomain and kmat blocks
    int n_vals_in_subdomain = size4 * block_dim2;
    int n_rows_in_subdomain = size2 * block_dim;
    const int subdomain_ind = tid / n_vals_in_subdomain;
    const int inner_sd_ind = tid % n_vals_in_subdomain;

    // Dense local matrix stored column-major for cuBLAS
    T* A = array_vals[subdomain_ind];
    const int row = inner_sd_ind % n_rows_in_subdomain;
    const int col = inner_sd_ind / n_rows_in_subdomain;
    A[inner_sd_ind] = (row == col) ? 1.0 : 0.0;
}

template <typename T>
__global__ void k_copyMatValuesToBatched_support(
    const int n_batch_vals,           // = nnz_batch_blocks * block_dim^2
    const int block_dim,
    const int size,                   // patch side length in nodes (size2=size*size)
    const int* __restrict__ d_kmatBlockInds, // length nnz_batch_blocks; jp indices into vals (kmat) blocks
    const int* __restrict__ d_sdBlockInds, // length nnz_batch_blocks; jp indices into array_vals blocks
    const T* __restrict__ vals,         // BSR block values, row-major within each block
    T** __restrict__ array_vals         // length batchSize; each points to dense n*n
) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= n_batch_vals) return;

    const int block_dim2 = block_dim * block_dim;
    const int size2 = size * size;
    const int size4 = size2 * size2;
    const int n = size2 * block_dim;                 // e.g. 24

    // Which (patch-block, entry-within-block)?
    const int batch_block_ind = tid / block_dim2;    // 0 .. nnz_batch_blocks-1
    const int inner_ind       = tid % block_dim2;    // 0 .. block_dim2-1

    // get the subdomain and kmat blocks
    const int subdomain_block_ind = d_sdBlockInds[batch_block_ind];
    const int subdomain_ind = subdomain_block_ind / size4;
    const int kmat_block_ind = d_kmatBlockInds[batch_block_ind];

    // get dense storage spots in the subdomain
    const int inner_block = subdomain_block_ind % size4;  // 0..size4-1 within subdomain
    const int i_node = inner_block % size2;               // row-node slot
    const int j_node = inner_block / size2;               // col-node slot
    const int p = inner_ind / block_dim;             // row dof inside block
    const int q = inner_ind % block_dim;             // col dof inside block
    const int row = i_node * block_dim + p;
    const int col = j_node * block_dim + q;

    // Dense local matrix stored column-major for cuBLAS
    T* A = array_vals[subdomain_ind];
    A[row + col * n] = vals[kmat_block_ind * block_dim2 + inner_ind];
}


template <typename T>
__global__ void k_zeroSubdomainVecs_support(
    const int n_rhs_vals,           // = n_batch * size2 * block_dim
    const int block_dim,
    const int size,                   // patch side length in nodes (size2=size*size)
    T** __restrict__ vec_sd_vals         // length batchSize; each points to dense n*n
) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= n_rhs_vals) return;

    const int size2 = size * size;
    const int n = size2 * block_dim;                 // e.g. 24

    // get the subdomain and kmat blocks
    const int subdomain_ind = tid / n;
    const int inner_sd_ind = tid % n;

    // Dense local matrix stored column-major for cuBLAS
    T* sd_vals = vec_sd_vals[subdomain_ind];
    sd_vals[inner_sd_ind] = 0.0;
}

template <typename T>
__global__ void k_copyRHSIntoBatched_support(
    const int n_batch_vals,           // = (batchSize * size2) * block_dim
    const int block_dim,
    const int size,
    const int* __restrict__ d_rhsDenseMap,
    const int* __restrict__ d_rhsSDMap, 
    const T* __restrict__ rhs,          // length N (global dofs)
    T** __restrict__ array_rhs          // length batchSize, each points to local rhs (n = size2*block_dim)
) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= n_batch_vals) return;

    const int size2 = size * size;

    const int nz_block_ind = tid / block_dim;      // 0 .. batchSize*size2 - 1
    const int inner_ind       = tid % block_dim;      // dof within node (0..block_dim-1)

    const int subdomain_block_ind = d_rhsSDMap[nz_block_ind];
    const int subdomain_ind = subdomain_block_ind / size2;
    const int inner_sd_node_ind = subdomain_block_ind % size2;
    const int global_node = d_rhsDenseMap[nz_block_ind];

    T* b = array_rhs[subdomain_ind];
    b[inner_sd_node_ind * block_dim + inner_ind] = rhs[global_node * block_dim + inner_ind];
}

template <typename T>
__global__ void k_copyBatchedIntoSoln_additiveSupport(
    const int n_batch_vals,            // = (batchSize * size2) * block_dim
    const int block_dim,
    const int size,
    const int* __restrict__ d_rhsDenseMap,
    const int* __restrict__ d_rhsSDMap, 
    T** __restrict__ array_soln,        // length batchSize, local solution vectors (n=size2*block_dim)
    T* __restrict__ soln               // global vector to accumulate into
) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= n_batch_vals) return;

    const int size2 = size * size;

    const int nz_block_ind = tid / block_dim;      // 0 .. batchSize*size2 - 1
    const int inner_ind       = tid % block_dim;      // dof within node (0..block_dim-1)

    const int subdomain_block_ind = d_rhsSDMap[nz_block_ind];
    const int subdomain_ind = subdomain_block_ind / size2;
    const int inner_sd_node_ind = subdomain_block_ind % size2;
    const int global_node = d_rhsDenseMap[nz_block_ind];

    const T* xloc = array_soln[subdomain_ind];
    const T val = xloc[inner_sd_node_ind * block_dim + inner_ind];

    // Accumulate because overlap causes collisions
    atomicAdd(&soln[global_node * block_dim + inner_ind], val);
}


template <typename T>
__global__ void k_zeroBatchedSolnOnElemBoundaries(
    const int nxe,
    const int n_batch_vals,            // = (batchSize * size2) * block_dim
    const int block_dim,
    const int size,
    const int* __restrict__ d_rhsDenseMap,
    const int* __restrict__ d_rhsSDMap, 
    T** __restrict__ array_soln        // length batchSize, local solution vectors (n=size2*block_dim)
) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= n_batch_vals) return;
    int nx = nxe + 1;
    int ny = nxe; // for temp cylinder case (this is HACK routine for my thesis here)
    // in general, you should not use Elem3x3, use default support 3x3 ASW smoother

    const int size2 = size * size;

    const int nz_block_ind = tid / block_dim;      // 0 .. batchSize*size2 - 1
    const int inner_ind       = tid % block_dim;      // dof within node (0..block_dim-1)

    const int subdomain_block_ind = d_rhsSDMap[nz_block_ind];
    const int subdomain_ind = subdomain_block_ind / size2;
    const int inner_sd_node_ind = subdomain_block_ind % size2;
    const int global_node = d_rhsDenseMap[nz_block_ind];

    // subdomain ind is equivalent to node index (for support-based)
    int ix = subdomain_ind % nx, iy = subdomain_ind / nx;

    T* xloc = array_soln[subdomain_ind];
    T val = xloc[inner_sd_node_ind * block_dim + inner_ind];

    // change value to zero if on boundary.. (zeros this part of additive solution)
    // bool on_bndry = ix == 0 || ix == nx - 1 || iy == 0 || iy == ny-1;
    bool on_bndry = ix == 0 || ix == nx - 1;
    T bndry_val = on_bndry ? 0.0 : val;

    // change teh value (only on bndry, but resets all values)
    xloc[inner_sd_node_ind * block_dim + inner_ind] = bndry_val;
}