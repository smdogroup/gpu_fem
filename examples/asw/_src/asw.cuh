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
