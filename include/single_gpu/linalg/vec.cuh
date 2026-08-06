#pragma once
#include "cuda_utils.h"

template <typename T, template <typename> class Vec>
__GLOBAL__ void apply_vec_bcs_kernel(Vec<int> bcs, T *data) {
    int nbcs = bcs.getSize();
    // no need for perm here since this is called on unreordered state of vectors like res, loads, etc.
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    for (int ibc = start; ibc < nbcs; ibc += blockDim.x) {
        int idof = bcs[ibc];
        data[idof] = 0.0;
    }
}

template <typename T, template <typename> class Vec>
__GLOBAL__ void permute_vec_kernel(int num_nodes, T *data, T *temp, int block_dim, int *perm) {
    // TODO : could also have each block in the grid do more work with for loop
    // for larger problems?

    // copying data from data to temp using permutation
    int orig_inode = blockIdx.x * blockDim.x + threadIdx.x;
    if (orig_inode < num_nodes) {
        for (int inner_dof = 0; inner_dof < block_dim; inner_dof++) {
            int orig_idof = block_dim * orig_inode + inner_dof;
            int perm_idof = Vec<T>::permuteDof(orig_idof, perm, block_dim);
            T val = data[orig_idof];
            temp[perm_idof] = val;
        }
    }
}

// template <typename T>
// __GLOBAL__ void vec_axpy_kernel(int N, T a, T *x, T *y) {
//     int ind = blockDim.x * blockIdx.x + threadIdx.x;
//     if (ind < N) {
//         y[ind] += a * x[ind];
//     }
// }

template <typename T, template <typename> class Vec>
__GLOBAL__ void vec_add_kernel(Vec<T> v1, Vec<T> v2, Vec<T> v3) {
    int thread_start = blockDim.x * blockIdx.x + threadIdx.x;
    int N = v1.getSize();

    int i = thread_start;
    if (i < N) {
        atomicAdd(&v3[i], v1[i] + v2[i]);
        // printf("adding ind %d\n", i);
    }
}

template <typename T>
__GLOBAL__ void vec_add_value_kernel(int N, T *data, int ind, T value) {
    if (ind < N) {
        atomicAdd(&data[ind], value);
    }
}

template <typename T, template <typename> class Vec>
__GLOBAL__ void removeRotationalDOF_kernel(int N1, int N2, T *v1, T *v2) {
    int thread_start = blockDim.x * blockIdx.x + threadIdx.x;

    int i2 = thread_start;
    int group_num = i2 / 3;
    int i1 = 6 * group_num + i2 % 3;

    if (i1 < N1 && i2 < N2) {
        v2[i2] = v1[i1];
    }
}

template <typename T, template <typename> class Vec>
__GLOBAL__ void addRotationalDOF_kernel(int N1, int N2, T *v1, T *v2) {
    int thread_start = blockDim.x * blockIdx.x + threadIdx.x;

    int i1 = thread_start;
    int group_num = i1 / 3;
    int i2 = 6 * group_num + i1 % 3;

    if (i1 < N1 && i2 < N2) {
        v2[i2] = v1[i1];
    }
}

template <typename T>
__global__ void k_set_full_vec_const_value(const int N, const T val, T *vec) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        vec[i] = val;
    }
}