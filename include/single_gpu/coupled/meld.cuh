#pragma once
#include "linalg/svd_utils.h"
#include "a2dcore.h"

template <typename T, int NN>
__GLOBAL__ void compute_avgdistsq_kernel(int nn, DeviceVec<int> aerostruct_conn, DeviceVec<T> xs0,
                                       DeviceVec<T> xa0, double beta, DeviceVec<T> avgdistsq) {
    // each block of threads if just computing ua for one aero node
    // among all the nearest neighbor struct nodes
    int aero_ind = blockIdx.x;
    int na = xa0.getSize() / 3;
    int ns = xs0.getSize() / 3;

    __SHARED__ int loc_conn[NN];
    __SHARED__ T loc_xs0[3 * NN];
    __SHARED__ T loc_xa0[3];
    __SHARED__ T sum_distsq[1];

    // copy data from global to shared
    int glob_start = aero_ind * nn + NN * blockIdx.y;
    bool active_thread = (glob_start + threadIdx.x) < na * nn;
    aerostruct_conn.copyValuesToShared(active_thread, threadIdx.x, NN, blockDim.x, glob_start,
                                       &loc_conn[0]);
    xa0.copyValuesToShared(true, threadIdx.x, 3, blockDim.x, 3 * aero_ind, &loc_xa0[0]);
    // need to use conn here so may need copyElemValuesToShared
    xs0.copyElemValuesToShared(true, threadIdx.x, blockDim.x, 3, NN, &loc_conn[0], &loc_xs0[0]);
    __syncthreads();

    // compute avg dist_sq for normalization
    memset(sum_distsq, 1e-10, 1 * sizeof(T));
    for (int inode = threadIdx.x; inode < NN; inode += blockDim.x) {
        // first compute the distance squared
        T distsq = 0.0;
        for (int idim = 0; idim < 3; idim++) {
            T delta = loc_xa0[idim] - loc_xs0[3 * inode + idim];
            distsq += delta * delta;
        }

        atomicAdd(&sum_distsq[0], distsq / (1.0 * nn));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(&avgdistsq[aero_ind], sum_distsq[0]);
    }
    
}

template <typename T, int NN>
__GLOBAL__ void compute_sum_weights_kernel(int nn, DeviceVec<int> aerostruct_conn, DeviceVec<T> xs0,
                                       DeviceVec<T> xa0, double beta, DeviceVec<T> glob_avgdistsq, DeviceVec<T> sum_weights) {
    // each block of threads if just computing ua for one aero node
    // among all the nearest neighbor struct nodes
    int aero_ind = blockIdx.x;
    int na = xa0.getSize() / 3;
    int ns = xs0.getSize() / 3;

    __SHARED__ int loc_conn[NN];
    __SHARED__ T loc_xs0[3 * NN];
    __SHARED__ T loc_xa0[3];
    __SHARED__ T loc_avgdistsq[1];
    __SHARED__ T sum[1];

    // copy data from global to shared
    int glob_start = aero_ind * nn + NN * blockIdx.y;
    bool active_thread = (glob_start + threadIdx.x) < na * nn;
    aerostruct_conn.copyValuesToShared(active_thread, threadIdx.x, NN, blockDim.x, glob_start,
                                       &loc_conn[0]);
    xa0.copyValuesToShared(true, threadIdx.x, 3, blockDim.x, 3 * aero_ind, &loc_xa0[0]);
    // need to use conn here so may need copyElemValuesToShared
    xs0.copyElemValuesToShared(true, threadIdx.x, blockDim.x, 3, NN, &loc_conn[0], &loc_xs0[0]);
    if (threadIdx.x == 0) {
        loc_avgdistsq[0] = glob_avgdistsq[aero_ind];
    }
    __syncthreads();

    // on thread data
    T avgdistsq = loc_avgdistsq[0];

    // compute weights (un-normalized first)
    memset(sum, 0.0, 1 * sizeof(T));
    for (int inode = threadIdx.x; inode < NN; inode += blockDim.x) {
        // first compute the distance squared
        T distsq = 0.0;
        for (int idim = 0; idim < 3; idim++) {
            T delta = loc_xa0[idim] - loc_xs0[3 * inode + idim];
            distsq += delta * delta;
        }

        T new_weight = exp(-beta * distsq / avgdistsq);
        atomicAdd(&sum[0], new_weight);
    }
    __syncthreads();
    // add to global
    if (threadIdx.x == 0) {
        atomicAdd(&sum_weights[aero_ind], sum[0]);
    } 
}

template <typename T, int NN>
__GLOBAL__ void compute_weights_kernel(int nn, DeviceVec<int> aerostruct_conn, DeviceVec<T> xs0,
                                       DeviceVec<T> xa0, double beta, DeviceVec<T> glob_avgdistsq, 
                                       DeviceVec<T> glob_sum_weights, DeviceVec<T> weights) {
    // each block of threads if just computing ua for one aero node
    // among all the nearest neighbor struct nodes
    int aero_ind = blockIdx.x;
    int na = xa0.getSize() / 3;
    int ns = xs0.getSize() / 3;

    __SHARED__ int loc_conn[NN];
    __SHARED__ T loc_w[NN];
    __SHARED__ T loc_xs0[3 * NN];
    __SHARED__ T loc_xa0[3];
    __SHARED__ T loc_avgdistsq[1];
    __SHARED__ T loc_sum[1];

    // copy data from global to shared
    int glob_start = aero_ind * nn + NN * blockIdx.y;
    bool active_thread = (glob_start + threadIdx.x) < na * nn;
    aerostruct_conn.copyValuesToShared(active_thread, threadIdx.x, NN, blockDim.x, glob_start,
                                       &loc_conn[0]);
    weights.copyValuesToShared(active_thread, threadIdx.x, nn, blockDim.x, glob_start,
                               &loc_w[0]);
    xa0.copyValuesToShared(true, threadIdx.x, 3, blockDim.x, 3 * aero_ind, &loc_xa0[0]);
    // need to use conn here so may need copyElemValuesToShared
    xs0.copyElemValuesToShared(true, threadIdx.x, blockDim.x, 3, NN, &loc_conn[0], &loc_xs0[0]);
    if (threadIdx.x == 0) {
        loc_avgdistsq[0] = glob_avgdistsq[aero_ind];
        loc_sum[0] = glob_sum_weights[aero_ind];
    }
    __syncthreads();

    // on thread data
    T avgdistsq = loc_avgdistsq[0];
    T sum = loc_sum[0];

    // compute weights (un-normalized first)
    memset(loc_w, 0.0, NN * sizeof(T));
    for (int inode = threadIdx.x; inode < NN; inode += blockDim.x) {
        // first compute the distance squared
        T distsq = 0.0;
        for (int idim = 0; idim < 3; idim++) {
            T delta = loc_xa0[idim] - loc_xs0[3 * inode + idim];
            distsq += delta * delta;
        }

        T new_weight = exp(-beta * distsq / avgdistsq);
        loc_w[inode] += new_weight / sum; // add in normalized weights
    }
    __syncthreads();

    // debug local weight sum
    T loc_wsum = 0.0;
    for (int inode = 0; inode < NN; inode++) {
        loc_wsum += loc_w[inode]; 
    }

    for (int inode = threadIdx.x; inode < NN; inode += blockDim.x) {
        atomicAdd(&weights[glob_start+inode], loc_w[inode]);
    }
}

template <typename T, int NN>
__GLOBAL__ void compute_weights_oneshot_kernel(int nn, DeviceVec<int> aerostruct_conn, DeviceVec<T> xs0,
                                       DeviceVec<T> xa0, double beta, DeviceVec<T> weights) {
    // each block of threads if just computing ua for one aero node
    // among all the nearest neighbor struct nodes
    int aero_ind = blockIdx.x;
    int na = xa0.getSize() / 3;
    int ns = xs0.getSize() / 3;

    __SHARED__ int loc_conn[NN];
    __SHARED__ T loc_w[NN];
    __SHARED__ T loc_xs0[3 * NN];
    __SHARED__ T loc_xa0[3];
    __SHARED__ T sum_distsq[1];
    __SHARED__ T sum[1];

    // copy data from global to shared
    int glob_start = aero_ind * nn + NN * blockIdx.y;
    bool active_thread = (glob_start + threadIdx.x) < na * nn;
    aerostruct_conn.copyValuesToShared(active_thread, threadIdx.x, NN, blockDim.x, glob_start,
                                       &loc_conn[0]);
    weights.copyValuesToShared(active_thread, threadIdx.x, nn, blockDim.x, glob_start,
                               &loc_w[0]);
    xa0.copyValuesToShared(true, threadIdx.x, 3, blockDim.x, 3 * aero_ind, &loc_xa0[0]);
    // need to use conn here so may need copyElemValuesToShared
    xs0.copyElemValuesToShared(true, threadIdx.x, blockDim.x, 3, NN, &loc_conn[0], &loc_xs0[0]);
    __syncthreads();

    // compute avg dist_sq for normalization
    memset(sum_distsq, 0.0, 1 * sizeof(T));
    for (int inode = threadIdx.x; inode < NN; inode += blockDim.x) {
        // first compute the distance squared
        T distsq = 0.0;
        for (int idim = 0; idim < 3; idim++) {
            T delta = loc_xa0[idim] - loc_xs0[3 * inode + idim];
            distsq += delta * delta;
        }

        atomicAdd(&sum_distsq[0], distsq);
    }
    __syncthreads();
    T avg_distsq = sum_distsq[0] / (1.0 * nn);
    avg_distsq += 1e-7;

    // compute weights (un-normalized first)
    memset(loc_w, 0.0, nn * sizeof(T));
    memset(sum, 0.0, 1 * sizeof(T));
    for (int inode = threadIdx.x; inode < nn; inode += blockDim.x) {
        // first compute the distance squared
        T distsq = 0.0;
        for (int idim = 0; idim < 3; idim++) {
            T delta = loc_xa0[idim] - loc_xs0[3 * inode + idim];
            distsq += delta * delta;
        }

        T new_weight = exp(-beta * distsq / avg_distsq);
        loc_w[inode] += new_weight;
        atomicAdd(&sum[0], new_weight);
    }
    __syncthreads();

    // normalize the weights so it becomes a partition of unity
    for (int inode = threadIdx.x; inode < nn; inode += blockDim.x) {
        loc_w[inode] /= sum[0];
    }
    __syncthreads();

    for (int inode = threadIdx.x; inode < NN; inode += blockDim.x) {
        atomicAdd(&weights[glob_start+inode], loc_w[inode]);
    }
}

template <typename T, int NN>
__GLOBAL__ void compute_centroid_kernel(int nn, T H_reg, DeviceVec<int> aerostruct_conn, 
                                      DeviceVec<T> weights, DeviceVec<T> xs0, DeviceVec<T> xs,
                                      DeviceVec<T> glob_xs0_bar, DeviceVec<T> glob_xs_bar) {
    // each block of threads if just computing ua for one aero node
    // among all the nearest neighbor struct nodes
    int aero_ind = blockIdx.x;
    int na = aerostruct_conn.getSize() / nn;
    int ns = xs0.getSize() / 3;
    
    __SHARED__ int loc_conn[NN];
    __SHARED__ T loc_w[NN];
    __SHARED__ T loc_xs0[3 * NN];
    __SHARED__ T loc_xs[3 * NN];

    __SHARED__ T loc_xs0_bar[3];
    __SHARED__ T loc_xs_bar[3];

    // copy data from global to shared
    int glob_start = aero_ind * nn + blockDim.x * blockIdx.y;
    bool active_thread = (glob_start + threadIdx.x) < na * nn;
    aerostruct_conn.copyValuesToShared(active_thread, threadIdx.x, NN, blockDim.x, glob_start,
                                       &loc_conn[0]);
    weights.copyValuesToShared(active_thread, threadIdx.x, NN, blockDim.x, glob_start,
                               &loc_w[0]);
    // need to use conn here so may need copyElemValuesToShared
    xs0.copyElemValuesToShared(true, threadIdx.x, blockDim.x, 3, NN, &loc_conn[0], &loc_xs0[0]);
    xs.copyElemValuesToShared(true, threadIdx.x, blockDim.x, 3, NN, &loc_conn[0], &loc_xs[0]);
    __syncthreads();

    // all of the following will be reduction-like operations
    // until H is computed (each thread helps in the work)

    // compute the centroids of the nearest neighbors
    // xs0_bar
    memset(&loc_xs0_bar[0], 0.0, 3 * sizeof(T));
    for (int i = threadIdx.x; i < 3 * NN; i += blockDim.x) {
        int inode = i / 3;
        int idim = i % 3;
        atomicAdd(&loc_xs0_bar[idim], loc_w[inode] * loc_xs0[i]);
    }

    // xs_bar
    memset(&loc_xs_bar[0], 0.0, 3 * sizeof(T));
    for (int i = threadIdx.x; i < 3 * NN; i += blockDim.x) {
        int inode = i / 3;
        int idim = i % 3;
        atomicAdd(&loc_xs_bar[idim], loc_w[inode] * loc_xs[i]);
    }
    __syncthreads();

    // now add into global memory
    for (int i = threadIdx.x; i < 3; i += blockDim.x) {
        atomicAdd(&glob_xs0_bar[3 * aero_ind + i], loc_xs0_bar[i]);
        atomicAdd(&glob_xs_bar[3 * aero_ind + i], loc_xs_bar[i]);
    }
}

template <typename T, int NN>
__GLOBAL__ void compute_covariance_kernel(int nn, T H_reg, DeviceVec<int> aerostruct_conn, 
                                      DeviceVec<T> weights, DeviceVec<T> xs0, DeviceVec<T> xs,
                                      DeviceVec<T> glob_xs0_bar, DeviceVec<T> glob_xs_bar, DeviceVec<T> glob_H) {
    // each block of threads if just computing ua for one aero node
    // among all the nearest neighbor struct nodes
    int aero_ind = blockIdx.x;
    int na = aerostruct_conn.getSize() / nn;
    int ns = xs0.getSize() / 3;
    
    __SHARED__ int loc_conn[NN];
    __SHARED__ T loc_w[NN];
    __SHARED__ T loc_xs0[3 * NN];
    __SHARED__ T loc_xs[3 * NN];

    __SHARED__ T loc_xs0_bar[3];
    __SHARED__ T loc_xs_bar[3];
    __SHARED__ T loc_H[9];

    // copy data from global to shared
    int glob_start = aero_ind * nn + NN * blockIdx.y;
    bool active_thread = (glob_start + threadIdx.x) < na * nn;
    aerostruct_conn.copyValuesToShared(active_thread, threadIdx.x, NN, blockDim.x, glob_start,
                                       &loc_conn[0]);
    weights.copyValuesToShared(active_thread, threadIdx.x, NN, blockDim.x, glob_start,
                               &loc_w[0]);
    // need to use conn here so may need copyElemValuesToShared
    xs0.copyElemValuesToShared(true, threadIdx.x, blockDim.x, 3, NN, &loc_conn[0], &loc_xs0[0]);
    xs.copyElemValuesToShared(true, threadIdx.x, blockDim.x, 3, NN, &loc_conn[0], &loc_xs[0]);
    glob_xs0_bar.copyValuesToShared(true, threadIdx.x, 3, blockDim.x, 3 * aero_ind, &loc_xs0_bar[0]);
    glob_xs_bar.copyValuesToShared(true, threadIdx.x, 3, blockDim.x, 3 * aero_ind, &loc_xs_bar[0]);
    __syncthreads();

    // all of the following will be reduction-like operations
    // until H is computed (each thread helps in the work)

    // get xs0_bar, xs_bar on this thread from shared memory
    T xs0_bar[3], xs_bar[3];
    for (int i = 0; i < 3; i++) {
        xs0_bar[i] = loc_xs0_bar[i];
        xs_bar[i] = loc_xs_bar[i];
    }

    // compute covariance H (reduction step across threads)
    memset(&loc_H[0], 0.0, 9 * sizeof(T));
    if (threadIdx.x == 0 && blockIdx.y == 0) {
        // regularization of H for stability
        loc_H[0] += H_reg;
        loc_H[4] += H_reg;
        loc_H[8] += H_reg;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 9 * NN; i += blockDim.x) {
        int inode = i / 9;
        int i9 = i % 9;
        int idim = i9 / 3;
        int jdim = i9 % 3;

        // H += w_{inode} * p_{inode,idim} * q_{inode,jdim}
        // where p = xS - xSbar for each node and dim, sim q = xS0 - xS0bar
        // could do warp shuffling here
        T h_val = loc_w[inode] * (loc_xs[3 * inode + idim] - xs_bar[idim]) *
                  (loc_xs0[3 * inode + jdim] - xs0_bar[jdim]);
        atomicAdd(&loc_H[i9], h_val);
    }
    __syncthreads();

    // now add into global memory
    for (int i = threadIdx.x; i < 9; i += blockDim.x) {
        atomicAdd(&glob_H[9 * aero_ind + i], loc_H[i]);
    }
}

template <typename T, bool exact_givens = true>
__HOST_DEVICE__ void compute_aero_disp(T xa0[3], T xs0_bar[3], T xs_bar[3], T H[9], T ua[3]) {
    // get SVD of H (call device function in svd_utils.h)
    T R[9];
    bool print = false;
    computeRotation<T, exact_givens>(H, R, print);

    // compute disp offsets, r = xa0 - xs0_bar
    T r[3];
    A2D::VecSumCore<T, 3>(1.0, xa0, -1.0, xs0_bar, r);

    // perform rotation and translations
    // rho = R * r + t
    T rho[3];
    // note for some reason it looks like RT in F2F MELD code
    A2D::MatVecCore<T, 3, 3>(R, r, rho);

    T xa[3];
    A2D::VecSumCore<T, 3>(xs_bar, rho, xa);

    // compute aero disps
    A2D::VecSumCore<T, 3>(1.0, xa, -1.0, xa0, ua);
}


template <typename T, int NA_PER_BLOCK>
__GLOBAL__ void transfer_disps_kernel(DeviceVec<T> glob_xa0, DeviceVec<T> glob_xs0_bar, 
    DeviceVec<T> glob_xs_bar, 
    DeviceVec<T> glob_H, DeviceVec<T> glob_ua)  {

    
    // each block of threads if just computing ua for one aero node
    // among all the nearest neighbor struct nodes
    int start_aero_ind = blockIdx.x * blockDim.x; // starting aero ind of the block
    int aero_ind = start_aero_ind + threadIdx.x;
    int na = glob_xa0.getSize() / 3;
    
    __SHARED__ T loc_xa0[3 * NA_PER_BLOCK];
    __SHARED__ T loc_xs0_bar[3 * NA_PER_BLOCK];
    __SHARED__ T loc_xs_bar[3 * NA_PER_BLOCK];
    __SHARED__ T loc_H[9 * NA_PER_BLOCK];
    __SHARED__ T loc_ua[3 * NA_PER_BLOCK];

    // copy data from global to shared
    bool active_thread = aero_ind < na;
    glob_xa0.copyValuesToShared2(true, threadIdx.x, 3 * NA_PER_BLOCK, blockDim.x, 3 * start_aero_ind, 3 * na, &loc_xa0[0]);
    glob_xs0_bar.copyValuesToShared2(true, threadIdx.x, 3 * NA_PER_BLOCK, blockDim.x, 3 * start_aero_ind,  3 * na, &loc_xs0_bar[0]);
    glob_xs_bar.copyValuesToShared2(true, threadIdx.x, 3 * NA_PER_BLOCK, blockDim.x, 3 * start_aero_ind,  3 * na, &loc_xs_bar[0]);
    glob_H.copyValuesToShared2(true, threadIdx.x, 9 * NA_PER_BLOCK, blockDim.x, 9 * start_aero_ind,  9 * na, &loc_H[0]);
    __syncthreads();

    if (active_thread) {
        // now get on thread H, xs0_bar, xs_bar, xa0 for single aero node
        T xs_bar[3], xs0_bar[3], H[9], xa0[3];
        for (int i = 0; i < 3; i++) {
            xs_bar[i] = loc_xs_bar[3 * threadIdx.x + i];
            xs0_bar[i] = loc_xs0_bar[3 * threadIdx.x + i];
            xa0[i] = loc_xa0[3 * threadIdx.x + i];
        }
        for (int i = 0; i < 9; i++) {
            H[i] = loc_H[9 * threadIdx.x + i];
        }

        T ua[3];
        compute_aero_disp<T>(xa0, xs0_bar, xs_bar, H, ua);

        // copy ua back into shared memory
        for (int i = 0; i < 3; i++) {
            loc_ua[3 * threadIdx.x + i] = ua[i];
        }
    } // end of active_thread check

    __syncthreads();

    // copy shared ua into global memory
    for (int i = threadIdx.x; i < 3 * NA_PER_BLOCK && (3* start_aero_ind + i) < 3 * na; i += blockDim.x) {
        glob_ua[3 * start_aero_ind + i] = loc_ua[i];
    }
}

template <typename T, int NN, bool exact_givens = true>
__GLOBAL__ void transfer_disps_oneshot_kernel(int nn, T H_reg, DeviceVec<int> aerostruct_conn, DeviceVec<T> weights,
                                      DeviceVec<T> xs0, DeviceVec<T> xs, DeviceVec<T> xa0,
                                      DeviceVec<T> ua) {
    // each block of threads if just computing ua for one aero node
    // among all the nearest neighbor struct nodes
    int aero_ind = blockIdx.x;
    int na = xa0.getSize() / 3;
    int ns = xs0.getSize() / 3;
    
    __SHARED__ int loc_conn[NN];
    __SHARED__ T loc_w[NN];
    __SHARED__ T loc_xs0[3 * NN];
    __SHARED__ T loc_xs[3 * NN];
    __SHARED__ T loc_xa0[3];

    __SHARED__ T xs0_bar[3];
    __SHARED__ T xs_bar[3];
    __SHARED__ T H[9];

    // copy data from global to shared
    int glob_start = aero_ind * nn;
    bool active_thread = (glob_start + threadIdx.x) < na * nn;
    aerostruct_conn.copyValuesToShared(active_thread, threadIdx.x, nn, blockDim.x, aero_ind * nn,
                                       &loc_conn[0]);
    weights.copyValuesToShared(active_thread, threadIdx.x, nn, blockDim.x, aero_ind * nn,
                               &loc_w[0]);
    xa0.copyValuesToShared(true, threadIdx.x, 3, blockDim.x, 3 * aero_ind, &loc_xa0[0]);
    // need to use conn here so may need copyElemValuesToShared
    xs0.copyElemValuesToShared(true, threadIdx.x, blockDim.x, 3, nn, &loc_conn[0], &loc_xs0[0]);
    xs.copyElemValuesToShared(true, threadIdx.x, blockDim.x, 3, nn, &loc_conn[0], &loc_xs[0]);
    __syncthreads();

    // all of the following will be reduction-like operations
    // until H is computed (each thread helps in the work)

    // compute the centroids of the nearest neighbors
    // xs0_bar
    memset(&xs0_bar[0], 0.0, 3 * sizeof(T));
    for (int i = threadIdx.x; i < 3 * nn; i += blockDim.x) {
        int inode = i / 3;
        int idim = i % 3;
        atomicAdd(&xs0_bar[idim], loc_w[inode] * loc_xs0[i]);
    }

    // xs_bar
    memset(&xs_bar[0], 0.0, 3 * sizeof(T));
    for (int i = threadIdx.x; i < 3 * nn; i += blockDim.x) {
        int inode = i / 3;
        int idim = i % 3;
        atomicAdd(&xs_bar[idim], loc_w[inode] * loc_xs[i]);
    }
    __syncthreads();

    // compute covariance H (reduction step across threads)
    memset(&H[0], 0.0, 9 * sizeof(T));
    for (int i = threadIdx.x; i < 9 * nn; i += blockDim.x) {
        int inode = i / 9;
        int i9 = i % 9;
        int idim = i9 / 3;
        int jdim = i9 % 3;

        // H += w_{inode} * p_{inode,idim} * q_{inode,jdim}
        // where p = xS - xSbar for each node and dim, sim q = xS0 - xS0bar
        // could do warp shuffling here
        T h_val = loc_w[inode] * (loc_xs[3 * inode + idim] - xs_bar[idim]) *
                  (loc_xs0[3 * inode + jdim] - xs0_bar[jdim]);
        atomicAdd(&H[i9], h_val);
    }
    __syncthreads();

    // diagonalization of u for stability
    // change later to use trace(H)
    H[0] += H_reg;
    H[4] += H_reg;
    H[8] += H_reg;

    // after computing H, each thread is going to do the same thing
    // and we'll just average the result (same answer), but computations are cheap

    // get SVD of H (call device function in svd_utils.h)
    T R[9];
    // bool print = (aero_ind == 522 || aero_ind == 521) && threadIdx.x == 0;
    bool print = (aero_ind == 521) && threadIdx.x == 0;
    // bool print = false;
    computeRotation<T, exact_givens>(H, R, print);

    // compute disp offsets, r = xa0 - xs0_bar
    T r[3];
    A2D::VecSumCore<T, 3>(1.0, loc_xa0, -1.0, xs0_bar, r);

    // perform rotation and translations
    // rho = R * r + t
    T rho[3];
    // note for some reason it looks like RT in F2F MELD code
    A2D::MatVecCore<T, 3, 3>(R, r, rho);

    T loc_xa[3];
    A2D::VecSumCore<T, 3>(xs_bar, rho, loc_xa);

    T loc_ua[3];
    A2D::VecSumCore<T, 3>(1.0, loc_xa, -1.0, loc_xa0, loc_ua);

    // debug states
    T loc_us[3];
    A2D::VecSumCore<T, 3>(1.0, xs_bar, -1.0, xs0_bar, loc_us);

    // update xa and u0 globally with add reduction by the blockDim.x
    // int nb = blockDim.x;
    // should probably do warp shuffle here among the threads to speed it up
    // before atomic add
    if (threadIdx.x == 0) {
        for (int i = 0; i < 3; i++) {
            // atomicAdd(&xa[3 * aero_ind+i], loc_xa[i]);
            atomicAdd(&ua[3 * aero_ind+i], loc_ua[i]);
        }
    }
}

template <typename T, typename T2, bool linear, bool exact_givens = true>
__HOST_DEVICE__ void compute_virtual_work_load(
    int idim, T weight, T xs0[3], T xs[3], T xa0[3], T xa[3], 
    T xs0_bar[3], T xs_bar[3], T H[9], T fa[3], T *virtual_work_load) {
    // struct disp at this neighbor node with AD
    T2 us[3];
    for (int i = 0; i < 3; i++) {
        us[i].value = xs[i] - xs0[i];
        us[i].deriv[0] = 0.0;
    }
    us[idim].deriv[0] = 1.0;

    // xs_bar with AD
    T2 xs_bar2[3];
    for (int i = 0; i < 3; i++) {
        xs_bar2[i].value = xs_bar[i];
        xs_bar2[i].deriv[0] = 0.0;
    }
    xs_bar2[idim].deriv[0] = weight;

    // compute AD version of H
    T2 H2[9];
    for (int i = 0; i < 9; i++) {
        // copy forward analysis
        H2[i].value = H[i];
        H2[i].deriv[0] = 0.0;
    }
    for (int jdim = 0; jdim < 3; jdim++) {
        // compute forward AD part
        if constexpr (!linear) {
            // only on this nn node contributes to H2 deriv here
            T2 temp = weight * (xs0[idim] + us[idim] - 
                xs_bar2[idim]) * (xs0[jdim] - xs0_bar[jdim]);
            H2[3 * idim + jdim].deriv[0] = temp.deriv[0];
        }
    }

    // AD version of rotation matrix
    T2 R[9];
    bool print = false;
    computeRotation<T2, exact_givens>(H2, R, print);

    // compute disp offsets, r = xa0 - xs0_bar
    T2 r2[3];
    {
        T r[3];
        A2D::VecSumCore<T, 3>(1.0, xa0, -1.0, xs0_bar, r);
        for (int i = 0; i < 3; i++) {
            r2[i] = r[i]; // passive AD type
        }
    }
    
    // perform rotation and translations
    // rho = R * r + t
    T2 rho[3];
    A2D::MatVecCore<T2, 3, 3>(R, r2, rho);

    // AD version of final/deformed aero coords
    T2 xa2[3];
    A2D::VecSumCore<T2, 3>(xs_bar2, rho, xa2);

    // final virtual work expression for the load in the idim direction
    T2 aero_dot = 0.0;
    for (int i = 0; i < 3; i++) {
        aero_dot += fa[i] * (xa2[i] - xa0[i]);
    }
    *virtual_work_load = aero_dot.deriv[0];
}

template <typename T, int NN_PER_BLOCK, bool linear, bool exact_givens = true>
__GLOBAL__ void transfer_loads_kernel(
    int nn, DeviceVec<int> aerostruct_conn, DeviceVec<T> weights,
    DeviceVec<T> glob_xs0, DeviceVec<T> glob_xs,
    DeviceVec<T> glob_xa0, DeviceVec<T> glob_xa, 
    DeviceVec<T> glob_xs0_bar, DeviceVec<T> glob_xs_bar, 
    DeviceVec<T> glob_H, DeviceVec<T> glob_fa, DeviceVec<T> glob_fs)  {

    using T2 = A2D::ADScalar<T,1>;

    // each block of threads if just computing ua for one aero node
    // among all the nearest neighbor struct nodes
    int aero_ind = blockIdx.x;
    int nn_block_start = blockIdx.y * NN_PER_BLOCK;
    int direc = threadIdx.y;

    int na = glob_xa0.getSize() / 3;
    int ns = glob_xs0.getSize() / 3;
    
    __SHARED__ int loc_conn[NN_PER_BLOCK];
    __SHARED__ T loc_w[NN_PER_BLOCK];
    __SHARED__ T loc_xa0[3];
    __SHARED__ T loc_xa[3];
    __SHARED__ T loc_xs0[3 * NN_PER_BLOCK];
    __SHARED__ T loc_xs[3 * NN_PER_BLOCK];
    __SHARED__ T loc_xs0_bar[3];
    __SHARED__ T loc_xs_bar[3];
    __SHARED__ T loc_H[9];
    __SHARED__ T loc_fa[3];
    __SHARED__ T loc_fs[3 * NN_PER_BLOCK];

    // copy data from global to shared
    int xy_thread_start = threadIdx.y * blockDim.x + threadIdx.x;
    int xy_thread_stride = blockDim.x * blockDim.y;
    int block_ind_start = aero_ind * nn + nn_block_start;
    int active_thread = (block_ind_start + xy_thread_start) < na * nn;
    aerostruct_conn.copyValuesToShared(active_thread, xy_thread_start, NN_PER_BLOCK, xy_thread_stride, block_ind_start,
        &loc_conn[0]);
    weights.copyValuesToShared(active_thread, xy_thread_start, NN_PER_BLOCK, xy_thread_stride, block_ind_start,
        &loc_w[0]);
    glob_xa0.copyValuesToShared(true, xy_thread_start, 3, xy_thread_stride, 3 * aero_ind, &loc_xa0[0]);
    glob_xa.copyValuesToShared(true, xy_thread_start, 3, xy_thread_stride, 3 * aero_ind, &loc_xa[0]);
    glob_xs0.copyElemValuesToShared(threadIdx.y == 0, threadIdx.x, blockDim.x, 3, NN_PER_BLOCK, &loc_conn[0], &loc_xs0[0]);
    glob_xs.copyElemValuesToShared(threadIdx.y == 0, threadIdx.x, blockDim.x, 3, NN_PER_BLOCK, &loc_conn[0], &loc_xs[0]);
    glob_xs0_bar.copyValuesToShared(true, xy_thread_start, 3, xy_thread_stride, 3 * aero_ind, &loc_xs0_bar[0]);
    glob_xs_bar.copyValuesToShared(true, xy_thread_start, 3, xy_thread_stride, 3 * aero_ind, &loc_xs_bar[0]);
    glob_H.copyValuesToShared(true, xy_thread_start, 9, xy_thread_stride, 9 * aero_ind, &loc_H[0]);
    glob_fa.copyValuesToShared(true, xy_thread_start, 3, xy_thread_stride, 3 * aero_ind, &loc_fa[0]);
    __syncthreads();

    bool active_thread2 = active_thread && (nn_block_start + threadIdx.x) < nn;
    if (active_thread2) {
        // now get on thread H, xs0_bar, xs_bar, xa0, xa for the aero node of this thread
        T xs_bar[3], xs0_bar[3], H[9], xa0[3], xa[3], xs0[3], xs[3], fa[3], weight = loc_w[threadIdx.x];
        for (int i = 0; i < 3; i++) {
            xs_bar[i] = loc_xs_bar[i];
            xs0_bar[i] = loc_xs0_bar[i];
            xa0[i] = loc_xa0[i];
            xa[i] = loc_xa[i];
            xs0[i] = loc_xs0[3 * threadIdx.x + i];
            xs[i] = loc_xs[3 * threadIdx.x + i];
            fa[i] = loc_fa[i];
        }
        for (int i = 0; i < 9; i++) {
            H[i] = loc_H[i];
        }

        T fs_load;
        compute_virtual_work_load<T, T2, linear, exact_givens>(
            direc, weight, xs0, xs, xa0, xa,
            xs0_bar, xs_bar, H, fa, &fs_load 
        );

        // add into global memory here
        int global_struct_node = loc_conn[threadIdx.x];
        int my_ind = 3 * global_struct_node + direc;
        atomicAdd(&glob_fs[my_ind], fs_load);
    } // end of active_thread check
}

template <typename T, int NN, bool linear, bool exact_givens = true>
__GLOBAL__ void transfer_loads_oneshot_kernel(int nn, T H_reg, DeviceVec<int> aerostruct_conn, DeviceVec<T> weights,
                                      DeviceVec<T> xs0, DeviceVec<T> xs, DeviceVec<T> xa0,
                                      DeviceVec<T> xa, DeviceVec<T> fa, DeviceVec<T> fs) {
    // each block of threads if just computing ua for one aero node
    // among all the nearest neighbor struct nodes
    int aero_ind = blockIdx.x;
    int na = xa0.getSize() / 3;
    int ns = xs0.getSize() / 3;

    using T2 = A2D::ADScalar<T, 1>;

    // #define NN 32 // need nn < 64 then
    // may want to check this and throw error if nn > NN          

    __SHARED__ int loc_conn[NN];
    __SHARED__ T loc_w[NN];
    __SHARED__ T loc_xs0[3 * NN];
    __SHARED__ T loc_xs[3 * NN];
    __SHARED__ T loc_xa0[3];
    __SHARED__ T loc_fa[3];

    __SHARED__ T xs0_bar[3];
    __SHARED__ T xs_bar[3];
    __SHARED__ T loc_d[3];
    __SHARED__ T H[9];

    int xy_thread_start = threadIdx.y * blockDim.x + threadIdx.x;
    int xy_thread_stride = blockDim.x * blockDim.y;

    // copy data from global to shared
    int glob_start = aero_ind * nn;
    bool active_thread = (glob_start + xy_thread_start) < na * nn;
    aerostruct_conn.copyValuesToShared(active_thread, xy_thread_start, nn, xy_thread_stride, aero_ind * nn,
                                       &loc_conn[0]);
    weights.copyValuesToShared(active_thread, xy_thread_start, nn, xy_thread_stride, aero_ind * nn,
                               &loc_w[0]);
    xa0.copyValuesToShared(true, xy_thread_start, 3, xy_thread_stride, 3 * aero_ind, &loc_xa0[0]);
    fa.copyValuesToShared(true, xy_thread_start, 3, xy_thread_stride, 3 * aero_ind, &loc_fa[0]);
    xs0.copyElemValuesToShared(threadIdx.y == 0, threadIdx.x, blockDim.x, 3, nn, &loc_conn[0], &loc_xs0[0]);
    xs.copyElemValuesToShared(threadIdx.y == 0, threadIdx.x, blockDim.x, 3, nn, &loc_conn[0], &loc_xs[0]);
    __syncthreads();

    // first need to get xs0_bar, xs_bar, no forward AD yet
    memset(&xs0_bar[0], 0.0, 3 * sizeof(T));
    memset(&xs_bar[0], 0.0, 3 * sizeof(T));
    for (int i = xy_thread_start; i < 3 * nn; i += xy_thread_stride) {
        int inode = i / 3;
        int idim = i % 3;
        atomicAdd(&xs0_bar[idim], loc_w[inode] * loc_xs0[i]);
    }
    __syncthreads();
    // return;

    // xs_bar
    for (int i = xy_thread_start; i < 3 * nn; i += xy_thread_stride) {
        int inode = i / 3;
        int idim = i % 3;
        atomicAdd(&xs_bar[idim], loc_w[inode] * loc_xs[i]);
    }
    __syncthreads();

    memset(&H[0], 0.0, 9 * sizeof(T));
    for (int i = xy_thread_start; i < 9 * nn; i += xy_thread_stride) {
        int inode = i / 9;
        int i9 = i % 9;
        int idim = i9 / 3;
        int jdim = i9 % 3;

        // H += w_{inode} * p_{inode,idim} * q_{inode,jdim}
        // where p = xS - xSbar for each node and dim, sim q = xS0 - xS0bar
        // could do warp shuffling here
        T h_val = loc_w[inode] * (loc_xs[3 * inode + idim] - xs_bar[idim]) *
                  (loc_xs0[3 * inode + jdim] - xs0_bar[jdim]);
        atomicAdd(&H[i9], h_val);
    }
    __syncthreads();

    // diagonalization of u for stability
    H[0] += H_reg;
    H[4] += H_reg;
    H[8] += H_reg;

    // return;

    // forward AD section on uA(uS,n) for single nn node n and x,y,z direc on each thread
    // ----------------------------------------------------------------------------------

    // GPU parallelization settings, each thread is doing one nearest neighbor and one spatial dim
    int inode = threadIdx.x; // nearest neighbor node
    int idim = threadIdx.y;
    int global_struct_node = loc_conn[inode];

    // setup dot(uS,n) for forward AD on this single node, single direc
    // T2 is the forward AD type
    T2 this_us[3];
    for (int i = 0; i < 3; i++) {
        this_us[i].value = loc_xs[3 * inode + i] - loc_xs0[3 * inode + i];
        this_us[i].deriv[0] = 0.0;
    }
    this_us[idim].deriv[0] = 1.0; // only one spatial dim deriv per thread


    // now dot(uS) => dot(xS_bar), only single node nonzero dot so easy computation
    // atomicAdd(&xs_bar[idim], loc_w[inode] * loc_xs[i]);
    T2 xs_bar2[3];
    for (int i = 0; i < 3; i++) {
        xs_bar2[i].value = xs_bar[i]; // copy forward analysis       
        xs_bar2[i].deriv[0] = 0.0; // don't know why but apparently I need this here
    }
    xs_bar2[idim].deriv[0] = loc_w[inode];

    // now compute dot(H) forward prop
    T2 H2[9];
    for (int i = 0; i < 9; i++) {
        // copy forward analysis
        H2[i].value = H[i];
        H2[i].deriv[0] = 0.0;
    }
    for (int jdim = 0; jdim < 3; jdim++) {
        // compute forward AD part
        T2 temp = loc_w[inode] * (loc_xs0[3 * inode + idim] + this_us[idim] - xs_bar2[idim]) * 
        (loc_xs0[3 * inode + jdim] - xs0_bar[jdim]);
        if constexpr (!linear) {
            // only include svd jacobian terms if nonlinear MELD
            H2[3 * idim + jdim].deriv[0] = temp.deriv[0];
        }
    }

    // now forward AD types through the SVD and final uA disp calculation

    // get SVD of H (call device function in svd_utils.h)
    T2 R[9];
    // constexpr bool print = aero_ind == 279 and global_struct_node == 71 and threadIdx.y == 0;
    // const bool print = global_struct_node == 217 and threadIdx.x == 0;
    const bool print = false;
    computeRotation<T2, exact_givens>(H2, R, print);    

    // compute disp offsets, r = xa0 - xs0_bar
    T r[3];
    A2D::VecSumCore<T, 3>(1.0, loc_xa0, -1.0, xs0_bar, r);
    T2 r2[3];
    for (int i = 0; i < 3; i++) {
        r2[i] = r[i]; // passive AD type
    }

    // perform rotation and translations
    // rho = R * r + t
    T2 rho[3];
    // note for some reason it looks like RT in F2F MELD code
    A2D::MatVecCore<T2, 3, 3>(R, r2, rho);

    T2 loc_xa[3];
    A2D::VecSumCore<T2, 3>(xs_bar2, rho, loc_xa);

    T2 loc_ua[3];
    for (int i = 0; i < 3; i++) {
        loc_ua[i] = loc_xa[i] - loc_xa0[i];
    }

    // now compute the dot product dot(uA) * fA where dot(cdot) is forward AD state
    // and then we'll add this into the fS as our calculation of (duA/duS,n)^T fA
    // for single node, single direction
    T2 aero_dot = loc_ua[0] * loc_fa[0] + loc_ua[1] * loc_fa[1] + loc_ua[2] * loc_fa[2];
    T fS_contribution = aero_dot.deriv[0];

    int my_ind = 3 * global_struct_node + idim;

    atomicAdd(&fs[my_ind], fS_contribution);
}