#pragma once
#include "svd_utils.h"

template <typename T>
__GLOBAL__ void compute_weights_kernel(int nn, DeviceVec<int> aerostruct_conn, DeviceVec<T> xs0,
                                       DeviceVec<T> xa0, double beta, DeviceVec<T> weights) {
    // each block of threads if just computing ua for one aero node
    // among all the nearest neighbor struct nodes
    int aero_ind = blockIdx.x;
    int na = xa0.getSize() / 3;
    int ns = xs0.getSize() / 3;

    // printf("xs0:");
    // printVec<double>(xs0.getSize(), xs0.getPtr());

    // printf("inside host weights kernel\n");

    #define NN 32 // need nn < 64 then
    // may want to check this

    __SHARED__ int loc_conn[NN];
    __SHARED__ T loc_w[NN];
    __SHARED__ T loc_xs0[3 * NN];
    __SHARED__ T loc_xa0[3];
    __SHARED__ T sum[1];

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
    __syncthreads();

    // printf("loc_conn\n");
    // printVec<int>(nn, &loc_conn[0]);
    // printf("loc_xs0\n");
    // printVec<double>(3 * nn, &loc_xs0[0]);
    // printf("loc_xa0\n");
    // printVec<double>(3, &loc_xa0[0]);

    // compute weights (un-normalized first)
    memset(loc_w, 0.0, nn * sizeof(T));
    memset(sum, 0.0, 1 * sizeof(T));
    for (int inode = threadIdx.x; inode < nn; inode += blockDim.x) {
        // first compute the distance squared
        T distsq = 0.0;
        for (int idim = 0; idim < 3; idim++) {
            T delta = loc_xa0[3 * inode + idim] - loc_xs0[3 * inode + idim];
            distsq += delta * delta;
        }

        T new_weight = exp(-beta * distsq);
        loc_w[inode] += new_weight;
        atomicAdd(&sum[0], new_weight);
    }
    __syncthreads();

    // normalize the weights so it becomes a partition of unity
    for (int inode = threadIdx.x; inode < nn; inode += blockDim.x) {
        loc_w[inode] /= sum[0];
    }
    __syncthreads();

    // if (threadIdx.x == 0) {
    //     printf("aero_ind %d\n", aero_ind);
    //     printf("loc_w\n");
    //     printVec<double>(nn, &loc_w[0]);
    // }

    int global_start = aero_ind * nn;
    for (int inode = threadIdx.x; inode < nn; inode += blockDim.x) {
        atomicAdd(&weights[global_start+inode], loc_w[inode]);
        // weights[global_start + inode] = loc_w[inode];
    }
}

template <typename T>
__GLOBAL__ void transfer_disps_kernel(int nn, DeviceVec<int> aerostruct_conn, DeviceVec<T> weights,
                                      DeviceVec<T> xs0, DeviceVec<T> us, DeviceVec<T> xa0, DeviceVec<T> ua) {
    // each block of threads if just computing ua for one aero node
    // among all the nearest neighbor struct nodes
    int aero_ind = blockIdx.x;
    int na = xa0.getSize() / 3;
    int ns = xs0.getSize() / 3;

    #define NN 32 // need nn < 64 then
    // may want to check this
    
    __SHARED__ int loc_conn[NN];
    __SHARED__ T loc_w[NN];
    __SHARED__ T loc_xs0[3 * NN];
    __SHARED__ T loc_us[3 * NN];
    __SHARED__ T loc_xa0[3];

    __SHARED__ T xs0_bar[3];
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
    us.copyElemValuesToShared(true, threadIdx.x, blockDim.x, 3, nn, &loc_conn[0], &loc_us[0]);
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
        T h_val = loc_w[inode] * (loc_xs0[3 * inode + idim] - xs0_bar[idim]) *
                  (loc_xs0[3 * inode + jdim] - xs0_bar[jdim]);
        atomicAdd(&H[i9], h_val);
    }
    __syncthreads();

    // compute Hbar = H - tr(H)
    T Hbar[9];
    for (int i = 0; i < 9; i++) {
        Hbar[i] = H[i];
    }
    T tr_H = Hbar[0] + Hbar[4] + Hbar[8];
    Hbar[0] -= tr_H;
    Hbar[4] -= tr_H;
    Hbar[8] -= tr_H;

    // add a small nugget matrix before the matrix inverse?
    T eta = tr_H * 1e-8; // 
    Hbar[0] += eta;
    Hbar[4] += eta;
    Hbar[8] += eta;

    // compute Hinv = Hbar^-1 of 3x3
    // this was not numerically stable for some points in the mesh
    T Hinv[9];
    A2D::MatInvCore<T,3>(Hbar, Hinv);

    // compute d = xa0 - xs0_bar (length-3 vec) with at same time computing,
    // compute dcross, skew-sym cross-product matrix
    T dcross[9];
    memset(dcross, 0.0, 9 * sizeof(T));
    dcross[2] = loc_xa0[1] - xs0_bar[1];
    dcross[3] = loc_xa0[2] - xs0_bar[2];
    dcross[7] = loc_xa0[0] - xs0_bar[0];

    dcross[1] = -dcross[3];
    dcross[5] = -dcross[7];
    dcross[6] = -dcross[2];
    

    // T det = A2D::MatDetCore<T,3>(Hbar);
    // if (abs(det) < 1e-12 and threadIdx.x == 0) {
    //     printf("det(Hbar) at aeronode %d = %.4e\n", aero_ind, det);
    // }

    // now find Tn = wn * (d^x Hbar^-1 qn^x + I) on the fly and find ua = sum_n Tn * us,n
    for (int inode = threadIdx.x; inode < nn; inode += blockDim.x) {
        T *_xs0 = &loc_xs0[3 * inode];

        // compute qn = xs0,n - xs0_bar as qn^x skew-sym on the fly
        T qcross[9];
        memset(qcross, 0.0, 9 * sizeof(T));
        qcross[2] = _xs0[1] - xs0_bar[1];
        qcross[3] = _xs0[2] - xs0_bar[2];
        qcross[7] = _xs0[0] - xs0_bar[0];

        qcross[1] = -qcross[3];
        qcross[5] = -qcross[7];
        qcross[6] = -qcross[2];


        // compute Tn = wn * (d^x Hbar^-1 qn^x + I) 3x3 matrix
        T Tn[9], tmp[9];
        A2D::MatMatMultCore3x3<T>(dcross, Hinv, Tn);
        A2D::MatMatMultCore3x3<T>(Tn, qcross, tmp);
        tmp[0] += 1.0;
        tmp[4] += 1.0;
        tmp[8] += 1.0;
        A2D::MatScaleCore<T,3,3>(loc_w[inode], tmp, Tn);        

        // add in Tn * uS,n into uA
        T loc_ua[3];
        A2D::MatVecCore<T,3,3>(Tn, &loc_us[3*inode], loc_ua);
        
        // add into the same aero node
        for (int i = 0; i < 3; i++) {
            atomicAdd(&ua[3*aero_ind + i], loc_ua[i]);
        }
    }
}

template <typename T>
__GLOBAL__ void transfer_loads_kernel(int nn, DeviceVec<int> aerostruct_conn, DeviceVec<T> weights,
            DeviceVec<T> xa0, DeviceVec<T> fa, DeviceVec<T> xs0, DeviceVec<T> fs) {
    // each block of threads if just computing ua for one aero node
    // among all the nearest neighbor struct nodes
    int aero_ind = blockIdx.x;
    int na = xa0.getSize() / 3;
    int ns = xs0.getSize() / 3;

    #define NN 32 // need nn < 64 then
    // may want to check this
    
    __SHARED__ int loc_conn[NN];
    __SHARED__ T loc_w[NN];
    __SHARED__ T loc_xs0[3 * NN];
    __SHARED__ T loc_fs[3 * NN];
    __SHARED__ T loc_xa0[3];
    __SHARED__ T loc_fa[3];

    __SHARED__ T xs0_bar[3];
    __SHARED__ T H[9];

    // copy data from global to shared
    int glob_start = aero_ind * nn;
    bool active_thread = (glob_start + threadIdx.x) < na * nn;
    aerostruct_conn.copyValuesToShared(active_thread, threadIdx.x, nn, blockDim.x, aero_ind * nn,
                                        &loc_conn[0]);
    weights.copyValuesToShared(active_thread, threadIdx.x, nn, blockDim.x, aero_ind * nn,
                                &loc_w[0]);
    xa0.copyValuesToShared(true, threadIdx.x, 3, blockDim.x, 3 * aero_ind, &loc_xa0[0]);
    fa.copyValuesToShared(true, threadIdx.x, 3, blockDim.x, 3 * aero_ind, &loc_fa[0]);
    // need to use conn here so may need copyElemValuesToShared
    xs0.copyElemValuesToShared(true, threadIdx.x, blockDim.x, 3, nn, &loc_conn[0], &loc_xs0[0]);
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
        T h_val = loc_w[inode] * (loc_xs0[3 * inode + idim] - xs0_bar[idim]) *
                    (loc_xs0[3 * inode + jdim] - xs0_bar[jdim]);
        atomicAdd(&H[i9], h_val);
    }
    __syncthreads();

    // compute Hbar = H - tr(H)
    T Hbar[9];
    for (int i = 0; i < 9; i++) {
        Hbar[i] = H[i];
    }
    T tr_H = Hbar[0] + Hbar[4] + Hbar[8];
    Hbar[0] -= tr_H;
    Hbar[4] -= tr_H;
    Hbar[8] -= tr_H;

    // add a small nugget matrix before the matrix inverse?
    T eta = tr_H * 1e-8; // 
    Hbar[0] += eta;
    Hbar[4] += eta;
    Hbar[8] += eta;

    // compute Hinv = Hbar^-1 of 3x3
    // this was not numerically stable for some points in the mesh
    T Hinv[9];
    A2D::MatInvCore<T,3>(Hbar, Hinv);

    // compute d = xa0 - xs0_bar (length-3 vec) with at same time computing,
    // compute dcross, skew-sym cross-product matrix
    T dcross[9];
    memset(dcross, 0.0, 9 * sizeof(T));
    dcross[2] = loc_xa0[1] - xs0_bar[1];
    dcross[3] = loc_xa0[2] - xs0_bar[2];
    dcross[7] = loc_xa0[0] - xs0_bar[0];

    dcross[1] = -dcross[3];
    dcross[5] = -dcross[7];
    dcross[6] = -dcross[2];
    

    // T det = A2D::MatDetCore<T,3>(Hbar);
    // if (abs(det) < 1e-12 and threadIdx.x == 0) {
    //     printf("det(Hbar) at aeronode %d = %.4e\n", aero_ind, det);
    // }

    // now find Tn = wn * (d^x Hbar^-1 qn^x + I) on the fly and find ua = sum_n Tn * us,n
    for (int inode = threadIdx.x; inode < nn; inode += blockDim.x) {
        T *_xs0 = &loc_xs0[3 * inode];

        // compute qn = xs0,n - xs0_bar as qn^x skew-sym on the fly
        T qcross[9];
        memset(qcross, 0.0, 9 * sizeof(T));
        qcross[2] = _xs0[1] - xs0_bar[1];
        qcross[3] = _xs0[2] - xs0_bar[2];
        qcross[7] = _xs0[0] - xs0_bar[0];

        qcross[1] = -qcross[3];
        qcross[5] = -qcross[7];
        qcross[6] = -qcross[2];


        // compute Tn = wn * (d^x Hbar^-1 qn^x + I) 3x3 matrix
        T Tn[9], tmp[9];
        A2D::MatMatMultCore3x3<T>(dcross, Hinv, Tn);
        A2D::MatMatMultCore3x3<T>(Tn, qcross, tmp);
        tmp[0] += 1.0;
        tmp[4] += 1.0;
        tmp[8] += 1.0;
        A2D::MatScaleCore<T,3,3>(loc_w[inode], tmp, Tn);        

        // add in Tn^T * fa into fs_n
        A2D::MatVecCore<T,3,3, A2D::MatOp::TRANSPOSE>(Tn, loc_fa, &loc_fs[3*inode]);
        
        int global_inode = loc_conn[inode];
        // if (aero_ind == 650 and threadIdx.x == 0) {
        //     printf("global fs node %d\n", global_inode);
        //     printf("loc_fa:");
        //     printVec<double>(3, loc_fa);
        //     printf("loc_fs:");
        //     printVec<double>(3, &loc_fs[3*inode]);
        // }

        // add loc_fs contribution back into the global fs
        for (int idim = 0; idim < 3; idim++) {
            atomicAdd(&fs[3*global_inode + idim], loc_fs[3*inode+idim]);
        }
    }
}