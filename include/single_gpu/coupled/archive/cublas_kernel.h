// keep in mind that the cublasDgemmStridedBatched actually did
// matrix multiplication and there is no batched linear solver
// except for in magma..

__HOST__ DeviceVec<T> &transferDisps(DeviceVec<T> &new_us) {
    printf("inside transferDisps\n");
    new_us.copyValuesTo(us);
    xs.zeroValues();

    // add us to xs0
    dim3 block1(32);
    int nblocks = (3 * ns + block1.x - 1) / block1.x;
    dim3 grid1(nblocks);
    vec_add_kernel<T><<<grid1, block1>>>(us, xs0, xs);
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("\tfinished vec_add_kernel for us\n");

    // zero out xa, ua before we change them
    xa.zeroValues();
    ua.zeroValues();
    global_H.zeroValues();
    global_rhs.zeroValues();

    // transfer disps on the kernel for each aero node
    dim3 block2(32);
    dim3 grid2(na);
    transfer_disps_H_kernel<T>
        <<<grid2, block2>>>(nn, aerostruct_conn, weights, xs0, us, xa0, global_H, global_rhs);
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("\tfinished transfer_disps_H_kernel\n");

    // auto h_globalH = global_H.createHostVec();
    // printf("h_globalH:");
    // printVec<double>(h_globalH.getSize(), h_globalH.getPtr());

    // auto h_global_rhs = global_rhs.createHostVec();
    // printf("h_global_rhs:");
    // printVec<double>(h_global_rhs.getSize(), h_global_rhs.getPtr());

    // use cublas to solve H^-1 * rhs for each
    cublasStatus_t blas_stat;
    cublasHandle_t cublas_handle;
    blas_stat = cublasCreate(&cublas_handle);
    const double alpha = 1.0;
    const double beta = 0.0;
    cublasDgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 3, 3, 3, &alpha,
                              global_H.getPtr(), 3, 9, global_rhs.getPtr(), 3, 3, &beta,
                              global_soln.getPtr(), 3, 3, na);

    dim3 block3(32);
    dim3 grid3(na);
    transfer_disps_ua_kernel<T>
        <<<grid3, block3>>>(nn, aerostruct_conn, weights, xs0, us, xa0, global_soln, ua);
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("\tfinished transfer_disps_ua_kernel\n");

    dim3 block4(32);
    nblocks = (3 * na + block4.x - 1) / block4.x;
    dim3 grid4(nblocks);
    vec_add_kernel<T><<<grid4, block4>>>(ua, xa0, xa);
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("\tfinished vec_add_kernel for ua\n");

    return ua;
}

template <typename T>
__GLOBAL__ void transfer_disps_H_kernel(int nn, DeviceVec<int> aerostruct_conn,
                                        DeviceVec<T> weights, DeviceVec<T> xs0, DeviceVec<T> us,
                                        DeviceVec<T> xa0, DeviceVec<T> global_H,
                                        DeviceVec<T> global_rhs) {
    // each block of threads if just computing ua for one aero node
    // among all the nearest neighbor struct nodes
    int aero_ind = blockIdx.x;
    int na = xa0.getSize() / 3;
    int ns = xs0.getSize() / 3;

#define NN 32  // need nn < 64 then
    // may want to check this

    __SHARED__ int loc_conn[NN];
    __SHARED__ T loc_w[NN];
    __SHARED__ T loc_xs0[3 * NN];
    __SHARED__ T loc_us[3 * NN];
    __SHARED__ T loc_xa0[3];

    __SHARED__ T xs0_bar[3];
    __SHARED__ T H[9];
    __SHARED__ T rhs[3];

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

    // now find Tn = wn * (d^x Hbar^-1 qn^x + I) on the fly and find ua = sum_n Tn * us,n
    memset(&rhs[0], 0.0, 3 * sizeof(T));
    for (int inode = threadIdx.x; inode < nn; inode += blockDim.x) {
        const T *_xs0 = &loc_xs0[3 * inode];
        const T *_us = &loc_us[3 * inode];

        // compute rhs = qn cross uS,n  and  qn = xs0,n - xs0_bar
        T qn[3];
        A2D::VecSumCore<T, 3>(1.0, _xs0, -1.0, xs0_bar, qn);
        T loc_rhs[3];
        A2D::VecCrossCore<T>(qn, _us, loc_rhs);

        // if (aero_ind == 40) {
        //     printf("loc_rhs %d:", threadIdx.x);
        //     printVec<double>(3, loc_rhs);
        // }

        // add into the same aero node
        for (int i = 0; i < 3; i++) {
            atomicAdd(&rhs[i], loc_rhs[i] * loc_w[inode]);
        }
    }

    // now add back to global
    if (threadIdx.x == 0) {
        if (aero_ind == 845) {
            printf("Hbar:");
            printVec<double>(9, Hbar);
            printf("rhs:");
            printVec<double>(3, rhs);
        }

        for (int i = 0; i < 9; i++) {
            atomicAdd(&global_H[9 * aero_ind + i], Hbar[i]);
        }
        for (int i = 0; i < 3; i++) {
            atomicAdd(&global_rhs[3 * aero_ind + i], rhs[i]);
        }
    }
}

template <typename T>
__GLOBAL__ void transfer_disps_ua_kernel(int nn, DeviceVec<int> aerostruct_conn,
                                         DeviceVec<T> weights, DeviceVec<T> xs0, DeviceVec<T> us,
                                         DeviceVec<T> xa0, DeviceVec<T> global_soln,
                                         DeviceVec<T> ua) {
    // each block of threads if just computing ua for one aero node
    // among all the nearest neighbor struct nodes
    int aero_ind = blockIdx.x;
    int na = xa0.getSize() / 3;
    int ns = xs0.getSize() / 3;

#define NN 32  // need nn < 64 then
    // may want to check this

    __SHARED__ int loc_conn[NN];
    __SHARED__ T loc_w[NN];
    __SHARED__ T loc_xs0[3 * NN];
    __SHARED__ T loc_us[3 * NN];
    __SHARED__ T loc_xa0[3];

    __SHARED__ T xs0_bar[3];
    __SHARED__ T loc_soln[3];
    __SHARED__ T loc_ua[3];

    // copy data from global to shared
    int glob_start = aero_ind * nn;
    bool active_thread = (glob_start + threadIdx.x) < na * nn;
    aerostruct_conn.copyValuesToShared(active_thread, threadIdx.x, nn, blockDim.x, aero_ind * nn,
                                       &loc_conn[0]);
    weights.copyValuesToShared(active_thread, threadIdx.x, nn, blockDim.x, aero_ind * nn,
                               &loc_w[0]);
    xa0.copyValuesToShared(true, threadIdx.x, 3, blockDim.x, 3 * aero_ind, &loc_xa0[0]);
    global_soln.copyValuesToShared(true, threadIdx.x, 3, blockDim.x, 3 * aero_ind, &loc_soln[0]);
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

    // compute d = xa0 - xs0_bar
    T d[3];
    A2D::VecSumCore<T, 3>(1.0, loc_xa0, -1.0, xs0_bar, d);

    // add the term ua += wn * us,n in
    memset(&loc_ua[0], 0.0, 3 * sizeof(T));
    for (int inode = threadIdx.x; inode < nn; inode += blockDim.x) {
        if (aero_ind == 649 && threadIdx.x == 0) {
            printf("loc_us at node %d:", inode);
            printVec<double>(3, &loc_us[3 * inode]);
        }
        // add into the same aero node
        for (int i = 0; i < 3; i++) {
            atomicAdd(&loc_ua[i], loc_us[3 * inode + i] * loc_w[inode]);
        }
    }

    // now add back to global
    if (threadIdx.x == 0) {
        T ua_diff[3];  // add term d cross soln in
        A2D::VecCrossCore<T>(d, loc_soln, ua_diff);
        A2D::VecSumCore<T, 3>(1.0, ua_diff, 1.0, loc_ua, loc_ua);
        for (int i = 0; i < 3; i++) {
            atomicAdd(&ua[3 * aero_ind + i], loc_ua[i]);
        }

        if (aero_ind == 649) {
            printf("loc_soln:");
            printVec<double>(3, loc_soln);
            printf("global_soln at node 614:");
            printVec<double>(3, &global_soln[3 * 845]);
            printf("d:");
            printVec<double>(3, d);
            printf("ua_diff:");
            printVec<double>(3, ua_diff);
            printf("loc_ua:");
            printVec<double>(3, loc_ua);
        }
    }
}