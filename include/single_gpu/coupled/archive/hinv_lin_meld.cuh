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
    T eta = 1e-2;
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
    

    T det = A2D::MatDetCore<T,3>(Hbar);
    if (abs(det) < 1e-12 and threadIdx.x == 0) {
        printf("det(Hbar) at aeronode %d = %.4e\n", aero_ind, det);
    }

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

        if (aero_ind == 692 && threadIdx.x == 0) {
            printf("detHbar = %.4e\n", det);

            printf("Hinv:");
            printVec<T>(9, Hinv);

            printf("dcross:");
            printVec<T>(9, dcross);

            printf("qcross:");
            printVec<T>(9, qcross);

            printf("Tn:");
            printVec<T>(9, Tn);

            printf("loc_us:");
            printVec<T>(3, loc_us);

            printf("loc_ua:");
            printVec<T>(3, loc_ua);
        }
        
        // add into the same aero node
        for (int i = 0; i < 3; i++) {
            atomicAdd(&ua[3*aero_ind + i], loc_ua[i]);
        }
    }
}