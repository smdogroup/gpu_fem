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

    // compute d = xa0 - xs0_bar (length-3 vec) with at same time computing,
    T d[3];
    A2D::VecSumCore<T, 3>(1.0, loc_xa0, -1.0, xs0_bar, d);
    
    // NOTE : was using exact Hinv before (was going to try pinv from least-squares solution)
    // but turns out least-squares system is even worse conditioned
    // btw about 20 out of 900 nodes in my plate problem were ill-conditioned det Hbar < 1e-12
    //   Solution: use iterative conjugate gradient solver since Hbar sym and iterative solvers
    //   work better on ill-conditioned systems
    T det = A2D::MatDetCore<T,3>(Hbar);
    // if (abs(det) < 1e-12 and threadIdx.x == 0) {
    //     printf("det(Hbar) at aeronode %d = %.4e\n", aero_ind, det);
    // }

    // now find Tn = wn * (d^x Hbar^-1 qn^x + I) on the fly and find ua = sum_n Tn * us,n
    for (int inode = threadIdx.x; inode < nn; inode += blockDim.x) {
        const T *_xs0 = &loc_xs0[3 * inode];
        const T *_us = &loc_us[3*inode];

        // solve the linear system Hbar * x = qn cross uS,n
        // then ua += wn * (d cross x + uS,n)

        // compute rhs = qn cross uS,n  and  qn = xs0,n - xs0_bar
        T qn[3];
        A2D::VecSumCore<T, 3>(1.0, _xs0, -1.0, xs0_bar, qn);
        T rhs[3];
        A2D::VecCrossCore<T>(qn, _us, rhs);

    
        // now use conjugate gradient iterative method to solve the system
        // Hbar * x = rhs
        // this as Hbar is sym and Hbar can be ill-conditioned on some nodes 
        // (iterative solvers are better at ill-conditioned systems)
        // ------------------------------------------------------
        T xk[3], xk1[3];
        memset(xk, 0.0, 3 * sizeof(T));

        T rk[3], rk1[3];
        T pk[3], pk1[3];
        T tmpvec[3];
        T alphak, alphak1, betak, tmp1, tmp2, norm;

        // compute initial resid r0 = b - A*x0 and search direc p0 = r0
        A2D::MatVecCoreScale<T,3,3>(-1.0, Hbar, xk, rk);
        A2D::VecSumCore<T,3>(1.0, rhs, 1.0, rk, rk);
        A2D::VecScaleCore<T,3>(1.0, rk, pk);

        if (threadIdx.x == 0 and aero_ind == 37) {
            printf("_us:");
            printVec<double>(3, _us);

            printf("qn:");
            printVec<double>(3, qn);

            printf("rhs:");
            printVec<double>(3, rhs);

            printf("rk:");
            printVec<double>(3, rk);
        }

        // max iterations at 20 for CG method
        int MAX_ITER = 10;
        for (int k = 0; k < MAX_ITER; k++) {
            // compute alpha_k = (r_k^T r_k) / (p_k^T A p_k)
            tmp1 = A2D::VecDotCore<T,3>(rk, rk)*(1.0 + 1e-12);
            A2D::MatVecCoreScale<T,3,3>(1.0, Hbar, pk, tmpvec);
            tmp2 = A2D::VecDotCore<T,3>(tmpvec, pk)*(1.0 + 1e-12);
            alphak = tmp1 / tmp2;

            // update solution x_k+1 = x_k + alpha_k * p_k
            A2D::VecSumCore<T, 3>(1.0, xk, alphak, pk, xk1);

            // update residual r_k+1 = r_k - alpha_k * A * pk
            A2D::VecSumCore<T, 3>(1.0, rk, -alphak, tmpvec, rk1);

            // compute scalar beta_k = (r_k+1^T r_k+1) / (r_k^T r_k)
            tmp2 = A2D::VecDotCore<T,3>(rk1, rk1);
            betak = tmp2 / tmp1;

            // update search direction p_k+1 = r_k+1 + beta_k * pk
            A2D::VecSumCore<T, 3>(betak, pk, 1.0, rk1, pk1);

            // can print residual ||r_k+1|| for debugging
            norm = sqrt(tmp2);
            if (threadIdx.x == 0 and aero_ind == 37) {
                printf("iter %d, resid norm %.4e, alphak %.4e, betak %.4e\n", k, norm, alphak, betak);
            }

            // set to previous iteration storage before going to next one
            A2D::VecScaleCore<T,3>(1.0, xk1, xk);
            A2D::VecScaleCore<T,3>(1.0, pk1, pk);
            A2D::VecScaleCore<T,3>(1.0, rk1, rk);

        }

        // check convergence based on ||r_k+1||, don't want to check inside
        // because if statement slows down GPU code
        if (threadIdx.x == 0 and aero_ind == 37) {
            printf("final resid norm %.4e\n", norm);
        }

        // now find ua contribution = wn * (d cross x + uS,n)
        T loc_ua[3];
        A2D::VecCrossCore<T>(d, xk, loc_ua);
        A2D::VecAddCore<T, 3>(_us, loc_ua);
        A2D::VecScaleCore<T,3>(loc_w[inode], loc_ua, tmpvec);
        A2D::VecScaleCore<T,3>(1.0, tmpvec, loc_ua);
        
        // add into the same aero node
        for (int i = 0; i < 3; i++) {
            atomicAdd(&ua[3*aero_ind + i], loc_ua[i]);
        }
    }
}