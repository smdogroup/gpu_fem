
#pragma once
#include "cuda_utils.h"
#include "linalg/vec.h"
#include "locate_point.h"
#ifdef USE_GPU
#include <cublas_v2.h>

#include "lin_meld.cuh"
#endif

// intended for the GPU here, later could generalize it if you want
// main MELD paper https://arc.aiaa.org/doi/epdf/10.2514/1.J057318
// this is the nonlinear version of MELD, there's also a linearized form
// the source code is in the FUNtoFEM github,
// https://github.com/smdogroup/funtofem/blob/master/src/MELD.cpp

template <typename T>
class LinearizedMELD {
   public:
    LinearizedMELD(DeviceVec<T> &xs0, DeviceVec<T> &xa0, T beta, int num_nearest, int sym)
        : xs0(xs0), xa0(xa0), beta(beta), nn(num_nearest), sym(sym) {
        // assumes 3D so that xs0, xa0 are 3*nS, 3*nA sizes
        ns = xs0.getSize() / 3;
        na = xa0.getSize() / 3;

        xs = DeviceVec<T>(3 * ns);
        us = DeviceVec<T>(3 * ns);

        xa = DeviceVec<T>(3 * na);
        ua = DeviceVec<T>(3 * na);

        fa = DeviceVec<T>(3 * na);
        fs = DeviceVec<T>(3 * ns);

        // global_H = DeviceVec<T>(9 * na);
        // global_rhs = DeviceVec<T>(3 * na);
        // global_soln = DeviceVec<T>(3 * na);
    }

    // function declarations (for ease of use)
    // ---------------------------------------------------------------

    __HOST__ void initialize();
    __HOST__ DeviceVec<T> transferDisps(DeviceVec<T> uS);
    __HOST__ DeviceVec<T> transferLoads(DeviceVec<T> fA);

    __HOST__ DeviceVec<T> &getStructDisps() { return us; }
    __HOST__ DeviceVec<T> &getAeroDisps() { return ua; }
    __HOST__ DeviceVec<T> &getStructDeformed() { return xs; }

    __HOST__ void _computeAeroStructConn();

    // ----------------------------------------------------------------
    // end of function declarations

    __HOST__ void initialize() {
        printf("inside initialize\n");

        // was going to maybe compute aero struct connectivity (nearest neighbors)
        // using octree, but instead just going to reuse CPU code for this from F2F MELD
        _computeAeroStructConn();
        printf("\tfinished aero struct conn\n");

        // compute weights (assumes fixed here even) => reinitialize under shape change
        weights = DeviceVec<double>(nn * na);
        dim3 block(32);
        dim3 grid(na);

        compute_weights_kernel<T><<<grid, block>>>(nn, aerostruct_conn, xs0, xa0, beta, weights);
        CHECK_CUDA(cudaDeviceSynchronize());
        printf("\tfinished weights kernel\n");
    }

    __HOST__ void _computeAeroStructConn() {
        HostVec<int> conn(nn * na);
        // printf("inside aero struct conn\n");
        auto h_xa0 = xa0.createHostVec();
        auto h_xs = xs0.createHostVec();

        int min_bin_size = 10;
        int npts = h_xs.getSize() / 3;
        auto *locator = new LocatePoint<T>(h_xs.getPtr(), npts, min_bin_size);
        // printf("created locate point\n");

        int *indx = new int[nn];
        T *dist = new T[nn];

        // For each aerodynamic node, copy the indices of the nearest n structural
        // nodes into the conn array
        for (int i = 0; i < na; i++) {
            T loc_xa0[3];
            memcpy(loc_xa0, &h_xa0[3 * i], 3 * sizeof(T));

            locator->locateKClosest(nn, indx, dist, loc_xa0);

            for (int k = 0; k < nn; k++) {
                // not doing reflections for now
                // if (indx[k] >= ns) {
                //     int kk = indx[k] - ns;
                //     conn[nn * i + k] = locate_to_reflected_index[kk];
                // } else {
                //     conn[nn * i + k] = indx[k];
                // }
                conn[nn * i + k] = indx[k];
            }
        }

        // cleanup
        // if (Xs_dup) {
        //     delete[] Xs_dup;
        // }
        if (indx) {
            delete[] indx;
        }
        if (dist) {
            delete[] dist;
        }
        delete locator;

        // then copy onto the device in
        aerostruct_conn = conn.createDeviceVec();
    }

    __HOST__ DeviceVec<T> transferDisps(DeviceVec<T> new_us) {
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

        // transfer disps on the kernel for each aero node
        dim3 block2(32);
        dim3 grid2(na);
        transfer_disps_kernel<T><<<grid2, block2>>>(nn, aerostruct_conn, weights, xs0, us, xa0, ua);
        CHECK_CUDA(cudaDeviceSynchronize());
        printf("\tfinished transfer_disps_kernel\n");

        dim3 block4(32);
        nblocks = (3 * na + block4.x - 1) / block4.x;
        dim3 grid4(nblocks);
        vec_add_kernel<T><<<grid4, block4>>>(ua, xa0, xa);
        CHECK_CUDA(cudaDeviceSynchronize());
        printf("\tfinished vec_add_kernel for ua\n");

        return ua;
    }

    __HOST__ DeviceVec<T> transferLoads(DeviceVec<T> new_fa) {
        printf("inside transferLoads\n");
        new_fa.copyValuesTo(fa);
        fs.zeroValues();

        dim3 block(32);
        dim3 grid(na);
        transfer_loads_kernel<T><<<grid, block>>>(nn, aerostruct_conn, weights, xa0, fa, xs0, fs);
        CHECK_CUDA(cudaDeviceSynchronize());
        printf("\tfinished transfer_loads_kernel\n");

        return fs;
    }

   private:
    DeviceVec<T> xs0, xa0, weights;
    DeviceVec<T> xs, xa;
    DeviceVec<T> us, ua;
    DeviceVec<T> fs, fa;

    // DeviceVec<T> global_H, global_rhs, global_soln;

    DeviceVec<int> aerostruct_conn;
    double beta;
    int sym, nn;
    int na, ns;
};