
#pragma once
#include <cassert>

#include "cuda_utils.h"
#include "linalg/vec.h"
#include "locate_point.h"
#ifdef USE_GPU
#include "meld.cuh"
#endif

// intended for the GPU here, later could generalize it if you want
// main MELD paper https://arc.aiaa.org/doi/epdf/10.2514/1.J057318
// this is the nonlinear version of MELD, there's also a linearized form
// the source code is in the FUNtoFEM github,
// https://github.com/smdogroup/funtofem/blob/master/src/MELD.cpp

template <typename T, int NN_PER_BLOCK_ = 32, bool linear_ = false, bool oneshot_ = false,
          bool exact_givens_ = true>
class MELD {
    static constexpr int NN_PER_BLOCK = NN_PER_BLOCK_;  // want NN_PER_BLOCK to be a multiple of 32
    static constexpr bool linear = linear_;
    static constexpr bool oneshot =
        oneshot_;  // whether load and disp transfer is in oneshot kernel or multiple kernels
                   // (multiple usually faster)
    static constexpr bool exact_givens = exact_givens_;  // exact givens has improved SVD jacobian

   public:
    MELD(DeviceVec<T> &xs0, DeviceVec<T> &xa0, T beta, int num_nearest, int sym, T H_reg,
         bool print = false)
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

        if constexpr (!oneshot) {
            xs0_bar = DeviceVec<T>(3 * na);
            xs_bar = DeviceVec<T>(3 * na);
            H = DeviceVec<T>(9 * na);
        }

        this->H_reg = H_reg;
        this->print = print;

        // auto h_xs0 = xs0.createHostVec();
        // printf("h_xs0 constructor:");
        // printVec<double>(10, h_xs0.getPtr());
    }

    template <int N0 = 6, int NF = 3>
    static HostVec<T> extractVarsVec(int nnodes, HostVec<T> vec) {
        HostVec<T> vec2(NF * nnodes);
        for (int inode = 0; inode < nnodes; inode++) {
            for (int j = 0; j < NF; j++) {
                vec2[NF * inode + j] = vec[N0 * inode + j];
            }
        }
        return vec2;
    }

    template <int N0 = 3, int NF = 6>
    static HostVec<T> expandVarsVec(int nnodes, HostVec<T> vec) {
        HostVec<T> vec2(NF * nnodes);
        for (int inode = 0; inode < nnodes; inode++) {
            for (int j = 0; j < N0; j++) {
                vec2[NF * inode + j] = vec[N0 * inode + j];
            }
        }
        return vec2;
    }

    __HOST__ DeviceVec<T> &getStructDisps() { return us; }
    __HOST__ DeviceVec<T> &getAeroDisps() { return ua; }
    __HOST__ DeviceVec<T> &getStructDeformed() { return xs; }

    __HOST__ void initialize() {
        if (print) printf("inside initialize\n");

        // was going to maybe compute aero struct connectivity (nearest neighbors)
        // using octree, but instead just going to reuse CPU code for this from F2F MELD
        computeAeroStructConn();
        if (print) printf("\tfinished aero struct conn\n");

        // compute weights (assumes fixed here even) => reinitialize under shape change
        weights = DeviceVec<double>(nn * na);

        if (print) printf("checkpt2\n");

        if constexpr (oneshot) {
            dim3 block(NN_PER_BLOCK);
            dim3 grid(na);

            if (print) printf("before oneshot true kernel\n");

            compute_weights_oneshot_kernel<T, NN_PER_BLOCK>
                <<<grid, block>>>(nn, aerostruct_conn, xs0, xa0, beta, weights);

            CHECK_CUDA(cudaDeviceSynchronize());
            if (print) printf("\tfinished weights kernel\n");
        } else {
            dim3 block(NN_PER_BLOCK);
            int NN_MULT = (nn + block.x - 1) / block.x;
            dim3 grid(na, NN_MULT);

            if (print) printf("before oneshot false kernels\n");

            avgdistsq = DeviceVec<T>(na);
            sum_weights = DeviceVec<T>(na);

            compute_avgdistsq_kernel<T, NN_PER_BLOCK>
                <<<grid, block>>>(nn, aerostruct_conn, xs0, xa0, beta, avgdistsq);
            CHECK_CUDA(cudaDeviceSynchronize());
            if (print) printf("finished avgdist kernel\n");

            compute_sum_weights_kernel<T, NN_PER_BLOCK>
                <<<grid, block>>>(nn, aerostruct_conn, xs0, xa0, beta, avgdistsq, sum_weights);
            CHECK_CUDA(cudaDeviceSynchronize());
            if (print) printf("finished sum_weights kernel\n");

            compute_weights_kernel<T, NN_PER_BLOCK><<<grid, block>>>(
                nn, aerostruct_conn, xs0, xa0, beta, avgdistsq, sum_weights, weights);
            CHECK_CUDA(cudaDeviceSynchronize());
            if (print) printf("finished weights kernel\n");
        }
        if (print) printf("skipped kernels?\n");
    }

    __HOST__ void computeAeroStructConn() {
        HostVec<int> conn(nn * na);
        // printf("inside aero struct conn\n");
        auto h_xa0 = xa0.createHostVec();
        auto h_xs = xs0.createHostVec();

        int min_bin_size = 10;
        int npts = h_xs.getSize() / 3;
        auto *locator = new LocatePoint<T>(h_xs.getPtr(), npts, min_bin_size);

        int *indx = new int[nn];
        T *dist = new T[nn];

        // For each aerodynamic node, copy the indices of the nearest n structural
        // nodes into the conn array
        for (int i = 0; i < na; i++) {
            T loc_xa0[3];
            memcpy(loc_xa0, &h_xa0[3 * i], 3 * sizeof(T));

            locator->locateKClosest(nn, indx, dist, loc_xa0);

            for (int k = 0; k < nn; k++) {
                conn[nn * i + k] = indx[k];
            }
        }

        if (indx) {
            delete[] indx;
        }
        if (dist) {
            delete[] dist;
        }
        delete locator;

        aerostruct_conn = conn.createDeviceVec();

        // free data
        h_xa0.free();
        h_xs.free();
    }

    __HOST__ DeviceVec<T> &transferDisps(DeviceVec<T> &new_us) {
        if (print) printf("inside transferDisps\n");
        new_us.copyValuesTo(us);
        xs.zeroValues();
        xa.zeroValues();
        ua.zeroValues();

        // add us to xs0 => xs
        DeviceVec<T>::add_vec(us, xs0, xs);

        // transfer displacements
        // ----------------------

        if constexpr (oneshot) {
            // for oneshot NN_PER_BLOCK needs to be equal to nn right now
            dim3 block(NN_PER_BLOCK);
            dim3 grid(na);

            transfer_disps_oneshot_kernel<T, NN_PER_BLOCK>
                <<<grid, block>>>(nn, H_reg, aerostruct_conn, weights, xs0, xs, xa0, ua);
        } else {
            // the not oneshot case should be faster than oneshot
            dim3 reduction_block(NN_PER_BLOCK);
            int NN_MULT = (nn + reduction_block.x - 1) / reduction_block.x;
            dim3 reduction_grid(na, NN_MULT);

            printf("NN_MULT = %d\n", NN_MULT);

            // we assume here that nn is an even multiple of NN_PER_BLOCK, so let's check it here
            assert(nn % NN_PER_BLOCK == 0);

            // reset aero node data arrays
            xs0_bar.zeroValues();
            xs_bar.zeroValues();
            H.zeroValues();

            // call kernel for xs0_bar, xs_bar centroid
            compute_centroid_kernel<T, NN_PER_BLOCK><<<reduction_grid, reduction_block>>>(
                nn, H_reg, aerostruct_conn, weights, xs0, xs, xs0_bar, xs_bar);
            CHECK_CUDA(cudaDeviceSynchronize());

            // call kernel for covariance matrix H
            compute_covariance_kernel<T, NN_PER_BLOCK><<<reduction_grid, reduction_block>>>(
                nn, H_reg, aerostruct_conn, weights, xs0, xs, xs0_bar, xs_bar, H);
            CHECK_CUDA(cudaDeviceSynchronize());

            // after reduction, now can parallelize over the aero nodes separately!
            // take xs0_bar, xs_bar, H on each aero node and compute SVD rotation R
            // and the final displacements from nonlinear MELD
            constexpr int NA_PER_BLOCK = 32;
            dim3 disp_block(NA_PER_BLOCK);
            int nblocks = (na + disp_block.x - 1) / disp_block.x;
            dim3 disp_grid(nblocks);

            // call kernel that now computes the rotation with SVD from H and final disps
            transfer_disps_kernel<T, NA_PER_BLOCK>
                <<<disp_grid, disp_block>>>(xa0, xs0_bar, xs_bar, H, ua);
        }

        CHECK_CUDA(cudaDeviceSynchronize());
        if (print) printf("\tfinished transfer_disps_kernel\n");

        // ----------------------

        // post add ua to xa0 => xa
        DeviceVec<T>::add_vec(ua, xa0, xa);

        return ua;
    }

    __HOST__ DeviceVec<T> transferLoads(DeviceVec<T> new_fa) {
        if (print) printf("inside transferLoads\n");
        new_fa.copyValuesTo(fa);
        fs.zeroValues();

        // transfer loads kernels
        // ----------------------

        if (print) printf("launch transfer_loads_kernel\n");
        if constexpr (oneshot) {
            // for oneshot NN_PER_BLOCK needs to be equal to nn right now
            dim3 block(NN_PER_BLOCK, 3);  // one warp for parallelization by spatial_dim
            dim3 grid(na);

            transfer_loads_oneshot_kernel<T, NN_PER_BLOCK, linear, exact_givens>
                <<<grid, block>>>(nn, H_reg, aerostruct_conn, weights, xs0, xs, xa0, xa, fa, fs);
        } else {
            // the not oneshot case should be faster than oneshot
            // also this one allows parallelizing to more than 64 nn (oneshot can only go up to 64
            // without hitting thread limits since H is reduced on each block first)

            // need to consider nearest neighbors for struct load coalescence, so have to
            // parallelize over that unlike the transfer_disps_kernel which only needs nearest
            // neighbors for centroid, H computations
            // imoprtant for NN_PER_BLOCK to be 32, but allows larger NN this way by second grid dim
            // and as H etc. precomputed
            dim3 loads_block(NN_PER_BLOCK, 3);  // 3 is for x,y,z load component
            int NN_mult = (nn + loads_block.x - 1) / loads_block.x;
            dim3 loads_grid(na, NN_mult);  // parallelize over all nearest neighbors too

            // we assume here that nn is an even multiple of NN_PER_BLOCK, so let's check it here
            assert(nn % NN_PER_BLOCK == 0);

            // call kernel that now computes the aero to struct load transfer
            transfer_loads_kernel<T, NN_PER_BLOCK, linear, exact_givens>
                <<<loads_grid, loads_block>>>(nn, aerostruct_conn, weights, xs0, xs, xa0, xa,
                                              xs0_bar, xs_bar, H, fa, fs);
        }

        // ----------------------

        CHECK_CUDA(cudaDeviceSynchronize());
        if (print) printf("\tdone with transfer_loads_kernel\n");

        return fs;
    }

    void free() {
        xs0.free();
        xs.free();
        xa0.free();
        xa.free();
        weights.free();
        fa.free();
        fs.free();
        xs0_bar.free();
        xs_bar.free();
        H.free();
        avgdistsq.free();
        sum_weights.free();
        aerostruct_conn.free();
    }

   private:
    DeviceVec<T> xs0, xa0, weights;
    DeviceVec<T> xs, xa;
    DeviceVec<T> us, ua;
    DeviceVec<T> fs, fa;
    DeviceVec<T> xs0_bar, xs_bar, H;
    DeviceVec<T> avgdistsq, sum_weights;
    DeviceVec<int> aerostruct_conn;
    double beta;
    int sym, nn;
    int na, ns;
    T H_reg;
    bool print;
};