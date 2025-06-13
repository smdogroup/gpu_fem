#pragma once
#include "../../cuda_utils.h"

template <typename T>
class QuadLinearQuadrature {
   public:
    // Required static data used by other classes
    static constexpr int32_t num_quad_pts = 4;
    static constexpr double rt3 = 0.577350269189626;

    // get one of the four quad pts
    __HOST_DEVICE__ static T getQuadraturePoint(int ind, T* pt) {
        double quad_pts[] = {-rt3, rt3};
        double quad_wts[] = {1.0, 1.0};
        pt[0] = quad_pts[ind % 2];
        pt[1] = quad_pts[ind / 2];
        return quad_wts[ind % 2] * quad_wts[ind / 2];
    }

    __HOST_DEVICE__ static T _get_point(int ind) {
        // switch statements are GPU friendly, treated as lookup table here
        switch(ind) {
            case 0: return -rt3;
            case 1: return rt3;
            default: return 0.0;
        }
    }

    __HOST_DEVICE__ static T getQuadraturePoint(int ind, T& xi, T &eta) {
        /* GPU friendly of getQuadraturePoint */
        xi = _get_point(ind % 2);
        eta = _get_point(ind / 2);
        return 1.0;
    }
};
