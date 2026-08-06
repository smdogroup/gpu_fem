#pragma once
#include "cuda_utils.h"

template <typename T>
class TriangleQuadrature {
 public:
  // Required static data used by other classes
  static constexpr int32_t num_quad_pts = 4;

  // get one of the three triangle quad points
  __HOST_DEVICE__ static T getQuadraturePoint(int ind, T* pt) {
    switch (ind) {
      case 0:
        pt[0] = 1.0 / 3.0;
        pt[1] = 1.0 / 3.0;
        return -27.0 / 48.0;
      case 1:
        pt[0] = 1.0 / 5.0;
        pt[1] = 3.0 / 5.0;
        return 25.0 / 48.0;
      case 2:
        pt[0] = 1.0 / 5.0;
        pt[1] = 1.0 / 5.0;
        return 25.0 / 48.0;
      case 3:
        pt[0] = 3.0 / 5.0;
        pt[1] = 1.0 / 5.0;
        return 25.0 / 48.0;
      default:
        break;
    }
    return 0; // fails
  }
};