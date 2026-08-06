#pragma once
#include "a2dcore.h"
#include "quadrature.h"

template <typename T, class Quadrature_>
class LinearTriangleGeo {
 public:
  // Required static data used by other classes
  using Quadrature = Quadrature_;

  // Required for loading nodal coordinates
  static constexpr int32_t spatial_dim = 2;

  // Required for knowning number of spatial coordinates per node
  static constexpr int32_t num_nodes = 3;

  // Number of quadrature points
  static constexpr int32_t num_quad_pts = Quadrature::num_quad_pts;

  // Data-size = spatial_dim * number of nodes
  static constexpr int32_t geo_data_size = 5 * num_quad_pts;

  // LINEAR interpolation gradient
  // static constexpr

  // jacobian and det
  __HOST_DEVICE__ static T interpParamGradient(const T* pt, const T* xpts,
                                               T* dXdxi) {
    // pt unused here
    // interpMat static for LINEAR triangle element
    constexpr T dNdxi[2 * num_nodes] = {-1, -1, 1, 0, 0, 1};

    A2D::MatMatMultCore<T, num_nodes, spatial_dim, num_nodes, 2, spatial_dim, 2,
                        A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL, false>(
        xpts, dNdxi, dXdxi);

    // return the determinant of dX/dxi
    return dXdxi[0] * dXdxi[3] - dXdxi[1] * dXdxi[2];
  }
};