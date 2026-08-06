#pragma once
#include "a2dcore.h"
#include "quadrature.h"

template <typename T, class Quadrature_>
class QuadraticTriangleBasis {
 public:
  using Quadrature = Quadrature_;

  // Required for loading solution data
  static constexpr int32_t num_nodes = 6;

  // Parametric dimension
  static constexpr int32_t param_dim = 2;

  __HOST_DEVICE__ static void getBasisGrad(const T* xi, T* dNdxi) {
    // compute the basis function gradients at each basis node

    // basis fcn 0 : N0 = 2x^2 + 4xy - 3x + 2y^2 - 3y + 1
    dNdxi[0] = 4 * xi[0] + 4 * xi[1] - 3;
    dNdxi[1] = 4 * xi[0] + 4 * xi[1] - 3;
    // basis fcn 1 : N1 = x * (2x - 1)
    dNdxi[2] = 4 * xi[0] - 1;
    dNdxi[3] = 0.0;
    // basis fcn 2 : N2 = y * (2y - 1)
    dNdxi[4] = 0.0;
    dNdxi[5] = 4 * xi[1] - 1;
    // basis fcn 3 : N3 = 4xy
    dNdxi[6] = 4 * xi[1];
    dNdxi[7] = 4 * xi[0];
    // basis fcn 4 : N4 = 4y * (-x - y + 1)
    dNdxi[8] = -4.0 * xi[1];
    dNdxi[9] = -4.0 * xi[0] - 8.0 * xi[1] + 4.0;
    // basis fcn 5 : N5 = 4x * (-x - y + 1)
    dNdxi[10] = -8.0 * xi[0] - 4.0 * xi[1] + 4.0;
    dNdxi[11] = -4.0 * xi[0];
  }

  __HOST_DEVICE__ static void getBasisGrad(int nodeInd, const T* xi, T* dNdxi) {
    // compute the gradient for one of the basis functions

    switch (nodeInd) {
      case 0:
        // basis fcn 0 : N0 = 2x^2 + 4xy - 3x + 2y^2 - 3y + 1
        dNdxi[0] = 4 * xi[0] + 4 * xi[1] - 3;
        dNdxi[1] = 4 * xi[0] + 4 * xi[1] - 3;
        break;
      case 1:
        // basis fcn 1 : N1 = x * (2x - 1)
        dNdxi[0] = 4 * xi[0] - 1;
        dNdxi[1] = 0.0;
        break;
      case 2:
        // basis fcn 2 : N2 = y * (2y - 1)
        dNdxi[0] = 0.0;
        dNdxi[1] = 4 * xi[1] - 1;
        break;
      case 3:
        // basis fcn 3 : N3 = 4xy
        dNdxi[0] = 4 * xi[1];
        dNdxi[1] = 4 * xi[0];
        break;
      case 4:
        // basis fcn 4 : N4 = 4y * (-x - y + 1)
        dNdxi[0] = -4.0 * xi[1];
        dNdxi[1] = -4.0 * xi[0] - 8.0 * xi[1] + 4.0;
        break;
      case 5:
        // basis fcn 5 : N5 = 4x * (-x - y + 1)
        dNdxi[0] = -8.0 * xi[0] - 4.0 * xi[1] + 4.0;
        dNdxi[1] = -4.0 * xi[0];
        break;
      default:
        break;
    }
  }

  // don't use explicit N
  // compute U, dU/dxi
  template <int vars_per_node>
  __HOST_DEVICE__ static void interpParamGradient(const T* xi, const T* Un,
                                                  T* dUdxi) {
    T dNdxi[num_nodes * 2];
    getBasisGrad(xi, dNdxi);

    // for (int i = 0; i < 2 * num_nodes; i++) {
    //   printf("dNdxi[%d] = %.8e\n", i, dNdxi[i]);
    // }

    A2D::MatMatMultCore<T, num_nodes, vars_per_node, num_nodes, 2,
                        vars_per_node, 2, A2D::MatOp::TRANSPOSE,
                        A2D::MatOp::NORMAL, false>(Un, dNdxi, dUdxi);
  }

  template <int vars_per_node>
  __HOST_DEVICE__ static void addInterpParamGradientSens(const T* xi,
                                                         const T* dUdxi_bar,
                                                         T* res) {
    T dNdxi[num_nodes * 2];
    getBasisGrad(xi, dNdxi);

    // res += dN/dxi * dUdxi_bar^T
    A2D::MatMatMultCore<T, num_nodes, 2, vars_per_node, 2, num_nodes,
                        vars_per_node, A2D::MatOp::NORMAL,
                        A2D::MatOp::TRANSPOSE, true>(dNdxi, dUdxi_bar, res);
  }
};