#pragma once
#include "a2dcore.h"
#include "quadrature.h"

/* this version of the chebyshev basis explicitly computes the constant coefficients for the basis 
   on each interpolation (very inefficient), in the second version (another file), I just pre-compute these myself
*/

template <typename T, class Quadrature_, int order = 2>
class ChebyshevQuadBasis {
   public:
    using Quadrature = Quadrature_;

    // Required for loading solution data
    static constexpr int32_t nx = order + 1; // num nodes in a single direction
    static constexpr int32_t num_nodes = nx * nx;
    static constexpr int32_t param_dim = 2;
    static constexpr double pi = 3.14159265358979323846;
    // no MITC for chebyshev, it's always fully integrated

    // isoperimetric has same #nodes geometry class inside it
    class LinearQuadGeo {
       public:
        // Required for loading nodal coordinates
        static constexpr int32_t spatial_dim = 3;

        // Required for knowning number of spatial coordinates per node
        static constexpr int32_t num_nodes = nx * nx;

        // Number of quadrature points
        static constexpr int32_t num_quad_pts = Quadrature::num_quad_pts;

        // Data-size = spatial_dim * number of nodes
        // static constexpr int32_t geo_data_size = 5 * num_quad_pts;
    };  // end of class LinearQuadGeo
    using Geo = LinearQuadGeo;

    __HOST_DEVICE__ static T eval_chebyshev_1d_poly(T x, int ind) {
        T val = 0.0;
        // GPU friendly if statements here..
        val += (ind == 0) * 1.0;
        val += (ind == 1) * x;
        val += (ind == 2) * (2.0 * x * x - 1.0);
        val += (ind == 3) * (4.0 * x * x * x - 3.0 * x);
        return val;
    }

    __HOST_DEVICE__ static T eval_chebyshev_1d_poly_deriv(T x, int ind) {
        T val = 0.0;
        // GPU friendly if statements here..
        // val += (ind == 0) * 0.0;
        val += (ind == 1) * 1.0;
        val += (ind == 2) * 4.0 * x;
        val += (ind == 3) * (12.0 * x * x - 3.0);
        return val;
    }

    __HOST_DEVICE__ static T get_chebyshev_gauss_point(int ind) {
        T theta = pi * 0.5 / nx;
        T a = cos(theta);
        return -1.0 / a * cos((2.0 * ind + 1) * theta);
    }

    __HOST_DEVICE__ static void getNodePoint(const int n, T pt[]) {
        pt[0] = get_chebyshev_gauss_point(n % nx);
        pt[1] = get_chebyshev_gauss_point(n / nx);
    }

    __HOST_DEVICE__ static T eval_chebyshev_1d(T xi, int k) {
        T xi_k = get_chebyshev_gauss_point(k);
        T theta = pi * 0.5 / nx;
        T a = cos(theta);

        T out = 0.0;
        for (int i = 0; i < nx; i++) {
            T Tik = eval_chebyshev_1d_poly(a * xi_k, i);
            T Ti_in = eval_chebyshev_1d_poly(a * xi, i);

            T denom = i == 0 ? nx : nx / 2.0;
            out += Tik * Ti_in / denom;
        }
        return out;
    }

    __HOST_DEVICE__ static T eval_chebyshev_1d_grad(T xi, int k) {
        T xi_k = get_chebyshev_gauss_point(k);
        T theta = pi * 0.5 / nx;
        T a = cos(theta);

        T out = 0.0;
        for (int i = 0; i < nx; i++) {
            T Tik = eval_chebyshev_1d_poly(a * xi_k, i);
            // the *a converst from d/dx (of chebyshev function) to d/dxi, then later we go to true spatial d/dx
            T Ti_in_deriv = eval_chebyshev_1d_poly_deriv(a * xi, i) * a; 
            T denom = i == 0 ? nx : nx / 2.0;
            out += Tik * Ti_in_deriv / denom;
        }
        return out;
    }

    __HOST_DEVICE__ static void eval_chebyshev_2d_basis(const T pt[], T N[]) {
        for (int i = 0; i < nx * nx; i++) {
            int ix = i % nx, iy = i / nx;
            N[i] = eval_chebyshev_1d(pt[0], ix) * eval_chebyshev_1d(pt[1], iy);
        }
    }

    __HOST_DEVICE__ static void eval_chebyshev_2d_basis_grad(const T pt[], T N[], T Nxi[], T Neta[]) {
        for (int i = 0; i < nx * nx; i++) {
            int ix = i % nx, iy = i / nx;
            N[i] = eval_chebyshev_1d(pt[0], ix) * eval_chebyshev_1d(pt[1], iy);
            Nxi[i] = eval_chebyshev_1d_grad(pt[0], ix) * eval_chebyshev_1d(pt[1], iy);
            Neta[i] = eval_chebyshev_1d(pt[0], ix) * eval_chebyshev_1d_grad(pt[1], iy);
        }
    }
    
    template <int vars_per_node, int num_fields>
    __HOST_DEVICE__ static void interpFields(const T pt[], const T values[], T field[]) {
        T N[num_nodes];
        eval_chebyshev_2d_basis(pt, N);

        for (int ifield = 0; ifield < num_fields; ifield++) {
            field[ifield] = 0.0;
            for (int inode = 0; inode < num_nodes; inode++) {
                field[ifield] += N[inode] * values[inode * vars_per_node + ifield];
            }
        }
    }  // end of interpFields method

    template <int vars_per_node, int num_fields>
    __HOST_DEVICE__ static void interpFieldsGrad(const T pt[], const T values[], T dxi[],
                                                 T deta[]) {
        T N[num_nodes], dNdxi[num_nodes], dNdeta[num_nodes];
        eval_chebyshev_2d_basis_grad(pt, N, dNdxi, dNdeta);

        for (int ifield = 0; ifield < num_fields; ifield++) {
            dxi[ifield] = 0.0;
            deta[ifield] = 0.0;
            for (int inode = 0; inode < num_nodes; inode++) {
                dxi[ifield] += dNdxi[inode] * values[inode * vars_per_node + ifield];
                deta[ifield] += dNdeta[inode] * values[inode * vars_per_node + ifield];
            }
        }
    }  // end of interpFieldsGrad method

    template <int vars_per_node, int num_fields>
    __HOST_DEVICE__ static void interpFieldsTranspose(const T pt[], const T field_bar[],
                                                      T values_bar[]) {
        T N[num_nodes]; 
        eval_chebyshev_2d_basis(pt, N);

        for (int ifield = 0; ifield < num_fields; ifield++) {
            for (int inode = 0; inode < num_nodes; inode++) {
                values_bar[inode * vars_per_node + ifield] += field_bar[ifield] * N[inode];
            }
        }
    }  // end of interpFieldsTranspose method

    template <int vars_per_node, int num_fields>
    __HOST_DEVICE__ static void interpFieldsGradTranspose(const T pt[], const T dxi_bar[],
                                                          const T deta_bar[], T values_bar[]) {
        T N[num_nodes], dNdxi[num_nodes], dNdeta[num_nodes];
        eval_chebyshev_2d_basis_grad(pt, N, dNdxi, dNdeta);

        for (int ifield = 0; ifield < num_fields; ifield++) {
            for (int inode = 0; inode < num_nodes; inode++) {
                values_bar[inode * vars_per_node + ifield] +=
                    dxi_bar[ifield] * dNdxi[inode] + deta_bar[ifield] * dNdeta[inode];
            }
        }
    }  // end of interpFieldsGrad method
};  // end of class ChebyshevQuadBasis
