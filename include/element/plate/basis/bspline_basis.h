#pragma once
#include "a2dcore.h"
#include "quadrature.h"

/* this is a regular bspline basis for isogeometric analysis (not NURBS yet, NURBS is TODO) */

// =====================================
// 1D bspline basis first
// =====================================

// supporting class that helps with compile-time specifications
template <typename T, int order>
class Bspline1D;  // forward declaration for specialization

template <typename T>
class Bspline1D<T, 2> {
   public:
    static constexpr int num_nodes = 3;

    __HOST_DEVICE__ __forceinline__ static T getXi(int i) {
        return (i == 0) ? -1.0 : (i == 1 ? 0.0 : 1.0);
    }

    // Quadratic bspline basis
    __HOST_DEVICE__ __forceinline__ static void evalBasis(const T xi, const bool bndry[2],
                                                          T N[num_nodes]) {
        T B0 = (1 - xi) * (1 - xi);
        T B1 = 2 * xi * (1 - xi);
        T B2 = xi * xi;

        // basis functions change slightly on boundary to go from C1 to C^{-1} continuity there
        N[0] = 0.5 * (1.0 + bndry[0]) * B0;
        N[1] = 0.5 * (B0 + B1 + B2) - 0.5 * (bndry[0] * B0 + bndry[1] * B2);
        N[2] = 0.5 * (1.0 + bndry[1]) * B2;
    }

    __HOST_DEVICE__ __forceinline__ static void evalBasisGrad(const T xi, const bool bndry[2],
                                                              T dN[num_nodes]) {
        T dB0 = -2 * (1 - xi);
        T dB1 = 2 * (1 - 2 * xi);
        T dB2 = 2 * xi;

        // basis functions change slightly on boundary to go from C1 to C^{-1} continuity there
        dN[0] = 0.5 * (1.0 + bndry[0]) * dB0;
        dN[1] = 0.5 * (dB0 + dB1 + dB2) - 0.5 * (bndry[0] * dB0 + bndry[1] * dB2);
        dN[2] = 0.5 * (1.0 + bndry[1]) * dB2;
    }
};

// TODO : higher order bspline bases later (optional)

template <typename T, class Quadrature_, int _order = 2>
class BsplineQuadBasis {
   public:
    using Quadrature = Quadrature_;
    using Basis1D = Bspline1D<T, _order>;
    static constexpr int32_t order = _order;

    // Required for loading solution data
    static constexpr int32_t nx = order + 1;  // num nodes in a single direction
    static constexpr int32_t num_nodes = nx * nx;
    static constexpr int32_t param_dim = 2;
    static constexpr T pi = 3.14159265358979323846;
    // no tying points, always fully integrated here

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

    //     __HOST_DEVICE__ static T getGaussPoint(int i) {
    //         // for use in assembler and structured prolong (external tools)
    //         return Basis1D::getXi(i);
    //     }

    //     // generic evalBasis call for interps in multigrid and FEA
    //     __HOST_DEVICE__ static void getBasis(const T pt[2], T N[num_nodes]) {
    //         eval_chebyshev_2d_basis(pt, N);
    //     }

    //     __HOST_DEVICE__ static void getNodePoint(const int n, T pt[]) {
    //         pt[0] = Basis1D::getXi(n % nx);
    //         pt[1] = Basis1D::getXi(n / nx);
    //     }

    //     __HOST_DEVICE__ static void eval_chebyshev_2d_basis(const T pt[], T N[]) {
    //         T N1[nx], N2[nx];
    //         Basis1D::evalBasis(pt[0], N1);
    //         Basis1D::evalBasis(pt[1], N2);

    // #pragma unroll
    //         for (int i = 0; i < nx * nx; i++) {
    //             int ix = i % nx, iy = i / nx;
    //             N[i] = N1[ix] * N2[iy];
    //         }
    //     }

    //     __HOST_DEVICE__ static void eval_chebyshev_2d_basis_grad(const T pt[], T Nxi[], T Neta[])
    //     {
    //         T N1[nx], dN1[nx];
    //         T N2[nx], dN2[nx];
    //         Basis1D::evalBasis(pt[0], N1), Basis1D::evalBasis(pt[1], N2);
    //         Basis1D::evalBasisGrad(pt[0], dN1), Basis1D::evalBasisGrad(pt[1], dN2);

    // #pragma unroll
    //         for (int i = 0; i < nx * nx; i++) {
    //             int i1 = i % nx, i2 = i / nx;
    //             Nxi[i] = dN1[i1] * N2[i2];
    //             Neta[i] = N1[i1] * dN2[i2];
    //         }
    //     }

    //     template <int vars_per_node, int num_fields>
    //     __HOST_DEVICE__ static void interpFields(const T pt[], const T values[], T field[]) {
    //         T N[num_nodes];
    //         eval_chebyshev_2d_basis(pt, N);

    // #pragma unroll
    //         for (int ifield = 0; ifield < num_fields; ifield++) {
    //             field[ifield] = 0.0;
    // #pragma unroll
    //             for (int inode = 0; inode < num_nodes; inode++) {
    //                 field[ifield] += N[inode] * values[inode * vars_per_node + ifield];
    //             }
    //         }
    //     }  // end of interpFields method

    //     template <int vars_per_node, int num_fields>
    //     __HOST_DEVICE__ static void interpFieldsGrad(const T pt[], const T values[], T dxi[],
    //                                                  T deta[]) {
    //         T dNdxi[num_nodes], dNdeta[num_nodes];
    //         eval_chebyshev_2d_basis_grad(pt, dNdxi, dNdeta);

    // #pragma unroll
    //         for (int ifield = 0; ifield < num_fields; ifield++) {
    //             dxi[ifield] = 0.0;
    //             deta[ifield] = 0.0;
    // #pragma unroll
    //             for (int inode = 0; inode < num_nodes; inode++) {
    //                 dxi[ifield] += dNdxi[inode] * values[inode * vars_per_node + ifield];
    //                 deta[ifield] += dNdeta[inode] * values[inode * vars_per_node + ifield];
    //             }
    //         }
    //     }  // end of interpFieldsGrad method

    //     template <int vars_per_node, int num_fields>
    //     __HOST_DEVICE__ static void interpFieldsTranspose(const T pt[], const T field_bar[],
    //                                                       T values_bar[]) {
    //         T N[num_nodes];
    //         eval_chebyshev_2d_basis(pt, N);

    // #pragma unroll
    //         for (int ifield = 0; ifield < num_fields; ifield++) {
    // #pragma unroll
    //             for (int inode = 0; inode < num_nodes; inode++) {
    //                 values_bar[inode * vars_per_node + ifield] += field_bar[ifield] * N[inode];
    //             }
    //         }
    //     }  // end of interpFieldsTranspose method

    //     template <int vars_per_node, int num_fields>
    //     __HOST_DEVICE__ static void interpFieldsGradTranspose(const T pt[], const T dxi_bar[],
    //                                                           const T deta_bar[], T values_bar[])
    //                                                           {
    //         T dNdxi[num_nodes], dNdeta[num_nodes];
    //         eval_chebyshev_2d_basis_grad(pt, dNdxi, dNdeta);

    // #pragma unroll
    //         for (int ifield = 0; ifield < num_fields; ifield++) {
    // #pragma unroll
    //             for (int inode = 0; inode < num_nodes; inode++) {
    //                 values_bar[inode * vars_per_node + ifield] +=
    //                     dxi_bar[ifield] * dNdxi[inode] + deta_bar[ifield] * dNdeta[inode];
    //             }
    //         }
    //     }  // end of interpFieldsGrad method
};  // end of class BsplineQuadBasis
