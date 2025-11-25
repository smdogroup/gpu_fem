#pragma once
#include "a2dcore.h"
#include "quadrature.h"

/* this 2nd version of the chebyshev basis pre-computes all constants beforehand
   and we just provide the overall basis functions, no re-compute each interp
   so much more GPU friendly
*/

// supporting class that helps with compile-time specifications
template <typename T, int order>
class ChebyShev1D;  // forward declaration for specialization

template <typename T>
class ChebyShev1D<T, 1> {
   public:
    static constexpr int num_nodes = 2;
    // static constexpr T gps[num_nodes] = { -1.0, 1.0 }; // not easy with GPU

    __HOST_DEVICE__ __forceinline__ static T getXi(int i) { return (i == 0) ? -1.0 : 1.0; }

    // Example basis functions at a given ξ (linear)
    __HOST_DEVICE__ __forceinline__ static void evalBasis(const T xi, T N[num_nodes]) {
        N[0] = 0.5 * (1.0 - xi);
        N[1] = 0.5 * (1.0 + xi);
    }

    __HOST_DEVICE__ __forceinline__ static void evalBasisGrad(const T xi, T dN[num_nodes]) {
        dN[0] = -0.5;
        dN[1] = 0.5;
    }
};

template <typename T>
class ChebyShev1D<T, 2> {
   public:
    static constexpr int num_nodes = 3;
    // static constexpr T gps[num_nodes] = { -1.0, 0.0, 1.0 }; // not easy with GPU

    __HOST_DEVICE__ __forceinline__ static T getXi(int i) {
        return (i == 0) ? -1.0 : (i == 1 ? 0.0 : 1.0);
    }

    // Quadratic Chebyshev basis
    __HOST_DEVICE__ __forceinline__ static void evalBasis(const T xi, T N[num_nodes]) {
        // can be determined from Chebyshev property that each basis func nz only at its GP
        N[0] = 0.5 * xi * (xi - 1.0);
        N[1] = 1.0 - xi * xi;
        N[2] = 0.5 * xi * (xi + 1.0);
    }

    __HOST_DEVICE__ __forceinline__ static void evalBasisGrad(const T xi, T dN[num_nodes]) {
        dN[0] = xi - 0.5;
        dN[1] = -2.0 * xi;
        dN[2] = xi + 0.5;
    }
};

/* only at third order do the gauss points (GPs) finally differ from standard Gauss-Lobatto points
 * [-1, -0.5, 0.5, 1] and help prevent locking better */
template <typename T>
class ChebyShev1D<T, 3> {
   public:
    static constexpr int num_nodes = 4;

    __HOST_DEVICE__ __forceinline__ static T getXi(int i) {
        if (i == 0) return -1.0;
        if (i == 1) return -0.41421356237309504880;  // -(√2 - 1)
        if (i == 2) return 0.41421356237309504880;   //  +(√2 - 1)
        return 1.0;
    }

    __HOST_DEVICE__ __forceinline__ static void evalBasis(const T x, T N[num_nodes]) {
        const T x2 = x * x;
        const T x3 = x2 * x;
        // N0
        N[0] = (-0.6035533905932737) * x3 + (0.6035533905932737) * x2 + (0.10355339059327376) * x +
               (-0.10355339059327376);
        // N1
        N[1] = (1.4571067811865475) * x3 + (-0.6035533905932737) * x2 + (-1.4571067811865475) * x +
               (0.6035533905932737);
        // N2
        N[2] = (-1.4571067811865475) * x3 + (-0.6035533905932737) * x2 + (1.4571067811865475) * x +
               (0.6035533905932737);
        // N3
        N[3] = (0.6035533905932737) * x3 + (0.6035533905932737) * x2 + (-0.10355339059327376) * x +
               (-0.10355339059327376);
    }

    __HOST_DEVICE__ __forceinline__ static void evalBasisGrad(const T x, T dN[num_nodes]) {
        const T x2 = x * x;
        // dN0/dx = 3*a0*x^2 + 2*b0*x + c0
        dN[0] = (-1.8106601717798211) * x2 + (1.2071067811865474) * x + (0.10355339059327376);
        // dN1/dx
        dN[1] = (4.3713203435596425) * x2 + (-1.2071067811865474) * x + (-1.4571067811865475);
        // dN2/dx
        dN[2] = (-4.3713203435596425) * x2 + (-1.2071067811865474) * x + (1.4571067811865475);
        // dN3/dx
        dN[3] = (1.8106601717798211) * x2 + (1.2071067811865474) * x + (-0.10355339059327376);
    }
};

template <typename T, class Quadrature_, int _order = 2>
class ChebyshevQuadBasis {
   public:
    using Quadrature = Quadrature_;
    using Basis1D = ChebyShev1D<T, _order>;
    static constexpr int32_t order = _order;

    // Required for loading solution data
    static constexpr int32_t nx = order + 1;  // num nodes in a single direction
    static constexpr int32_t num_nodes = nx * nx;
    static constexpr int32_t param_dim = 2;
    static constexpr T pi = 3.14159265358979323846;
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

    __HOST_DEVICE__ static T getGaussPoint(int i) {
        // for use in assembler and structured prolong (external tools)
        return Basis1D::getXi(i);
    }

    // generic evalBasis call for interps in multigrid and FEA
    __HOST_DEVICE__ static void getBasis(const T pt[2], T N[num_nodes]) {
        eval_chebyshev_2d_basis(pt, N);
    }

    __HOST_DEVICE__ static void getNodePoint(const int n, T pt[]) {
        pt[0] = Basis1D::getXi(n % nx);
        pt[1] = Basis1D::getXi(n / nx);
    }

    __HOST_DEVICE__ static void eval_chebyshev_2d_basis(const T pt[], T N[]) {
        T N1[nx], N2[nx];
        Basis1D::evalBasis(pt[0], N1);
        Basis1D::evalBasis(pt[1], N2);

#pragma unroll
        for (int i = 0; i < nx * nx; i++) {
            int ix = i % nx, iy = i / nx;
            N[i] = N1[ix] * N2[iy];
        }
    }

    __HOST_DEVICE__ static void eval_chebyshev_2d_basis_grad(const T pt[], T Nxi[], T Neta[]) {
        T N1[nx], dN1[nx];
        T N2[nx], dN2[nx];
        Basis1D::evalBasis(pt[0], N1), Basis1D::evalBasis(pt[1], N2);
        Basis1D::evalBasisGrad(pt[0], dN1), Basis1D::evalBasisGrad(pt[1], dN2);

#pragma unroll
        for (int i = 0; i < nx * nx; i++) {
            int i1 = i % nx, i2 = i / nx;
            Nxi[i] = dN1[i1] * N2[i2];
            Neta[i] = N1[i1] * dN2[i2];
        }
    }

    template <int vars_per_node, int num_fields>
    __HOST_DEVICE__ static void interpFields(const T pt[], const T values[], T field[]) {
        T N[num_nodes];
        eval_chebyshev_2d_basis(pt, N);

#pragma unroll
        for (int ifield = 0; ifield < num_fields; ifield++) {
            field[ifield] = 0.0;
#pragma unroll
            for (int inode = 0; inode < num_nodes; inode++) {
                field[ifield] += N[inode] * values[inode * vars_per_node + ifield];
            }
        }
    }  // end of interpFields method

    template <int vars_per_node, int num_fields>
    __HOST_DEVICE__ static void interpFieldsGrad(const T pt[], const T values[], T dxi[],
                                                 T deta[]) {
        T dNdxi[num_nodes], dNdeta[num_nodes];
        eval_chebyshev_2d_basis_grad(pt, dNdxi, dNdeta);

#pragma unroll
        for (int ifield = 0; ifield < num_fields; ifield++) {
            dxi[ifield] = 0.0;
            deta[ifield] = 0.0;
#pragma unroll
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

#pragma unroll
        for (int ifield = 0; ifield < num_fields; ifield++) {
#pragma unroll
            for (int inode = 0; inode < num_nodes; inode++) {
                values_bar[inode * vars_per_node + ifield] += field_bar[ifield] * N[inode];
            }
        }
    }  // end of interpFieldsTranspose method

    template <int vars_per_node, int num_fields>
    __HOST_DEVICE__ static void interpFieldsGradTranspose(const T pt[], const T dxi_bar[],
                                                          const T deta_bar[], T values_bar[]) {
        T dNdxi[num_nodes], dNdeta[num_nodes];
        eval_chebyshev_2d_basis_grad(pt, dNdxi, dNdeta);

#pragma unroll
        for (int ifield = 0; ifield < num_fields; ifield++) {
#pragma unroll
            for (int inode = 0; inode < num_nodes; inode++) {
                values_bar[inode * vars_per_node + ifield] +=
                    dxi_bar[ifield] * dNdxi[inode] + deta_bar[ifield] * dNdeta[inode];
            }
        }
    }  // end of interpFieldsGrad method

    template <int vars_per_node, int num_fields>
    __HOST_DEVICE__ static void interpFieldsMixedGrad(const T pt[], const T values[], T d2mixed[]) {
        // Compute 1D basis and gradients
        T na[nx], nb[nx];
        T dna[nx], dnb[nx];
        Basis1D::evalBasis(pt[0], na), Basis1D::evalBasis(pt[1], nb);
        Basis1D::evalBasisGrad(pt[0], dna), Basis1D::evalBasisGrad(pt[1], dnb);

        for (int ifield = 0; ifield < num_fields; ifield++) {
            T val = 0.0;

            // Loop over 2D nodes: N(xi_i, eta_j) = na[i] * nb[j]
            for (int j = 0; j < nx; j++) {
                for (int i = 0; i < nx; i++) {
                    const int inode = nx * j + i;

                    // Mixed derivative:  dN/dxi * dN/deta = dna[i] * dnb[j]
                    const T Nmixed = dna[i] * dnb[j];

                    val += Nmixed * values[inode * vars_per_node + ifield];
                }
            }

            d2mixed[ifield] = val;
        }
    }

    template <int vars_per_node, int num_fields>
    __HOST_DEVICE__ static void interpFieldsMixedGradTranspose(const T pt[], const T d2mixed_b[],
                                                               T values_b[]) {
        // Compute 1D basis and gradients
        T na[nx], nb[nx];
        T dna[nx], dnb[nx];
        Basis1D::evalBasis(pt[0], na), Basis1D::evalBasis(pt[1], nb);
        Basis1D::evalBasisGrad(pt[0], dna), Basis1D::evalBasisGrad(pt[1], dnb);

        // Loop over 2D nodes to accumulate adjoints
        for (int j = 0; j < nx; j++) {
            for (int i = 0; i < nx; i++) {
                const int inode = nx * j + i;
                T coeff = dna[i] * dnb[j];  // mixed derivative

                for (int ifield = 0; ifield < num_fields; ifield++) {
                    values_b[inode * vars_per_node + ifield] += coeff * d2mixed_b[ifield];
                }
            }
        }
    }
};  // end of class ChebyshevQuadBasis
