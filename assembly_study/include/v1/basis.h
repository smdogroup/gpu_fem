#pragma once
#include "_basis_utils.h"
#include "a2d/_a2dshell.h"
#include "a2dcore.h"
#include "quadrature.h"
#include "shell_utils.h"

template <typename T, class Quadrature_, int order_ = 2>
class ShellQuadBasisV1 {
   public:
    using Quadrature = Quadrature_;

    // Required for loading solution data
    static constexpr int32_t order = order_;
    static constexpr int32_t num_nodes = order * order;
    static constexpr int32_t param_dim = 2;

    // MITC method => number of tying points for each strain component
    static constexpr int32_t mitcLevel2 = order * (order - 1);
    static constexpr int32_t mitcLevel3 = (order - 1) * (order - 1);
    static constexpr int32_t num_tying_components = 5;  // five gij components for tying strains
    static constexpr int32_t num_all_tying_points = 4 * mitcLevel2 + mitcLevel3;

    // isoperimetric has same #nodes geometry class inside it
    class LinearQuadGeo {
       public:
        // Required for loading nodal coordinates
        static constexpr int32_t spatial_dim = 3;

        // Required for knowning number of spatial coordinates per node
        static constexpr int32_t num_nodes = 4;

        // Number of quadrature points
        static constexpr int32_t num_quad_pts = Quadrature::num_quad_pts;

        // Data-size = spatial_dim * number of nodes
        // static constexpr int32_t geo_data_size = 5 * num_quad_pts;
    };  // end of class LinearQuadGeo
    using Geo = LinearQuadGeo;

    __HOST_DEVICE__ static constexpr int32_t num_tying_points(int icomp) {
        // g11, g22, g12, g23, g13, g33=0 (not included g33)
        int32_t _num_tying_points[] = {mitcLevel2, mitcLevel2, mitcLevel3, mitcLevel2, mitcLevel2};
        return _num_tying_points[icomp];
    }

    __HOST_DEVICE__ static constexpr int32_t tying_point_offsets(int icomp) {
        // g11, g22, g12, g23, g13, g33=0 (not included g33)
        int32_t _tying_point_offsets[] = {0, mitcLevel2, 2 * mitcLevel2,
                                          2 * mitcLevel2 + mitcLevel3, 3 * mitcLevel2 + mitcLevel3};
        return _tying_point_offsets[icomp];
    }

    // shape functions here
    __HOST_DEVICE__ static void lagrangeLobatto1D(const T &u, T *N) {
        if constexpr (order == 2) {
            N[0] = 0.5 * (1.0 - u);
            N[1] = 0.5 * (1.0 + u);
        }
    }

    template <int tyingOrder>
    __HOST_DEVICE__ static void lagrangeLobatto1D_tying(const T &u, T *N) {
        if constexpr (tyingOrder == 1) {
            N[0] = 1.0;
        } else if constexpr (tyingOrder == 2) {
            N[0] = 0.5 * (1.0 - u);
            N[1] = 0.5 * (1.0 + u);
        }
    }

    __HOST_DEVICE__ static void lagrangeLobatto1DGrad(const T u, T *N, T *Nd) {
        if constexpr (order == 2) {
            N[0] = 0.5 * (1.0 - u);
            N[1] = 0.5 * (1.0 + u);

            // dN/du
            Nd[0] = -0.5;
            Nd[1] = 0.5;
        }
    }

    __HOST_DEVICE__ static void lagrangeLobatto2D(const T &xi, const T &eta, T *N) {
        // compute N_i(u) shape function values at each node
        T na[order], nb[order];
        lagrangeLobatto1D(xi, na);
        lagrangeLobatto1D(eta, nb);

        // now compute 2D N_a(xi) * N_b(eta) for each node
        for (int ieta = 0; ieta < order; ieta++) {
            for (int ixi = 0; ixi < order; ixi++) {
                N[order * ieta + ixi] = na[ixi] * nb[ieta];
            }
        }
    }  // end of lagrangeLobatto2D

    __HOST_DEVICE__ static void lagrangeLobatto2DGrad(const T xi, const T eta, T *N, T *dNdxi,
                                                      T *dNdeta) {
        // compute N_i(u) shape function values at each node
        T na[order], nb[order];
        T dna[order], dnb[order];
        lagrangeLobatto1DGrad(xi, na, dna);
        lagrangeLobatto1DGrad(eta, nb, dnb);

        // now compute 2D N_a(xi) * N_b(eta) for each node
        for (int ieta = 0; ieta < order; ieta++) {
            for (int ixi = 0; ixi < order; ixi++) {
                N[order * ieta + ixi] = na[ixi] * nb[ieta];
                dNdxi[order * ieta + ixi] = dna[ixi] * nb[ieta];
                dNdeta[order * ieta + ixi] = na[ixi] * dnb[ieta];
            }
        }
    }  // end of lagrangeLobatto2D_grad

    __HOST_DEVICE__ static T lagrangeLobatto2DGradEtaLight(const int inode, const T &xi,
                                                           const T &eta) {
        return lagrangeLobatto1DLight(inode % order, xi) *
               lagrangeLobatto1DGradLight(inode / order, eta);
    }

    __HOST_DEVICE__ static void assembleFrame(const T a[], const T b[], const T c[], T frame[]) {
        for (int i = 0; i < 3; i++) {
            frame[3 * i] = a[i];
            frame[3 * i + 1] = b[i];
            frame[3 * i + 2] = c[i];
        }
    }

    __HOST_DEVICE__ static void extractFrame(const T frame[], T a[], T b[], T c[]) {
        for (int i = 0; i < 3; i++) {
            a[i] = frame[3 * i];
            b[i] = frame[3 * i + 1];
            c[i] = frame[3 * i + 2];
        }
    }

    __HOST_DEVICE__ static void getNodePoint(const int n, T pt[]) {
        // get xi, eta coordinates of each node point
        pt[0] = -1.0 + (2.0 / (order - 1)) * (n % order);
        pt[1] = -1.0 + (2.0 / (order - 1)) * (n / order);
    }

    template <int vars_per_node, int num_fields>
    __HOST_DEVICE__ static void interpFields(const T pt[], const T values[], T field[]) {
        T N[num_nodes];
        lagrangeLobatto2D(pt[0], pt[1], N);

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
        lagrangeLobatto2DGrad(pt[0], pt[1], N, dNdxi, dNdeta);

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
        T N[num_nodes];  // TODO : double check this method (can we store less
                         // floats here also?)
        lagrangeLobatto2D(pt[0], pt[1], N);

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
        lagrangeLobatto2DGrad(pt[0], pt[1], N, dNdxi, dNdeta);

        for (int ifield = 0; ifield < num_fields; ifield++) {
            for (int inode = 0; inode < num_nodes; inode++) {
                values_bar[inode * vars_per_node + ifield] +=
                    dxi_bar[ifield] * dNdxi[inode] + deta_bar[ifield] * dNdeta[inode];
            }
        }
    }  // end of interpFieldsGradTranspose method

    __HOST_DEVICE__ static void ShellComputeNodeNormal(const T pt[], const T dXdxi[],
                                                       const T dXdeta[], T n0[]) {
        // compute the shell node normal at a single node given already the pre-computed spatial
        // gradients
        T tmp[3];
        A2D::VecCrossCore<T>(dXdxi, dXdeta, tmp);
        T norm = sqrt(A2D::VecDotCore<T, 3>(tmp, tmp));
        A2D::VecScaleCore<T, 3>(1.0 / norm, tmp, n0);
    }

    __HOST_DEVICE__ static void getTyingKnots(T red_knots[], T full_knots[]) {
        if constexpr (order == 2) {
            red_knots[0] = 0.0;
            full_knots[0] = -1.0;
            full_knots[1] = 1.0;
        }
    }

    // section for tying point interpolations
    template <int icomp>
    __HOST_DEVICE__ static void getTyingPoint(const int n, T pt[]) {
        T red_knots[order * (order - 1)], full_knots[order * order];
        getTyingKnots(red_knots, full_knots);

        // use constexpr here to prevent warp divergence
        if constexpr (icomp == 0 || icomp == 4) {  // g11 or g13
            // 1-dir uses reduced knots for 1j strains
            // also order*(order-1) matrix but order-1 columns
            pt[0] = red_knots[n % (order - 1)];
            pt[1] = full_knots[n / (order - 1)];
        } else if constexpr (icomp == 1 || icomp == 3) {  // g22 or g23
            // 2-dir uses reduced knots for 2j strains
            // also order*(order-1) matrix but order columns
            pt[0] = full_knots[n % order];
            pt[1] = red_knots[n / order];
        } else if constexpr (icomp == 2) {  // g12
            // 1-dir and 2-dir use reduced knots here
            pt[0] = red_knots[n % (order - 1)];
            pt[1] = red_knots[n / (order - 1)];
        }
    }

    template <int icomp>
    __HOST_DEVICE__ static void getTyingInterp(const T pt[], T N[]) {
        // get 1d knot vectors
        // T red_knots[(order-1)], full_knots[order];
        // getTyingKnots(red_knots, full_knots);

        T na[order], nb[order];
        lagrangeLobatto1D_tying<order>(pt[0], na);
        lagrangeLobatto1D_tying<order>(pt[1], nb);

        T nar[order], nbr[order];
        lagrangeLobatto1D_tying<order - 1>(pt[0], nar);
        lagrangeLobatto1D_tying<order - 1>(pt[1], nbr);

        if constexpr (icomp == 0) {
            // g11
            for (int j = 0; j < order; j++) {
                for (int i = 0; i < order - 1; i++, N++) {
                    N[0] = nar[i] * nb[j];
                }
            }
        } else if constexpr (icomp == 2) {
            // g12
            for (int j = 0; j < order - 1; j++) {
                for (int i = 0; i < order - 1; i++, N++) {
                    N[0] = nar[i] * nbr[j];
                }
            }
        } else if constexpr (icomp == 4) {
            // g13
            for (int j = 0; j < order; j++) {
                for (int i = 0; i < order - 1; i++, N++) {
                    N[0] = nar[i] * nb[j];
                }
            }
        } else if constexpr (icomp == 1) {
            // g22
            for (int j = 0; j < order - 1; j++) {
                for (int i = 0; i < order; i++, N++) {
                    N[0] = na[i] * nbr[j];
                }
            }
        } else if constexpr (icomp == 3) {
            // g23
            for (int j = 0; j < order - 1; j++) {
                for (int i = 0; i < order; i++, N++) {
                    N[0] = na[i] * nbr[j];
                }
            }
        }
    }  // end of getTyingInterp

    __HOST_DEVICE__ static void ShellComputeNodeNormal(const T pt[], const T xpts[], T n0[]) {
        // compute the shell node normal at a single node given already the pre-computed spatial
        // gradients
        T Xxi[3], Xeta[3];
        interpFieldsGrad<3, 3>(pt, xpts, Xxi, Xeta);
        ShellComputeNodeNormal(pt, Xxi, Xeta, n0);
    }
};
