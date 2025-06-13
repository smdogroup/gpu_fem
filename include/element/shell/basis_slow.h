#pragma once
#include "_basis_utils.h"
#include "a2dcore.h"
#include "a2dshell.h"
#include "quadrature.h"
#include "shell_utils.h"

template <typename T, class Quadrature_, int order_ = 2>
class ShellQuadBasisOld {
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

    // template <int vars_per_node>
    // __HOST_DEVICE__ static void assembleFrameFast(const T &xi, const T &eta, const T values[],
    //                                               const T &n0x, const T &n0y, const T &n0z,
    //                                               T frame[]) {
    //     /* fast implementation of assembleFrame for xpts */

    //     for (int i = 0; i < 3; i++) {
    //         frame[3 * i] = interpFieldsGradFast<XI, vars_per_node>(i, xi, eta, values);
    //         frame[3 * i + 1] = interpFieldsGradFast<ETA, vars_per_node>(i, xi, eta, values);
    //     }
    //     frame[2] = n0x;
    //     frame[5] = n0y;
    //     frame[8] = n0z;
    // }

    __HOST_DEVICE__ static void extractFrame(const T frame[], T a[], T b[], T c[]) {
        for (int i = 0; i < 3; i++) {
            a[i] = frame[3 * i];
            b[i] = frame[3 * i + 1];
            c[i] = frame[3 * i + 2];
        }
    }

    __HOST_DEVICE__ static void getNodePoint(const int n, T &xi, T &eta) {
        // get xi, eta coordinates of each node point
        xi = -1.0 + (2.0 / (order - 1)) * (n % order);
        eta = -1.0 + (2.0 / (order - 1)) * (n / order);
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

    // __HOST_DEVICE__ static void ShellComputeNodeNormalFast(const T &xi, const T &eta,
    //                                                        const T xpts[], T n0[]) {
    //     /* exploring a faster GPU method to get the node normal */

    //     for (int i = 0; i < 3; i++) {
    //         int j = (i + 1) % 3;
    //         int k = (i + 2) % 3;
    //         // cycles through so no sign change needed
    //         // X,xi * Y,eta
    //         T val1 = interpFieldsGradFast<XI, 3>(j, xi, eta, xpts);
    //         T val2 = interpFieldsGradFast<ETA, 3>(k, xi, eta, xpts);
    //         n0[i] = val1 * val2;

    //         // X,eta * Y,xi
    //         val1 = interpFieldsGradFast<ETA, 3>(j, xi, eta, xpts);
    //         val2 = interpFieldsGradFast<XI, 3>(k, xi, eta, xpts);
    //         n0[i] -= val1 * val2;
    //     }

    //     // normalize the unit vector
    //     T norm = sqrt(A2D::VecDotCore<T, 3>(n0, n0));
    //     for (int i = 0; i < 3; i++) n0[i] /= 1.0;
    // }

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

    // template <int icomp>
    // __HOST_DEVICE__ static T getTyingInterpLight(const T pt[], const T ety[]) {
    //     // get 1d knot vectors
    //     // T red_knots[(order-1)], full_knots[order];
    //     // getTyingKnots(red_knots, full_knots);

    //     T dot = 0.0;

    //     if constexpr (icomp == 0) {
    //         // g11
    //         for (int j = 0; j < order; j++) {
    //             for (int i = 0; i < order - 1; i++, ety++) {
    //                 // nar[i] * nb[j]
    //                 dot += lagrangeLobatto1D_tyingLight<order - 1>(i, pt[0]) *
    //                        lagrangeLobatto1D_tyingLight<order>(j, pt[1]) * ety[0];
    //             }
    //         }
    //     } else if constexpr (icomp == 2) {
    //         // g12
    //         for (int j = 0; j < order - 1; j++) {
    //             for (int i = 0; i < order - 1; i++, ety++) {
    //                 // nar[i] * nbr[j]
    //                 dot += lagrangeLobatto1D_tyingLight<order - 1>(i, pt[0]) *
    //                        lagrangeLobatto1D_tyingLight<order - 1>(j, pt[1]) * ety[0];
    //             }
    //         }
    //     } else if constexpr (icomp == 4) {
    //         // g13
    //         for (int j = 0; j < order; j++) {
    //             for (int i = 0; i < order - 1; i++, ety++) {
    //                 // nar[i] * nb[j]
    //                 dot += lagrangeLobatto1D_tyingLight<order - 1>(i, pt[0]) *
    //                        lagrangeLobatto1D_tyingLight<order>(j, pt[1]) * ety[0];
    //             }
    //         }
    //     } else if constexpr (icomp == 1) {
    //         // g22
    //         for (int j = 0; j < order - 1; j++) {
    //             for (int i = 0; i < order; i++, ety++) {
    //                 // na[i] * nbr[j]
    //                 dot += lagrangeLobatto1D_tyingLight<order>(i, pt[0]) *
    //                        lagrangeLobatto1D_tyingLight<order - 1>(j, pt[1]) * ety[0];
    //             }
    //         }
    //     } else if constexpr (icomp == 3) {
    //         // g23
    //         for (int j = 0; j < order - 1; j++) {
    //             for (int i = 0; i < order; i++, ety++) {
    //                 // na[i] * nbr[j]
    //                 dot += lagrangeLobatto1D_tyingLight<order>(i, pt[0]) *
    //                        lagrangeLobatto1D_tyingLight<order - 1>(j, pt[1]) * ety[0];
    //             }
    //         }
    //     }
    //     return dot;
    // }  // end of getTyingInterp

    // template <int icomp>
    // __HOST_DEVICE__ static void addTyingInterpTransposeLight(const T pt[], const T out_bar,
    //                                                          T ety_bar[]) {
    //     if constexpr (icomp == 0) {
    //         // g11
    //         for (int j = 0; j < order; j++) {
    //             for (int i = 0; i < order - 1; i++, ety_bar++) {
    //                 // nar[i] * nb[j]
    //                 ety_bar[0] += lagrangeLobatto1D_tyingLight<order - 1>(i, pt[0]) *
    //                               lagrangeLobatto1D_tyingLight<order>(j, pt[1]) * out_bar;
    //             }
    //         }
    //     } else if constexpr (icomp == 2) {
    //         // g12
    //         for (int j = 0; j < order - 1; j++) {
    //             for (int i = 0; i < order - 1; i++, ety_bar++) {
    //                 // nar[i] * nbr[j]
    //                 ety_bar[0] += lagrangeLobatto1D_tyingLight<order - 1>(i, pt[0]) *
    //                               lagrangeLobatto1D_tyingLight<order - 1>(j, pt[1]) * out_bar;
    //             }
    //         }
    //     } else if constexpr (icomp == 4) {
    //         // g13
    //         for (int j = 0; j < order; j++) {
    //             for (int i = 0; i < order - 1; i++, ety_bar++) {
    //                 // nar[i] * nb[j]
    //                 ety_bar[0] += lagrangeLobatto1D_tyingLight<order - 1>(i, pt[0]) *
    //                               lagrangeLobatto1D_tyingLight<order>(j, pt[1]) * out_bar;
    //             }
    //         }
    //     } else if constexpr (icomp == 1) {
    //         // g22
    //         for (int j = 0; j < order - 1; j++) {
    //             for (int i = 0; i < order; i++, ety_bar++) {
    //                 // na[i] * nbr[j]
    //                 ety_bar[0] += lagrangeLobatto1D_tyingLight<order>(i, pt[0]) *
    //                               lagrangeLobatto1D_tyingLight<order - 1>(j, pt[1]) * out_bar;
    //             }
    //         }
    //     } else if constexpr (icomp == 3) {
    //         // g23
    //         for (int j = 0; j < order - 1; j++) {
    //             for (int i = 0; i < order; i++, ety_bar++) {
    //                 // na[i] * nbr[j]
    //                 ety_bar[0] += lagrangeLobatto1D_tyingLight<order>(i, pt[0]) *
    //                               lagrangeLobatto1D_tyingLight<order - 1>(j, pt[1]) * out_bar;
    //             }
    //         }
    //     }
    // }  // end of addTyingInterpTransposeLight
};

template <typename T, class Basis>
__HOST_DEVICE__ void ShellComputeNodeNormals(const T Xpts[], T fn[]) {
    // the nodal normal vectors are used for director methods
    // fn is 3*num_nodes each node normals
    // Xdn is list of shell frames
    for (int inode = 0; inode < Basis::num_nodes; inode++) {
        T pt[2];
        Basis::getNodePoint(inode, pt);

        // compute the computational coord gradients of Xpts for xi, eta
        T dXdxi[3], dXdeta[3];
        Basis::template interpFieldsGrad<3, 3>(pt, Xpts, dXdxi, dXdeta);

        // call helper function to get node normal at single node
        Basis::ShellComputeNodeNormal(pt, dXdxi, dXdeta, &fn[3 * inode]);
    }
}

template <typename T, class Basis>
__HOST_DEVICE__ static void interpTyingStrain(const T pt[], const T ety[], T gty[]) {
    // given quadpt pt[] and ety[] the tying strains at each tying point from MITC
    // in order {g11-n1, g11-n2, ..., g11-nN, g12-n1, g12-n2,...}
    // interp the final tying strain {g11, g12, g13, g22, g23} at this point with
    // g33 = 0 also
    int32_t offset;

    // probably can add a few scope blocks to make more efficient on the GPU

    // get g11 tying strain
    // ------------------------------------
    {
        offset = Basis::tying_point_offsets(0);
        constexpr int num_tying_g11 = Basis::num_tying_points(0);
        T N_g11[num_tying_g11];  // TODO : can we store less floats here?
        Basis::template getTyingInterp<0>(pt, N_g11);
        gty[0] = A2D::VecDotCore<T, num_tying_g11>(N_g11, &ety[offset]);
    }

    // get g22 tying strain
    // ------------------------------------
    {
        offset = Basis::tying_point_offsets(1);
        constexpr int num_tying_g22 = Basis::num_tying_points(1);
        T N_g22[num_tying_g22];
        Basis::template getTyingInterp<1>(pt, N_g22);
        gty[3] = A2D::VecDotCore<T, num_tying_g22>(N_g22, &ety[offset]);
    }

    // get g12 tying strain
    // ------------------------------------
    {
        offset = Basis::tying_point_offsets(2);
        constexpr int num_tying_g12 = Basis::num_tying_points(2);
        T N_g12[num_tying_g12];
        Basis::template getTyingInterp<2>(pt, N_g12);
        gty[1] = A2D::VecDotCore<T, num_tying_g12>(N_g12, &ety[offset]);
    }

    // get g23 tying strain
    // ------------------------------------
    {
        offset = Basis::tying_point_offsets(3);
        constexpr int num_tying_g23 = Basis::num_tying_points(3);
        T N_g23[num_tying_g23];
        Basis::template getTyingInterp<3>(pt, N_g23);
        gty[4] = A2D::VecDotCore<T, num_tying_g23>(N_g23, &ety[offset]);
    }

    // get g13 tying strain
    // ------------------------------------
    {
        offset = Basis::tying_point_offsets(4);
        constexpr int num_tying_g13 = Basis::num_tying_points(4);
        T N_g13[num_tying_g13];
        Basis::template getTyingInterp<4>(pt, N_g13);
        gty[2] = A2D::VecDotCore<T, num_tying_g13>(N_g13, &ety[offset]);
    }

    // get g33 tying strain
    // --------------------
    gty[5] = 0.0;
}

template <typename T, class Basis>
__HOST_DEVICE__ static void interpTyingStrainLight(const T pt[], const T ety[], T gty[]) {
    // given quadpt pt[] and ety[] the tying strains at each tying point from MITC
    // in order {g11-n1, g11-n2, ..., g11-nN, g12-n1, g12-n2,...}
    // interp the final tying strain {g11, g12, g13, g22, g23} at this point with
    // g33 = 0 also
    int32_t offset;

    // g11 tying strain
    offset = Basis::tying_point_offsets(0);
    gty[0] = Basis::template getTyingInterpLight<0>(pt, &ety[offset]);

    // g22 tying strain
    offset = Basis::tying_point_offsets(1);
    gty[3] = Basis::template getTyingInterpLight<1>(pt, &ety[offset]);

    // g12 tying strain
    offset = Basis::tying_point_offsets(2);
    gty[1] = Basis::template getTyingInterpLight<2>(pt, &ety[offset]);

    // g23 tying strain
    offset = Basis::tying_point_offsets(3);
    gty[4] = Basis::template getTyingInterpLight<3>(pt, &ety[offset]);

    // g13 tying strain
    offset = Basis::tying_point_offsets(4);
    gty[2] = Basis::template getTyingInterpLight<4>(pt, &ety[offset]);

    // get g33 tying strain
    gty[5] = 0.0;
}

template <typename T, class Basis>
__HOST_DEVICE__ static void interpTyingStrainTranspose(const T pt[], const T gty_bar[],
                                                       T ety_bar[]) {
    // given quadpt pt[] and ety[] the tying strains at each tying point from MITC
    // in order {g11-n1, g11-n2, ..., g11-nN, g22-n1, g22-n2,...}
    // interp the final tying strain {g11, g22, g12, g23, g13} in ety_bar storage to
    // with symMat storage also
    int32_t offset;

    // get g11 tying strain
    // ------------------------------------
    {
        offset = Basis::tying_point_offsets(0);
        constexpr int num_tying_g11 = Basis::num_tying_points(0);
        T N_g11[num_tying_g11];  // TODO : can we store less floats here?
        Basis::template getTyingInterp<0>(pt, N_g11);
        A2D::VecAddCore<T, num_tying_g11>(gty_bar[0], N_g11, &ety_bar[offset]);
    }

    // get g22 tying strain
    // ------------------------------------
    {
        offset = Basis::tying_point_offsets(1);
        constexpr int num_tying_g22 = Basis::num_tying_points(1);
        T N_g22[num_tying_g22];
        Basis::template getTyingInterp<1>(pt, N_g22);
        A2D::VecAddCore<T, num_tying_g22>(gty_bar[3], N_g22, &ety_bar[offset]);
    }

    // get g12 tying strain
    // ------------------------------------
    {
        offset = Basis::tying_point_offsets(2);
        constexpr int num_tying_g12 = Basis::num_tying_points(2);
        T N_g12[num_tying_g12];
        Basis::template getTyingInterp<2>(pt, N_g12);
        A2D::VecAddCore<T, num_tying_g12>(gty_bar[1], N_g12, &ety_bar[offset]);
    }

    // get g23 tying strain
    // ------------------------------------
    {
        offset = Basis::tying_point_offsets(3);
        constexpr int num_tying_g23 = Basis::num_tying_points(3);
        T N_g23[num_tying_g23];
        Basis::template getTyingInterp<3>(pt, N_g23);
        A2D::VecAddCore<T, num_tying_g23>(gty_bar[4], N_g23, &ety_bar[offset]);
    }

    // get g13 tying strain
    // ------------------------------------
    {
        offset = Basis::tying_point_offsets(4);
        constexpr int num_tying_g13 = Basis::num_tying_points(4);
        T N_g13[num_tying_g13];
        Basis::template getTyingInterp<4>(pt, N_g13);
        A2D::VecAddCore<T, num_tying_g13>(gty_bar[2], N_g13, &ety_bar[offset]);
    }

    // get g33 tying strain
    // --------------------
    // zero so do nothing
}

template <typename T, class Basis>
__HOST_DEVICE__ static void addInterpTyingStrainTransposeLight(const T pt[], const T gty_bar[],
                                                               T ety_bar[]) {
    // given quadpt pt[] and ety[] the tying strains at each tying point from MITC
    // in order {g11-n1, g11-n2, ..., g11-nN, g22-n1, g22-n2,...}
    // interp the final tying strain {g11, g22, g12, g23, g13} in ety_bar storage to
    // with symMat storage also
    // int32_t offset;
    // constexpr int order = Basis::order;

    // g11 super actually fast ?
    // offset = Basis::tying_point_offsets(0);
    // offset = 0;
    // ety_bar[0] = 0.5; // fine, 32 registers, 4.8e-4 seconds
    // ety_bar[0] = 0.5 * (1.0 - pt[1]); // fine, 32 registers, 4.9e-4 sec
    // so multiply by gty_bar[0] breaks it
    // ety_bar[0] = 0.5 * (1.0 - pt[1]) * gty_bar[0]; // bad (136 registers, 3.6e-3 sec)
    ety_bar[0] = gty_bar[0];  // this alone breaks the registers
    // ety_bar[offset+1] = 0.5 * (1.0 + pt[1]) * gty_bar[0];
    return;

    // g11 fast (pragma unroll here?)
    // offset = Basis::tying_point_offsets(0);
    // #pragma unroll
    // for (int j = 0; j < order; j++) {
    // #pragma unroll
    //    for (int i = 0; i < order - 1; i++, offset++) {
    //        ety_bar[offset] += 1.0 * 0.5 * (1.0 + (-1.0 + 2.0 * j) * pt[1]) * gty_bar[0];
    //    }
    // }

    // g11
    // offset = Basis::tying_point_offsets(0);
    // #pragma unroll
    // for (int j = 0; j < order; j++) {
    //     T nbj = Basis::lagrangeLobatto1D_tyingLight<order>(j, pt[1]);
    //     #pragma unroll
    //     for (int i = 0 ; i < order - 1; i++, ety_bar++) {
    //         ety_bar[0] += Basis::lagrangeLobatto1D_tyingLight<order-1>(i, pt[0]) * nbj *
    //         gty_bar[0];
    //     }
    // }

    // // g12
    // offset = Basis::tying_point_offsets(1);
    // #pragma unroll
    // for (int j = 0; j < order - 1; j++) {
    //     T nbj = Basis::lagrangeLobatto1D_tyingLight<order-1>(j, pt[1]);
    //     #pragma unroll
    //     for (int i = 0 ; i < order - 1; i++, ety_bar++) {
    //         ety_bar[0] += Basis::lagrangeLobatto1D_tyingLight<order-1>(i, pt[0]) * nbj *
    //         gty_bar[3];
    //     }
    // }

    // // g13
    // offset = Basis::tying_point_offsets(2);
    // #pragma unroll
    // for (int j = 0; j < order; j++) {
    //     T nbj = Basis::lagrangeLobatto1D_tyingLight<order>(j, pt[1]);
    //     #pragma unroll
    //     for (int i = 0 ; i < order - 1; i++, ety_bar++) {
    //         ety_bar[0] += Basis::lagrangeLobatto1D_tyingLight<order-1>(i, pt[0]) * nbj *
    //         gty_bar[1];
    //     }
    // }

    // // g22
    // offset = Basis::tying_point_offsets(3);
    // #pragma unroll
    // for (int j = 0; j < order - 1; j++) {
    //     T nbj = Basis::lagrangeLobatto1D_tyingLight<order-1>(j, pt[1]);
    //     #pragma unroll
    //     for (int i = 0 ; i < order - 1; i++, ety_bar++) {
    //         ety_bar[0] += Basis::lagrangeLobatto1D_tyingLight<order>(i, pt[0]) * nbj *
    //         gty_bar[4];
    //     }
    // }

    // // g23
    // offset = Basis::tying_point_offsets(4);
    // #pragma unroll
    // for (int j = 0; j < order - 1; j++) {
    //     T nbj = Basis::lagrangeLobatto1D_tyingLight<order-1>(j, pt[1]);
    //     #pragma unroll
    //     for (int i = 0 ; i < order; i++, ety_bar++) {
    //         ety_bar[0] += Basis::lagrangeLobatto1D_tyingLight<order>(i, pt[0]) * nbj *
    //         gty_bar[2];
    //     }
    // }

    // really slow register version
    // // g11 tying strain
    // offset = Basis::tying_point_offsets(0);
    // Basis::template addTyingInterpTransposeLight<0>(pt, gty_bar[0], &ety_bar[offset]);

    // // g22 tying strain
    // offset = Basis::tying_point_offsets(1);
    // Basis::template addTyingInterpTransposeLight<1>(pt, gty_bar[3], &ety_bar[offset]);

    // // g12 tying strain
    // offset = Basis::tying_point_offsets(2);
    // Basis::template addTyingInterpTransposeLight<2>(pt, gty_bar[1], &ety_bar[offset]);

    // // g23 tying strain
    // offset = Basis::tying_point_offsets(3);
    // Basis::template addTyingInterpTransposeLight<3>(pt, gty_bar[4], &ety_bar[offset]);

    // // g13 tying strain
    // offset = Basis::tying_point_offsets(4);
    // Basis::template addTyingInterpTransposeLight<4>(pt, gty_bar[2], &ety_bar[offset]);

    // get g33 tying strain
    // --------------------
    // zero so do nothing
}

template <typename T, int vars_per_node, class Basis, class Data>
__HOST_DEVICE__ T ShellComputeDispGrad(const T pt[], const T refAxis[], const T xpts[],
                                       const T vars[], const T fn[], const T d[], const T ety[],
                                       T u0x[], T u1x[], A2D::SymMat<T, 3> &e0ty) {
    // Xd, Xdz frame assembly scope
    A2D::Mat<T, 3, 3> Tmat;
    T Xd[9], Xdz[9];
    {
        // interpolation of normals and xpts for disp grads
        T Xxi[3], Xeta[3], nxi[3], neta[3], n0[3];
        Basis::template interpFields<3, 3>(pt, fn, n0);
        Basis::template interpFieldsGrad<3, 3>(pt, xpts, Xxi, Xeta);
        Basis::template interpFieldsGrad<3, 3>(pt, fn, nxi, neta);

        // assemble frames dX/dxi in comp coord
        A2D::Vec<T, 3> zero;
        Basis::assembleFrame(Xxi, Xeta, n0, Xd);
        Basis::assembleFrame(nxi, neta, zero.get_data(), Xdz);

        // compute shell trasnform
        ShellComputeTransform<T, Data>(refAxis, Xxi, Xeta, n0, Tmat.get_data());
    }  // Xd, Xdz frame assembly scope

    {  // u0x, u1x frame assembly scope
        // interp directors
        T d0[3], d0xi[3], d0eta[3];
        Basis::template interpFields<3, 3>(pt, d, d0);
        Basis::template interpFieldsGrad<3, 3>(pt, d, d0xi, d0eta);

        // interp midplane displacements
        T u0xi[3], u0eta[3];
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, u0xi, u0eta);

        // assemble the frames for u0x, u1x in computational space first
        // then we transfer it to
        A2D::Vec<T, 3> zero;
        Basis::assembleFrame(u0xi, u0eta, d0, u0x);
        Basis::assembleFrame(d0xi, d0eta, zero.get_data(), u1x);
    }  // end of u0x, u1x frame assembly scope

    // u0x, u1x conversion to physical space scope
    T XdinvT[9], detXd;
    {
        // invert the Xd transformation
        T Xdinv[9], tmp[9];
        A2D::MatInvCore<T, 3>(Xd, Xdinv);
        detXd = A2D::MatDetCore<T, 3>(Xd);

        // compute XdinvT = Xdinv*T
        A2D::MatMatMultCore3x3<T>(Xdinv, Tmat.get_data(), XdinvT);

        // compute XdinvzT = -Xdinv*Xdz*Xdinv*T
        T XdinvzT[9];
        A2D::MatMatMultCore3x3Scale<T>(-1.0, Xdinv, Xdz, tmp);
        A2D::MatMatMultCore3x3<T>(tmp, XdinvT, XdinvzT);

        // compute u1x = T^{T}*u1d*XdinvT + T^{T}*u0d*XdinvzT
        A2D::MatMatMultCore3x3<T>(u1x, XdinvT, tmp);
        A2D::MatMatMultCore3x3Add<T>(u0x, XdinvzT, tmp);
        A2D::MatMatMultCore3x3<T, A2D::MatOp::TRANSPOSE>(Tmat.get_data(), tmp, u1x);

        // compute u0x = T^{T}*u0d*Xdinv*T
        A2D::MatMatMultCore3x3<T>(u0x, XdinvT, tmp);
        A2D::MatMatMultCore3x3<T, A2D::MatOp::TRANSPOSE>(Tmat.get_data(), tmp, u0x);
    }  // end of u0x, u1x conversion to physical space scope

    {  // interp and rotate tying strains scope
        A2D::SymMat<T, 3> gty;
        interpTyingStrain<T, Basis>(pt, ety, gty.get_data());

        A2D::SymMatRotateFrame<T, 3>(XdinvT, gty, e0ty);
    }  // end of interp and rotate tying strains scope

    return detXd;
}

template <typename T, int vars_per_node, class Basis, class Data>
__HOST_DEVICE__ void ShellComputeDispGradHfwd(const T pt[], const T refAxis[], const T xpts[],
                                              const T p_vars[], const T fn[], const T p_d[],
                                              const T p_ety[], T p_u0x[], T p_u1x[],
                                              A2D::SymMat<T, 3> &p_e0ty) {
    // this is purely linear function, so hforward equiv to forward analysis
    ShellComputeDispGrad<T, vars_per_node, Basis, Data>(pt, refAxis, xpts, p_vars, fn, p_d, p_ety,
                                                        p_u0x, p_u1x, p_e0ty);
}

template <typename T, int vars_per_node, class Basis, class Data>
__HOST_DEVICE__ void ShellComputeDispGradSens(const T pt[], const T refAxis[], const T xpts[],
                                              const T vars[], const T fn[], const T u0x_bar[],
                                              const T u1x_bar[], A2D::SymMat<T, 3> &e0ty_bar,
                                              T res[], T d_bar[], T ety_bar[]) {
    // define some custom matrix multiplies
    constexpr A2D::MatOp NORM = A2D::MatOp::NORMAL;
    constexpr A2D::MatOp TRANS = A2D::MatOp::TRANSPOSE;

    // Xd, Xdz frame assembly scope
    A2D::Mat<T, 3, 3> Tmat;
    T Xd[9], Xdz[9];
    {
        // interpolation of normals and xpts for disp grads
        T Xxi[3], Xeta[3], nxi[3], neta[3], n0[3];
        Basis::template interpFields<3, 3>(pt, fn, n0);
        Basis::template interpFieldsGrad<3, 3>(pt, xpts, Xxi, Xeta);
        Basis::template interpFieldsGrad<3, 3>(pt, fn, nxi, neta);

        // assemble frames dX/dxi in comp coord
        A2D::Vec<T, 3> zero;
        Basis::assembleFrame(Xxi, Xeta, n0, Xd);
        Basis::assembleFrame(nxi, neta, zero.get_data(), Xdz);

        // compute shell trasnform
        ShellComputeTransform<T, Data>(refAxis, Xxi, Xeta, n0, Tmat.get_data());
    }  // Xd, Xdz frame assembly scope

    T XdinvT[9];
    {  // scope block for backprop of u0x_bar, u1x_bar to res
        // invert the Xd transformation
        T Xdinv[9], tmp[9];
        A2D::MatInvCore<T, 3>(Xd, Xdinv);

        // compute XdinvT = Xdinv*T
        A2D::MatMatMultCore3x3<T>(Xdinv, Tmat.get_data(), XdinvT);

        // compute XdinvzT = -Xdinv*Xdz*Xdinv*T
        T XdinvzT[9];
        A2D::MatMatMultCore3x3Scale<T>(-1.0, Xdinv, Xdz, tmp);
        A2D::MatMatMultCore3x3<T>(tmp, XdinvT, XdinvzT);

        // scope for u0d_barT
        {
            T u0d_barT[9];

            // u0d_bar^t = XdinvT * u0x_bar^t * T^t
            A2D::MatMatMultCore3x3<T, NORM, TRANS>(XdinvT, u0x_bar, tmp);
            A2D::MatMatMultCore3x3<T, NORM, TRANS>(tmp, Tmat.get_data(), u0d_barT);

            // u0d_bar^t += XdinvzT * u1x_bar^t * T^t
            // think this should be add here..
            A2D::MatMatMultCore3x3<T, NORM, TRANS>(XdinvzT, u1x_bar, tmp);
            A2D::MatMatMultCore3x3Add<T, NORM, TRANS>(tmp, Tmat.get_data(), u0d_barT);

            // transfer back to u0xi, u0eta, d0 bar (with transpose so columns now available in
            // rows)
            Basis::template interpFieldsTranspose<3, 3>(pt, &u0d_barT[6], d_bar);
            Basis::template interpFieldsGradTranspose<vars_per_node, 3>(pt, &u0d_barT[0],
                                                                        &u0d_barT[3], res);

        }  // end of u0d_barT scope

        // scope for u1d_barT
        {
            T u1d_barT[9];

            // u1d_barT^t = XdinvT * u1x_bar^t * T^t
            A2D::MatMatMultCore3x3<T, NORM, TRANS>(XdinvT, u1x_bar, tmp);
            A2D::MatMatMultCore3x3<T, NORM, TRANS>(tmp, Tmat.get_data(), u1d_barT);

            // prev : cols of u1d are {d0xi, d0eta, zero} => extract from rows of u1d_bar^T => d_bar
            Basis::template interpFieldsGradTranspose<3, 3>(pt, &u1d_barT[0], &u1d_barT[3], d_bar);
        }  // end of u0d_barT scope
    }

    // backprop interp tying strain sens scope block
    {
        A2D::SymMat<T, 3> gty_bar;
        A2D::SymMat3x3RotateFrameReverse<T>(XdinvT, e0ty_bar.get_data(), gty_bar.get_data());
        interpTyingStrainTranspose<T, Basis>(pt, gty_bar.get_data(), ety_bar);
    }  // end of interp tying strain sens scope
}  // ShellComputeDispGradSens

template <typename T, int vars_per_node, class Basis, class Data>
__HOST_DEVICE__ void ShellComputeDispGradHrev(const T pt[], const T refAxis[], const T xpts[],
                                              const T vars[], const T fn[], const T h_u0x[],
                                              const T h_u1x[], A2D::SymMat<T, 3> &h_e0ty,
                                              T matCol[], T h_d[], T h_ety[]) {
    // this is purely linear function, so hreverse equivalent to 1st order reverse (aka Sens
    // function)
    ShellComputeDispGradSens<T, vars_per_node, Basis, Data>(pt, refAxis, xpts, vars, fn, h_u0x,
                                                            h_u1x, h_e0ty, matCol, h_d, h_ety);
}

// function declarations (for ease of use)
// -------------------------------------------------------

/**
__HOST_DEVICE__ static constexpr int32_t num_tying_points(int icomp);
__HOST_DEVICE__ static constexpr int32_t tying_point_offsets(int icomp);
class LinearQuadGeo;
__HOST_DEVICE__ static void lagrangeLobatto1D(const T u, T *N);

template <int tyingOrder>
__HOST_DEVICE__ static void lagrangeLobatto1D_tying(const T u, T *N);

__HOST_DEVICE__ static void lagrangeLobatto1DGrad(const T u, T *N, T *Nd);
__HOST_DEVICE__ static void lagrangeLobatto2D(const T xi, const T eta, T *N);
__HOST_DEVICE__ static void lagrangeLobatto2DGrad(const T xi, const T eta, T *N, T *dNdxi,
                                                  T *dNdeta);
__HOST_DEVICE__ static void assembleFrame(const T a[], const T b[], const T c[], T frame[]);
__HOST_DEVICE__ static void extractFrame(const T frame[], T a[], T b[], T c[]);
__HOST_DEVICE__ static void getNodePoint(const int n, T pt[]);

template <int vars_per_node, int num_fields>
__HOST_DEVICE__ static void interpFields(const T pt[], const T values[], T field[]);

template <int vars_per_node, int num_fields>
__HOST_DEVICE__ static void interpFieldsGrad(const T pt[], const T values[], T dxi[], T deta[]);

template <int vars_per_node, int num_fields>
__HOST_DEVICE__ static void interpFieldsTranspose(const T pt[], const T field_bar[],
                                                  T values_bar[]);

template <int vars_per_node, int num_fields>
__HOST_DEVICE__ static void interpFieldsGradTranspose(const T pt[], const T dxi_bar[],
                                                      const T deta_bar[], T values_bar[]);

__HOST_DEVICE__ static void getTyingKnots(T red_knots[], T full_knots[]);

template <int icomp>
__HOST_DEVICE__ static void getTyingPoint(const int n, T pt[]);
template <int icomp>
__HOST_DEVICE__ static void getTyingInterp(const T pt[], T N[]);

 * defined below / outside the class
 *
 * template <typename T, class Basis>
 * __HOST_DEVICE__ static void interpTyingStrain(const T pt[], const T ety[], T gty[]);
 *
 * template <typename T, class Basis>
 * __HOST_DEVICE__ static void interpTyingStrain(const T pt[], const T ety[], T gty[])
 *
 * template <typename T, class Basis>
 * __HOST_DEVICE__ static void interpTyingStrainTranspose(const T pt[], const T gty_bar[],
                                                   T ety_bar[])
 *
 * template <typename T, int vars_per_node, class Basis, class Data>
 * __HOST_DEVICE__ T ShellComputeDispGrad(const T pt[], const T refAxis[], const T xpts[],
 *            const T vars[], const T fn[], const T d[], const T ety[],
 *            T u0x[], T u1x[], A2D::SymMat<T, 3> &e0ty)
 *
 * template <typename T, int vars_per_node, class Basis, class Data>
 * __HOST_DEVICE__ void ShellComputeDispGradHfwd(const T pt[], const T refAxis[], const T
 * xpts[],
 *                                          const T p_vars[], const T fn[], const T p_d[],
 *                                        const T p_ety[], T p_u0x[], T p_u1x[],
 *                                        A2D::SymMat<T, 3> &p_e0ty)
 *
 *
 * template <typename T, int vars_per_node, class Basis, class Data>
 * __HOST_DEVICE__ void ShellComputeDispGradSens(const T pt[], const T refAxis[], const T
 * xpts[],
 *      const T vars[], const T fn[], const T u0x_bar[],
 *                           const T u1x_bar[], A2D::SymMat<T, 3> &e0ty_bar,
 *                           T res[], T d_bar[], T ety_bar[])
 *
 * template <typename T, int vars_per_node, class Basis, class Data>
 * __HOST_DEVICE__ void ShellComputeDispGradHrev(const T pt[], const T refAxis[], const T
 * xpts[],
 *      const T vars[], const T fn[], const T h_u0x[],
 *                                        const T h_u1x[], A2D::SymMat<T, 3> &h_e0ty,
 *                                        T matCol[], T h_d[], T h_ety[])
 *
 */

// -------------------------------------------------------
// end of function declarations