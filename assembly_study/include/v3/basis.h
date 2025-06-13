#pragma once
#include "../v1/_basis_utils.h"
#include "a2dcore.h"
#include "quadrature.h"

// only considering order = 2 here, can generalize later if I want..
template <typename T, class Quadrature_>
class ShellQuadBasisV3 {
   public:
    using Quadrature = Quadrature_;

    // order and number of nodes
    static constexpr int32_t order = 2;
    static constexpr int32_t num_nodes = 4;
    static constexpr int32_t num_tying = 9;

    // isoperimetric has same #nodes geometry class inside it
    class LinearQuadGeo {
       public:
        static constexpr int32_t spatial_dim = 3;
        static constexpr int32_t num_nodes = 4;
        static constexpr int32_t num_quad_pts = Quadrature::num_quad_pts;
    };  // end of class LinearQuadGeo
    using Geo = LinearQuadGeo;

    __HOST_DEVICE__ static void getNodePoint(const int n, T pt[]) {
        // get xi, eta coordinates of each node point
        pt[0] = -1.0 + (2.0 / (order - 1)) * (n % order);
        pt[1] = -1.0 + (2.0 / (order - 1)) * (n / order);
    }

    __HOST_DEVICE__ static T lagrangeLobatto1DLight(const int i, const T &u) {
        // for higher order, could use product formula and a const data of the nodal points in order
        // to get this on the fly (is possible)
        if constexpr (order == 2) {
            return 0.5 * (1.0 + (-1.0 + 2.0 * i) * u);
        }
    }

    __HOST_DEVICE__ static T lagrangeLobatto1DGradLight(const int i, const T &u) {
        if constexpr (order == 2) {
            return 0.5 * (-1.0 + 2.0 * i);
        }
    }

    __HOST_DEVICE__ static T lagrangeLobatto2DLight(const int ind, const T &xi, const T &eta) {
        /* on the fly interp */
        return lagrangeLobatto1DLight(ind % order, xi) * lagrangeLobatto1DLight(ind / order, eta);
    }  // end of lagrangeLobatto2DLight

    template <COMP_VAR deriv>
    __HOST_DEVICE__ static T lagrangeLobatto2DGradLight(const int inode, const T &xi,
                                                        const T &eta) {
        /* deriv == 0 is xi deriv, deriv == 1 is eta deriv */
        if constexpr (deriv == XI) {
            return lagrangeLobatto1DGradLight(inode % order, xi) *
                   lagrangeLobatto1DLight(inode / order, eta);
        } else if (deriv == ETA) {
            return lagrangeLobatto1DLight(inode % order, xi) *
                   lagrangeLobatto1DGradLight(inode / order, eta);
        }
    }  // end of lagrangeLobatto2D_gradLight

    template <int vars_per_node>
    __HOST_DEVICE__ static T interpFieldsLight(const int ifield, const T pt[], const T values[]) {
        /* light version of interpFields that just gets one value only of the output vector */
        // xpts[ifield] = sum_inode N[inode] * values[inode * vars_per_node + ifield] for single
        // ifield

        T out = 0.0;
        for (int inode = 0; inode < num_nodes; inode++) {
            out += lagrangeLobatto2DLight(inode, pt[0], pt[1]) *
                   values[vars_per_node * inode + ifield];
        }
        return out;
    }  // end of interpFieldsLight method

    template <COMP_VAR DERIV, int vars_per_node>
    __HOST_DEVICE__ static T interpFieldsGradLight(const int ifield, const T pt[],
                                                   const T values[]) {
        /* light version of interpFieldsGrad, where ideriv is either 0 or 1 */
        T out = 0.0;
        for (int inode = 0; inode < num_nodes; inode++) {
            out += lagrangeLobatto2DGradLight<DERIV>(inode, pt[0], pt[1]) *
                   values[vars_per_node * inode + ifield];
        }
        return out;
    }  // end of interpFieldsGradLight method

    template <COMP_VAR DERIV, int vars_per_node>
    __HOST_DEVICE__ static T interpFieldsGradLight(const int inode, const int ifield, const T pt[],
                                                   const T values[]) {
        /* light version of interpFieldsGrad, where ideriv is either 0 or 1 */
        return lagrangeLobatto2DGradLight<DERIV>(inode, pt[0], pt[1]) *
               values[vars_per_node * inode + ifield];
    }  // end of interpFieldsGradLight method

    template <COMP_VAR i1, COMP_VAR i2, int vpn1, int vpn2, int nfields>
    __HOST_DEVICE__ static T interpFieldsGradDotLight(const T pt[], const T vec1[],
                                                      const T vec2[]) {
        /* i1, i2 represent xi, eta through 0,1
            then this is dot product <dvec1/di1, dvec2/di2> where i1,i2 represent either xi or eta
           each
        */
        T dot = 0.0;
        for (int ifield = 0; ifield < nfields; ifield++) {
            T dv1_di1 = 0.0, dv2_di2 = 0.0;
            for (int inode = 0; inode < num_nodes; inode++) {
                dv1_di1 += lagrangeLobatto2DGradLight<i1>(inode, pt[0], pt[1]) *
                           vec1[vpn1 * inode + ifield];
                dv2_di2 += lagrangeLobatto2DGradLight<i2>(inode, pt[0], pt[1]) *
                           vec2[vpn2 * inode + ifield];
            }
            dot += dv1_di1 * dv2_di2;
        }
        return dot;
    }  // end of interpFieldsGradDotLight method

    template <COMP_VAR i1, COMP_VAR i2, int vpn1, int vpn2, int nfields>
    __HOST_DEVICE__ static void addInterpFieldsGradDotSensLight(const T scale, const T pt[],
                                                                const T vec1[], T vec2_bar[]) {
        /* i1, i2 represent xi, eta through 0,1
            dot product of forward analysis is <dvec1/di1, dvec2/di2> assume here vec2 is for
           derivative
        */

        T dv1_di1;
        for (int ifield = 0; ifield < nfields; ifield++) {
            dv1_di1 = 0.0;
            for (int inode = 0; inode < num_nodes; inode++) {
                dv1_di1 += lagrangeLobatto2DGradLight<i1>(inode, pt[0], pt[1]) *
                           vec1[vpn1 * inode + ifield];
            }

            for (int inode = 0; inode < num_nodes; inode++) {
                T loc_scale = scale * lagrangeLobatto2DGradLight<i2>(inode, pt[0], pt[1]);
                // return; // 32 registers per thread
                vec2_bar[vpn2 * inode + ifield] += loc_scale * dv1_di1;
            }
        }
    }  // end of interpFieldsGradDotLight method

    template <COMP_VAR i1, int vpn1, int nfields>
    __HOST_DEVICE__ static T interpFieldsGradRightDotLight(const T pt[], const T vec1[],
                                                           const T vec2[]) {
        /* i1, i2 represent xi, eta through 0,1
            then this is dot product <dvec1/di1, vec2> where i1 is comp derivative
        */
        T dot = 0.0;
        for (int ifield = 0; ifield < nfields; ifield++) {
            T dv1_di1 = 0.0;
            for (int inode = 0; inode < num_nodes; inode++) {
                dv1_di1 += lagrangeLobatto2DGradLight<i1>(inode, pt[0], pt[1]) *
                           vec1[vpn1 * inode + ifield];
            }
            dot += dv1_di1 * vec2[ifield];
        }
        return dot;
    }  // end of interpFieldsGradDotLight method

    template <COMP_VAR i1, int vpn1, int nfields>
    __HOST_DEVICE__ static void interpFieldsGradRightDotLight_LeftSens(const T scale, const T pt[],
                                                                       T vec1_bar[],
                                                                       const T vec2[]) {
        /* i1, i2 represent xi, eta through 0,1
            then this is dot product <dvec1/di1, vec2> where i1 is comp derivative
        */
        for (int ifield = 0; ifield < nfields; ifield++) {
            for (int inode = 0; inode < num_nodes; inode++) {
                T jac = vec2[ifield] * lagrangeLobatto2DGradLight<i1>(inode, pt[0], pt[1]);
                vec1_bar[vpn1 * inode + ifield] += scale * jac;
            }
        }
    }  // end of interpFieldsGradDotLight method

    template <COMP_VAR i1, int vpn1, int nfields>
    __HOST_DEVICE__ static void interpFieldsGradRightDotLight_RightSens(const T scale, const T pt[],
                                                                        const T vec1[],
                                                                        T vec2_bar[]) {
        /* i1, i2 represent xi, eta through 0,1
            then this is dot product <dvec1/di1, vec2> where i1 is comp derivative
        */
        for (int ifield = 0; ifield < nfields; ifield++) {
            T dv1_di1 = 0.0;
            for (int inode = 0; inode < num_nodes; inode++) {
                dv1_di1 += lagrangeLobatto2DGradLight<i1>(inode, pt[0], pt[1]) *
                           vec1[vpn1 * inode + ifield];
            }
            vec2_bar[ifield] += dv1_di1 * scale;
        }
    }  // end of interpFieldsGradDotLight method

    template <int vars_per_node, int num_fields>
    __HOST_DEVICE__ static void interpFieldsTransposeLight(const T pt[], const T field_bar[],
                                                           T values_bar[]) {
        /* light version of interpFields that just gets one value only of the output vector */
        // xpts[ifield] = sum_inode N[inode] * values[inode * vars_per_node + ifield] for single
        // ifield

        for (int ifield = 0; ifield < num_fields; ifield++) {
            for (int inode = 0; inode < num_nodes; inode++) {
                values_bar[inode * vars_per_node + ifield] +=
                    field_bar[ifield] * lagrangeLobatto2DLight(inode, pt[0], pt[1]);
            }
        }
    }  // end of interpFieldsLight method

    template <int vars_per_node, int num_fields>
    __HOST_DEVICE__ static void interpFieldsGradTransposeLight(const T pt[], const T dxi_bar[],
                                                               const T deta_bar[], T values_bar[]) {
        for (int ifield = 0; ifield < num_fields; ifield++) {
            for (int inode = 0; inode < num_nodes; inode++) {
                int ind = inode * vars_per_node + ifield;
                values_bar[ind] +=
                    dxi_bar[ifield] * lagrangeLobatto2DGradLight<XI>(inode, pt[0], pt[1]);
                values_bar[ind] +=
                    deta_bar[ifield] * lagrangeLobatto2DGradLight<ETA>(inode, pt[0], pt[1]);
            }
        }
    }  // end of interpFieldsGradTransposeLight method

    __HOST_DEVICE__ static void ShellComputeNodeNormalLight(const T pt[], const T xpts[], T n0[]) {
        // compute the shell node normal at a single node given already the pre-computed spatial
        // gradients
        T Xxi[3], Xeta[3];
        for (int i = 0; i < 3; i++) {
            Xxi[i] = interpFieldsGradLight<XI, 3>(i, pt, xpts);
            Xeta[i] = interpFieldsGradLight<ETA, 3>(i, pt, xpts);
        }

        // compute and normalize X,xi cross X,eta
        T tmp[3];
        A2D::VecCrossCore<T>(Xxi, Xeta, tmp);
        T norm = sqrt(A2D::VecDotCore<T, 3>(tmp, tmp));
        norm = 1.0 / norm;
        A2D::VecScaleCore<T, 3>(norm, tmp, n0);
    }

    __HOST_DEVICE__ static void interpNodeNormalLight(const T quad_pt[], const T xpts[], T n0[]) {
// compute the shell node normal at a single node given already the pre-computed spatial
// gradients
#pragma unroll
        for (int ifield = 0; ifield < 3; ifield++) n0[ifield] = 0.0;

#pragma unroll
        for (int inode = 0; inode < num_nodes; inode++) {
            T fn[3], node_pt[2];
            getNodePoint(inode, node_pt);
            ShellComputeNodeNormalLight(node_pt, xpts, fn);

            for (int ifield = 0; ifield < 3; ifield++) {
                n0[ifield] += lagrangeLobatto2DLight(inode, quad_pt[0], quad_pt[1]) * fn[ifield];
            }
        }
    }
};