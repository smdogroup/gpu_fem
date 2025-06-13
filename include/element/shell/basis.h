#pragma once
#include "_basis_utils.h"
#include "a2dcore.h"
#include "a2dshell.h"
#include "quadrature.h"
#include "shell_utils.h"

// still include slow versions
#include "basis_slow.h"

template <typename T, class Quadrature_, int order_ = 2>
class ShellQuadBasis : public ShellQuadBasisOld<T, Quadrature_, order_> {
   public:
    using Quadrature = Quadrature_;

    // Required for loading solution data
    static constexpr int32_t order = order_;
    static constexpr int32_t num_nodes = order * order;
    static constexpr int32_t param_dim = 2;

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

    __HOST_DEVICE__ static T lagrangeLobatto1DLight(const int i, const T &u) {
        // for higher order, could use product formula and a const data of the nodal points in order
        // to get this on the fly (is possible)
        if constexpr (order == 2) {
            return 0.5 * (1.0 + (-1.0 + 2.0 * i) * u);
        }
    }

    template <int tyingOrder>
    __HOST_DEVICE__ static T lagrangeLobatto1D_tyingLight(int i, const T &u) {
        if constexpr (tyingOrder == 1) {
            return 1.0;
        } else if constexpr (tyingOrder == 2) {
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
    __HOST_DEVICE__ static void assembleFrameLight(const T pt[], const T values[], const T normal[],
                                                   T frame[]) {
        /* light implementation of assembleFrame for xpts */

#pragma unroll
        for (int i = 0; i < 3; i++) {
            frame[3 * i] = interpFieldsGradLight<XI, vars_per_node>(i, pt, values);
            frame[3 * i + 1] = interpFieldsGradLight<ETA, vars_per_node>(i, pt, values);
            frame[3 * i + 2] = normal[i];
        }
    }

    template <int vars_per_node>
    __HOST_DEVICE__ static void assembleFrameLightSens(const T pt[], const T frame_bar[],
                                                       T normal_bar[3], T values_bar[]) {
        /* light implementation of assembleFrame for xpts */

#pragma unroll
        for (int i = 0; i < 3; i++) {
#pragma unroll
            for (int inode = 0; inode < num_nodes; inode++) {
                values_bar[vars_per_node * inode + i] +=
                    lagrangeLobatto2DGradLight<XI>(inode, pt[0], pt[1]) * frame_bar[3 * i];
                values_bar[vars_per_node * inode + i] +=
                    lagrangeLobatto2DGradLight<ETA>(inode, pt[0], pt[1]) * frame_bar[3 * i + 1];
            }
            normal_bar[i] = frame_bar[3 * i + 2];
        }
    }

    __HOST_DEVICE__ static void getNodePoint(const int n, T pt[]) {
        // get xi, eta coordinates of each node point
        pt[0] = -1.0 + (2.0 / (order - 1)) * (n % order);
        pt[1] = -1.0 + (2.0 / (order - 1)) * (n / order);
    }

    __HOST_DEVICE__ static void assembleXptFrameLight(const T pt[], const T xpts[], T frame[]) {
        /* light implementation of assembleFrame for xpts */
        T n0[3];
        interpNodeNormalLight(pt, xpts, n0);
#pragma unroll
        for (int i = 0; i < 3; i++) {
            frame[3 * i] = interpFieldsGradLight<XI, 3>(i, pt, xpts);
            frame[3 * i + 1] = interpFieldsGradLight<ETA, 3>(i, pt, xpts);
            frame[3 * i + 2] = n0[i];
        }
    }

    __HOST_DEVICE__ static T getDetXd(const T pt[], const T xpts[]) {
        // get detXd at quadpt
        T Xd[9];
        assembleXptFrameLight(pt, xpts, Xd);

        // now compute det of Xd jacobian
        T detXd = A2D::MatDetCore<T, 3>(Xd);
        return detXd;
    }

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

    template <int vars_per_node>
    __HOST_DEVICE__ static T addInterpFieldsLight(const int inode, const int ifield, const T pt[],
                                                  const T values[]) {
        /* light version of interpFields that just gets one value only of the output vector */
        // xpts[ifield] = sum_inode N[inode] * values[inode * vars_per_node + ifield] for single
        // ifield
        return lagrangeLobatto2DLight(inode, pt[0], pt[1]) * values[vars_per_node * inode + ifield];
    }  // end of interpFieldsLight method

    template <int vars_per_node>
    __HOST_DEVICE__ static T interpFieldsFast(const int ifield, const T &xi, const T &eta,
                                              const T values[]) {
        /* fast version of interpFields that just gets one value only of the output vector */
        T out = 0.0;
        for (int inode = 0; inode < num_nodes; inode++) {
            out += lagrangeLobatto2DLight(inode, xi, eta) * values[vars_per_node * inode + ifield];
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

    template <COMP_VAR DERIV, int vars_per_node>
    __HOST_DEVICE__ static T interpFieldsGradFast(const int ifield, const T &xi, const T &eta,
                                                  const T values[]) {
        /* light version of interpFieldsGrad, where ideriv is either 0 or 1 */
        T out = 0.0;
        for (int inode = 0; inode < num_nodes; inode++) {
            out += lagrangeLobatto2DGradLight<DERIV>(inode, xi, eta) *
                   values[vars_per_node * inode + ifield];
        }
        return out;
    }  // end of interpFieldsGradFasat method

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

    __HOST_DEVICE__ static void ShellComputeNormalFrameLight(const T quad_pt[], const T xpts[],
                                                             T Xdz[]) {
        // compute the shell node normal at a single node given already the pre-computed spatial
        // gradients: assembles the frame (n0,xi; n0,eta; 0-vec)

        T n0[3], node_pt[2];
#pragma unroll
        for (int i = 0; i < 9; i++) Xdz[i] = 0.0;
#pragma unroll
        for (int inode = 0; inode < num_nodes; inode++) {
            // compute the node normal at each basis node
            getNodePoint(inode, node_pt);
            ShellComputeNodeNormalLight(node_pt, xpts, n0);

            // add into Xdz at the pt (which is usually quad_pt)
            for (int idim = 0; idim < 3; idim++) {
                Xdz[3 * idim] +=
                    n0[idim] * lagrangeLobatto2DGradLight<XI>(inode, quad_pt[0], quad_pt[1]);
                Xdz[3 * idim + 1] +=
                    n0[idim] * lagrangeLobatto2DGradLight<ETA>(inode, quad_pt[0], quad_pt[1]);
            }
        }
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

};  // end of class ShellQuadBasis

// Basis related utils

template <typename T, class Data, class Basis>
__HOST_DEVICE__ static T getFrameRotation(const T refAxis[], const T pt[], const T xpts[],
                                          T XdinvT[]) {
    /* get XdinvT the full frame rotation */
    T Tmat[9], n0[3];
    Basis::ShellComputeNodeNormalLight(pt, xpts, n0);

    // assemble Xd frame (here it is Xd)
    Basis::assembleFrameLight<3>(pt, xpts, n0, Tmat);

    // invert the Xd transformation (so Xdinv stored in XdinvT now)
    A2D::MatInvCore<T, 3>(Tmat, XdinvT);
    T detXd = A2D::MatDetCore<T, 3>(XdinvT);

    // compute the shell transform based on the ref axis in Data object
    ShellComputeTransformLight<T, Basis, Data>(refAxis, pt, xpts, n0, Tmat);

    // compute XdinvT = Xdinv*T
    A2D::MatMatMultCore3x3<T>(XdinvT, Tmat, XdinvT);

    return detXd;
}  // end of Xd and shell transform scope
