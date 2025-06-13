#pragma once

#include "cuda_utils.h"

template <typename T, int offset_ = 3>
class LinearizedRotationV1 {
   public:
    static const int32_t offset = offset_;
    static const int32_t num_params = 3;

    // TODO fix so in base class only
    __HOST_DEVICE__ static void setMatSkew(const T a, const T b[], T C[]) {
        C[0] = 0.0;
        C[1] = -a * b[2];
        C[2] = a * b[1];

        C[3] = a * b[2];
        C[4] = 0.0;
        C[5] = -a * b[0];

        C[6] = -a * b[1];
        C[7] = a * b[0];
        C[8] = 0.0;
    }

    __HOST_DEVICE__ static void setMatSkewSens(const T a, const T C_bar[], T b_bar[]) {
        b_bar[0] += a * (C_bar[7] - C_bar[5]);
        b_bar[1] += a * (C_bar[2] - C_bar[6]);
        b_bar[2] += a * (C_bar[3] - C_bar[1]);
    }

    template <int vars_per_node, int num_nodes>
    __HOST_DEVICE__ static void computeRotationMat(const T vars[], T C[]) {
        const T *q = &vars[offset];
        for (int inode = 0; inode < num_nodes; inode++) {
            // C = I - q^x
            setMatSkew(-1.0, q, C);
            C[0] = C[4] = C[8] = 1.0;

            C += 9;
            q += vars_per_node;
        }
    }

    template <int vars_per_node, int num_nodes>
    __HOST_DEVICE__ static void computeRotationMatSens(const T C_bar[], T res[]) {
        T *r = &res[offset];
        for (int inode = 0; inode < num_nodes; inode++) {
            // C = I - q^x
            setMatSkewSens(-1.0, C_bar, r);
            C_bar += 9;
            r += vars_per_node;
        }
    }

    __HOST_DEVICE__ static T evalDrillStrain(const T u0x[], const T Ct[]) {
        // compute rotation penalty
        return 0.5 * (Ct[3] + u0x[3] - Ct[1] - u0x[1]);
    }

    __HOST_DEVICE__ static void evalDrillStrainSens(const T &scale, T u0x_bar[], T Ct_bar[]) {
        // compute rotation penalty
        // etn = 0.5 * (Ct[3] + u0x[3] - Ct[1] - u0x[1]);
        for (int i = 0; i < 9; i++) {
            u0x_bar[i] = 0.0;
            Ct_bar[i] = 0.0;
        }
        u0x_bar[1] = -0.5 * scale;
        u0x_bar[3] = 0.5 * scale;
        Ct_bar[1] = -0.5 * scale;
        Ct_bar[3] = 0.5 * scale;
    }

    template <int vars_per_node, int num_nodes>
    __HOST_DEVICE__ static void computeDirector(const T vars[], const T t[], T d[]) {
        const T *q = &vars[offset];
        for (int inode = 0; inode < num_nodes; inode++) {
            A2D::VecCrossCore<T>(q, t, d);
            t += 3;
            d += 3;
            q += vars_per_node;
        }
    }

    template <class Basis, int vars_per_node, int num_nodes>
    __HOST_DEVICE__ static void interpDirectorLight(const T pt[], const T xpts[], const T vars[],
                                                    T d0[]) {
        const T *q = &vars[offset];
#pragma unroll
        for (int i = 0; i < 3; i++) d0[i] = 0.0;

#pragma unroll
        for (int inode = 0; inode < num_nodes; inode++) {
            T n0[3], node_pt[2], d[3];
            Basis::getNodePoint(inode, node_pt);
            Basis::ShellComputeNodeNormal(node_pt, xpts, n0);

            A2D::VecCrossCore<T>(q, n0, d);
            q += vars_per_node;

#pragma unroll
            for (int ifield = 0; ifield < 3; ifield++) {
                d0[ifield] += Basis::lagrangeLobatto2DLight(inode, pt[0], pt[1]) * d[ifield];
            }
        }
    }

    template <class Basis, int vars_per_node, int num_nodes>
    __HOST_DEVICE__ static void interpDirectorLightSens(const T scale, const T pt[], const T xpts[],
                                                        const T d0_bar[], T res[]) {
        T *q_bar = &res[offset];
#pragma unroll
        for (int inode = 0; inode < num_nodes; inode++) {
            T n0[3], node_pt[2], d_bar[3];
            Basis::getNodePoint(inode, node_pt);
            Basis::ShellComputeNodeNormal(node_pt, xpts, n0);

#pragma unroll
            for (int ifield = 0; ifield < 3; ifield++) {
                T jac = Basis::lagrangeLobatto2DLight(inode, pt[0], pt[1]);
                d_bar[ifield] = scale * jac * d0_bar[ifield];
            }

            A2D::VecCrossCoreAdd<T>(n0, d_bar, q_bar);
            q_bar += vars_per_node;
        }
    }

    template <int vars_per_node, int num_nodes>
    __HOST_DEVICE__ static void computeDirectorHfwd(const T p_vars[], const T t[], T p_d[]) {
        // since linear, just call reg forward analysis
        computeDirector<vars_per_node, num_nodes>(p_vars, t, p_d);
    }

    template <int vars_per_node, int num_nodes>
    __HOST_DEVICE__ static void computeDirectorSens(const T t[], const T d_bar[], T res[]) {
        T *q_bar = &res[offset];
        for (int inode = 0; inode < num_nodes; inode++) {
            // easy to show backprop of cross product is also cross product operation
            // if y = x cross n with n constant then x = n cross y
            A2D::VecCrossCoreAdd<T>(t, d_bar, q_bar);
            t += 3;
            d_bar += 3;
            q_bar += vars_per_node;
        }
    }

    template <int vars_per_node, int num_nodes>
    __HOST_DEVICE__ static void computeDirectorHrev(const T t[], const T h_d[], T matCol[]) {
        // since linear, just call reg 1st order reverse
        computeDirectorSens<vars_per_node, num_nodes>(t, h_d, matCol);
    }

};  // end of class LinearizedRotation
