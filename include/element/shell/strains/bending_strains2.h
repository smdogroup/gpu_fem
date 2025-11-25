#pragma once
#include "../../../cuda_utils.h"

template <typename T, int vars_per_node, class Basis>
__DEVICE__ void computeBendingDispGrad(const T pt[], const T vars[], const T d[], const T Tmat[],
                                       const T XdinvT[], const T XdinvzT[], T u0x[], T u1x[]) {
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
        assembleFrame<T>(u0xi, u0eta, d0, u0x);
        assembleFrame<T>(d0xi, d0eta, zero.get_data(), u1x);
    }  // end of u0x, u1x frame assembly scope
    __syncthreads();

    // u0x, u1x conversion to shell frame
    {
        T tmp[9];

        // compute u1x = T^{T}*u1d*XdinvT + T^{T}*u0d*XdinvzT
        A2D::MatMatMultCore3x3<T>(u1x, XdinvT, tmp);
        A2D::MatMatMultCore3x3Add<T>(u0x, XdinvzT, tmp);
        A2D::MatMatMultCore3x3<T, A2D::MatOp::TRANSPOSE>(Tmat, tmp, u1x);

        // compute u0x = T^{T}*u0d*Xdinv*T
        A2D::MatMatMultCore3x3<T>(u0x, XdinvT, tmp);
        A2D::MatMatMultCore3x3<T, A2D::MatOp::TRANSPOSE>(Tmat, tmp, u0x);
    }  // end of u0x, u1x conversion to conversion to shell frame
}

template <typename T, int vars_per_node, class Basis>
__DEVICE__ void computeBendingDispGradSens(const T pt[], const T Tmat[], const T XdinvT[],
                                           const T XdinvzT[], const T u0xb[], const T u1xb[],
                                           T d_bar[], T res[]) {
    /* since this computation is linear, this can be reused for hrev also */

    constexpr A2D::MatOp NORM = A2D::MatOp::NORMAL;
    constexpr A2D::MatOp TRANS = A2D::MatOp::TRANSPOSE;

    // scope for u0d_barT
    T tmp[9];
    {
        T u0d_barT[9];

        // u0d_bar^t = XdinvT * u0x_bar^t * T^t
        A2D::MatMatMultCore3x3<T, NORM, TRANS>(XdinvT, u0xb, tmp);
        A2D::MatMatMultCore3x3<T, NORM, TRANS>(tmp, Tmat, u0d_barT);

        // u0d_bar^t += XdinvzT * u1x_bar^t * T^t
        A2D::MatMatMultCore3x3<T, NORM, TRANS>(XdinvzT, u1xb, tmp);
        A2D::MatMatMultCore3x3Add<T, NORM, TRANS>(tmp, Tmat, u0d_barT);

        // transfer back to u0xi, u0eta, d0 bar (with transpose so columns now available in
        // rows)
        Basis::template interpFieldsTranspose<3, 3>(pt, &u0d_barT[6], d_bar);
        Basis::template interpFieldsGradTranspose<vars_per_node, 3>(pt, &u0d_barT[0], &u0d_barT[3],
                                                                    res);
    }  // end of u0d_barT scope
    __syncthreads();

    // scope for u1d_barT
    {
        T u1d_barT[9];

        // u1d_barT^t = XdinvT * u1x_bar^t * T^t
        A2D::MatMatMultCore3x3<T, NORM, TRANS>(XdinvT, u1xb, tmp);
        A2D::MatMatMultCore3x3<T, NORM, TRANS>(tmp, Tmat, u1d_barT);

        // prev : cols of u1d are {d0xi, d0eta, zero} => extract from rows of u1d_bar^T => d_bar
        Basis::template interpFieldsGradTranspose<3, 3>(pt, &u1d_barT[0], &u1d_barT[3], d_bar);
    }  // end of u0d_barT scope
}  // ShellComputeDispGradSens

template <typename T, bool is_nonlinear = false>
__DEVICE__ void computeBendingStrain(const T u0x[9], const T u1x[9], T ek[3]) {
    /* just computing the linear part of the strains for now, will add back NL later */
    ek[0] = u1x[0];
    ek[1] = u1x[4];
    ek[2] = u1x[1] + u1x[3];

    if constexpr (is_nonlinear) {
        ek[0] += u0x[0] * u1x[0] + u0x[3] * u1x[3] + u0x[6] * u1x[6];  // k11
        ek[1] += u0x[1] * u1x[1] + u0x[4] * u1x[4] + u0x[7] * u1x[7];  // k22
        ek[2] += u0x[0] * u1x[1] + u0x[3] * u1x[4] + u0x[6] * u1x[7] + u1x[0] * u0x[1] +
                 u1x[3] * u0x[4] + u1x[6] * u0x[7];  // k12
    }
}

template <typename T, bool is_nonlinear = false>
__DEVICE__ void computeBendingStrainHfwd(const T u0x[9], const T u1x[9], const T u0xF[9],
                                         const T u1xF[9], T ekF[3]) {
    /* just computing the linear part of the strains for now, will add back NL later */

    // linear part
    computeBendingStrain<T, false>(u0xF, u1xF, ekF);

    if constexpr (is_nonlinear) {
        ekF[0] += u0xF[0] * u1x[0] + u0x[0] * u1xF[0] + u0xF[3] * u1x[3] + u0x[3] * u1xF[3] +
                  u0xF[6] * u1x[6] + u0x[6] * u1xF[6];  // k11
        ekF[1] += u0xF[1] * u1x[1] + u0x[1] * u1xF[1] + u0xF[4] * u1x[4] + u0x[4] * u1xF[4] +
                  u0xF[7] * u1x[7] + u0x[7] * u1xF[7];  // k22
        ekF[2] += u0xF[0] * u1x[1] + u0x[0] * u1xF[1] + u0xF[3] * u1x[4] + u0x[3] * u1xF[4] +
                  u0xF[6] * u1x[7] + u0x[6] * u1xF[7] + u1xF[0] * u0x[1] + u1x[0] * u0xF[1] +
                  u1xF[3] * u0x[4] + u1x[3] * u0xF[4] + u1xF[6] * u0x[7] + u1x[6] * u0xF[7];  // k12
    }
}

template <typename T, bool is_nonlinear = false>
__DEVICE__ void computeBendingStrainSens(const T ekb[3], const T u0x[9], const T u1x[9], T u0xb[9],
                                         T u1xb[9]) {
    /* just computing the linear part of the strains for now, will add back NL later */
    u1xb[0] += ekb[0];
    u1xb[4] += ekb[1];
    u1xb[1] += ekb[2];
    u1xb[3] += ekb[2];

    if constexpr (is_nonlinear) {
        // k11 computation
        u0xb[0] += u1x[0] * ekb[0];
        u1xb[0] += u0x[0] * ekb[0];
        u0xb[3] += u1x[3] * ekb[0];
        u1xb[3] += u0x[3] * ekb[0];
        u0xb[6] += u1x[6] * ekb[0];
        u1xb[6] += u0x[6] * ekb[0];
        // k22 computation
        u0xb[1] += u1x[1] * ekb[1];
        u1xb[1] += u0x[1] * ekb[1];
        u0xb[4] += u1x[4] * ekb[1];
        u1xb[4] += u0x[4] * ekb[1];
        u0xb[7] += u1x[7] * ekb[1];
        u1xb[7] += u0x[7] * ekb[1];
        // k12 computation
        u0xb[0] += u1x[1] * ekb[2];
        u1xb[0] += u0x[1] * ekb[2];
        u0xb[1] += u1x[0] * ekb[2];
        u1xb[1] += u0x[0] * ekb[2];
        u0xb[3] += u1x[4] * ekb[2];
        u1xb[3] += u0x[4] * ekb[2];
        u0xb[4] += u1x[3] * ekb[2];
        u1xb[4] += u0x[3] * ekb[2];
        u0xb[6] += u1x[7] * ekb[2];
        u1xb[6] += u0x[7] * ekb[2];
        u0xb[7] += u1x[6] * ekb[2];
        u1xb[7] += u0x[6] * ekb[2];
    }
}

template <typename T, bool is_nonlinear = false>
__DEVICE__ void computeBendingStrainHrev(const T ekh[3], const T ekb[3], const T u0x[9],
                                         const T u1x[9], const T u0xp[9], const T u1xp[9],
                                         T u0xh[9], T u1xh[9]) {
    /* just computing the linear part of the strains for now, will add back NL later */

    // nonlinear backprop grad style strains_h => disp grads_h
    computeBendingStrainSens<T, is_nonlinear>(ekh, u0x, u1x, u0xh, u1xh);

    if constexpr (is_nonlinear) {
        // only nonlinear part is bending strains
        // k11 computation
        u0xh[0] += u1xp[0] * ekb[0];
        u1xh[0] += u0xp[0] * ekb[0];
        u0xh[3] += u1xp[3] * ekb[0];
        u1xh[3] += u0xp[3] * ekb[0];
        u0xh[6] += u1xp[6] * ekb[0];
        u1xh[6] += u0xp[6] * ekb[0];
        // k22 computatio
        u0xh[1] += u1xp[1] * ekb[1];
        u1xh[1] += u0xp[1] * ekb[1];
        u0xh[4] += u1xp[4] * ekb[1];
        u1xh[4] += u0xp[4] * ekb[1];
        u0xh[7] += u1xp[7] * ekb[1];
        u1xh[7] += u0xp[7] * ekb[1];
        // k12 computatio
        u0xh[0] += u1xp[1] * ekb[2];
        u1xh[0] += u0xp[1] * ekb[2];
        u0xh[1] += u1xp[0] * ekb[2];
        u1xh[1] += u0xp[0] * ekb[2];
        u0xh[3] += u1xp[4] * ekb[2];
        u1xh[3] += u0xp[4] * ekb[2];
        u0xh[4] += u1xp[3] * ekb[2];
        u1xh[4] += u0xp[3] * ekb[2];
        u0xh[6] += u1xp[7] * ekb[2];
        u1xh[6] += u0xp[7] * ekb[2];
        u0xh[7] += u1xp[6] * ekb[2];
        u1xh[7] += u0xp[6] * ekb[2];
    }
}