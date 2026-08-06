#pragma once
#include "cuda_utils.h"

/* there is only one drill strain in the shell element */

template <typename T, int vars_per_node, class Basis, class Director>
__HOST_DEVICE__ void ShellComputeDrillStrainFast(const T pt[], const T Tmat[], const T XdinvT[],
                                                 const T vars[], T et[1]) {
    // assemble u0xn frame
    T u0xn[9];
    {
        // compute midplane disp field gradients
        T u0xi[3], u0eta[3];
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, u0xi, u0eta);

        A2D::Vec<T, 3> zero;
        assembleFrame<T>(u0xi, u0eta, zero.get_data(), u0xn);
    }

    // interpolate the displacements to the quadpt
    T u0[6], C[9];
    Basis::template interpFields<vars_per_node, vars_per_node>(pt, vars, u0);
    Director::template computeRotationMatSinglePt(&u0[Director::offset], C);

    // now rotate the rotation mat by shell transform
    T tmp[9];
    A2D::MatMatMultCore3x3<T, A2D::MatOp::TRANSPOSE>(Tmat, C, tmp);
    A2D::MatMatMultCore3x3<T>(tmp, Tmat, C);

    // Compute transformation u0x = T^T * u0xn * (Xdinv*T)
    A2D::MatMatMultCore3x3<T>(u0xn, XdinvT, tmp);
    A2D::MatMatMultCore3x3<T, A2D::MatOp::TRANSPOSE>(Tmat, tmp, u0xn);

    // compute the drill strain
    et[0] = Director::evalDrillStrain(u0xn, C);

}  // end of method ShellComputeDrillStrain

template <typename T, int vars_per_node, class Basis, class Director>
__HOST_DEVICE__ void ShellComputeDrillStrainFastSens(const T pt[], const T Tmat[], const T XdinvT[],
                                                     const T et_bar[1], T res[]) {
    constexpr A2D::MatOp NORM = A2D::MatOp::NORMAL;
    constexpr A2D::MatOp TRANS = A2D::MatOp::TRANSPOSE;

    T u0xn_bar[9], C_bar[9];  // really u0x_bar at this point (but don't want two vars for it)
    Director::evalDrillStrainSens(et_bar[0], u0xn_bar, C_bar);

    T tmp[9];
    {
        // C_bar = T * C2_bar * T^t (in reverse)
        A2D::MatMatMultCore3x3<T>(Tmat, C_bar, tmp);
        A2D::MatMatMultCore3x3<T, NORM, TRANS>(tmp, Tmat, C_bar);

        // reverse C(vars) to C_bar => res
        T u0_sens[6];
        memset(u0_sens, 0.0, 6 * sizeof(T));
        Director::template computeRotationMatSinglePtSens(C_bar, &u0_sens[Director::offset]);

        // reverse the interp step here
        Basis::template interpFieldsTranspose<vars_per_node, vars_per_node>(pt, u0_sens, res);
    }

    // backprop u0x_bar to u0xn_bar^T
    {
        // u0xn_bar = T * u0x_bar * XdinvT^t
        // transpose version for convenience of cols avail in rows
        // u0xn_bar^t = XdinvT * u0x_bar^t * T^t (u0xn_bar now holds transpose u0xn_bar^t)
        A2D::MatMatMultCore3x3<T, NORM, TRANS>(XdinvT, u0xn_bar, tmp);
        A2D::MatMatMultCore3x3<T, NORM, TRANS>(tmp, Tmat, u0xn_bar);

        // reverse the interpolations u0xn_bar to res
        // because we have u0xn_bar^T stored, each row is u0xi_bar, u0eta_bar
        Basis::template interpFieldsGradTranspose<vars_per_node, 3>(pt, &u0xn_bar[0], &u0xn_bar[3],
                                                                    res);
    }
}  // end of method ShellComputeDrillStrainSens
