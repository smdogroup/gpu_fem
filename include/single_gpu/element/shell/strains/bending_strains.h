#pragma once
#include "cuda_utils.h"

template <typename T, int vars_per_node, class Basis, class Data>
__HOST_DEVICE__ T computeBendingDispGrad(const T pt[], const T refAxis[], const T xpts[],
                                         const T vars[], const T fn[], const T d[], T XdinvT[],
                                         T u0x[], T u1x[]) {
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
        assembleFrame<T>(Xxi, Xeta, n0, Xd);
        assembleFrame<T>(nxi, neta, zero.get_data(), Xdz);

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
        assembleFrame<T>(u0xi, u0eta, d0, u0x);
        assembleFrame<T>(d0xi, d0eta, zero.get_data(), u1x);
    }  // end of u0x, u1x frame assembly scope

    // u0x, u1x conversion to physical space scope
    T detXd;
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

    return detXd;
}

template <typename T, int vars_per_node, class Basis, class Data>
__HOST_DEVICE__ void computeBendingDispGradHfwd(const T pt[], const T refAxis[], const T xpts[],
                                                const T p_vars[], const T fn[], const T p_d[],
                                                T XdinvT[9], T p_u0x[], T p_u1x[]) {
    // this is purely linear function, so hforward equiv to forward analysis
    computeBendingDispGrad<T, vars_per_node, Basis, Data>(pt, refAxis, xpts, p_vars, fn, p_d,
                                                          XdinvT, p_u0x, p_u1x);
}

template <typename T, int vars_per_node, class Basis, class Data>
__HOST_DEVICE__ void computeBendingDispGradSens(const T pt[], const T refAxis[], const T xpts[],
                                                const T vars[], const T fn[], const T u0x_bar[],
                                                const T u1x_bar[], T XdinvT[9], T res[],
                                                T d_bar[]) {
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
        assembleFrame<T>(Xxi, Xeta, n0, Xd);
        assembleFrame<T>(nxi, neta, zero.get_data(), Xdz);

        // compute shell trasnform
        ShellComputeTransform<T, Data>(refAxis, Xxi, Xeta, n0, Tmat.get_data());
    }  // Xd, Xdz frame assembly scope

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
}  // ShellComputeDispGradSens

template <typename T, int vars_per_node, class Basis, class Data>
__HOST_DEVICE__ void computeBendingDispGradHrev(const T pt[], const T refAxis[], const T xpts[],
                                                const T vars[], const T fn[], const T h_u0x[],
                                                const T h_u1x[], T XdinvT[9], T matCol[], T h_d[]) {
    // this is purely linear function, so hreverse equivalent to 1st order reverse (aka Sens
    // function)
    computeBendingDispGradSens<T, vars_per_node, Basis, Data>(pt, refAxis, xpts, vars, fn, h_u0x,
                                                              h_u1x, XdinvT, matCol, h_d);
}