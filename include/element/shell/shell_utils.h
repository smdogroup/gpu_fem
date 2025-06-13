#pragma once
#include "../../cuda_utils.h"
#include "_basis_utils.h"

// for now still include the old shell utils methods
#include "shell_utils_slow.h"
// below are all the newer fast ones on GPU

template <typename T, class Basis, class Data>
__HOST_DEVICE__ void ShellComputeTransformLight(const T refAxis[], const T pt[], const T xpts[],
                                                const T n0[], T Tmat[]) {
// make the normal a unit vector, store it in Tmat (Tmat assembled in transpose form first for
// convenience, then transposed later)
#pragma unroll
    for (int i = 0; i < 9; i++) Tmat[i] = 0.0;  // zero out
#pragma unroll
    for (int i = 0; i < 3; i++) {
        Tmat[6 + i] = n0[i];
    }
    T *n = &Tmat[6];
    T norm = sqrt(A2D::VecDotCore<T, 3>(n, n));
#pragma unroll
    for (int i = 0; i < 3; i++) {
        n[i] /= norm;
    }

    // set t1
    if constexpr (Data::has_ref_axis) {
        // shell ref axis transform
        A2D::VecAddCore<T, 3>(1.0, refAxis, &Tmat[0]);
    } else {  // doesn't have ref axis
              // shell natural transform, set to dX/dxi
#pragma unroll
        for (int i = 0; i < 3; i++) {
            Tmat[i] = Basis::template interpFieldsGradLight<XI, 3>(i, pt, xpts);
        }
    }

    // remove normal component from t1 (of n)
    T *t1 = &Tmat[0], *tmp = &Tmat[3];
    T d = A2D::VecDotCore<T, 3>(t1, n);  // t1 cross n
    A2D::VecAddCore<T, 3>(T(1.0), t1, tmp);
    A2D::VecAddCore<T, 3>(-d, n, tmp);  // t1 - (t1 dot n) * n => t2 (temp store in t2)
    norm = sqrt(A2D::VecDotCore<T, 3>(tmp, tmp));
#pragma unroll
    for (int i = 0; i < 3; i++) {
        t1[i] = tmp[i] / norm;
    }

    // compute t2 by cross product
    T *t2 = tmp;
    A2D::VecCrossCore<T>(n, t1, t2);  // nhat cross t1hat => t2hat

// transpose Tmat to standard column major from row major format
#pragma unroll
    for (int i = 0; i < 3; i++) {
#pragma unroll
        for (int j = 0; j < i; j++) {
            // in place transpose swap
            T tmp = Tmat[3 * i + j];
            Tmat[3 * i + j] = Tmat[3 * j + i];
            Tmat[3 * j + i] = tmp;
        }
    }
}

template <typename T, class Data, class Basis>
__HOST_DEVICE__ void ShellComputeNodalTransforms(const int inode, const T xpts[],
                                                 const Data physData, T Tmat[9], T XdinvT[9]) {
    T node_pt[2];
    Basis::getNodePoint(inode, node_pt);

    // get shell transform and Xdn frame scope
    T Xdinv[9];
    {
        T n0[3];
        Basis::ShellComputeNodeNormalLight(node_pt, xpts, n0);

        // assemble Xd frame (Tmat treated here as Xd)
        T *Xd = &Tmat[0];
        Basis::template assembleFrameLight<3>(node_pt, xpts, n0, Xd);
        A2D::MatInvCore<T, 3>(Xd, Xdinv);

        // compute the shell transform based on the ref axis in Data object
        ShellComputeTransformLight<T, Basis, Data>(physData.refAxis, node_pt, xpts, n0, Tmat);
    }  // end of Xd and shell transform scope

    // get full transform product
    A2D::MatMatMultCore3x3<T>(Xdinv, Tmat, XdinvT);
}

template <typename T, class Data, class Basis, class Quadrature>
__HOST_DEVICE__ void ShellComputeQuadptTransforms(const int iquad, const T xpts[],
                                                  const Data physData, T Tmat[9], T XdinvT[9],
                                                  T XdinvzT[9]) {
    T quad_pt[2];
    Quadrature::getQuadraturePoint(iquad, quad_pt);

    // get shell transform and Xdn frame scope
    T Xdinv[9], Xdz[9];
    {
        T n0[3];
        Basis::interpNodeNormalLight(quad_pt, xpts, n0);

        // assemble Xd frame (Tmat treated here as Xd)
        T *Xd = &Tmat[0];
        Basis::template assembleFrameLight<3>(quad_pt, xpts, n0, Xd);
        A2D::MatInvCore<T, 3>(Xd, Xdinv);

        // compute Xdz
        Basis::ShellComputeNormalFrameLight(quad_pt, xpts, Xdz);

        // compute the shell transform based on the ref axis in Data object
        ShellComputeTransformLight<T, Basis, Data>(physData.refAxis, quad_pt, xpts, n0, Tmat);
    }  // end of Xd and shell transform scope

    // get full transform product XdinvT
    A2D::MatMatMultCore3x3<T>(Xdinv, Tmat, XdinvT);

    // compute XdinvzT = -1.0 * Xdinv * Xdz * XdinvT
    T tmp[9];
    A2D::MatMatMultCore3x3Scale<T>(-1.0, Xdinv, Xdz, tmp);
    A2D::MatMatMultCore3x3<T>(tmp, XdinvT, XdinvzT);
}

template <typename T, int vars_per_node, class Basis, class Director>
__HOST_DEVICE__ void ShellComputeDrillStrainFast(const T quad_pt[], const T xpts[], const T vars[],
                                                 const T Tmat[], const T XdinvT[], T &et) {
    // instead of storing etn[4], we add to interpolated et on the fly..
    et = 0.0;
    for (int inode = 0; inode < Basis::num_nodes; inode++) {
        T node_pt[2];
        Basis::getNodePoint(inode, node_pt);

        // const T Tmat = &sTmat[9 * inode];
        // const T XdinvT = &sXdinvT[9 * inode];

        // // assemble u0xn frame scope
        T u0xn[9], zero[3];
        {
            for (int i = 0; i < 3; i++) zero[i] = 0.0;
            Basis::template assembleFrameLight<vars_per_node>(node_pt, vars, zero, u0xn);
        }

        // compute rotation matrix at this node
        T C[9], tmp[9];
        Director::template computeRotationMat<vars_per_node, 1>(&vars[vars_per_node * inode], C);

        // compute Ct = T^T * C * T
        using MatOp = A2D::MatOp;
        A2D::MatMatMultCore3x3<T, MatOp::TRANSPOSE>(&Tmat[9 * inode], C, tmp);
        A2D::MatMatMultCore3x3<T>(tmp, &Tmat[9 * inode], C);

        // Compute transformation u0x = T^T * u0xn * (Xdinv*T)
        A2D::MatMatMultCore3x3<T>(u0xn, &XdinvT[9 * inode], tmp);
        A2D::MatMatMultCore3x3<T, MatOp::TRANSPOSE>(&Tmat[9 * inode], tmp, u0xn);

        // compute the drill strain
        T etn = Director::evalDrillStrain(u0xn, C);

        et += etn * Basis::lagrangeLobatto2DLight(inode, quad_pt[0], quad_pt[1]);
    }  // end of node for loop

}  // end of method ShellComputeDrillStrainFast

template <typename T, int vars_per_node, class Basis, class Director>
__HOST_DEVICE__ void ShellComputeDrillStrainSensFast(const T quad_pt[], const T xpts[],
                                                     const T vars[], const T Tmat[],
                                                     const T XdinvT[], const T et_bar[], T res[]) {
    // TODO : do we actually need Ctn, Tn, XdinvTn, u0xn here?

    // first interpolate back to nodal level
    A2D::Vec<T, Basis::num_nodes> etn_bar;
    Basis::template interpFieldsTransposeLight<1, 1>(quad_pt, et_bar, etn_bar.get_data());

    constexpr A2D::MatOp NORM = A2D::MatOp::NORMAL;
    constexpr A2D::MatOp TRANS = A2D::MatOp::TRANSPOSE;

    for (int inode = 0; inode < Basis::num_nodes; inode++) {
        T pt[2];
        Basis::getNodePoint(inode, pt);

        T u0xn_bar[9], C_bar[9];  // really u0x_bar at this point (but don't want two vars for it)
        Director::evalDrillStrainSens(etn_bar[inode], u0xn_bar, C_bar);

        T tmp[9];
        {
            // C_bar = T * C2_bar * T^t (in reverse)
            A2D::MatMatMultCore3x3<T>(&Tmat[9 * inode], C_bar, tmp);
            A2D::MatMatMultCore3x3<T, NORM, TRANS>(tmp, &Tmat[9 * inode], C_bar);

            // reverse C(vars) to C_bar => res
            Director::template computeRotationMatSens<vars_per_node, 1>(
                C_bar, &res[vars_per_node * inode]);
        }

        // backprop u0x_bar to u0xn_bar^T
        {
            // u0xn_bar = T * u0x_bar * XdinvT^t
            // transpose version for convenience of cols avail in rows
            // u0xn_bar^t = XdinvT * u0x_bar^t * T^t (u0xn_bar now holds transpose u0xn_bar^t)
            A2D::MatMatMultCore3x3<T, NORM, TRANS>(&XdinvT[9 * inode], u0xn_bar, tmp);
            A2D::MatMatMultCore3x3<T, NORM, TRANS>(tmp, &Tmat[9 * inode], u0xn_bar);

            // reverse the interpolations u0xn_bar to res
            // because we have u0xn_bar^T stored, each row is u0xi_bar, u0eta_bar
            Basis::template interpFieldsGradTransposeLight<vars_per_node, 3>(pt, &u0xn_bar[0],
                                                                             &u0xn_bar[3], res);
        }

    }  // end of node for loop
}  // end of method ShellComputeDrillStrainSensFast

template <typename T, class Physics, class Basis, class Director>
__HOST_DEVICE__ static void computeInterpTyingStrainFast(const T quad_pt[], const T xpts[],
                                                         const T vars[], T gty[]) {
    // using unrolled loop here for efficiency (if statements and for loops not
    // great for device)
    // int32_t offset, num_tying;
    static constexpr bool is_nonlinear = Physics::is_nonlinear;
    static constexpr int vars_per_node = Physics::vars_per_node;
    static constexpr int order = Basis::order;

    T tying_pt[2], ety, d0[3], n0[3];

    // if statements by order allows us to inline and manually unroll the loops given the order,
    // which reduces registers (instead of for loops which may not
    // go to registers if array indexing in for loops

    if constexpr (order == 2) {
        // g11 tying strain = X,xi * U0,xi + 0.5 * U0,xi * U0,xi
        // -----------------------------------------------------
#pragma unroll
        for (int itying = 0; itying < 2; itying++) {
            tying_pt[0] = 0.0;
            tying_pt[1] = -1 + itying * 2;
            ety = Basis::template interpFieldsGradDotLight<XI, XI, 3, vars_per_node, 3>(tying_pt,
                                                                                        xpts, vars);
            if constexpr (is_nonlinear)
                ety += 0.5 * Basis::template interpFieldsGradDotLight<XI, XI, vars_per_node,
                                                                      vars_per_node, 3>(tying_pt,
                                                                                        vars, vars);
            gty[0] += ety * Basis::template lagrangeLobatto1D_tyingLight<2>(itying, quad_pt[1]);
        }

        // g12 tying strain = 0.5 * (X,eta * U0,xi + X,xi * U0,eta) + 0.5 * U0,xi * U0,eta
        // -----------------------------------------------------
#pragma unroll
        for (int itying = 0; itying < 1; itying++) {
            tying_pt[0] = 0.0;
            tying_pt[1] = 0.0;
            Basis::template getTyingPoint<2>(itying, tying_pt);
            ety = 0.5 * Basis::template interpFieldsGradDotLight<ETA, XI, 3, vars_per_node, 3>(
                            tying_pt, xpts, vars);
            ety += 0.5 * Basis::template interpFieldsGradDotLight<XI, ETA, 3, vars_per_node, 3>(
                             tying_pt, xpts, vars);
            if constexpr (is_nonlinear)
                ety += 0.5 * Basis::template interpFieldsGradDotLight<XI, ETA, vars_per_node,
                                                                      vars_per_node, 3>(tying_pt,
                                                                                        vars, vars);
            gty[1] += ety;  // only one tying ponit, just becomes the quadpt (MITC)
        }

        // g13 = 0.5 * (X,xi * d0 + U,xi * n0) + 0.5 * U,xi * d0
        // -----------------------------------------------------
#pragma unroll
        for (int itying = 0; itying < 2; itying++) {
            tying_pt[0] = 0.0;
            tying_pt[1] = -1 + itying * 2;

            Director::template interpDirectorLight<Basis, vars_per_node, Basis::num_nodes>(
                tying_pt, xpts, vars, d0);
            ety = 0.5 * Basis::template interpFieldsGradRightDotLight<XI, 3, 3>(tying_pt, xpts, d0);
            {
                Basis::interpNodeNormalLight(tying_pt, xpts, n0);
                ety += 0.5 * Basis::template interpFieldsGradRightDotLight<XI, vars_per_node, 3>(
                                 tying_pt, vars, n0);
            }

            if constexpr (is_nonlinear)
                ety += 0.5 * Basis::template interpFieldsGradRightDotLight<XI, vars_per_node, 3>(
                                 tying_pt, vars, d0);
            gty[2] += ety * Basis::template lagrangeLobatto1D_tyingLight<2>(itying, quad_pt[1]);
        }

        // g22 tying strain = X,eta * U0,eta + 0.5 * U0,eta * U0,eta
        // -----------------------------------------------------
#pragma unroll
        for (int itying = 0; itying < 2; itying++) {
            tying_pt[0] = -1 + itying * 2;
            tying_pt[1] = 0.0;
            ety = Basis::template interpFieldsGradDotLight<ETA, ETA, 3, vars_per_node, 3>(
                tying_pt, xpts, vars);
            if constexpr (is_nonlinear)
                ety += 0.5 * Basis::template interpFieldsGradDotLight<ETA, ETA, vars_per_node,
                                                                      vars_per_node, 3>(tying_pt,
                                                                                        vars, vars);
            gty[3] += ety * Basis::template lagrangeLobatto1D_tyingLight<2>(itying, quad_pt[0]);
        }

        // g23 tying strain = X,eta * U0,eta + 0.5 * U0,eta * U0,eta
        // -----------------------------------------------------
#pragma unroll
        for (int itying = 0; itying < 2; itying++) {
            tying_pt[0] = -1 + itying * 2;
            tying_pt[1] = 0.0;

            Director::template interpDirectorLight<Basis, vars_per_node, Basis::num_nodes>(
                tying_pt, xpts, vars, d0);
            ety =
                0.5 * Basis::template interpFieldsGradRightDotLight<ETA, 3, 3>(tying_pt, xpts, d0);
            {
                Basis::interpNodeNormalLight(tying_pt, xpts, n0);
                ety += 0.5 * Basis::template interpFieldsGradRightDotLight<ETA, vars_per_node, 3>(
                                 tying_pt, vars, n0);
            }

            if constexpr (is_nonlinear)
                ety += 0.5 * Basis::template interpFieldsGradRightDotLight<ETA, vars_per_node, 3>(
                                 tying_pt, vars, d0);
            gty[4] += ety * Basis::template lagrangeLobatto1D_tyingLight<2>(itying, quad_pt[0]);
        }

        // g33 = 0, repr gty[5]
    }

    // if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0)
    //     printf("gty = %.4e, %.4e, %.4e, %.4e, %.4e\n", gty[0], gty[1], gty[2], gty[3], gty[4]);
}  // end of computeInterpTyingStrainFast

template <typename T, class Physics, class Basis, class Director>
__HOST_DEVICE__ static void computeInterpTyingStrainFastSens(const T quad_pt[], const T xpts[],
                                                             const T vars[], const T gty_bar[],
                                                             T res[]) {
    // using unrolled loop here for efficiency (if statements and for loops not
    // great for device)
    // int32_t offset, num_tying;
    static constexpr bool is_nonlinear = Physics::is_nonlinear;
    static constexpr int vars_per_node = Physics::vars_per_node;
    static constexpr int order = Basis::order;

    T tying_pt[2], ety_bar, d0[3], n0[3];

    // if statements by order allows us to inline and manually unroll the loops given the order,
    // which reduces registers (instead of for loops which may not
    // go to registers if array indexing in for loops

    if constexpr (order == 2) {
        // g11 tying strain = X,xi * U0,xi + 0.5 * U0,xi * U0,xi
        // -----------------------------------------------------
#pragma unroll
        for (int itying = 0; itying < 2; itying++) {
            tying_pt[0] = 0.0;
            tying_pt[1] = -1 + itying * 2;

            ety_bar =
                gty_bar[0] * Basis::template lagrangeLobatto1D_tyingLight<2>(itying, quad_pt[1]);

            Basis::template addInterpFieldsGradDotSensLight<XI, XI, 3, vars_per_node, 3>(
                ety_bar, tying_pt, xpts, res);

            if constexpr (is_nonlinear)
                Basis::template addInterpFieldsGradDotSensLight<XI, XI, vars_per_node,
                                                                vars_per_node, 3>(ety_bar, tying_pt,
                                                                                  vars, res);
        }

        // g12 tying strain = 0.5 * (X,eta * U0,xi + X,xi * U0,eta) + 0.5 * U0,xi * U0,eta
        // -----------------------------------------------------
#pragma unroll
        for (int itying = 0; itying < 1; itying++) {
            tying_pt[0] = 0.0;
            tying_pt[1] = 0.0;

            ety_bar = 0.5 * gty_bar[1];
            Basis::template addInterpFieldsGradDotSensLight<ETA, XI, 3, vars_per_node, 3>(
                ety_bar, tying_pt, xpts, res);
            Basis::template addInterpFieldsGradDotSensLight<XI, ETA, 3, vars_per_node, 3>(
                ety_bar, tying_pt, xpts, res);

            if constexpr (is_nonlinear) {
                Basis::template addInterpFieldsGradDotSensLight<XI, ETA, vars_per_node,
                                                                vars_per_node, 3>(ety_bar, tying_pt,
                                                                                  vars, res);
                Basis::template addInterpFieldsGradDotSensLight<ETA, XI, vars_per_node,
                                                                vars_per_node, 3>(ety_bar, tying_pt,
                                                                                  vars, res);
            }
        }

        // g13
#pragma unroll
        for (int itying = 0; itying < 2; itying++) {
            tying_pt[0] = 0.0;
            tying_pt[1] = -1 + itying * 2;

            ety_bar = 0.5 * gty_bar[2] *
                      Basis::template lagrangeLobatto1D_tyingLight<2>(itying, quad_pt[1]);

            A2D::Vec<T, 3> d0_bar;
            Basis::template interpFieldsGradRightDotLight_RightSens<XI, 3, 3>(
                ety_bar, tying_pt, xpts, d0_bar.get_data());
            Basis::template interpNodeNormalLight(tying_pt, xpts, n0);
            Basis::template interpFieldsGradRightDotLight_LeftSens<XI, vars_per_node, 3>(
                ety_bar, tying_pt, res, n0);

            // nonlinear g13 strain term += 0.5 * U,eta dot d0
            if constexpr (is_nonlinear) {
                Director::template interpDirectorLight<Basis, vars_per_node, Basis::num_nodes>(
                    tying_pt, xpts, vars, d0);

                Basis::template interpFieldsGradRightDotLight_LeftSens<XI, vars_per_node, 3>(
                    ety_bar, tying_pt, res, d0);
                Basis::template interpFieldsGradRightDotLight_RightSens<XI, vars_per_node, 3>(
                    ety_bar, tying_pt, vars, d0_bar.get_data());
            }
            Director::template interpDirectorLightSens<Basis, vars_per_node, Basis::num_nodes>(
                1.0, tying_pt, xpts, d0_bar.get_data(), res);
        }

        // g22
#pragma unroll  // for low num_tying can speed up?
        for (int itying = 0; itying < 2; itying++) {
            tying_pt[0] = -1 + itying * 2;
            tying_pt[1] = 0.0;

            ety_bar =
                gty_bar[3] * Basis::template lagrangeLobatto1D_tyingLight<2>(itying, quad_pt[0]);

            Basis::template addInterpFieldsGradDotSensLight<ETA, ETA, 3, vars_per_node, 3>(
                ety_bar, tying_pt, xpts, res);

            if constexpr (is_nonlinear) {
                // backprop g22 nl term 1/2 * U0,eta dot U0,eta
                Basis::template addInterpFieldsGradDotSensLight<ETA, ETA, vars_per_node,
                                                                vars_per_node, 3>(ety_bar, tying_pt,
                                                                                  vars, res);
            }
        }  // end of itying for loop for g22

        // g23
#pragma unroll
        for (int itying = 0; itying < 2; itying++) {
            tying_pt[0] = -1 + itying * 2;
            tying_pt[1] = 0.0;

            ety_bar = 0.5 * gty_bar[4] *
                      Basis::template lagrangeLobatto1D_tyingLight<2>(itying, quad_pt[0]);

            A2D::Vec<T, 3> d0_bar;
            Basis::template interpFieldsGradRightDotLight_RightSens<ETA, 3, 3>(
                ety_bar, tying_pt, xpts, d0_bar.get_data());
            Basis::template interpNodeNormalLight(tying_pt, xpts, n0);
            Basis::template interpFieldsGradRightDotLight_LeftSens<ETA, vars_per_node, 3>(
                ety_bar, tying_pt, res, n0);

            // nonlinear g23 strain term += 0.5 * U,eta dot d0
            if constexpr (is_nonlinear) {
                Director::template interpDirectorLight<Basis, vars_per_node, Basis::num_nodes>(
                    tying_pt, xpts, vars, d0);

                Basis::template interpFieldsGradRightDotLight_LeftSens<ETA, vars_per_node, 3>(
                    ety_bar, tying_pt, res, d0);
                Basis::template interpFieldsGradRightDotLight_RightSens<ETA, vars_per_node, 3>(
                    ety_bar, tying_pt, vars, d0_bar.get_data());
            }
            Director::template interpDirectorLightSens<Basis, vars_per_node, Basis::num_nodes>(
                1.0, tying_pt, xpts, d0_bar.get_data(), res);
        }

        // g33 = 0
    }
}  // end of computeInterpTyingStrainFastSens

template <typename T, int vars_per_node, class Basis, class Director>
__HOST_DEVICE__ void assembleDirectorFrame(const T quad_pt[], const T xpts[], const T vars[],
                                           T u1x_0[]) {
    // goal is to assemble the 3x3 frame (d0,xi; d0,eta; 0-vec) efficiently
    T node_pt[2], n0[3], d[3];
    const T *q = &vars[Director::offset];

#pragma unroll
    for (int i = 0; i < 9; i++) {
        u1x_0[i] = 0.0;
    }

#pragma unroll
    for (int inode = 0; inode < Basis::num_nodes; inode++) {
        // compute the director at this point (unrolled interpDirectorLight here)
        Basis::getNodePoint(inode, node_pt);
        Basis::ShellComputeNodeNormal(node_pt, xpts, n0);

        A2D::VecCrossCore<T>(q, n0, d);
        q += vars_per_node;

#pragma unroll
        for (int idim = 0; idim < 3; idim++) {
            // add into Xd at the pt (which is usually quad_pt)
            u1x_0[3 * idim] += d[idim] * Basis::template lagrangeLobatto2DGradLight<XI>(
                                             inode, quad_pt[0], quad_pt[1]);
            u1x_0[3 * idim + 1] += d[idim] * Basis::template lagrangeLobatto2DGradLight<ETA>(
                                                 inode, quad_pt[0], quad_pt[1]);
        }
    }
}

template <typename T, int vars_per_node, class Basis, class Director>
__HOST_DEVICE__ void assembleDirectorFrameSens(const T quad_pt[], const T xpts[], const T u1x_bar[],
                                               T res[]) {
    // goal is to assemble the 3x3 frame (d0,xi; d0,eta; 0-vec) efficiently
    T d_bar[3], node_pt[2], n0[3], d[3];
    T *q_bar = &res[Director::offset];

#pragma unroll
    for (int inode = 0; inode < Basis::num_nodes; inode++) {
        // compute the director at this point (unrolled interpDirectorLight here)
        Basis::getNodePoint(inode, node_pt);
        Basis::ShellComputeNodeNormal(node_pt, xpts, n0);

#pragma unroll
        for (int idim = 0; idim < 3; idim++) {
            d_bar[idim] = u1x_bar[3 * idim] * Basis::template lagrangeLobatto2DGradLight<XI>(
                                                  inode, quad_pt[0], quad_pt[1]);
            d_bar[idim] += u1x_bar[3 * idim + 1] * Basis::template lagrangeLobatto2DGradLight<ETA>(
                                                       inode, quad_pt[0], quad_pt[1]);
        }

        A2D::VecCrossCoreAdd<T>(n0, d_bar, q_bar);
        q_bar += vars_per_node;
    }
}

template <typename T, int vars_per_node, class Basis, class Director, bool is_nonlinear>
__HOST_DEVICE__ void computeBendingStrains(const T quad_pt[], const T xpts[], const T vars[],
                                           const T Tmat[9], const T XdinvT[9], const T XdinvzT[9],
                                           T u0x[9], T u1x[9], T ek[3]) {
    // get init u0x frame
    {
        T d0[3];
        Director::template interpDirectorLight<Basis, vars_per_node, Basis::num_nodes>(
            quad_pt, xpts, vars, d0);
        Basis::template assembleFrameLight<vars_per_node>(quad_pt, vars, d0, u0x);
    }

    // get init u1x frame = (d0,xi; d0,eta; 0-vec)
    { assembleDirectorFrame<T, vars_per_node, Basis, Director>(quad_pt, xpts, vars, u1x); }

    // now compute transformed u0x, u1x with shared memory calculations
    T tmp[9];
    // compute u1x = T^{T}*u1d*XdinvT + T^{T}*u0d*XdinvzT
    {
        A2D::MatMatMultCore3x3<T>(u1x, XdinvT, tmp);
        A2D::MatMatMultCore3x3Add<T>(u0x, XdinvzT, tmp);
        A2D::MatMatMultCore3x3<T, A2D::MatOp::TRANSPOSE>(Tmat, tmp, u1x);
    }

    // compute u0x = T^{T}*u0d*Xdinv*T
    {
        A2D::MatMatMultCore3x3<T>(u0x, XdinvT, tmp);
        A2D::MatMatMultCore3x3<T, A2D::MatOp::TRANSPOSE>(Tmat, tmp, u0x);
    }

    // immediately go to the bending strains (the hope is that using this instead of u0x[9], u1x[9])
    // may promote fewer registers through inlining A2D functions
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

template <typename T, int vars_per_node, class Basis, class Director, bool is_nonlinear>
__HOST_DEVICE__ void computeBendingStrainSens(const T quad_pt[], const T xpts[], const T vars[],
                                              const T Tmat[9], const T XdinvT[9],
                                              const T XdinvzT[9], const T ek_bar[3], const T u0x[9],
                                              const T u1x[9], T res[]) {
    // TODO:
    T u0x_bar[9], u1x_bar[9], tmp[9];

#pragma unroll
    for (int i = 0; i < 9; i++) {
        u0x_bar[i] = 0.0;
        u1x_bar[i] = 0.0;
    }

    // linear part
    u1x_bar[0] = ek_bar[0];
    u1x_bar[4] = ek_bar[1];
    u1x_bar[1] = ek_bar[2];
    u1x_bar[3] = ek_bar[2];

    // nonlinear part
    if constexpr (is_nonlinear) {
        // k11 part, ek[0] = u0x[0] * u1x[0] + u0x[3] * u1x[3] + u0x[6] * u1x[6]
        u1x_bar[0] += u0x[0] * ek_bar[0];
        u1x_bar[3] += u0x[3] * ek_bar[0];
        u1x_bar[6] += u0x[6] * ek_bar[0];
        u0x_bar[0] += u1x[0] * ek_bar[0];
        u0x_bar[3] += u1x[3] * ek_bar[0];
        u0x_bar[6] += u1x[6] * ek_bar[0];

        // k22 part, ek[1] = u0x[1] * u1x[1] + u0x[4] * u1x[4] + u0x[7] * u1x[7]
        u1x_bar[1] += u0x[1] * ek_bar[1];
        u1x_bar[4] += u0x[4] * ek_bar[1];
        u1x_bar[7] += u0x[7] * ek_bar[1];
        u0x_bar[1] += u1x[1] * ek_bar[1];
        u0x_bar[4] += u1x[4] * ek_bar[1];
        u0x_bar[7] += u1x[7] * ek_bar[1];

        // k12 part, ek[2] = u0x[0] * u1x[1] + u0x[3] * u1x[4] + u0x[6] * u1x[7] + u1x[0] * u0x[1] +
        //              u1x[3] * u0x[4] + u1x[6] * u0x[7];  // k12
        u0x_bar[0] += u1x[1] * ek_bar[2];
        u1x_bar[0] += u0x[1] * ek_bar[2];
        u0x_bar[1] += u1x[0] * ek_bar[2];
        u1x_bar[1] += u0x[0] * ek_bar[2];
        u0x_bar[3] += u1x[4] * ek_bar[2];
        u1x_bar[3] += u0x[4] * ek_bar[2];
        u0x_bar[4] += u1x[3] * ek_bar[2];
        u1x_bar[4] += u0x[3] * ek_bar[2];
        u0x_bar[6] += u1x[7] * ek_bar[2];
        u1x_bar[6] += u0x[7] * ek_bar[2];
        u0x_bar[7] += u1x[6] * ek_bar[2];
        u1x_bar[7] += u0x[6] * ek_bar[2];
    }

    // now backprop through the u0x, u1x parts
    constexpr A2D::MatOp NORM = A2D::MatOp::NORMAL;
    constexpr A2D::MatOp TRANS = A2D::MatOp::TRANSPOSE;

    // reverse of u1x' += T^t * u0x * XdinvzT and u0x' = T^t * u0x * XdinvT
    // becomes u0x_bar^t = XdinvzT * u1x'_bar * T^t + XdinvT * u0x'_bar * T^t
    {
        // or un-transposed version
        // u0x_bar += T * u0x'_bar * XdinvT^t
        A2D::MatMatMultCore3x3<T>(Tmat, u0x_bar, tmp);
        A2D::MatMatMultCore3x3<T, NORM, TRANS>(tmp, XdinvT, u0x_bar);

        // u0x_bar += T * u1x'_bar^t * XdinvzT^t
        A2D::MatMatMultCore3x3<T>(Tmat, u1x_bar, tmp);
        A2D::MatMatMultCore3x3Add<T, NORM, TRANS>(tmp, XdinvzT, u0x_bar);
    }

    // reverse of fw: u1x' = T^t * u1x * XdinvT
    // first u1x_bar^t = XdinvT * u1x'_bar * T^t
    {
        // or un-transposed version, u1x_bar = T * u1x'_bar^t * XdinvT^t
        A2D::MatMatMultCore3x3<T>(Tmat, u1x_bar, tmp);
        A2D::MatMatMultCore3x3<T, NORM, TRANS>(tmp, XdinvT, u1x_bar);
    }

    // so now u0x_bar, u1x_bar are at un-transformed stage in reverse
    // backprop now to the variables
    // -----------------------------

    // backprop u0x_bar to vars or res
    {
        T d0_bar[3];
        Basis::template assembleFrameLightSens<vars_per_node>(quad_pt, u0x_bar, d0_bar, res);

        Director::template interpDirectorLightSens<Basis, vars_per_node, Basis::num_nodes>(
            1.0, quad_pt, xpts, d0_bar, res);
    }

    // backprop u1x_bar to vars or res
    assembleDirectorFrameSens<T, vars_per_node, Basis, Director>(quad_pt, xpts, u1x_bar, res);
}
