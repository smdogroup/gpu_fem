#pragma once
#include "cuda_utils.h"

/* the tying strains include the 3 membrane and 2 transverse shear strains (5 of them in total) */

// in the MITC (mixed interpolation of tensorial component shell), lower order Gauss quadrature is
// used
//    for each strain to prevent transverse shear and membrane locking
// e.g. for a first order element with 2nd order Gauss quadrature that has 4 quadpts for full
// integration,
//    the two transverse shear strains gam_13 and gam_23 use 2 quadpts, reduced along the 1 and 2
//    directions respectively , while the membrane shear strain gam_12 uses only 1 quadpt (reduced
//    in both directions)
// this is an improved version of reduced integration as it prevents zero energy modes while
// somewhat reducing the integration to prevent locking however, it is not as multigrid friendly as
// fully integrated elements, though fully integrated normally lock (with some exceptions for
// advanced elements.. see my paper)

// NOTE : only the forward + sens methods have is_nonlinear separate from the physics is_nonlinear
// this is so that we are free to call the linear part in Hfwd and Hrev

template <typename T, class Physics, class Basis, bool is_nonlinear>
__HOST_DEVICE__ static void computeMITCTyingStrain(const T Xpts[], const T fn[], const T vars[],
                                                   const T d[], T ety[]) {
    // using unrolled loop here for efficiency (if statements and for loops not
    // great for device)
    int32_t offset, num_tying;
    static constexpr int vars_per_node = Physics::vars_per_node;

    // get g11 tying strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(0);
    num_tying = Basis::num_tying_points(0);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<0>(itying, pt);

        // Interpolate the field value
        T Uxi[3], Ueta[3], Xxi[3], Xeta[3];
        Basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi, Xeta);
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, Uxi, Ueta);

        // store g11 strain
        ety[offset + itying] = A2D::VecDotCore<T, 3>(Uxi, Xxi);

        // nonlinear g11 strain term
        if constexpr (is_nonlinear) {
            ety[offset + itying] += 0.5 * A2D::VecDotCore<T, 3>(Uxi, Uxi);
        }

    }  // end of itying for loop for g11

    // get g22 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(1);
    num_tying = Basis::num_tying_points(1);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<1>(itying, pt);

        // Interpolate the field value
        T Uxi[3], Ueta[3], Xxi[3], Xeta[3];
        Basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi, Xeta);
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, Uxi, Ueta);

        // store g22 strain
        ety[offset + itying] = A2D::VecDotCore<T, 3>(Ueta, Xeta);

        // nonlinear g22 strain term
        if constexpr (is_nonlinear) {
            ety[offset + itying] += 0.5 * A2D::VecDotCore<T, 3>(Ueta, Ueta);
        }

    }  // end of itying for loop for g22

    // get g12 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(2);
    num_tying = Basis::num_tying_points(2);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<2>(itying, pt);

        // Interpolate the field value
        T Uxi[3], Ueta[3], Xxi[3], Xeta[3];
        Basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi, Xeta);
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, Uxi, Ueta);

        // store g12 strain
        ety[offset + itying] =
            0.5 * (A2D::VecDotCore<T, 3>(Uxi, Xeta) + A2D::VecDotCore<T, 3>(Ueta, Xxi));

        // nonlinear g12 strain term
        if constexpr (is_nonlinear) {
            ety[offset + itying] += 0.5 * A2D::VecDotCore<T, 3>(Uxi, Ueta);
        }
    }  // end of itying for loop for g12

    // get g23 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(3);
    num_tying = Basis::num_tying_points(3);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<3>(itying, pt);

        // Interpolate the field value
        T Uxi[3], Ueta[3], Xxi[3], Xeta[3];
        Basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi, Xeta);
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, Uxi, Ueta);

        T d0[3], n0[3];
        Basis::template interpFields<3, 3>(pt, d, d0);
        Basis::template interpFields<3, 3>(pt, fn, n0);

        // store g23 strain
        ety[offset + itying] =
            0.5 * (A2D::VecDotCore<T, 3>(Xeta, d0) + A2D::VecDotCore<T, 3>(n0, Ueta));

        // nonlinear g23 strain term
        if constexpr (is_nonlinear) {
            ety[offset + itying] += 0.5 * A2D::VecDotCore<T, 3>(d0, Ueta);
        }
    }  // end of itying for loop for g23

    // get g13 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(4);
    num_tying = Basis::num_tying_points(4);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<4>(itying, pt);

        // Interpolate the field value
        T Uxi[3], Ueta[3], Xxi[3], Xeta[3];
        Basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi, Xeta);
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, Uxi, Ueta);

        T d0[3], n0[3];
        Basis::template interpFields<3, 3>(pt, d, d0);
        Basis::template interpFields<3, 3>(pt, fn, n0);

        // store g13 strain
        ety[offset + itying] =
            0.5 * (A2D::VecDotCore<T, 3>(Xxi, d0) + A2D::VecDotCore<T, 3>(n0, Uxi));

        // nonlinear g13 strain term
        if constexpr (is_nonlinear) {
            ety[offset + itying] += 0.5 * A2D::VecDotCore<T, 3>(d0, Uxi);
        }
    }  // end of itying for loop for g13

}  // end of computeTyingStrain

template <typename T, class Physics, class Basis>
__HOST_DEVICE__ static void computeMITCTyingStrainHfwd(const T Xpts[], const T fn[], const T vars[],
                                                       const T d[], const T p_vars[], const T p_d[],
                                                       T p_ety[]) {
    // linear part
    computeMITCTyingStrain<T, Physics, Basis, false>(Xpts, fn, p_vars, p_d, p_ety);

    // using unrolled loop here for efficiency (if statements and for loops not
    // great for device)
    int32_t offset, num_tying;
    static constexpr bool is_nonlinear = Physics::is_nonlinear;
    static constexpr int vars_per_node = Physics::vars_per_node;

    if constexpr (!is_nonlinear) {
        return;  // exit early for linear part
    }
    // return;  // temporary debug check

    // remaining nonlinear extra terms of Hfwd

    // get g11 tying strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(0);
    num_tying = Basis::num_tying_points(0);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<0>(itying, pt);

        // Interpolate the field value
        T Uxi[3], p_Uxi[3], zero[3];
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, Uxi, zero);
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, p_vars, p_Uxi, zero);
        p_ety[offset + itying] += A2D::VecDotCore<T, 3>(Uxi, p_Uxi);

    }  // end of itying for loop for g11

    // get g22 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(1);
    num_tying = Basis::num_tying_points(1);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<1>(itying, pt);

        // Interpolate the field value
        T Ueta[3], p_Ueta[3], zero[3];
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, zero, Ueta);
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, p_vars, zero, p_Ueta);
        p_ety[offset + itying] += A2D::VecDotCore<T, 3>(Ueta, p_Ueta);

    }  // end of itying for loop for g22

    // get g12 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(2);
    num_tying = Basis::num_tying_points(2);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<2>(itying, pt);

        // Interpolate the field value
        T Uxi[3], Ueta[3], p_Uxi[3], p_Ueta[3];
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, Uxi, Ueta);
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, p_vars, p_Uxi, p_Ueta);
        p_ety[offset + itying] += 0.5 * A2D::VecDotCore<T, 3>(Uxi, p_Ueta);
        p_ety[offset + itying] += 0.5 * A2D::VecDotCore<T, 3>(Ueta, p_Uxi);
    }  // end of itying for loop for g12

    // get g23 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(3);
    num_tying = Basis::num_tying_points(3);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<3>(itying, pt);

        // Interpolate the field value
        T Ueta[3], p_Ueta[3], zero[3], d0[3], p_d0[3];
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, zero, Ueta);
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, p_vars, zero, p_Ueta);
        Basis::template interpFields<3, 3>(pt, d, d0);
        Basis::template interpFields<3, 3>(pt, p_d, p_d0);
        p_ety[offset + itying] += 0.5 * A2D::VecDotCore<T, 3>(d0, p_Ueta);
        p_ety[offset + itying] += 0.5 * A2D::VecDotCore<T, 3>(p_d0, Ueta);
    }  // end of itying for loop for g23

    // get g13 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(4);
    num_tying = Basis::num_tying_points(4);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<4>(itying, pt);

        // Interpolate the field value
        T Uxi[3], p_Uxi[3], zero[3], d0[3], p_d0[3];
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, Uxi, zero);
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, p_vars, p_Uxi, zero);
        Basis::template interpFields<3, 3>(pt, d, d0);
        Basis::template interpFields<3, 3>(pt, p_d, p_d0);
        p_ety[offset + itying] += 0.5 * A2D::VecDotCore<T, 3>(d0, p_Uxi);
        p_ety[offset + itying] += 0.5 * A2D::VecDotCore<T, 3>(p_d0, Uxi);
    }  // end of itying for loop for g13
}

template <typename T, class Physics, class Basis, bool is_nonlinear>
__HOST_DEVICE__ static void computeMITCTyingStrainSens(const T Xpts[], const T fn[], const T vars[],
                                                       const T d[], const T ety_bar[], T res[],
                                                       T d_bar[]) {
    // using unrolled loop here for efficiency (if statements and for loops not
    // great for device)
    int32_t offset, num_tying;
    static constexpr int vars_per_node = Physics::vars_per_node;

    // get g11 tying strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(0);
    num_tying = Basis::num_tying_points(0);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<0>(itying, pt);
        //   ety[offset + itying] = A2D::VecDotCore<T, 3>(Uxi, Xxi);

        T Xxi[3], Xeta[3];
        Basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi, Xeta);

        A2D::Vec<T, 3> Uxi_bar, zero;
        A2D::VecAddCore<T, 3>(ety_bar[offset + itying], Xxi, Uxi_bar.get_data());

        // nonlinear g11 strain term backprop
        if constexpr (is_nonlinear) {
            T Uxi[3], Ueta[3];
            Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, Uxi, Ueta);
            // recall forward analysis nonlinear g11 term:
            // ety[offset + itying] += 0.5 * A2D::VecDotCore<T, 3>(Uxi, Uxi);
            A2D::VecAddCore<T, 3>(ety_bar[offset + itying], Uxi, Uxi_bar.get_data());
        }

        Basis::template interpFieldsGradTranspose<vars_per_node, 3>(pt, Uxi_bar.get_data(),
                                                                    zero.get_data(), res);

    }  // end of itying for loop for g11

    // get g22 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(1);
    num_tying = Basis::num_tying_points(1);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<1>(itying, pt);
        //   ety[offset + itying] = A2D::VecDotCore<T, 3>(Ueta, Xeta);

        T Xxi[3], Xeta[3];
        Basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi, Xeta);

        A2D::Vec<T, 3> Ueta_bar, zero;
        A2D::VecAddCore<T, 3>(ety_bar[offset + itying], Xeta, Ueta_bar.get_data());

        // nonlinear g22 strain term backprop
        if constexpr (is_nonlinear) {
            T Uxi[3], Ueta[3];
            Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, Uxi, Ueta);
            // recall forward analysis nonlinear g22 term:
            // ety[offset + itying] += 0.5 * A2D::VecDotCore<T, 3>(Ueta, Ueta);
            A2D::VecAddCore<T, 3>(ety_bar[offset + itying], Ueta, Ueta_bar.get_data());
        }

        Basis::template interpFieldsGradTranspose<vars_per_node, 3>(pt, zero.get_data(),
                                                                    Ueta_bar.get_data(), res);

    }  // end of itying for loop for g22

    // get g12 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(2);
    num_tying = Basis::num_tying_points(2);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<2>(itying, pt);
        //   ety[offset + itying] = 0.5 * (A2D::VecDotCore<T, 3>(Uxi, Xeta) +
        //                                 A2D::VecDotCore<T, 3>(Ueta, Xxi));

        T Xxi[3], Xeta[3];
        Basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi, Xeta);

        A2D::Vec<T, 3> Uxi_bar, Ueta_bar;
        A2D::VecAddCore<T, 3>(0.5 * ety_bar[offset + itying], Xxi, Ueta_bar.get_data());
        A2D::VecAddCore<T, 3>(0.5 * ety_bar[offset + itying], Xeta, Uxi_bar.get_data());

        // nonlinear g12 strain term backprop
        if constexpr (is_nonlinear) {
            T Uxi[3], Ueta[3];
            Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, Uxi, Ueta);
            // recall forward analysis nonlinear g12 term:
            // ety[offset + itying] += 0.5 * A2D::VecDotCore<T, 3>(Uxi, Ueta);
            A2D::VecAddCore<T, 3>(0.5 * ety_bar[offset + itying], Uxi, Ueta_bar.get_data());
            A2D::VecAddCore<T, 3>(0.5 * ety_bar[offset + itying], Ueta, Uxi_bar.get_data());
        }

        Basis::template interpFieldsGradTranspose<vars_per_node, 3>(pt, Uxi_bar.get_data(),
                                                                    Ueta_bar.get_data(), res);
    }  // end of itying for loop for g12

    // get g23 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(3);
    num_tying = Basis::num_tying_points(3);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<3>(itying, pt);
        //   ety[offset + itying] = 0.5 * (A2D::VecDotCore<T, 3>(Xeta, d0) +
        //                                 A2D::VecDotCore<T, 3>(n0, Ueta));

        T Xxi[3], Xeta[3], n0[3];
        Basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi, Xeta);
        Basis::template interpFields<3, 3>(pt, fn, n0);

        A2D::Vec<T, 3> zero, d0_bar, Ueta_bar;
        A2D::VecAddCore<T, 3>(0.5 * ety_bar[offset + itying], Xeta, d0_bar.get_data());
        A2D::VecAddCore<T, 3>(0.5 * ety_bar[offset + itying], n0, Ueta_bar.get_data());

        // nonlinear g23 strain term backprop
        if constexpr (is_nonlinear) {
            T d0[3];
            Basis::template interpFields<3, 3>(pt, d, d0);
            T Uxi[3], Ueta[3];
            Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, Uxi, Ueta);
            // recall forward analysis nonlinear g23 term:
            // ety[offset + itying] += 0.5 * A2D::VecDotCore<T, 3>(d0, Ueta);
            A2D::VecAddCore<T, 3>(0.5 * ety_bar[offset + itying], Ueta, d0_bar.get_data());
            A2D::VecAddCore<T, 3>(0.5 * ety_bar[offset + itying], d0, Ueta_bar.get_data());
        }

        Basis::template interpFieldsGradTranspose<vars_per_node, 3>(pt, zero.get_data(),
                                                                    Ueta_bar.get_data(), res);
        Basis::template interpFieldsTranspose<3, 3>(pt, d0_bar.get_data(), d_bar);
    }  // end of itying for loop for g23

    // get g13 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(4);
    num_tying = Basis::num_tying_points(4);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<4>(itying, pt);
        //   ety[offset + itying] = 0.5 * (A2D::VecDotCore<T, 3>(Xxi, d0) +
        //                                 A2D::VecDotCore<T, 3>(n0, Uxi));

        T Xxi[3], Xeta[3], n0[3];
        Basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi, Xeta);
        Basis::template interpFields<3, 3>(pt, fn, n0);

        A2D::Vec<T, 3> zero, Uxi_bar, d0_bar;
        A2D::VecAddCore<T, 3>(0.5 * ety_bar[offset + itying], Xxi, d0_bar.get_data());
        A2D::VecAddCore<T, 3>(0.5 * ety_bar[offset + itying], n0, Uxi_bar.get_data());

        // nonlinear g23 strain term backprop
        if constexpr (is_nonlinear) {
            T d0[3];
            Basis::template interpFields<3, 3>(pt, d, d0);
            T Uxi[3], Ueta[3];
            Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, Uxi, Ueta);
            // recall forward analysis nonlinear g22 term:
            // ety[offset + itying] += 0.5 * A2D::VecDotCore<T, 3>(d0, Uxi);
            A2D::VecAddCore<T, 3>(0.5 * ety_bar[offset + itying], Uxi, d0_bar.get_data());
            A2D::VecAddCore<T, 3>(0.5 * ety_bar[offset + itying], d0, Uxi_bar.get_data());
        }

        Basis::template interpFieldsGradTranspose<vars_per_node, 3>(pt, Uxi_bar.get_data(),
                                                                    zero.get_data(), res);
        Basis::template interpFieldsTranspose<3, 3>(pt, d0_bar.get_data(), d_bar);

    }  // end of itying for loop for g13

}  // end of computeTyingStrainSens

template <typename T, class Physics, class Basis>
__HOST_DEVICE__ static void computeMITCTyingStrainHrev(const T Xpts[], const T fn[], const T vars[],
                                                       const T d[], const T p_vars[], const T p_d[],
                                                       const T ety_bar[], const T h_ety[],
                                                       T matCol[], T h_d[]) {
    // 2nd order backprop terms, linear part
    static constexpr bool is_nonlinear = Physics::is_nonlinear;
    computeMITCTyingStrainSens<T, Physics, Basis, is_nonlinear>(Xpts, fn, vars, d, h_ety, matCol,
                                                                h_d);

    if constexpr (!is_nonlinear) {
        return;
    }
    // return;  // temp debug

    // remaining part is nonlinear mixed term
    // ybar_i * d^2y_i/dxk/dxl * xdot_l mixed term with forward inputs here
    // only valid for nonlinear case
    int32_t offset, num_tying;
    static constexpr int vars_per_node = Physics::vars_per_node;

    // get g11 tying strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(0);
    num_tying = Basis::num_tying_points(0);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<0>(itying, pt);
        //   ety[offset + itying] NL = 1/2 * Uxi dot Uxi;
        // g11_bar * d^2g11/dUxi/dUxi * p_Uxi term

        T p_Uxi[3];
        A2D::Vec<T, 3> Uxi_hat, zero;
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, p_vars, p_Uxi, zero.get_data());
        // had 1/2 * ety_bar before but now I'm trying 1 since looks like sens above
        A2D::VecAddCore<T, 3>(ety_bar[offset + itying], p_Uxi, Uxi_hat.get_data());

        Basis::template interpFieldsGradTranspose<vars_per_node, 3>(pt, Uxi_hat.get_data(),
                                                                    zero.get_data(), matCol);

    }  // end of itying for loop for g11

    // return;  // temp debug

    // get g22 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(1);
    num_tying = Basis::num_tying_points(1);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<1>(itying, pt);
        //   ety[offset + itying] NL = 1/2 * Ueta dot Ueta;
        // g22_bar * d^2g22/dUeta/dUeta * p_Ueta term

        T p_Ueta[3];
        A2D::Vec<T, 3> Ueta_hat, zero;
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, p_vars, zero.get_data(), p_Ueta);
        // had 1/2 * ety_bar before but now I'm trying 1 since looks like sens above
        A2D::VecAddCore<T, 3>(ety_bar[offset + itying], p_Ueta, Ueta_hat.get_data());

        Basis::template interpFieldsGradTranspose<vars_per_node, 3>(pt, zero.get_data(),
                                                                    Ueta_hat.get_data(), matCol);

    }  // end of itying for loop for g22

    // get g12 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(2);
    num_tying = Basis::num_tying_points(2);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<2>(itying, pt);
        // ety[offset + itying] NL = 1/2 * Uxi dot Ueta
        // g12_bar * d^2g22/dUxi/dUeta * p_Uxi + again but swap Ueta,Uxi (term)

        T p_Uxi[3], p_Ueta[3];
        A2D::Vec<T, 3> Uxi_hat, Ueta_hat;
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, p_vars, p_Uxi, p_Ueta);
        A2D::VecAddCore<T, 3>(0.5 * ety_bar[offset + itying], p_Uxi, Ueta_hat.get_data());
        A2D::VecAddCore<T, 3>(0.5 * ety_bar[offset + itying], p_Ueta, Uxi_hat.get_data());

        Basis::template interpFieldsGradTranspose<vars_per_node, 3>(pt, Uxi_hat.get_data(),
                                                                    Ueta_hat.get_data(), matCol);
    }  // end of itying for loop for g12

    // get g23 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(3);
    num_tying = Basis::num_tying_points(3);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<3>(itying, pt);
        // ety[offset + itying] NL = 1/2 * d0 dot Ueta
        // g23_bar * d^2g22/dd0/dUeta * p_Ueta + again but swap Ueta,d0 (term)

        T p_Ueta[3];
        A2D::Vec<T, 3> d0_hat, Ueta_hat, zero;
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, p_vars, zero.get_data(), p_Ueta);
        T p_d0[3];
        Basis::template interpFields<3, 3>(pt, p_d, p_d0);

        A2D::VecAddCore<T, 3>(0.5 * ety_bar[offset + itying], p_d0, Ueta_hat.get_data());
        A2D::VecAddCore<T, 3>(0.5 * ety_bar[offset + itying], p_Ueta, d0_hat.get_data());

        Basis::template interpFieldsGradTranspose<vars_per_node, 3>(pt, zero.get_data(),
                                                                    Ueta_hat.get_data(), matCol);
        Basis::template interpFieldsTranspose<3, 3>(pt, d0_hat.get_data(), h_d);
    }  // end of itying for loop for g23

    // get g13 strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(4);
    num_tying = Basis::num_tying_points(4);
#pragma unroll  // for low num_tying can speed up?
    for (int itying = 0; itying < num_tying; itying++) {
        T pt[2];
        Basis::template getTyingPoint<4>(itying, pt);
        // ety[offset + itying] NL = 1/2 * d0 dot Uxi
        // g13_bar * d^2g22/dd0/dUxi * p_Uxi + again but swap Uxi,d0 (term)

        T p_Uxi[3];
        A2D::Vec<T, 3> d0_hat, Uxi_hat, zero;
        Basis::template interpFieldsGrad<vars_per_node, 3>(pt, p_vars, p_Uxi, zero.get_data());
        T p_d0[3];
        Basis::template interpFields<3, 3>(pt, p_d, p_d0);

        A2D::VecAddCore<T, 3>(0.5 * ety_bar[offset + itying], p_d0, Uxi_hat.get_data());
        A2D::VecAddCore<T, 3>(0.5 * ety_bar[offset + itying], p_Uxi, d0_hat.get_data());

        Basis::template interpFieldsGradTranspose<vars_per_node, 3>(pt, Uxi_hat.get_data(),
                                                                    zero.get_data(), matCol);
        Basis::template interpFieldsTranspose<3, 3>(pt, d0_hat.get_data(), h_d);

    }  // end of itying for loop for g13
}

// interp section
// ------------------

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
    offset = Basis::tying_point_offsets(0);
    constexpr int num_tying_g11 = Basis::num_tying_points(0);
    T N_g11[num_tying_g11];  // TODO : can we store less floats here?
    Basis::template getTyingInterp<0>(pt, N_g11);
    gty[0] = A2D::VecDotCore<T, num_tying_g11>(N_g11, &ety[offset]);

    // get g22 tying strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(1);
    constexpr int num_tying_g22 = Basis::num_tying_points(1);
    T N_g22[num_tying_g22];
    Basis::template getTyingInterp<1>(pt, N_g22);
    gty[3] = A2D::VecDotCore<T, num_tying_g22>(N_g22, &ety[offset]);

    // get g12 tying strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(2);
    constexpr int num_tying_g12 = Basis::num_tying_points(2);
    T N_g12[num_tying_g12];
    Basis::template getTyingInterp<2>(pt, N_g12);
    gty[1] = A2D::VecDotCore<T, num_tying_g12>(N_g12, &ety[offset]);

    // get g23 tying strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(3);
    constexpr int num_tying_g23 = Basis::num_tying_points(3);
    T N_g23[num_tying_g23];
    Basis::template getTyingInterp<3>(pt, N_g23);
    gty[4] = A2D::VecDotCore<T, num_tying_g23>(N_g23, &ety[offset]);

    // get g13 tying strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(4);
    constexpr int num_tying_g13 = Basis::num_tying_points(4);
    T N_g13[num_tying_g13];
    Basis::template getTyingInterp<4>(pt, N_g13);
    gty[2] = A2D::VecDotCore<T, num_tying_g13>(N_g13, &ety[offset]);

    // get g33 tying strain
    // --------------------
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

    // TODO : is this really the most efficient way to do this? Profile on GPU

    // get g11 tying strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(0);
    constexpr int num_tying_g11 = Basis::num_tying_points(0);
    T N_g11[num_tying_g11];  // TODO : can we store less floats here?
    Basis::template getTyingInterp<0>(pt, N_g11);
    A2D::VecAddCore<T, num_tying_g11>(gty_bar[0], N_g11, &ety_bar[offset]);

    // get g22 tying strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(1);
    constexpr int num_tying_g22 = Basis::num_tying_points(1);
    T N_g22[num_tying_g22];
    Basis::template getTyingInterp<1>(pt, N_g22);
    A2D::VecAddCore<T, num_tying_g22>(gty_bar[3], N_g22, &ety_bar[offset]);

    // get g12 tying strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(2);
    constexpr int num_tying_g12 = Basis::num_tying_points(2);
    T N_g12[num_tying_g12];
    Basis::template getTyingInterp<2>(pt, N_g12);
    A2D::VecAddCore<T, num_tying_g12>(gty_bar[1], N_g12, &ety_bar[offset]);

    // get g23 tying strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(3);
    constexpr int num_tying_g23 = Basis::num_tying_points(3);
    T N_g23[num_tying_g23];
    Basis::template getTyingInterp<3>(pt, N_g23);
    A2D::VecAddCore<T, num_tying_g23>(gty_bar[4], N_g23, &ety_bar[offset]);

    // get g13 tying strain
    // ------------------------------------
    offset = Basis::tying_point_offsets(4);
    constexpr int num_tying_g13 = Basis::num_tying_points(4);
    T N_g13[num_tying_g13];
    Basis::template getTyingInterp<4>(pt, N_g13);
    A2D::VecAddCore<T, num_tying_g13>(gty_bar[2], N_g13, &ety_bar[offset]);

    // get g33 tying strain
    // --------------------
    // zero so do nothing
}