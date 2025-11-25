#pragma once
#include "../../../cuda_utils.h"

/* the tying strains include the 3 membrane and 2 transverse shear strains (5 of them in total) */

// the term 'full' tying strains, means all 5 tying strains are simply computed at the desired
// quadpt for full integration this is for the fully integrated element

// NOTE : only the forward + sens methods have is_nonlinear separate from the physics is_nonlinear
// this is so that we are free to call the linear part in Hfwd and Hrev

template <typename T, class Physics, class Basis, bool is_nonlinear>
__DEVICE__ static void computeFullTyingStrain(const T pt[], const T Xpts[], const T fn[],
                                              const T vars[], const T d[], T gty[]) {
    static constexpr int vars_per_node = Physics::std_vpn;

    // Interpolate the field values
    T Uxi[3], Ueta[3], Xxi[3], Xeta[3];
    Basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi, Xeta);
    Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, Uxi, Ueta);

    // 3 membrane strains g11, g22, g12
    // ---------------------------------

    // g11 strain
    gty[0] = A2D::VecDotCore<T, 3>(Uxi, Xxi);
    if constexpr (is_nonlinear) {
        gty[0] += 0.5 * A2D::VecDotCore<T, 3>(Uxi, Uxi);
    }

    // g22 strain
    gty[3] = A2D::VecDotCore<T, 3>(Ueta, Xeta);
    if constexpr (is_nonlinear) {
        gty[3] += 0.5 * A2D::VecDotCore<T, 3>(Ueta, Ueta);
    }

    // g12 strain
    gty[1] = 0.5 * (A2D::VecDotCore<T, 3>(Uxi, Xeta) + A2D::VecDotCore<T, 3>(Ueta, Xxi));
    if constexpr (is_nonlinear) {
        gty[1] += 0.5 * A2D::VecDotCore<T, 3>(Uxi, Ueta);
    }
    __syncthreads();

    // 2 transverse shear strains (and g33 = 0)
    //    cause inextensible director (by defn)
    // ----------------------------------------

    T d0[3], n0[3];
    Basis::template interpFields<3, 3>(pt, d, d0);
    Basis::template interpFields<3, 3>(pt, fn, n0);

    // g23 strain
    gty[4] = 0.5 * (A2D::VecDotCore<T, 3>(Xeta, d0) + A2D::VecDotCore<T, 3>(n0, Ueta));
    if constexpr (is_nonlinear) {
        gty[4] += 0.5 * A2D::VecDotCore<T, 3>(d0, Ueta);
    }

    // g13 strain
    gty[2] = 0.5 * (A2D::VecDotCore<T, 3>(Xxi, d0) + A2D::VecDotCore<T, 3>(n0, Uxi));
    if constexpr (is_nonlinear) {
        gty[2] += 0.5 * A2D::VecDotCore<T, 3>(d0, Uxi);
    }

    // g33 strain
    gty[5] = 0.0;

}  // end of computeFullTyingStrain

template <typename T, class Physics, class Basis>
__DEVICE__ static void computeFullTyingStrainHfwd(const T pt[], const T Xpts[], const T fn[],
                                                  const T vars[], const T d[], const T p_vars[],
                                                  const T p_d[], T p_gty[]) {
    // linear part
    computeFullTyingStrain<T, Physics, Basis, false>(pt, Xpts, fn, p_vars, p_d, p_gty);

    static constexpr bool is_nonlinear = Physics::is_nonlinear;
    if constexpr (!is_nonlinear) {
        return;  // exit early for linear part
    }
    // otherwise continue and get nonlinear part of the derivatives
    __syncthreads();

    static constexpr int vars_per_node = Physics::std_vpn;

    // Interpolate the field values
    T Uxi[3], Ueta[3], p_Uxi[3], p_Ueta[3];
    Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, Uxi, Ueta);
    Basis::template interpFieldsGrad<vars_per_node, 3>(pt, p_vars, p_Uxi, p_Ueta);

    // 3 membrane strains g11, g22, g12
    // ---------------------------------

    // g11 strain
    p_gty[0] += A2D::VecDotCore<T, 3>(Uxi, p_Uxi);

    // g22 strain
    p_gty[3] += A2D::VecDotCore<T, 3>(Ueta, p_Ueta);

    // g12 strain
    p_gty[1] += 0.5 * A2D::VecDotCore<T, 3>(Uxi, p_Ueta);
    p_gty[1] += 0.5 * A2D::VecDotCore<T, 3>(Ueta, p_Uxi);

    __syncthreads();

    // 2 transverse shear strains (and g33 = 0)
    //    cause inextensible director (by defn)
    // ----------------------------------------

    T d0[3], p_d0[3];
    Basis::template interpFields<3, 3>(pt, d, d0);
    Basis::template interpFields<3, 3>(pt, p_d, p_d0);

    // g23 strain
    p_gty[4] += 0.5 * A2D::VecDotCore<T, 3>(d0, p_Ueta);
    p_gty[4] += 0.5 * A2D::VecDotCore<T, 3>(p_d0, Ueta);

    // g13 strain
    p_gty[2] += 0.5 * A2D::VecDotCore<T, 3>(d0, p_Uxi);
    p_gty[2] += 0.5 * A2D::VecDotCore<T, 3>(p_d0, Uxi);

    // g33 strain = 0.0

}  // end of computeFullTyingStrainHfwd

template <typename T, class Physics, class Basis, bool is_nonlinear>
__DEVICE__ static void computeFullTyingStrainSens(const T pt[], const T Xpts[], const T fn[],
                                                  const T vars[], const T d[], const T gty_bar[],
                                                  T res[], T d_bar[]) {
    static constexpr int vars_per_node = Physics::std_vpn;

    // Interpolate the field values
    T Uxi[3], Ueta[3], Xxi[3], Xeta[3];
    Basis::template interpFieldsGrad<3, 3>(pt, Xpts, Xxi, Xeta);
    Basis::template interpFieldsGrad<vars_per_node, 3>(pt, vars, Uxi, Ueta);
    __syncthreads();

    // backprop senses for interp
    A2D::Vec<T, 3> Uxi_bar, Ueta_bar, d0_bar;

    // 3 membrane strains g11, g22, g12
    // ---------------------------------

    // g11 strain
    A2D::VecAddCore<T, 3>(gty_bar[0], Xxi, Uxi_bar.get_data());
    if constexpr (is_nonlinear) {
        A2D::VecAddCore<T, 3>(gty_bar[0], Uxi, Uxi_bar.get_data());
    }

    // g22 strain
    A2D::VecAddCore<T, 3>(gty_bar[3], Xeta, Ueta_bar.get_data());
    if constexpr (is_nonlinear) {
        A2D::VecAddCore<T, 3>(gty_bar[3], Ueta, Ueta_bar.get_data());
    }

    // g12 strain
    A2D::VecAddCore<T, 3>(0.5 * gty_bar[1], Xxi, Ueta_bar.get_data());
    A2D::VecAddCore<T, 3>(0.5 * gty_bar[1], Xeta, Uxi_bar.get_data());
    if constexpr (is_nonlinear) {
        A2D::VecAddCore<T, 3>(0.5 * gty_bar[1], Uxi, Ueta_bar.get_data());
        A2D::VecAddCore<T, 3>(0.5 * gty_bar[1], Ueta, Uxi_bar.get_data());
    }
    __syncthreads();

    // 2 transverse shear strains (and g33 = 0)
    //    cause inextensible director (by defn)
    // ----------------------------------------

    T d0[3], n0[3];
    Basis::template interpFields<3, 3>(pt, d, d0);
    Basis::template interpFields<3, 3>(pt, fn, n0);

    // g23 strain
    A2D::VecAddCore<T, 3>(0.5 * gty_bar[4], Xeta, d0_bar.get_data());
    A2D::VecAddCore<T, 3>(0.5 * gty_bar[4], n0, Ueta_bar.get_data());
    if constexpr (is_nonlinear) {
        A2D::VecAddCore<T, 3>(0.5 * gty_bar[4], Ueta, d0_bar.get_data());
        A2D::VecAddCore<T, 3>(0.5 * gty_bar[4], d0, Ueta_bar.get_data());
    }

    // g13 strain
    A2D::VecAddCore<T, 3>(0.5 * gty_bar[2], Xxi, d0_bar.get_data());
    A2D::VecAddCore<T, 3>(0.5 * gty_bar[2], n0, Uxi_bar.get_data());
    if constexpr (is_nonlinear) {
        A2D::VecAddCore<T, 3>(0.5 * gty_bar[2], Uxi, d0_bar.get_data());
        A2D::VecAddCore<T, 3>(0.5 * gty_bar[2], d0, Uxi_bar.get_data());
    }

    __syncthreads();

    // g33 strain == 0

    // final backprop from interp levels to element residual
    // -----------------------------------------------------

    // backprop for Uxi, Ueta interp step
    Basis::template interpFieldsGradTranspose<vars_per_node, 3>(pt, Uxi_bar.get_data(),
                                                                Ueta_bar.get_data(), res);

    // backprop for d0 interp step
    Basis::template interpFieldsTranspose<3, 3>(pt, d0_bar.get_data(), d_bar);

}  // end of computeFullTyingStrainSens

template <typename T, class Physics, class Basis>
__DEVICE__ static void computeFullTyingStrainHrev(const T pt[], const T Xpts[], const T fn[],
                                                  const T vars[], const T d[], const T p_vars[],
                                                  const T p_d[], const T gty_bar[], const T h_gty[],
                                                  T matCol[], T h_d[]) {
    // 2nd order backprop terms, linear part
    static constexpr bool is_nonlinear = Physics::is_nonlinear;
    computeFullTyingStrainSens<T, Physics, Basis, is_nonlinear>(pt, Xpts, fn, vars, d, h_gty,
                                                                matCol, h_d);

    if constexpr (!is_nonlinear) {
        return;
    }
    __syncthreads();
    // remaining part is nonlinear mixed term
    // ybar_i * d^2y_i/dxk/dxl * xdot_l mixed term with forward inputs here
    // only valid for nonlinear case (aka uses gradient and second derivs of this step, nonlinear
    // part)

    static constexpr int vars_per_node = Physics::std_vpn;

    // Interpolate the field values
    T p_Uxi[3], p_Ueta[3];
    Basis::template interpFieldsGrad<vars_per_node, 3>(pt, p_vars, p_Uxi, p_Ueta);

    // backprop projected hessians for interp
    A2D::Vec<T, 3> Uxi_hat, Ueta_hat, d0_hat;

    // 3 membrane strains g11, g22, g12
    // ---------------------------------

    // g11 strain
    A2D::VecAddCore<T, 3>(gty_bar[0], p_Uxi, Uxi_hat.get_data());

    // g22 strain
    A2D::VecAddCore<T, 3>(gty_bar[3], p_Ueta, Ueta_hat.get_data());

    // g12 strain
    A2D::VecAddCore<T, 3>(0.5 * gty_bar[1], p_Uxi, Ueta_hat.get_data());
    A2D::VecAddCore<T, 3>(0.5 * gty_bar[1], p_Ueta, Uxi_hat.get_data());

    __syncthreads();

    // 2 transverse shear strains (and g33 = 0)
    //    cause inextensible director (by defn)
    // ----------------------------------------

    T p_d0[3];
    Basis::template interpFields<3, 3>(pt, p_d, p_d0);

    // g23 strain
    A2D::VecAddCore<T, 3>(0.5 * gty_bar[4], p_d0, Ueta_hat.get_data());
    A2D::VecAddCore<T, 3>(0.5 * gty_bar[4], p_Ueta, d0_hat.get_data());

    // g13 strain
    A2D::VecAddCore<T, 3>(0.5 * gty_bar[2], p_d0, Uxi_hat.get_data());
    A2D::VecAddCore<T, 3>(0.5 * gty_bar[2], p_Uxi, d0_hat.get_data());

    // g33 strain == 0

    // final backprop from interp levels to element residual
    // -----------------------------------------------------

    // backprop for Uxi, Ueta interp step
    Basis::template interpFieldsGradTranspose<vars_per_node, 3>(pt, Uxi_hat.get_data(),
                                                                Ueta_hat.get_data(), matCol);

    // backprop for d0 interp step
    Basis::template interpFieldsTranspose<3, 3>(pt, d0_hat.get_data(), h_d);
}