#pragma once
#include "../../../cuda_utils.h"

template <typename T, class Basis>
__HOST_DEVICE__ T getTransformMatrix(const T pt[2], const bool bndry[4], const T xpts[],
                                     T Xdinv[4]) {
    // first compute dx/xi, dy/deta etc using basis interp
    T Xxi[3], Xeta[3];
    Basis::template interpFieldsGrad<3, 3>(pt, bndry, xpts, Xxi, Xeta);
    T Xd[4];  // d(x,y)/d(xi,eta) matrix
    Xd[0] = Xxi[0], Xd[2] = Xeta[0];
    Xd[1] = Xxi[1], Xd[3] = Xeta[1];
    // now compute inverse matrix and determinant
    T detJ = Xd[0] * Xd[3] - Xd[1] * Xd[2];
    // T Xdinv[4];
    Xdinv[0] = Xd[3] / detJ;
    Xdinv[1] = -Xd[1] / detJ;
    Xdinv[2] = -Xd[2] / detJ;
    Xdinv[3] = Xd[0] / detJ;
    // get scale from quadpt and area + transform scale
    return detJ;
}

// compute bending strains
// -----------------------------
template <typename T, int vars_per_node, class Basis>
__HOST_DEVICE__ void computeBendingStrain(const T pt[2], const bool bndry[4], const T Xdinv[4],
                                          const T vars[], A2D::Vec<T, 3> &ek) {
    // compute rotation gradients in comp coords (first three values are u,v,w are ignored and
    // compiled away?)
    T Uxi[vars_per_node], Ueta[vars_per_node];
    Basis::template interpFieldsGrad<vars_per_node, vars_per_node>(pt, bndry, vars, Uxi, Ueta);

    // d(thx)/dx
    ek[0] = Xdinv[0] * Uxi[3] + Xdinv[2] * Ueta[3];
    // d(thy)/dy
    ek[1] = Xdinv[1] * Uxi[4] + Xdinv[3] * Ueta[4];
    // d(thx)/dy + d(thy)/dx
    ek[2] = Xdinv[1] * Uxi[3] + Xdinv[3] * Ueta[3];
    ek[2] += Xdinv[0] * Uxi[4] + Xdinv[2] * Ueta[4];
}

template <typename T, int vars_per_node, class Basis>
__HOST_DEVICE__ void computeBendingStrainTranspose(const T pt[2], const bool bndry[4],
                                                   const T Xdinv[4], A2D::Vec<T, 3> &ek_bar,
                                                   T res[]) {
    // adjoints of Uxi and Ueta
    T xi_bar[vars_per_node] = {0};
    T eta_bar[vars_per_node] = {0};

    // ek[0] = d(thx)/dx
    xi_bar[3] += Xdinv[0] * ek_bar[0];
    eta_bar[3] += Xdinv[2] * ek_bar[0];

    // ek[1] = d(thy)/dy
    xi_bar[4] += Xdinv[1] * ek_bar[1];
    eta_bar[4] += Xdinv[3] * ek_bar[1];

    // ek[2] = d(thx)/dy + d(thy)/dx
    xi_bar[3] += Xdinv[1] * ek_bar[2];
    eta_bar[3] += Xdinv[3] * ek_bar[2];

    xi_bar[4] += Xdinv[0] * ek_bar[2];
    eta_bar[4] += Xdinv[2] * ek_bar[2];

    // push adjoints back through interpolation
    Basis::template interpFieldsGradTranspose<vars_per_node, vars_per_node>(pt, bndry, xi_bar,
                                                                            eta_bar, res);
}

// compute membrane + trv shear tying strains
template <typename T, int vars_per_node, class Basis>
__DEVICE__ static void computeFullTyingStrain(const T pt[2], const bool bndry[4], const T xpts[],
                                              const T vars[], T gty[6]) {
    // Interpolate the field values
    T Uxi[3], Ueta[3], Xxi[3], Xeta[3];
    Basis::template interpFieldsGrad<3, 3>(pt, bndry, xpts, Xxi, Xeta);
    Basis::template interpFieldsGrad<vars_per_node, 3>(pt, bndry, vars, Uxi, Ueta);

    // no nonlinear allowed yet.. (do AIG shell for that)

    // 3 membrane strains g11, g22, g12
    // ---------------------------------

    // g11 strain
    gty[0] = A2D::VecDotCore<T, 3>(Uxi, Xxi);

    // g22 strain
    gty[3] = A2D::VecDotCore<T, 3>(Ueta, Xeta);

    // g12 strain
    gty[1] = 0.5 * (A2D::VecDotCore<T, 3>(Uxi, Xeta) + A2D::VecDotCore<T, 3>(Ueta, Xxi));

    // 2 transverse shear strains (and g33 = 0)
    //    cause inextensible director (by defn)
    // ----------------------------------------

    T U[vars_per_node];
    Basis::template interpFields<vars_per_node, vars_per_node>(pt, vars, U);

    // g23 strain
    gty[4] = 0.5 * (Xeta[1] * U[4] + Ueta[2]);

    // g13 strain
    gty[2] = 0.5 * (Xxi[0] * U[3] + Uxi[2]);

    // g33 strain
    gty[5] = 0.0;

}  // end of computeFullTyingStrain

template <typename T, int vars_per_node, class Basis>
__DEVICE__ static void computeFullTyingStrainSens(const T pt[], const bool bndry[4], const T xpts[],
                                                  const T gty_bar[], T res[]) {
    // Interpolate the field values
    T Xxi[3], Xeta[3];
    Basis::template interpFieldsGrad<3, 3>(pt, bndry, xpts, Xxi, Xeta);

    // backprop senses for interp
    A2D::Vec<T, 3> Uxi_bar, Ueta_bar, d0_bar;

    // 3 membrane strains g11, g22, g12
    // ---------------------------------

    // g11 strain
    A2D::VecAddCore<T, 3>(gty_bar[0], Xxi, Uxi_bar.get_data());

    // g22 strain
    A2D::VecAddCore<T, 3>(gty_bar[3], Xeta, Ueta_bar.get_data());

    // g12 strain
    A2D::VecAddCore<T, 3>(0.5 * gty_bar[1], Xxi, Ueta_bar.get_data());
    A2D::VecAddCore<T, 3>(0.5 * gty_bar[1], Xeta, Uxi_bar.get_data());

    // 2 transverse shear strains (and g33 = 0)
    //    cause inextensible director (by defn)
    // ----------------------------------------

    T U_bar[6] = {0};

    // g23 strain
    U_bar[4] += Xeta[1] * 0.5 * gty_bar[4];
    Ueta_bar[2] += 0.5 * gty_bar[4];

    // g13 strain
    U_bar[3] += 0.5 * gty_bar[2] * Xxi[0];
    Uxi_bar[2] += 0.5 * gty_bar[2];

    // g33 strain == 0

    // final backprop from interp levels to element residual
    // -----------------------------------------------------

    // backprop for Uxi, Ueta interp step
    Basis::template interpFieldsGradTranspose<vars_per_node, 3>(pt, bndry, Uxi_bar.get_data(),
                                                                Ueta_bar.get_data(), res);

    // backprop for d0 interp step
    Basis::template interpFieldsTranspose<vars_per_node, vars_per_node>(pt, bndry, U_bar, res);

}  // end of computeFullTyingStrainSens

template <typename T, int vars_per_node, class Basis>
__DEVICE__ static void computeDrillStrain(const T pt[2], const bool bndry[4], const T Xdinv[4],
                                          const T vars[], T et[1]) {
    // Interpolate the field values
    T Uxi[3], Ueta[3], U[vars_per_node];
    Basis::template interpFields<vars_per_node, vars_per_node>(pt, vars, U);
    Basis::template interpFieldsGrad<vars_per_node, 3>(pt, bndry, vars, Uxi, Ueta);

    T du_dy = Xdinv[0] * Uxi[0] + Xdinv[2] * Ueta[0];
    T dv_dx = Xdinv[1] * Uxi[1] + Xdinv[3] * Ueta[1];

    et[0] = U[5] - 0.5 * (du_dy - dv_dx);
}

template <typename T, int vars_per_node, class Basis>
__DEVICE__ static void computeDrillStrainTranspose(const T pt[2], const bool bndry[4],
                                                   const T Xdinv[4], const T et_bar[1],
                                                   T vars_bar[]) {
    // Adjoint variables
    T U_bar[vars_per_node] = {0.0};
    T Uxi_bar[3] = {0.0, 0.0, 0.0};
    T Ueta_bar[3] = {0.0, 0.0, 0.0};

    const T scale = et_bar[0];

    // et = U[5] - 0.5*(du_dy - dv_dx)
    U_bar[5] += scale;

    const T d_du_dy = -0.5 * scale;
    const T d_dv_dx = +0.5 * scale;

    // du_dy = Xdinv[0]*Uxi[0] + Xdinv[2]*Ueta[0]
    Uxi_bar[0] += Xdinv[0] * d_du_dy;
    Ueta_bar[0] += Xdinv[2] * d_du_dy;

    // dv_dx = Xdinv[1]*Uxi[1] + Xdinv[3]*Ueta[1]
    Uxi_bar[1] += Xdinv[1] * d_dv_dx;
    Ueta_bar[1] += Xdinv[3] * d_dv_dx;

    // Push adjoints back through basis operators
    Basis::template interpFieldsTranspose<vars_per_node, vars_per_node>(pt, U_bar, vars_bar);

    Basis::template interpFieldsGradTranspose<vars_per_node, 3>(pt, bndry, Uxi_bar, Ueta_bar,
                                                                vars_bar);
}