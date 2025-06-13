#pragma once
#include "a2dcore.h"
#include "cuda_utils.h"

// could be an internal class for each type of physics
// but I would like physics to be linear vs nonlinear and this class has methods which get ABD, etc.
template <typename T, bool has_ref_axis_ = true>
class ShellIsotropicDataV2 {
   public:
    // static constexpr double transverseShearCorrectionFactor = 5.0 / 6.0;
    // static constexpr double drillingRegularization = 10.0;  // default drilling regularization

    ShellIsotropicDataV2() = default;
    static constexpr bool has_ref_axis = has_ref_axis_;

    // constructor with ref Axis
    template <bool U = has_ref_axis, typename std::enable_if<U, int>::type = 0>
    __HOST_DEVICE__ ShellIsotropicDataV2(T E_, T nu_, T thick_, T refAxis_[], T ys_ = 1.0,
                                         T rho_ = 1.0, T tOffset_ = 0.0)
        : E(E_), nu(nu_), thick(thick_), ys(ys_), rho(rho_), tOffset(tOffset_) {
        // deep copy the ref axis
        for (int i = 0; i < 3; i++) {
            refAxis[i] = refAxis_[i];
        }
    }

    // constructor without refAxis input
    template <bool U = has_ref_axis, typename std::enable_if<!U, int>::type = 0>
    __HOST_DEVICE__ ShellIsotropicDataV2(T E_, T nu_, T thick_, T ys_ = 1.0, T rho_ = 1.0,
                                         T tOffset_ = 0.0)
        : E(E_), nu(nu_), thick(thick_), ys(ys_), rho(rho_), tOffset(tOffset_) {}

    // constitutive methods
    __HOST_DEVICE__ static void evalTangentStiffness2D(const T E_, const T nu_, T C[]) {
        // isotropic C matrix for stress = C * strain in-plane with no twist
        T D = E_ / (1.0 - nu_ * nu_);
        C[0] = D;
        C[1] = nu_ * D;
        C[2] = 0.0;
        C[3] = D;
        C[4] = 0.0;
        T G = E_ / 2.0 / (1.0 + nu_);
        C[5] = G;
    }

    __HOST_DEVICE__ static void stiffnessMatrixProd(const T E_, const T nu_, const T scale,
                                                    const T strain[], T stress[]) {
        T D = E_ / (1.0 - nu_ * nu_);
        // sym mat prod
        stress[0] = D * strain[0] + nu_ * D * strain[1];  // e11 => s11
        stress[1] = nu_ * D * strain[0] + D * strain[1];  // e22 => s22
        T G = E_ / 2.0 / (1.0 + nu_);
        stress[2] = G * strain[2];  // e12 => s12
        for (int i = 0; i < 3; i++) stress[i] *= scale;
    }

    __HOST_DEVICE__ static T getTransShearCorrFactor() { return T(5.0 / 6.0); }
    __HOST_DEVICE__ static T getDrillingRegularization() { return T(10.0); }

    // TODO : add thickness offset (rarely used but to be complete.. can allow B matrix to be
    // nonzero)
    T E, nu, thick, tOffset, ys, rho;
    T refAxis[3];
};