#pragma once

#include "../a2d/a2disostress.h"
#include "../a2d/a2dshellstrain1.h"
#include "../data/iso_stiff.h"
#include "a2dcore.h"
#include "isotropic_shell.h"

template <typename T, class Data_, bool isNonlinear = false, bool hellingerReissner_ = false>
class StiffenedIsotropicShell : public IsotropicShell<T, Data_, isNonlinear, hellingerReissner_> {
   public:
    using Data = Data_;
    using IsoShellClass = IsotropicShell<T, Data_, isNonlinear, hellingerReissner_>;
    static constexpr bool hellingerReissner = hellingerReissner_;

    // u, v, w, thx, thy, thz
    static constexpr int32_t vars_per_node = hellingerReissner ? 11 : 6;
    static constexpr int32_t std_vpn = 6;
    // whether strain is linear or nonlinear (in this case linear)
    static constexpr A2D::ShellStrainType STRAIN_TYPE =
        isNonlinear ? A2D::ShellStrainType::NONLINEAR : A2D::ShellStrainType::LINEAR;
    static constexpr bool is_nonlinear = isNonlinear;
    static constexpr int num_dvs = 4;

    __HOST_DEVICE__ static void computeTyingStress(const T &scale, const Data &data,
                                                   A2D::SymMat<T, 3> &e, A2D::SymMat<T, 3> &s) {
        /* compute membrane + trv shear stresses from the tying strains */

        /* 1) panel contributions to tying stress */
        IsoShellClass::computeTyingStress(scale, data, e, s);

        /* 2) stiffener stiffness contributions */
        // e11_stiff = E * A / stiffPitch
        T scaled_A11 = scale * data.getStiffenerArea() / data.stiffPitch;
        s[0] += scaled_A11 * e[0];

        // trv shear stiffnesses
        T scaled_A44 = Data::getTransShearCorrFactor() * scaled_A11;
        s[2] += scaled_A44 * e[2];
        s[4] += scaled_A44 * e[4];
    }

    __HOST_DEVICE__ static void computeBendingStress(const T &scale, const Data &data,
                                                     const A2D::Vec<T, 3> &e, A2D::Vec<T, 3> &s) {
        /* compute membrane + trv shear stresses from the tying strains */

        /* 1) panel bending stiffness contributions */

        // TODO : take about the centroid so B = 0, needs modification here
        // probably won't be able to call the previous method directly
        IsoShellClass::computeBendingStress(scale, data, e, s);

        /* 2) stiffener stiffness contributions */
        // M11 = E * I / sp * k11 (axial bending moment from bending stress)
        // TODO : also need modification so B = 0 here too (take about overall centroid)
        T stiff_D11 = data.E * data.getStiffenerI11();  // includes sp norm
        s[0] += scale * stiff_D11 * e[0];
    }

    __HOST_DEVICE__ static void computeDrillStress(const T &scale, const Data &data, const T ed[1],
                                                   T sd[1]) {
        // panel contribution to drill stress
        IsoShellClass::computeDrillStress(scale, data, ed, sd);

        // stiffener contribution to drill stress
        T G = data.E / 2.0 / (1.0 + data.nu);
        T ks_drill = Data::getTransShearCorrFactor() * Data::getDrillingRegularization();
        T stiff_A66 = scale * ks_drill * G * data.getStiffenerArea() / data.stiffPitch;
        sd[0] += scale * stiff_A66 * ed[0];
    }

    __HOST_DEVICE__ static void getMassMoments(const Data &physData, T moments[]) {
        // for mass residual + jacobian (unsteady analyses)

        // panel contribution to mass moments
        IsoShellClass::getMassMoments(physData, moments);

        // stiffener contribution to mass moments
        const T &rho = physData.rho;
        const T &sp = physData.stiffPitch;

        // TODO : add thick offset part to this also?
        // and take about centroid so B neq 0
        moments[0] += rho * physData.getStiffenerArea() / sp;
        moments[2] += rho * physData.getStiffenerI11();
    }

    template <typename T2>
    __HOST_DEVICE__ static void computeQuadptStresses(const Data &physData, const T &scale,
                                                      A2D::ADObj<A2D::Mat<T2, 3, 3>> &u0x,
                                                      A2D::ADObj<A2D::Mat<T2, 3, 3>> &u1x,
                                                      A2D::ADObj<A2D::SymMat<T2, 3>> &e0ty,
                                                      A2D::ADObj<A2D::Vec<T2, 1>> &et,
                                                      A2D::ADObj<A2D::Vec<T2, 9>> &E,
                                                      A2D::ADObj<A2D::Vec<T2, 9>> &S) {
        // call panel contribution
        IsoShellClass::computeQuadptStresses(physData, scale, u0x, u1x, e0ty, et, E, S);

        // TODO : add stiffener contributions here
        //   when not added (only affects visualized stresses, not strains), and no affect on
        //   optimization
    }  // end of computeQuadptStrains

    template <typename T2>
    __HOST_DEVICE__ static void compute_strain_adjoint_res_product(
        const Data &physData, const T &scale, const A2D::Mat<T2, 3, 3> &u0x,
        const A2D::Mat<T2, 3, 3> &u1x, const A2D::SymMat<T2, 3> &e0ty, const A2D::Vec<T2, 1> &et,
        const A2D::Mat<T2, 3, 3> &psi_u0x, const A2D::Mat<T2, 3, 3> &psi_u1x,
        const A2D::SymMat<T2, 3> &psi_e0ty, const A2D::Vec<T2, 1> &psi_et, T loc_dv_sens[]) {
        /* compute psi[E]^T * d^2Pi/dE/dx product at strain level (equiv to back at disp level) */
        A2D::Vec<T2, 9> E;
        A2D::Vec<T2, 9> psi_E;

        // first compute strains on the regular disps
        if constexpr (STRAIN_TYPE == A2D::ShellStrainType::LINEAR) {
            A2D::LinearShellStrainCore<T>(u0x.get_data(), u1x.get_data(), e0ty.get_data(),
                                          et.get_data(), E.get_data());
        } else {
            A2D::NonlinearShellStrainCore<T>(u0x.get_data(), u1x.get_data(), e0ty.get_data(),
                                             et.get_data(), E.get_data());
        }

        // then compute adjoint strains using (linearized hfwd)
        if constexpr (STRAIN_TYPE == A2D::ShellStrainType::LINEAR) {
            A2D::LinearShellStrainForwardCore<T>(psi_u0x.get_data(), psi_u1x.get_data(),
                                                 psi_e0ty.get_data(), psi_et.get_data(),
                                                 psi_E.get_data());
        } else {
            A2D::NonlinearShellStrainForwardCore<T>(
                u0x.get_data(), u1x.get_data(), e0ty.get_data(), et.get_data(), psi_u0x.get_data(),
                psi_u1x.get_data(), psi_e0ty.get_data(), psi_et.get_data(), psi_E.get_data());
        }

        // then compute psi_E^T dEbar/dx the strain sensitivity design var sens for each design var
        physData.evalStrainDVSensProduct(scale, E.get_data(), psi_E.get_data(), loc_dv_sens);

    }  // end of computeWeakRes

    template <typename T2>
    __HOST_DEVICE__ static void computeFailureIndex(const Data physData, A2D::Mat<T2, 3, 3> u0x,
                                                    A2D::Mat<T2, 3, 3> u1x, A2D::SymMat<T2, 3> e0ty,
                                                    A2D::Vec<T2, 1> et, const T &rhoKS,
                                                    const T &safetyFactor, T &fail_index) {
        A2D::Vec<T2, 9> E;

        if constexpr (STRAIN_TYPE == A2D::ShellStrainType::LINEAR) {
            A2D::LinearShellStrainCore<T>(u0x.get_data(), u1x.get_data(), e0ty.get_data(),
                                          et.get_data(), E.get_data());
        } else {
            A2D::NonlinearShellStrainCore<T>(u0x.get_data(), u1x.get_data(), e0ty.get_data(),
                                             et.get_data(), E.get_data());
        }

        fail_index = physData.evalFailure(rhoKS, safetyFactor, E.get_data());

    }  // end of computeFailureIndex

    template <typename T2>
    __HOST_DEVICE__ static void computeFailureIndexDVSens(
        const Data physData, A2D::Mat<T2, 3, 3> u0x, A2D::Mat<T2, 3, 3> u1x,
        A2D::SymMat<T2, 3> e0ty, A2D::Vec<T2, 1> et, const T &rhoKS, const T &safetyFactor,
        const T &scale, T dv_sens[]) {
        /* compute df/dx constribution to ks failure */
        A2D::Vec<T2, 9> E;
        if constexpr (STRAIN_TYPE == A2D::ShellStrainType::LINEAR) {
            A2D::LinearShellStrainCore<T>(u0x.get_data(), u1x.get_data(), e0ty.get_data(),
                                          et.get_data(), E.get_data());
        } else {
            A2D::NonlinearShellStrainCore<T>(u0x.get_data(), u1x.get_data(), e0ty.get_data(),
                                             et.get_data(), E.get_data());
        }

        physData.evalFailureDVSens(rhoKS, safetyFactor, E.get_data(), scale, dv_sens);

    }  // end of computeFailureIndexDVSens

    template <typename T2>
    __HOST_DEVICE__ static void computeFailureIndexSVSens(const Data physData, const T &rhoKS,
                                                          const T &safetyFactor, const T &scale,
                                                          A2D::ADObj<A2D::Mat<T2, 3, 3>> &u0x,
                                                          A2D::ADObj<A2D::Mat<T2, 3, 3>> &u1x,
                                                          A2D::ADObj<A2D::SymMat<T2, 3>> &e0ty,
                                                          A2D::ADObj<A2D::Vec<T2, 1>> &et) {
        /* compute df/du RHS constribution to ks failure */
        A2D::ADObj<A2D::Vec<T2, 9>> E;
        if constexpr (STRAIN_TYPE == A2D::ShellStrainType::LINEAR) {
            A2D::LinearShellStrainCore<T>(u0x.value().get_data(), u1x.value().get_data(),
                                          e0ty.value().get_data(), et.value().get_data(),
                                          E.value().get_data());
        } else {
            A2D::NonlinearShellStrainCore<T>(u0x.value().get_data(), u1x.value().get_data(),
                                             e0ty.value().get_data(), et.value().get_data(),
                                             E.value().get_data());
        }

        // go from E forward to E backwards (the strain)
        physData.evalFailureStrainSens(scale, rhoKS, safetyFactor, E.value().get_data(),
                                       E.bvalue().get_data());

        if constexpr (STRAIN_TYPE == A2D::ShellStrainType::LINEAR) {
            A2D::LinearShellStrainReverseCore<T>(E.bvalue().get_data(), u0x.bvalue().get_data(),
                                                 u1x.bvalue().get_data(), e0ty.bvalue().get_data(),
                                                 et.bvalue().get_data());
        } else {
            A2D::NonlinearShellStrainReverseCore<T>(
                E.bvalue().get_data(), u0x.value().get_data(), u1x.value().get_data(),
                e0ty.value().get_data(), et.value().get_data(), u0x.bvalue().get_data(),
                u1x.bvalue().get_data(), e0ty.bvalue().get_data(), et.bvalue().get_data());
        }
    }  // end of computeFailureIndexDVSens
};
