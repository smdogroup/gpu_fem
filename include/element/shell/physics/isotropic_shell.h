#pragma once

#include "../a2d/a2disostress.h"
#include "../a2d/a2dshellstrain1.h"
#include "../data/isotropic.h"
#include "a2dcore.h"

template <typename T, class Data_, bool isNonlinear = false, bool hellingerReissner_ = false>
class IsotropicShell {
   public:
    using Data = Data_;
    static constexpr bool hellingerReissner = hellingerReissner_;

    // ensure the data is only allowed to be ShellIsotropicData
    // static_assert(std::is_same<Data_, ShellIsotropicData>::value,
    //               "Error IsotropicShell physics class must use Data class 'ShellIsotropicData'");

    // u, v, w, thx, thy, thz
    static constexpr int32_t vars_per_node = hellingerReissner ? 11 : 6;
    static constexpr int32_t std_vpn = 6;
    // whether strain is linear or nonlinear (in this case linear)
    static constexpr A2D::ShellStrainType STRAIN_TYPE =
        isNonlinear ? A2D::ShellStrainType::NONLINEAR : A2D::ShellStrainType::LINEAR;
    static constexpr bool is_nonlinear = isNonlinear;
    static constexpr int num_dvs = 1;

    __HOST_DEVICE__ static void computeTyingStress(const T &scale, const Data &data,
                                                   A2D::SymMat<T, 3> &e, A2D::SymMat<T, 3> &s) {
        /* compute membrane + trv shear stresses from the tying strains */

        T C[6];
        Data::evalTangentStiffness2D(data.E, data.nu, C);
        // [C[0], C[1], C[2]]
        // [C[1], C[3], C[4]]
        // [C[2], C[4], C[5]]

        // recall strain + stresses indices
        // 0 - e11, 1 - e12, 2 - e13, 3 - e22, 4 - e23, 5 - e33 (e33 = 0 cause inextensible)
        // membrane strains = [e[0], e[3], e[1]]
        // trv shear strains = [e[2], e[4]]

        // compute membrane stresses or in-plane loads, N = A * em
        s[0] = C[0] * e[0] + C[1] * e[3] + C[2] * e[1];
        s[3] = C[1] * e[0] + C[3] * e[3] + C[4] * e[1];
        s[1] = C[2] * e[0] + C[4] * e[3] + C[5] * e[1];
        s[0] *= data.thick * scale;
        s[3] *= data.thick * scale;
        s[1] *= data.thick * scale;

        // compute trv shear stresses, Q = As * es
        T As = Data::getTransShearCorrFactor() * data.thick * scale * C[5];
        s[2] = As * e[2];
        s[4] = As * e[4];
    }

    __HOST_DEVICE__ static void computeBendingStress(const T &scale, const Data &data,
                                                     const A2D::Vec<T, 3> &e, A2D::Vec<T, 3> &s) {
        /* compute membrane + trv shear stresses from the tying strains */

        T C[6];
        Data::evalTangentStiffness2D(data.E, data.nu, C);
        // [C[0], C[1], C[2]]
        // [C[1], C[3], C[4]]
        // [C[2], C[4], C[5]]

        // compute membrane stresses or in-plane loads, N = A * em
        s[0] = C[0] * e[0] + C[1] * e[1] + C[2] * e[2];
        s[1] = C[1] * e[0] + C[3] * e[1] + C[4] * e[2];
        s[2] = C[2] * e[0] + C[4] * e[1] + C[5] * e[2];
        T I = data.getPanelIzz();
        s[0] *= I * scale;
        s[1] *= I * scale;
        s[2] *= I * scale;
    }

    __HOST_DEVICE__ static void computeDrillStress(const T &scale, const Data &data, const T ed[1],
                                                   T sd[1]) {
        T G = data.E / 2.0 / (1.0 + data.nu);
        T As = Data::getTransShearCorrFactor() * G * data.thick;
        T drill = Data::getDrillingRegularization() * As;
        sd[0] = scale * drill * ed[0];
    }

    template <typename T2>
    __HOST_DEVICE__ static void computeStrainEnergy(const Data physData, const T scale,
                                                    A2D::ADObj<A2D::Mat<T2, 3, 3>> &u0x,
                                                    A2D::ADObj<A2D::Mat<T2, 3, 3>> &u1x,
                                                    A2D::ADObj<A2D::SymMat<T2, 3>> &e0ty,
                                                    A2D::ADObj<A2D::Vec<T2, 1>> &et,
                                                    A2D::ADObj<T2> &Uelem) {
        A2D::ADObj<A2D::Vec<T2, 9>> E, S;
        A2D::ADObj<T2> ES_dot;

        // use stack to compute shell strains, stresses and then to strain energy
        auto strain_energy_stack =
            A2D::MakeStack(A2D::ShellStrain<STRAIN_TYPE>(u0x, u1x, e0ty, et, E),
                           A2D::IsotropicShellStress<T, Data>(
                               physData.E, physData.nu, physData.thick, physData.tOffset, E, S),
                           // A2D::VecScale(1.0, E, S), // debugging statement
                           A2D::VecDot(E, S, ES_dot), A2D::Eval(T2(0.5 * scale) * ES_dot, Uelem));
        // printf("Uelem = %.8e\n", Uelem.value());

    }  // end of computeStrainEnergy

    // could template by ADType = ADObj or A2DObj later to allow different
    // derivative levels maybe
    template <typename T2>
    __HOST_DEVICE__ static void computeWeakRes(const Data &physData, const T &scale,
                                               A2D::ADObj<A2D::Mat<T2, 3, 3>> &u0x,
                                               A2D::ADObj<A2D::Mat<T2, 3, 3>> &u1x,
                                               A2D::ADObj<A2D::SymMat<T2, 3>> &e0ty,
                                               A2D::ADObj<A2D::Vec<T2, 1>> &et) {
        // using ADVec = A2D::ADObj<A2D::Vec<T2,9>>;
        A2D::ADObj<A2D::Vec<T2, 9>> E, S;
        A2D::ADObj<T2> ES_dot, Uelem;
        // isotropicShellStress expression uses many fewer floats than storing ABD
        // matrix

        // use stack to compute shell strains, stresses and then to strain energy
        auto strain_energy_stack =
            A2D::MakeStack(A2D::ShellStrain<STRAIN_TYPE>(u0x, u1x, e0ty, et, E),
                           A2D::IsotropicShellStress<T, Data>(
                               physData.E, physData.nu, physData.thick, physData.tOffset, E, S),
                           // no 1/2 here to match TACS formulation (just scales eqns) [is removing
                           // the 0.5 correct?]
                           A2D::VecDot(E, S, ES_dot), A2D::Eval(T2(scale) * ES_dot, Uelem));
        // printf("Uelem = %.8e\n", Uelem.value());

        Uelem.bvalue() = 1.0;
        strain_energy_stack.reverse();
        // bvalue outputs stored in u0x, u1x, e0ty, et and are backpropagated
    }  // end of computeWeakRes

    template <typename T2>
    __HOST_DEVICE__ static void computeWeakJacobianCol(const Data &physData, const T &scale,
                                                       A2D::A2DObj<A2D::Mat<T2, 3, 3>> &u0x,
                                                       A2D::A2DObj<A2D::Mat<T2, 3, 3>> &u1x,
                                                       A2D::A2DObj<A2D::SymMat<T2, 3>> &e0ty,
                                                       A2D::A2DObj<A2D::Vec<T2, 1>> &et) {
        // computes a projected Hessian (or jacobian column)

        // using ADVec = A2D::ADObj<A2D::Vec<T2,9>>;
        A2D::A2DObj<A2D::Vec<T2, 9>> E, S;
        A2D::A2DObj<T2> ES_dot, Uelem;

        // use stack to compute shell strains, stresses and then to strain energy
        auto strain_energy_stack =
            A2D::MakeStack(A2D::ShellStrain<STRAIN_TYPE>(u0x, u1x, e0ty, et, E),
                           A2D::IsotropicShellStress<T, Data>(
                               physData.E, physData.nu, physData.thick, physData.tOffset, E, S),
                           // no 1/2 here to match TACS formulation (just scales eqns) [is removing
                           // the 0.5 correct?]
                           A2D::VecDot(E, S, ES_dot), A2D::Eval(T2(scale) * ES_dot, Uelem));
        // note TACS differentiates based on 2 * Uelem here.. hmm
        // printf("Uelem = %.8e\n", Uelem.value());

        Uelem.bvalue() = 1.0;
        strain_energy_stack.hproduct();  // computes projected hessians
        // bvalue outputs stored in u0x, u1x, e0ty, et and are backpropagated
    }  // end of JacobianCol

    __HOST_DEVICE__ static void getMassMoments(const Data &physData, T moments[]) {
        // for mass residual + jacobian (unsteady analyses)
        const T &rho = physData.rho;
        const T &t = physData.thick;
        const T &tOffset = physData.tOffset;

        // taken from TACSIsoShellConstitutive.cpp (evalMassMoments)
        moments[0] = rho * t;
        moments[1] = -rho * t * t * tOffset;
        moments[2] = rho * t * t * t * (tOffset * tOffset + 1.0 / 12.0);
    }

    template <typename T2>
    __HOST_DEVICE__ static void computeQuadptStresses(const Data &physData, const T &scale,
                                                      A2D::ADObj<A2D::Mat<T2, 3, 3>> &u0x,
                                                      A2D::ADObj<A2D::Mat<T2, 3, 3>> &u1x,
                                                      A2D::ADObj<A2D::SymMat<T2, 3>> &e0ty,
                                                      A2D::ADObj<A2D::Vec<T2, 1>> &et,
                                                      A2D::ADObj<A2D::Vec<T2, 9>> &E,
                                                      A2D::ADObj<A2D::Vec<T2, 9>> &S) {
        // use stack to compute shell strains, stresses and then to strain energy
        auto strain_energy_stack =
            A2D::MakeStack(A2D::ShellStrain<STRAIN_TYPE>(u0x, u1x, e0ty, et, E),
                           A2D::IsotropicShellStress<T, Data>(
                               physData.E, physData.nu, physData.thick, physData.tOffset, E, S));
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