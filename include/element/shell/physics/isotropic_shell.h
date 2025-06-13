#pragma once

#include "../a2dshell.h"
#include "../data/isotropic.h"
#include "a2dcore.h"

template <typename T, class Data_, bool isNonlinear = false>
class IsotropicShell {
   public:
    using Data = Data_;

    // ensure the data is only allowed to be ShellIsotropicData
    // static_assert(std::is_same<Data_, ShellIsotropicData>::value,
    //               "Error IsotropicShell physics class must use Data class 'ShellIsotropicData'");

    // u, v, w, thx, thy, thz
    static constexpr int32_t vars_per_node = 6;
    // whether strain is linear or nonlinear (in this case linear)
    static constexpr A2D::ShellStrainType STRAIN_TYPE =
        isNonlinear ? A2D::ShellStrainType::NONLINEAR : A2D::ShellStrainType::LINEAR;
    static constexpr bool is_nonlinear = isNonlinear;
    static constexpr int num_dvs = 1;

    template <typename T2>
    __HOST_DEVICE__ static void computeStrainEnergy(const Data physData, const T scale,
                                                    A2D::ADObj<A2D::Mat<T2, 3, 3>> u0x,
                                                    A2D::ADObj<A2D::Mat<T2, 3, 3>> u1x,
                                                    A2D::ADObj<A2D::SymMat<T2, 3>> e0ty,
                                                    A2D::ADObj<A2D::Vec<T2, 1>> et,
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

        // printf("ek_f:");
        // printVec<T>(3, &E.value().get_data()[3]);

        Uelem.bvalue() = 1.0;
        strain_energy_stack.reverse();

        // printf("ek_b:");
        // printVec<T>(3, &E.bvalue().get_data()[3]);
        // bvalue outputs stored in u0x, u1x, e0ty, et and are backpropagated
    }  // end of computeWeakRes

    template <typename T2>
    __HOST_DEVICE__ static void compute_drill_strain_grad(const Data &physData, const T &scale,
                                                          A2D::ADObj<A2D::Vec<T2, 1>> &et) {
        // compute drilling stiffness
        T drill;
        {  // TODO : could just compute G here separately.., less data
            T C[6], E = physData.E, nu = physData.nu, thick = physData.thick;
            Data::evalTangentStiffness2D(E, nu, C);
            T As = Data::getTransShearCorrFactor() * thick * C[5];
            drill = Data::getDrillingRegularization() * As;
        }

        // or I could just do this..
        T2 drill_strain = et.value()[0];
        T2 drill_stress = drill * drill_strain;

        et.bvalue()[0] = scale * drill_stress;  // backprop from strain energy

        // can also use stack, but not really necessary for this one
    }

    template <typename T2>
    __HOST_DEVICE__ static void compute_tying_strain_midplane_grad(
        const Data &physData, const T &scale, A2D::ADObj<A2D::SymMat<T, 3>> &e0ty) {
        /* compute gradient of energy term with midplane strains */

        // compute the A matrix * strain (assume B = 0, sym laminate)
        T E = physData.E, nu = physData.nu, thick = physData.thick;
        T e0[3], s0[3];  // midplane stress and strains
        A2D::SymMat<T, 3> &e0ty_f = e0ty.value();
        e0[0] = e0ty_f[0];        // e11
        e0[1] = e0ty_f[3];        // e22
        e0[2] = 2.0 * e0ty_f[1];  // e12
        // nonlinearity is computed during "computeTyingStrain" not here

        // Nij = A * Eij; A = C * thick
        Data::stiffnessMatrixProd(E, nu, thick, e0, s0);
        // now put in bvalue() as the scale*stres is the midplane energy gradient
        A2D::SymMat<T, 3> &e0ty_b = e0ty.bvalue();
        e0ty_b[0] = scale * s0[0];
        e0ty_b[3] = scale * s0[1];
        e0ty_b[1] = 2.0 * scale * s0[2];
        //  this scale*stress is the midplane energy gradient
    }

    template <typename T2>
    __HOST_DEVICE__ static void compute_tying_strain_transverse_grad(
        const Data &physData, const T &scale, A2D::ADObj<A2D::SymMat<T, 3>> &e0ty) {
        /* compute gradient of energy term with transverse shear strains */
        T As;
        {
            T C[6], E = physData.E, nu = physData.nu, thick = physData.thick;
            Data::evalTangentStiffness2D(E, nu, C);
            As = Data::getTransShearCorrFactor() * thick * C[5];
        }
        A2D::SymMat<T, 3> &e0ty_f = e0ty.value();
        A2D::SymMat<T, 3> &e0ty_b = e0ty.bvalue();

        // scale * stress is the gradient of e0ty in these entries
        e0ty_b[2] += 4.0 * scale * As * e0ty_f[2];  // e13, transverse shear
        e0ty_b[4] += 4.0 * scale * As * e0ty_f[4];  // e23, transverse shear
        // can also use stack, but not really necessary for this one
    }

    template <typename T2>
    __HOST_DEVICE__ static void compute_bending_strain_grad(const Data &physData, const T &scale,
                                                            A2D::ADObj<A2D::Vec<T, 3>> &ek) {
        /* compute gradient of energy term with midplane strains */

        // compute the D matrix * strain (assume B = 0, sym laminate)
        T E = physData.E, nu = physData.nu, thick = physData.thick;
        T I = thick * thick * thick / 12.0;
        T *ek_f = ek.value().get_data();
        T *ek_b = ek.bvalue().get_data();

        // Mi = D * ek_i; D = C * I
        Data::stiffnessMatrixProd(E, nu, I * scale, ek_f, ek_b);
    }

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
<<<<<<< HEAD
};
=======

    template <typename T2>
    __HOST_DEVICE__ static void computeWeakResThickDVSens(const Data &physData, const T &scale,
                                                          A2D::ADObj<A2D::Mat<T2, 3, 3>> &u0x,
                                                          A2D::ADObj<A2D::Mat<T2, 3, 3>> &u1x,
                                                          A2D::ADObj<A2D::SymMat<T2, 3>> &e0ty,
                                                          A2D::ADObj<A2D::Vec<T2, 1>> &et) {
        // manually for linear isotropic case for now
        A2D::Vec<T, 9> e, eb, s, sb;
        auto &u0xF = u0x.value();
        auto &u0xb = u0x.bvalue();
        auto &u1xF = u1x.value();
        auto &u1xb = u1x.bvalue();
        auto &e0tyF = e0ty.value();
        auto &e0tyb = e0ty.bvalue();
        auto &etF = et.value();
        auto &etb = et.bvalue();

        // linear shell strains
        // Evaluate the in-plane strains from the tying strain expressions
        e[0] = e0tyF[0];        // e11
        e[1] = e0tyF[3];        // e22
        e[2] = 2.0 * e0tyF[1];  // e12

        // Compute the bending strain
        e[3] = u1xF[0];            // k11
        e[4] = u1xF[4];            // k22
        e[5] = u1xF[1] + u1xF[3];  // k12

        // Add the components of the shear strain
        e[6] = 2.0 * e0tyF[4];  // e23, transverse shear
        e[7] = 2.0 * e0tyF[2];  // e13, transverse shear
        e[8] = etF[0];          // e12 (drill strain)

        printf("e:");
        printVec<T>(9, e.get_data());

        // forward stresses derivs (dstress/dthick) are dU/dstrain/dx
        T C[6];
        Data::evalTangentStiffness2D(physData.E, physData.nu, C);
        T thick = physData.thick;
        A2D::SymMatVecCoreScale3x3<T, false>(1.0, C, e.get_data(), s.get_data());
        A2D::SymMatVecCoreScale3x3<T, true>(thick * thick / 4.0, C, &e.get_data()[3],
                                            &s.get_data()[3]);
        T dAs = Data::getTransShearCorrFactor() * C[5];
        s[6] = dAs * e[6];
        s[7] = dAs * e[7];
        T ddrill = Data::getDrillingRegularization() * dAs;
        s[8] = ddrill * e[8];
        for (int i = 0; i < 9; i++) eb[i] = scale * s[i];

        // strain derivs back to input disp grad derivs
        e0tyb[0] += eb[0];        // e1
        e0tyb[3] += eb[1];        // e22
        e0tyb[1] += 2.0 * eb[2];  // e12

        // Compute the bending strain
        u1xb[0] += eb[3];  // k11
        u1xb[4] += eb[4];  // k22
        u1xb[1] += eb[5];  // k12
        u1xb[3] += eb[5];  // k12

        // Add the components of the shear strain
        e0tyb[4] += 2.0 * eb[6];  // e23, transverse shear
        e0tyb[2] += 2.0 * eb[7];  // e13, transverse shear
        etb[0] += eb[8];          // e12 (drill strain)
    }                             // end of computeWeakRes

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
>>>>>>> 42213dc9efa552f5caa9f39073da390944f4589e
