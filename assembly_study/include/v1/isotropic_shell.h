#pragma once

#include "a2d/_a2dshell.h"
#include "a2dcore.h"
#include "isotropic_data.h"

template <typename T, class Data_, bool isNonlinear = false>
class IsotropicShellV1 {
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

        Uelem.bvalue() = 1.0;
        strain_energy_stack.reverse();
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
    }  // end of computeWeakRes
};
