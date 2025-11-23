#pragma once

#include "a2dcore.h"
#include "a2dshell.h"
#include "shell/data/composite_stiffened.h"
#include "shell/data/composite_unstiffened.h"

template <typename T, class Data_, bool isNonlinear = false>
class CompositeShell {
   public:
    using Data = Data_;

    // ensure the data is only allowed to be ShellIsotropicData
    static_assert(std::is_same<Data_, ShellCompositeStiffenedData>::value ||
                      std::is_same<Data_, ShellCompositeUnstiffenedData>::value,
                  "Error IsotropicShell physics class must use Data class "
                  "'ShellCompositeStiffenedData' or 'ShellCompositeUnstiffenedData'");

    // u, v, w, thx, thy, thz
    static constexpr int32_t vars_per_node = 6;
    // whether strain is linear or nonlinear (in this case linear)
    static constexpr A2D::ShellStrainType STRAIN_TYPE =
        isNonlinear ? A2D::ShellStrainType::NONLINEAR : A2D::ShellStrainType::LINEAR;

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
    }  // end of computeWeakRes
};