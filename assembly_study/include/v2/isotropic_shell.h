#pragma once

#include "a2dcore.h"
#include "isotropic_data.h"

template <typename T, class Data_, bool isNonlinear = false>
class IsotropicShellV2 {
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
};
