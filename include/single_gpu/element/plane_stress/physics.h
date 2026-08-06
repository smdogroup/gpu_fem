#pragma once

#include "a2dcore.h"
#include "quadrature.h"

template <typename T, class Quadrature_, A2D::GreenStrainType strainType>
class PlaneStressPhysics {
 public:
  using Quadrature = Quadrature_;

  // Variables at each node (u, v)
  static constexpr int32_t vars_per_node = 2;

  class IsotropicData {
   public:
    IsotropicData() = default;
    // __HOST_DEVICE__ IsotropicData() : E(0.0), nu(0.0), t(0.0) {}
    __HOST_DEVICE__ IsotropicData(double E_, double nu_, double t_)
        : E(E_), nu(nu_), t(t_) {}
    double E, nu, t;
  };

  using Data = IsotropicData;

  template <typename T2>
  __HOST_DEVICE__ static void computeWeakRes(const Data physData, const T scale,
                                             const A2D::Mat<T2, 2, 2>& dUdx_,
                                             A2D::Mat<T2, 2, 2>& weak_res) {
    A2D::ADObj<A2D::Mat<T2, 2, 2>> dUdx(dUdx_);
    A2D::ADObj<A2D::SymMat<T2, 2>> strain, stress;
    A2D::ADObj<T2> energy;

    const double mu = 0.5 * physData.E / (1.0 + physData.nu);
    const double lambda = 2 * mu * physData.nu / (1.0 - physData.nu);

    // auto strain_energy_stack =
    //     A2D::MakeStack(A2D::MatGreenStrain<strainType>(dUdx, strain),
    //                    A2D::SymIsotropic(mu, lambda, strain, stress),
    //                    A2D::SymMatMultTrace(strain, stress, energy));
    // energy.bvalue() = 0.5 * scale * physData.t;
    // strain_energy_stack.reverse();

    // A2D stacks don't currently work on the GPU so we'll try reversing
    // through the stack manually
    auto strainExpr = A2D::MatGreenStrain<strainType>(dUdx, strain);
    strainExpr.eval();
    auto stressExpr = SymIsotropic(mu, lambda, strain, stress);
    stressExpr.eval();
    auto energyExpr = SymMatMultTrace(strain, stress, energy);
    energyExpr.eval();
    // Compute strain energy derivative w.r.t state gradient
    energy.bvalue() =
        0.5 * scale * physData.t;  // Set the seed value (0.5 because energy =
                                   // 0.5 * sigma : epsilon)
    energyExpr.reverse();          // Reverse mode AD back through the stack
    stressExpr.reverse();
    strainExpr.reverse();

    weak_res.copy(dUdx.bvalue());
  }
};
