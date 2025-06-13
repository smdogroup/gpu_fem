#include "linalg/_linalg.h"
#include "mesh/TACSMeshLoader.h"
#include <chrono>

#include "include/time_assembler.h"

// shell imports
#include "include/v1/v1.h"
#include "include/v2/v2.h"
#include "include/v3/v3.h"

int main() {
  // using T = float;
  using T = double;
  constexpr bool is_nonlinear = true; // true
  
  constexpr bool just_drill_strain = true;
  // constexpr bool just_drill_strain = false;
  constexpr bool is_jac = true;

  // constexpr int version = 1;
  // constexpr int version = 2;
  constexpr int version = 3;

  if constexpr (version == 1) {
    // no just drill strain setting for version 1
    using Quad = QuadLinearQuadratureV1<T>;
    using Director = LinearizedRotationV1<T>;
    using Basis = ShellQuadBasisV1<T, Quad, 2>;
    using Data = ShellIsotropicDataV1<T, false>;
    using Physics = IsotropicShellV1<T, Data, is_nonlinear>;
    using ElemGroup = ShellElementGroupV1<T, Director, Basis, Physics>;
    using Assembler = ElementAssemblerV1<T, ElemGroup, VecType, BsrMat>;

    time_assembler<T, Data, Assembler, is_jac>();

  } else if constexpr (version == 2) {
    constexpr bool full_strain = !just_drill_strain;

    using Quad = QuadLinearQuadratureV2<T>;
    using Director = LinearizedRotationV2<T>;
    using Basis = ShellQuadBasisV2<T, Quad, 2>;
    using Data = ShellIsotropicDataV2<T, false>;
    using Physics = IsotropicShellV2<T, Data, is_nonlinear>;
    using ElemGroup = ShellElementGroupV2<T, Director, Basis, Physics, full_strain>;
    using Assembler = ElementAssemblerV2<T, ElemGroup, VecType, BsrMat>;

    time_assembler<T, Data, Assembler, is_jac>();
  } else if constexpr (version == 3) {
    // exploring pre-computing xpt shell transform data vs computing all in oneshot
    // 1 - shared, 2 - local, 3 - oneshot
    // constexpr int kernel_option = 1; // failed call, 7.97e-5
    // constexpr int kernel_option = 2; // just mem, 2.27e-4
    // constexpr int kernel_option = 3; // 2.29e-4 (fastest but uses more smem)
    constexpr int kernel_option = 4; // 2.38e-4    (fewest smem, prob best for jac, somewhat more variable though..)
    // constexpr int kernel_option = 5; // 2.61e-4 (slower with Tmat, XdinvT in smem than registers)
    // constexpr int kernel_option = 6; // 4.74e-4 (about twice as slow and uses 98 vs 80 registers, so it does matter to do scalar triple products and less loading)

    using Quad = QuadLinearQuadratureV3<T>;
    using Director = LinearizedRotationV3<T>;
    using Basis = ShellQuadBasisV3<T, Quad>;
    using Data = ShellIsotropicDataV3<T, false>;
    using Physics = IsotropicShellV3<T, Data, is_nonlinear>;
    using ElemGroup = ShellElementGroupV3<T, Director, Basis, Physics, kernel_option>; //, full_strain>;
    using Assembler = ElementAssemblerV3<T, ElemGroup, VecType, BsrMat>;

    time_assembler<T, Data, Assembler, is_jac>();
  }
  

  return 0;
}