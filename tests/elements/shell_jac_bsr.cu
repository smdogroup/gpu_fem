#include "utils//local_utils.h"
#include "chrono"
#include "linalg/_linalg.h"
#include "../test_commons.h"

// shell imports
#include "element/shell/director/linear_rotation.h"
#include "element/shell/physics/isotropic_shell.h"

// lagrange MITC element
#include "element/shell/basis/lagrange_basis.h"
#include "element/shell/mitc_shell.h"

template <bool is_nonlinear>
void test_kelem_gpu() {

  using T = double;
  // bool print = false;
  // bool print = is_nonlinear;
  bool print = true;

  // using Quad = QuadLinearQuadrature<T>;
  using Quad = QuadQuadraticQuadrature<T>;
  using Director = LinearizedRotation<T>;
  // using Basis = LagrangeQuadBasis<T, Quad, 1>;
  using Basis = LagrangeQuadBasis<T, Quad, 2>;
  using Geo = Basis::Geo;

  constexpr bool has_ref_axis = false;
  using Data = ShellIsotropicData<T, has_ref_axis>;
  using Physics = IsotropicShell<T, Data, is_nonlinear>;
  using Assembler = MITCShellAssembler<T, Director, Basis, Physics, DeviceVec, BsrMat>;

  // printf("running!\n");
  int num_bcs = 2;
  bool structured = true;
  // auto assembler = createOneElementAssembler<Assembler>(num_bcs, structured);
  auto assembler = createOneElementAssembler_order2<Assembler>(num_bcs);

  // init variables u
  int num_vars = assembler.get_num_vars();
  auto res = assembler.createVarsVec();
  auto h_vars = HostVec<T>(num_vars);
  auto p_vars = HostVec<T>(num_vars);
  auto p_vars2 = HostVec<T>(num_vars);

  // fixed perturbations of the host and pert vars
  for (int ivar = 0; ivar < 24; ivar++) {
    h_vars[ivar] = (1.4543 + 6.4323 * ivar) * 1e-6;
    h_vars[ivar] *= 1e6;
    p_vars[ivar] = (-1.4543 + 2.312 * 6.4323 * ivar);
    p_vars2[ivar] = (-1.4543 * 1.024343 + 2.812 * -9.4323 * ivar);
  }

  auto vars = h_vars.createDeviceVec();
  assembler.set_variables(vars);

  auto mat = createBsrMat<Assembler, VecType<T>>(assembler);

  // time add residual method
  auto start = std::chrono::high_resolution_clock::now();

  assembler.add_jacobian_fast(mat);
  assembler.add_residual_fast(res);

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  // print res, jac
  auto h_res = res.createHostVec();
  auto h_mat = mat.createHostVec();
  T jac_TD = 0.0;
  T res_TD = 0.0;
  for (int i = 0; i < 24; i++) {
    res_TD += p_vars[i] * h_res[i];
    for (int j = 0; j < 24; j++) {
      jac_TD += p_vars[i] * p_vars2[j] * h_mat[24 * i + j];
    }
  }

  if (print) {
    printf("Analytic Jacobian GPU\n");
    printf("res TD = %.8e\n", res_TD);
    printf("jac TD = %.8e\n", jac_TD);
  }

  // print residual
  if (print) {
    printf("res: ");
    printVec<double>(num_vars, h_res.getPtr());
  }

  const double *h_mat_ptr = h_mat.getPtr();
  if (print) {
    // print each block node of 36 values

    for (int bnode = 0; bnode < 1; bnode++) {
    // for (int bnode = 0; bnode < 16; bnode++) {
      printf("------------------------\nkmat block %d:\n------------------------\n", bnode);
      for (int j = 0; j < 6; j++) {
        printf("row %d: ", j);
        printVec<double>(6, &h_mat_ptr[6 * j]);
      }
    }
    printf("------------------------\n\n");
  }

  printKernelTiming(duration.count());
}

int main(void) {

  test_kelem_gpu<false>(); // linear
  // test_kelem_gpu<true>(); // nonlinear

  return 0;
};