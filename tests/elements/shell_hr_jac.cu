#include "utils//local_utils.h"
#include "chrono"
#include "linalg/_linalg.h"
#include "../test_commons.h"

// shell imports
#include "element/shell/director/linear_rotation.h"
#include "element/shell/physics/isotropic_shell.h"

// lagrange MITC element
#include "element/shell/basis/lagrange_basis.h"
#include "element/shell/hr_shell.h"

template <bool is_nonlinear>
void test_kelem_gpu() {

  using T = double;
  // bool print = false;
  // bool print = is_nonlinear;
  bool print = true;
  using Quad = QuadLinearQuadrature<T>;
  using Director = LinearizedRotation<T>;
  using Basis = LagrangeQuadBasis<T, Quad, 1>;
  using Geo = Basis::Geo;

  constexpr bool has_ref_axis = false;
  using Data = ShellIsotropicData<T, has_ref_axis>;
  const bool HR = true;
  using Physics = IsotropicShell<T, Data, is_nonlinear, HR>;
  using Assembler = HellingerReissnerShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;

  // printf("running!\n");
  int num_bcs = 2;
  auto assembler = createOneElementAssembler<Assembler>(num_bcs);

  // init variables u
  int num_vars = assembler.get_num_vars();
  auto res = assembler.createVarsVec();
  auto h_vars = HostVec<T>(num_vars);
  auto p_vars = HostVec<T>(num_vars);
  auto p_vars2 = HostVec<T>(num_vars);

  // fixed perturbations of the host and pert vars
  for (int ivar = 0; ivar < 44; ivar++) {
    h_vars[ivar] = (1.4543 + 6.4323 * ivar) * 1e-6;
    h_vars[ivar] *= 1e6;
    p_vars[ivar] = (-1.4543 + 2.312 * 6.4323 * ivar);
    p_vars2[ivar] = (-1.4543 * 1.024343 + 2.812 * -9.4323 * ivar);
  }

  auto vars = h_vars.createDeviceVec();
  assembler.set_variables(vars);

  auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);

  // time add residual method
  auto start = std::chrono::high_resolution_clock::now();

  // assembler.add_jacobian(res, mat);
  assembler.add_jacobian_fast(kmat);
  assembler.add_residual_fast(res);

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  // print res, jac
  auto h_res = res.createHostVec();
  auto h_mat = kmat.getVec().createHostVec();
  T jac_TD = 0.0;
  T res_TD = 0.0;
  for (int i = 0; i < 44; i++) {
    res_TD += p_vars[i] * h_res[i];
    for (int j = 0; j < 44; j++) {
      // read BSR nodal order
      int inode = i / 11, jnode = j / 11;
      int ii = i % 11, jj = j % 11;
      int bnode = 4 * inode + jnode;
      int inz = 121 * bnode + (11 * ii + jj);

      jac_TD += p_vars[i] * h_vars[j] * h_mat[inz];
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
    for (int i = 0; i < 44; i++) {  // i < 2
      printf("kmat row %d: ", i);
      printVec<double>(num_vars, &h_mat_ptr[num_vars * i]);
    }
  }  

  // if constexpr (is_nonlinear) {
  //   double max_ref = max(576, cpu_kelem_ref_nl);
  //   double max_abs_err = abs_err(h_mat, cpu_kelem_ref_nl);
  //   double my_rel_err = max_abs_err / max_ref;
  //   bool passed = my_rel_err < 1e-10;
  //   T max_rel_err = rel_err(h_mat, cpu_kelem_ref_nl, 1e-9);
  //   printTestReport("shell elem-jac geom nonlinear", passed, my_rel_err);
  //   printf("\tabs err %.4e, max ref %.4e, norm err %.4e, max rel err %.4e\n", max_abs_err, max_ref, my_rel_err, max_rel_err);
  // } else {
  //   double max_ref = max(576, cpu_kelem_ref_lin);
  //   double max_abs_err = abs_err(h_mat, cpu_kelem_ref_lin);
  //   double my_rel_err = max_abs_err / max_ref;
  //   bool passed = my_rel_err < 1e-10;
  //   T max_rel_err = rel_err(h_mat, cpu_kelem_ref_lin, 1e-9);
  //   printTestReport("shell elem-jac linear", passed, my_rel_err);
  //   printf("\tabs err %.4e, max ref %.4e, norm err %.4e, max rel err %.4e\n", max_abs_err, max_ref, my_rel_err, max_rel_err);
  // }

  // temp debug
  // constexpr bool debug = false;
  // if constexpr (is_nonlinear && debug) {
  //   for (int i = 0; i < 576; i++) {
  //     double val1 = h_mat[i];
  //     double val2 = cpu_kelem_ref_nl[i];
  //     double my_rel_err = rel_err(val1, val2);
  //     int row = i / 24;
  //     int col = i % 24;
  //     printf("row %d, col %d : GPU %.4e, CPU %.4e, rel_err %.4e\n", row, col, val1, val2, my_rel_err);
  //   }
  // }

  printKernelTiming(duration.count());
}

int main(void) {

  test_kelem_gpu<false>(); // linear
  test_kelem_gpu<true>(); // nonlinear

  return 0;
};