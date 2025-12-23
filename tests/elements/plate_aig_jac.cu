#include "utils//local_utils.h"
#include "chrono"
#include "linalg/_linalg.h"
#include "../test_commons.h"

// shell imports
// #include "element/shell/director/linear_rotation.h"
#include "element/shell/physics/isotropic_shell.h"

// aig plate
#include "element/plate/basis/bspline_basis.h"
#include "element/plate/aig_plate.h"

template <bool is_nonlinear>
void test_kelem_gpu() {

  using T = double;
  // bool print = false;
  // bool print = is_nonlinear;
  bool print = true;
  using Quad = QuadQuadraticQuadrature<T>;
  // using Director = LinearizedRotation<T>;
  using Basis = BsplineQuadBasis<T, Quad, 2>;
  using Geo = Basis::Geo;

  constexpr bool has_ref_axis = false;
  using Data = ShellIsotropicData<T, has_ref_axis>;
  const bool HR = false;
  using Physics = IsotropicShell<T, Data, is_nonlinear, HR>;
  using Assembler = AsymptoticIsogeometricPlateAssembler<T, Basis, Physics, VecType, BsrMat>;

  // printf("running!\n");
  int num_bcs = 2;
  printf("create one element assembler\n");

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

  // printf("try set variables\n");
  auto vars = h_vars.createDeviceVec();
  assembler.set_variables(vars);

  // printf("create BSRmat\n");
  auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);

  // time add residual method
  auto start = std::chrono::high_resolution_clock::now();

  assembler.add_residual_fast(res);
  CHECK_CUDA(cudaDeviceSynchronize());

  // assembler.add_jacobian(res, mat);
  printf("add jacobian fast\n");
  assembler.add_jacobian_fast(kmat);
  CHECK_CUDA(cudaDeviceSynchronize());
  printf("\tdone with add jacobian fast\n");

  assembler.apply_bcs(kmat);

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  // print res, jac
  auto h_res = res.createHostVec();
  auto h_mat = kmat.getVec().createHostVec();
  T jac_TD = 0.0;
  T res_TD = 0.0;
  for (int i = 0; i < 24; i++) {
    res_TD += p_vars[i] * h_res[i];
    for (int j = 0; j < 24; j++) {
      // read BSR nodal order
      int inode = i / 6, jnode = j / 6;
      int ii = i % 6, jj = j % 6;
      int bnode = 4 * inode + jnode;
      int inz = 36 * bnode + (6 * ii + jj);

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
    // print each block node of 36 values

    // for (int bnode = 0; bnode < 1; bnode++) {
    for (int inz = 0; inz < 81; inz++) {
        const T *h_mat_block = &h_mat_ptr[36 * inz];
        printf("------------------------\nkmat block (%d, %d):\n------------------------\n", inz % 9, inz / 9);
        for (int j = 0; j < 6; j++) {
            printf("row %d: ", j);
            printVec<double>(6, &h_mat_block[6 * j]);
        }
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
  // test_kelem_gpu<true>(); // nonlinear

  return 0;
};