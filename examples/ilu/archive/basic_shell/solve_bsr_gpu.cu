#include "_local_utils.h"
#include "chrono"
#include "linalg/linalg.h"
#include "shell/shell.h"

// get residual directional derivative analytically on the CPU

int main(void) {
  using T = double;

  using Quad = QuadLinearQuadrature<T>;
  using Director = LinearizedRotation<T>;
  using Basis = ShellQuadBasis<T, Quad, 2>;
  using Geo = Basis::Geo;

  constexpr bool has_ref_axis = false;
  using Data = ShellIsotropicData<T, has_ref_axis>;
  using Physics = IsotropicShell<T, Data>;

  using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
  using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;

  int num_bcs = 10;

  // int num_elements = 1e3;
  // int num_geo_nodes = 1e2;
  // int num_vars_nodes = 1e2;

  int num_geo_nodes = 100;
  int num_vars_nodes = num_geo_nodes;
  int num_elements = 1e3;

  auto assembler = createFakeAssembler<Assembler>(
      num_bcs, num_elements, num_geo_nodes, num_vars_nodes);

  // init variables u
  bool randomize = true;
  auto vars = assembler.createVarsVec(randomize);
  assembler.set_variables(vars);

  auto res = assembler.createVarsVec();
  auto soln = assembler.createVarsVec();
  auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);

  // time add residual method
  auto start = std::chrono::high_resolution_clock::now();
  assembler.add_jacobian(res, kmat);
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  // apply the BCs to the res and kmat
  assembler.apply_bcs(res);
  assembler.apply_bcs(kmat);

  // now do cusparse solve
  auto start2 = std::chrono::high_resolution_clock::now();
  cusparse_solve<T>(kmat, res, soln);
  auto stop2 = std::chrono::high_resolution_clock::now();
  auto duration2 =
      std::chrono::duration_cast<std::chrono::microseconds>(stop2 - start2);

  // compute total direc derivative of analytic residual

  // print some of the data of host residual
  auto h_res = res.createHostVec();
  auto h_soln = soln.createHostVec();
  auto h_kmat = kmat.createHostVec();

  printf("kmat: ");
  printVec<double>(24, h_kmat.getPtr());
  printf("\n");

  printf("soln: ");
  printVec<double>(24, h_soln.getPtr());
  printf("\n");

  printf("rhs: ");
  printVec<double>(24, h_res.getPtr());
  printf("\n");

  printf("took %d microseconds to run add jacobian\n", (int)duration.count());
  printf("took %d microseconds to run cusparse solve\n", (int)duration2.count());

  return 0;
};