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
  using Assembler = ElementAssembler<T, ElemGroup, VecType, DenseMat>;

  int num_bcs = 10;
  int num_elements = 1e3;
  int num_geo_nodes = 1e2;
  int num_vars_nodes = 1e2;

  auto assembler = createFakeAssembler<Assembler>(
      num_bcs, num_elements, num_geo_nodes, num_vars_nodes);

  // init variables u
  bool randomize = true;
  auto vars = assembler.createVarsVec(randomize);
  assembler.set_variables(vars);

  auto res = assembler.createVarsVec();

  // time add residual method
  auto start = std::chrono::high_resolution_clock::now();

  assembler.add_residual(res);

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  // compute total direc derivative of analytic residual

  // print data of host residual
  auto h_res = res.createHostVec();
  printf("res (first 24 entries): ");
  printVec<double>(24, h_res.getPtr());

  printf("took %d microseconds to run add residual\n", (int)duration.count());

  return 0;
};