#include "linalg/_linalg.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"
#include "solvers/_solvers.h"
#include <chrono>

// shell imports
#include "assembler.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/shell_elem_group.h"
#include "element/shell/shell_elem_group_v2.h" // new one for unittesting

int main() {
  using T = double;

  auto start0 = std::chrono::high_resolution_clock::now();
  bool print = true;

  // uCRM mesh files can be found at:
  // https://data.niaid.nih.gov/resources?id=mendeley_gpk4zn73xn
  bool mesh_print = false;
  TACSMeshLoader<T> mesh_loader{mesh_print};
  // mesh_loader.scanBDFFile("../uCRM/CRM_box_2nd.bdf");
  mesh_loader.scanBDFFile("../uCRM-135_wingbox_fine.bdf");

  using Quad = QuadLinearQuadrature<T>;
  using Director = LinearizedRotation<T>;
  using Basis = ShellQuadBasis<T, Quad, 2>;
  using Geo = Basis::Geo;

  constexpr bool has_ref_axis = false;
  constexpr bool is_nonlinear = true;
  using Data = ShellIsotropicData<T, has_ref_axis>;
  using Physics = IsotropicShell<T, Data, is_nonlinear>;

  // try using new ElemGroup to speedup assembly
  using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
  // using ElemGroup = ShellElementGroupV2<T, Director, Basis, Physics>;

  using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;

  double E = 70e9, nu = 0.3, thick = 0.02;  // material & thick properties

  // make the assembler from the uCRM mesh
  auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));

  // BSR factorization
  auto start1 = std::chrono::high_resolution_clock::now();
  auto& bsr_data = assembler.getBsrData();
  // double fillin = 10.0;  // 10.0
  // bsr_data.AMD_reordering();
  // bsr_data.compute_full_LU_pattern(fillin, print);

  // bsr_data.AMD_reordering();
  // bsr_data.qorder_reordering(0.2);
  // bsr_data.compute_ILUk_pattern(5, fillin, print);
  // bsr_data.compute_full_LU_pattern(fillin, print);

  assembler.moveBsrDataToDevice();
  auto end1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> compute_nz_time = end1 - start1;

  // setup kmat and initial vecs
  auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
  auto res = assembler.createVarsVec();
  auto soln = assembler.createVarsVec();

  // debug
  int nvars = assembler.get_num_vars();
  auto h_soln1 = soln.createHostVec();
  for (int i = 0; i < nvars; i++) {
      h_soln1[i] = 1.1343 + 2.3142 * i + 4.132 * i * i;
      h_soln1[i] *= 1e-6;
  }
  auto soln2 = h_soln1.createDeviceVec();
  assembler.set_variables(soln2);

  assembler.apply_bcs(res); // warmup call
  // assembler.add_residual(res, print); // warmup call
  assembler.add_residual(res, print);

  // check residual not zero
  printf("check resid\n");
  auto h_res = res.createHostVec();
  printVec<T>(100, h_res.getPtr());

  return 0;
}