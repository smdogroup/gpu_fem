
#include "linalg/_linalg.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"
#include "solvers/_solvers.h"
#include <chrono>

// shell imports
#include "assembler.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/shell_elem_group.h"

int main() {
  using T = double;

  auto start0 = std::chrono::high_resolution_clock::now();

  // uCRM mesh files can be found at:
  // https://data.niaid.nih.gov/resources?id=mendeley_gpk4zn73xn
  TACSMeshLoader<T> mesh_loader{};
  // mesh_loader.scanBDFFile("../uCRM/CRM_box_2nd.bdf");
  mesh_loader.scanBDFFile("uCRM-135_wingbox_fine.bdf");

  using Quad = QuadLinearQuadrature<T>;
  using Director = LinearizedRotation<T>;
  using Basis = LagrangeQuadBasis<T, Quad, 1>;
  using Geo = Basis::Geo;

  constexpr bool has_ref_axis = false;
  constexpr bool is_nonlinear = true;
  using Data = ShellIsotropicData<T, has_ref_axis>;
  using Physics = IsotropicShell<T, Data, is_nonlinear>;

  using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
  using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;

  double E = 70e9, nu = 0.3, thick = 0.02;  // material & thick properties

  // make the assembler from the uCRM mesh
  auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));

  // BSR factorization
  auto start1 = std::chrono::high_resolution_clock::now();
  auto& bsr_data = assembler.getBsrData();
  double fillin = 10.0;  // 10.0
  bool print = true;
  bsr_data.AMD_reordering();
  bsr_data.compute_full_LU_pattern(fillin, print);
  assembler.moveBsrDataToDevice();
  auto end1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> compute_nz_time = end1 - start1;

  // get the loads
  int nvars = assembler.get_num_vars();
  int nnodes = assembler.get_num_nodes();
  HostVec<T> h_loads(nvars);
  double load_mag = 3.0;
  double *h_loads_ptr = h_loads.getPtr();
  for (int inode = 0; inode < nnodes; inode++) {
    h_loads_ptr[6 * inode + 2] = load_mag;
  }
  auto loads = h_loads.createDeviceVec();
  assembler.apply_bcs(loads);

  // setup kmat and initial vecs
  auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
  auto soln = assembler.createVarsVec();
  auto res = assembler.createVarsVec();
  auto rhs = assembler.createVarsVec();
  auto vars = assembler.createVarsVec();

  // newton solve => go to 10x the 1m up disp from initial loads
  int num_load_factors = 10, num_newton = 50;
  T min_load_factor = 0.1, max_load_factor = 2.3, abs_tol = 1e-8,
    rel_tol = 1e-8;
  auto solve_func = CUSPARSE::direct_LU_solve<T>;
  std::string outputPrefix = "out/uCRM_";
  auto start2 = std::chrono::high_resolution_clock::now();
  newton_solve<T, BsrMat<DeviceVec<T>>, DeviceVec<T>, Assembler>(
      solve_func, kmat, loads, soln, assembler, res, rhs, vars,
      num_load_factors, min_load_factor, max_load_factor, num_newton, abs_tol,
      rel_tol, outputPrefix, print);

  auto end2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> nl_solve_time = end2 - start2;

  // print some of the data of host residual
  auto h_vars = vars.createHostVec();
  printToVTK<Assembler, HostVec<T>>(assembler, h_vars, "out/uCRM_nl.vtk");

  printf("uCRM NL case on GPU with load factor %.4f\n", max_load_factor);
  printf("\tcompute nz time = %.4f\n", compute_nz_time.count());
  printf("\tnonlinear solve time = %.4f\n", nl_solve_time.count());
  

  // free data
  assembler.free();
  h_loads.free();
  kmat.free();
  soln.free();
  res.free();
  vars.free();
  h_vars.free();
  rhs.free();
};