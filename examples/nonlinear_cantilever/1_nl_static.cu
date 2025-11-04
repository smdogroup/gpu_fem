#include <iostream>
#include <sstream>

#include "chrono"
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "mesh/TACSMeshLoader.h"

// shell imports
#include "element/shell/director/linear_rotation.h"
#include "element/shell/physics/isotropic_shell.h"

// lagrange MITC element
#include "element/shell/basis/lagrange_basis.h"
#include "element/shell/mitc_shell.h"

template <typename T, class Assembler>
HostVec<T> getTipLoads(Assembler &assembler, T length, T beam_tip_force) {
  // find nodes within tolerance of x=10.0
  int num_nodes = assembler.get_num_nodes();
  int num_vars = assembler.get_num_vars();
  HostVec<T> h_loads(num_vars);
  DeviceVec<T> d_xpts = assembler.getXpts();
  auto h_xpts = d_xpts.createHostVec();
  int num_tip_nodes = 0;
  for (int inode = 0; inode < num_nodes; inode++) {
    if (abs(h_xpts[3 * inode] - length) < 1e-6) {
      num_tip_nodes++;
    }
  }
  T nodal_force = beam_tip_force / num_tip_nodes;
  printf("nodal force = %.4e\n", nodal_force);
  for (int inode = 0; inode < num_nodes; inode++) {
    if (abs(h_xpts[3 * inode] - length) < 1e-6) {
      h_loads[6 * inode + 2] = beam_tip_force / num_tip_nodes;
    }
  }
  return h_loads;
}

/**
 solve on CPU with cusparse for debugging
 **/
int main(int argc, char **argv) {
  using T = double;

  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;

  std::ios::sync_with_stdio(false);  // always flush print immediately

  TACSMeshLoader mesh_loader{comm};
  mesh_loader.scanBDFFile("Beam.bdf");

  using Quad = QuadLinearQuadrature<T>;
  using Director = LinearizedRotation<T>;
  using Basis = LagrangeQuadBasis<T, Quad, 2>;
  using Geo = Basis::Geo;

  constexpr bool has_ref_axis = false;
  constexpr bool is_nonlinear = true;
  // constexpr bool is_nonlinear = false;
  using Data = ShellIsotropicData<T, has_ref_axis>;
  using Physics = IsotropicShell<T, Data, is_nonlinear>;
  using Assembler = MITCShellAssembler<T, Director, Basis, Physics, DeviceVec, BsrMat>;

  // material & thick properties
  double E = 1.2e6, nu = 0.0, thick = 0.1;
  auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));

  // perform a factorization on the rowPtr, colPtr (before creating matrix)
  auto& bsr_data = assembler.getBsrData();
  double fillin = 10.0;  // 10.0
  bool print = true;
  // bsr_data.AMD_ordering();
  bsr_data.compute_full_LU_pattern(fillin, print);
  assembler.moveBsrDataToDevice();

  // compute load magnitude of tip force
  double length = 10.0, width = 1.0;
  double Izz = width * thick * thick * thick / 12.0;
  double beam_tip_force = 4.0 * E * Izz / length / length;

  // compute loads
  auto h_loads = getTipLoads<T>(assembler, length, beam_tip_force);
  auto d_loads = h_loads.createDeviceVec();
  assembler.apply_bcs(d_loads);

  // setup kmat and initial vecs
  auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
  auto soln = assembler.createVarsVec();
  auto res = assembler.createVarsVec();
  auto rhs = assembler.createVarsVec();
  auto vars = assembler.createVarsVec();

  // newton solve
  int num_load_factors = 20, num_newton = 30;
  T min_load_factor = 0.05, max_load_factor = 1.00, abs_tol = 1e-8,
    rel_tol = 1e-7;
  auto solve_func = CUSPARSE::direct_LU_solve<T>;
  std::string outputPrefix = "out/beam_";
  newton_solve<T, BsrMat<DeviceVec<T>>, DeviceVec<T>, Assembler>(
      solve_func, kmat, d_loads, soln, assembler, res, rhs, vars,
      num_load_factors, min_load_factor, max_load_factor, num_newton, abs_tol,
      rel_tol, outputPrefix, print);

  // get tip displacements
  int num_nodes = assembler.get_num_nodes();
  auto xpts = assembler.getXpts();
  auto h_xpts = xpts.createHostVec();
  auto h_soln = vars.createHostVec();
  for (int inode = 0; inode < num_nodes; inode++) {
    if (abs(h_xpts[3 * inode] - length) < 1e-6) {
      printf("End disp (u,v,w):");
      printVec<T>(3, &h_soln.getPtr()[6*inode]);
    }
  }

  // using linear rotationType (not default quadratic)
  // perfect match now!
  printf("ref (u,v,w): -2.218, 0.0, 5.791\n");

  return 0;
};