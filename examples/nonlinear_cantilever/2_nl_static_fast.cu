#include <iostream>
#include <sstream>

#include "chrono"
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "multigrid/solvers/direct/cusp_directLU.h"
#include "mesh/TACSMeshLoader.h"

// shell imports
#include "element/shell/director/linear_rotation.h"
#include "element/shell/physics/isotropic_shell.h"

// lagrange MITC element
#include "element/shell/basis/lagrange_basis.h"
#include "element/shell/mitc_shell.h"

// new nonlinear solvers
#include "solvers/nonlinear_static/inexact_newton.h"
#include "solvers/nonlinear_static/continuation.h"

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
  using Basis = LagrangeQuadBasis<T, Quad, 1>;
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
  // auto soln = assembler.createVarsVec();
  // auto res = assembler.createVarsVec();
  // auto rhs = assembler.createVarsVec();
  auto vars = assembler.createVarsVec();

  /* new nonlinear solver of Ali's on GPU */

  // build the inexact newton + outer continuation solver
  using Mat = BsrMat<DeviceVec<T>>;
  using Vec = DeviceVec<T>;
  using LinearSolver = CusparseMGDirectLU<T, Assembler>;
  using INK = InexactNewtonSolver<T, Mat, Vec, Assembler, LinearSolver>;
  using NL = NonlinearContinuationSolver<T, Vec, Assembler, INK>;

  cublasHandle_t cublasHandle = NULL;
  CHECK_CUBLAS(cublasCreate(&cublasHandle));
  cusparseHandle_t cusparseHandle = NULL;
  CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

  LinearSolver *solver = new LinearSolver(cublasHandle, cusparseHandle, assembler, kmat);
  INK *inner_solver = new INK(cublasHandle, assembler, kmat, d_loads, solver);
  NL *nl_solver = new NL(cublasHandle, assembler, inner_solver);

  // now try calling it
  T lambda0 = 0.2;
  // T lambda0 = 0.05;
  nl_solver->solve(vars, lambda0);

  // permute vars to output order (from solve order?)
  auto d_bsr_data = assembler.getBsrData();
  vars.permuteData(d_bsr_data.block_dim, d_bsr_data.perm);

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

  // also write soln to VTK
  printToVTK<Assembler,HostVec<T>>(assembler, h_soln, "out/nl_beam_fast.vtk");

  return 0;
};