#include <iostream>
#include <sstream>

#include "chrono"
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "coupled/_coupled.h"
// // #include "coupled/aero_solver.h"
// // #include "coupled/struct_solver.h"
// // #include "coupled/coupled_analysis.h"
// #include "coupled/meld.h"

// shell imports
#include "assembler.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/shell_elem_group.h"

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
int main(void) {
  using T = double;

  std::ios::sync_with_stdio(false);  // always flush print immediately

  using Quad = QuadLinearQuadrature<T>;
  using Director = LinearizedRotation<T>;
  using Basis = LagrangeQuadBasis<T, Quad, 1>;
  using Geo = Basis::Geo;

  constexpr bool has_ref_axis = false;
  constexpr bool is_nonlinear = true;
  // constexpr bool is_nonlinear = false;
  using Data = ShellIsotropicData<T, has_ref_axis>;
  using Physics = IsotropicShell<T, Data, is_nonlinear>;

  using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
  using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;

  // define coupled analysis types
  // -----------------------------
  using StructSolver = TacsNonlinearStaticNewton<T, Assembler>;
  using AeroSolver = FixedAeroSolver<T, DeviceVec<T>>;
  using Transfer = MELD<T, 32>;
  using CoupledDriver = FuntofemCoupledAnalysis<T, DeviceVec<T>, StructSolver, AeroSolver, Transfer>;


  // build the Tacs prelim objects
  // -----------------------------

  // material & thick properties
  double E = 1.2e6, nu = 0.0, thick = 0.1;

  // make the coarse struct mesh (mesh generated with nex = 16 elems along length)
  TACSMeshLoader<T> mesh_loader{};
  mesh_loader.scanBDFFile("Beam.bdf");
  auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));
  auto xs0 = assembler.getXpts();
  int ns = assembler.get_num_nodes();

  // make the fine aero mesh (generated with nex = 37 elems along length)
  TACSMeshLoader<T> mesh_loader2{};
  mesh_loader2.scanBDFFile("Beam-fine.bdf");
  auto assembler_aero = Assembler::createFromBDF(mesh_loader2, Data(E, nu, thick));
  auto xa0 = assembler_aero.getXpts();
  int na = assembler_aero.get_num_nodes();  

  // perform a factorization on the rowPtr, colPtr (before creating matrix)
  auto& bsr_data = assembler.getBsrData();
  double fillin = 10.0;  // 10.0
  bool print = true;
  bsr_data.compute_full_LU_pattern(fillin, print);
  assembler.moveBsrDataToDevice();

  // compute load magnitude of tip force
  double length = 10.0, width = 1.0;
  double Izz = width * thick * thick * thick / 12.0;
  double beam_tip_force = 4.0 * E * Izz / length / length;

  // compute loads
  auto h_loads = getTipLoads<T>(assembler_aero, length, beam_tip_force);
  auto d_loads = h_loads.createDeviceVec();
  assembler_aero.apply_bcs(d_loads);

  // setup kmat
  auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
  auto linear_solve = CUSPARSE::direct_LU_solve<T>;
  
  // make the solvers and transfer scheme
  // ------------------------------------

  int num_load_factors = 20, num_newton = 30;
  StructSolver struct_solver = StructSolver(assembler, kmat, linear_solve, num_load_factors, num_newton);
  // make the struct linear solver
  // TacsLinearStatic struct_solver = TacsLinearStatic(assembler, kmat, linear_solve);  

  AeroSolver aero_solver = AeroSolver(na, d_loads);

  T beta = 3.0, Hreg = 1e-4;
  int sym = -1, nn = 32;
  Transfer transfer = Transfer(xs0, xa0, beta, nn, sym, Hreg);
  transfer.initialize();

  // make coupled analysis object
  // ----------------------------

  int num_coupled_steps = 10;
  CoupledDriver driver = CoupledDriver(struct_solver, aero_solver, transfer, num_coupled_steps);
  driver.solve_forward();

  return 0;
};