#include <iostream>
#include <sstream>

#include "chrono"
#include "coupled/_coupled.h"
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
// // #include "coupled/aero_solver.h"
// // #include "coupled/struct_solver.h"
// // #include "coupled/coupled_analysis.h"
// #include "coupled/meld.h"
#include "_crm_utils.h"

// shell imports
#include "assembler.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/shell_elem_group.h"

/**
 solve on CPU with cusparse for debugging
 **/
int main(void) {
  using T = double;

  // important user settings
  // -----------------------
  // constexpr bool nonlinear_strain = false; // Kevin had nonlinear strain = false
  // but I only get the disp anomaly for nonlinear strain true
  constexpr bool nonlinear_strain = false;
  static constexpr int MELD_NN_PER_BLOCK = 32;

  // type definitions
  // ----------------

  std::ios::sync_with_stdio(false);  // always flush print immediately

  using Quad = QuadLinearQuadrature<T>;
  using Director = LinearizedRotation<T>;
  using Basis = ShellQuadBasis<T, Quad, 2>;
  using Geo = Basis::Geo;

  constexpr bool has_ref_axis = false;
  using Data = ShellIsotropicData<T, has_ref_axis>;
  using Physics = IsotropicShell<T, Data, nonlinear_strain>;

  using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
  using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;

  using AeroSolver = FixedAeroSolver<T, DeviceVec<T>>;
  using Transfer = MELD<T, MELD_NN_PER_BLOCK>;
  // nonlinear MELD has high loads on spars right now

  // build the Tacs prelim objects
  // -----------------------------

  double E = 70e9, nu = 0.3, thick = 0.03;  // material & thick properties

  // load the medium mesh for the struct mesh
  // uCRM mesh files can be found at:
  // https://data.niaid.nih.gov/resources?id=mendeley_gpk4zn73xn
  // ----------------------------------------

  TACSMeshLoader<T> mesh_loader{};
  mesh_loader.scanBDFFile("CRM_box_2nd.bdf");
  auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));
  int ns = assembler.get_num_nodes();
  int ns_vars = assembler.get_num_vars();

  auto xs0 = assembler.getXpts();

  // load the coarse mesh for the aero surf
  // --------------------------------------

  double *xyz_forces;
  auto _assembler_aero = makeFun3dAeroSurfMeshFromDat<Assembler>("fun3d_forces.dat", &xyz_forces);
  auto xa0 = _assembler_aero.getXpts();
  int na = _assembler_aero.get_num_nodes();
  auto h_xa0 = xa0.createHostVec();

  // printToVTK<Assembler, HostVec<T>>(_assembler_aero, h_xa0, "xa0.vtk");

  // perform a factorization on the rowPtr, colPtr (before creating matrix)
  auto& bsr_data = assembler.getBsrData();
  double fillin = 10.0;  // 10.0
  bool print = true;
  bsr_data.AMD_reordering();
  bsr_data.compute_full_LU_pattern(fillin, print);
  assembler.moveBsrDataToDevice();

  // compute loads from aero
  printf("here4\n");
  auto h_loads = HostVec<T>(6 * na);
  for (int i = 0; i < na; i++) {
    h_loads[6*i] = xyz_forces[3*i];
    h_loads[6*i+1] = xyz_forces[3*i+1];
    h_loads[6*i+2] = xyz_forces[3*i+2];
  }
  printf("here5-0\n");
  auto d_loads = h_loads.createDeviceVec();
  printf("here5\n");

  // setup kmat
  auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
  auto linear_solve = CUSPARSE::direct_LU_solve<T>;

  // make transfer scheme
  printf("here6\n");
  AeroSolver aero_solver = AeroSolver(na, d_loads);
  T beta = 0.1, Hreg = 0.0; // 1e-4, 0
  int sym = -1, nn = 128;
  bool meld_print = false;
  Transfer transfer =
      Transfer(xs0, xa0, beta, nn, sym, Hreg, meld_print);
  transfer.initialize();
  printf("here7\n");

  // make the solvers and transfer scheme
  // ------------------------------------
  
  if constexpr (nonlinear_strain) {
    // define coupled analysis types
    // -----------------------------
    using StructSolver = TacsNonlinearStaticNewton<T, Assembler>;

    printf("here8\n");
    int num_load_factors = 10, num_newton = 30;
    double abs_tol = 1e-6, rel_tol = 1e-7;
    bool struct_print = true;
    bool write_intermediate_vtk = false;
    TacsNonlinearStaticNewton<T, Assembler> struct_solver =
        TacsNonlinearStaticNewton<T, Assembler>(assembler, kmat, linear_solve,
                                                num_load_factors, num_newton,
                                                struct_print, abs_tol, rel_tol, write_intermediate_vtk);

    // test coupled driver
    // testCoupledDriver<T>(struct_solver, aero_solver, transfer, assembler);
    printf("here9\n");
    bool aero_point_data = true;
    testCoupledDriverManual<T>(struct_solver, aero_solver, transfer, assembler,
                               _assembler_aero, aero_point_data);

    struct_solver.writeSoln("out/uCRM_coupled_us.vtk");

    struct_solver.free();

  } else {
    using StructSolver = TacsLinearStatic<T, Assembler>;

    bool struct_print = true;
    auto struct_solver = TacsLinearStatic<T, Assembler>(
        assembler, kmat, linear_solve, struct_print);

    // test coupled driver
    // testCoupledDriver<T>(struct_solver, aero_solver, transfer, assembler);
    bool aero_point_data = true;
    testCoupledDriverManual<T>(struct_solver, aero_solver, transfer,
    assembler, _assembler_aero, aero_point_data);

    struct_solver.writeSoln("out/uCRM_coupled_us.vtk");

    struct_solver.free();
  }

  // free
  assembler.free();
  _assembler_aero.free();
  d_loads.free();
  aero_solver.free();
  transfer.free();

  return 0;
};