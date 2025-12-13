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
DeviceVec<T> getTipLoads(Assembler &assembler, T load_mag) {
  // find nodes within tolerance of x=10.0
  int num_nodes = assembler.get_num_nodes();
  int num_vars = assembler.get_num_vars();
  HostVec<T> h_loads(num_vars);
    double *h_loads_ptr = h_loads.getPtr();
    for (int inode = 0; inode < num_nodes; inode++) {
        h_loads_ptr[6*inode+2] = load_mag;
    }
    auto loads = h_loads.createDeviceVec();
    assembler.apply_bcs(loads);
  return loads;
}

template <typename T, class StructSolver, class AeroSolver, class Transfer>
void testCoupledDriver(StructSolver struct_solver, AeroSolver aero_solver, Transfer transfer) {
  using CoupledDriver = FuntofemCoupledAnalysis<T, DeviceVec<T>, StructSolver, AeroSolver, Transfer>;
  int num_coupled_steps = 2;
  bool demo = true; // demo settings for nonlinear struct (just resets it)
  CoupledDriver driver = CoupledDriver(struct_solver, aero_solver, transfer, num_coupled_steps, demo);
  driver.solve_forward();
}

template <typename T, class StructSolver, class AeroSolver, class Transfer, class Assembler>
void testCoupledDriverManual(StructSolver struct_solver, AeroSolver aero_solver, Transfer transfer, Assembler assembler, Assembler _assembler_aero) {
  
  int ns = assembler.get_num_nodes();
  int na = _assembler_aero.get_num_nodes();

  // break out the coupled loop manually for testing
  auto us = DeviceVec<T>(3 * ns);
  auto h_us = us.createHostVec();
  auto h_us_ext = MELD<T>::template expandVarsVec<3,6>(ns, h_us);
  printToVTK<Assembler,HostVec<T>>(assembler, h_us_ext, "uCRM_us-0.vtk"); // zero, good

  auto ua = transfer.transferDisps(us);
  auto h_ua = ua.createHostVec();
  auto h_ua_ext = MELD<T>::template expandVarsVec<3,6>(na, h_ua);
  printToVTK<Assembler,HostVec<T>>(_assembler_aero, h_ua_ext, "uCRM_ua-0.vtk"); // small, near zero good

  auto fa = aero_solver.getAeroLoads();
  auto h_fa = fa.createHostVec();
  auto h_fa_ext = MELD<T>::template expandVarsVec<3,6>(na, h_fa);
  printToVTK<Assembler,HostVec<T>>(_assembler_aero, h_fa_ext, "uCRM_fa-0.vtk"); // Z force only, good

  auto fs = transfer.transferLoads(fa);
  auto fs_ext = fs.addRotationalDOF();
  auto h_fs_ext = fs_ext.createHostVec();
  printToVTK<Assembler,HostVec<T>>(assembler, h_fs_ext, "uCRM_fs-0.vtk"); // z force only, could be smoother though (increased beta), good

  struct_solver.solve(fs_ext);
  auto us1 = struct_solver.getStructDisps();
  auto h_us1 = us1.createHostVec();
  auto h_us1_ext = MELD<T>::template expandVarsVec<3,6>(ns, h_us1);
  printToVTK<Assembler,HostVec<T>>(assembler, h_us1_ext, "uCRM_us-1.vtk"); // zero everywhere? weird, fix this..

  // second loop
  auto ua1 = transfer.transferDisps(us1);
  auto h_ua1 = ua1.createHostVec();
  auto h_ua1_ext = MELD<T>::template expandVarsVec<3,6>(na, h_ua1);
  printToVTK<Assembler,HostVec<T>>(_assembler_aero, h_ua1_ext, "uCRM_ua-1.vtk");

  auto fa1 = aero_solver.getAeroLoads();
  auto h_fa1 = fa1.createHostVec();
  auto h_fa1_ext = MELD<T>::template expandVarsVec<3,6>(na, h_fa1);
  printToVTK<Assembler,HostVec<T>>(_assembler_aero, h_fa1_ext, "uCRM_fa-1.vtk");

  auto fs1 = transfer.transferLoads(fa1);
  auto fs1_ext = fs1.addRotationalDOF();
  auto h_fs1_ext = fs1_ext.createHostVec();
  printToVTK<Assembler,HostVec<T>>(assembler, h_fs1_ext, "uCRM_fs-1.vtk");

  struct_solver.solve(fs1_ext);
  auto us2 = struct_solver.getStructDisps();
  auto h_us2 = us2.createHostVec();
  auto h_us2_ext = MELD<T>::template expandVarsVec<3,6>(ns, h_us2);
  printToVTK<Assembler,HostVec<T>>(assembler, h_us2_ext, "uCRM_us-2.vtk");
}

// void makeBoundingBoxMesh() {

// }

/**
 solve on CPU with cusparse for debugging
 **/
int main(void) {
  using T = double;
  // user setting to change to nonlinear or linear here
  constexpr bool is_nonlinear = false;

  std::ios::sync_with_stdio(false);  // always flush print immediately

  using Quad = QuadLinearQuadrature<T>;
  using Director = LinearizedRotation<T>;
  using Basis = ShellQuadBasis<T, Quad, 2>;
  using Geo = Basis::Geo;

  constexpr bool has_ref_axis = false;
  // constexpr bool is_nonlinear = false;
  using Data = ShellIsotropicData<T, has_ref_axis>;
  using Physics = IsotropicShell<T, Data, is_nonlinear>;

  using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
  using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;

  using AeroSolver = FixedAeroSolver<T, DeviceVec<T>>;
  static constexpr int NN = 64; // 32 (may be too small at 32)
  using Transfer = MELD<T, NN>;


  // build the Tacs prelim objects
  // -----------------------------

  double E = 70e9, nu = 0.3, thick = 0.005; // material & thick properties

    // load the medium mesh for the struct mesh
    // uCRM mesh files can be found at: https://data.niaid.nih.gov/resources?id=mendeley_gpk4zn73xn
    // ----------------------------------------
        
    TACSMeshLoader<T> mesh_loader{};
    mesh_loader.scanBDFFile("uCRM-135_wingbox_medium.bdf");
    auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));
    int ns = assembler.get_num_nodes();
    int ns_vars = assembler.get_num_vars();

    auto xs0 = assembler.getXpts();

    // load the coarse mesh for the aero surf
    // --------------------------------------

    TACSMeshLoader<T> mesh_loader_aero{};
    mesh_loader_aero.scanBDFFile("uCRM-135_wingbox_coarse.bdf");
    // mesh_loader_aero.scanBDFFile("tacs.dat"); // must use quad elements here rn
    auto _assembler_aero = Assembler::createFromBDF(mesh_loader_aero, Data(E, nu, thick));
    int na = _assembler_aero.get_num_nodes();
    int na_vars = _assembler_aero.get_num_vars();

    auto xa0 = _assembler_aero.getXpts();
    auto h_xa0 = xa0.createHostVec();

    // // print two different aero nodes
    // printf("aero node 9532 at:");
    // printVec<T>(3, &h_xa0[3 * 9532]);
    // printf("aero node 1621 at:");
    // printVec<T>(3, &h_xa0[3 * 1621]);

  // perform a factorization on the rowPtr, colPtr (before creating matrix)
  double fillin = 10.0;  // 10.0
  assembler.symbolic_factorization(fillin, true);

  // compute loads
  double load_mag = 1.0;
  auto d_loads = getTipLoads<T>(_assembler_aero, load_mag);

  // setup kmat
  auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
  auto linear_solve = CUSPARSE::direct_LU_solve<T>;

  // make transfer scheme
  AeroSolver aero_solver = AeroSolver(na, d_loads);

  T beta = 1e-1, Hreg = 1e-4;
  int sym = -1;
  Transfer transfer = Transfer(xs0, xa0, beta, NN, sym, Hreg);
  transfer.initialize();
  
  // make the solvers and transfer scheme
  // ------------------------------------

  if constexpr (is_nonlinear) {
    // define coupled analysis types
    // -----------------------------
    using StructSolver = TacsNonlinearStaticNewton<T, Assembler>;

    int num_load_factors = 2, num_newton = 30;
    double abs_tol = 1e-8, rel_tol = 1e-9;
    bool struct_print = true;
    TacsNonlinearStaticNewton<T, Assembler> struct_solver = TacsNonlinearStaticNewton<T, Assembler>(assembler, kmat, linear_solve, num_load_factors, num_newton, struct_print, abs_tol, rel_tol);
    
    // test coupled driver
    // testCoupledDriver<T>(struct_solver, aero_solver, transfer);
    testCoupledDriverManual<T>(struct_solver, aero_solver, transfer, assembler, _assembler_aero);


  } else {
    using StructSolver = TacsLinearStatic<T, Assembler>;

    bool struct_print = true;
    auto struct_solver = TacsLinearStatic<T, Assembler>(assembler, kmat, linear_solve, struct_print);  

    // test coupled driver
    // testCoupledDriver<T>(struct_solver, aero_solver, transfer);
    testCoupledDriverManual<T>(struct_solver, aero_solver, transfer, assembler, _assembler_aero);

  }

  

  

  return 0;
};