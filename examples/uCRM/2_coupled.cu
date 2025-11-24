#include <iostream>
#include <sstream>

#include "chrono"
#include "coupled/_coupled.h"
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "_src/crm_utils.h"

// shell imports
#include "assembler.h"
#include "element/shell/director/linear_rotation.h"
#include "element/shell/physics/isotropic_shell.h"

// lagrange MITC element
#include "element/shell/basis/lagrange_basis.h"
#include "element/shell/mitc_shell.h"
/* command line args:
    [meld|coupled] [--linear]

    examples:
    ./2_coupled.out meld            to run meld
    ./2_coupled.out coupled         to run coupled nonlinear
    ./2_coupled.out coupled --linear  to run linear case
*/

// helper functions
// ----------------
// ----------------

template <typename T>
void verify_conservation(HostVec<T> h_us, HostVec<T> h_ua, HostVec<T> h_fs, HostVec<T> h_fa) {
  // compute total aero forces
  T fA_tot[3], fS_tot[3];
  int na = h_fa.getSize() / 3;
  int ns = h_fs.getSize() / 3;
  memset(fA_tot, 0.0, 3 * sizeof(T));
  memset(fS_tot, 0.0, 3 * sizeof(T));

  for (int ia = 0; ia < na; ia++) {
      for (int idim = 0; idim < 3; idim++) {
          fA_tot[idim] += h_fa[3 * ia + idim];
      }
  }

  for (int is = 0; is < ns; is++) {
      for (int idim = 0; idim < 3; idim++) {
          fS_tot[idim] += h_fs[3 * is + idim];
      }
  }

  printf("fA_tot:");
  printVec<double>(3, &fA_tot[0]);

  printf("fS_tot:");
  printVec<double>(3, &fS_tot[0]);

  // compute total work done
  T W_A = 0.0, W_S = 0.0;
  for (int ia = 0; ia < na; ia++) {
      for (int idim = 0; idim < 3; idim++) {
          W_A += h_fa[3 * ia + idim] * h_ua[3 * ia + idim];
      }
  }
  for (int is = 0; is < ns; is++) {
      for (int idim = 0; idim < 3; idim++) {
          W_S += h_fs[3 * is + idim] * h_us[3 * is + idim];
      }
  }

  // compute the relative error
  T W_rel_err = abs(W_A - W_S) / abs(W_S);

  printf("W_A %.6e, W_S %.6e, rel err %.6e\n", W_A, W_S, W_rel_err);
}

void meld_demo(MPI_Comm &comm) {
  using T = double;

  auto start0 = std::chrono::high_resolution_clock::now();

  using Quad = QuadLinearQuadrature<T>;
  using Director = LinearizedRotation<T>;
  using Basis = LagrangeQuadBasis<T, Quad, 1>;
  using Geo = Basis::Geo;

  constexpr bool has_ref_axis = false;
  constexpr bool is_nonlinear = false;
  using Data = ShellIsotropicData<T, has_ref_axis>;
  using Physics = IsotropicShell<T, Data, is_nonlinear>;
  using Assembler = MITCShellAssembler<T, Director, Basis, Physics, DeviceVec, BsrMat>;

  double E = 70e9, nu = 0.3, thick = 0.005;  // material & thick properties

  // load the medium mesh for the struct mesh
  // uCRM mesh files can be found at:
  // https://data.niaid.nih.gov/resources?id=mendeley_gpk4zn73xn
  // ----------------------------------------

  TACSMeshLoader mesh_loader{comm};
  mesh_loader.scanBDFFile("CRM_box_2nd.bdf");
  // mesh_loader.scanBDFFile("uCRM-135_wingbox_medium.bdf");
  auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));
  int ns = assembler.get_num_nodes();
  int ns_vars = assembler.get_num_vars();

  auto xs0 = assembler.getXpts();

  // make struct disps
  auto h_us = HostVec<T>(3 * ns);
  double disp_mag = 1.0;
  // double disp_mag = 0.0;
  auto h_xs0 = xs0.createHostVec();
  for (int i = 0; i < ns; i++) {
    // h_us[3 * i + 2] = disp_mag;
    // h_us[3*i+2] = h_xs0[3*i]/50 * disp_mag;
    h_us[3 * i + 2] = disp_mag * (1.0 - std::sin(h_xs0[3*i]/50 * 3.1415 * 1.0));
  }
  auto us = h_us.createDeviceVec();

  // load the coarse mesh for the aero surf
  // --------------------------------------

//   TACSMeshLoader<T> mesh_loader_aero{};
//   mesh_loader_aero.scanBDFFile("uCRM-135_wingbox_coarse.bdf");
//   auto _assembler_aero =
//       Assembler::createFromBDF(mesh_loader_aero, Data(E, nu, thick));
  int nx = 51;  // 101
  auto _assembler_aero = makeAeroSurfMesh<Assembler>(nx, nx);
  int na = _assembler_aero.get_num_nodes();
  int na_vars = _assembler_aero.get_num_vars();

  auto xa0 = _assembler_aero.getXpts();
  auto h_xa0 = xa0.createHostVec();

  // make aero loads
  auto h_fa = HostVec<T>(3 * na);
  double load_mag = 1.0;
  for (int i = 0; i < na; i++) {
    h_fa[3 * i + 2] =
        -load_mag * (std::sin(h_xa0[3 * i] / 50 * 3.1415 * 2.0) *
                    std::sin(h_xa0[3 * i + 1] / 40 * 3.1415 * 1.0));
  }
  auto fa = h_fa.createDeviceVec();

  // make the MELD transfer scheme
  // -----------------------------

  // T beta = 1e-3, Hreg = 1e-8;
  // T beta = 0.1, Hreg = 1e-4;
  T beta = 1e-1, Hreg = 1e-8;
  int sym = -1, nn = 128;
  static constexpr int NN_PER_BLOCK = 32;
  bool meld_print = true;
  using TransferScheme = MELD<T, NN_PER_BLOCK>;
  auto meld = TransferScheme(xs0, xa0, beta, nn, sym, Hreg, meld_print);
  meld.initialize();

  // disp transfer
  // -------------

  auto ua = meld.transferDisps(us);

  // visualization
  auto h_us_ext = MELD<T>::expandVarsVec<3, 6>(ns, h_us);
  printToVTK<Assembler, HostVec<T>>(assembler, h_us_ext, "out/uCRM_us.vtk");

  // now extend to full 6-length size (MELD only handles length 3)
  auto h_ua = ua.createHostVec();
  auto h_ua_ext = MELD<T>::expandVarsVec<3, 6>(na, h_ua);

  printToVTK<Assembler, HostVec<T>>(_assembler_aero, h_ua_ext, "out/uCRM_ua.vtk");

  // printf("ua:");
  // printVec<T>(100, h_ua.getPtr());

  // load transfer
  // -------------

  auto fs = meld.transferLoads(fa);
  // DeviceVec<T> fs = DeviceVec<T>(3 * na);
  // for (int i = 0; i < 10; i++) {
  //     fs = meld.transferLoads(fa);
  // }

  // extend both fs, fa to length 6 vars_per_node not 3
  auto h_fs = fs.createHostVec();
  auto h_fa_ext = MELD<T>::expandVarsVec<3, 6>(na, h_fa);
  auto h_fs_ext = MELD<T>::expandVarsVec<3, 6>(ns, h_fs);

  // visualization of fa and fs
  printToVTK<Assembler, HostVec<T>>(_assembler_aero, h_fa_ext, "out/uCRM_fa.vtk");
  printToVTK<Assembler, HostVec<T>>(assembler, h_fs_ext, "out/uCRM_fs.vtk");

  // verify conservation of force and work
  verify_conservation<T>(h_us, h_ua, h_fs, h_fa);

  // free data
  assembler.free();
  meld.free();
  h_xs0.free();
  h_us.free();
  h_xa0.free();
  h_fa.free();
  h_us_ext.free();
  h_fa_ext.free();
  h_fs_ext.free();
}

template <bool nonlinear_strain>
void coupled_analysis(MPI_Comm &comm) {
  using T = double;

  // important user settings
  // -----------------------
  static constexpr int MELD_NN_PER_BLOCK = 32;
  double load_mag = 30.0; // 1.0 (small)

  // type definitions
  // ----------------

  std::ios::sync_with_stdio(false);  // always flush print immediately

  using Quad = QuadLinearQuadrature<T>;
  using Director = LinearizedRotation<T>;
  using Basis = LagrangeQuadBasis<T, Quad, 1>;
  using Geo = Basis::Geo;

  constexpr bool has_ref_axis = false;
  using Data = ShellIsotropicData<T, has_ref_axis>;
  using Physics = IsotropicShell<T, Data, nonlinear_strain>;
  using Assembler = MITCShellAssembler<T, Director, Basis, Physics, DeviceVec, BsrMat>;

  using AeroSolver = FixedAeroSolver<T, DeviceVec<T>>;
  using Transfer = MELD<T, MELD_NN_PER_BLOCK>;
  // nonlinear MELD has high loads on spars right now

  // build the Tacs prelim objects
  // -----------------------------

  double E = 70e9, nu = 0.3, thick = 0.02;  // material & thick properties

  // load the medium mesh for the struct mesh
  // uCRM mesh files can be found at:
  // https://data.niaid.nih.gov/resources?id=mendeley_gpk4zn73xn
  // ----------------------------------------

  TACSMeshLoader mesh_loader{comm};
  mesh_loader.scanBDFFile("CRM_box_2nd.bdf");
  auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));
  int ns = assembler.get_num_nodes();
  int ns_vars = assembler.get_num_vars();

  auto xs0 = assembler.getXpts();

  // load the coarse mesh for the aero surf
  // --------------------------------------

  int nx = 51;  // 101
  auto _assembler_aero = makeAeroSurfMesh<Assembler>(nx, nx);
  auto xa0 = _assembler_aero.getXpts();
  int na = _assembler_aero.get_num_nodes();
  auto h_xa0 = xa0.createHostVec();

  // perform a factorization on the rowPtr, colPtr (before creating matrix)
  auto& bsr_data = assembler.getBsrData();
  double fillin = 10.0;  // 10.0
  bool print = true;
  bsr_data.AMD_reordering();
  bsr_data.compute_full_LU_pattern(fillin, print);
  assembler.moveBsrDataToDevice();

  // compute loads
  auto d_loads = getSurfLoads<T>(_assembler_aero, load_mag);

  // setup kmat
  auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
  auto linear_solve = CUSPARSE::direct_LU_solve<T>;

  // make transfer scheme
  AeroSolver aero_solver = AeroSolver(na, d_loads);
  T beta = 0.1, Hreg = 1e-4;
  int sym = -1, nn = 128;
  bool meld_print = false;
  Transfer transfer =
      Transfer(xs0, xa0, beta, nn, sym, Hreg, meld_print);
  transfer.initialize();

  // make the solvers and transfer scheme
  // ------------------------------------

  if constexpr (nonlinear_strain) {
    // define coupled analysis types
    // -----------------------------
    using StructSolver = TacsNonlinearStaticNewton<T, Assembler>;

    int num_load_factors = 10, num_newton = 30;
    double abs_tol = 1e-8, rel_tol = 1e-8;
    bool struct_print = true;
    bool write_intermediate_vtk = false;
    TacsNonlinearStaticNewton<T, Assembler> struct_solver =
        TacsNonlinearStaticNewton<T, Assembler>(assembler, kmat, linear_solve,
                                                num_load_factors, num_newton,
                                                struct_print, abs_tol, rel_tol, write_intermediate_vtk);

    // test coupled driver
    testCoupledDriver<T>(struct_solver, aero_solver, transfer, assembler);
    // testCoupledDriverManual<T>(struct_solver, aero_solver, transfer, assembler,
    //                            _assembler_aero);

    struct_solver.writeSoln("out/uCRM_coupled_us.vtk");

    struct_solver.free();

  } else {
    using StructSolver = TacsLinearStatic<T, Assembler>;

    bool struct_print = true;
    auto struct_solver = TacsLinearStatic<T, Assembler>(
        assembler, kmat, linear_solve, struct_print);

    // test coupled driver
    // testCoupledDriver<T>(struct_solver, aero_solver, transfer, assembler);
    testCoupledDriverManual<T>(struct_solver, aero_solver, transfer,
    assembler, _assembler_aero);

    struct_solver.writeSoln("out/uCRM_coupled_us.vtk");

    struct_solver.free();
  }

  // free
  assembler.free();
  _assembler_aero.free();
  d_loads.free();
  aero_solver.free();
  transfer.free();
}

/**
 solve on CPU with cusparse for debugging
 **/
int main(int argc, char **argv) {
    /* command line args:
      ./2_coupled.out meld            to run meld
      ./2_coupled.out coupled         to run coupled nonlinear
      ./2_coupled.out coupled linear  to run linear case
    */

    // Intialize MPI and declare communicator
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
  
    bool run_linear = false;
    bool run_full_coupled = true;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        char* arg = argv[i];
        to_lowercase(arg);  // Make comparison case-insensitive

        if (strcmp(arg, "meld") == 0) {
            run_full_coupled = false;
        } else if (strcmp(arg, "coupled") == 0) {
            run_full_coupled = true;
        } else if (strcmp(arg, "--linear") == 0) {
            run_linear = true;
        } else {
            int rank;
            MPI_Comm_rank(comm, &rank);
            if (rank == 0) {
                std::cerr << "Unknown argument: " << argv[i] << std::endl;
                std::cerr << "Usage: " << argv[0] << " [meld|coupled] [--linear]" << std::endl;
            }
            MPI_Finalize();
            return 1;
        }
    }
  
    if (run_full_coupled) {
      // bool input here is constexpr bool nonlinear_strain
      if (run_linear) {
        coupled_analysis<false>(comm);
      } else {
        coupled_analysis<true>(comm);
      }
    } else {
        meld_demo(comm);
    }

    MPI_Finalize();

  return 0;
};