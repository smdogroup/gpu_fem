
#include "linalg/_linalg.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"
#include "solvers/_solvers.h"

// shell imports
#include "assembler.h"
#include "element/shell/director/linear_rotation.h"
#include "element/shell/physics/isotropic_shell.h"

// lagrange MITC element
#include "element/shell/basis/lagrange_basis.h"
#include "element/shell/mitc_shell.h"

// case utils
#include "_src/crm_utils.h"


/* command line args:
    [linear|nonlinear] [--iterative]

    examples:
    ./1_static.out linear      to run linear
    ./1_static.out nonlinear   to run nonlinear
    add the option --iterative to make it switch from full_LU (only for linear)
*/

// helper functions
// ----------------
// ----------------

void solve_linear(MPI_Comm &comm, bool full_LU = true) {
  using T = double;

  auto start0 = std::chrono::high_resolution_clock::now();

  // uCRM mesh files can be found at:
  // https://data.niaid.nih.gov/resources?id=mendeley_gpk4zn73xn
  TACSMeshLoader mesh_loader{comm};
  mesh_loader.scanBDFFile("CRM_box_2nd.bdf");
  // mesh_loader.scanBDFFile("uCRM-135_wingbox_medium.bdf");

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

  // make the assembler from the uCRM mesh
  auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));

  int ndvs = assembler.get_num_dvs();
  printf("ndvs %d\n", ndvs);
  T thick2 = 1e-2;
  HostVec<T> h_dvs(ndvs, thick2);
  auto global_dvs = h_dvs.createDeviceVec();
  assembler.set_design_variables(global_dvs);

  // T mass = assembler._compute_mass();
  // printf("mass %.4e\n", mass);

  // BSR factorization
  auto& bsr_data = assembler.getBsrData();
  double fillin = 10.0;  // 10.0
  bool print = true;
  if (full_LU) {
    bsr_data.AMD_reordering();
    bsr_data.compute_full_LU_pattern(fillin, print);
  } else {
    bsr_data.RCM_reordering();
    bsr_data.qorder_reordering(0.5); // qordering not working well for some reason..
    bsr_data.compute_ILUk_pattern(0, fillin); // 10, 20 (for BiCGStab)
    // bsr_data.compute_full_LU_pattern(fillin, print);
  }
  assembler.moveBsrDataToDevice();

  // get the loads
  int nvars = assembler.get_num_vars();
  int nnodes = assembler.get_num_nodes();
  HostVec<T> h_loads(nvars);
  double load_mag = 10.0;
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
  auto vars = assembler.createVarsVec();

  // assemble the kmat
  assembler.set_variables(vars);
  // assembler.add_jacobian(res, kmat);
  assembler.add_jacobian_fast(kmat);
  assembler.apply_bcs(res);
  assembler.apply_bcs(kmat);

  
  // // now add diag nugget to stabilize ILU factor
  // kmat.add_diag_to_each_block(1e-3); // 1e-3 relative to trace abs value

  // solve the linear system
  if (full_LU) {
      CUSPARSE::direct_LU_solve(kmat, loads, soln);
  } else {
      int n_iter = 200, max_iter = 200;
      T abs_tol = 1e-11, rel_tol = 1e-15;
      bool print = true;
      constexpr bool right = true, modifiedGS = true; // better with modifiedGS true, yeah it is..
      CUSPARSE::GMRES_solve<T, right, modifiedGS>(kmat, loads, soln, n_iter, max_iter, abs_tol, rel_tol, print);
      // CUSPARSE::BiCGStab_solve<T>(kmat, loads, soln, n_iter, abs_tol, rel_tol, print);
      // CUSPARSE::PCG_solve<T>(kmat, loads, soln, n_iter, abs_tol, rel_tol, print);
      // CUSPARSE::GMRES_DR_solve<T, right, modifiedGS>(kmat, loads, soln, 4, 2, 8, abs_tol, rel_tol, print, true, 1);
  }

  // print some of the data of host residual
  auto h_soln = soln.createHostVec();
  printToVTK<Assembler, HostVec<T>>(assembler, h_soln, "out/uCRM.vtk");

  // free data
  assembler.free();
  h_loads.free();
  kmat.free();
  soln.free();
  res.free();
  vars.free();
  h_soln.free();
}

void solve_nonlinear(MPI_Comm &comm) {
  using T = double;

  auto start0 = std::chrono::high_resolution_clock::now();

  // uCRM mesh files can be found at:
  // https://data.niaid.nih.gov/resources?id=mendeley_gpk4zn73xn
  TACSMeshLoader mesh_loader{comm};
  mesh_loader.scanBDFFile("CRM_box_2nd.bdf");
  // mesh_loader.scanBDFFile("uCRM-135_wingbox_medium.bdf");

  using Quad = QuadLinearQuadrature<T>;
  using Director = LinearizedRotation<T>;
  using Basis = LagrangeQuadBasis<T, Quad, 1>;
  using Geo = Basis::Geo;

  constexpr bool has_ref_axis = false;
  constexpr bool is_nonlinear = true;
  using Data = ShellIsotropicData<T, has_ref_axis>;
  using Physics = IsotropicShell<T, Data, is_nonlinear>;
  using Assembler = MITCShellAssembler<T, Director, Basis, Physics, DeviceVec, BsrMat>;

  double E = 70e9, nu = 0.3, thick = 0.02;  // material & thick properties
  // double E = 70e9, nu = 0.3, thick = 0.005;  // material & thick properties

  // make the assembler from the uCRM mesh
  auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));

  // BSR factorization
  auto& bsr_data = assembler.getBsrData();
  double fillin = 10.0;  // 10.0
  bool print = true;
  bsr_data.AMD_reordering();
  bsr_data.compute_full_LU_pattern(fillin, print);
  assembler.moveBsrDataToDevice();

  // get the loads
  int nvars = assembler.get_num_vars();
  int nnodes = assembler.get_num_nodes();
  HostVec<T> h_loads(nvars);
  double load_mag = 15.0; // 9.0 with 40 load steps, now 15.0 with 70 load steps
  // double load_mag = 1.0;
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
  int num_load_factors = 70, num_newton = 50;
  T min_load_factor = 0.1, max_load_factor = 23.0, abs_tol = 1e-8,
    rel_tol = 1e-8;
  auto solve_func = CUSPARSE::direct_LU_solve<T>;
  bool write_vtk = true;
  std::string outputPrefix = "out/uCRM_";

  const bool fast_assembly = true;
  // const bool fast_assembly = false;
  newton_solve<T, BsrMat<DeviceVec<T>>, DeviceVec<T>, Assembler, fast_assembly>(
      solve_func, kmat, loads, soln, assembler, res, rhs, vars,
      num_load_factors, min_load_factor, max_load_factor, num_newton, abs_tol,
      rel_tol, outputPrefix, print, write_vtk);

  // print some of the data of host residual
  auto h_vars = vars.createHostVec();
  printToVTK<Assembler, HostVec<T>>(assembler, h_vars, "out/uCRM_nl.vtk");

  // free data
  assembler.free();
  h_loads.free();
  kmat.free();
  soln.free();
  res.free();
  vars.free();
  h_vars.free();
  rhs.free();
}

int main(int argc, char **argv) {
    /* command line args:
       ./1_static.out linear      to run linear
       ./1_static.out nonlinear   to run nonlinear
       add the option --iterative to make it switch from full_LU (only for linear)
    */

    // Intialize MPI and declare communicator
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    bool run_linear = false;
    bool full_LU = true;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        char* arg = argv[i];
        to_lowercase(arg);  // Make comparison case-insensitive

        if (strcmp(arg, "linear") == 0) {
            run_linear = true;
        } else if (strcmp(arg, "nonlinear") == 0) {
            run_linear = false;
        } else if (strcmp(arg, "--iterative") == 0) {
            full_LU = false;
        } else {
            int rank;
            MPI_Comm_rank(comm, &rank);
            if (rank == 0) {
                std::cerr << "Unknown argument: " << argv[i] << std::endl;
                std::cerr << "Usage: " << argv[0] << " [linear|nonlinear] [--iterative]" << std::endl;
            }
            MPI_Finalize();
            return 1;
        }
    }
  
    if (run_linear) {
        solve_linear(comm, full_LU);
    } else {
        solve_nonlinear(comm);
    }

    MPI_Finalize();
    return 0;
};
