
#include "linalg/_linalg.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"
#include "solvers/_solvers.h"

// shell imports
#include "assembler.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/shell_elem_group.h"
#include "_src/crm_utils.h"


/* command line args:
    [linear|nonlinear]

    examples:
    ./3_unsteady.out linear      to run linear
    ./3_unsteady.out nonlinear   to run nonlinear
*/

// helper functions
// ----------------
// ----------------

template <typename T>
void vec_scale(int N, T scale, T *myvec) {
    for (int i = 0; i < N; i++) myvec[i] *= scale;
}

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

  using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
  using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;

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
  bsr_data.AMD_reordering();
  bsr_data.compute_full_LU_pattern(fillin, print);
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

  int ndof = assembler.get_num_vars();

    // setup kmat and initial vecs
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto mass_mat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto res = assembler.createVarsVec();

    // assemble mass matrix
    assembler.add_mass_jacobian(res, mass_mat, true);
    assembler.apply_bcs(mass_mat);

    // assemble the kmat
    assembler.add_jacobian(res, kmat);
    assembler.apply_bcs(res);
    assembler.apply_bcs(kmat);

    // time settings
    int num_timesteps = 1000;
    double dt = 0.01;

    // compute the forces on the structure
    T *h_forces = new T[ndof * num_timesteps];
    memset(h_forces, 0.0, ndof * num_timesteps * sizeof(T));
    for (int itime = 0; itime < num_timesteps; itime++) {
        // copy from static loads to unsteady loads
        memcpy(&h_forces[itime * ndof], h_loads_ptr, ndof * sizeof(T));
        T time = dt * itime;
        T omega = 1.431;
        // T omega = 4.0;
        T scale = 10.0 * std::sin(3.14159 * omega * time);
        // cblas_dscal(ndof, scale, &h_forces[itime * ndof], 1);
        vec_scale(ndof, scale, &h_forces[itime * ndof]);
    }
    auto forces = HostVec<T>(ndof * num_timesteps, h_forces).createDeviceVec();

    // create the linear gen alpha integrator
    auto integrator = LGAIntegrator(mass_mat, kmat, forces, ndof, num_timesteps, dt);

    // now solve and write to vtk
    print = true;
    integrator.solve(print);
    int stride = 2;
    integrator.writeToVTK<Assembler>(assembler, "out/ucrm_dyn", stride);

    integrator.free();
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

  using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
  using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;

  double E = 70e9, nu = 0.3, thick = 0.02;  // material & thick properties

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
  double load_mag = 1.0; // 9.0 with 40 load steps, now 15.0 with 70 load steps
  double *h_loads_ptr = h_loads.getPtr();
  for (int inode = 0; inode < nnodes; inode++) {
    h_loads_ptr[6 * inode + 2] = load_mag;
  }
  auto loads = h_loads.createDeviceVec();
  assembler.apply_bcs(loads);

  int ndof = assembler.get_num_vars();

    // setup kmat and initial vecs
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto mass_mat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto res = assembler.createVarsVec();

    // time settings
    int num_timesteps = 1000;
    double dt = 0.01;

    // compute the forces on the structure
    T *h_forces = new T[ndof * num_timesteps];
    memset(h_forces, 0.0, ndof * num_timesteps * sizeof(T));
    for (int itime = 0; itime < num_timesteps; itime++) {
        // copy from static loads to unsteady loads
        memcpy(&h_forces[itime * ndof], h_loads_ptr, ndof * sizeof(T));
        T time = dt * itime;
        T omega = 1.431;
        // T omega = 4.0;
        T scale = 1.0 * std::sin(3.14159 * omega * time);
        // cblas_dscal(ndof, scale, &h_forces[itime * ndof], 1);
        vec_scale(ndof, scale, &h_forces[itime * ndof]);
    }
    auto forces = HostVec<T>(ndof * num_timesteps, h_forces).createDeviceVec();

    int print_freq = 10, max_newton_steps = 30;
    T rel_tol = 1e-15, abs_tol = 1e-8;
    bool lin_print = false;

    // create the linear gen alpha integrator
    auto solve_func = CUSPARSE::direct_LU_solve<T>;
    auto integrator = NLGAIntegrator<Assembler>(
        solve_func, assembler, mass_mat, kmat, 
        forces, ndof, num_timesteps, dt, print_freq, 
        rel_tol, abs_tol, max_newton_steps, lin_print);

    // now solve and write to vtk
    print = true;
    integrator.solve(print);
    int stride = 2;
    integrator.writeToVTK(assembler, "out/ucrm_dyn", stride);
    integrator.free();
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

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        char* arg = argv[i];
        to_lowercase(arg);  // Make comparison case-insensitive

        if (strcmp(arg, "linear") == 0) {
            run_linear = true;
        } else if (strcmp(arg, "nonlinear") == 0) {
            run_linear = false;
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
        solve_linear(comm);
    } else {
        solve_nonlinear(comm);
    }

    MPI_Finalize();
    return 0;
};
