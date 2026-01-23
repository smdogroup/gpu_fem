// general gpu_fem imports
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"

// shell imports
#include "assembler.h"
#include "element/shell/director/linear_rotation.h"
#include "element/shell/physics/isotropic_shell.h"

// lagrange MITC element
#include "element/shell/basis/lagrange_basis.h"
#include "element/shell/mitc_shell.h"

// chebyshev element
#include "element/shell/basis/chebyshev_basis.h"
#include "element/shell/fint_shell.h"

// local multigrid imports
#include "multigrid/grid.h"
#include "multigrid/utils/fea.h"
#include "multigrid/smoothers/cheb4_poly.h"
#include "multigrid/smoothers/mc_smooth1.h"
#include "multigrid/prolongation/unstructured.h"
#include "multigrid/prolongation/structured.h"
#include "multigrid/solvers/gmg.h"
#include <string>
#include <chrono>

#include <type_traits>

// new multigrid imports for K-cycles, etc.
#include "_src/sa_amg.h"
#include "_src/_rigid_modes.cuh"
#include "multigrid/solvers/krylov/bsr_gmres.h"
#include "multigrid/solvers/krylov/bsr_pcg.h"

/* command line args:
    [direct/mg] [--nxe int] [--SR float] [--nvcyc int]
    * nxe must be power of 2

    examples:
    ./1_plate.out direct --nxe 2048 --SR 100.0    to run direct cylinder solve on 2048 x 2048 elem grid with slenderness ratio 100
    ./1_plate.out mg --nxe 2048 --SR 100.0    to run geometric multigrid cylinder solve on 2048 x 2048 elem grid with slenderness ratio 100
*/

void to_lowercase(char *str) {
    for (; *str; ++str) {
        *str = std::tolower(*str);
    }
}

template <typename T, class Assembler>
void amg_solve(MPI_Comm &comm, int level, double SR, int nsmooth, int ninnercyc, T omegas, T omegap, int ORDER) {
    // geometric multigrid method here..
    // need to make a number of grids..

    using Basis = typename Assembler::Basis;
    using Physics = typename Assembler::Phys;
    using Data = typename Physics::Data;
    const SCALER scaler  = LINE_SEARCH;
    const bool is_bsr = false; // no difference in intra-nodal (default old working prolong)
    using Prolongation = UnstructuredProlongation<Assembler, Basis, is_bsr>; 
    using FAssembler = FakeAssembler<T>;
    using Smoother = ChebyshevPolynomialSmoother<FAssembler>; // uses fake assembler for smoother so can also build on coarser grids
    using GRID = SingleGrid<Assembler, Prolongation, Smoother, LINE_SEARCH>;

    // const bool ORTHOG_PROJECTOR = true;
    const bool ORTHOG_PROJECTOR = false;
    using AMG = SmoothAggregationAMG<T, Smoother, ORTHOG_PROJECTOR>;
    using PCG = PCGSolver<T, GRID>;

    cublasHandle_t cublasHandle = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));
    cusparseHandle_t cusparseHandle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

    
    // make the fine grid assembler and loads
    // read the ESP/CAPS => nastran mesh for TACS
    TACSMeshLoader mesh_loader{comm};
    std::string fname = "../gmg/3_aob_wing/meshes/aob_wing_L" + std::to_string(level) + ".bdf";
    mesh_loader.scanBDFFile(fname.c_str());
    double E = 70e9, nu = 0.3, thick = 2.0 / SR;  // material & thick properties (start thicker first try)
    // TODO : run optimized design from AOB case
    printf("making assembler+GMG for mesh '%s'\n", fname.c_str());
    
    // create the TACS Assembler from the mesh loader
    auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));

    // create the loads (really only needed on finer mesh.. TBD how to setup nonlinear case..)
    int nvars = assembler.get_num_vars();
    int nnodes = nvars / 6;
    HostVec<T> h_loads(nvars);
    double load_mag = 10.0;
    double *my_loads = h_loads.getPtr();
    for (int inode = 0; inode < nnodes; inode++) {
        my_loads[6 * inode + 2] = load_mag;
    }

    // build the kmat
    auto &bsr_data = assembler.getBsrData();
    bsr_data.compute_nofill_pattern();
    assembler.moveBsrDataToDevice();
    auto loads = assembler.createVarsVec(my_loads);
    assembler.apply_bcs(loads);
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto soln = assembler.createVarsVec();
    int N = soln.getSize();
    int block_dim = bsr_data.block_dim; // should be 6 here
    // int nnodes = N / block_dim;
    auto fake_assembler = FAssembler(bsr_data, nnodes);
    auto d_bcs = assembler.getBCs();

    // assemble the kmat
    assembler.add_jacobian_fast(kmat);
    // delay bcs until after forming SA-aggregates

    auto start0 = std::chrono::high_resolution_clock::now();
    // build smoother on fine grid
    Smoother *fine_smoother = new Smoother(cublasHandle, cusparseHandle, fake_assembler, kmat, omegas, ORDER, nsmooth);
    
    // make fine rigid body modes array on device
    auto d_xpts = assembler.getXpts();
    auto fine_rbm = DeviceVec<T>(6 * N); // each of 6 rigid body modes
    k_compute_linear_rigid_body_modes<T><<<(nnodes + 31) / 32, 32>>>(nnodes, block_dim, d_xpts.getPtr(), fine_rbm.getPtr());

    T *h_xpts = d_xpts.createHostVec().getPtr();
    // printf("h_xpts: ");
    // printVec<T>(3 * nnodes, h_xpts);

    // make fine grid AMG solver
    // TODO : add coarse_node_th and sparse_th as command line inputs also
    int coarse_node_th = 600; // this value is problem dependent
    T sparse_th = 0.13;
    // omegaJac is not omegap input
    // T omegaJac = 0.3;
    // T omegaJac = 0.6; // for smooth prolongator (smaller is sometimes better, this should be another input)
    // T omegaJac = 1.8;
    // T omegaJac = 0.8631319920631012;
    // T sparse_th = 0.15; // instead of 0.25 for strength of connections
    printf("MAIN: build fine AMG solver\n");
    AMG *fine_amg = new AMG(cublasHandle, cusparseHandle, fine_smoother, nnodes, kmat, fine_rbm, coarse_node_th, sparse_th, omegap, nsmooth);
    assembler.apply_bcs(kmat); // now apply bcs after tentative aggregate pattern formed
    fine_amg->post_apply_bcs(d_bcs);
    auto end0 = std::chrono::high_resolution_clock::now();

    // assist in making smoothers at coarser levels
    printf("MAIN: build fine AMG solver\n");
    AMG *c_amg = fine_amg;
    bool first = true;
    while (c_amg != nullptr && c_amg->is_coarse_mg || first) {
        // build smoother for coarser problem (but it can't use assembler though..)
        auto c_bsr_data = c_amg->get_coarse_bsr_data();
        auto coarse_kmat = c_amg->get_coarse_kmat();
        int c_nnodes = c_amg->get_num_aggregates();
        printf("MAIN: build coarse system with %d aggregates\n", c_nnodes);
        auto fake_c_assembler = FAssembler(c_bsr_data, c_nnodes);
        Smoother *c_smoother = new Smoother(cublasHandle, cusparseHandle, fake_c_assembler, coarse_kmat, omegas, ORDER, nsmooth);

        // build coarser system
        printf("MAIN: build coarse system\n");
        c_amg->build_coarse_system(fake_c_assembler, c_smoother);
        printf("\tMAIN: done building coarse system\n");

        // then set current amg (c_amg) to coarser problem
        c_amg = c_amg->coarse_mg;
        first = false;
    }

    // build prolongation and fine grid also (unnecessary but required arg of PCG solver right now for some reason)
    auto prolongation = new Prolongation(cusparseHandle, assembler, 10);
    auto grid = new GRID(assembler, prolongation, fine_smoother, kmat, loads, cublasHandle, cusparseHandle);

    // now build PCG / Krylov solver with AMG as preconditioner
    // int level = 0;
    // create the preconditioner and GMRES solver now
    auto options = SolverOptions();
    options.ncycles = 800; // number of max PCG cycles
    options.print_freq = 10;

    printf("MAIN: build KRYLOV\n");

    // only use GMRES if SR > 100
    const int N_SUBSPACE = 100;
    using GMRES = GMRESSolver<T, GRID, N_SUBSPACE>;
    int MAX_ITER = N_SUBSPACE;
    auto pc = fine_amg;
    auto linear_solver = new GMRES(cublasHandle, cusparseHandle, grid, pc, options, MAX_ITER);
    // auto linear_solver = new PCG(cublasHandle, cusparseHandle, grid, pc, options, level);
    T init_resid = linear_solver->getResidualNorm(loads, soln);
    
    // out settings
    linear_solver->set_rel_tol(1e-6);
    linear_solver->set_abs_tol(1e-6);
    linear_solver->set_print(true);

    // perform the Krylov linear solve
    auto start1 = std::chrono::high_resolution_clock::now();
    bool check_conv = true;
    printf("MAIN: KRYLOV solve\n");
    linear_solver->solve(loads, soln, check_conv);
    // DEBUG: just pc
    // pc->solve(loads, soln);
    printf("\tMAIN: done with solve\n");
    T final_resid = linear_solver->getResidualNorm(loads, soln);

    CHECK_CUDA(cudaDeviceSynchronize());
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> startup_time = end0 - start0;
    std::chrono::duration<double> solve_time = end1 - start1;
    int ndof = N;
    double total = startup_time.count() + solve_time.count();
    // double mem_MB = kmg->get_memory_usage_mb();
    printf("AOB-wing SA-AMG solve, ndof %d : startup time %.2e, solve time %.2e, total %.2e", ndof, startup_time.count(), solve_time.count(), total);

    // compute log residual reduction per unit time
    T log_red_rate = (log(init_resid) - log(final_resid)) / log(10.0) / solve_time.count();
    printf("\tSA-AMG-GMRES on AOB-wing case with %d level and %.4e SR\n", level, SR);
    printf("\tinit resid %.4e => final resid %.4e in %.2e sec, log10(reduction)/sec = %.6e\n", init_resid, final_resid, solve_time.count(), log_red_rate);

    // print some of the data of host residual
    auto h_soln = soln.createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_soln, "out/cylinder_mg_lin.vtk");
}


template <typename T, class Assembler>
void solve_linear_direct(MPI_Comm &comm, int level, double SR) {
  
    using Basis = typename Assembler::Basis;
    using Physics = typename Assembler::Phys;
    using Data = typename Physics::Data;

  auto start0 = std::chrono::high_resolution_clock::now();

  TACSMeshLoader mesh_loader{comm};
  std::string fname = "../gmg/3_aob_wing/meshes/aob_wing_L" + std::to_string(level) + ".bdf";
  mesh_loader.scanBDFFile(fname.c_str());

  //   double E = 70e9, nu = 0.3, thick = 0.005;  // material & thick properties
  double E = 70e9, nu = 0.3, thick = 2.0 / SR;  // material & thick properties

  // make the assembler from the uCRM mesh
  auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));

  // TODO : set this in from optimized design from AOB case

  // BSR factorization
  auto& bsr_data = assembler.getBsrData();
  double fillin = 10.0;  // 10.0
  bool print = true;
  bsr_data.AMD_reordering();

//   // TRY INSTEAD Mc REORDERING
//   int num_colors, *_color_rowp, *nodal_num_comps, *node_geom_ind;
//   GRID::get_nodal_geom_indices(assembler, nodal_num_comps, node_geom_ind);
//   bsr_data.multicolor_junction_reordering_v2(node_geom_ind, num_colors, _color_rowp);

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

  // setup kmat and initial vecs
  auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
  auto soln = assembler.createVarsVec();
  auto res = assembler.createVarsVec();
  auto vars = assembler.createVarsVec();

  // assemble the kmat
  assembler.set_variables(vars);
  
  CHECK_CUDA(cudaDeviceSynchronize());
  auto starta = std::chrono::high_resolution_clock::now();
//   assembler.add_jacobian(res, kmat);
  assembler.add_jacobian_fast(kmat);
  CHECK_CUDA(cudaDeviceSynchronize());
  auto enda = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> assemb_time = enda - starta;

  assembler.apply_bcs(res);
  assembler.apply_bcs(kmat);

  CHECK_CUDA(cudaDeviceSynchronize());
  auto start1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> startup_time = start1 - start0;

  // solve the linear system
  CUSPARSE::direct_LU_solve(kmat, loads, soln);

  CHECK_CUDA(cudaDeviceSynchronize());
  auto end1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> solve_time = end1 - start1;
  std::chrono::duration<double> total_time = end1 - start0;

  size_t bytes_per_double = sizeof(double);
  double mem_mb = static_cast<double>(bytes_per_double) * static_cast<double>(bsr_data.nnzb) * 36.0 / 1024.0 / 1024.0;
  printf("direct LU solve on #dof %d, uses memory(MB) %.2e\n", nvars, mem_mb);
  printf("\tassembly %.2e and ovr startup %.2e, solve time %.2e and total time %.2e (sec)\n", assemb_time.count(), startup_time.count(), solve_time.count(), total_time.count());

  // print some of the data of host residual
  auto h_soln = soln.createHostVec();
  printToVTK<Assembler, HostVec<T>>(assembler, h_soln, "out/aob_direct_L" + std::to_string(level) + ".vtk");

  // free data
  assembler.free();
  h_loads.free();
  kmat.free();
  soln.free();
  res.free();
  vars.free();
  h_soln.free();
}

template <typename T, class Assembler>
void gatekeeper_method(std::string solver_type, MPI_Comm &comm, int level, double SR, int nsmooth, int ninnercyc, T omegas, T omegap, int ORDER) {
    if (solver_type != "direct") {
        amg_solve<T, Assembler>(comm, level, SR, nsmooth, ninnercyc, omegas, omegap, ORDER);
    } else {
        solve_linear_direct<T, Assembler>(comm, level, SR);
    }
}

int main(int argc, char **argv) {
    // Intialize MPI and declare communicator
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    // DEFAULTS
    int level = 1; // level mesh to solve.. level 4 also a good starting setting (big case)
    double SR = 100.0; // default
    double omegas = 0.3; // omega for smoother
    double omegap = 0.3; // omega for smooth prolongation
    int ORDER = 8; // for chebyshev

    int nsmooth = 1; // typically faster right now
    int ninnercyc = 1; // inner V-cycles to precond K-cycle
    std::string solver_type = "multigrid";

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        char* arg = argv[i];
        to_lowercase(arg);

        if (strcmp(arg, "--solver") == 0) {
            if (i + 1 < argc) {
                solver_type = argv[++i];
            } else {
                std::cerr << "Missing value for --solver\n";
                return 1;
            }
        } else if (strcmp(arg, "--level") == 0) {
            if (i + 1 < argc) {
                level = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --level\n";
                return 1;
            }
        } else if (strcmp(arg, "--omegas") == 0) {
            if (i + 1 < argc) {
                omegas = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --omegas\n";
                return 1;
            }
        } else if (strcmp(arg, "--omegap") == 0) {
            if (i + 1 < argc) {
                omegap = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --omegap\n";
                return 1;
            }
        } else if (strcmp(arg, "--sr") == 0) {
            if (i + 1 < argc) {
                SR = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --SR\n";
                return 1;
            }
        } else if (strcmp(arg, "--nsmooth") == 0) {
            if (i + 1 < argc) {
                nsmooth = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --nsmooth\n";
                return 1;
            }
        } else if (strcmp(arg, "--order") == 0) {
            if (i + 1 < argc) {
                ORDER = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --order\n";
                return 1;
            }
        } else if (strcmp(arg, "--ninnercyc") == 0) {
            if (i + 1 < argc) {
                ninnercyc = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --nsmooth\n";
                return 1;
            }
        } else {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            std::cerr << "Usage: " << argv[0] << " [direct/mg] [--nxe value] [--SR value] [--cycle char] [--nsmooth int] [--ninnercyc int]" << std::endl;
            return 1;
        }
    }

    // type specifications here
    using T = double;   
    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    constexpr bool has_ref_axis = false;
    constexpr bool is_nonlinear = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;

    // MITC4 shells
    using Basis = LagrangeQuadBasis<T, Quad, 1>;
    using Assembler = MITCShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;

    printf("AOB-wing mesh with MITC4 elements, level %d and SR %.2e\n------------\n", level, SR);
    gatekeeper_method<T, Assembler>(solver_type, comm, level, SR, nsmooth, ninnercyc, omegas, omegap, ORDER);

    return 0;

    
}