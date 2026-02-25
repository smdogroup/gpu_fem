
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

// chebyshev element
#include "element/shell/basis/chebyshev_basis.h"
#include "element/shell/fint_shell.h"

// local multigrid imports
#include "multigrid/grid.h"
#include "multigrid/utils/fea.h"
#include "multigrid/smoothers/_wingbox_coloring.h"
#include "multigrid/smoothers/cheb4_poly.h"
#include "multigrid/smoothers/asw_unstruct.h"
#include "multigrid/prolongation/unstructured.h"
#include "multigrid/solvers/gmg.h"
#include <string>
#include <chrono>

// new multigrid imports for K-cycles, etc.
#include "multigrid/solvers/solve_utils.h"
#include "multigrid/solvers/direct/cusp_directLU.h"
#include "multigrid/solvers/krylov/bsr_pcg.h"
#include "multigrid/solvers/multilevel/kcycle.h"
#include "multigrid/solvers/multilevel/twolevel.h"

/* argparse options:
[mg/direct/debug] [--level int]
*/

void to_lowercase(char *str) {
    for (; *str; ++str) {
        *str = std::tolower(*str);
    }
}

std::string time_string(int itime) {
    std::string _time = std::to_string(itime);
    if (itime < 10) {
        return "00" + _time;
    } else if (itime < 100) {
        return "0" + _time;
    } else {
        return _time;
    }
}

template <typename T, class Assembler>
void solve_linear_multigrid(MPI_Comm &comm, int level, double SR, int nsmooth, int ninnercyc, T omega, std::string cycle_type) {
    // geometric multigrid method here..
    // need to make a number of grids..
    // level gives the finest level here..

    using Basis = typename Assembler::Basis;
    using Physics = typename Assembler::Phys;
    using Data = typename Physics::Data;
    // using Smoother = ChebyshevPolynomialSmoother<Assembler>;
    using Smoother = UnstructuredQuadElementAdditiveSchwarzSmoother<T, Assembler>;
    const bool is_bsr = true; // need this one if want to smooth prolongation
    // const bool is_bsr = false; // no difference in intra-nodal (default old working prolong)
    using Prolongation = UnstructuredProlongation<Assembler, Basis, is_bsr>; 
    using GRID = SingleGrid<Assembler, Prolongation, Smoother, LINE_SEARCH>;
    using CoarseSolver = CusparseMGDirectLU<T, Assembler>;
    using MG = GeometricMultigridSolver<GRID, CoarseSolver>;

    // for K-cycles
    using KrylovSolve = PCGSolver<T, GRID>;
    using TwoLevelSolve = MultigridTwoLevelSolver<GRID>;
    using KMG = MultilevelKcycleSolver<GRID, CoarseSolver, TwoLevelSolve, KrylovSolve>;

    auto start0 = std::chrono::high_resolution_clock::now();

    // create cublas and cusparse handles (single one each)
    // -----------------------------------------------------
    cublasHandle_t cublasHandle = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));
    cusparseHandle_t cusparseHandle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

    // order of chebyshev polynomial smoother (can do higher for greater smoothing effect)
    // int ORDER = 4;
    // int ORDER = 8;

    // hopefully this doesn't construct the object?
    MG *mg;
    KMG *kmg;

    bool is_kcycle = cycle_type == "K";
    if (is_kcycle) {
        kmg = new KMG();
    } else {
        mg = new MG();
    }

    // make each wing multigrid object.. with L0 the coarsest mesh, L3 finest 
    //   (this way mg.grids is still finest to coarsest meshes order by convention)
    for (int i = level; i >= 0; i--) {

        // read the ESP/CAPS => nastran mesh for TACS
        TACSMeshLoader mesh_loader{comm};
        std::string fname = "meshes/aob_wing_L" + std::to_string(i) + ".bdf";
        mesh_loader.scanBDFFile(fname.c_str());
        double E = 70e9, nu = 0.3, thick = 2.0 / SR;  // material & thick properties (start thicker first try)
        // TODO : run optimized design from AOB case
        printf("making assembler+GMG for mesh '%s'\n", fname.c_str());
        
        // create the TACS Assembler from the mesh loader
        auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));

        // create the loads (really only needed on finer mesh.. TBD how to setup nonlinear case..)
        int nvars = assembler.get_num_vars();
        int nnodes = assembler.get_num_nodes();
        HostVec<T> h_loads(nvars);
        double load_mag = 10.0;
        double *my_loads = h_loads.getPtr();
        for (int inode = 0; inode < nnodes; inode++) {
            my_loads[6 * inode + 2] = load_mag;
        }

        // do multicolor junction reordering
        auto &bsr_data = assembler.getBsrData();
        int num_colors, *_color_rowp;

        bool coarsest_grid = i == 0;
        if (!coarsest_grid) {
            // don't do coloring for additive schwarz smoother..
            // WingboxMultiColoring<Assembler>::apply_coloring(assembler, bsr_data, num_colors, _color_rowp);
            bsr_data.compute_nofill_pattern();
        } else {
            // full LU pattern for coarsest grid
            bsr_data.AMD_reordering();
            bsr_data.compute_full_LU_pattern(10.0, false);
            num_colors = 0;
            _color_rowp = new int[2];
            _color_rowp[0] = 0, _color_rowp[1] = nnodes;
        }
        auto h_color_rowp = HostVec<int>(num_colors + 1, _color_rowp);
        assembler.moveBsrDataToDevice();

        // now compute loads, bcs and assemble kmat
        auto loads = assembler.createVarsVec(my_loads);
        assembler.apply_bcs(loads);
        auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
        auto vars = assembler.createVarsVec();
        assembler.set_variables(vars);
        auto res = assembler.createVarsVec();
        auto starta = std::chrono::high_resolution_clock::now();
        // assembler.add_jacobian(res, kmat);
        const int elems_per_blockk = 1; // 1 versus 2 elements => similar runtime (1 slightly better)
        // const int elems_per_blockk = 2;
        assembler.template add_jacobian_fast<elems_per_blockk>(kmat);
        assembler.apply_bcs(kmat);
        CHECK_CUDA(cudaDeviceSynchronize());
        auto enda = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> assembly_time = enda - starta;
        printf("\tassemble kmat in %.2e sec\n", assembly_time.count());

        // build smoother and prolongations
        // auto smoother = new Smoother(cublasHandle, cusparseHandle, assembler, kmat, omega, ORDER); // this was chebyshev
        auto smoother = new Smoother(cublasHandle, cusparseHandle, assembler, kmat, 
            omega, nsmooth);
        int ELEM_MAX = 10; // num nearby elements of each fine node for nz pattern construction
        // int ELEM_MAX = 4;
        auto prolongation = new Prolongation(cusparseHandle, assembler, ELEM_MAX);
        auto grid = GRID(assembler, prolongation, smoother, kmat, loads, cublasHandle, cusparseHandle);

	    // testing this new feature out
        // smoother->setup_cg_lanczos(grid.d_defect); // was for CG-lanczos

        if (is_kcycle) {
            kmg->grids.push_back(grid);
        } else {
            mg->grids.push_back(grid);
            if (coarsest_grid) mg->coarse_solver = new CoarseSolver(cublasHandle, cusparseHandle, 
                assembler, kmat);
        }
    }

    // register the coarse assemblers to the prolongations..
    if (is_kcycle) {
        kmg->template init_prolongations<Basis>();
    } else {
        mg->template init_prolongations<Basis>();
    }

    auto end0 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> startup_time = end0 - start0;

    T init_resid_nrm = is_kcycle ? kmg->grids[0].getResidNorm() : mg->grids[0].getResidNorm();

    printf("starting %s cycle solve\n", cycle_type.c_str());
    int pre_smooth = nsmooth, post_smooth = nsmooth;
    // best was V(4,4) before
    // bool print = false;
    bool print = true;
    T atol = 1e-6, rtol = 1e-6;
    int n_cycles = SR >= 100.0 ? 1000 : 200;
    // bool time = false;
    bool time = true;
    int print_freq = 5;

    bool double_smooth = false;
    // bool double_smooth = true; // true tends to be slightly faster sometimes

    if (is_kcycle) {
        int n_krylov = 500;
        kmg->init_outer_solver(cublasHandle, cusparseHandle, nsmooth, ninnercyc, 
            n_krylov, omega, atol, rtol, print_freq, print, double_smooth);    
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    auto start1 = std::chrono::high_resolution_clock::now();

    if (is_kcycle) {
        int nlevels = kmg->grids.size();
        for (int i = 0; i < nlevels; i++) {
            kmg->grids[i].smoother->factor();
        } 
    } else {
        int nlevels = mg->grids.size();
        for (int i = 0; i < nlevels; i++) {
            mg->grids[i].smoother->factor();
        } 
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end_factor = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> asw_factor_time = end_factor - start1;
    printf("ASW factor time %.4e\n", asw_factor_time.count());

    // fastest is K-cycle usually
    if (cycle_type == "V") {
        mg->vcycle_solve(0, pre_smooth, post_smooth, n_cycles, print, atol, rtol, double_smooth, print_freq, time); //(good option)
    } else if (cycle_type == "W") {
        mg->wcycle_solve(0, pre_smooth, post_smooth, n_cycles, print, atol, rtol);
    } else if (cycle_type == "F") {
        mg->fcycle_solve(0, pre_smooth, post_smooth, n_cycles, print, atol, rtol, double_smooth, print_freq, time); // also decent
    } else if (cycle_type == "K") {
        kmg->solve();
    }
    

    CHECK_CUDA(cudaDeviceSynchronize());
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = end1 - start1;
    int ndof = cycle_type == "K" ? kmg->grids[0].N : mg->grids[0].N;
    double total = startup_time.count() + solve_time.count();
    double mem_MB = is_kcycle ? kmg->get_memory_usage_mb() : mg->get_memory_usage_mb();
    printf("wingbox GMG solve, ndof %d : startup time %.2e, solve time %.2e, total %.2e, with mem(MB) %.2e\n", ndof, startup_time.count(), solve_time.count(), total, mem_MB);

    if (is_kcycle) {
        // double check with true resid nrm
        T resid_nrm = kmg->grids[0].getResidNorm();
        printf("init resid_nrm = %.2e => final resid_nrm = %.2e\n", init_resid_nrm, resid_nrm);

        // print some of the data of host residual
        int *d_perm = kmg->grids[0].d_perm;
        auto h_soln = kmg->grids[0].d_soln.createPermuteVec(6, d_perm).createHostVec();
        printToVTK<Assembler,HostVec<T>>(kmg->grids[0].assembler, h_soln, "out/aob_wing_mg.vtk");
    } else {
        // double check with true resid nrm
        T resid_nrm = mg->grids[0].getResidNorm();
        printf("init resid_nrm = %.2e => final resid_nrm = %.2e\n", init_resid_nrm, resid_nrm);

        // print some of the data of host residual
        int *d_perm = mg->grids[0].d_perm;
        auto h_soln = mg->grids[0].d_soln.createPermuteVec(6, d_perm).createHostVec();
        printToVTK<Assembler,HostVec<T>>(mg->grids[0].assembler, h_soln, "out/aob_wing_mg.vtk");
    }
}

template <typename T, class Assembler>
void solve_linear_direct(MPI_Comm &comm, int level, double SR) {
  
    using Basis = typename Assembler::Basis;
    using Physics = typename Assembler::Phys;
    using Data = typename Physics::Data;

  auto start0 = std::chrono::high_resolution_clock::now();

  TACSMeshLoader mesh_loader{comm};
  std::string fname = "meshes/aob_wing_L" + std::to_string(level) + ".bdf";
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
void gatekeeper_method(bool is_multigrid, MPI_Comm &comm, int level, double SR, int nsmooth, int ninnercyc, T omega, std::string cycle_type) {
    if (is_multigrid) {
        solve_linear_multigrid<T, Assembler>(comm, level, SR, nsmooth, ninnercyc, omega, cycle_type);
    } else {
        solve_linear_direct<T, Assembler>(comm, level, SR);
    }
}

int main(int argc, char **argv) {

    // Intialize MPI and declare communicator
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    // DEFAULTS
    int level = 3; // level mesh to solve.. level 4 also a good starting setting (big case)
    bool is_multigrid = true;
    // bool is_debug = false;
    double SR = 300.0;
    // low SR can use less smoothing and inner cyc
    int nsmooth = 4; // may need more here (esp for MITC elements, but CFI can use less)
    int ninnercyc = 1; // inner V-cycles to precond K-cycle
    // omega = 0.1 or 0.15 before spectral radius
    // but omega < 1, particularly omega = 0.3 tends to conv fastest for me
    double omega = 0.15; // starting omega for AOB wing (slightly better than 0.1 and 0.2 or 0.25)
    std::string cycle_type = "K"; // "V", "F", "W", "K"
    
    // MITC4 loses some performance, CFI4 faster for linear (but can't use for NL cause locking needs line searches)
    std::string elem_type = "MITC4";
    // std::string elem_type = "CFI4"; // 'MITC4', 'CFI4', 'CFI9'

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        char* arg = argv[i];
        to_lowercase(arg);

        if (strcmp(arg, "direct") == 0) {
            is_multigrid = false;
        } else if (strcmp(arg, "mg") == 0) {
            is_multigrid = true;
        } else if (strcmp(arg, "--sr") == 0) {
            if (i + 1 < argc) {
                SR = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --SR\n";
                return 1;
            }
        } else if (strcmp(arg, "--omega") == 0) {
            if (i + 1 < argc) {
                omega = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --omega\n";
                return 1;
            }
        } else if (strcmp(arg, "--level") == 0) {
            if (i + 1 < argc) {
                level = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --level\n";
                return 1;
            }
        } else if (strcmp(arg, "--cycle") == 0) {
            if (i + 1 < argc) {
                cycle_type = argv[++i];
            } else {
                std::cerr << "Missing value for --level\n";
                return 1;
            }
        } else if (strcmp(arg, "--elem") == 0) {
            if (i + 1 < argc) {
                elem_type = argv[++i];
            } else {
                std::cerr << "Missing value for --elem\n";
                return 1;
            }
        } else if (strcmp(arg, "--nsmooth") == 0) {
            if (i + 1 < argc) {
                nsmooth = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --nsmooth\n";
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
            std::cerr << "Usage: " << argv[0] << " [direct/mg] [--level int] [--SR double] [--cycle char] [--nsmooth int] [--ninnercyc int]" << std::endl;
            return 1;
        }
    }

    // type specifications here
    using T = double;   
    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    constexpr bool has_ref_axis = false;
    constexpr bool is_nonlinear = false;
    // constexpr bool is_nonlinear = true;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;

    printf("AOB mesh with %s elements, level %d and SR %.2e\n------------\n", elem_type.c_str(), level, SR);
    if (elem_type == "MITC4") {
        using Basis = LagrangeQuadBasis<T, Quad, 1>;
        using Assembler = MITCShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
        gatekeeper_method<T, Assembler>(is_multigrid, comm, level, SR, nsmooth, ninnercyc, omega, cycle_type);
    } else if (elem_type == "CFI4") {
        using Basis = ChebyshevQuadBasis<T, Quad, 1>;
        using Assembler = FullyIntegratedShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
        gatekeeper_method<T, Assembler>(is_multigrid, comm, level, SR, nsmooth, ninnercyc, omega, cycle_type);
    } else if (elem_type == "CFI9") {
        using Basis = ChebyshevQuadBasis<T, Quad, 2>;
        using Assembler = FullyIntegratedShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
        gatekeeper_method<T, Assembler>(is_multigrid, comm, level, SR, nsmooth, ninnercyc, omega, cycle_type);
    } else {
        printf("ERROR : didn't run anything, elem type not in available types (see main function)\n");
    }

    MPI_Finalize();
    return 0;
};
