
#include "linalg/_linalg.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"
#include "solvers/_solvers.h"

// case imports
#include "_src/comp_reader.h"

// new nonlinear solvers
#include "solvers/nonlinear_static/inexact_newton.h"
#include "solvers/nonlinear_static/continuation.h"

// shell imports
#include "assembler.h"
#include "element/shell/director/linear_rotation.h"
#include "element/shell/physics/iso_stiff_shell.h"

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

template <typename T>
T get_max_disp(DeviceVec<T> &d_soln, int idof = 2) {
    T *h_soln = d_soln.createHostVec().getPtr();
    int nvars = d_soln.getSize();
    int nnodes = nvars / 6;
    T my_max = 0.0;
    for (int inode = 0; inode < nnodes; inode++) {
        T val = abs(h_soln[6 * inode + idof]);
        if (val > my_max) my_max = val;
    }
    return my_max;
}



template <typename T, class Assembler>
void solve_nonlinear_multigrid(MPI_Comm &comm, int level, 
    int nsmooth, int ninnercyc, std::string cycle_type, T omega, int ORDER, T omega_min = 0.25, T omega_max = 1.0, int n_krylov = 50, double total_force = 1.0) {
    // geometric multigrid method here..
    // need to make a number of grids..
    // level gives the finest level here..

    using Basis = typename Assembler::Basis;
    using Physics = typename Assembler::Phys;
    using Data = typename Physics::Data;
    using Smoother = ChebyshevPolynomialSmoother<Assembler>;
    const bool is_bsr = true; // need this one if want to smooth prolongation
    // const bool is_bsr = false; // no difference in intra-nodal (default old working prolong)
    using Prolongation = UnstructuredProlongation<Assembler, Basis, is_bsr>; 
    
    // in linear case (CFI4 + line search gets around locking issue, cause needs to be CFI9 order or greater to avoid locking, thus improving high slender perf)
    // in nonlinear case though (Vcycle line search breaks down at high NL and CFI4 can't be used, so need MITC4 and no line search for now more robust)
    // I had slowly realized that omega larger to 1 (then just omega_LS = 1 aka no rescaling was better)
    using GRID = SingleGrid<Assembler, Prolongation, Smoother, LINE_SEARCH>;
    // using GRID = SingleGrid<Assembler, Prolongation, Smoother, NONE>; // scales by one each time
    // wondering if NONE may help the Vcycle in hard NL case
    using CoarseSolver = CusparseMGDirectLU<T, Assembler>;
    using MG = GeometricMultigridSolver<GRID, CoarseSolver>;

    // for K-cycles
    // constexpr bool full_approx_scheme = true;
    constexpr bool full_approx_scheme = false; // not fully implemented yet so false
    using KrylovSolve = PCGSolver<T, GRID>;
    using TwoLevelSolve = MultigridTwoLevelSolver<GRID, full_approx_scheme>;
    using KMG = MultilevelKcycleSolver<GRID, CoarseSolver, TwoLevelSolve, KrylovSolve>;

    // create cublas and cusparse handles (single one each)
    // -----------------------------------------------------
    cublasHandle_t cublasHandle = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));
    cusparseHandle_t cusparseHandle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

    auto start0 = std::chrono::high_resolution_clock::now();

    // hopefully this doesn't construct the object?
    MG *mg;
    KMG *kmg;

    bool is_kcycle = cycle_type == "K";
    if (!is_kcycle) {
        printf("non K-cycle not supported for this script yet (just would have to change it a bit) with one or two function calls or another routine maybe\n");
        return;
    }

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
        std::string fname = "../../multigrid/3_aob_wing/meshes/aob_wing_L" + std::to_string(i) + ".bdf";
        mesh_loader.scanBDFFile(fname.c_str());

        // now set component data using new helper method
        HostVec<Data> comp_data(mesh_loader.getNumComponents());
        std::string design_filename = "design/AOB-design.txt";
        build_AOB_component_data<T, Data>(mesh_loader, comp_data, design_filename);
        
        printf("making assembler+GMG for mesh '%s'\n", fname.c_str());
        // create the TACS Assembler from the mesh loader
        auto assembler = Assembler::createFromBDFComponent(mesh_loader, comp_data);
        printf("\tdone making assembler\n");

        // create the loads (really only needed on finer mesh.. TBD how to setup nonlinear case..)
        int nvars = assembler.get_num_vars();
        int nnodes = assembler.get_num_nodes();
        HostVec<T> h_loads(nvars);
        int level_exp = 1 << level;
        // printf("level_exp %d\n", level_exp);
        double load_mag = total_force / nnodes; // estimate for nodal load mag
        double *my_loads = h_loads.getPtr();
        for (int inode = 0; inode < nnodes; inode++) {
            my_loads[6 * inode + 2] = load_mag;
        }
        // TODO : change this to only put loads on upper + lower skins..

        // do multicolor junction reordering
        printf("perform coloring\n");
        auto &bsr_data = assembler.getBsrData();
        int num_colors, *_color_rowp;

        bool coarsest_grid = i == 0;
        if (!coarsest_grid) {
            WingboxMultiColoring<Assembler>::apply_coloring(assembler, bsr_data, num_colors, _color_rowp);
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
        // printf("assemble jacobian\n");
        assembler.template add_jacobian_fast<elems_per_blockk>(kmat);
        CHECK_CUDA(cudaDeviceSynchronize());
        // printf("\tdone assemble jacobian\n");
        // return; // temp debug
        assembler.apply_bcs(kmat);
        CHECK_CUDA(cudaDeviceSynchronize());
        auto enda = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> assembly_time = enda - starta;
        printf("\tassemble kmat time %.2e\n", assembly_time.count());

        // build smoother and prolongations
        // bool smooth_debug = true;
        bool smooth_debug = false;
        auto smoother = new Smoother(cublasHandle, cusparseHandle, assembler, kmat, omega, ORDER, 1, smooth_debug);
        int ELEM_MAX = 10; // num nearby elements of each fine node for nz pattern construction
        // int ELEM_MAX = 4;
        auto prolongation = new Prolongation(cusparseHandle, assembler, ELEM_MAX);
        auto grid = GRID(assembler, prolongation, smoother, kmat, loads, 
            cublasHandle, cusparseHandle, omega_min, omega_max);

        smoother->setup_cg_lanczos(grid.d_defect);

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

    CHECK_CUDA(cudaDeviceSynchronize());
    auto start1 = std::chrono::high_resolution_clock::now();
    printf("starting %s cycle solve\n", cycle_type.c_str());
    int pre_smooth = nsmooth, post_smooth = nsmooth;
    bool print = false; // true
    T atol = 1e-4, rtol = 1e-6;
    bool time = true; // false
    int print_freq = 5;
    bool double_smooth = true; // true tends to be slightly faster sometimes

    if (is_kcycle) {
        kmg->init_outer_solver(cublasHandle, cusparseHandle, nsmooth, ninnercyc, 
            n_krylov, omega, atol, rtol, print_freq, print, double_smooth);    
    }

    std::vector<GRID>& grids = is_kcycle ? kmg->grids : mg->grids;

    // fine grid states
    auto& fine_assembler = grids[0].assembler;
    auto fine_soln = fine_assembler.createVarsVec();
    auto fine_res = fine_assembler.createVarsVec();
    auto fine_rhs = fine_assembler.createVarsVec();
    auto fine_loads = fine_assembler.createVarsVec();
    auto fine_vars = fine_assembler.createVarsVec();
    auto& fine_kmat = grids[0].Kmat;

    // get fine loads from fine grid init rhs
    bool perm_out = true;
    grids[0].getDefect(fine_loads, perm_out);
    fine_assembler.apply_bcs(fine_loads);

    // 1) do a linear solve here
    // -------------------------------------------------------

    kmg->set_print(true);
    bool fail = kmg->solve();
    kmg->set_print(false);
    if (fail) {
        printf("K-GMG linear solve failed\n");
    } else {
        printf("K-GMG linear solve succeeded\n");
    }
    int *d_perm = kmg->grids[0].d_perm;
    auto h_soln = kmg->grids[0].d_soln.createPermuteVec(6, d_perm).createHostVec();
    // printf("print to VTK\n");
    if (level < 3) printToVTK<Assembler,HostVec<T>>(kmg->grids[0].assembler, h_soln, "out/wing_mg_lin.vtk");
    // printf("\tdone with print to VTK\n");
    T lin_max_disp = get_max_disp(kmg->grids[0].d_soln);
    // printf("done with linear solve section\n");

    // ------------------------------------------------------------
    // 2) solve nonlinear Newton-Raphson load-step scheme

    // build the inexact newton + outer continuation solver
    using Mat = BsrMat<DeviceVec<T>>;
    using Vec = DeviceVec<T>;
    constexpr bool DO_LINE_SEARCH = true;
    // constexpr bool DO_LINE_SEARCH = false;
    using INK = InexactNewtonSolver<T, Mat, Vec, Assembler, KMG, DO_LINE_SEARCH>;
    using NL = NonlinearContinuationSolver<T, Vec, Assembler, INK>;

    bool use_predictor = true; // sometimes works on wing and sometimes not
    // bool use_predictor = false; // need to do something else to the predictor like smooth it for NL MG case.

    // 100x less load than other case, so need lower atol (it's higher SR)
    T initLinSolveRtol = 5e-2;
    T linSolveAtol = 1e-4;
    T minLinSolveTol = 1e-3;
    T maxLinSolveTol = 0.3;
    // printf("building INK and NL solver\n");
    INK *inner_solver = new INK(cublasHandle, fine_assembler, fine_kmat, fine_loads, kmg, initLinSolveRtol, linSolveAtol, minLinSolveTol, maxLinSolveTol);
    NL *nl_solver = new NL(cublasHandle, fine_assembler, inner_solver, use_predictor);
    // printf("\tdone building INK and NL solver\n");

    // now try calling it
    T lambda0 = 0.2;
    T inner_atol = 1e-4;
    T lambdaf = 1.0; 
    // T inner_crtol = 1e-3, inner_frtol = 1e-6;
    // T inner_crtol = 1e-3, inner_frtol = 1e-4;
    T inner_crtol = 1e-4, inner_frtol = 1e-5; // prevents divergence better. with predictor..
    // printf("begin NL solve\n");
    bool nl_fail = nl_solver->solve(fine_vars, lambda0, inner_atol, lambdaf, inner_crtol, inner_frtol);
    T nl_max_disp = get_max_disp(fine_vars);
    // printf("done with continuation solve - DEBUG PRINT\n");

    // print some of the data of host residual
    auto h_vars = fine_vars.createHostVec();
    if (level > 3) printToVTK<Assembler,HostVec<T>>(fine_assembler, h_vars, "out/wing_mg_nl.vtk");

    // important to know reduction for how NL regime we are
    T ratio = nl_max_disp / lin_max_disp;
    printf("lin max disp %.8e, nl max disp %.8e, ratio = %.8e\n", lin_max_disp, nl_max_disp, ratio);

    // ==================================================


    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = end1 - start1;
    int ndof = fine_assembler.get_num_vars();
    double total = startup_time.count() + solve_time.count();
    printf("nonlinear Newton-Raphson KMG solve of wing geom, ndof %d : startup time %.2e, solve time %.2e, total %.2e\n", ndof, startup_time.count(), solve_time.count(), total);

    // free and cleanup
    // --------------------

    // nl_solver.free();
    kmg->free();
    fine_assembler.free();
}





template <typename T, class Assembler>
void solve_nonlinear_direct(MPI_Comm &comm, int level, double total_force) {
  
    using Basis = typename Assembler::Basis;
    using Physics = typename Assembler::Phys;
    using Data = typename Physics::Data;

  auto start0 = std::chrono::high_resolution_clock::now();

  TACSMeshLoader mesh_loader{comm};
  std::string fname = "../../multigrid/3_aob_wing/meshes/aob_wing_L" + std::to_string(level) + ".bdf";
  mesh_loader.scanBDFFile(fname.c_str());

    // now set component data using new helper method
    HostVec<Data> comp_data(mesh_loader.getNumComponents());
    std::string design_filename = "design/AOB-design.txt";
    build_AOB_component_data<T, Data>(mesh_loader, comp_data, design_filename);
    
    printf("making assembler+GMG for mesh '%s'\n", fname.c_str());
    // create the TACS Assembler from the mesh loader
    auto assembler = Assembler::createFromBDFComponent(mesh_loader, comp_data);


  // TODO : set this in from optimized design from AOB case

  // BSR factorization
  auto& bsr_data = assembler.getBsrData();
  double fillin = 10.0;  // 10.0
  bool print = true;
  bsr_data.AMD_reordering();
  bsr_data.compute_full_LU_pattern(fillin, print);
  assembler.moveBsrDataToDevice();

  cublasHandle_t cublasHandle = NULL;
  CHECK_CUBLAS(cublasCreate(&cublasHandle));
  cusparseHandle_t cusparseHandle = NULL;
  CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

  // get the loads
  int nvars = assembler.get_num_vars();
  int nnodes = assembler.get_num_nodes();
  HostVec<T> h_loads(nvars);
  int level_exp = 1 << level;
  double load_mag = total_force / nnodes; // estimate for nodal load mag
//   double SR1 = (300.0 / SR);
//   double SR3 = SR1 * SR1 * SR1;
//   load_mag *= SR3;
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
  auto rhs = assembler.createVarsVec();

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

    // compare to pure linear solve (to see how nonlinear)
    // ==================================

    assembler.add_jacobian_fast(kmat);
    assembler.apply_bcs(kmat);
    CUSPARSE::direct_LU_solve(kmat, loads, soln);
    T lin_max_disp = get_max_disp(soln);
    auto h_soln = soln.createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_soln, "out/wing_direct_lin.vtk");


    // new nonlinear solver
    // ======================

    // build the inexact newton + outer continuation solver
    using Mat = BsrMat<DeviceVec<T>>;
    using Vec = DeviceVec<T>;
    using LinearSolver = CusparseMGDirectLU<T, Assembler>;
    using INK = InexactNewtonSolver<T, Mat, Vec, Assembler, LinearSolver>;
    using NL = NonlinearContinuationSolver<T, Vec, Assembler, INK>;

    bool use_predictor = true;
    // bool use_predictor = false;

    LinearSolver *solver = new LinearSolver(cublasHandle, cusparseHandle, assembler, kmat);
    INK *inner_solver = new INK(cublasHandle, assembler, kmat, loads, solver);
    NL *nl_solver = new NL(cublasHandle, assembler, inner_solver, use_predictor);

    // now try calling it
    T lambda0 = 0.2;
    // T lambda0 = 0.1;
    // T lambda0 = 0.05;
    // T inner_atol = 1e-5;
    T inner_atol = 1e-8;
    nl_solver->solve(vars, lambda0, inner_atol);
    T nl_max_disp = get_max_disp(vars);

    // print some of the data of host residual
    auto h_vars = vars.createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_vars, "out/wing_direct_nl.vtk");

    // important to know reduction for how NL regime we are
    T ratio = nl_max_disp / lin_max_disp;
    printf("lin max disp %.8e, nl max disp %.8e, ratio = %.8e\n", lin_max_disp, nl_max_disp, ratio);

    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = end1 - start1;
    int ndof = assembler.get_num_vars();
    double total = startup_time.count() + solve_time.count();
    printf("nonlinear Newton-Raphson Direct-LU solve of wing geom, ndof %d : startup time %.2e, solve time %.2e, total %.2e\n", ndof, startup_time.count(), solve_time.count(), total);

    // free and cleanup
    // --------------------
    
    // nl_solver.free();
    assembler.free();
}

template <typename T, class Assembler>
void gatekeeper_method(bool is_multigrid, MPI_Comm &comm, int level, int nsmooth, 
    int ninnercyc, std::string cycle_type, T omega, int ORDER, T omega_min, T omega_max, int n_krylov, double total_force) {
    if (is_multigrid) {
        solve_nonlinear_multigrid<T, Assembler>(comm, level, nsmooth, ninnercyc, cycle_type, omega, ORDER, omega_min, omega_max, n_krylov, total_force);
    } else {
        solve_nonlinear_direct<T, Assembler>(comm, level, total_force);
    }
}

int main(int argc, char **argv) {

    // Intialize MPI and declare communicator
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    // DEFAULTS
    // int level = 1; // for debug
    int level = 2;
    // int level = 3; // level mesh to solve.. level 4 also a good starting setting (big case)
    bool is_multigrid = true;
    // bool is_debug = false;
    int ORDER = 8;
    double omega = 0.9; // after spectral radius norm (which appears to work as omega > 1 diverges, omega < 1 conv)
    // line search breaking down a lot..
    double omegaLS_min = 1.0; // default min line search omega
    double omegaLS_max = 1.0;

    // oh looks like it is just straight up buckling at these load mags.. oof.. not even direct converges at force = 1e6
    // buckling happens around --force 3e5 I think rn (even for direct solve)
    // some mistake in bending stiffnesses then..
    // force > 2.5e5 for MITC4 buckles, force > 9e5 for CFI4 buckles
    // double force = 9e5; // buckles for CFI4
    double force = 7e5; // just below buckling

    int n_krylov = 500; // default n_krylov (may need to increase for L4 mesh)
    int nsmooth = 1; // may need more here (esp for MITC elements, but CFI can use less)
    int ninnercyc = 1; // inner V-cycles to precond K-cycle
    std::string cycle_type = "K"; // "V", "F", "W", "K"
    // can need ninnercyc = 2 for L4 mesh to converge better..
    
    // while MITC4 needed for other unstiff panel wing case (as line seraches would break down with CFI4 at higher mesh levels)
    // only CFI4 solves this more slender (but stiffened panel design), low SR in span direction, but chord-tip high SR (unstiff still may affect performance)
    // std::string elem_type = "MITC4"; // 'MITC4', 'CFI4', 'CFI9'
    std::string elem_type = "CFI4"; // 'MITC4', 'CFI4', 'CFI9'

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        char* arg = argv[i];
        to_lowercase(arg);

        if (strcmp(arg, "direct") == 0) {
            is_multigrid = false;
        } else if (strcmp(arg, "mg") == 0) {
            is_multigrid = true;
        } else if (strcmp(arg, "--omegamin") == 0) {
	    omegaLS_min = std::atof(argv[++i]);
	} else if (strcmp(arg, "--omegamax") == 0) {
	    omegaLS_max = std::atof(argv[++i]);
	} else if (strcmp(arg, "--nkrylov") == 0) {
	    n_krylov = std::atoi(argv[++i]);
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
        } else if (strcmp(arg, "--force") == 0) {
            if (i + 1 < argc) {
                force = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --load\n";
                return 1;
            }
        } else if (strcmp(arg, "--omega") == 0) {
            if (i + 1 < argc) {
                omega = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --omega\n";
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
            std::cerr << "Usage: " << argv[0] << " [direct/mg] [--level int] [--cycle char] [--nsmooth int] [--ninnercyc int]" << std::endl;
            return 1;
        }
    }

    // type specifications here
    using T = double;   
    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    // unlike other cases, do want it now for panel length.. 
    // and so buckling in-plane loads in right direc
    constexpr bool has_ref_axis = true; 
    constexpr bool is_nonlinear = true;
    using Data = StiffenedIsotropicShellData<T, has_ref_axis>;
    using Physics = StiffenedIsotropicShell<T, Data, is_nonlinear>;

    printf("AOB mesh with nonlinear %s elements, level %d and with optimized design\n------------\n", elem_type.c_str(), level);
    if (elem_type == "MITC4") {
        using Basis = LagrangeQuadBasis<T, Quad, 1>;
        using Assembler = MITCShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
        gatekeeper_method<T, Assembler>(is_multigrid, comm, level, nsmooth, ninnercyc, cycle_type, omega, ORDER, omegaLS_min, omegaLS_max, n_krylov, force);
    } else if (elem_type == "CFI4") {
        using Basis = ChebyshevQuadBasis<T, Quad, 1>;
        using Assembler = FullyIntegratedShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
        gatekeeper_method<T, Assembler>(is_multigrid, comm, level, nsmooth, ninnercyc, cycle_type, omega, ORDER, omegaLS_min, omegaLS_max, n_krylov, force);
    } else if (elem_type == "CFI9") {
        using Basis = ChebyshevQuadBasis<T, Quad, 2>;
        using Assembler = FullyIntegratedShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
        gatekeeper_method<T, Assembler>(is_multigrid, comm, level, nsmooth, ninnercyc, cycle_type, omega, ORDER, omegaLS_min, omegaLS_max, n_krylov, force);
    } else {
        printf("ERROR : didn't run anything, elem type not in available types (see main function)\n");
    }

    MPI_Finalize();
    return 0;
};
