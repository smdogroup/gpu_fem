// general gpu_fem imports
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"

// shell imports
#include "assembler.h"
#include "element/shell/director/linear_rotation.h"
#include "element/shell/physics/isotropic_shell.h"


// new nonlinear solvers
#include "solvers/nonlinear_static/inexact_newton.h"
#include "solvers/nonlinear_static/continuation.h"

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
#include "multigrid/smoothers/asw_struct.h"
#include "multigrid/prolongation/structured.h"
#include "multigrid/solvers/gmg.h"
#include <string>
#include <chrono>

#include <type_traits>

// new multigrid imports for K-cycles, etc.
#include "multigrid/solvers/solve_utils.h"
#include "multigrid/solvers/direct/cusp_directLU.h"
#include "multigrid/solvers/krylov/bsr_pcg.h"
#include "multigrid/solvers/multilevel/kcycle.h"
#include "multigrid/solvers/multilevel/twolevel.h"

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

template <typename T, class Smoother, class Assembler>
void multigrid_solve(std::string smoother_type, int nxe, double SR, int nsmooth, int ninnercyc, T omega, int ORDER) {
    // geometric multigrid method here..
    // need to make a number of grids..

    using Basis = typename Assembler::Basis;
    using Physics = typename Assembler::Phys;
    const SCALER scaler  = LINE_SEARCH;
    using Prolongation = StructuredProlongation<Assembler, CYLINDER>;
    using GRID = SingleGrid<Assembler, Prolongation, Smoother, scaler>;
    using CoarseSolver = CusparseMGDirectLU<T, Assembler>;
    // using MG = GeometricMultigridSolver<GRID, CoarseSolver>;
    using ASW = StructuredAdditiveSchwarzSmoother<T, Assembler, S_CYLINDER>;
    
    // for K-cycles
    using KrylovSolve = PCGSolver<T, GRID>;
    using TwoLevelSolve = MultigridTwoLevelSolver<GRID>;
    using KMG = MultilevelKcycleSolver<GRID, CoarseSolver, TwoLevelSolve, KrylovSolve>;

    auto start0 = std::chrono::high_resolution_clock::now();

    KMG *kmg = new KMG();

    // T omegaLS_min = 0.25, omegaLS_max = 2.0;
    T omegaLS_min = 1e-2, omegaLS_max = 4.0;

    cublasHandle_t cublasHandle = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));
    cusparseHandle_t cusparseHandle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

    int pre_nxe_min = nxe > 32 ? 32 : 4;
    int nxe_min = pre_nxe_min;
    for (int c_nxe = nxe; c_nxe >= pre_nxe_min; c_nxe /= 2) {
        nxe_min = c_nxe;
    }

    // make each grid
    for (int c_nxe = nxe; c_nxe >= nxe_min; c_nxe /= 2) {
        // make the assembler
        int c_nhe = c_nxe;
        double L = 1.0, R = 0.5, thick = L / SR;
        double E = 70e9, nu = 0.3;
        // double rho = 2500, ys = 350e6;
        bool imperfection = false; // option for geom imperfection
        int imp_x = 1, imp_hoop = 1; // no imperfection this input doesn't matter rn..
        auto assembler = createCylinderAssembler<Assembler>(c_nxe, c_nhe, L, R, E, nu, thick, imperfection, imp_x, imp_hoop);
        constexpr bool compressive = false;
        const int load_case = 3; // petal and chirp load
        T pressure = 8.0e6;
        double uniform_force = pressure * 1.0 * 1.0;
        double nodal_loads = uniform_force; // / (nxe - 1) / (nxe - 1);
        nodal_loads *= (100.0 / SR) * (100.0 / SR) * (100.0 / SR);
        double Q = 1.0; // load magnitude
        T *my_loads = getCylinderLoads<T,  Basis,Physics, load_case>(nxe, nxe, L, R, nodal_loads);
        printf("making grid with nxe %d\n", c_nxe);

        auto &bsr_data = assembler.getBsrData();
        int num_colors, *_color_rowp;

        // make the grid
        bool full_LU = c_nxe == nxe_min;
        if (full_LU) {
            bsr_data.AMD_reordering();
            bsr_data.compute_full_LU_pattern(10.0, false);
        } else {
            if constexpr (std::is_same_v<Smoother,MulticolorGSSmoother_V1<Assembler>>) {
                bsr_data.multicolor_reordering(num_colors, _color_rowp);
            }
            bsr_data.compute_nofill_pattern();
        }
        // auto grid = *GRID::buildFromAssembler(assembler, my_loads, full_LU, reorder);
        auto h_color_rowp = HostVec<int>(num_colors + 1, _color_rowp);

        assembler.moveBsrDataToDevice();
        auto loads = assembler.createVarsVec(my_loads);
        assembler.apply_bcs(loads);
        auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
        auto res = assembler.createVarsVec();
        int N = res.getSize();

        // assemble the kmat
        auto start0 = std::chrono::high_resolution_clock::now();
        assembler.add_jacobian(res, kmat);
        // assembler.apply_bcs(res);
        assembler.apply_bcs(kmat);
        CHECK_CUDA(cudaDeviceSynchronize());
        auto end0 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> assembly_time = end0 - start0;
        printf("\tassemble kmat in %.2e sec\n", assembly_time.count());

        // build smoother and prolongations..
        Smoother *smoother = nullptr;
        if constexpr (std::is_same_v<Smoother, ChebyshevPolynomialSmoother<Assembler, false>>) {
            // both chebyshev or jacobi here (jacobi is special case of order 1)
            if (smoother_type == "chebyshev") {
                // int ORDER = 8;
                // int ORDER = 4;
                smoother = new Smoother(cublasHandle, cusparseHandle, assembler, kmat, omega, ORDER, nsmooth);
            } else {
                int _ORDER = 1;
                smoother = new Smoother(cublasHandle, cusparseHandle, assembler, kmat, omega, _ORDER, nsmooth);
            }
        } else if constexpr (std::is_same_v<Smoother,MulticolorGSSmoother_V1<Assembler>>) {
            bool symmetric = true;
            smoother = new Smoother(cublasHandle, cusparseHandle, assembler, kmat, h_color_rowp, omega, symmetric, nsmooth);
        } else if constexpr (std::is_same_v<Smoother, ASW>) {
            int size = 2;
            smoother = new Smoother(cublasHandle, cusparseHandle, assembler, kmat, c_nxe + 1, c_nxe, 
            omega, nsmooth, size);
            // smoother->factor();
        } else {
            static_assert(sizeof(Smoother) == 0, "Unsupported smoother type");
        }
        // smoother->factor();


        auto prolongation = new Prolongation(assembler);
        auto grid = GRID(assembler, prolongation, smoother, kmat, loads, cublasHandle, cusparseHandle, omegaLS_min, omegaLS_max);
        kmg->grids.push_back(grid);
    }

    // register the coarse assemblers to the prolongations..
    kmg->template init_prolongations<Basis>();

    auto end0 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> startup_time = end0 - start0;

    int pre_smooth = nsmooth, post_smooth = nsmooth; // need a little extra smoothing on cylinder (compare to plate).. (cause of curvature I think..)
    bool print = true;
    // bool print = false;
    T atol = 1e-6, rtol = 1e-6;
    bool double_smooth = true; // twice as many smoothing steps at lower levels (similar cost, better conv?)

    int n_cycles = 500; // max # cycles
    int print_freq = 3;
    int n_krylov = 500;
    kmg->init_outer_solver(cublasHandle, cusparseHandle, nsmooth, ninnercyc, n_krylov, omega, atol, rtol, print_freq, print, double_smooth);    

    // create solution and right hand side vecs
    int N_fine = kmg->grids[0].N;
    auto rhs = DeviceVec<T>(N_fine);
    auto soln = DeviceVec<T>(N_fine);
    kmg->grids[0].d_defect.copyValuesTo(rhs);


    // get the loads out again cause they were permuted by grid
    // create the loads and kmat
    auto assembler = kmg->grids[0].assembler;
    // assembler.apply_bcs(loads);
    // auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto kmat = kmg->grids[0].Kmat;
    auto res = assembler.createVarsVec();
    auto lin_soln = assembler.createVarsVec();
    auto vars = assembler.createVarsVec();
    auto loads2 = assembler.createVarsVec();
    int N = res.getSize();
    bool perm_out = true;
    kmg->grids[0].getDefect(loads2, perm_out);

    // get initial residual
    KrylovSolve* outer_solver = static_cast<KrylovSolve*>(kmg->outer_solver); 
    T init_resid = outer_solver->getResidualNorm(rhs, soln);

    CHECK_CUDA(cudaDeviceSynchronize());
    auto startL = std::chrono::high_resolution_clock::now();

    if constexpr (std::is_same_v<Smoother, ASW>) {
        int nlevels = kmg->grids.size();
        for (int i = 0; i < nlevels; i++) {
            kmg->grids[i].smoother->factor();
        }
    } 

    kmg->coarse_solver->factor(); // factor
    bool fail = kmg->solve(rhs, soln);


    CHECK_CUDA(cudaDeviceSynchronize());
    auto endL = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = endL - startL;
    printf("\tASW-linear solve time %.2e on %d nxe\n", solve_time.count(), nxe);

    // // print to VTK (permuting from solve to vis order)
    int *d_perm = outer_solver->grid->d_perm;
    auto h_soln = soln.createPermuteVec(6, d_perm).createHostVec();
    printToVTK<Assembler,HostVec<T>>(outer_solver->grid->assembler, h_soln, "out/cyl_kry_lin.vtk");
    // T lin_max_disp = get_max_disp(soln);

    if (fail) {
        printf("\tPCG linear solve failed, so not proceeding to nonlinear solves\n");
        return;
    }

    // -----------------------------------------------------------
    // 2) actually try Newton-mg solve here (this is just V1, later versions may use FMG cycle so less extra work needs to be done on fine grids)
    //     i.e. you can do most of hte nonlinear solves to get in basin of attraction on coarser grids first.. (then nonlinear fine grid at end only, or some FMG cycle)

    // new nonlinear solver
    // ======================

    // build the inexact newton + outer continuation solver
    using Mat = BsrMat<DeviceVec<T>>;
    using Vec = DeviceVec<T>;
    using INK = InexactNewtonSolver<T, Mat, Vec, Assembler, KMG>;
    using NL = NonlinearContinuationSolver<T, Vec, Assembler, INK>;

    T initLinSolveRtol = 1e-3;
    T initLinSolveAtol = 1e-4;
    T minRtol = 1e-4, maxRtol = 1e-2; // don't want min rtol too low, cause GMRES will run out of steps (with GSMC precond)

    outer_solver->set_print(false);

    INK *inner_solver = new INK(cublasHandle, assembler, kmat, loads2, kmg, initLinSolveRtol, initLinSolveAtol, minRtol, maxRtol);
    // bool use_predictor = true, debug = false;
    bool use_predictor = true, debug = false;
    NL *nl_solver = new NL(cublasHandle, assembler, inner_solver, use_predictor, debug);

    // now try calling it
    T lambda0 = 0.2;
    // T lambda0 = 0.05;
    // T inner_atol = 1e-2;
    // T inner_atol = 1e-4;
    // T inner_atol = 1e-4;
    T inner_atol = 1e-6;
    T lambdaf = 1.0;

    CHECK_CUDA(cudaDeviceSynchronize());
    auto start1 = std::chrono::high_resolution_clock::now();
    nl_solver->solve(vars, lambda0, inner_atol, lambdaf);

    CHECK_CUDA(cudaDeviceSynchronize());
    auto end1 = std::chrono::high_resolution_clock::now();

    // T nl_max_disp = get_max_disp(vars);
    T lin_max_disp = 1.0;
    T nl_max_disp = 0.0;

    // print some of the data of host residual
    auto h_vars = vars.createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_vars, "out/cylinder_kry_nl.vtk");

    // important to know reduction for how NL regime we are
    T ratio = nl_max_disp / lin_max_disp;
    printf("lin max disp %.8e, nl max disp %.8e, ratio = %.8e\n", lin_max_disp, nl_max_disp, ratio);

    // auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> nl_solve_time = end1 - start1;
    int ndof = assembler.get_num_vars();
    double total = startup_time.count() + nl_solve_time.count();
    printf("nonlinear Newton-Raphson ASW-PCG solve of plate geom, ndof %d : startup time %.2e, solve time %.2e, total %.2e\n", ndof, startup_time.count(), nl_solve_time.count(), total);

    int num_newton = inner_solver->get_num_lin_solves();
    printf("\tlin solve time %.3e, nl solve time %.3e, #lin-solves %d\n", solve_time.count(), nl_solve_time.count(), num_newton);

    if (fail) {
        printf("\tnonlinear solver failed\n");
        return;
    }
}

template <typename T, class Assembler>
void solve_direct(int nxe, double SR) {

    /* direct NL solve used to check that how NL the problem is and how */

    using Basis = typename Assembler::Basis;
    using Physics = typename Assembler::Phys;
    const SCALER scaler  = LINE_SEARCH;
    using Smoother = MulticolorGSSmoother_V1<Assembler>;
    using Prolongation = StructuredProlongation<Assembler, CYLINDER>;
    using GRID = SingleGrid<Assembler, Prolongation, Smoother, scaler>;

    // for K-cycles
    // linear solver
    using Precond = CusparseMGDirectLU<T, Assembler>;
    using PCG = PCGSolver<T, GRID>;

    // create cublas and cusparse handles (single one each)
    // -----------------------------------------------------
    cublasHandle_t cublasHandle = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));
    cusparseHandle_t cusparseHandle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));


    CHECK_CUDA(cudaDeviceSynchronize());
    auto start0 = std::chrono::high_resolution_clock::now();

    double L = 1.0, R = 0.5, thick = L / SR;
    double E = 70e9, nu = 0.3;
    // double rho = 2500, ys = 350e6;
    bool imperfection = false; // option for geom imperfection
    int imp_x = 1, imp_hoop = 1; // no imperfection this input doesn't matter rn..
    auto assembler = createCylinderAssembler<Assembler>(nxe, nxe, L, R, E, nu, thick, imperfection, imp_x, imp_hoop);
    constexpr bool compressive = false;
    const int load_case = 3; // petal and chirp load
    T pressure = 5.0e7;
    double uniform_force = pressure * 1.0 * 1.0;
    double nodal_loads = uniform_force; // / (nxe - 1) / (nxe - 1);
    nodal_loads *= (100.0 / SR) * (100.0 / SR) * (100.0 / SR);
    double Q = 1.0; // load magnitude
    T *my_loads = getCylinderLoads<T,  Basis,Physics, load_case>(nxe, nxe, L, R, nodal_loads);
    printf("making grid with nxe %d\n", nxe);

    // perform multicolor reordering
    auto &bsr_data = assembler.getBsrData();
    bsr_data.AMD_reordering();
    bsr_data.compute_full_LU_pattern(10.0, false);

    // T *_color_rowp = new T[2];
    auto h_color_rowp = HostVec<int>(2);
    assembler.moveBsrDataToDevice();

    // create the loads and kmat
    auto loads = assembler.createVarsVec(my_loads);
    assembler.apply_bcs(loads);
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto res = assembler.createVarsVec();
    auto lin_soln = assembler.createVarsVec();
    auto vars = assembler.createVarsVec();
    auto loads2 = assembler.createVarsVec();
    int N = res.getSize();

    // assemble the kmat
    auto startkmat = std::chrono::high_resolution_clock::now();
    assembler.add_jacobian_fast(kmat);
    assembler.apply_bcs(kmat);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto endkmat = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> assembly_time = endkmat - startkmat;
    printf("\tassemble kmat in %.2e sec\n", assembly_time.count());

    // build smoother and prolongations..
    // nsmooth steps per precond set in the solver
    T omegaMC = 1.0;
    int nsmooth = 1; // not used
    auto smoother = new Smoother(cublasHandle, cusparseHandle, assembler, kmat, h_color_rowp, omegaMC, false, nsmooth);
    auto prolongation = new Prolongation(assembler);
    auto grid = new GRID(assembler, prolongation, smoother, kmat, loads, cublasHandle, cusparseHandle);

    // the ILU preconditioner
    auto pc = new Precond(cublasHandle, cusparseHandle, assembler, kmat); // turns out the smoother does work somewhat

    // create the preconditioner and GMRES solver now
    auto options = SolverOptions();
    options.ncycles = 800; // number of max PCG cycles
    options.print_freq = 10;

    // PCG solver
    auto linear_solver = new PCG(cublasHandle, cusparseHandle, grid, pc, options);

    // only use GMRES if SR > 100
    // const int N_SUBSPACE = 100;
    // using GMRES = GMRESSolver<T, GRID, N_SUBSPACE>;
    // int MAX_ITER = N_SUBSPACE;
    // auto linear_solver = new GMRES(cublasHandle, cusparseHandle, grid, pc, options, MAX_ITER);


    // out settings
    linear_solver->set_rel_tol(1e-6);
    linear_solver->set_abs_tol(1e-6);
    linear_solver->set_print(true);

    auto endstartup = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> startup_time = endstartup - start0;

    // run the linear solver
    CHECK_CUDA(cudaDeviceSynchronize());
    auto start_solve = std::chrono::high_resolution_clock::now();
    // run factor again so that we give fair comparison
    // pc->factor_matrix();
    pc->factor();
    

    // get initial residual
    T init_resid = linear_solver->getResidualNorm(grid->d_defect, lin_soln);

    // linear solve
    bool fail = linear_solver->solve(grid->d_defect, lin_soln, true);
    
    // final residual
    T final_resid = linear_solver->getResidualNorm(grid->d_defect, lin_soln);

    CHECK_CUDA(cudaDeviceSynchronize());
    auto end_solve = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = end_solve - start_solve;

    // compute log residual reduction per unit time
    // it's converging about 1e14 resid drop, only need like 1e7 so half
    T log_resid_drop = (log(init_resid) - log(final_resid)) / log(10.0);
    // T log_resid_cap = log(1e6) / log(10.0); // cap out past 1e6 because don't need deeper than this really for Newton-Krylov..
    T log_red_rate =  log_resid_drop / solve_time.count();
    printf("\nDirectLU-PCG on cylinder case with %d nxe and %.4e SR\n", nxe, SR);
    printf("\tinit resid %.4e => final resid %.4e in %.2e sec, log10(reduction)/sec = %.6e\n", init_resid, final_resid, solve_time.count(), log_red_rate);

    int nx = nxe + 1;
    int ndof = nx * nx * 6;
    double total = startup_time.count() + solve_time.count();
    size_t bytes_per_double = sizeof(double);
    double mem_mb = static_cast<double>(bytes_per_double) * static_cast<double>(bsr_data.nnzb) * 36.0 / 1024.0 / 1024.0;
    printf("fullLU-memory in MB %.4e with NDOF %d\n", mem_mb, ndof);

    // // print to VTK (permuting from solve to vis order)
    int *d_perm = linear_solver->grid->d_perm;
    auto h_soln = lin_soln.createPermuteVec(6, d_perm).createHostVec();
    printToVTK<Assembler,HostVec<T>>(linear_solver->grid->assembler, h_soln, "out/cylinder_lin.vtk");
    // T lin_max_disp = get_max_disp(lin_soln);

    if (fail) {
        printf("\tPCG linear solver failed\n");
        return;
    }
}

template <typename T, class Smoother, class Assembler>
void gatekeeper_method(std::string smoother_type, int nxe, double SR, int nsmooth, int ninnercyc, T omega, int ORDER) {
    if (smoother_type != "direct") {
        multigrid_solve<T, Smoother, Assembler>(smoother_type, nxe, SR, nsmooth, ninnercyc, omega, ORDER);
    } else {
        solve_direct<T, Assembler>(nxe, SR);
    }
}

int main(int argc, char **argv) {
    // input ----------
    int nxe = 128; // default value
    double SR = 10.0; // default
    int n_vcycles = 50;
    double omega = 0.15;
    int ORDER = 8; // for chebyshev

    int nsmooth = 2; // typically faster right now
    int ninnercyc = 1; // inner V-cycles to precond K-cycle

    // chebyshev, jacobi, gsmc, direct 
    std::string smoother_type = "chebyshev";

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        char* arg = argv[i];
        to_lowercase(arg);

        if (strcmp(arg, "--nxe") == 0) {
            if (i + 1 < argc) {
                nxe = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --nxe\n";
                return 1;
            }
        } else if (strcmp(arg, "--omega") == 0) {
            if (i + 1 < argc) {
                omega = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --omega\n";
                return 1;
            }
        } else if (strcmp(arg, "--sr") == 0) {
            if (i + 1 < argc) {
                SR = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --SR\n";
                return 1;
            }
        } else if (strcmp(arg, "--smoother") == 0) {
            if (i + 1 < argc) {
                smoother_type = argv[++i];
            } else {
                std::cerr << "Missing value for --smoother\n";
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
    constexpr bool is_nonlinear = true;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;

    // MITC4 shells
    using Basis = LagrangeQuadBasis<T, Quad, 1>;
    using Assembler = MITCShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;

    printf("cylinder mesh with MITC4 elements, nxe %d and SR %.2e\n------------\n", nxe, SR);
    if (smoother_type == "chebyshev" || smoother_type == "jacobi") {
        using Smoother = ChebyshevPolynomialSmoother<Assembler, false>;
        gatekeeper_method<T, Smoother, Assembler>(smoother_type, nxe, SR, nsmooth, ninnercyc, omega, ORDER);
    } else if (smoother_type == "gsmc" || smoother_type == "direct") {
        using Smoother = MulticolorGSSmoother_V1<Assembler>; // still calls direct later if direct
        gatekeeper_method<T, Smoother, Assembler>(smoother_type, nxe, SR, nsmooth, ninnercyc, omega, ORDER);
    } else if (smoother_type == "asw") {
        using Smoother = StructuredAdditiveSchwarzSmoother<T, Assembler, S_CYLINDER>; // still calls direct later if direct
        gatekeeper_method<T, Smoother, Assembler>(smoother_type, nxe, SR, nsmooth, ninnercyc, omega, ORDER);
    }

    return 0;

    
}