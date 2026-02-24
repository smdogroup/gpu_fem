// general gpu_fem imports
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"

// new nonlinear solvers
#include "solvers/nonlinear_static/inexact_newton.h"
#include "solvers/nonlinear_static/continuation.h"

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

// hellinger reissner element
#include "element/shell/hr_shell.h"

// local multigrid imports
#include "multigrid/grid.h"
#include "multigrid/utils/fea.h"
#include "multigrid/smoothers/cheb4_poly.h"
#include "multigrid/smoothers/mc_smooth1.h"
#include "multigrid/prolongation/structured.h"
#include "multigrid/solvers/gmg.h"
#include <string>
#include <chrono>

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
    ./1_static_gmg.out direct --nxe 2048 --SR 100.0    to run direct plate solve on 2048 x 2048 elem grid with slenderness ratio 100
    ./1_static_gmg.out mg --nxe 2048 --SR 100.0    to run geometric multigrid plate solve on 2048 x 2048 elem grid with slenderness ratio 100
*/

// NOTE : weird BCs might be slowing down the multigrid + nonlinear solver conv here (we get weird nonlinear direct convergence initially as well, something we don't see for wings)

void to_lowercase(char *str) {
    for (; *str; ++str) {
        *str = std::tolower(*str);
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
void multigrid_solve(int nxe, double SR, int nsmooth, int ninnercyc, std::string cycle_type, T omega, T pressure = 5.0e7) {
    // geometric multigrid method here..
    // need to make a number of grids..
    using Basis = typename Assembler::Basis;
    using Physics = typename Assembler::Phys;
    // using Smoother = MulticolorGSSmoother_V1<Assembler>;
    using Smoother = ChebyshevPolynomialSmoother<Assembler>;
    using Prolongation = StructuredProlongation<Assembler, CYLINDER>;
    
    // sometimes line search helps, sometimes not
    using GRID = SingleGrid<Assembler, Prolongation, Smoother, LINE_SEARCH>;
    // using GRID = SingleGrid<Assembler, Prolongation, Smoother, NONE>;
    
    using CoarseSolver = CusparseMGDirectLU<T, Assembler>;
    using MG = GeometricMultigridSolver<GRID, CoarseSolver>;

    // for K-cycles
    // constexpr bool is_nonlinear = true;
    constexpr bool full_approx_scheme = false;
    using KrylovSolve = PCGSolver<T, GRID>;
    using TwoLevelSolve = MultigridTwoLevelSolver<GRID, full_approx_scheme>;
    using KMG = MultilevelKcycleSolver<GRID, CoarseSolver, TwoLevelSolve, KrylovSolve>;

    // create cublas and cusparse handles (single one each)
    // -----------------------------------------------------
    cublasHandle_t cublasHandle = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));
    cusparseHandle_t cusparseHandle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));


    CHECK_CUDA(cudaDeviceSynchronize());
    auto start0 = std::chrono::high_resolution_clock::now();
    
    MG *mg;
    KMG *kmg;

    bool is_kcycle = cycle_type == "K";
    if (is_kcycle) {
        kmg = new KMG();
    } else {
        mg = new MG();
    }

    // some important settings
    // T omegaMC = 1.5; // for GS-SOR
    // T omegaMC = 0.75;
    // T omegaMC = 0.7;

    T omegaLS_min = 1e-2, omegaLS_max = 4.0;

    // T omegaLS_min = 0.1, omegaLS_max = 2.0;
    // T omegaLS_min = 0.25, omegaLS_max = 2.0;
    // T omegaLS_min = 0.5, omegaLS_max = 2.0;

    // get nxe_min for not exactly power of 2 case
    int nxe_start = 32 / Basis::order;
    // int nxe_start = 64 / Basis::order; // higher load frequency here needs a bit finer mesh for coarsest grid
    // int pre_nxe_min = nxe > 16 ? 16 : 4; // 2x slower with this setting (takes more V-cycles)
    int pre_nxe_min = nxe > nxe_start ? nxe_start : 4; // but on higher nxe, this one is more robust somehow
    // int pre_nxe_min = nxe > 64 ? 64 : 4; // solved about 33% faster with this as coarsest grid (for nxe = 256, but prob need faster direct solver on GPU)

    int nxe_min = pre_nxe_min;
    for (int c_nxe = nxe; c_nxe >= pre_nxe_min; c_nxe /= 2) {
        nxe_min = c_nxe;
    }

    if (cycle_type != "K") {
        printf("only does Kcycle mg in this example rn (just haven't adapted generally)\n");
        return;
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
        constexpr int load_case = 3; // petal and chirp load
        // double nodal_loads = uniform_force; // (don't normalize anymore, integrated out) / (nxe - 1) / (nxe - 1);
        // T *my_loads = getCylinderLoads<T,  Basis, Physics, load_case>(c_nxe, c_nhe, L, R, pressure);
        // T *my_loads = getCylinderLoadsRobust<T,  Assembler>(assembler, c_nxe, c_nhe, L, R, pressure);
        T *my_loads = getCylinderLoadsSimple<T,  Assembler>(assembler, c_nxe, c_nhe, L, R, pressure);
        printf("making grid with nxe %d\n", c_nxe);

        auto &bsr_data = assembler.getBsrData();
        int num_colors, *_color_rowp;

        // make the grid
        bool full_LU = c_nxe == nxe_min;
        if (full_LU) {
            bsr_data.AMD_reordering();
            bsr_data.compute_full_LU_pattern(10.0, false);
        } else {
            bsr_data.multicolor_reordering(num_colors, _color_rowp);
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
        // assembler.add_jacobian(res, kmat);
        assembler.add_jacobian_fast(kmat);
        // assembler.apply_bcs(res);
        assembler.apply_bcs(kmat);
        CHECK_CUDA(cudaDeviceSynchronize());
        auto end0 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> assembly_time = end0 - start0;
        printf("\tassemble kmat in %.2e sec\n", assembly_time.count());

        // build smoother and prolongations..
        // auto smoother = new Smoother(cublasHandle, cusparseHandle, assembler, kmat, h_color_rowp, omegaMC);
        int ORDER = 4;
        auto smoother = new Smoother(cublasHandle, cusparseHandle, assembler, kmat, omega, ORDER);
        auto prolongation = new Prolongation(assembler);
        auto grid = GRID(assembler, prolongation, smoother, kmat, loads, cublasHandle, cusparseHandle, omegaLS_min, omegaLS_max);

        smoother->setup_cg_lanczos(grid.d_defect);
        
        if (is_kcycle) {
            kmg->grids.push_back(grid);
        } else {
            mg->grids.push_back(grid);
            if (full_LU) mg->coarse_solver = new CoarseSolver(cublasHandle, cusparseHandle, assembler, kmat);
        }
    }

    // register the coarse assemblers to the prolongations..
    if (is_kcycle) {
        kmg->template init_prolongations<Basis>();
    } else {
        mg->template init_prolongations<Basis>();
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    auto end0 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> startup_time = end0 - start0;

    T init_resid_nrm = is_kcycle ? kmg->grids[0].getResidNorm() : mg->grids[0].getResidNorm();

    CHECK_CUDA(cudaDeviceSynchronize());

    int pre_smooth = nsmooth, post_smooth = nsmooth; // need a little extra smoothing on cylinder (compare to plate).. (cause of curvature I think..)
    // bool print = true;
    bool print = false;
    T atol = 1e-6, rtol = 1e-6;
    bool double_smooth = true; // twice as many smoothing steps at lower levels (similar cost, better conv?)

    // int n_cycles = 500; // max # cycles
    int print_freq = 3;

    if (is_kcycle) {
        // int n_krylov = 500;
        // int n_krylov = 10;
        // int n_krylov = 20;
        // int n_krylov = 40;
        int n_krylov = 200;
        kmg->init_outer_solver(cublasHandle, cusparseHandle, nsmooth, ninnercyc, n_krylov, omega, atol, rtol, print_freq, print, double_smooth);    
    }

    std::vector<GRID>& grids = kmg->grids;

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

    CHECK_CUDA(cudaDeviceSynchronize());
    auto start_lin = std::chrono::high_resolution_clock::now();

    kmg->solve();

    CHECK_CUDA(cudaDeviceSynchronize());
    auto end_lin = std::chrono::high_resolution_clock::now();

    kmg->set_print(false);

    T lin_max_disp = get_max_disp(kmg->grids[0].d_soln);
    printf("lin max disp = %.4e\n", lin_max_disp);

    std::chrono::duration<double> lin_time = end_lin - start_lin;

    int *d_perm = kmg->grids[0].d_perm;
    auto h_soln = kmg->grids[0].d_soln.createPermuteVec(6, d_perm).createHostVec();
    // printToVTK<Assembler,HostVec<T>>(kmg->grids[0].assembler, h_soln, "out/cylinder_mg_lin.vtk");
    


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

    INK *inner_solver = new INK(cublasHandle, fine_assembler, fine_kmat, fine_loads, kmg);
    // bool use_predictor = true, debug = false;
    bool use_predictor = false, debug = false;
    NL *nl_solver = new NL(cublasHandle, fine_assembler, inner_solver, use_predictor, debug);

    // now try calling it
    T lambda0 = 0.2;
    // T lambda0 = 0.05;
    // T inner_atol = 1e-2;
    // T inner_atol = 1e-4;
    // T inner_atol = 1e-4;
    T inner_atol = 1e-6;

    CHECK_CUDA(cudaDeviceSynchronize());
    auto start_nl = std::chrono::high_resolution_clock::now();

    nl_solver->solve(fine_vars, lambda0, inner_atol);

    CHECK_CUDA(cudaDeviceSynchronize());
    auto end_nl = std::chrono::high_resolution_clock::now();

    
    T nl_max_disp = get_max_disp(fine_vars);

    // print some of the data of host residual
    auto h_vars = fine_vars.createHostVec();
    // printToVTK<Assembler,HostVec<T>>(fine_assembler, h_vars, "out/cylinder_mg_nl.vtk");

    // important to know reduction for how NL regime we are
    T ratio = nl_max_disp / lin_max_disp;
    printf("lin max disp %.8e, nl max disp %.8e, ratio = %.8e\n", lin_max_disp, nl_max_disp, ratio);

    std::chrono::duration<double> nl_time = end_nl - start_nl;
    int ndof = fine_assembler.get_num_vars();
    // double total = startup_time.count() + solve_time.count();
    printf("lin solve time %.4e, nl solve time %.4e for ndof %d\n", lin_time.count(), nl_time.count(), ndof);
    // printf("nonlinear Newton-Raphson KMG solve of cylinder geom, ndof %d : startup time %.2e, solve time %.2e, total %.2e\n", ndof, startup_time.count(), solve_time.count(), total);

    // free and cleanup
    // --------------------

    // nl_solver.free();
    kmg->free();
    fine_assembler.free();
}

template <typename T, class Assembler>
void solve_direct(int nxe, double SR, T pressure = 5.0e7) {

    /* direct NL solve used to check that how NL the problem is and how */

    using Basis = typename Assembler::Basis;
    using Physics = typename Assembler::Phys;

    CHECK_CUDA(cudaDeviceSynchronize());
    auto start0 = std::chrono::high_resolution_clock::now();

    double L = 1.0, R = 0.5, thick = L / SR;
    double E = 70e9, nu = 0.3;
    // double rho = 2500, ys = 350e6;
    bool imperfection = false; // option for geom imperfection
    int imp_x = 1, imp_hoop = 1; // no imperfection this input doesn't matter rn..
    auto assembler = createCylinderAssembler<Assembler>(nxe, nxe, L, R, E, nu, thick, imperfection, imp_x, imp_hoop);
    constexpr int load_case = 3; // petal and chirp load
    // double nodal_loads = uniform_force; // (don't normalize anymore, integrated out) / (nxe - 1) / (nxe - 1);
    // T *my_loads = getCylinderLoads<T,  Basis, Physics, load_case>(nxe, nxe, L, R, pressure);
    // T *my_loads = getCylinderLoadsRobust<T,  Assembler>(assembler, nxe, nxe, L, R, pressure);
    T *my_loads = getCylinderLoadsSimple<T,  Assembler>(assembler, nxe, nxe, L, R, pressure);
    printf("making grid with nxe %d\n", nxe);

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


    // double Q = 1.0e5;
    // T *my_loads = getPlatePointLoad<T, Physics>(nxe, nye, Lx, Ly, Q);
    // double in_plane_frac = 0.3;
    // T *my_loads = getPlateNonlinearLoads<T, Physics>(nxe, nye, Lx, Ly, Q, in_plane_frac);
    // T *my_loads = getPlateLoads<T, Physics>(nxe, nye, Lx, Ly, Q);
    auto loads = assembler.createVarsVec(my_loads);
    assembler.apply_bcs(loads);

    // setup kmat and initial vecs
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto soln = assembler.createVarsVec();
    auto res = assembler.createVarsVec();
    auto rhs = assembler.createVarsVec();
    auto vars = assembler.createVarsVec();

    CHECK_CUDA(cudaDeviceSynchronize());
    auto start1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> startup_time = start1 - start0;

    // newton solve
    // int num_load_factors = 50, num_newton = 10;
    // T min_load_factor = 1.0 / (num_load_factors - 1), max_load_factor = 1.0, abs_tol = 1e-8,
    //     rel_tol = 1e-8;
    // auto solve_func = CUSPARSE::direct_LU_solve<T>;
    // std::string outputPrefix = "out/plate_";
    // // bool write_vtk = true;
    // bool write_vtk = false;
    // old slow nonlinear solve (less robust)
    // const bool fast_assembly = true;
    // // const bool fast_assembly = false;
    // newton_solve<T, BsrMat<DeviceVec<T>>, DeviceVec<T>, Assembler, fast_assembly>(
    //     solve_func, kmat, loads, soln, assembler, res, rhs, vars,
    //     num_load_factors, min_load_factor, max_load_factor, num_newton, abs_tol,
    //     rel_tol, outputPrefix, print, write_vtk);

    // compare to pure linear solve (to see how nonlinear)
    // ==================================

    printf("add jacobian\n");
    assembler.add_jacobian_fast(kmat);

    // TEMP DEBUG for HR element before bcs applied (check Kelem)
    // if constexpr (Physics::hellingerReissner) {
    //     T *h_kmat_vals = kmat.getVec().createHostVec().getPtr();
    //     int vpn = Physics::vars_per_node;
    //     printf("HR element with %d vpn\n", vpn);
    //     printf("h_kmat_vals for one block node in [-H, G; G, 0] format: \n");
    //     for (int row = 0; row < vpn; row++) {
    //         T *kmat_row = &h_kmat_vals[vpn * row];
    //         printVec<T>(5, kmat_row);
    //         printf("\t");
    //         printVec<T>(6, &kmat_row[5]); 
    //         if (row == 4) printf("\n===========\n"); // divider between rows 0-4 and 5-10
    //     }
    // }
    // return;

    assembler.apply_bcs(kmat); // comment out for temp debug
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("\tdone with jacobian\n");
    // return;

    printf("try linear solve\n");
    CUSPARSE::direct_LU_solve(kmat, loads, soln);
    printf("\tdone with linear solve\n");

    T lin_max_disp = get_max_disp(soln);
    auto h_soln = soln.createHostVec();
    printf("lin max disp = %.4e\n", lin_max_disp);
    printf("print solution\n");
    printToVTK<Assembler,HostVec<T>>(assembler, h_soln, "out/cylinder_lin.vtk");
    printf("\tprinted solution\n");
    

    // new nonlinear solver
    // ======================

    // build the inexact newton + outer continuation solver
    using Mat = BsrMat<DeviceVec<T>>;
    using Vec = DeviceVec<T>;
    using LinearSolver = CusparseMGDirectLU<T, Assembler>;
    using INK = InexactNewtonSolver<T, Mat, Vec, Assembler, LinearSolver>;
    using NL = NonlinearContinuationSolver<T, Vec, Assembler, INK>;

    LinearSolver *solver = new LinearSolver(cublasHandle, cusparseHandle, assembler, kmat);
    INK *inner_solver = new INK(cublasHandle, assembler, kmat, loads, solver);
    NL *nl_solver = new NL(cublasHandle, assembler, inner_solver);

    // now try calling it
    T lambda0 = 0.2;
    // T lambda0 = 0.05;
    nl_solver->solve(vars, lambda0);
    T nl_max_disp = get_max_disp(vars);

    // print some of the data of host residual
    auto h_vars = vars.createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_vars, "out/cylinder_nl.vtk");

    // important to know reduction for how NL regime we are
    T ratio = nl_max_disp / lin_max_disp;
    printf("lin max disp %.8e, nl max disp %.8e, ratio = %.8e\n", lin_max_disp, nl_max_disp, ratio);

    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = end1 - start1;
    int ndof = assembler.get_num_vars();
    double total = startup_time.count() + solve_time.count();
    printf("nonlinear Newton-Raphson Direct-LU solve of cylinder geom, ndof %d : startup time %.2e, solve time %.2e, total %.2e\n", ndof, startup_time.count(), solve_time.count(), total);

    // free and cleanup
    // --------------------
    
    // nl_solver.free();
    assembler.free();
}

template <typename T, class Assembler>
void gatekeeper_method(bool is_multigrid, int nxe, double SR, int nsmooth, int ninnercyc, std::string cycle_type, T omega, T load_mag = 5.0e7) {
    if (is_multigrid) {
        multigrid_solve<T, Assembler>(nxe, SR, nsmooth, ninnercyc, cycle_type, omega, load_mag);
    } else {
        solve_direct<T, Assembler>(nxe, SR, load_mag);
    }
}

int main(int argc, char **argv) {
    // input ----------
    bool is_multigrid = true;
    int nxe = 128;
    double SR = 100.0; // default, the less slender it is, solves much faster
    int n_vcycles = 50;
    // double pressure = 8.0e6;
    // double pressure = 8e7;
    double pressure = 32e7;
    double omega = 0.8; // works better with omega near 1

    // old GSMC settings
    int nsmooth = 2;
    // int nsmooth = 1; // use nsmooth = 2 for 3rd order elements (needed)
    int ninnercyc = 1;
    std::string cycle_type = "K"; // "V", "F", "W", "K"
    // std::string elem_type = "MITC4"; // 'MITC4', 'CFI4', 'CFI9', 'HR4'
    // std::string elem_type = "CFI4"; // careful CFI4 shear locks some (need better element here)
    std::string elem_type = "CFI9"; 

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        char* arg = argv[i];
        to_lowercase(arg);

        if (strcmp(arg, "direct") == 0) {
            is_multigrid = false;
        } else if (strcmp(arg, "mg") == 0) {
            is_multigrid = true;
        } else if (strcmp(arg, "--nxe") == 0) {
            if (i + 1 < argc) {
                nxe = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --nxe\n";
                return 1;
            }
        }  else if (strcmp(arg, "--sr") == 0) {
            if (i + 1 < argc) {
                SR = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --SR\n";
                return 1;
            }
        }  else if (strcmp(arg, "--omega") == 0) {
            if (i + 1 < argc) {
                omega = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --omega\n";
                return 1;
            }
        } else if (strcmp(arg, "--pressure") == 0) {
            if (i + 1 < argc) {
                pressure = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --load\n";
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
            std::cerr << "Usage: " << argv[0] << " [direct/mg] [--nxe value] [--SR value] [--cycle char] [--nsmooth int] [--ninnercyc int]" << std::endl;
            return 1;
        }
    }

    // type specifications here
    using T = double;   
    using Director = LinearizedRotation<T>;
    constexpr bool has_ref_axis = false;
    constexpr bool is_nonlinear = true; // this is a nonlinear GMG case
    using Data = ShellIsotropicData<T, has_ref_axis>;

    printf("cylinder mesh with geomNL %s elements, nxe %d and SR %.2e\n------------\n", elem_type.c_str(), nxe, SR);
    if (elem_type == "MITC4") {
        using Physics = IsotropicShell<T, Data, is_nonlinear>;
        using Quad = QuadLinearQuadrature<T>;
        using Basis = LagrangeQuadBasis<T, Quad, 1>;
        using Assembler = MITCShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
        gatekeeper_method<T, Assembler>(is_multigrid, nxe, SR, nsmooth, ninnercyc, cycle_type, omega, pressure);
    // MITC higher order don't really work? but LFI16 would..
    } else if (elem_type == "MITC9") {
        using Physics = IsotropicShell<T, Data, is_nonlinear>;
        using Quad = QuadQuadraticQuadrature<T>;
        using Basis = LagrangeQuadBasis<T, Quad, 2>;
        using Assembler = MITCShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
        gatekeeper_method<T, Assembler>(is_multigrid, nxe, SR, nsmooth, ninnercyc, cycle_type, omega, pressure);
    } else if (elem_type == "MITC16") {
        using Physics = IsotropicShell<T, Data, is_nonlinear>;
        using Quad = QuadCubicQuadrature<T>;
        using Basis = LagrangeQuadBasis<T, Quad, 3>;
        using Assembler = MITCShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
        gatekeeper_method<T, Assembler>(is_multigrid, nxe, SR, nsmooth, ninnercyc, cycle_type, omega, pressure);
    } else if (elem_type == "CFI4") {
        using Physics = IsotropicShell<T, Data, is_nonlinear>;
        using Quad = QuadLinearQuadrature<T>;
        using Basis = ChebyshevQuadBasis<T, Quad, 1>;
        using Assembler = FullyIntegratedShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
        gatekeeper_method<T, Assembler>(is_multigrid, nxe, SR, nsmooth, ninnercyc, cycle_type, omega, pressure);
    } else if (elem_type == "CFI9") {
        using Physics = IsotropicShell<T, Data, is_nonlinear>;
        // probably do need quadratic, but need to fix assembly issues with 9 quadpts
        using Quad = QuadQuadraticQuadrature<T>;
        using Basis = ChebyshevQuadBasis<T, Quad, 2>;
        using Assembler = FullyIntegratedShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
        gatekeeper_method<T, Assembler>(is_multigrid, nxe, SR, nsmooth, ninnercyc, cycle_type, omega, pressure);
    } else if (elem_type == "CFI16") {
        using Physics = IsotropicShell<T, Data, is_nonlinear>;
        // probably do need quadratic, but need to fix assembly issues with 9 quadpts
        using Quad = QuadCubicQuadrature<T>;
        using Basis = ChebyshevQuadBasis<T, Quad, 3>;
        using Assembler = FullyIntegratedShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
        gatekeeper_method<T, Assembler>(is_multigrid, nxe, SR, nsmooth, ninnercyc, cycle_type, omega, pressure);
    } else if (elem_type == "HR4") {
        // hellinger-reissner element
        const bool HR = true; // whether is HR element (then physics has 5 extra DOF at start for strain-gap)
        using Physics = IsotropicShell<T, Data, is_nonlinear, HR>;
        using Quad = QuadLinearQuadrature<T>;
        using Basis = LagrangeQuadBasis<T, Quad, 1>;
        using Assembler = HellingerReissnerShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
        gatekeeper_method<T, Assembler>(is_multigrid, nxe, SR, nsmooth, ninnercyc, cycle_type, omega, pressure);
    } else {
        printf("ERROR : didn't run anything, elem type not in available types (see main function)\n");
    }

    return 0;

    
}
