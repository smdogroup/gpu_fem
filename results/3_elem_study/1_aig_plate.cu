// general gpu_fem imports
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"

#include <iomanip>

// new nonlinear solvers
#include "solvers/nonlinear_static/inexact_newton.h"
#include "solvers/nonlinear_static/continuation.h"

// shell imports
#include "element/shell/physics/isotropic_shell.h"

// aig plate
#include "element/plate/basis/bspline_basis.h"
#include "element/plate/aig_plate.h"

// local multigrid imports
#include "multigrid/grid.h"
#include "multigrid/utils/fea.h"
#include "multigrid/smoothers/cheb4_poly.h"
#include "multigrid/smoothers/mc_smooth1.h"
#include "multigrid/prolongation/structured_iga.h"
#include "multigrid/solvers/gmg.h"
#include <string>
#include <chrono>

// new multigrid imports for K-cycles, etc.
#include "multigrid/solvers/solve_utils.h"
#include "multigrid/solvers/direct/cusp_directLU.h"
#include "multigrid/solvers/krylov/bsr_pcg.h"
#include "multigrid/solvers/multilevel/kcycle.h"
#include "multigrid/solvers/multilevel/twolevel.h"

/*
to converge higher order shells, need more smoothing steps, like this
./0_plate.out --SR 300.0 --elem CFI9 --nxe 256 --nsmooth 8 --omega 0.7

lower order shells can use nsmooth = 1 and omega default
*/

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

template <typename T, class Physics>
T get_max_disp(DeviceVec<T> &d_soln, int idof = 2) {
    int block_dim = Physics::vars_per_node;
    int offset = block_dim - 6;

    T *h_soln = d_soln.createHostVec().getPtr();
    int nvars = d_soln.getSize();
    int nnodes = nvars / block_dim;
    T my_max = 0.0;
    for (int inode = 0; inode < nnodes; inode++) {
        T val = abs(h_soln[block_dim * inode + offset + idof]);
        if (val > my_max) my_max = val;
    }
    return my_max;
}

template <typename T, class Assembler>
void multigrid_solve(std::string elem_type, int nxe, double SR, int ORDER, int nsmooth, int ninnercyc, std::string cycle_type, T omega, T pressure = 5.0e7) {
    // geometric multigrid method here..
    // need to make a number of grids..
    using Basis = typename Assembler::Basis;
    using Physics = typename Assembler::Phys;
    // using Smoother = MulticolorGSSmoother_V1<Assembler>;
    using Smoother = ChebyshevPolynomialSmoother<Assembler>;
    using Prolongation = StructuredIGAProlongation<Assembler>;
    
    // sometimes line search helps, sometimes not
    // using GRID = SingleGrid<Assembler, Prolongation, Smoother, LINE_SEARCH>;
    using GRID = SingleGrid<Assembler, Prolongation, Smoother, NONE>;
    
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

    // T omegaLS_min = 0.01, omegaLS_max = 4.0;
    T omegaLS_min, omegaLS_max;
    // if (Basis::order > 1 || SR >= 1000.0) {
    //     omegaLS_min = 0.01, omegaLS_max = 4.0;
    // } else {
    //     omegaLS_min = 0.1, omegaLS_max = 2.0;
    // }
    // T omegaLS_min = 0.25, omegaLS_max = 2.0;
    // omegaLS_min = 0.5, omegaLS_max = 2.0;

    
    if (Basis::ISOGEOM) {
        // needs some improvement in defect / smoothed prolongation, otherwise can't do line searches rn..
        // equiv to basically not doing the line search
        omegaLS_min = 1.0, omegaLS_max = 1.0;
    }
    

    if (Basis::order == 1) {
        ORDER = 8;
    }

    // get nxe_min for not exactly power of 2 case
    // int nxe_start = 16 / Basis::order;
    int pre_nxe_min;

    if (Basis::order == 1) {
        int nxe_start = 32;
        pre_nxe_min = nxe > nxe_start ? nxe_start : 8;
    } else if (Basis::order == 2) {
        int nxe_start = 16;
        pre_nxe_min = nxe > nxe_start ? nxe_start : 4;
    } else if (Basis::order == 3) {
        int nxe_start = 8;
        pre_nxe_min = nxe > nxe_start ? nxe_start : 4;
    }

    int nxe_min = pre_nxe_min;
    for (int c_nxe = nxe; c_nxe >= pre_nxe_min; c_nxe /= 2) {
        nxe_min = c_nxe;
    }

    // if (cycle_type != "K") {
    //     printf("only does Kcycle mg in this example rn (just haven't adapted generally)\n");
    //     return;
    // }

    // make each grid
    for (int c_nxe = nxe; c_nxe >= nxe_min; c_nxe /= 2) {
        // make the assembler
        int c_nye = c_nxe;
        double Lx = 1.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 1.0 / SR, rho = 2500, ys = 350e6;
        int nxe_per_comp = c_nxe, nye_per_comp = c_nye; // for now (should have 25 grids)
        auto assembler = createPlateAssembler<Assembler>(c_nxe, c_nye, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp);
        double uniform_force = pressure * 1.0 * 1.0;
        double nodal_loads = uniform_force; // / (c_nxe - 1) / (c_nye - 1); // no longer need to normalize (it's an integral of pressure now)
        nodal_loads *= (100.0 / SR) * (100.0 / SR) * (100.0 / SR);
        // T *my_loads = getPlateLoads<T, Basis, Physics>(c_nxe, c_nye, Lx, Ly, nodal_loads);
        int m = 3, n = 1;
        bool uniform_load = false;
        // bool uniform_load = true; // makes it const load not sine load anymore
        // T *my_loads = getPlateMeshConvLoads<T, Assembler>(assembler, c_nxe, c_nye, Lx, Ly,nodal_loads, uniform_load, m, n);
        printf("making grid with nxe %d\n", c_nxe);

        int ndof = assembler.get_num_vars();
        int num_nodes = ndof / 6;
        T *my_loads = new T[ndof]; // TODO : change back to plate mesh conv loads later..
        memset(my_loads, 0.0, ndof * sizeof(T));
        for (int inode = 0; inode < num_nodes; inode++) {
            my_loads[6 * inode + 2] = pressure;
        }

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
        // int ORDER = Basis::order > 1 ? 4 : 8;
        // int ORDER = 8; // order 8 doesn't work with CFI9 and CFI16 for some reason..
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

    // T init_resid_nrm = is_kcycle ? kmg->grids[0].getResidNorm() : mg->grids[0].getResidNorm();

    int pre_smooth = nsmooth, post_smooth = nsmooth; // need a little extra smoothing on cylinder (compare to plate).. (cause of curvature I think..)
    // bool print = true;
    bool print = false;
    // T atol = 1e-6, rtol = 1e-6;
    // T rtol = 1e-9;
    // T rtol = 1e-11;
    // T atol = 1e-7;

    // T rtol = 1e-10, atol = 1e-5;
    // T rtol = 1e-13, atol = 1e-13;

    // don't need full converge for lower order elements (they won't get that far prob..)
    T atol = 1e-13;
    T rtol;
    if (Basis::order == 1) {
        rtol = 1e-6;
    } else if (Basis::order == 2) {
        rtol = 1e-9;
    } else if (Basis::order == 3) {
        rtol = 1e-13;
    }

    // bool double_smooth = true; // twice as many smoothing steps at lower levels (similar cost, better conv?)
    bool double_smooth = false;

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
    // std::vector<GRID> &grids = mg->grids;

    // fine grid states
    auto& fine_assembler_prev = grids[0].assembler;
    auto fine_soln = fine_assembler_prev.createVarsVec();
    auto fine_res = fine_assembler_prev.createVarsVec();
    auto fine_rhs = fine_assembler_prev.createVarsVec();
    auto fine_loads = fine_assembler_prev.createVarsVec();
    auto fine_vars = fine_assembler_prev.createVarsVec();
    auto& fine_kmat_prev = grids[0].Kmat;

    // get fine loads from fine grid init rhs
    bool perm_out = true;
    grids[0].getDefect(fine_loads, perm_out);
    fine_assembler_prev.apply_bcs(fine_loads);

    // debug prolongation first
    // -------------------------

    // // also make a fine direct solver (for debug), full LU pattern
    // // make the assembler
    // double Lx = 1.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 1.0 / SR, rho = 2500, ys = 350e6;
    // int nxe_per_comp = nxe, nye_per_comp = nxe; // for now (should have 25 grids)
    // auto fine_assembler = createPlateAssembler<Assembler>(nxe, nxe, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp);
    // printf("making fine grid direct solver with nxe %d\n", nxe);

    // auto &fine_bsr_data = fine_assembler.getBsrData();
    // // make the grid
    // fine_bsr_data.AMD_reordering();
    // fine_bsr_data.compute_full_LU_pattern(10.0, false);
    // fine_assembler.moveBsrDataToDevice();
    // // auto loads = fine_assembler.createVarsVec(my_loads);
    // // fine_assembler.apply_bcs(loads);
    // auto fine_kmat = createBsrMat<Assembler, VecType<T>>(fine_assembler);
    // fine_assembler.add_jacobian_fast(fine_kmat);
    // fine_assembler.apply_bcs(fine_kmat);

    // // fine grid direct solver
    // auto fine_solver = new CoarseSolver(cublasHandle, cusparseHandle, fine_assembler, fine_kmat);
    // fine_solver->factor_matrix();

    // // ====================================

    // if (cycle_type == "V") {
    //     // debug solve
    //     int n_cycles = 10;
    //     mg->template debug_vcycle_solve<Assembler>(fine_solver, fine_bsr_data, 0, pre_smooth, post_smooth, n_cycles, print, atol, rtol, double_smooth, print_freq); //(good option)
    // } else {
    //     printf("rerun and change to V-cycle for now (debug)\n");
    // }

    // printf("end early: doing debug of vcycle solve for now\n");
    // return;



    // 1) do a linear solve here
    // -------------------------------------------------------

    CHECK_CUDA(cudaDeviceSynchronize());
    auto start_lin = std::chrono::high_resolution_clock::now();

    kmg->set_print(true);
    bool fail = kmg->solve();
    kmg->set_print(false);

    if (fail) {
        printf("failed lin solve\n");
        return;
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    auto end_lin = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> lin_solve_time = end_lin - start_lin;

    printf("lin solve time in %.4e sec\n", lin_solve_time);

    T lin_max_disp = get_max_disp<T, Physics>(kmg->grids[0].d_soln);
    printf("lin max disp = %.4e\n", lin_max_disp);

    int *d_perm = kmg->grids[0].d_perm;
    auto h_soln = kmg->grids[0].d_soln.createPermuteVec(6, d_perm).createHostVec();
    // printToVTK<Assembler,HostVec<T>>(kmg->grids[0].assembler, h_soln, "out/plate_mg_lin.vtk");


    // // -----------------------------------------------------------
    // // 2) actually try Newton-mg solve here (this is just V1, later versions may use FMG cycle so less extra work needs to be done on fine grids)
    // //     i.e. you can do most of hte nonlinear solves to get in basin of attraction on coarser grids first.. (then nonlinear fine grid at end only, or some FMG cycle)

    // // new nonlinear solver
    // // ======================

    // std::chrono::duration<double> solve_time;
    // T nl_max_disp = 0.0;

    // if (false) { // temp
    // // if (Basis::order == 1) {
    //     // build the inexact newton + outer continuation solver
    //     using Mat = BsrMat<DeviceVec<T>>;
    //     using Vec = DeviceVec<T>;
    //     using INK = InexactNewtonSolver<T, Mat, Vec, Assembler, KMG>;
    //     using NL = NonlinearContinuationSolver<T, Vec, Assembler, INK>;

    //     INK *inner_solver = new INK(cublasHandle, fine_assembler_prev, fine_kmat_prev, fine_loads, kmg);
    //     // bool use_predictor = true, debug = false;
    //     bool use_predictor = false, debug = false;
    //     NL *nl_solver = new NL(cublasHandle, fine_assembler_prev, inner_solver, use_predictor, debug);

    //     // now try calling it
    //     T lambda0 = 0.2;
    //     // T lambda0 = 0.05;
    //     // T inner_atol = 1e-2;
    //     // T inner_atol = 1e-4;
    //     // T inner_atol = 1e-4;
    //     T inner_atol = 1e-6;

    //     CHECK_CUDA(cudaDeviceSynchronize());
    //     auto start1 = std::chrono::high_resolution_clock::now();

    //     if constexpr (!Physics::hellingerReissner) {
    //         nl_solver->solve(fine_vars, lambda0, inner_atol);
    //     }

    //     CHECK_CUDA(cudaDeviceSynchronize());
    //     auto end1 = std::chrono::high_resolution_clock::now();
    //     solve_time = end1 - start1;

    //     nl_max_disp = get_max_disp<T, Physics>(fine_vars);

    //     // print some of the data of host residual
    //     auto h_vars = fine_vars.createHostVec();
    //     printToVTK<Assembler,HostVec<T>>(fine_assembler_prev, h_vars, "out/plate_mg_nl.vtk");

    //     // important to know reduction for how NL regime we are
    //     T ratio = nl_max_disp / lin_max_disp;
    //     printf("lin max disp %.8e, nl max disp %.8e, ratio = %.8e\n", lin_max_disp, nl_max_disp, ratio);

    //     int ndof1 = fine_assembler_prev.get_num_vars();
    //     double total = startup_time.count() + solve_time.count();
    // }

    // int ndof = fine_assembler_prev.get_num_vars();
    // printf("nonlinear Newton-Raphson KMG solve of plate geom, ndof %d : startup time %.2e, solve time %.2e\n", ndof, startup_time.count(), solve_time.count());


    
    // // write to csv (this particular run)
    // // ---------------------------------------
    // std::ofstream csv("csv/_plate.csv", std::ios::app);
    // if (csv.tellp() == 0)
    //     csv << "t/R,nxe,NDOF,elem_type,lin_disp,nl_disp,lin_runtime(s),nl_runtime(s),solver\n";

    // double lin_runtime = lin_solve_time.count(), nl_runtime = solve_time.count();
    // // Set high precision for CSV output
    // csv << std::setprecision(15) << std::scientific;
    // csv << (1.0/SR) << "," << nxe << "," << ndof << ","
    //     << elem_type << "," << lin_max_disp << "," << nl_max_disp << ","
    //     << lin_runtime << "," << nl_runtime << ","
    //     << "gmg\n";

    // // free and cleanup
    // // --------------------

    // // nl_solver.free();
    // kmg->free();
    // fine_assembler.free();
}

template <typename T, class Assembler>
void solve_direct(std::string elem_type, int nxe, double SR, T pressure = 5.0e7) {

    /* direct NL solve used to check that how NL the problem is and how */

    using Basis = typename Assembler::Basis;
    using Physics = typename Assembler::Phys;

    CHECK_CUDA(cudaDeviceSynchronize());
    auto start0 = std::chrono::high_resolution_clock::now();

    int nye = nxe;
    double Lx = 1.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 1.0 / SR, rho = 2500, ys = 350e6;
    int nxe_per_comp = nxe, nye_per_comp = nye; // for now (should have 25 grids)
    auto assembler = createPlateAssembler<Assembler>(nxe, nye, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp);

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

    // get plate loads
    double uniform_force = pressure * 1.0 * 1.0;
    double nodal_loads = uniform_force; // / (nxe - 1) / (nxe - 1); // no longer need to normalize (it's an integral of pressure now)
    nodal_loads *= (100.0 / SR) * (100.0 / SR) * (100.0 / SR);
    // T *my_loads = getPlatePointLoad<T, Physics>(c_nxe, c_nye, Lx, Ly, Q);
    // T *my_loads = getPlateLoads<T, Basis, Physics>(nxe, nye, Lx, Ly, nodal_loads);
    // int m = 3, n = 1;
    // T *my_loads = getPlateMeshConvLoads<T, Assembler>(assembler, nxe, nxe, Lx, Ly, m, n, nodal_loads);
    int m = 3, n = 1;
    bool uniform_load = false;
    // bool uniform_load = true; // makes it const load not sine load anymore
    // T *my_loads = getPlateMeshConvLoads<T, Assembler>(assembler, nxe, nxe, Lx, Ly,nodal_loads, uniform_load, m, n);
    int ndof = assembler.get_num_vars();
    int num_nodes = ndof / 6;
    T *my_loads = new T[ndof]; // TODO : change back to plate mesh conv loads later..
    memset(my_loads, 0.0, ndof * sizeof(T));
    for (int inode = 0; inode < num_nodes; inode++) {
        my_loads[6 * inode + 2] = pressure;
    }

    // print loads
    // printf("loads: ");
    // printVec<T>(11 * 3, my_loads);


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

    // CHECK_CUDA(cudaDeviceSynchronize());
    // auto h_mat1 = kmat.getVec().createHostVec();
    // const double *h_mat_ptr1 = h_mat1.getPtr();
    // for (int inz = 0; inz < 81; inz++) {
    //     const T *h_mat_block1 = &h_mat_ptr1[36 * inz];
    //     printf("------------------------\nkmat block (%d, %d):\n------------------------\n", inz % 9, inz / 9);
    //     for (int j = 0; j < 6; j++) {
    //         printf("row %d: ", j);
    //         printVec<double>(6, &h_mat_block1[6 * j]);
    //     }
    // }

    // bool include_cols = true; // maybe default include_cols = true not working well for IGA case?
    assembler.apply_bcs(kmat,true); //, include_cols); // comment out for temp debug
    CHECK_CUDA(cudaDeviceSynchronize());
    // int nnz = kmat.get_nnz();
    // printf("\tdone with jacobian with nnz %d\n", nnz);
    // // return;
    // CHECK_CUDA(cudaDeviceSynchronize());
    auto start_lin = std::chrono::high_resolution_clock::now();

    // auto h_mat = kmat.getVec().createHostVec();
    // const double *h_mat_ptr = h_mat.getPtr();
    // for (int inz = 0; inz < 81; inz++) {
    //     const T *h_mat_block = &h_mat_ptr[36 * inz];
    //     printf("------------------------\nkmat block (%d, %d):\n------------------------\n", inz % 9, inz / 9);
    //     for (int j = 0; j < 6; j++) {
    //         printf("row %d: ", j);
    //         printVec<double>(6, &h_mat_block[6 * j]);
    //     }
    // }


    printf("try linear solve\n");
    CUSPARSE::direct_LU_solve(kmat, loads, soln);
    printf("\tdone with linear solve\n");
    
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end_lin = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> lin_solve_time = end_lin - start_lin;

    T lin_max_disp = get_max_disp<T, Physics>(soln);
    printf("lin max disp = %.4e\n", lin_max_disp);

    auto h_soln = soln.createHostVec();
    printf("print solution\n");
    printToVTK<Assembler,HostVec<T>>(assembler, h_soln, "out/plate_lin.vtk");
    printf("\tprinted solution\n");
    

    // new nonlinear solver
    // ======================

    std::chrono::duration<double> solve_time;
    T nl_max_disp = 0.0;

    if (false) {
    // if (Basis::order == 1) {
        // build the inexact newton + outer continuation solver
        using Mat = BsrMat<DeviceVec<T>>;
        using Vec = DeviceVec<T>;
        using LinearSolver = CusparseMGDirectLU<T, Assembler>;
        // const bool DO_LINE_SEARCH = !Physics::hellingerReissner;
        const bool DO_LINE_SEARCH = true;
        using INK = InexactNewtonSolver<T, Mat, Vec, Assembler, LinearSolver, DO_LINE_SEARCH>;
        using NL = NonlinearContinuationSolver<T, Vec, Assembler, INK>;

        LinearSolver *solver = new LinearSolver(cublasHandle, cusparseHandle, assembler, kmat);
        INK *inner_solver = new INK(cublasHandle, assembler, kmat, loads, solver);
        NL *nl_solver = new NL(cublasHandle, assembler, inner_solver);

        // now try calling it
        T lambda0 = 0.2;
        // T lambda0 = 0.05;
        CHECK_CUDA(cudaDeviceSynchronize());
        auto start11 = std::chrono::high_resolution_clock::now();
        
        // skip hellinger-reissner NL solve for now
        if constexpr (!Physics::hellingerReissner) {
            nl_solver->solve(vars, lambda0);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        auto end1 = std::chrono::high_resolution_clock::now();
        solve_time = end1 - start11;

        nl_max_disp = get_max_disp<T, Physics>(vars);

        // print some of the data of host residual
        auto h_vars = vars.createHostVec();
        printToVTK<Assembler,HostVec<T>>(assembler, h_vars, "out/plate_nl.vtk");

        // important to know reduction for how NL regime we are
        T ratio = nl_max_disp / lin_max_disp;
        printf("lin max disp %.8e, nl max disp %.8e, ratio = %.8e\n", lin_max_disp, nl_max_disp, ratio);
    }

    
    double total = startup_time.count() + solve_time.count();
    printf("nonlinear Newton-Raphson Direct-LU solve of plate geom, ndof %d : startup time %.2e, solve time %.2e, total %.2e\n", ndof, startup_time.count(), solve_time.count(), total);

    // write to csv (this particular run)
    // ---------------------------------------
    std::ofstream csv("csv/_plate.csv", std::ios::app);
    if (csv.tellp() == 0)
        csv << "t/R,nxe,NDOF,elem_type,lin_disp,nl_disp,lin_runtime(s),nl_runtime(s)\n";

    double lin_runtime = lin_solve_time.count(), nl_runtime = solve_time.count();
    // Set high precision for CSV output
    csv << std::setprecision(15) << std::scientific;
    csv << (1.0/SR) << "," << nxe << "," << ndof << ","
        << elem_type << "," << lin_max_disp << "," << nl_max_disp << ","
        << lin_runtime << "," << nl_runtime << ","
        << "direct\n";

    // free and cleanup
    // --------------------
    
    // nl_solver.free();
    assembler.free();
}

template <typename T, class Assembler>
void gatekeeper_method(std::string elem_type, bool is_multigrid, int nxe, double SR, int ORDER, int nsmooth, int ninnercyc, std::string cycle_type, T omega, T load_mag = 5.0e7) {
    if (is_multigrid) {
        multigrid_solve<T, Assembler>(elem_type, nxe, SR, ORDER, nsmooth, ninnercyc, cycle_type, omega, load_mag);
    } else {
        solve_direct<T, Assembler>(elem_type, nxe, SR, load_mag);
    }
}

int main(int argc, char **argv) {
    // input ----------
    bool is_multigrid = true;
    int nxe = 128;
    double SR = 100.0; // default, the less slender it is, solves much faster
    int n_vcycles = 50;
    double pressure = 2.0e6;
    double omega = 0.8; // works better with omega near 1

    // old GSMC settings
    // int nsmooth = 1; // use nsmooth = 2 for 3rd order elements (needed)
    int nsmooth = 2;
    int ninnercyc = 1;
    // int ORDER = 4;
    int ORDER = 8;
    std::string cycle_type = "K"; // "V", "F", "W", "K"
    // std::string cycle_type = "V";

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
                std::cerr << "Missing value for --ORDER\n";
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
    constexpr bool has_ref_axis = false;
    constexpr bool is_nonlinear = false; // this is a linear GMG case
    using Data = ShellIsotropicData<T, has_ref_axis>;

    printf("plate mesh with geomNL AIG9 elements, nxe %d and SR %.2e\n------------\n", nxe, SR);
    
    // runs like AIG9 when I finish implementing
    using Physics = IsotropicShell<T, Data, is_nonlinear>;
    using Quad = QuadQuadraticQuadrature<T>;
    using Basis = BsplineQuadBasis<T, Quad, 2>;
    using Assembler = AsymptoticIsogeometricPlateAssembler<T, Basis, Physics, VecType, BsrMat>;
    // SR = 5.0;
    gatekeeper_method<T, Assembler>("AIG9", is_multigrid, nxe, SR, ORDER, nsmooth, ninnercyc, cycle_type, omega, pressure);

    return 0;

    
}
