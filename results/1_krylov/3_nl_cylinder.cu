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
#include "multigrid/prolongation/structured.h"
#include "multigrid/smoothers/mc_smooth1.h"
// #include "multigrid/smoothers/damped_jacobi.h"
#include "multigrid/smoothers/cheb4_poly.h"
#include "multigrid/smoothers/asw_struct.h"
#include "multigrid/smoothers/spai.h"
#include "multigrid/solvers/gmg.h"
#include <string>
#include <chrono>

// finalsolver
#include "multigrid/solvers/direct/cusp_directLU.h"
#include "multigrid/solvers/solve_utils.h"
#include "multigrid/solvers/krylov/bsr_gmres.h"
#include "multigrid/solvers/krylov/bsr_pcg.h"


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
void chebyshev_polynomial_solve(int nxe, double SR, int nsmooth, T omegaMC = 1.5, T pressure = 5.0e7, int ORDER = 1) {
    /* damped jacobi / chebyshev_polynomial PCG solve */

    using Basis = typename Assembler::Basis;
    using Physics = typename Assembler::Phys;
    const SCALER scaler  = LINE_SEARCH;
    using Smoother = ChebyshevPolynomialSmoother<Assembler, false>; 
    using Prolongation = StructuredProlongation<Assembler, CYLINDER>;
    using GRID = SingleGrid<Assembler, Prolongation, Smoother, scaler>;

    // for K-cycles
    // linear solver
    using Precond = Smoother;
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
    double uniform_force = pressure * 1.0 * 1.0;
    double nodal_loads = uniform_force; // / (nxe - 1) / (nxe - 1);
    nodal_loads *= (100.0 / SR) * (100.0 / SR) * (100.0 / SR);
    double Q = 1.0; // load magnitude
    T *my_loads = getCylinderLoads<T,  Basis,Physics, load_case>(nxe, nxe, L, R, nodal_loads);
    printf("making grid with nxe %d\n", nxe);

    // perform multicolor reordering
    auto &bsr_data = assembler.getBsrData();
    int num_colors, *_color_rowp;
    bsr_data.multicolor_reordering(num_colors, _color_rowp);
    bsr_data.compute_nofill_pattern();
    auto h_color_rowp = HostVec<int>(num_colors + 1, _color_rowp);
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
    printf("\tassemble kmat in %.3e sec\n", assembly_time.count());

    // build smoother and prolongations..
    // nsmooth steps per precond set in the solver
    int ORDER_AND_SMOOTH = ORDER * nsmooth;
    auto smoother = new Smoother(cublasHandle, cusparseHandle, assembler, kmat, omegaMC, ORDER, nsmooth);
    auto prolongation = new Prolongation(assembler);
    auto grid = new GRID(assembler, prolongation, smoother, kmat, loads, cublasHandle, cusparseHandle);

    // create the preconditioner and GMRES solver now
    auto pc = smoother; // turns out the smoother does work somewhat
    // BaseSolver *pc = nullptr; // if want to try no precond for comparison (TEMP DEBUG)
    auto options = SolverOptions();
    options.ncycles = 4000; // number of max PCG cycles
    options.print_freq = 10;
    auto pcg_solver = new PCG(cublasHandle, cusparseHandle, grid, pc, options);
    pcg_solver->set_rel_tol(1e-6);
    pcg_solver->set_abs_tol(1e-6);
    pcg_solver->set_print(true);


    auto endstartup = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> startup_time = endstartup - start0;

    // run the linear solver
    CHECK_CUDA(cudaDeviceSynchronize());
    auto start_solve = std::chrono::high_resolution_clock::now();

    // pc->factor();

    // printf("running GSMC-GMRES linear solve\n");
    bool fail = pcg_solver->solve(grid->d_defect, lin_soln, true);
    // printf("\t\ndone with GSMC-GMRES linear solve\n");


    CHECK_CUDA(cudaDeviceSynchronize());
    auto end_solve = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = end_solve - start_solve;
    printf("\tCP-linear solve time %.2e on %d nxe\n", solve_time.count(), nxe);

    // // print to VTK (permuting from solve to vis order)
    int *d_perm = pcg_solver->grid->d_perm;
    auto h_soln = lin_soln.createPermuteVec(6, d_perm).createHostVec();
    printToVTK<Assembler,HostVec<T>>(pcg_solver->grid->assembler, h_soln, "out/cyl_kry_lin.vtk");
    T lin_max_disp = get_max_disp(lin_soln);

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
    using INK = InexactNewtonSolver<T, Mat, Vec, Assembler, PCG>;
    using NL = NonlinearContinuationSolver<T, Vec, Assembler, INK>;

    T initLinSolveRtol = 1e-3;
    T initLinSolveAtol = 1e-4;
    T minRtol = 1e-4, maxRtol = 1e-2; // don't want min rtol too low, cause GMRES will run out of steps (with GSMC precond)

    // get the loads out again cause they were permuted by grid
    bool perm_out = true;
    grid->getDefect(loads2, perm_out);

    pcg_solver->set_print(false);

    INK *inner_solver = new INK(cublasHandle, assembler, kmat, loads2, pcg_solver, initLinSolveRtol, initLinSolveAtol, minRtol, maxRtol);
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

    T nl_max_disp = get_max_disp(vars);

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
    printf("nonlinear Newton-Raphson CP-PCG solve of plate geom, ndof %d : startup time %.2e, solve time %.2e, total %.2e\n", ndof, startup_time.count(), nl_solve_time.count(), total);

    int num_newton = inner_solver->get_num_lin_solves();
    printf("\tlin solve time %.3e, nl solve time %.3e, #lin-solves %d\n", solve_time.count(), nl_solve_time.count(), num_newton);

    if (fail) {
        printf("\tnonlinear solver failed\n");
        return;
    }
}


template <typename T, class Assembler>
void asw_solve(int nxe, double SR, T omega, int n_smooth, int size, T pressure = 5.0e7) {
    /* SPAI-GMRES solve */

    using Basis = typename Assembler::Basis;
    using Physics = typename Assembler::Phys;
    const SCALER scaler  = LINE_SEARCH;
    using Smoother = StructuredAdditiveSchwarzSmoother<T, Assembler, S_CYLINDER>;
    using Prolongation = StructuredProlongation<Assembler, CYLINDER>;
    using GRID = SingleGrid<Assembler, Prolongation, Smoother, scaler>;

    // for K-cycles
    // linear solver
    // using Precond = CusparseMGDirectLU<T, Assembler>;
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
    double uniform_force = pressure * 1.0 * 1.0;
    double nodal_loads = uniform_force; // / (nxe - 1) / (nxe - 1);
    nodal_loads *= (100.0 / SR) * (100.0 / SR) * (100.0 / SR);
    double Q = 1.0; // load magnitude
    T *my_loads = getCylinderLoads<T,  Basis,Physics, load_case>(nxe, nxe, L, R, nodal_loads);
    printf("making grid with nxe %d\n", nxe);

    // perform multicolor reordering
    auto &bsr_data = assembler.getBsrData();
    bsr_data.compute_nofill_pattern();

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
    printf("\tassemble kmat in %.3e sec\n", assembly_time.count());

    // build smoother and prolongations..
    // auto smoother = new Smoother(cublasHandle, cusparseHandle, assembler, kmat, h_color_rowp, omegaMC, false, nsmooth);
    printf("making ASW smoother\n");
    // int size = 2; // size x size coupled blocks of smoothing
    auto smoother = new Smoother(cublasHandle, cusparseHandle, assembler, kmat, nxe + 1, nxe, 
        omega, n_smooth, size);
    printf("\tdone making ASW smoother\n");
    auto prolongation = new Prolongation(assembler);
    auto grid = new GRID(assembler, prolongation, smoother, kmat, loads, cublasHandle, cusparseHandle);
    auto pc = smoother;

    // create the preconditioner and GMRES solver now
    auto options = SolverOptions();
    options.ncycles = 4000; // number of max PCG cycles
    options.print_freq = 10;

    // PCG solver
    auto linear_solver = new PCG(cublasHandle, cusparseHandle, grid, pc, options);

    // // can maybe use BiCGStab if need be..
    // // only use GMRES if SR > 100
    // const int N_SUBSPACE = 200; // 100
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

    pc->factor();

    // printf("running GSMC-GMRES linear solve\n");
    bool fail = linear_solver->solve(grid->d_defect, lin_soln, true);
    // printf("\t\ndone with GSMC-GMRES linear solve\n");


    CHECK_CUDA(cudaDeviceSynchronize());
    auto end_solve = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = end_solve - start_solve;
    printf("\tASW-linear solve time %.2e on %d nxe\n", solve_time.count(), nxe);

    // // print to VTK (permuting from solve to vis order)
    int *d_perm = linear_solver->grid->d_perm;
    auto h_soln = lin_soln.createPermuteVec(6, d_perm).createHostVec();
    printToVTK<Assembler,HostVec<T>>(linear_solver->grid->assembler, h_soln, "out/cyl_kry_lin.vtk");
    T lin_max_disp = get_max_disp(lin_soln);

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
    using INK = InexactNewtonSolver<T, Mat, Vec, Assembler, PCG>;
    using NL = NonlinearContinuationSolver<T, Vec, Assembler, INK>;

    T initLinSolveRtol = 1e-3;
    T initLinSolveAtol = 1e-4;
    T minRtol = 1e-4, maxRtol = 1e-2; // don't want min rtol too low, cause GMRES will run out of steps (with GSMC precond)

    // get the loads out again cause they were permuted by grid
    bool perm_out = true;
    grid->getDefect(loads2, perm_out);

    linear_solver->set_print(false);

    INK *inner_solver = new INK(cublasHandle, assembler, kmat, loads2, linear_solver, initLinSolveRtol, initLinSolveAtol, minRtol, maxRtol);
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

    T nl_max_disp = get_max_disp(vars);

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
void gsmc_solve(int nxe, double SR, int nsmooth, T omegaMC = 1.5, T pressure = 5.0e7) {
    /* gauss-seidel multicolor GMRES solve */

    using Basis = typename Assembler::Basis;
    using Physics = typename Assembler::Phys;
    const SCALER scaler  = LINE_SEARCH;
    using Smoother = MulticolorGSSmoother_V1<Assembler>;
    using Prolongation = StructuredProlongation<Assembler, CYLINDER>;
    using GRID = SingleGrid<Assembler, Prolongation, Smoother, scaler>;

    // for K-cycles
    // linear solver
    using Precond = Smoother;
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
    double uniform_force = pressure * 1.0 * 1.0;
    double nodal_loads = uniform_force; // / (nxe - 1) / (nxe - 1);
    nodal_loads *= (100.0 / SR) * (100.0 / SR) * (100.0 / SR);
    double Q = 1.0; // load magnitude
    T *my_loads = getCylinderLoads<T,  Basis,Physics, load_case>(nxe, nxe, L, R, nodal_loads);
    printf("making grid with nxe %d\n", nxe);

    // perform multicolor reordering
    auto &bsr_data = assembler.getBsrData();
    int num_colors, *_color_rowp;
    bsr_data.multicolor_reordering(num_colors, _color_rowp);
    bsr_data.compute_nofill_pattern();
    auto h_color_rowp = HostVec<int>(num_colors + 1, _color_rowp);
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
    printf("\tassemble kmat in %.3e sec\n", assembly_time.count());

    // build smoother and prolongations..
    // nsmooth steps per precond set in the solver
    bool symmetric = true; // actually helps quite a bit for PCG (symmetric)
    // bool symmetric = false;
    auto smoother = new Smoother(cublasHandle, cusparseHandle, assembler, kmat, h_color_rowp, omegaMC, symmetric, nsmooth);
    auto prolongation = new Prolongation(assembler);
    auto grid = new GRID(assembler, prolongation, smoother, kmat, loads, cublasHandle, cusparseHandle);

    // create the preconditioner and GMRES solver now
    auto pc = smoother; // turns out the smoother does work somewhat
    // BaseSolver *pc = nullptr; // if want to try no precond for comparison (TEMP DEBUG)
    auto options = SolverOptions();
    options.ncycles = 4000; // number of max PCG cycles
    options.print_freq = 10;
    auto pcg_solver = new PCG(cublasHandle, cusparseHandle, grid, pc, options);
    pcg_solver->set_rel_tol(1e-6);
    pcg_solver->set_abs_tol(1e-6);
    pcg_solver->set_print(true);


    auto endstartup = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> startup_time = endstartup - start0;

    // run the linear solver
    CHECK_CUDA(cudaDeviceSynchronize());
    auto start_solve = std::chrono::high_resolution_clock::now();
    // printf("running GSMC-GMRES linear solve\n");
    bool fail = pcg_solver->solve(grid->d_defect, lin_soln, true);
    // printf("\t\ndone with GSMC-GMRES linear solve\n");


    CHECK_CUDA(cudaDeviceSynchronize());
    auto end_solve = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = end_solve - start_solve;
    printf("\tGSMC-linear solve time %.2e on %d nxe\n", solve_time.count(), nxe);

    // // print to VTK (permuting from solve to vis order)
    int *d_perm = pcg_solver->grid->d_perm;
    auto h_soln = lin_soln.createPermuteVec(6, d_perm).createHostVec();
    printToVTK<Assembler,HostVec<T>>(pcg_solver->grid->assembler, h_soln, "out/cyl_kry_lin.vtk");
    T lin_max_disp = get_max_disp(lin_soln);

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
    using INK = InexactNewtonSolver<T, Mat, Vec, Assembler, PCG>;
    using NL = NonlinearContinuationSolver<T, Vec, Assembler, INK>;

    T initLinSolveRtol = 1e-3;
    T initLinSolveAtol = 1e-4;
    T minRtol = 1e-4, maxRtol = 1e-2; // don't want min rtol too low, cause GMRES will run out of steps (with GSMC precond)

    // get the loads out again cause they were permuted by grid
    bool perm_out = true;
    grid->getDefect(loads2, perm_out);

    pcg_solver->set_print(false);

    INK *inner_solver = new INK(cublasHandle, assembler, kmat, loads2, pcg_solver, initLinSolveRtol, initLinSolveAtol, minRtol, maxRtol);
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

    T nl_max_disp = get_max_disp(vars);

    // print some of the data of host residual
    auto h_vars = vars.createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_vars, "out/cylinder_kry_nl.vtk");

    // important to know reduction for how NL regime we are
    T ratio = nl_max_disp / lin_max_disp;
    printf("lin max disp %.8e, nl max disp %.8e, ratio = %.8e\n", lin_max_disp, nl_max_disp, ratio);

    std::chrono::duration<double> nl_solve_time = end1 - start1;
    int ndof = assembler.get_num_vars();
    double total = startup_time.count() + nl_solve_time.count();
    printf("nonlinear Newton-Raphson GSMC-GMRES solve of plate geom, ndof %d : startup time %.2e, solve time %.2e, total %.2e\n", ndof, startup_time.count(), nl_solve_time.count(), total);

    int num_newton = inner_solver->get_num_lin_solves();
    printf("\tlin solve time %.3e, nl solve time %.3e, #lin-solves %d\n", solve_time.count(), nl_solve_time.count(), num_newton);

    if (fail) {
        printf("\tnonlinear solver failed\n");
        return;
    }
}

template <typename T, class Assembler>
void ilu_solve(int nxe, double SR, T qorder, int fill_level, int nsmooth, T omega = 1.5, T pressure = 5.0e7) {
    /* gauss-seidel multicolor GMRES solve */

    using Basis = typename Assembler::Basis;
    using Physics = typename Assembler::Phys;
    const SCALER scaler  = LINE_SEARCH;
    using Smoother = MulticolorGSSmoother_V1<Assembler>;
    using Prolongation = StructuredProlongation<Assembler, CYLINDER>;
    using GRID = SingleGrid<Assembler, Prolongation, Smoother, scaler>;

    // for K-cycles
    const bool MULTI_SMOOTH = true;
    using Precond = CusparseMGDirectLU<T, Assembler, MULTI_SMOOTH>;
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
    double uniform_force = pressure * 1.0 * 1.0;
    double nodal_loads = uniform_force; // / (nxe - 1) / (nxe - 1);
    nodal_loads *= (100.0 / SR) * (100.0 / SR) * (100.0 / SR);
    double Q = 1.0; // load magnitude
    T *my_loads = getCylinderLoads<T,  Basis,Physics, load_case>(nxe, nxe, L, R, nodal_loads);
    printf("making grid with nxe %d\n", nxe);

    // perform multicolor reordering
    auto &bsr_data = assembler.getBsrData();

    int kmat_orig_nnzb = bsr_data.nnzb;


    bsr_data.compute_nofill_pattern();
    // bsr_data.RCM_reordering();
    // bsr_data.AMD_reordering();
    bsr_data.qorder_reordering(qorder);
    bsr_data.compute_ILUk_pattern(fill_level, 10.0);

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
    printf("\tassemble kmat in %.3e sec\n", assembly_time.count());



    if (omega != 0.9) {
        printf("WARNING - choose omega like 0.9 or 0.7 for ILU(k) smoother typically and 2 to 4 smoothing steps\n");
    }

    // build smoother and prolongations..
    // nsmooth steps per precond set in the solver
    T omegaMC = 1.0;
    // int nsmooth = 1; // not used
    auto smoother = new Smoother(cublasHandle, cusparseHandle, assembler, kmat, h_color_rowp, omegaMC, false, nsmooth);
    auto prolongation = new Prolongation(assembler);
    auto grid = new GRID(assembler, prolongation, smoother, kmat, loads, cublasHandle, cusparseHandle);

    // the ILU preconditioner
    auto pc = new Precond(cublasHandle, cusparseHandle, assembler, kmat, omega, nsmooth); // turns out the smoother does work somewhat

    // create the preconditioner and GMRES solver now
    auto options = SolverOptions();
    options.ncycles = 4000; // number of max PCG cycles
    options.print_freq = 10;

    // PCG solver
    // auto linear_solver = new PCG(cublasHandle, cusparseHandle, grid, pc, options);

    // only use GMRES if SR > 100
    const int N_SUBSPACE = 300;
    // const int N_SUBSPACE = 150;
    using GMRES = GMRESSolver<T, GRID, N_SUBSPACE>;
    int MAX_ITER = 2000;
    auto linear_solver = new GMRES(cublasHandle, cusparseHandle, grid, pc, options, MAX_ITER);


    // out settings
    linear_solver->set_rel_tol(1e-6);
    linear_solver->set_abs_tol(1e-6);
    linear_solver->set_print(true);


    auto endstartup = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> startup_time = endstartup - start0;

    // run the linear solver
    CHECK_CUDA(cudaDeviceSynchronize());
    auto start_solve = std::chrono::high_resolution_clock::now();

    pc->factor();

    // printf("running GSMC-GMRES linear solve\n");
    bool fail = linear_solver->solve(grid->d_defect, lin_soln, true);
    // printf("\t\ndone with GSMC-GMRES linear solve\n");


    CHECK_CUDA(cudaDeviceSynchronize());
    auto end_solve = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = end_solve - start_solve;
    printf("\tILU-linear solve time %.2e on %d nxe\n", solve_time.count(), nxe);

    // // print to VTK (permuting from solve to vis order)
    int *d_perm = linear_solver->grid->d_perm;
    auto h_soln = lin_soln.createPermuteVec(6, d_perm).createHostVec();
    printToVTK<Assembler,HostVec<T>>(linear_solver->grid->assembler, h_soln, "out/cyl_kry_lin.vtk");
    T lin_max_disp = get_max_disp(lin_soln);

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
    using INK = InexactNewtonSolver<T, Mat, Vec, Assembler, GMRES>;
    using NL = NonlinearContinuationSolver<T, Vec, Assembler, INK>;

    T initLinSolveRtol = 1e-3;
    T initLinSolveAtol = 1e-4;
    T minRtol = 1e-4, maxRtol = 1e-2; // don't want min rtol too low, cause GMRES will run out of steps (with GSMC precond)

    // get the loads out again cause they were permuted by grid
    bool perm_out = true;
    grid->getDefect(loads2, perm_out);

    linear_solver->set_print(false);

    INK *inner_solver = new INK(cublasHandle, assembler, kmat, loads2, linear_solver, initLinSolveRtol, initLinSolveAtol, minRtol, maxRtol);
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

    T nl_max_disp = get_max_disp(vars);

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
    printf("nonlinear Newton-Raphson ILU-GMRES solve of plate geom, ndof %d : startup time %.2e, solve time %.2e, total %.2e\n", ndof, startup_time.count(), nl_solve_time.count(), total);

    int num_newton = inner_solver->get_num_lin_solves();
    printf("\tlin solve time %.3e, nl solve time %.3e, #lin-solves %d\n", solve_time.count(), nl_solve_time.count(), num_newton);

    if (fail) {
        printf("\tnonlinear solver failed\n");
        return;
    }
}

template <typename T, class Assembler>
void spai_solve(int nxe, double SR, int fill_level, int optim, T pressure = 5.0e7) {
    /* SPAI-GMRES solve */

    using Basis = typename Assembler::Basis;
    using Physics = typename Assembler::Phys;
    const SCALER scaler  = LINE_SEARCH;
    using Smoother = SPAI<T, Assembler>;
    using Prolongation = StructuredProlongation<Assembler, CYLINDER>;
    using GRID = SingleGrid<Assembler, Prolongation, Smoother, scaler>;

    // for K-cycles
    // linear solver
    // using Precond = CusparseMGDirectLU<T, Assembler>;
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
    double uniform_force = pressure * 1.0 * 1.0;
    double nodal_loads = uniform_force; // / (nxe - 1) / (nxe - 1);
    nodal_loads *= (100.0 / SR) * (100.0 / SR) * (100.0 / SR);
    double Q = 1.0; // load magnitude
    T *my_loads = getCylinderLoads<T,  Basis,Physics, load_case>(nxe, nxe, L, R, nodal_loads);
    printf("making grid with nxe %d\n", nxe);

    // perform multicolor reordering
    auto &bsr_data = assembler.getBsrData();
    bsr_data.compute_nofill_pattern();

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
    printf("\tassemble kmat in %.3e sec\n", assembly_time.count());

    // build smoother and prolongations..
    // auto smoother = new Smoother(cublasHandle, cusparseHandle, assembler, kmat, h_color_rowp, omegaMC, false, nsmooth);
    printf("making SPAI smoother\n");
    auto smoother = new Smoother(cublasHandle, cusparseHandle, assembler, kmat, fill_level, optim);
    printf("\tdone making SPAI smoother\n");
    auto prolongation = new Prolongation(assembler);
    auto grid = new GRID(assembler, prolongation, smoother, kmat, loads, cublasHandle, cusparseHandle);

    // the ILU preconditioner
    // auto pc = new Precond(cublasHandle, cusparseHandle, assembler, kmat); // turns out the smoother does work somewhat
    auto pc = smoother;

    // create the preconditioner and GMRES solver now
    auto options = SolverOptions();
    options.ncycles = 800; // number of max PCG cycles
    options.print_freq = 10;

    // PCG solver
    // auto linear_solver = new PCG(cublasHandle, cusparseHandle, grid, pc, options);

    // can maybe use BiCGStab if need be..
    // only use GMRES if SR > 100
    // const int N_SUBSPACE = 200; // 100
    const int N_SUBSPACE = 150;
    using GMRES = GMRESSolver<T, GRID, N_SUBSPACE>;
    int MAX_ITER = 4000;
    auto linear_solver = new GMRES(cublasHandle, cusparseHandle, grid, pc, options, MAX_ITER);

    // out settings
    linear_solver->set_rel_tol(1e-6);
    linear_solver->set_abs_tol(1e-6);
    linear_solver->set_print(true);

    auto endstartup = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> startup_time = endstartup - start0;

    // run the linear solver
    CHECK_CUDA(cudaDeviceSynchronize());
    auto start_solve = std::chrono::high_resolution_clock::now();

    pc->factor(); // spai optimization

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
    T log_red_rate = (log(init_resid) - log(final_resid)) / log(10.0) / solve_time.count();
    printf("\nSPAI-GMRES on cylinder case with %d nxe and %.4e SR\n", nxe, SR);
    printf("\tinit resid %.4e => final resid %.4e in %.3e sec, log10(reduction)/sec = %.6e\n", init_resid, final_resid, solve_time.count(), log_red_rate);

    // // print to VTK (permuting from solve to vis order)
    int *d_perm = linear_solver->grid->d_perm;
    auto h_soln = lin_soln.createPermuteVec(6, d_perm).createHostVec();
    printToVTK<Assembler,HostVec<T>>(linear_solver->grid->assembler, h_soln, "./out/plate_kry_lin.vtk");
    T lin_max_disp = get_max_disp(lin_soln);

    int nx = nxe + 1;
    int ndof = nx * nx * 6;
    double total = startup_time.count() + solve_time.count();
    size_t bytes_per_double = sizeof(double);
    double mem_mb = static_cast<double>(bytes_per_double) * static_cast<double>(bsr_data.nnzb) * 36.0 / 1024.0 / 1024.0;
    printf("SPAI(%d)-memory in MB %.4e with NDOF %d\n", fill_level, mem_mb, ndof);

    if (!fail) {
        T pc_compl = smoother->precond_complexity();

        // write to csv (this particular run)
        // ---------------------------------------
        std::ofstream csv("./out/cylinder-times.csv", std::ios::app);
        if (csv.tellp() == 0)
            csv << "t/R,nxe,NDOF,solver,pc_complexity,lin_runtime(s)\n";
        // Set high precision for CSV output
        csv << std::setprecision(15) << std::scientific;
        csv << (1.0/SR) << "," << nxe << "," << N << ","
            << "SPAI" << "," << pc_compl << "," << solve_time.count() << "\n";
    }


    if (fail) {
        printf("\tPCG linear solver failed\n");
        return;
    }
}

template <typename T, class Assembler>
void solve_direct(int nxe, double SR, T pressure = 5.0e7) {

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
    double uniform_force = pressure * 1.0 * 1.0;
    double nodal_loads = uniform_force; // / (nxe - 1) / (nxe - 1);
    nodal_loads *= (100.0 / SR) * (100.0 / SR) * (100.0 / SR);
    double Q = 1.0; // load magnitude
    T *my_loads = getCylinderLoads<T,  Basis, Physics, load_case>(nxe, nxe, L, R, nodal_loads);
    printf("making grid with nxe %d\n", nxe);

    // perform multicolor reordering
    auto &bsr_data = assembler.getBsrData();
    int kmat_orig_nnzb = bsr_data.nnzb;
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
    printf("\tassemble kmat in %.3e sec\n", assembly_time.count());

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
    options.ncycles = 4000; // number of max PCG cycles
    options.print_freq = 10;

    // PCG solver
    // auto linear_solver = new PCG(cublasHandle, cusparseHandle, grid, pc, options);

    // only use GMRES if SR > 100
    const int N_SUBSPACE = 100;
    using GMRES = GMRESSolver<T, GRID, N_SUBSPACE>;
    int MAX_ITER = N_SUBSPACE;
    auto linear_solver = new GMRES(cublasHandle, cusparseHandle, grid, pc, options, MAX_ITER);


    // out settings
    linear_solver->set_rel_tol(1e-6);
    linear_solver->set_abs_tol(1e-6);
    linear_solver->set_print(true);


    auto endstartup = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> startup_time = endstartup - start0;

    // run the linear solver
    CHECK_CUDA(cudaDeviceSynchronize());
    auto start_solve = std::chrono::high_resolution_clock::now();

    pc->factor();

    // printf("running GSMC-GMRES linear solve\n");
    bool fail = linear_solver->solve(grid->d_defect, lin_soln, true);
    // printf("\t\ndone with GSMC-GMRES linear solve\n");


    CHECK_CUDA(cudaDeviceSynchronize());
    auto end_solve = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = end_solve - start_solve;
    printf("\tLU-linear solve time %.2e on %d nxe\n", solve_time.count(), nxe);

    // // print to VTK (permuting from solve to vis order)
    int *d_perm = linear_solver->grid->d_perm;
    auto h_soln = lin_soln.createPermuteVec(6, d_perm).createHostVec();
    printToVTK<Assembler,HostVec<T>>(linear_solver->grid->assembler, h_soln, "out/cyl_kry_lin.vtk");
    T lin_max_disp = get_max_disp(lin_soln);

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
    using INK = InexactNewtonSolver<T, Mat, Vec, Assembler, GMRES>;
    using NL = NonlinearContinuationSolver<T, Vec, Assembler, INK>;

    T initLinSolveRtol = 1e-3;
    T initLinSolveAtol = 1e-4;
    T minRtol = 1e-4, maxRtol = 1e-2; // don't want min rtol too low, cause GMRES will run out of steps (with GSMC precond)

    // get the loads out again cause they were permuted by grid
    bool perm_out = true;
    grid->getDefect(loads2, perm_out);

    linear_solver->set_print(false);

    INK *inner_solver = new INK(cublasHandle, assembler, kmat, loads2, linear_solver, initLinSolveRtol, initLinSolveAtol, minRtol, maxRtol);
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

    T nl_max_disp = get_max_disp(vars);

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
    printf("nonlinear Newton-Raphson LU solve of plate geom, ndof %d : startup time %.2e, solve time %.2e, total %.2e\n", ndof, startup_time.count(), nl_solve_time.count(), total);

    int num_newton = inner_solver->get_num_lin_solves();
    printf("\tlin solve time %.3e, nl solve time %.3e, #lin-solves %d\n", solve_time.count(), nl_solve_time.count(), num_newton);

    if (fail) {
        printf("\tnonlinear solver failed\n");
        return;
    }
}

template <typename T, class Assembler>
void gatekeeper_method(std::string solver_type, int nxe, double SR, int nsmooth, T omega, int ORDER, T qorder, T load_mag = 5.0e7) {
    if (solver_type == "direct") {
        solve_direct<T, Assembler>(nxe, SR, load_mag);
    } else if (solver_type == "gsmc") {
        gsmc_solve<T, Assembler>(nxe, SR, nsmooth, omega, load_mag);
    } else if (solver_type == "jacobi") {
        int _ORDER = 1; // = damped jacobi
        chebyshev_polynomial_solve<T, Assembler>(nxe, SR, nsmooth, omega, load_mag, _ORDER);
    } else if (solver_type == "chebyshev") {
        // int ORDER = 4; // = 4th order chebyshev
        chebyshev_polynomial_solve<T, Assembler>(nxe, SR, nsmooth, omega, load_mag, ORDER);
    } else if (solver_type == "ilu0") {
        ilu_solve<T, Assembler>(nxe, SR, qorder, 0, nsmooth, omega, load_mag);
    } else if (solver_type == "ilu1") {
        ilu_solve<T, Assembler>(nxe, SR, qorder, 1, nsmooth, omega, load_mag);
    } else if (solver_type == "ilu2") {
        ilu_solve<T, Assembler>(nxe, SR, qorder, 2, nsmooth, omega, load_mag);
    } else if (solver_type == "ilu3") {
        ilu_solve<T, Assembler>(nxe, SR, qorder, 3, nsmooth, omega, load_mag);
    } else if (solver_type == "ilu4") {
        ilu_solve<T, Assembler>(nxe, SR, qorder, 4, nsmooth, omega, load_mag);
    } else if (solver_type == "ilu5") {
        ilu_solve<T, Assembler>(nxe, SR, qorder, 5, nsmooth, omega, load_mag);
    } else if (solver_type == "asw") {
        omega = 0.2;
        printf("ASW : setting omega = 0.2 and size to 2, comment this out to change it\n");
        int size = 2; // recommend omega = 0.2 here, nsmooth = 2
        asw_solve<T, Assembler>(nxe, SR, omega, nsmooth, size, load_mag);
    } else if (solver_type == "spai") {
        int fill_level = 1;
        // int fill_level = 2;
        // int optim = 10;
        int optim = 20;
        spai_solve<T, Assembler>(nxe, SR, fill_level, optim, load_mag);
    }
}

int main(int argc, char **argv) {
    // input ----------
    std::string solver_type = "gsmc";
    int nxe = 128; // default value (three grids)
    double SR = 10.0; // default, the less slender it is, solves much faster
    double pressure = 8.0e6;
    double omega = 0.35; // default omega
    int ORDER = 8; // for chebyshev polynomial
    double qorder = 1.0;
    int nsmooth = 2; 

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
        } else if (strcmp(arg, "--nxe") == 0) {
            if (i + 1 < argc) {
                nxe = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --nxe\n";
                return 1;
            }
        } else if (strcmp(arg, "--order") == 0) {
            if (i + 1 < argc) {
                ORDER = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --ORDER\n";
                return 1;
            }
        } else if (strcmp(arg, "--omega") == 0) {
            if (i + 1 < argc) {
                omega = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --omega\n";
                return 1;
            }
        } else if (strcmp(arg, "--qorder") == 0) {
            if (i + 1 < argc) {
                qorder = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --qorder\n";
                return 1;
            }
        } else if (strcmp(arg, "--sr") == 0) {
            if (i + 1 < argc) {
                SR = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --SR\n";
                return 1;
            }
        } else if (strcmp(arg, "--pressure") == 0) {
            if (i + 1 < argc) {
                pressure = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --load\n";
                return 1;
            }
        } else if (strcmp(arg, "--nsmooth") == 0) {
            if (i + 1 < argc) {
                nsmooth = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --nsmooth\n";
                return 1;
            }
        } else {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            std::cerr << "Usage: " << argv[0] << " [direct/krylov] [--nxe value] [--SR value] [--nsmooth int]" << std::endl;
            return 1;
        }
    }

    // type specifications here
    using T = double;   
    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    constexpr bool has_ref_axis = false;
    constexpr bool is_nonlinear = true; // this is a nonlinear GMG case
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;

    // have to use MITC4 shells cause this is before diff element types in paper
    using Basis = LagrangeQuadBasis<T, Quad, 1>;
    using Assembler = MITCShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
    gatekeeper_method<T, Assembler>(solver_type, nxe, SR, nsmooth, omega, ORDER, qorder, pressure);
    

    return 0;

    
}
