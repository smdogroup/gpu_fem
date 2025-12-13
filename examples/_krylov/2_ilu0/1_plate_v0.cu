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

// local multigrid imports
#include "multigrid/grid.h"
#include "multigrid/utils/fea.h"
#include "multigrid/prolongation/structured.h"
#include "multigrid/smoothers/mc_smooth1.h"
#include "multigrid/solvers/gmg.h"
#include <string>
#include <chrono>

// finalsolver
#include "multigrid/solvers/direct/cusp_directLU.h"
#include "multigrid/solvers/solve_utils.h"
#include "multigrid/solvers/krylov/bsr_gmres.h"


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
void gmres_solve(int nxe, double SR, int nsmooth, T pressure = 5.0e7) {
    /* gauss-seidel multicolor GMRES solve */

    using Basis = typename Assembler::Basis;
    using Physics = typename Assembler::Phys;
    const SCALER scaler  = LINE_SEARCH;
    using Smoother = MulticolorGSSmoother_V1<Assembler>;
    using Prolongation = StructuredProlongation<Assembler, PLATE>; // technically don't need this here.. but GMRES solver uses grid right now..
    using GRID = SingleGrid<Assembler, Prolongation, Smoother, scaler>;

    // for K-cycles
    // const int N_SUBSPACE = 50;
    const int N_SUBSPACE = 100;
    // becomes ILU(0) preconditioner / smoother when you force no fill pattern on it
    using Precond = CusparseMGDirectLU<T, Assembler>;
    using GMRES = GMRESSolver<T, GRID, N_SUBSPACE>;

    // create cublas and cusparse handles (single one each)
    // -----------------------------------------------------
    cublasHandle_t cublasHandle = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));
    cusparseHandle_t cusparseHandle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));


    CHECK_CUDA(cudaDeviceSynchronize());
    auto start0 = std::chrono::high_resolution_clock::now();


    double Lx = 1.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 1.0 / SR, rho = 2500, ys = 350e6;
    int nxe_per_comp = nxe / 4, nye_per_comp = nxe/4; // for now (should have 25 grids)
    auto assembler = createPlateAssembler<Assembler>(nxe, nxe, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp);
    double uniform_force = pressure * 1.0 * 1.0;
    double nodal_loads = uniform_force / (nxe - 1) / (nxe - 1);
    nodal_loads *= (100.0 / SR) * (100.0 / SR) * (100.0 / SR);
    T *my_loads = getPlateLoads<T, Physics>(nxe, nxe, Lx, Ly, nodal_loads);
    printf("making grid with nxe %d\n", nxe);

    // perform multicolor reordering
    auto &bsr_data = assembler.getBsrData();
    int num_colors, *_color_rowp;
    
    // can do multicolor ILU(0) much more parallel if you want
    bsr_data.multicolor_reordering(num_colors, _color_rowp);
    
    // num_colors = 1, _color_rowp = new int[1];
    // double qdiag_strength = 0.25; // higher number means closer to diag (>1), <1 means wider number
    // bsr_data.qorder_reordering(qdiag_strength);

    bsr_data.compute_nofill_pattern(); // no fill so it's ILU(0) ordering
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
    printf("\tassemble kmat time %.2e\n", assembly_time.count());

    // build smoother and prolongations..
    // nsmooth steps per precond set in the solver
    auto smoother = new Smoother(cublasHandle, cusparseHandle, assembler, kmat, h_color_rowp, 1.0, false, nsmooth);
    auto prolongation = new Prolongation(assembler);
    auto grid = new GRID(assembler, prolongation, smoother, kmat, loads, cublasHandle, cusparseHandle);
    
    // create the preconditioner and GMRES solver now
    auto pc = new Precond(cublasHandle, cusparseHandle, assembler, kmat);
    auto options = SolverOptions();
    options.print_freq = 10;
    int MAX_ITER = N_SUBSPACE;
    auto gmres_solver = new GMRES(cublasHandle, cusparseHandle, grid, pc, options, MAX_ITER);
    gmres_solver->set_rel_tol(1e-4);
    gmres_solver->set_abs_tol(1e-6);
    gmres_solver->set_print(true);

    auto endstartup = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> startup_time = endstartup - start0;

    // run the linear solver
    auto start_solve = std::chrono::high_resolution_clock::now();
    // printf("running GSMC-GMRES linear solve\n");
    gmres_solver->solve(grid->d_defect, lin_soln, true);
    // printf("\t\ndone with GSMC-GMRES linear solve\n");

    auto end_solve = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = end_solve - start_solve;
    printf("\tGMRES solve time %.2e on %d nxe\n", solve_time.count(), nxe);

    // // print to VTK (permuting from solve to vis order)
    int *d_perm = gmres_solver->grid->d_perm;
    auto h_soln = lin_soln.createPermuteVec(6, d_perm).createHostVec();
    printToVTK<Assembler,HostVec<T>>(gmres_solver->grid->assembler, h_soln, "out/plate_kry_lin.vtk");
    T lin_max_disp = get_max_disp(lin_soln);


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

    gmres_solver->set_print(false);

    INK *inner_solver = new INK(cublasHandle, assembler, kmat, loads2, gmres_solver, initLinSolveRtol, initLinSolveAtol, minRtol, maxRtol);
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

    auto start1 = std::chrono::high_resolution_clock::now();
    nl_solver->solve(vars, lambda0, inner_atol, lambdaf);
    T nl_max_disp = get_max_disp(vars);

    // print some of the data of host residual
    auto h_vars = vars.createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_vars, "out/plate_kry_nl.vtk");

    // important to know reduction for how NL regime we are
    T ratio = nl_max_disp / lin_max_disp;
    printf("lin max disp %.8e, nl max disp %.8e, ratio = %.8e\n", lin_max_disp, nl_max_disp, ratio);

    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> nl_solve_time = end1 - start1;
    int ndof = assembler.get_num_vars();
    double total = startup_time.count() + nl_solve_time.count();
    printf("nonlinear Newton-Raphson ILU0-GMRES solve of plate geom, ndof %d : startup time %.2e, solve time %.2e, total %.2e\n", ndof, startup_time.count(), nl_solve_time.count(), total);

    // // free and cleanup
    // // --------------------

    // //  nl_solver.free();
    // kmg->free();
    // fine_assembler.free();
}

template <typename T, class Assembler>
void solve_direct(int nxe, double SR, T pressure = 5.0e7) {

    /* direct NL solve used to check that how NL the problem is and how */

    using Basis = typename Assembler::Basis;
    using Physics = typename Assembler::Phys;

    CHECK_CUDA(cudaDeviceSynchronize());
    auto start0 = std::chrono::high_resolution_clock::now();

    int nye = nxe;
    double Lx = 1.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 1.0 / SR, rho = 2500, ys = 350e6;
    int nxe_per_comp = nxe / 4, nye_per_comp = nye/4; // for now (should have 25 grids)
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
    double nodal_loads = uniform_force / (nxe - 1) / (nxe - 1);
    nodal_loads *= (100.0 / SR) * (100.0 / SR) * (100.0 / SR);
    // T *my_loads = getPlatePointLoad<T, Physics>(c_nxe, c_nye, Lx, Ly, Q);
    T *my_loads = getPlateLoads<T, Physics>(nxe, nye, Lx, Ly, nodal_loads);

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

    assembler.add_jacobian_fast(kmat);
    assembler.apply_bcs(kmat);
    CUSPARSE::direct_LU_solve(kmat, loads, soln);
    T lin_max_disp = get_max_disp(soln);
    auto h_soln = soln.createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_soln, "out/plate_direct_lin.vtk");
    

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
    printToVTK<Assembler,HostVec<T>>(assembler, h_vars, "out/plate_direct_nl.vtk");

    // important to know reduction for how NL regime we are
    T ratio = nl_max_disp / lin_max_disp;
    printf("lin max disp %.8e, nl max disp %.8e, ratio = %.8e\n", lin_max_disp, nl_max_disp, ratio);

    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = end1 - start1;
    int ndof = assembler.get_num_vars();
    double total = startup_time.count() + solve_time.count();
    printf("nonlinear Newton-Raphson Direct-LU solve of plate geom, ndof %d : startup time %.2e, solve time %.2e, total %.2e\n", ndof, startup_time.count(), solve_time.count(), total);

    // free and cleanup
    // --------------------
    
    // nl_solver.free();
    assembler.free();
}

template <typename T, class Assembler>
void gatekeeper_method(bool is_krylov, int nxe, double SR, int nsmooth, T load_mag = 5.0e7) {
    if (is_krylov) {
        gmres_solve<T, Assembler>(nxe, SR, nsmooth, load_mag);
    } else {
        solve_direct<T, Assembler>(nxe, SR, load_mag);
    }
}

int main(int argc, char **argv) {
    // input ----------
    bool is_krylov = true;
    int nxe = 128; // default value (three grids)
    double SR = 10.0; // default, the less slender it is, solves much faster
    double pressure = 8.0e6;

    int nsmooth = 40; 
    // std::string elem_type = "MITC4"; // 'MITC4', 'CFI4', 'CFI9'
    std::string elem_type = "CFI4"; // careful CFI4 shear locks some (need better element here)

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        char* arg = argv[i];
        to_lowercase(arg);

        if (strcmp(arg, "direct") == 0) {
            is_krylov = false;
        } else if (strcmp(arg, "krylov") == 0) {
            is_krylov = true;
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
        } else if (strcmp(arg, "--pressure") == 0) {
            if (i + 1 < argc) {
                pressure = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --load\n";
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

    printf("plate mesh with geomNL %s elements, nxe %d and SR %.2e\n------------\n", elem_type.c_str(), nxe, SR);
    if (elem_type == "MITC4") {
        using Basis = LagrangeQuadBasis<T, Quad, 1>;
        using Assembler = MITCShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
        gatekeeper_method<T, Assembler>(is_krylov, nxe, SR, nsmooth, pressure);
    } else if (elem_type == "CFI4") {
        using Basis = ChebyshevQuadBasis<T, Quad, 1>;
        using Assembler = FullyIntegratedShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
        gatekeeper_method<T, Assembler>(is_krylov, nxe, SR, nsmooth, pressure);
    } else if (elem_type == "CFI9") {
        using Basis = ChebyshevQuadBasis<T, Quad, 2>;
        using Assembler = FullyIntegratedShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
        gatekeeper_method<T, Assembler>(is_krylov, nxe, SR, nsmooth, pressure);
    } else {
        printf("ERROR : didn't run anything, elem type not in available types (see main function)\n");
    }
    

    return 0;

    
}
