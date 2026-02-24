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

// wing stuff
#include "multigrid/prolongation/unstructured.h"
#include "multigrid/smoothers/_wingbox_coloring.h"

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
void gsmc_gmres_solve(int level, MPI_Comm comm, double SR, int nsmooth, T omegaMC, T force = 5.0e7) {
    /* gauss-seidel multicolor GMRES solve */

    using Basis = typename Assembler::Basis;
    using Physics = typename Assembler::Phys;
    const SCALER scaler  = LINE_SEARCH;
    using Smoother = MulticolorGSSmoother_V1<Assembler>;
    using Prolongation = UnstructuredProlongation<Assembler, Basis, true>; 
    using GRID = SingleGrid<Assembler, Prolongation, Smoother, scaler>;
    using Data = ShellIsotropicData<T, false>;

    // for K-cycles
    // const int N_SUBSPACE = 50;
    const int N_SUBSPACE = 100;
    using Precond = Smoother;
    using GMRES = GMRESSolver<T, GRID, N_SUBSPACE>;

    // create cublas and cusparse handles (single one each)
    // -----------------------------------------------------
    cublasHandle_t cublasHandle = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));
    cusparseHandle_t cusparseHandle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));


    CHECK_CUDA(cudaDeviceSynchronize());
    auto start0 = std::chrono::high_resolution_clock::now();


    // create the assembler
    TACSMeshLoader mesh_loader{comm};
    std::string fname = "../../multigrid/3_aob_wing/meshes/aob_wing_L" + std::to_string(level) + ".bdf";
    mesh_loader.scanBDFFile(fname.c_str());
    double E = 70e9, nu = 0.3, thick = 2.0 / SR;  // material & thick properties (start thicker first try)
    printf("making assembler for mesh '%s'\n", fname.c_str());
    auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));

    // make the loads on the wing
    int nvars = assembler.get_num_vars();
    int nnodes = assembler.get_num_nodes();
    HostVec<T> h_loads(nvars);
    double load_mag = force / nnodes; // estimate for nodal load mag
    double *my_loads = h_loads.getPtr();
    for (int inode = 0; inode < nnodes; inode++) {
        my_loads[6 * inode + 2] = load_mag;
    }

    // perform multicolor reordering
    bool custom_wing_coloring = true;
    // bool custom_wing_coloring = false;

    auto &bsr_data = assembler.getBsrData();
    int num_colors, *_color_rowp;
    if (custom_wing_coloring) {
        WingboxMultiColoring<Assembler>::apply_coloring(assembler, bsr_data, num_colors, _color_rowp);
    } else {
        bsr_data.multicolor_reordering(num_colors, _color_rowp);
    }
    bsr_data.compute_nofill_pattern();
    auto h_color_rowp = HostVec<int>(num_colors + 1, _color_rowp);
    assembler.moveBsrDataToDevice();

    // create the device loads and kmat
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
    auto smoother = new Smoother(cublasHandle, cusparseHandle, assembler, kmat, h_color_rowp, omegaMC, false, nsmooth);
    int ELEM_MAX = 10; // num nearby elements of each fine node for nz pattern construction
    auto prolongation = new Prolongation(cusparseHandle, assembler, ELEM_MAX);
    auto grid = new GRID(assembler, prolongation, smoother, kmat, loads, cublasHandle, cusparseHandle);
    
    // create the preconditioner and GMRES solver now
    auto pc = smoother;
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
    bool fail = gmres_solver->solve(grid->d_defect, lin_soln, true);
    // printf("\t\ndone with GSMC-GMRES linear solve\n");

    auto end_solve = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = end_solve - start_solve;
    printf("\tGMRES solve time %.2e on %d level\n", solve_time.count(), level);

    // // print to VTK (permuting from solve to vis order)
    int *d_perm = gmres_solver->grid->d_perm;
    auto h_soln = lin_soln.createPermuteVec(6, d_perm).createHostVec();
    printToVTK<Assembler,HostVec<T>>(gmres_solver->grid->assembler, h_soln, "out/wing_kry_lin.vtk");
    T lin_max_disp = get_max_disp(lin_soln);

    if (fail) {
        printf("\tGMRES solved failed, so not proceeding to nonlinear solves\n");
        return;
    }

    // return; // for now DEBUG


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
    printToVTK<Assembler,HostVec<T>>(assembler, h_vars, "out/wing_kry_nl.vtk");

    // important to know reduction for how NL regime we are
    T ratio = nl_max_disp / lin_max_disp;
    printf("lin max disp %.8e, nl max disp %.8e, ratio = %.8e\n", lin_max_disp, nl_max_disp, ratio);

    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> nl_solve_time = end1 - start1;
    int ndof = assembler.get_num_vars();
    double total = startup_time.count() + nl_solve_time.count();
    printf("nonlinear Newton-Raphson GSMC-GMRES solve of AOB-WING geom, ndof %d : startup time %.2e, solve time %.2e, total %.2e\n", ndof, startup_time.count(), nl_solve_time.count(), total);

    // // free and cleanup
    // // --------------------

    // //  nl_solver.free();
    // kmg->free();
    // fine_assembler.free();
}

template <typename T, class Assembler>
void solve_direct(int level, MPI_Comm comm, double SR, T force = 5.0e7) {

    /* direct NL solve used to check that how NL the problem is and how */

    using Basis = typename Assembler::Basis;
    using Physics = typename Assembler::Phys;
    using Data = ShellIsotropicData<T, false>;

    CHECK_CUDA(cudaDeviceSynchronize());
    auto start0 = std::chrono::high_resolution_clock::now();

    // create the assembler
    TACSMeshLoader mesh_loader{comm};
    std::string fname = "../../multigrid/3_aob_wing/meshes/aob_wing_L" + std::to_string(level) + ".bdf";
    mesh_loader.scanBDFFile(fname.c_str());
    double E = 70e9, nu = 0.3, thick = 2.0 / SR;  // material & thick properties (start thicker first try)
    printf("making assembler for mesh '%s'\n", fname.c_str());
    auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));

    // make the loads on the wing
    int nvars = assembler.get_num_vars();
    int nnodes = assembler.get_num_nodes();
    HostVec<T> h_loads(nvars);
    double load_mag = force / nnodes; // estimate for nodal load mag
    double *my_loads = h_loads.getPtr();
    for (int inode = 0; inode < nnodes; inode++) {
        my_loads[6 * inode + 2] = load_mag;
    }

    auto &bsr_data = assembler.getBsrData();
    double fillin = 10.0;  // 10.0
    bool print = true;
    bsr_data.AMD_reordering();
    bsr_data.compute_full_LU_pattern(fillin, print);
    assembler.moveBsrDataToDevice();

    cublasHandle_t cublasHandle = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));
    cusparseHandle_t cusparseHandle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

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
    printToVTK<Assembler,HostVec<T>>(assembler, h_vars, "out/wing_direct_nl.vtk");

    // important to know reduction for how NL regime we are
    T ratio = nl_max_disp / lin_max_disp;
    printf("lin max disp %.8e, nl max disp %.8e, ratio = %.8e\n", lin_max_disp, nl_max_disp, ratio);

    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = end1 - start1;
    int ndof = assembler.get_num_vars();
    double total = startup_time.count() + solve_time.count();
    printf("nonlinear Newton-Raphson Direct-LU solve of AOB-WING geom, ndof %d : startup time %.2e, solve time %.2e, total %.2e\n", ndof, startup_time.count(), solve_time.count(), total);

    // free and cleanup
    // --------------------
    
    // nl_solver.free();
    assembler.free();
}

template <typename T, class Assembler>
void gatekeeper_method(bool is_krylov, MPI_Comm comm, int level, double SR, int nsmooth, T omega, T force = 5.0e7) {
    if (is_krylov) {
        gsmc_gmres_solve<T, Assembler>(level, comm, SR, nsmooth, omega, force);
    } else {
        solve_direct<T, Assembler>(level, comm, SR, force);
    }
}

int main(int argc, char **argv) {
    // Intialize MPI and declare communicator
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    // input ----------
    bool is_krylov = true;
    double SR = 10.0; // default, the less slender it is, solves much faster
    double force = 4e7;
    int level = 2;
    double omega = 0.7; // default MC smoother omega

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
        } else if (strcmp(arg, "--level") == 0) {
            if (i + 1 < argc) {
                level = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --level\n";
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
        } else if (strcmp(arg, "--force") == 0) {
            if (i + 1 < argc) {
                force = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --force\n";
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

    printf("AOB-wing mesh with geomNL %s elements, level %d and SR %.2e\n------------\n", elem_type.c_str(), level, SR);
    if (elem_type == "MITC4") {
        using Basis = LagrangeQuadBasis<T, Quad, 1>;
        using Assembler = MITCShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
        gatekeeper_method<T, Assembler>(is_krylov, comm, level, SR, nsmooth, omega, force);
    } else if (elem_type == "CFI4") {
        using Basis = ChebyshevQuadBasis<T, Quad, 1>;
        using Assembler = FullyIntegratedShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
        gatekeeper_method<T, Assembler>(is_krylov, comm, level, SR, nsmooth, omega, force);
    } else if (elem_type == "CFI9") {
        using Basis = ChebyshevQuadBasis<T, Quad, 2>;
        using Assembler = FullyIntegratedShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
        gatekeeper_method<T, Assembler>(is_krylov, comm, level, SR, nsmooth, omega, force);
    } else {
        printf("ERROR : didn't run anything, elem type not in available types (see main function)\n");
    }
    

    return 0;

    
}
