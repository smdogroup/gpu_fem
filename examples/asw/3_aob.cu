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
// #include "multigrid/smoothers/damped_jacobi.h"
#include "multigrid/smoothers/cheb4_poly.h"
#include "multigrid/solvers/gmg.h"
#include <string>
#include <chrono>

#include "multigrid/smoothers/asw_unstruct.h"

// wing stuff
#include "multigrid/prolongation/unstructured.h"
#include "multigrid/smoothers/_wingbox_coloring.h"

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
void asw_solve(int level, MPI_Comm comm, double SR, int nsmooth, T omega = 1.5, T force = 5.0e7) {
    /* damped jacobi / chebyshev_polynomial PCG solve */

    using Basis = typename Assembler::Basis;
    using Physics = typename Assembler::Phys;
    const SCALER scaler  = LINE_SEARCH;
    // using Smoother = ChebyshevPolynomialSmoother<Assembler, false>; 
    using Smoother = UnstructuredQuadElementAdditiveSchwarzSmoother<T, Assembler>;
    using Prolongation = UnstructuredProlongation<Assembler, Basis, true>; 
    using GRID = SingleGrid<Assembler, Prolongation, Smoother, scaler>;
    using Data = ShellIsotropicData<T, false>;

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

    // create the assembler
    TACSMeshLoader mesh_loader{comm};
    std::string fname = "../gmg/3_aob_wing/meshes/aob_wing_L" + std::to_string(level) + ".bdf";
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
    // bool custom_wing_coloring = true;
    bool custom_wing_coloring = false; // need this off to properly do schwarz method
    // as schwarz doesn't handle node permutations yet (can fix later though)

    auto &bsr_data = assembler.getBsrData();
    // int num_colors, *_color_rowp;
    // if (custom_wing_coloring) {
    //     WingboxMultiColoring<Assembler>::apply_coloring(assembler, bsr_data, num_colors, _color_rowp);
    // } else {
    //     bsr_data.multicolor_reordering(num_colors, _color_rowp);
    // }
    bsr_data.compute_nofill_pattern();
    // auto h_color_rowp = HostVec<int>(num_colors + 1, _color_rowp);
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
    auto smoother = new Smoother(cublasHandle, cusparseHandle, assembler, kmat, 
        omega, nsmooth);
    // int ORDER_AND_SMOOTH = ORDER * nsmooth;
    // auto smoother = new Smoother(cublasHandle, cusparseHandle, assembler, kmat, omegaMC, ORDER, nsmooth);
    int ELEM_MAX = 10; // num nearby elements of each fine node for nz pattern construction
    auto prolongation = new Prolongation(cusparseHandle, assembler, ELEM_MAX);
    auto grid = new GRID(assembler, prolongation, smoother, kmat, loads, cublasHandle, cusparseHandle);

    // create the preconditioner and GMRES solver now
    auto pc = smoother; // turns out the smoother does work somewhat
    // BaseSolver *pc = nullptr; // if want to try no precond for comparison (TEMP DEBUG)
    auto options = SolverOptions();
    options.ncycles = 200; // number of max PCG cycles
    options.print_freq = 10;
    
    // auto linear_solver = new PCG(cublasHandle, cusparseHandle, grid, pc, options);

    const int N_SUBSPACE = 800;
    using GMRES = GMRESSolver<T, GRID, N_SUBSPACE>;
    int MAX_ITER = N_SUBSPACE;
    auto linear_solver = new GMRES(cublasHandle, cusparseHandle, grid, pc, options, MAX_ITER);

    linear_solver->set_rel_tol(1e-6);
    linear_solver->set_abs_tol(1e-6);
    linear_solver->set_print(true);

    auto endstartup = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> startup_time = endstartup - start0;

    // run the linear solver
    CHECK_CUDA(cudaDeviceSynchronize());
    auto start_solve = std::chrono::high_resolution_clock::now();

    // get initial residual
    T init_resid = linear_solver->getResidualNorm(grid->d_defect, lin_soln);

    

    // linear solve
    bool fail = linear_solver->solve(grid->d_defect, lin_soln, true);
    // bool fail = pc->solve(grid->d_defect, lin_soln, true); // uncomment this to just see preconditioner solve
    
    // final residual
    T final_resid = linear_solver->getResidualNorm(grid->d_defect, lin_soln);

    CHECK_CUDA(cudaDeviceSynchronize());
    auto end_solve = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = end_solve - start_solve;

    // compute log residual reduction per unit time
    T log_red_rate = (log(init_resid) - log(final_resid)) / log(10.0) / solve_time.count();
    printf("\nASW-PCG on AOB wingbox case with %d level and %.4e SR\n", level, SR);
    printf("\tinit resid %.4e => final resid %.4e in %.2e sec, log10(reduction)/sec = %.6e\n", init_resid, final_resid, solve_time.count(), log_red_rate);

    // // print to VTK (permuting from solve to vis order)
    int *d_perm = linear_solver->grid->d_perm;
    auto h_soln = lin_soln.createPermuteVec(6, d_perm).createHostVec();
    printToVTK<Assembler,HostVec<T>>(linear_solver->grid->assembler, h_soln, "out/wing_kry_lin.vtk");
    T lin_max_disp = get_max_disp(lin_soln);

    if (fail) {
        printf("\tPCG linear solver failed\n");
        return;
    }
}

template <typename T, class Assembler>
void solve_direct(int level, MPI_Comm comm, double SR, T force = 5.0e7) {

    /* direct NL solve used to check that how NL the problem is and how */

    using Basis = typename Assembler::Basis;
    using Physics = typename Assembler::Phys;
    const SCALER scaler  = LINE_SEARCH;
    using Smoother = MulticolorGSSmoother_V1<Assembler>;
    using Prolongation = UnstructuredProlongation<Assembler, Basis, true>; 
    using GRID = SingleGrid<Assembler, Prolongation, Smoother, scaler>;
    using Data = ShellIsotropicData<T, false>;

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

    // create the assembler
    TACSMeshLoader mesh_loader{comm};
    std::string fname = "../gmg/3_aob_wing/meshes/aob_wing_L" + std::to_string(level) + ".bdf";
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
    int ELEM_MAX = 10; // num nearby elements of each fine node for nz pattern construction
    auto prolongation = new Prolongation(cusparseHandle, assembler, ELEM_MAX);
    auto grid = new GRID(assembler, prolongation, smoother, kmat, loads, cublasHandle, cusparseHandle);

    // the ILU preconditioner
    auto pc = new Precond(cublasHandle, cusparseHandle, assembler, kmat); // turns out the smoother does work somewhat

    // create the preconditioner and GMRES solver now
    auto options = SolverOptions();
    options.ncycles = 800; // number of max PCG cycles
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
    printf("\nGSMC-PCG on AOB wingbox case with %d level and %.4e SR\n", level, SR);
    printf("\tinit resid %.4e => final resid %.4e in %.2e sec, log10(reduction)/sec = %.6e\n", init_resid, final_resid, solve_time.count(), log_red_rate);

    // // print to VTK (permuting from solve to vis order)
    int *d_perm = linear_solver->grid->d_perm;
    auto h_soln = lin_soln.createPermuteVec(6, d_perm).createHostVec();
    printToVTK<Assembler,HostVec<T>>(linear_solver->grid->assembler, h_soln, "out/plate_kry_lin.vtk");
    T lin_max_disp = get_max_disp(lin_soln);

    if (fail) {
        printf("\tPCG linear solver failed\n");
        return;
    }
}

template <typename T, class Assembler>
void gatekeeper_method(std::string solver_type, int level, MPI_Comm comm, double SR, int nsmooth, T omega, T load_mag = 5.0e7) {
    if (solver_type == "direct") {
        solve_direct<T, Assembler>(level, comm, SR, load_mag);
    } else if (solver_type == "asw") {
        asw_solve<T, Assembler>(level, comm, SR, nsmooth, omega, load_mag);
    }
}

int main(int argc, char **argv) {
    // Intialize MPI and declare communicator
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    // input ----------
    std::string solver_type = "asw";
    int level = 1; // level 1 wingbox
    double SR = 100.0; // default, the less slender it is, solves much faster
    double pressure = 8.0e6;
    double omega = 0.2; // default omega
    int nsmooth = 64; 

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
    gatekeeper_method<T, Assembler>(solver_type, level, comm, SR, nsmooth, omega, pressure);
    

    return 0;

    
}
