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
#include "multigrid/smoothers/_wingbox_coloring.h"
#include "multigrid/smoothers/cheb4_poly.h"
#include "multigrid/smoothers/mc_smooth1.h"
#include "multigrid/prolongation/unstructured.h"
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
void multigrid_solve(MPI_Comm &comm, int level, std::string smoother_type, double SR, int nsmooth, int ninnercyc, T omega, int ORDER) {
    // geometric multigrid method here..
    // need to make a number of grids..

    using Basis = typename Assembler::Basis;
    using Physics = typename Assembler::Phys;
    using Data = typename Physics::Data;
    const SCALER scaler  = LINE_SEARCH;
    const bool is_bsr = false; // no difference in intra-nodal (default old working prolong)
    using Prolongation = UnstructuredProlongation<Assembler, Basis, is_bsr>; 
    using GRID = SingleGrid<Assembler, Prolongation, Smoother, scaler>;
    using CoarseSolver = CusparseMGDirectLU<T, Assembler>;
    // using MG = GeometricMultigridSolver<GRID, CoarseSolver>;
    
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
    
    for (int i = level; i >= 0; i--) {

        // read the ESP/CAPS => nastran mesh for TACS
        TACSMeshLoader mesh_loader{comm};
        std::string fname = "../../multigrid/3_aob_wing/meshes/aob_wing_L" + std::to_string(i) + ".bdf";
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
        assembler.template add_jacobian_fast<elems_per_blockk>(kmat);
        assembler.apply_bcs(kmat);
        CHECK_CUDA(cudaDeviceSynchronize());
        auto enda = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> assembly_time = enda - starta;
        printf("\tassemble kmat in %.2e sec\n", assembly_time.count());

        // build smoother and prolongations
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
        } else {
            static_assert(sizeof(Smoother) == 0, "Unsupported smoother type");
        }

        int ELEM_MAX = 10; // num nearby elements of each fine node for nz pattern construction
        // int ELEM_MAX = 4;
        auto prolongation = new Prolongation(cusparseHandle, assembler, ELEM_MAX);
        auto grid = GRID(assembler, prolongation, smoother, kmat, loads, cublasHandle, cusparseHandle);
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


    // get initial residual
    KrylovSolve* outer_solver = static_cast<KrylovSolve*>(kmg->outer_solver); 
    T init_resid = outer_solver->getResidualNorm(rhs, soln);

    CHECK_CUDA(cudaDeviceSynchronize());
    auto start1 = std::chrono::high_resolution_clock::now();

    // fastest is K-cycle usually
    kmg->coarse_solver->factor(); // factor
    kmg->solve(rhs, soln);

    // get final residual
    T final_resid = outer_solver->getResidualNorm(rhs, soln);

    CHECK_CUDA(cudaDeviceSynchronize());
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = end1 - start1;
    int ndof = kmg->grids[0].N;
    double total = startup_time.count() + solve_time.count();
    double mem_MB = kmg->get_memory_usage_mb();
    printf("AOB-wing GMG solve, ndof %d : startup time %.2e, solve time %.2e, total %.2e, with mem(MB) %.2e\n", ndof, startup_time.count(), solve_time.count(), total, mem_MB);

    // compute log residual reduction per unit time
    T log_red_rate = (log(init_resid) - log(final_resid)) / log(10.0) / solve_time.count();
    printf("\nGMG-PCG on AOB-wing case with %d level and %.4e SR\n", level, SR);
    printf("\tinit resid %.4e => final resid %.4e in %.2e sec, log10(reduction)/sec = %.6e\n", init_resid, final_resid, solve_time.count(), log_red_rate);

    // print some of the data of host residual
    int *d_perm = kmg->grids[0].d_perm;
    auto h_soln = soln.createPermuteVec(6, d_perm).createHostVec();
    printToVTK<Assembler,HostVec<T>>(kmg->grids[0].assembler, h_soln, "out/wing_mg_lin.vtk");
}

template <typename T, class Assembler>
void solve_direct(MPI_Comm &comm, int level, double SR) {

    /* direct NL solve used to check that how NL the problem is and how */

    using Basis = typename Assembler::Basis;
    using Physics = typename Assembler::Phys;
    using Data = typename Physics::Data;
    const SCALER scaler  = LINE_SEARCH;
    using Smoother = MulticolorGSSmoother_V1<Assembler>;
    const bool is_bsr = false; // no difference in intra-nodal (default old working prolong)
    using Prolongation = UnstructuredProlongation<Assembler, Basis, is_bsr>;
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

    TACSMeshLoader mesh_loader{comm};
    std::string fname = "../../multigrid/3_aob_wing/meshes/aob_wing_L" + std::to_string(level) + ".bdf";
    mesh_loader.scanBDFFile(fname.c_str());

    //   double E = 70e9, nu = 0.3, thick = 0.005;  // material & thick properties
    double E = 70e9, nu = 0.3, thick = 2.0 / SR;  // material & thick properties

    // make the assembler from the uCRM mesh
    auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));

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

    printf("making grid with level %d\n", level);

    // perform multicolor reordering
    auto &bsr_data = assembler.getBsrData();
    bsr_data.AMD_reordering();
    bsr_data.compute_full_LU_pattern(10.0, false);

    // T *_color_rowp = new T[2];
    auto h_color_rowp = HostVec<int>(2);
    assembler.moveBsrDataToDevice();

    // create the loads and kmat
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
    // int ELEM_MAX = 4;
    auto prolongation = new Prolongation(cusparseHandle, assembler, ELEM_MAX);
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
    printf("\nDirectLU-PCG on AOB-wing case with %d level and %.4e SR\n", level, SR);
    printf("\tinit resid %.4e => final resid %.4e in %.2e sec, log10(reduction)/sec = %.6e\n", init_resid, final_resid, solve_time.count(), log_red_rate);

    int ndof = assembler.get_num_vars();
    double total = startup_time.count() + solve_time.count();
    size_t bytes_per_double = sizeof(double);
    double mem_mb = static_cast<double>(bytes_per_double) * static_cast<double>(bsr_data.nnzb) * 36.0 / 1024.0 / 1024.0;
    printf("fullLU-memory in MB %.4e with NDOF %d\n", mem_mb, ndof);

    // // print to VTK (permuting from solve to vis order)
    int *d_perm = linear_solver->grid->d_perm;
    auto h_soln = lin_soln.createPermuteVec(6, d_perm).createHostVec();
    printToVTK<Assembler,HostVec<T>>(linear_solver->grid->assembler, h_soln, "out/wing_lin.vtk");
    // T lin_max_disp = get_max_disp(lin_soln);

    if (fail) {
        printf("\tPCG linear solver failed\n");
        return;
    }
}

template <typename T, class Smoother, class Assembler>
void gatekeeper_method(std::string smoother_type, MPI_Comm &comm, int level, double SR, int nsmooth, int ninnercyc, T omega, int ORDER) {
    if (smoother_type != "direct") {
        multigrid_solve<T, Smoother, Assembler>(comm, level, smoother_type, SR, nsmooth, ninnercyc, omega, ORDER);
    } else {
        solve_direct<T, Assembler>(comm, level, SR);
    }
}

int main(int argc, char **argv) {
    // input ----------
    int level = 1; // default value
    double SR = 50.0; // default
    int n_vcycles = 50;
    double omega = 0.3;
    int ORDER = 8; // for chebyshev

    // Intialize MPI and declare communicator
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int nsmooth = 1; // typically faster right now
    int ninnercyc = 1; // inner V-cycles to precond K-cycle

    // chebyshev, jacobi, gsmc, direct 
    std::string smoother_type = "chebyshev";

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        char* arg = argv[i];
        to_lowercase(arg);

        if (strcmp(arg, "--level") == 0) {
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
    constexpr bool is_nonlinear = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;

    // MITC4 shells
    using Basis = LagrangeQuadBasis<T, Quad, 1>;
    using Assembler = MITCShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;

    printf("AOB-wing level %d mesh with MITC4 elements, and SR %.2e\n------------\n", level, SR);
    if (smoother_type == "chebyshev" || smoother_type == "jacobi") {
        using Smoother = ChebyshevPolynomialSmoother<Assembler, false>;
        gatekeeper_method<T, Smoother, Assembler>(smoother_type, comm, level, SR, nsmooth, ninnercyc, omega, ORDER);
    } else if (smoother_type == "gsmc" || smoother_type == "direct") {
        using Smoother = MulticolorGSSmoother_V1<Assembler>; // still calls direct later if direct
        gatekeeper_method<T, Smoother, Assembler>(smoother_type, comm, level, SR, nsmooth, ninnercyc, omega, ORDER);
    }

    return 0;

    
}