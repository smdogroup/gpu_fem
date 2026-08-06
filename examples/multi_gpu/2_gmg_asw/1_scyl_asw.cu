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

// local multigrid imports
#include "multigrid/grid.h"
#include "multigrid/utils/fea.h"
#include "multigrid/prolongation/structured.h"
#include "multigrid/smoothers/asw_unstruct.h"
#include "multigrid/smoothers/mc_smooth1.h"
#include "multigrid/solvers/gmg.h"
#include <string>
#include <chrono>

// finalsolver
#include "multigrid/solvers/direct/cusp_directLU.h"
#include "multigrid/solvers/solve_utils.h"
#include "multigrid/solvers/krylov/bsr_pcg.h"
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


template <typename T, class Assembler, bool USE_CHEBYSHEV = false>
void asw_solve(int nxe, double SR, T omega, int n_smooth, int size, T pressure = 5.0e7) {
    /* SPAI-GMRES solve */

    using Basis = typename Assembler::Basis;
    using Physics = typename Assembler::Phys;
    const SCALER scaler  = LINE_SEARCH;
    using Smoother = UnstructuredQuadElementAdditiveSchwarzSmoother<T, Assembler>;
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


    if (nxe * nxe < 100) {
        printf("rhs after bcs\n");
        T *h_loads = loads.createHostVec().getPtr();
        printf("h_vec(nnodes=%d) single GPU\n", (int)N/6);
        for (int i = 0; i < (int)(N/6); i++) {
            T *h_block = &h_loads[6 * i];
            printf("singleGPU-node[%d]: ", i);
            printVec<T>(6, h_block);
        }
    }
    

    // assemble the kmat
    auto startkmat = std::chrono::high_resolution_clock::now();
    assembler.add_jacobian_fast(kmat);

    CHECK_CUDA(cudaDeviceSynchronize());
    auto endkmat = std::chrono::high_resolution_clock::now();

    // printout kmat for debugging purposes
    int mb = bsr_data.mb;
    int nb = bsr_data.nb;
    int nnzb = bsr_data.nnzb;
    int nnz = 36 * nnzb;
    int *h_rowp = DeviceVec<int>(mb + 1, bsr_data.rowp).createHostVec().getPtr();
    int *h_cols = DeviceVec<int>(nnzb, bsr_data.cols).createHostVec().getPtr();
    T *h_vals = DeviceVec<T>(nnz, kmat.getPtr()).createHostVec().getPtr();
    if (mb <= 100) {
        printf("Kmat before bcs on single GPU with nnz(%d) ------\n", nnz);
        for (int row = 0; row < mb; row++) {
            for (int jp = h_rowp[row]; jp < h_rowp[row + 1]; jp++) {
                int col = h_cols[jp];
                T *h_block = &h_vals[36 * jp];

                printf("block (%d,%d)\n", row, col);
                for (int i = 0; i < 6; i++) {
                    T *h_row = &h_block[6 * i];
                    printVec<T>(6, h_row);
                }
            }
        }
    }

    assembler.apply_bcs(kmat);
    CHECK_CUDA(cudaDeviceSynchronize());
    std::chrono::duration<double> assembly_time = endkmat - startkmat;
    printf("\tassemble kmat in %.3e sec\n", assembly_time.count());

    T *h_vals2 = DeviceVec<T>(nnz, kmat.getPtr()).createHostVec().getPtr();
    if (mb <= 100) {
        printf("Kmat after bcs on single GPU with nnz(%d) ------\n", nnz);
        for (int row = 0; row < mb; row++) {
            for (int jp = h_rowp[row]; jp < h_rowp[row + 1]; jp++) {
                int col = h_cols[jp];
                T *h_block = &h_vals2[36 * jp];

                printf("block (%d,%d)\n", row, col);
                for (int i = 0; i < 6; i++) {
                    T *h_row = &h_block[6 * i];
                    printVec<T>(6, h_row);
                }
            }
        }
    }

    // build smoother and prolongations..
    // auto smoother = new Smoother(cublasHandle, cusparseHandle, assembler, kmat, h_color_rowp, omegaMC, false, nsmooth);
    printf("making ASW smoother\n");
    // int size = 2; // size x size coupled blocks of smoothing
    auto smoother = new Smoother(cublasHandle, cusparseHandle, assembler, kmat, 
        omega, n_smooth);
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

    // TEST before solve
    // -----------------------------

    auto d_test_vec = assembler.createVarsVec();
    int nnodes = assembler.get_num_vars() / 6;
    if (nxe * nxe < 100) {
        linear_solver->test_mult(loads, d_test_vec);
        auto h_test_vec = d_test_vec.createHostVec();
        T *h_test_ptr = h_test_vec.getPtr();
        printf("test mat-vec\n");
        for (int i = 0; i < nnodes; i++) {
            T *h_block = &h_test_ptr[6 * i];
            printf("test node[%d]: ", i);
            printVec<T>(6, h_block);
        }
    }

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

    pc->factor(); // ASW factor time

    if (nxe * nxe < 100) {
        d_test_vec.zeroValues();
        linear_solver->test_precond(loads, d_test_vec);
        auto h_test_vec2 = d_test_vec.createHostVec();
        T *h_test_ptr2 = h_test_vec2.getPtr();
        printf("test precond-vec\n");
        for (int i = 0; i < nnodes; i++) {
            T *h_block = &h_test_ptr2[6 * i];
            printf("test node[%d]: ", i);
            printVec<T>(6, h_block);
        }

    }
    
    // get initial residual
    T init_resid = linear_solver->getResidualNorm(grid->d_defect, lin_soln);

    // linear solve
    bool fail = linear_solver->solve(grid->d_defect, lin_soln, true);
    // bool fail = smoother->solve(grid->d_defect, lin_soln); // just preconditioner solve
    
    // final residual
    T final_resid = linear_solver->getResidualNorm(grid->d_defect, lin_soln);

    CHECK_CUDA(cudaDeviceSynchronize());
    auto end_solve = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = end_solve - start_solve;

    // compute log residual reduction per unit time
    T log_red_rate = (log(init_resid) - log(final_resid)) / log(10.0) / solve_time.count();
    printf("\nASW-GMRES on cylinder case with %d nxe and %.4e SR\n", nxe, SR);
    printf("\tinit resid %.4e => final resid %.4e in %.3e sec, log10(reduction)/sec = %.6e\n", init_resid, final_resid, solve_time.count(), log_red_rate);

    // // print to VTK (permuting from solve to vis order)
    int *d_perm = linear_solver->grid->d_perm;
    auto h_soln = lin_soln.createPermuteVec(6, d_perm).createHostVec();
    printToVTK<Assembler,HostVec<T>>(linear_solver->grid->assembler, h_soln, "./out/plate_kry_lin_single.vtk");
    T lin_max_disp = get_max_disp(lin_soln);

    int nx = nxe + 1;
    int ndof = nx * nx * 6;
    double total = startup_time.count() + solve_time.count();
    size_t bytes_per_double = sizeof(double);
    double mem_mb = static_cast<double>(bytes_per_double) * static_cast<double>(bsr_data.nnzb) * 36.0 / 1024.0 / 1024.0;
    printf("ASW-PCG memory in MB %.4e with NDOF %d\n", mem_mb, ndof);

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
            << "ASW" << "," << pc_compl << "," << solve_time.count() << "\n";
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

    pc->factor(); // run factor again so fair comparison

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
    T log_red_rate =  log_resid_drop / solve_time.count(); // 0.5 * 
    printf("\nDirectLU-PCG on cylinder case with %d nxe and %.4e SR\n", nxe, SR);
    printf("\tinit resid %.4e => final resid %.4e in %.3e sec, log10(reduction)/sec = %.6e\n", init_resid, final_resid, solve_time.count(), log_red_rate);

    int nx = nxe + 1;
    int ndof = nx * nx * 6;
    double total = startup_time.count() + solve_time.count();
    size_t bytes_per_double = sizeof(double);
    double mem_mb = static_cast<double>(bytes_per_double) * static_cast<double>(bsr_data.nnzb) * 36.0 / 1024.0 / 1024.0;
    printf("fullLU-memory in MB %.4e with NDOF %d\n", mem_mb, ndof);

    // // print to VTK (permuting from solve to vis order)
    int *d_perm = linear_solver->grid->d_perm;
    auto h_soln = lin_soln.createPermuteVec(6, d_perm).createHostVec();
    printToVTK<Assembler,HostVec<T>>(linear_solver->grid->assembler, h_soln, "out/plate_kry_lin.vtk");
    T lin_max_disp = get_max_disp(lin_soln);

    if (!fail) {
        T pc_compl = pc->precond_complexity(kmat_orig_nnzb);

        // write to csv (this particular run)
        // ---------------------------------------
        std::ofstream csv("./out/cylinder-times.csv", std::ios::app);
        if (csv.tellp() == 0)
            csv << "t/R,nxe,NDOF,solver,pc_complexity,lin_runtime(s)\n";
        // Set high precision for CSV output
        csv << std::setprecision(15) << std::scientific;
        csv << (1.0/SR) << "," << nxe << "," << N << ","
            << "LU" << "," << pc_compl << "," << solve_time.count() << "\n";
    }

    if (fail) {
        printf("\tPCG linear solver failed\n");
        return;
    }
}

template <typename T, class Assembler>
void gatekeeper_method(std::string solver_type, int nxe, double SR, int nsmooth, T omega, int ORDER, T qorder, T load_mag = 5.0e7) {
    if (solver_type == "direct") {
        solve_direct<T, Assembler>(nxe, SR, load_mag);
    } else if (solver_type == "asw2") {
        printf("ASW : setting omega = 0.2 and size to 2, comment this out to change it\n");
        int size = 2; // recommend omega = 0.2 here, nsmooth = 2
        asw_solve<T, Assembler, false>(nxe, SR, omega, nsmooth, size, load_mag);
    }
}

int main(int argc, char **argv) {
    // input ----------
    std::string solver_type = "asw2";
    int nxe = 128; // default value (three grids)
    double SR = 10.0; // default, the less slender it is, solves much faster
    double pressure = 8.0e6;
    double omega = 0.2; // default omega
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
