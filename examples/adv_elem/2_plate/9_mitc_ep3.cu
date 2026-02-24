// """
// plate geometric multigrid with MITC4 shells
// * uses MITC-EP (energy-smooth prolongation)
// * 3x3 node support-ASW (additive schwarz subdomain smoother)
// * default is K-cycle with V-cycle precond (PCG solver), other options V-cycle, K-cycle and F-cycle solver
// """

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
#include "multigrid/smoothers/cheb4_poly.h"
#include "multigrid/smoothers/asw_support.h"
#include "multigrid/prolongation/unstruct_smooth.h"
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
    ./1_plate.out direct --nxe 2048 --SR 100.0    to run direct cylinder solve on 2048 x 2048 elem grid with slenderness ratio 100
    ./1_plate.out mg --nxe 2048 --SR 100.0    to run geometric multigrid cylinder solve on 2048 x 2048 elem grid with slenderness ratio 100
*/

void to_lowercase(char *str) {
    for (; *str; ++str) {
        *str = std::tolower(*str);
    }
}

template <typename T, class Assembler>
void multigrid_solve(int nxe, double SR, int nsmooth, int ninnercyc, int nsmooth_mat, T omega, std::string cycle_type) {
    // geometric multigrid method here..
    // need to make a number of grids..

    using Basis = typename Assembler::Basis;
    using Physics = typename Assembler::Phys;
    const SCALER scaler  = LINE_SEARCH;
    using ProlongSmoother = ChebyshevPolynomialSmoother<Assembler>;
    using Smoother = UnstructuredQuadSupportAdditiveSchwarzSmoother<T, Assembler>;
    using Prolongation = UnstructuredSmoothProlongation<Assembler, Basis, ProlongSmoother>;
    using GRID = SingleGrid<Assembler, Prolongation, Smoother, scaler>;
    using CoarseSolver = CusparseMGDirectLU<T, Assembler>;
    using MG = GeometricMultigridSolver<GRID, CoarseSolver>;
    
    // for K-cycles
    using KrylovSolve = PCGSolver<T, GRID>;
    using TwoLevelSolve = MultigridTwoLevelSolver<GRID>;
    using KMG = MultilevelKcycleSolver<GRID, CoarseSolver, TwoLevelSolve, KrylovSolve>;

    auto start0 = std::chrono::high_resolution_clock::now();

    MG *mg;
    KMG *kmg;

    // T omegaLS_min = 0.25, omegaLS_max = 2.0;
    T omegaLS_min = 1e-2, omegaLS_max = 4.0;

    cublasHandle_t cublasHandle = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));
    cusparseHandle_t cusparseHandle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

    bool is_kcycle = cycle_type == "K";
    if (is_kcycle) {
        kmg = new KMG();
    } else {
        mg = new MG();
    }

    // int pre_nxe_min = nxe > 32 ? 32 : 8;
    // int pre_nxe_min = 2;
    int pre_nxe_min = nxe > 32 ? 32 : (nxe / 2);
    int nxe_min = pre_nxe_min;
    for (int c_nxe = nxe; c_nxe >= pre_nxe_min; c_nxe /= 2) {
        nxe_min = c_nxe;
    }

    // make each grid
    for (int c_nxe = nxe; c_nxe >= nxe_min; c_nxe /= 2) {
        // make the assembler
        int c_nye = c_nxe;
        double Lx = 1.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 1.0 / SR, rho = 2500, ys = 350e6;
        int nxe_per_comp = c_nxe, nye_per_comp = c_nye; // for now (should have 25 grids)
        auto assembler = createPlateAssembler<Assembler>(c_nxe, c_nye, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp);
        double Q = 1.0; // load magnitude
        T *my_loads = getPlateLoads<T, Basis, Physics>(c_nxe, c_nye, Lx, Ly, Q);
        printf("making grid with nxe %d => ", c_nxe);

        auto &bsr_data = assembler.getBsrData();
        int num_colors, *_color_rowp;

        // make the grid
        bool full_LU = c_nxe == nxe_min;
        if (full_LU) {
            bsr_data.AMD_reordering();
            bsr_data.compute_full_LU_pattern(10.0, false);
        } else {
            bsr_data.compute_nofill_pattern();
        }

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
        printf("assemble kmat in %.2e sec\n", assembly_time.count());

        // build smoother and prolongations..
        // printf("nsmooth %d, omega = %.4e\n", nsmooth, omega);
        auto smoother = new Smoother(cublasHandle, cusparseHandle, assembler, kmat, 
            omega, nsmooth);
        int ORDER = 1; // equiv to Jacobi-prolong smoother
        T omega_p = 0.9;
        auto prolong_smoother = new ProlongSmoother(cublasHandle, cusparseHandle, assembler, kmat, omega_p, ORDER);
        int ELEM_MAX = 10;
        auto prolongation = new Prolongation(cusparseHandle, assembler, prolong_smoother, ELEM_MAX, nsmooth_mat);
        auto grid = GRID(assembler, prolongation, smoother, kmat, loads, cublasHandle, cusparseHandle, omegaLS_min, omegaLS_max);
        
        // 10 iterations of CG lanczos
        prolong_smoother->setup_cg_lanczos(grid.d_defect, 10);

        if (is_kcycle) {
            kmg->grids.push_back(grid);
        } else {
            mg->grids.push_back(grid);
            if (full_LU) mg->coarse_solver = new CoarseSolver(cublasHandle, cusparseHandle, assembler, kmat);
        }
    }

    assert(is_kcycle); // for first try..
    auto &grids = kmg->grids;
    kmg->template init_prolongations<Basis>();

    // ===========================================
    // end of locking aware prolongation
    // ===========================================

    auto end0 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> startup_time = end0 - start0;

    int pre_smooth = nsmooth, post_smooth = nsmooth; // need a little extra smoothing on cylinder (compare to plate).. (cause of curvature I think..)
    bool print = true;
    // bool print = false;
    T atol = 1e-10, rtol = 1e-6;
    bool double_smooth = false;
    // bool double_smooth = true; // twice as many smoothing steps at lower levels (similar cost, better conv?)

    int n_cycles = 500; // max # cycles
    int print_freq = 3;

    if (is_kcycle) {
        int n_krylov = 500;
        kmg->init_outer_solver(cublasHandle, cusparseHandle, nsmooth, ninnercyc, n_krylov, omega, atol, rtol, print_freq, print, double_smooth);    
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    auto start1 = std::chrono::high_resolution_clock::now();

    if (is_kcycle) {
        int nlevels = kmg->grids.size();
        for (int i = 0; i < nlevels; i++) {
            kmg->grids[i].smoother->factor();
        } 
    } else {
        int nlevels = mg->grids.size();
        for (int i = 0; i < nlevels; i++) {
            mg->grids[i].smoother->factor();
        } 
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end_factor = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> asw_factor_time = end_factor - start1;
    printf("ASW factor time %.4e\n", asw_factor_time.count());

    // fastest is K-cycle usually
    if (cycle_type == "V") {
        mg->vcycle_solve(0, pre_smooth, post_smooth, n_cycles, print, atol, rtol, double_smooth, print_freq); //(good option)
    } else if (cycle_type == "W") {
        mg->wcycle_solve(0, pre_smooth, post_smooth, n_cycles, print, atol, rtol);
    } else if (cycle_type == "F") {
        mg->fcycle_solve(0, pre_smooth, post_smooth, n_cycles, print, atol, rtol, double_smooth, print_freq); // also decent
    } else if (cycle_type == "K") {
        kmg->solve(); // best
    }


    CHECK_CUDA(cudaDeviceSynchronize());
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = end1 - start1;
    int ndof = cycle_type == "K" ? kmg->grids[0].N : mg->grids[0].N;
    double total = startup_time.count() + solve_time.count();
    double mem_MB = is_kcycle ? kmg->get_memory_usage_mb() : mg->get_memory_usage_mb();
    printf("plate GMG solve, ndof %d : startup time %.2e, solve time %.2e, total %.2e, with mem(MB) %.2e\n", ndof, startup_time.count(), solve_time.count(), total, mem_MB);

    if (is_kcycle) {
        // print some of the data of host residual
        int *d_perm = kmg->grids[0].d_perm;
        auto h_soln = kmg->grids[0].d_soln.createPermuteVec(6, d_perm).createHostVec();
        printToVTK<Assembler,HostVec<T>>(kmg->grids[0].assembler, h_soln, "out/plate_mg.vtk");
    } else {
        // print some of the data of host residual
        int *d_perm = mg->grids[0].d_perm;
        auto h_soln = mg->grids[0].d_soln.createPermuteVec(6, d_perm).createHostVec();
        printToVTK<Assembler,HostVec<T>>(mg->grids[0].assembler, h_soln, "out/plate_mg.vtk");
    }
}

template <typename T, class Assembler>
void direct_solve(int nxe, double SR) {
    using Basis = typename Assembler::Basis;
    using Physics = typename Assembler::Phys;

    int c_nxe = nxe;
    int c_nye = c_nxe;
    double Lx = 1.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 1.0 / SR, rho = 2500, ys = 350e6;
    int nxe_per_comp = c_nxe / 4, nye_per_comp = c_nye/4; // for now (should have 25 grids)
    auto assembler = createPlateAssembler<Assembler>(c_nxe, c_nye, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp);
    double Q = 1.0; // load magnitude
    T *my_loads = getPlateLoads<T, Basis, Physics>(c_nxe, c_nye, Lx, Ly, Q);
    printf("making grid with nxe %d\n", c_nxe);

    // BSR symbolic factorization
    // must pass by ref to not corrupt pointers
    auto& bsr_data = assembler.getBsrData();
    double fillin = 10.0;  // 10.0
    bool print = true;
    bsr_data.AMD_reordering();
    bsr_data.compute_full_LU_pattern(fillin, print);
    assembler.moveBsrDataToDevice();

    auto loads = assembler.createVarsVec(my_loads);
    assembler.apply_bcs(loads);

    // setup kmat and initial vecs
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto soln = assembler.createVarsVec();
    auto res = assembler.createVarsVec();
    auto vars = assembler.createVarsVec();

    // assemble the kmat
    assembler.add_jacobian_fast(kmat);
    assembler.add_residual_fast(res);
    // assembler.add_jacobian(res, kmat);
    assembler.apply_bcs(res);
    assembler.apply_bcs(kmat);

    CHECK_CUDA(cudaDeviceSynchronize());
    auto start1 = std::chrono::high_resolution_clock::now();

    // solve the linear system
    CUSPARSE::direct_LU_solve(kmat, loads, soln);

    CHECK_CUDA(cudaDeviceSynchronize());
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = end1 - start1;

    size_t bytes_per_double = sizeof(double);
    double mem_mb = static_cast<double>(bytes_per_double) * static_cast<double>(bsr_data.nnzb) * 36.0 / 1024.0 / 1024.0;
    int ndof = assembler.get_num_vars();
    printf("plate direct solve, ndof %d : solve time %.2e, with mem (MB) %.2e\n", ndof, solve_time.count(), mem_mb);

    // print some of the data of host residual
    auto h_soln = soln.createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_soln, "out/plate.vtk");
}

template <typename T, class Assembler>
void gatekeeper_method(bool is_multigrid, int nxe, double SR, int nsmooth, int ninnercyc, int nsmooth_mat, T omega, std::string cycle_type) {
    if (is_multigrid) {
        multigrid_solve<T, Assembler>(nxe, SR, nsmooth, ninnercyc, nsmooth_mat, omega, cycle_type);
    } else {
        direct_solve<T, Assembler>(nxe, SR);
    }
}

int main(int argc, char **argv) {
    // input ----------
    bool is_multigrid = true;
    // int nxe = 256; // default value
    int nxe = 256; // for comparison with python GMG
    // int nxe = 4;
    double SR = 1e3; // default
    double omega = 0.1; // smaller omega for ASW

    int nsmooth = 2; // typically faster right now
    int ninnercyc = 1;
    int nsmooth_mat = 2; // more iterations not converging yet
    // int ninnercyc = 2; // inner V-cycles to precond K-cycle (ends up being a bit faster here..)
    std::string cycle_type = "K"; // "V", "F", "W", "K"
    // std::string cycle_type = "V"; // "V", "F", "W", "K"

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
        } else if (strcmp(arg, "--nsmooth_mat") == 0) {
            if (i + 1 < argc) {
                nsmooth_mat = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --nsmooth_mat\n";
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

    printf("plate mesh with MITC4-EP elements, nxe %d and SR %.2e\n------------\n", nxe, SR);
    using Basis = LagrangeQuadBasis<T, Quad, 1>;
    using Assembler = MITCShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
    gatekeeper_method<T, Assembler>(is_multigrid, nxe, SR, nsmooth, ninnercyc, nsmooth_mat, omega, cycle_type);
    

    return 0;

    
}
