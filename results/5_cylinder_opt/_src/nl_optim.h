#include <iostream>
#include <sstream>

#include "chrono"
#include "coupled/_coupled.h"
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"

// shell imports
#include "assembler.h"
#include "element/shell/basis/lagrange_basis.h"
#include "element/shell/director/linear_rotation.h"
#include "element/shell/mitc_shell.h"
#include "element/shell/physics/isotropic_shell.h"

// new nonlinear solvers
#include "solvers/nonlinear_static/continuation.h"
#include "solvers/nonlinear_static/inexact_newton.h"
#include "solvers/nonlinear_static/nl_interface.h"

// multigrid imports
#include "multigrid/grid.h"
#include "multigrid/utils/fea.h"
// #include "multigrid/smoothers/mc_smooth1.h"
#include "multigrid/prolongation/structured.h"
#include "multigrid/smoothers/cheb4_poly.h"
// #include "multigrid/solvers/gmg.h"

// new multigrid imports for K-cycles, etc.
#include "multigrid/interface.h"
#include "multigrid/solvers/direct/cusp_directLU.h"
#include "multigrid/solvers/krylov/bsr_pcg.h"
#include "multigrid/solvers/multilevel/kcycle.h"
#include "multigrid/solvers/multilevel/twolevel.h"
#include "multigrid/solvers/solve_utils.h"

// copied and modified from ../uCRM/_src/optim.h (uCRM optimization example)

class NonlinearCylinderSolver {
   public:
    using T = double;
    // FEM typedefs
    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = LagrangeQuadBasis<T, Quad, 1>;
    using Data = ShellIsotropicData<T, false>;
    using Physics = IsotropicShell<T, Data, false>;
    using Assembler = MITCShellAssembler<T, Director, Basis, Physics, DeviceVec, BsrMat>;
    using CoarseSolver = CusparseMGDirectLU<T, Assembler>;

    // multigrid objects
    using Smoother = ChebyshevPolynomialSmoother<Assembler>;
    using Prolongation = StructuredProlongation<Assembler, CYLINDER>;
    using GRID = SingleGrid<Assembler, Prolongation, Smoother, LINE_SEARCH>;
    // using MG = GeometricMultigridSolver<GRID>; // old V-cycle solver

    // for K-cycles
    using KrylovSolve = PCGSolver<T, GRID>;
    using TwoLevelSolve = MultigridTwoLevelSolver<GRID>;
    using KMG = MultilevelKcycleSolver<GRID, CoarseSolver, TwoLevelSolve, KrylovSolve>;
    // using StructSolver = TacsMGInterface<T, Assembler, KMG>;

    // build the inexact newton + outer continuation solver
    using Mat = BsrMat<DeviceVec<T>>;
    using Vec = DeviceVec<T>;
    using INK = InexactNewtonSolver<T, Mat, Vec, Assembler, KMG>;
    using NL = NonlinearContinuationSolver<T, Vec, Assembler, INK>;

    using StructSolver = TACSNLInterface<T, Assembler, KMG, NL>;

    // functions
    using DMass = Mass<T, DeviceVec>;
    using DKSFail = KSFailure<T, DeviceVec>;

    NonlinearCylinderSolver(double rhoKS = 100.0, double safety_factor = 1.5,
                            double load_mag = 100.0, T omega = 1.0, int nxe = 100, int nx_comp = 5,
                            int ny_comp = 5, double SR = 50.0, int ORDER = 8, 
                            int nsmooth = 1, int ninnercyc = 1, double in_plane_frac = 0.1,
                            bool print = false) {
        // 1) Build mesh & assembler
        assert(nxe % nx_comp == 0);  // evenly divisible by number of elems_per_comp
        int nye = nxe;
        assert(nye % ny_comp == 0);
        num_lin_solves = 0;  // set num lin solves to 0

        CHECK_CUBLAS(cublasCreate(&cublasHandle));
        CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

        // start building multigrid object
        mg = new KMG();

        // get nxe_min for not exactly power of 2 case
        int pre_pre_nxe_min = max(32, nx_comp);
        int pre_nxe_min = nxe > pre_pre_nxe_min ? pre_pre_nxe_min : 4;
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
            double rho = 2500, ys = 350e6;
            bool imperfection = false;    // option for geom imperfection
            int imp_x = 1, imp_hoop = 1;  // no imperfection this input doesn't matter rn..
            auto assembler =
                createCylinderAssembler<Assembler>(c_nxe, c_nhe, L, R, E, nu, thick, imperfection,
                                                   imp_x, imp_hoop, rho, ys, nx_comp, ny_comp);
            constexpr bool compressive = false;
            constexpr int load_case = 3;  // petal and chirp load
            // double nodal_loads = uniform_force; // (don't normalize anymore, integrated out) /
            // (nxe - 1) / (nxe - 1); T *my_loads = getCylinderLoads<T,  Basis, Physics,
            // load_case>(c_nxe, c_nhe, L, R, pressure);
            T *my_loads =
                getCylinderLoadsRobust<T, Assembler>(assembler, c_nxe, c_nhe, L, R, load_mag, in_plane_frac);
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
            assembler.add_jacobian(res, kmat);
            // assembler.apply_bcs(res);
            assembler.apply_bcs(kmat);
            CHECK_CUDA(cudaDeviceSynchronize());
            auto end0 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> assembly_time = end0 - start0;
            printf("\tassemble kmat time %.2e\n", assembly_time.count());

            // build smoother and prolongations..
            auto smoother =
                new Smoother(cublasHandle, cusparseHandle, assembler, kmat, omega, ORDER);
            auto prolongation = new Prolongation(assembler);
            T omegaLS_min = 0.01, omegaLS_max = 4.0;
            auto grid = GRID(assembler, prolongation, smoother, kmat, loads, cublasHandle,
                             cusparseHandle, omegaLS_min, omegaLS_max);

            smoother->setup_cg_lanczos(grid.d_defect);  // CG-Lanczos

            mg->grids.push_back(grid);
            if (full_LU)
                mg->coarse_solver = new CoarseSolver(cublasHandle, cusparseHandle, assembler, kmat);
        }

        mg->template init_prolongations<Basis>();
        // end of startup

        // int n_cycles = 200, pre_smooth = 1, post_smooth = 1, print_freq = 3;
        // bool print = true;
        // bool print = false;
        bool double_smooth = true;
        // int nsmooth = 1, ninnercyc = 1, 
        int print_freq = 3;
        int n_krylov = 50;
        T atol = 1e-6, rtol = 1e-6;
        // bool double_smooth = false;  // actually faster sometimes

        // mg->init_outer_solver(nsmooth, ninnercyc, n_krylov, omega, atol, rtol, print_freq,
        // print);
        mg->init_outer_solver(cublasHandle, cusparseHandle, nsmooth, ninnercyc, n_krylov, omega,
                              atol, rtol, print_freq, print, double_smooth);

        // get struct loads on finest grid
        auto fine_grid = mg->grids[0];
        d_loads = DeviceVec<T>(fine_grid.N);
        mg->grids[0].getDefect(d_loads);

        // initialize any vecs needed at this level
        auto &assembler = mg->grids[0].assembler;
        nvars = assembler.get_num_vars();
        int nn = assembler.get_num_nodes();
        soln = DeviceVec<T>(nvars);
        ndvs = assembler.get_num_dvs();
        d_dvs = DeviceVec<T>(ndvs, /*initial=*/0.02);

        // fine grid, create the nonlinear solvers and NL solver interface
        // -------------------------
        T initLinSolveRtol = 1e-2;
        // T initLinSolveRtol = 1e-1; // which makes it run faster?
        T linSolveAtol = 1e-4;
        // T restart_dlam = 1e-2; // default
        T restart_dlam = 0.05; // tolerance for just trying to newton solve immediately to lam = 1.0

        inner_solver = new INK(cublasHandle, assembler, mg->grids[0].Kmat, d_loads, mg, initLinSolveRtol, linSolveAtol, 1e-4, 0.25, restart_dlam);
        bool use_predictor = true, debug = false;
        // bool use_predictor = false, debug = false;
        nl_solver = new NL(cublasHandle, assembler, inner_solver, use_predictor, debug);
        solver = new StructSolver(cublasHandle, nl_solver, assembler, mg, print);

        // 5) Functions
        mass = std::make_unique<DMass>();
        ksfail = std::make_unique<DKSFail>(rhoKS, safety_factor);

        dvs_changed = true;
        first_solve = true;
    }

    void set_design_variables(const std::vector<T> &dvs) {
        /* check if dvs changed before running new analysis (make sure this works right) */
        dvs_changed = (dvs.size() != prev_dvs.size());
        if (!dvs_changed) {
            for (int i = 0; i < dvs.size(); i++) {
                if (dvs[i] != prev_dvs[i]) {
                    dvs_changed = true;
                    break;
                }
            }
        }

        // if (first_solve) {clear
        //     dvs_changed = true;
        //     first_solve = false;
        // }
        dvs_changed = true;  // debug

        if (dvs_changed) {
            prev_dvs = dvs;
            CHECK_CUDA(
                cudaMemcpy(d_dvs.getPtr(), dvs.data(), ndvs * sizeof(T), cudaMemcpyHostToDevice));
            solver->set_design_variables(d_dvs);
        }
    }

    int get_num_vars() const { return nvars; }
    int get_num_dvs() const { return ndvs; }
    void writeSolution(const std::string &filename) const { solver->writeSoln(filename); }

    void solve() {
        if (dvs_changed) {
            printf("design changed, new solve\n");

            solver->solve();
            solver->copy_solution_out(soln);
        } else {
            // reload old state
            printf("design didn't change, reload vals\n");
            solver->copy_solution_in(soln);
        }
    }

    T evalFunction(const std::string &name) {
        if (name == "mass")
            return solver->evalFunction(*mass);
        else if (name == "ksfailure")
            return solver->evalFunction(*ksfail);
        throw std::invalid_argument("Unknown func");
    }

    // fills host array of length ndvs
    void evalFunctionSens(const std::string &name, T *out_h_sens) {
        double *dptr;
        if (name == "mass") {
            solver->solve_adjoint(*mass);  // mass is adjoint free, so no lin solve dont here..
            dptr = mass->dv_sens.getPtr();
        } else if (name == "ksfailure") {
            solver->solve_adjoint(*ksfail);
            num_lin_solves++;
            dptr = ksfail->dv_sens.getPtr();
        } else {
            throw std::invalid_argument("Unknown func");
        }
        CHECK_CUDA(cudaMemcpy(out_h_sens, dptr, ndvs * sizeof(T), cudaMemcpyDeviceToHost));
    }
    int get_num_lin_solves() { return mg->get_num_lin_solves(); }

    void free() {
        solver->free();
        // assembler->free();
        d_loads.free();
        d_dvs.free();
        // tear down MPI if *we* initialized it
        int mpi_inited = 0;
        MPI_Finalized(&mpi_inited);
        if (!mpi_inited) {
            MPI_Finalize();
        }
    }

   private:
    // std::unique_ptr<Assembler> assembler;
    // std::unique_ptr<StructSolver> solver;
    StructSolver *solver;
    std::unique_ptr<DMass> mass;
    std::unique_ptr<DKSFail> ksfail;

    cublasHandle_t cublasHandle = NULL;
    cusparseHandle_t cusparseHandle = NULL;

    INK *inner_solver;
    NL *nl_solver;

    KMG *mg;  // multigrid object

    int num_lin_solves = 0;

    int ndvs = 0, nvars = 0;
    DeviceVec<T> d_loads, d_dvs;
    std::vector<T> prev_dvs;
    bool dvs_changed;
    bool first_solve;
    DeviceVec<T> soln;
};