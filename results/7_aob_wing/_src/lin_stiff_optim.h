#pragma once

#include <iostream>
#include <sstream>

#include "chrono"
#include "linalg/_linalg.h"
#include "mesh/TACSMeshLoader.h"
// #include "coupled/_coupled.h"
#include "solvers/_solvers.h"

// shell imports
#include "assembler.h"
#include "element/shell/basis/lagrange_basis.h"
#include "element/shell/director/linear_rotation.h"
#include "element/shell/mitc_shell.h"
// #include "element/shell/physics/isotropic_shell.h"
#include "element/shell/physics/iso_stiff_shell.h"

// multigrid imports
#include "multigrid/grid.h"
#include "multigrid/utils/fea.h"
// #include "multigrid/smoothers/mc_smooth1.h"
// #include "multigrid/prolongation/structured.h"
#include "multigrid/prolongation/unstructured.h"
#include "multigrid/smoothers/_wingbox_coloring.h"
#include "multigrid/smoothers/cheb4_poly.h"
// #include "multigrid/solvers/gmg.h"

// case imports
#include "comp_reader.h"
#include "loads_util.h"

// new multigrid imports for K-cycles, etc.
#include "multigrid/interface.h"
#include "multigrid/solvers/direct/cusp_directLU.h"
#include "multigrid/solvers/krylov/bsr_pcg.h"
#include "multigrid/solvers/multilevel/kcycle.h"
#include "multigrid/solvers/multilevel/twolevel.h"
#include "multigrid/solvers/solve_utils.h"

// copied and modified from ../uCRM/_src/optim.h (uCRM optimization example)

class LinearStiffenedWingSolver {
   public:
    using T = double;
    // FEM typedefs
    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    // static constexpr bool has_ref_axis = false;
    static const bool has_ref_axis =
        true;  // only need ref axis for stiffened wing  with buckling loads
    using Data = StiffenedIsotropicShellData<T, has_ref_axis>;
    static const bool is_nonlinear = false;  // cause linear wing
    using Physics = StiffenedIsotropicShell<T, Data, is_nonlinear>;

    // element type
    // MITC4 shell
    // using Basis = LagrangeQuadBasis<T, Quad, 1>;
    // using Assembler = MITCShellAssembler<T, Director, Basis, Physics, DeviceVec, BsrMat>;

    // CFI4 shell (be careful it can lock though)
    using Basis = ChebyshevQuadBasis<T, Quad, 1>;
    using Assembler = FullyIntegratedShellAssembler<T, Director, Basis, Physics, DeviceVec, BsrMat>;

    // multigrid objects
    using CoarseSolver = CusparseMGDirectLU<T, Assembler>;
    using Smoother = ChebyshevPolynomialSmoother<Assembler>;
    static const bool is_bsr = false;  // no difference in intra-nodal (default old working prolong)
    using Prolongation = UnstructuredProlongation<Assembler, Basis, is_bsr>;
    using GRID = SingleGrid<Assembler, Prolongation, Smoother, LINE_SEARCH>;
    // using MG = GeometricMultigridSolver<GRID>; // old V-cycle solver

    // for K-cycles
    using KrylovSolve = PCGSolver<T, GRID>;
    using TwoLevelSolve = MultigridTwoLevelSolver<GRID>;
    using KMG = MultilevelKcycleSolver<GRID, CoarseSolver, TwoLevelSolve, KrylovSolve>;
    using StructSolver = TacsMGInterface<T, Assembler, KMG>;

    // functions
    using DMass = Mass<T, DeviceVec>;
    using DKSFail = KSFailure<T, DeviceVec>;

    LinearStiffenedWingSolver(double rhoKS = 100.0, double safety_factor = 1.5,
                              double force = 684e3, T omega = 1.0, int level = 2, T rtol = 1e-6,
                              int ORDER = 8, int nsmooth = 1, int ninnercyc = 1,
                              bool print = false) {
        // --- SAFE MPI INIT ---
        int already_init = 0;
        MPI_Initialized(&already_init);
        if (!already_init) {
            // init only if Python / mpirun / mpi4py hasn’t done so
            MPI_Init(NULL, NULL);
            // (Optionally: store that YOU initialized MPI if you later want to call MPI_Finalize)
        }

        MPI_Comm comm = MPI_COMM_WORLD;

        // 1) Build mesh & assembler
        num_lin_solves = 0;  // set num lin solves to 0
        CHECK_CUBLAS(cublasCreate(&cublasHandle));
        CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

        double SR =
            20.0;  // default slenderness (to modify thicknesses, change them in the DVs instead)

        // start building multigrid object
        mg = new KMG();

        // make each grid
        for (int i = level; i >= 0; i--) {
            // read the ESP/CAPS => nastran mesh for TACS
            TACSMeshLoader mesh_loader{comm};
            std::string fname = "../../examples/multigrid/3_aob_wing/meshes/aob_wing_L" +
                                std::to_string(i) + ".bdf";
            mesh_loader.scanBDFFile(fname.c_str());

            // IF STIFFENED WING with REF AXIS:
            // ===============================================

            HostVec<Data> comp_data(mesh_loader.getNumComponents());
            std::string design_filename = "design/AOB-design.txt";
            build_AOB_component_data<T, Data>(mesh_loader, comp_data, design_filename);
            printf("making assembler+GMG for mesh '%s'\n", fname.c_str());
            // create the TACS Assembler from the mesh loader
            auto assembler = Assembler::createFromBDFComponent(mesh_loader, comp_data);
            printf("\tdone making assembler\n");

            // IF UNSTIFFENED WING without ref axis, isotropic (no buckling)
            // =================================================
            // double E = 70e9, nu = 0.3, thick = 2.0 / SR;
            // double rho = 2500, ys = 350e6;
            // printf("making assembler+GMG for mesh '%s'\n", fname.c_str());
            // auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick, rho, ys));

            // apply lower skin press loads
            int nvars = assembler.get_num_vars();
            int nnodes = assembler.get_num_nodes();
            T *wing_loads;
            addSkinLoadsToWing<double, Basis, Assembler>(assembler, wing_loads, force);

            // do multicolor junction reordering
            printf("perform coloring\n");
            auto &bsr_data = assembler.getBsrData();
            int num_colors, *_color_rowp;

            bool coarsest_grid = i == 0;
            if (!coarsest_grid) {
                WingboxMultiColoring<Assembler>::apply_coloring(assembler, bsr_data, num_colors,
                                                                _color_rowp);
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
            auto loads = assembler.createVarsVec(wing_loads);
            assembler.apply_bcs(loads);
            auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
            auto vars = assembler.createVarsVec();
            assembler.set_variables(vars);
            auto res = assembler.createVarsVec();
            auto starta = std::chrono::high_resolution_clock::now();
            const int elems_per_blockk =
                1;  // 1 versus 2 elements => similar runtime (1 slightly better)
            assembler.template add_jacobian_fast<elems_per_blockk>(kmat);
            assembler.apply_bcs(kmat);
            CHECK_CUDA(cudaDeviceSynchronize());
            auto enda = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> assembly_time = enda - starta;
            printf("\tassemble kmat in %.2e sec\n", assembly_time.count());

            // build smoother and prolongations
            // // bool smooth_debug = true;
            // bool smooth_debug = false;
            auto smoother =
                new Smoother(cublasHandle, cusparseHandle, assembler, kmat, omega, ORDER);
            int ELEM_MAX = 10;  // num nearby elements of each fine node for nz pattern construction
            // int ELEM_MAX = 4;
            auto prolongation = new Prolongation(cusparseHandle, assembler, ELEM_MAX);
            // T omegaLS_min = 0.01, omegaLS_max = 4.0;
            T omegaLS_min = 0.1, omegaLS_max = 2.0;
            auto grid = GRID(assembler, prolongation, smoother, kmat, loads, cublasHandle,
                             cusparseHandle, omegaLS_min, omegaLS_max);

            smoother->setup_cg_lanczos(grid.d_defect);

            mg->grids.push_back(grid);
        }

        mg->template init_prolongations<Basis>();
        // end of startup

        // int n_cycles = 200, pre_smooth = 1, post_smooth = 1, print_freq = 3;
        // bool print = true;
        // bool print = false;
        bool double_smooth = true;
        // bool double_smooth = false;
        // int nsmooth = 1, ninnercyc = 1, print_freq = 3;
        int print_freq = 3;
        int n_krylov = 50;
        T atol = 1e-4;  //, rtol = 1e-6;
        // bool double_smooth = false;  // actually faster sometimes

        // mg->init_outer_solver(nsmooth, ninnercyc, n_krylov, omega, atol, rtol, print_freq,
        // print);
        mg->init_outer_solver(cublasHandle, cusparseHandle, nsmooth, ninnercyc, n_krylov, omega,
                              atol, rtol, print_freq, print, double_smooth);
        solver = new StructSolver(*mg, print);

        // mg->solve();

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

            solver->solve(d_loads);
            num_lin_solves++;
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
    int get_num_lin_solves() { return num_lin_solves; }

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

    KMG *mg;  // multigrid object

    int num_lin_solves = 0;

    int ndvs = 0, nvars = 0;
    DeviceVec<T> d_loads, d_dvs;
    std::vector<T> prev_dvs;
    bool dvs_changed;
    bool first_solve;
    DeviceVec<T> soln;
};