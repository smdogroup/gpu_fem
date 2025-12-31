#include <iostream>
#include <sstream>

#include "chrono"
#include "coupled/_coupled.h"
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"

// new nonlinear solvers
#include "solvers/nonlinear_static/continuation.h"
#include "solvers/nonlinear_static/inexact_newton.h"

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

// multigrid imports
#include "multigrid/grid.h"
#include "multigrid/prolongation/structured.h"
#include "multigrid/smoothers/mc_smooth1.h"
#include "multigrid/solvers/gmg.h"
#include "multigrid/utils/fea.h"

// new multigrid imports for K-cycles, etc.
#include "multigrid/interface.h"
#include "multigrid/solvers/direct/cusp_directLU.h"
#include "multigrid/solvers/krylov/bsr_pcg.h"
#include "multigrid/solvers/multilevel/kcycle.h"
#include "multigrid/solvers/multilevel/twolevel.h"

class NonlinearPlateGPUSolver {
   public:
    using T = double;
    // FEM typedefs
    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Data = ShellIsotropicData<T, false>;
    using Physics = IsotropicShell<T, Data, true>;

    // if want MITC basis
    // using Basis = LagrangeQuadBasis<T, Quad, 1>;
    // using Assembler = MITCShellAssembler<T, Director, Basis, Physics, DeviceVec, BsrMat>;

    // if want chebyshev basis
    using Basis = ChebyshevQuadBasis<T, Quad, 1>;
    using Assembler = FullyIntegratedShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;

    using CoarseSolver = CusparseMGDirectLU<T, Assembler>;
    using LUsolver = CoarseSolver;

    // multigrid objects
    using Smoother = MulticolorGSSmoother_V1<Assembler>;
    using Prolongation = StructuredProlongation<Assembler, PLATE>;
    using GRID = SingleGrid<Assembler, Prolongation, Smoother, LINE_SEARCH>;
    using MG = GeometricMultigridSolver<GRID, CoarseSolver>;  // old V-cycle solver

    // for K-cycles
    using KrylovSolve = PCGSolver<T, GRID>;
    using TwoLevelSolve = MultigridTwoLevelSolver<GRID, false>;
    using KMG = MultilevelKcycleSolver<GRID, CoarseSolver, TwoLevelSolve, KrylovSolve>;

    // new nonlinear solver
    using Mat = BsrMat<DeviceVec<T>>;
    using Vec = DeviceVec<T>;
    using INK = InexactNewtonSolver<T, Mat, Vec, Assembler, KMG>;
    using NL = NonlinearContinuationSolver<T, Vec, Assembler, INK>;

    NonlinearPlateGPUSolver(double pressure = 100.0, double omega = 1.5, int nxe = 100,
                            double SR = 50.0, bool use_predictor = true, bool kmg_print = false,
                            bool nl_debug = false, bool debug_gmg_ = false) {
        // 1) Build mesh & assembler

        // create cublas and cusparse handles (single one each)
        // -----------------------------------------------------
        cublasHandle = NULL;
        cusparseHandle = NULL;
        CHECK_CUBLAS(cublasCreate(&cublasHandle));
        CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

        mg = nullptr;
        fine_LU_solver = nullptr;
        fine_LU_assembler = nullptr;

        debug_gmg = debug_gmg_;

        T omegaLS_min = 0.25, omegaLS_max = 2.0;

        // start building multigrid objects
        kmg = new KMG();
        if (debug_gmg) mg = new MG();

        // get nxe_min for not exactly power of 2 case
        int pre_nxe_min = nxe > 32 ? 32 : 4;
        int nxe_min = pre_nxe_min;
        for (int c_nxe = nxe; c_nxe >= pre_nxe_min; c_nxe /= 2) {
            nxe_min = c_nxe;
        }

        // make each grid
        int nx_comp = 1;
        for (int c_nxe = nxe; c_nxe >= nxe_min; c_nxe /= 2) {
            // make the assembler
            int c_nye = c_nxe;
            double Lx = 1.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 1.0 / SR, rho = 2500, ys = 350e6;
            int nxe_per_comp = c_nxe / nx_comp, nye_per_comp = c_nye / nx_comp;
            auto assembler = createPlateAssembler<Assembler>(c_nxe, c_nye, Lx, Ly, E, nu, thick,
                                                             rho, ys, nxe_per_comp, nye_per_comp);
            double uniform_force = pressure * 1.0 * 1.0;
            double nodal_loads = uniform_force / (c_nxe - 1) / (c_nye - 1);
            nodal_loads *= (100.0 / SR) * (100.0 / SR) * (100.0 / SR);
            T *my_loads = getPlateLoads<T, Physics>(c_nxe, c_nye, Lx, Ly, nodal_loads);
            printf("making grid with nxe %d\n", c_nxe);

            if (c_nxe == nxe && debug_gmg) {
                // make also the fine LU assembler (with different full LU pattern)
                auto _fine_LU_assemb = createPlateAssembler<Assembler>(
                    c_nxe, c_nye, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp);
                fine_LU_assembler = &_fine_LU_assemb;
                auto h_LU_bsr_data = &(fine_LU_assembler->getBsrData());
                h_LU_bsr_data->AMD_reordering();
                h_LU_bsr_data->compute_full_LU_pattern(10.0, false);
                fine_LU_assembler->moveBsrDataToDevice();
                fine_LU_kmat = createBsrMat<Assembler, VecType<T>>(*fine_LU_assembler);
                fine_LU_assembler->add_jacobian_fast(fine_LU_kmat);
                fine_LU_assembler->apply_bcs(fine_LU_kmat);
                fine_LU_bsr_data = fine_LU_assembler->getBsrData();

                // then make the fine LU linear solver
                fine_LU_solver =
                    new LUsolver(cublasHandle, cusparseHandle, *fine_LU_assembler, fine_LU_kmat);
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
            if (c_nxe == nxe) {
                nvars = res.getSize();
            }

            // assemble the kmat
            auto start0 = std::chrono::high_resolution_clock::now();
            assembler.add_jacobian_fast(kmat);
            assembler.apply_bcs(kmat);
            CHECK_CUDA(cudaDeviceSynchronize());
            auto end0 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> assembly_time = end0 - start0;
            printf("\tassemble kmat time %.2e\n", assembly_time.count());

            // build smoother and prolongations..
            auto smoother =
                new Smoother(cublasHandle, cusparseHandle, assembler, kmat, h_color_rowp, omega);
            auto prolongation = new Prolongation(assembler);
            auto grid = GRID(assembler, prolongation, smoother, kmat, loads, cublasHandle,
                             cusparseHandle, omegaLS_min, omegaLS_max);

            kmg->grids.push_back(grid);
            if (debug_gmg) {
                mg->grids.push_back(grid);
                if (full_LU)
                    mg->coarse_solver =
                        new CoarseSolver(cublasHandle, cusparseHandle, assembler, kmat);
            }
        }

        printf("done with grid creation\n");

        kmg->template init_prolongations<Basis>();
        if (debug_gmg) mg->template init_prolongations<Basis>();
        // end of startup

        bool double_smooth = true;
        int nsmooth = 2, ninnercyc = 2, print_freq = 3;
        int n_krylov = 20;
        T atol = 1e-6, rtol = 1e-6;

        kmg->init_outer_solver(cublasHandle, cusparseHandle, nsmooth, ninnercyc, n_krylov, omega,
                               atol, rtol, print_freq, kmg_print, double_smooth);
        // grids = &kmg->grids;

        // fine grid states
        fine_assembler = &kmg->grids[0].assembler;
        fine_soln = fine_assembler->createVarsVec();
        fine_res = fine_assembler->createVarsVec();
        fine_rhs = fine_assembler->createVarsVec();
        fine_loads = fine_assembler->createVarsVec();
        fine_vars = fine_assembler->createVarsVec();
        fine_kmat = kmg->grids[0].Kmat;

        temp1 = fine_assembler->createVarsVec();
        temp2 = fine_assembler->createVarsVec();
        temp3 = fine_assembler->createVarsVec();
        temp4 = fine_assembler->createVarsVec();
        temp5 = fine_assembler->createVarsVec();
        temp6 = fine_assembler->createVarsVec();

        h_temp_vec = HostVec<T>(nvars);

        // get fine loads from fine grid init rhs
        bool perm_out = true;
        kmg->grids[0].getDefect(fine_loads, perm_out);
        fine_assembler->apply_bcs(fine_loads);

        // printf("done with most of constructor 1\n");

        // build the inexact newton + outer continuation solver
        inner_solver = new INK(cublasHandle, *fine_assembler, fine_kmat, fine_loads, kmg);
        nl_solver = new NL(cublasHandle, *fine_assembler, inner_solver, use_predictor, nl_debug);

        // printf("done with most of constructor 2\n");
    }

    void continuationSolve(const T *u_0, T *u_f, const T lambda0 = 0.2, const T lambdaf = 1.0,
                           const T inner_atol = 1e-8) {
        /* perform a continuation solve from u_0 to u_f (output)*/
        set_variables(u_0);
        nl_solver->solve(fine_vars, lambda0, inner_atol, lambdaf);
        get_variables(u_f);
    }

    void vcycleSolve(const T *u, const T lambda, T *du, int n_cycles = 40) {
        /* run V-cycle solve at current nonlinear state, with prints */

        if (!debug_gmg) {
            printf("debug_gmg not active can't do Vcycle solve\n");
        }
        setGridDefect(u, lambda);
        mg->vcycle_solve(0, 2, 2, n_cycles, true, 1e-6, 1e-6, true, 3);
        getFineGridSoln(du);
    }

    void kcycleSolve(const T *u, const T lambda, T *du) {
        /* run K-cycle solve at current nonlinear state, with prints */

        setGridDefect(u, lambda);
        kmg->set_print(true);
        kmg->solve();
        kmg->set_print(false);
        getFineGridSoln(du);
    }

    T getResidual(const T *u, T lambda, T *res_out) {
        // get the current fine grid residual
        set_variables(u);
        T res_nrm = inner_solver->getResidual(lambda, fine_res);
        fine_vars.copyValuesTo(h_temp_vec);
        memcpy(res_out, h_temp_vec.getPtr(), nvars * sizeof(T));
        return res_nrm;
    }

    void setGridDefect(const T *u, T lambda, bool set_fine_LU = false) {
        /* set the defect on fine grid for V-cycle GMG or KMG solver */

        // update fine grid assembly and the coarser grids in GMG solver (with new vars)
        set_variables(u);  // sets into fine assembler
        fine_assembler->add_jacobian_fast(fine_kmat);
        fine_assembler->apply_bcs(fine_kmat);
        if (mg) mg->update_after_assembly(fine_vars);
        kmg->update_after_assembly(fine_vars);

        // compute and set the residual into the fine grid for GMG
        inner_solver->getResidual(lambda, fine_res);
        kmg->grids[0].setDefect(fine_res);

        // also set into the fine LU solver and asembler too
        if (set_fine_LU && debug_gmg) {
            printf("factoring full LU fine grid again\n");
            // fine_LU_assembler->set_variables(fine_vars); // fine_LU_assembler doesn't work on
            // second step here fine_LU_assembler->add_jacobian_fast(fine_LU_kmat);
            // fine_LU_assembler->apply_bcs(fine_LU_kmat);
            fine_LU_solver->assemble_matrix(fine_vars);
            fine_LU_solver->update_after_assembly(fine_vars);
            printf("fine LU solver update after assembly worked\n");
        }
    }

    void getCoarseFineStep(T *i_defect, T *ism_defect, T *cf_soln, T *ch_defect, T *fsm_defect,
                           T *lu_soln, bool smooth = true) {
        /* goal here is to run one V-cycle (can repeat this process), getting the current cf disp
         * and its original defect for comparison to fine LU solve */

        if (!debug_gmg) {
            printf(
                "need to turn on debug_gmg in order to be able to see coarse-fine and fineLU "
                "solves\n");
            return;
        }

        // 1) i_defect is init defect, 2) ism_defect is init smooth defect, 3) cf_soln is coarse
        // fine delta soln, 4) ch_defect is prolong change in defect, 5) fsm_defect is final
        // smoothed defect
        int start_level = 0, pre_smooth = 2, post_smooth = 2;
        mg->getCoarseFineStep(start_level, pre_smooth, post_smooth, temp1, temp2, temp3, temp4,
                              temp5, smooth, true);

        // printf("done running internal coarse fine step\n");

        // copy device vecs out to host (perm was done in mg->getCoarseFineStep method itself)
        copy_dvec_out(temp1, i_defect);
        copy_dvec_out(temp2, ism_defect);
        copy_dvec_out(temp3, cf_soln);
        copy_dvec_out(temp4, ch_defect);
        copy_dvec_out(temp5, fsm_defect);

        // NOTE : need to make sure setDefect had the fine LU set to true
        // also run the fine grid LU solver on the previous defect, to get 6) lu_soln
        temp2.permuteData(
            6, fine_LU_bsr_data.iperm);  // to solve order of fine LU (as ism defect here)
        fine_LU_solver->solve(temp2, temp6);
        temp6.permuteData(6, fine_LU_bsr_data.perm);  // permute to vis order
        copy_dvec_out(temp6, lu_soln);
    }

    void writeSolution(const std::string &filename, T *u) {
        memcpy(h_temp_vec.getPtr(), u,
               nvars * sizeof(T));  // copy to host temp vec for printing
        printToVTK<Assembler, HostVec<T>>(*fine_assembler, h_temp_vec, filename);
    }

    int get_num_vars() const { return nvars; }

    void free() {
        // free up all the data on the device
        if (kmg) kmg->free();
        if (mg) mg->free();
        if (nl_solver) nl_solver->free();
        if (inner_solver) inner_solver->free();
        if (fine_assembler) fine_assembler->free();
        if (fine_LU_solver) fine_LU_solver->free();
        if (fine_LU_assembler) fine_LU_assembler->free();
        if (debug_gmg) fine_LU_kmat.free();
        fine_kmat.free();

        // free up vecs
        h_temp_vec.free();
        temp1.free();
        temp2.free();
        temp3.free();
        temp4.free();
        temp5.free();
        temp6.free();
        fine_soln.free();
        fine_res.free();
        fine_rhs.free();
        fine_loads.free();
        fine_vars.free();

        // destroy the handles
        cublasDestroy(cublasHandle);
        cusparseDestroy(cusparseHandle);
    }

   private:
    void getFineGridSoln(T *u_out) {
        kmg->grids[0].d_soln.copyValuesTo(fine_soln);
        fine_soln.permuteData(6, kmg->grids[0].d_perm);  // permute to vis order
        fine_soln.copyValuesTo(h_temp_vec);
        memcpy(u_out, h_temp_vec.getPtr(), nvars * sizeof(T));
    }

    void set_variables(const T *u_in) {
        memcpy(h_temp_vec.getPtr(), u_in, nvars * sizeof(T));
        h_temp_vec.copyValuesTo(fine_vars);
        fine_assembler->set_variables(fine_vars);
    }

    void get_variables(T *u_out) {
        fine_vars.copyValuesTo(h_temp_vec);
        memcpy(u_out, h_temp_vec.getPtr(), nvars * sizeof(T));
    }

    void copy_dvec_out(DeviceVec<T> &vec, T *h_out) {
        /* copy device vec out to host */
        vec.copyValuesTo(h_temp_vec);
        memcpy(h_out, h_temp_vec.getPtr(), nvars * sizeof(T));
    }

    KMG *kmg;  // multigrid object
    MG *mg;
    INK *inner_solver;  // nonlinear solvers
    NL *nl_solver;
    Assembler *fine_assembler;
    bool debug_gmg;

    // fine LU solver stuff
    LUsolver *fine_LU_solver;
    Assembler *fine_LU_assembler;
    Mat fine_LU_kmat;
    BsrData fine_LU_bsr_data;

    HostVec<T> h_temp_vec;
    Vec fine_soln, fine_res, fine_rhs, fine_loads, fine_vars;
    Mat fine_kmat;

    Vec temp1, temp2, temp3, temp4, temp5, temp6;

    int nvars;

    cublasHandle_t cublasHandle;
    cusparseHandle_t cusparseHandle;
};