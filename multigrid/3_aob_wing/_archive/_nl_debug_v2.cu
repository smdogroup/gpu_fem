
#include "linalg/_linalg.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"
#include "solvers/_solvers.h"

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
#include "multigrid/smoothers/_wingbox_coloring.h"
#include "multigrid/smoothers/mc_smooth1.h"
#include "multigrid/prolongation/unstructured.h"
#include "multigrid/solvers/gmg.h"
#include <string>
#include <chrono>

// new multigrid imports for K-cycles, etc.
#include "multigrid/solvers/solve_utils.h"
#include "multigrid/solvers/direct/cusp_directLU.h"
#include "multigrid/solvers/krylov/bsr_pcg.h"
#include "multigrid/solvers/multilevel/kcycle.h"
#include "multigrid/solvers/multilevel/twolevel.h"

/* argparse options:
[mg/direct/debug] [--level int]
*/

void to_lowercase(char *str) {
    for (; *str; ++str) {
        *str = std::tolower(*str);
    }
}

std::string time_string(int itime) {
    std::string _time = std::to_string(itime);
    if (itime < 10) {
        return "00" + _time;
    } else if (itime < 100) {
        return "0" + _time;
    } else {
        return _time;
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
void solve_nonlinear_multigrid(MPI_Comm &comm, int level, double SR, 
    int nsmooth, int ninnercyc, std::string cycle_type, double total_force = 1.0) {
    // geometric multigrid method here..
    // need to make a number of grids..
    // level gives the finest level here..

    using Basis = typename Assembler::Basis;
    using Physics = typename Assembler::Phys;
    using Data = typename Physics::Data;
    using Smoother = MulticolorGSSmoother_V1<Assembler>;
    const bool is_bsr = true; // need this one if want to smooth prolongation
    // const bool is_bsr = false; // no difference in intra-nodal (default old working prolong)
    using Prolongation = UnstructuredProlongation<Assembler, Basis, is_bsr>; 
    using GRID = SingleGrid<Assembler, Prolongation, Smoother, LINE_SEARCH>;
    using CoarseSolver = CusparseMGDirectLU<T, Assembler>;
    using MG = GeometricMultigridSolver<GRID, CoarseSolver>;

    // for K-cycles
    // constexpr bool full_approx_scheme = true;
    constexpr bool full_approx_scheme = false; // not fully implemented yet so false
    using KrylovSolve = PCGSolver<T, GRID>;
    using TwoLevelSolve = MultigridTwoLevelSolver<GRID, full_approx_scheme>;
    using KMG = MultilevelKcycleSolver<GRID, CoarseSolver, TwoLevelSolve, KrylovSolve>;

    // create cublas and cusparse handles (single one each)
    // -----------------------------------------------------
    cublasHandle_t cublasHandle = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));
    cusparseHandle_t cusparseHandle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

    auto start0 = std::chrono::high_resolution_clock::now();

    // hopefully this doesn't construct the object?
    MG *mg;
    KMG *kmg;

    bool is_kcycle = cycle_type == "K";
    if (!is_kcycle) {
        printf("non K-cycle not supported for this script yet (just would have to change it a bit) with one or two function calls or another routine maybe\n");
        return;
    }

    if (is_kcycle) {
        kmg = new KMG();
    } else {
        mg = new MG();
    }

    // make each wing multigrid object.. with L0 the coarsest mesh, L3 finest 
    //   (this way mg.grids is still finest to coarsest meshes order by convention)
    for (int i = level; i >= 0; i--) {

        // read the ESP/CAPS => nastran mesh for TACS
        TACSMeshLoader mesh_loader{comm};

        // temp debug i+1
        std::string fname = "meshes/aob_wing_L" + std::to_string(i+1) + ".bdf";
        
        // std::string fname = "meshes/aob_wing_L" + std::to_string(i) + ".bdf";
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
        int level_exp = 1 << level;
        // printf("level_exp %d\n", level_exp);
        double load_mag = total_force / nnodes; // estimate for nodal load mag
        double *my_loads = h_loads.getPtr();
        for (int inode = 0; inode < nnodes; inode++) {
            my_loads[6 * inode + 2] = load_mag;
        }

        // SET DVs
        // -------------------

        if (SR == 0.0) {
            T h_dvs_ptr[111] = {0.004818181818181818, 0.0047272727272727275, 0.0047272727272727275, 0.018636363636363635, 0.00490909090909091, 0.004818181818181818, 0.018636363636363635, 0.017954545454545456, 0.017954545454545456, 0.0046363636363636355, 0.0046363636363636355, 0.004818181818181818, 0.0047272727272727275, 0.01931818181818182, 0.005, 0.00490909090909091, 0.01931818181818182, 0.00490909090909091, 0.01727272727272727, 0.01727272727272727, 0.004545454545454545, 0.004545454545454545, 0.0046363636363636355, 0.02, 0.005, 0.02, 0.005, 0.01659090909090909, 0.01659090909090909, 0.004454545454545454, 0.004454545454545454, 0.004545454545454545, 0.015909090909090907, 0.015909090909090907, 0.004363636363636364, 0.004363636363636364, 0.004454545454545454, 0.015227272727272728, 0.015227272727272728, 0.004272727272727273, 0.004272727272727273, 0.004363636363636364, 0.014545454545454545, 0.014545454545454545, 0.0041818181818181815, 0.0041818181818181815, 0.004272727272727273, 0.013863636363636363, 0.013863636363636363, 0.00409090909090909, 0.00409090909090909, 0.0041818181818181815, 0.01318181818181818, 0.01318181818181818, 0.004, 0.004, 0.00409090909090909, 0.0125, 0.0125, 0.003909090909090909, 0.003909090909090909, 0.004, 0.01181818181818182, 0.01181818181818182, 0.003818181818181818, 0.003818181818181818, 0.003909090909090909, 0.011136363636363635, 0.011136363636363635, 0.0037272727272727275, 0.0037272727272727275, 0.003818181818181818, 0.010454545454545454, 0.010454545454545454, 0.0036363636363636364, 0.0036363636363636364, 0.0037272727272727275, 0.009772727272727273, 0.009772727272727273, 0.003545454545454545, 0.003545454545454545, 0.0036363636363636364, 0.00909090909090909, 0.00909090909090909, 0.003454545454545455, 0.003454545454545455, 0.003545454545454545, 0.00840909090909091, 0.00840909090909091, 0.003363636363636364, 0.003363636363636364, 0.003454545454545455, 0.007727272727272727, 0.007727272727272727, 0.003272727272727273, 0.003272727272727273, 0.003363636363636364, 0.007045454545454546, 0.007045454545454546, 0.003181818181818182, 0.003181818181818182, 0.003272727272727273, 0.006363636363636364, 0.006363636363636364, 0.0030909090909090908, 0.0030909090909090908, 0.003181818181818182, 0.005681818181818181, 0.005681818181818181, 0.003, 0.0030909090909090908};
            auto h_dvs = HostVec<T>(111, h_dvs_ptr);
            auto global_dvs = h_dvs.createDeviceVec();
            assembler.set_design_variables(global_dvs);
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
        printf("\tassemble kmat time %.2e\n", assembly_time.count());

        // CHECK_CUDA(cudaDeviceSynchronize());
        // auto startar = std::chrono::high_resolution_clock::now();
        // // const int elems_per_blockr = 32;
        // const int elems_per_blockr = 8;
        // // const int elems_per_blockr = 4;
        // assembler.template add_residual_fast<elems_per_blockr>(res);
        // // assembler.add_residual(res);
        // CHECK_CUDA(cudaDeviceSynchronize());
        // auto endar = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> assemb_resid_time = endar - startar;
        // printf("\tassemble resid time %.2e\n", assemb_resid_time.count());

        // return;

        // build smoother and prolongations
        T omega = 1.5; // for GS-SOR
        auto smoother = new Smoother(cublasHandle, cusparseHandle, assembler, kmat, h_color_rowp, omega);
        int ELEM_MAX = 10; // num nearby elements of each fine node for nz pattern construction
        // int ELEM_MAX = 4;
        auto prolongation = new Prolongation(cusparseHandle, assembler, ELEM_MAX);
        auto grid = GRID(assembler, prolongation, smoother, kmat, loads, cublasHandle, cusparseHandle);

        if (is_kcycle) {
            kmg->grids.push_back(grid);
        } else {
            mg->grids.push_back(grid);
            if (coarsest_grid) mg->coarse_solver = new CoarseSolver(cublasHandle, cusparseHandle, 
                assembler, kmat);
        }
    }

    // register the coarse assemblers to the prolongations..
    if (is_kcycle) {
        kmg->template init_prolongations<Basis>();
    } else {
        mg->template init_prolongations<Basis>();
    }

    auto end0 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> startup_time = end0 - start0;

    T init_resid_nrm = is_kcycle ? kmg->grids[0].getResidNorm() : mg->grids[0].getResidNorm();

    CHECK_CUDA(cudaDeviceSynchronize());
    auto start1 = std::chrono::high_resolution_clock::now();
    printf("starting %s cycle solve\n", cycle_type.c_str());
    int pre_smooth = nsmooth, post_smooth = nsmooth;
    // best was V(4,4) before
    bool print = false;
    // bool print = true;
    T atol = 1e-9, rtol = 1e-9;
    T omega2 = 1.5; // really is set up there
    int n_cycles = SR >= 100.0 ? 1000 : 200;
    // bool time = false;
    bool time = true;
    int print_freq = 5;

    // bool double_smooth = false;
    bool double_smooth = true; // true tends to be slightly faster sometimes

    if (is_kcycle) {
        // int n_krylov = 500;
        // int n_krylov = 20;
        int n_krylov = 50;
        kmg->init_outer_solver(cublasHandle, cusparseHandle, nsmooth, ninnercyc, 
            n_krylov, omega2, atol, rtol, print_freq, print, double_smooth);    
    }

    std::vector<GRID>& grids = is_kcycle ? kmg->grids : mg->grids;

    // fine grid states
    auto& fine_assembler = grids[0].assembler;
    auto fine_soln = fine_assembler.createVarsVec();
    auto fine_res = fine_assembler.createVarsVec();
    auto fine_rhs = fine_assembler.createVarsVec();
    auto fine_loads = fine_assembler.createVarsVec();
    auto fine_vars = fine_assembler.createVarsVec();
    auto& fine_kmat = grids[0].Kmat;

    // get fine loads from fine grid init rhs
    bool perm_out = true;
    grids[0].getDefect(fine_loads, perm_out);
    fine_assembler.apply_bcs(fine_loads);

    // ---------------------------------------------------
    // 0) demo restrict fine to coarse soln

    // // first solve on fine grid (with initial linear defect)
    // kmg->solve();

    // // // now pass soln down to the coarse grid
    // grids[1].restrict_soln(grids[0].d_soln);

    // int *d_perm = kmg->grids[0].d_perm;
    // auto h_soln = kmg->grids[0].d_soln.createPermuteVec(6, d_perm).createHostVec();
    // printToVTK<Assembler,HostVec<T>>(kmg->grids[0].assembler, h_soln, "out/wing_lin0.vtk");

    // int *d_perm1 = kmg->grids[1].d_perm;
    // auto h_soln1 = kmg->grids[1].d_vars.createPermuteVec(6, d_perm1).createHostVec();
    // printToVTK<Assembler,HostVec<T>>(kmg->grids[1].assembler, h_soln1, "out/wing_lin1.vtk");
    // return;

    // 1) do a linear solve here
    // -------------------------------------------------------

    kmg->solve();
    int *d_perm = kmg->grids[0].d_perm;
    auto h_soln = kmg->grids[0].d_soln.createPermuteVec(6, d_perm).createHostVec();
    printToVTK<Assembler,HostVec<T>>(kmg->grids[0].assembler, h_soln, "out/wing_mg_lin.vtk");
    T lin_max_disp = get_max_disp(kmg->grids[0].d_soln);

    // ------------------------------------------------------------
    // 2) solve nonlinear Newton-Raphson load-step scheme

    // build the inexact newton + outer continuation solver
    using Mat = BsrMat<DeviceVec<T>>;
    using Vec = DeviceVec<T>;
    using INK = InexactNewtonSolver<T, Mat, Vec, Assembler, KMG>;
    using NL = NonlinearContinuationSolver<T, Vec, Assembler, INK>;

    // bool use_predictor = true;
    bool use_predictor = false;

    INK *inner_solver = new INK(cublasHandle, fine_assembler, fine_kmat, fine_loads, kmg);
    NL *nl_solver = new NL(cublasHandle, fine_assembler, inner_solver, use_predictor);

    // now try calling it
    T lambda0 = 0.2;
    // T lambda0 = 0.1;
    // T lambda0 = 0.05;
    T inner_atol = 1e-3;
    // T inner_atol = 1e-8;
    nl_solver->solve(fine_vars, lambda0, inner_atol);
    T nl_max_disp = get_max_disp(fine_vars);
    // printf("done with continuation solve - DEBUG PRINT\n");


    // DEBUG (after exiting at failed state)
    // ==================================================

    bool debug = true;
    if (debug) {
        auto h_vars0 = fine_vars.createHostVec();
        printToVTK<Assembler,HostVec<T>>(fine_assembler, h_vars0, "out/wing_failed_state.vtk");

        // compute residual
        kmg->set_print(true); // turn on print for the outer solver
        T lambda = nl_solver->get_last_lambda();
        inner_solver->debug_solve(lambda, 1e-3, 1e-8, fine_vars, fine_res);
        printf("done with inner solver debug solve - DEBUG PRINT\n");

        // add grids and coarse solver from the kmg to gmg V-cycle solver
        mg = new MG();
        for (int i = 0; i < grids.size(); i++) {
            mg->grids.push_back(grids[i]);
        }
        printf("pushed back grids\n");
        mg->coarse_solver = static_cast<CoarseSolver*>(kmg->coarse_solver);

        // make also a fine grid direct solver..
        auto fine_solver = new CoarseSolver(cublasHandle, cusparseHandle, 
                grids[0].assembler, grids[0].Kmat);

        // test the fine solver out first on the residual (to make sure it works reasonably well..)
        auto h_res = fine_res.createHostVec();
        printToVTK<Assembler,HostVec<T>>(fine_assembler, h_res, "out/debug/wing_fine_res.vtk");
        
        fine_res.permuteData(6, grids[0].d_iperm);
        fine_solver->solve(fine_res, fine_soln);
        fine_soln.permuteData(6, grids[0].d_perm);
        auto h_solnf = fine_soln.createHostVec();
        printToVTK<Assembler,HostVec<T>>(fine_assembler, h_solnf, "out/debug/wing_fine_exact_soln.vtk");
        fine_res.permuteData(6, grids[0].d_perm);

        // can either run with previous defect sitting in fine grid
        // AND not setDefect

        // OR reset the defect to the fine residual.. you pick
        grids[0].setDefect(fine_res);
        grids[0].d_soln.zeroValues();

        // now try solving V-cycles manually
        printf("BEGIN V-cycle solve\n");
        mg->template debug_vcycle_solve<Assembler>(fine_solver, 0, 4, 4, 10, true, 1e-8, 1e-8, true, 5);
    }
    


    // ==================================================

    // print some of the data of host residual
    auto h_vars = fine_vars.createHostVec();
    printToVTK<Assembler,HostVec<T>>(fine_assembler, h_vars, "out/wing_mg_nl.vtk");

    // important to know reduction for how NL regime we are
    T ratio = nl_max_disp / lin_max_disp;
    printf("lin max disp %.8e, nl max disp %.8e, ratio = %.8e\n", lin_max_disp, nl_max_disp, ratio);

    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = end1 - start1;
    int ndof = fine_assembler.get_num_vars();
    double total = startup_time.count() + solve_time.count();
    printf("nonlinear Newton-Raphson KMG solve of wing geom, ndof %d : startup time %.2e, solve time %.2e, total %.2e\n", ndof, startup_time.count(), solve_time.count(), total);

    // free and cleanup
    // --------------------

    // nl_solver.free();
    kmg->free();
    fine_assembler.free();
}

template <typename T, class Assembler>
void solve_nonlinear_direct(MPI_Comm &comm, int level, double SR, double total_force) {
  
    using Basis = typename Assembler::Basis;
    using Physics = typename Assembler::Phys;
    using Data = typename Physics::Data;

  auto start0 = std::chrono::high_resolution_clock::now();

  TACSMeshLoader mesh_loader{comm};
  std::string fname = "meshes/aob_wing_L" + std::to_string(level) + ".bdf";
  mesh_loader.scanBDFFile(fname.c_str());

  //   double E = 70e9, nu = 0.3, thick = 0.005;  // material & thick properties
  double E = 70e9, nu = 0.3, thick = 2.0 / SR;  // material & thick properties

  // make the assembler from the uCRM mesh
  auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));

  // see _get_thicks.py and _thicks.txt (for this design)
  if (SR == 0.0) {
    T h_dvs_ptr[111] = {0.004818181818181818, 0.0047272727272727275, 0.0047272727272727275, 0.018636363636363635, 0.00490909090909091, 0.004818181818181818, 0.018636363636363635, 0.017954545454545456, 0.017954545454545456, 0.0046363636363636355, 0.0046363636363636355, 0.004818181818181818, 0.0047272727272727275, 0.01931818181818182, 0.005, 0.00490909090909091, 0.01931818181818182, 0.00490909090909091, 0.01727272727272727, 0.01727272727272727, 0.004545454545454545, 0.004545454545454545, 0.0046363636363636355, 0.02, 0.005, 0.02, 0.005, 0.01659090909090909, 0.01659090909090909, 0.004454545454545454, 0.004454545454545454, 0.004545454545454545, 0.015909090909090907, 0.015909090909090907, 0.004363636363636364, 0.004363636363636364, 0.004454545454545454, 0.015227272727272728, 0.015227272727272728, 0.004272727272727273, 0.004272727272727273, 0.004363636363636364, 0.014545454545454545, 0.014545454545454545, 0.0041818181818181815, 0.0041818181818181815, 0.004272727272727273, 0.013863636363636363, 0.013863636363636363, 0.00409090909090909, 0.00409090909090909, 0.0041818181818181815, 0.01318181818181818, 0.01318181818181818, 0.004, 0.004, 0.00409090909090909, 0.0125, 0.0125, 0.003909090909090909, 0.003909090909090909, 0.004, 0.01181818181818182, 0.01181818181818182, 0.003818181818181818, 0.003818181818181818, 0.003909090909090909, 0.011136363636363635, 0.011136363636363635, 0.0037272727272727275, 0.0037272727272727275, 0.003818181818181818, 0.010454545454545454, 0.010454545454545454, 0.0036363636363636364, 0.0036363636363636364, 0.0037272727272727275, 0.009772727272727273, 0.009772727272727273, 0.003545454545454545, 0.003545454545454545, 0.0036363636363636364, 0.00909090909090909, 0.00909090909090909, 0.003454545454545455, 0.003454545454545455, 0.003545454545454545, 0.00840909090909091, 0.00840909090909091, 0.003363636363636364, 0.003363636363636364, 0.003454545454545455, 0.007727272727272727, 0.007727272727272727, 0.003272727272727273, 0.003272727272727273, 0.003363636363636364, 0.007045454545454546, 0.007045454545454546, 0.003181818181818182, 0.003181818181818182, 0.003272727272727273, 0.006363636363636364, 0.006363636363636364, 0.0030909090909090908, 0.0030909090909090908, 0.003181818181818182, 0.005681818181818181, 0.005681818181818181, 0.003, 0.0030909090909090908};
    auto h_dvs = HostVec<T>(111, h_dvs_ptr);
    auto global_dvs = h_dvs.createDeviceVec();
    assembler.set_design_variables(global_dvs);
  }

  // TODO : set this in from optimized design from AOB case

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

  // get the loads
  int nvars = assembler.get_num_vars();
  int nnodes = assembler.get_num_nodes();
  HostVec<T> h_loads(nvars);
  int level_exp = 1 << level;
  double load_mag = total_force / nnodes; // estimate for nodal load mag
//   double SR1 = (300.0 / SR);
//   double SR3 = SR1 * SR1 * SR1;
//   load_mag *= SR3;
  double *h_loads_ptr = h_loads.getPtr();
  for (int inode = 0; inode < nnodes; inode++) {
    h_loads_ptr[6 * inode + 2] = load_mag;
  }
  auto loads = h_loads.createDeviceVec();
  assembler.apply_bcs(loads);

  // setup kmat and initial vecs
  auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
  auto soln = assembler.createVarsVec();
  auto res = assembler.createVarsVec();
  auto vars = assembler.createVarsVec();
  auto rhs = assembler.createVarsVec();

  // assemble the kmat
  assembler.set_variables(vars);
  
  CHECK_CUDA(cudaDeviceSynchronize());
  auto starta = std::chrono::high_resolution_clock::now();
//   assembler.add_jacobian(res, kmat);
  assembler.add_jacobian_fast(kmat);
  CHECK_CUDA(cudaDeviceSynchronize());
  auto enda = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> assemb_time = enda - starta;

  assembler.apply_bcs(res);
  assembler.apply_bcs(kmat);

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
    // T lambda0 = 0.1;
    // T lambda0 = 0.05;
    // T inner_atol = 1e-5;
    T inner_atol = 1e-8;
    nl_solver->solve(vars, lambda0, inner_atol);
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
    printf("nonlinear Newton-Raphson Direct-LU solve of wing geom, ndof %d : startup time %.2e, solve time %.2e, total %.2e\n", ndof, startup_time.count(), solve_time.count(), total);

    // free and cleanup
    // --------------------
    
    // nl_solver.free();
    assembler.free();
}

template <typename T, class Assembler>
void gatekeeper_method(bool is_multigrid, MPI_Comm &comm, int level, double SR, int nsmooth, int ninnercyc, std::string cycle_type, double total_force) {
    if (is_multigrid) {
        solve_nonlinear_multigrid<T, Assembler>(comm, level, SR, nsmooth, ninnercyc, cycle_type, total_force);
    } else {
        solve_nonlinear_direct<T, Assembler>(comm, level, SR, total_force);
    }
}

int main(int argc, char **argv) {

    // Intialize MPI and declare communicator
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    // DEFAULTS
    int level = 1; // level mesh to solve.. level 4 also a good starting setting (big case)
    bool is_multigrid = true;
    // bool is_debug = false;
    
    // very slender (buckles earlier)
    // double force = 6e5;
    // double SR = 100.0; // so that uses optimal design from AOB paper

    // less slender harder to buckle and can step into NL better (SR depends on wing length too, uCRM can be a bit more slender maybe because of the better narrower design, less likely to buckle, like beam)
    double force = 2e7;
    double SR = 20.0; // so that uses optimal design from AOB paper

    int nsmooth = 4; // may need more here (esp for MITC elements, but CFI can use less)
    int ninnercyc = 2; // inner V-cycles to precond K-cycle
    std::string cycle_type = "K"; // "V", "F", "W", "K"

    // probably need more locking / multigrid friendly element than either of these (CFI4 is locking, while MITC4 has bad GMG performance)
    std::string elem_type = "CFI4"; // 'MITC4', 'CFI4', 'CFI9'
    // std::string elem_type = "MITC4"; // 'MITC4', 'CFI4', 'CFI9'

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        char* arg = argv[i];
        to_lowercase(arg);

        if (strcmp(arg, "direct") == 0) {
            is_multigrid = false;
        } else if (strcmp(arg, "mg") == 0) {
            is_multigrid = true;
        } else if (strcmp(arg, "--sr") == 0) {
            if (i + 1 < argc) {
                SR = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --SR\n";
                return 1;
            }
        } else if (strcmp(arg, "--level") == 0) {
            if (i + 1 < argc) {
                level = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --level\n";
                return 1;
            }
        } else if (strcmp(arg, "--cycle") == 0) {
            if (i + 1 < argc) {
                cycle_type = argv[++i];
            } else {
                std::cerr << "Missing value for --level\n";
                return 1;
            }
        } else if (strcmp(arg, "--force") == 0) {
            if (i + 1 < argc) {
                force = std::atof(argv[++i]);
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
        } else if (strcmp(arg, "--ninnercyc") == 0) {
            if (i + 1 < argc) {
                ninnercyc = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --nsmooth\n";
                return 1;
            }
        } else {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            std::cerr << "Usage: " << argv[0] << " [direct/mg] [--level int] [--SR double] [--cycle char] [--nsmooth int] [--ninnercyc int]" << std::endl;
            return 1;
        }
    }

    // type specifications here
    using T = double;   
    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    constexpr bool has_ref_axis = false;
    // constexpr bool is_nonlinear = false;
    constexpr bool is_nonlinear = true;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;

    printf("AOB mesh with nonlinear %s elements, level %d and SR %.2e\n------------\n", elem_type.c_str(), level, SR);
    if (elem_type == "MITC4") {
        using Basis = LagrangeQuadBasis<T, Quad, 1>;
        using Assembler = MITCShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
        gatekeeper_method<T, Assembler>(is_multigrid, comm, level, SR, nsmooth, ninnercyc, cycle_type, force);
    } else if (elem_type == "CFI4") {
        using Basis = ChebyshevQuadBasis<T, Quad, 1>;
        using Assembler = FullyIntegratedShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
        gatekeeper_method<T, Assembler>(is_multigrid, comm, level, SR, nsmooth, ninnercyc, cycle_type, force);
    } else if (elem_type == "CFI9") {
        using Basis = ChebyshevQuadBasis<T, Quad, 2>;
        using Assembler = FullyIntegratedShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
        gatekeeper_method<T, Assembler>(is_multigrid, comm, level, SR, nsmooth, ninnercyc, cycle_type, force);
    } else {
        printf("ERROR : didn't run anything, elem type not in available types (see main function)\n");
    }

    MPI_Finalize();
    return 0;
};
