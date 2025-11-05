
#include "linalg/_linalg.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"
#include "solvers/_solvers.h"

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

template <typename T, class Assembler>
void solve_nonlinear_multigrid(MPI_Comm &comm, int level, double SR, int nsmooth, int ninnercyc, std::string cycle_type) {
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
    using KrylovSolve = PCGSolver<T, GRID>;
    using TwoLevelSolve = MultigridTwoLevelSolver<GRID>;
    using KMG = MultilevelKcycleSolver<GRID, CoarseSolver, TwoLevelSolve, KrylovSolve>;

    auto start0 = std::chrono::high_resolution_clock::now();

    // hopefully this doesn't construct the object?
    MG *mg;
    KMG *kmg;

    bool is_kcycle = cycle_type == "K";
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
        std::string fname = "meshes/aob_wing_L" + std::to_string(i) + ".bdf";
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
        printf("level_exp %d\n", level_exp);
        double load_mag = 40.0 * (16.0 / level_exp / level_exp); // 4x the linear loads (so get more geomNL deflections)
        double *my_loads = h_loads.getPtr();
        for (int inode = 0; inode < nnodes; inode++) {
            my_loads[6 * inode + 2] = load_mag;
        }

        // SET DVs
        // -------------------

        // see _get_thicks.py and _thicks.txt (for this design)
        T h_dvs_ptr[111] = {0.004818181818181818, 0.0047272727272727275, 0.0047272727272727275, 0.018636363636363635, 0.00490909090909091, 0.004818181818181818, 0.018636363636363635, 0.017954545454545456, 0.017954545454545456, 0.0046363636363636355, 0.0046363636363636355, 0.004818181818181818, 0.0047272727272727275, 0.01931818181818182, 0.005, 0.00490909090909091, 0.01931818181818182, 0.00490909090909091, 0.01727272727272727, 0.01727272727272727, 0.004545454545454545, 0.004545454545454545, 0.0046363636363636355, 0.02, 0.005, 0.02, 0.005, 0.01659090909090909, 0.01659090909090909, 0.004454545454545454, 0.004454545454545454, 0.004545454545454545, 0.015909090909090907, 0.015909090909090907, 0.004363636363636364, 0.004363636363636364, 0.004454545454545454, 0.015227272727272728, 0.015227272727272728, 0.004272727272727273, 0.004272727272727273, 0.004363636363636364, 0.014545454545454545, 0.014545454545454545, 0.0041818181818181815, 0.0041818181818181815, 0.004272727272727273, 0.013863636363636363, 0.013863636363636363, 0.00409090909090909, 0.00409090909090909, 0.0041818181818181815, 0.01318181818181818, 0.01318181818181818, 0.004, 0.004, 0.00409090909090909, 0.0125, 0.0125, 0.003909090909090909, 0.003909090909090909, 0.004, 0.01181818181818182, 0.01181818181818182, 0.003818181818181818, 0.003818181818181818, 0.003909090909090909, 0.011136363636363635, 0.011136363636363635, 0.0037272727272727275, 0.0037272727272727275, 0.003818181818181818, 0.010454545454545454, 0.010454545454545454, 0.0036363636363636364, 0.0036363636363636364, 0.0037272727272727275, 0.009772727272727273, 0.009772727272727273, 0.003545454545454545, 0.003545454545454545, 0.0036363636363636364, 0.00909090909090909, 0.00909090909090909, 0.003454545454545455, 0.003454545454545455, 0.003545454545454545, 0.00840909090909091, 0.00840909090909091, 0.003363636363636364, 0.003363636363636364, 0.003454545454545455, 0.007727272727272727, 0.007727272727272727, 0.003272727272727273, 0.003272727272727273, 0.003363636363636364, 0.007045454545454546, 0.007045454545454546, 0.003181818181818182, 0.003181818181818182, 0.003272727272727273, 0.006363636363636364, 0.006363636363636364, 0.0030909090909090908, 0.0030909090909090908, 0.003181818181818182, 0.005681818181818181, 0.005681818181818181, 0.003, 0.0030909090909090908};
        auto h_dvs = HostVec<T>(111, h_dvs_ptr);
        auto global_dvs = h_dvs.createDeviceVec();
        assembler.set_design_variables(global_dvs);

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
        auto smoother = new Smoother(assembler, kmat, h_color_rowp, omega);
        int ELEM_MAX = 10; // num nearby elements of each fine node for nz pattern construction
        // int ELEM_MAX = 4;
        auto prolongation = new Prolongation(assembler, ELEM_MAX);
        auto grid = GRID(assembler, prolongation, smoother, kmat, loads);

        if (is_kcycle) {
            kmg->grids.push_back(grid);
        } else {
            mg->grids.push_back(grid);
            if (coarsest_grid) mg->coarse_solver = new CoarseSolver(assembler, kmat);
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
    // bool print = false;
    bool print = true;
    T atol = 1e-6, rtol = 1e-6;
    T omega2 = 1.5; // really is set up there
    int n_cycles = SR >= 100.0 ? 1000 : 200;
    // bool time = false;
    bool time = true;
    int print_freq = 5;

    // bool double_smooth = false;
    bool double_smooth = true; // true tends to be slightly faster sometimes

    if (is_kcycle) {
        int n_krylov = 500;
        kmg->init_outer_solver(nsmooth, ninnercyc, n_krylov, omega2, atol, rtol, print_freq, print, double_smooth);    
    }

    std::vector<GRID>& grids = is_kcycle ? kmg->grids : mg->grids;

    // ---------------------------------------------------
    // 1) demo restrict fine to coarse soln

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

    // ------------------------------------------------------------
    // 2) solve nonlinear Newton-Raphson load-step scheme

    // try solving a nonlinear load step scheme (Newton-mg)
    int num_load_factors = 20, num_newton = 10; // num_load_factors = 10
    // careful, multigrid solve currently breaks down (may need galerkin coarse grids) when the wing buckles
    T max_load_factor = 1.0; 
    // T max_load_factor = 0.1; // with old thicknesses 2.0 / SR each panel (not optimized design), can't get large deflections before buckling
    T min_load_factor = max_load_factor / (num_load_factors - 1);
    T abs_tol = 1e-8, rel_tol = 1e-8;
    std::string outputPrefix = "out/wing_nl_mg_";
    bool write_vtk = level <= 2; // otherwise it's too slow
    // bool write_vtk = false;

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

    for (int iload = 0; iload < num_load_factors; iload++) {
        T load_factor =
            min_load_factor + (max_load_factor - min_load_factor) * iload / (num_load_factors - 1);

        T init_res = 1e50;
        if (print) {
            printf("load step %d / %d : load factor %.4e\n", iload, num_load_factors, load_factor);
        }

        for (int inewton = 0; inewton < num_newton; inewton++) {

            // update the fine grid stiffness matrix and residual
            fine_assembler.set_variables(fine_vars);
            fine_assembler.add_jacobian_fast(fine_kmat);
            fine_assembler.add_residual_fast(fine_res);
            fine_assembler.apply_bcs(fine_res);
            fine_assembler.apply_bcs(fine_kmat);

            // pass current states to coarse grids and update their assemblies
            grids[0].setStateVars(fine_vars); // set vars into finest grid (so we can pass this down to coarser grids)

            if (is_kcycle) {
                kmg->update_coarse_grid_states(); // restrict state variables to coarse grids
                kmg->update_coarse_grid_jacobians(); // compute coarse grid NL stiffness matrices
                kmg->update_after_assembly(); // updates any dependent matrices like Dinv
            } else {
                mg->update_coarse_grid_states(); // restrict state variables to coarse grids
                mg->update_coarse_grid_jacobians(); // compute coarse grid NL stiffness matrices
                mg->update_after_assembly(); // updates any dependent matrices like Dinv
            }

            // if (iload == 1 && inewton == 1) {
            //     // break;
            //     // show the solution on the coarse grid
            //     printf("writing iload 1, inewton 1 debug files\n");
            //     int *d_perm = kmg->grids[0].d_perm;
            //     auto h_vars = kmg->grids[0].d_vars.createPermuteVec(6, d_perm).createHostVec();
            //     printToVTK<Assembler,HostVec<T>>(kmg->grids[0].assembler, h_vars, "out/wing_nl_debug0.vtk");

            //     int *d_perm1 = kmg->grids[1].d_perm;
            //     auto h_vars1 = kmg->grids[1].d_vars.createPermuteVec(6, d_perm1).createHostVec();
            //     printToVTK<Assembler,HostVec<T>>(kmg->grids[1].assembler, h_vars1, "out/wing_nl_debug1.vtk");

            //     int *d_perm2 = kmg->grids[2].d_perm;
            //     auto h_vars2 = kmg->grids[2].d_vars.createPermuteVec(6, d_perm2).createHostVec();
            //     printToVTK<Assembler,HostVec<T>>(kmg->grids[2].assembler, h_vars2, "out/wing_nl_debug2.vtk");
            // }

            // compute the new RHS for load factor schemes (on fine grid)
            fine_rhs.zeroValues();
            CUBLAS::axpy(load_factor, fine_loads, fine_rhs);
            CUBLAS::axpy(-1.0, fine_res, fine_rhs);
            fine_assembler.apply_bcs(fine_rhs);
            double rhs_norm = CUBLAS::get_vec_norm(fine_rhs);
            grids[0].setDefect(fine_rhs, perm_out);
            grids[0].zeroSolution();

            if (iload == 0 && inewton == 0) {
                // set the abs tol of outer K-cycle krylov solver to rtol * init norm here (this will have it do less work on inner solves)
                if (is_kcycle) {
                    kmg->outer_solver->set_abs_tol(rtol * rhs_norm);
                }
            }

            // debug the V-cycle process here
            // if (iload == 1 && inewton == 1) {

            //     // assume two grids only

            //     // 1) print the fine grid defect and vars
            //     int *d_perm = kmg->grids[0].d_perm;
            //     auto h_defect0 = kmg->grids[0].d_defect.createPermuteVec(6, d_perm).createHostVec();
            //     printToVTK<Assembler,HostVec<T>>(kmg->grids[0].assembler, h_defect0, "out/wing_defect0.vtk");
            //     auto h_vars0 = kmg->grids[0].d_vars.createPermuteVec(6, d_perm).createHostVec();
            //     printToVTK<Assembler,HostVec<T>>(kmg->grids[0].assembler, h_vars0, "out/wing_vars0.vtk");

            //     // V-cycle transfer fine grid defect to the coarse grid
            //     grids[1].restrict_defect(grids[0].d_defect);

            //     // 2) print the coarse grid defect and vars
            //     int *d_perm1 = kmg->grids[1].d_perm;
            //     auto h_defect1 = kmg->grids[1].d_defect.createPermuteVec(6, d_perm1).createHostVec();
            //     printToVTK<Assembler,HostVec<T>>(kmg->grids[0].assembler, h_defect1, "out/wing_defect1.vtk");
            //     auto h_vars1 = kmg->grids[1].d_vars.createPermuteVec(6, d_perm1).createHostVec();
            //     printToVTK<Assembler,HostVec<T>>(kmg->grids[0].assembler, h_vars1, "out/wing_vars1.vtk");

            //     // 3) do a coarse grid solve

            //     return;
            // }

            // solve the linear system using GMG solver for soln = u - u0 (and update variables)
            if (cycle_type == "V") {
                mg->vcycle_solve(0, pre_smooth, post_smooth, n_cycles, print, atol, rtol, double_smooth, print_freq, time); //(good option)
            } else if (cycle_type == "W") {
                mg->wcycle_solve(0, pre_smooth, post_smooth, n_cycles, print, atol, rtol);
            } else if (cycle_type == "F") {
                mg->fcycle_solve(0, pre_smooth, post_smooth, n_cycles, print, atol, rtol, double_smooth, print_freq, time); // also decent
            } else if (cycle_type == "K") {
                kmg->solve();
            }

            grids[0].getSolution(fine_soln, perm_out);
            double soln_norm = CUBLAS::get_vec_norm(fine_soln);
            CUBLAS::axpy(1.0, fine_soln, fine_vars);

            // compute the residual (much cheaper computation on GPU)
            fine_assembler.set_variables(fine_vars);
            fine_assembler.add_residual_fast(fine_res);
            fine_assembler.apply_bcs(fine_res);
            fine_rhs.zeroValues();
            CUBLAS::axpy(load_factor, fine_loads, fine_rhs);
            CUBLAS::axpy(-1.0, fine_res, fine_rhs);
            fine_assembler.apply_bcs(fine_rhs);
            double full_resid_norm = CUBLAS::get_vec_norm(fine_rhs);

            // check + report convergence metrics
            if (inewton == 0) {
                init_res = full_resid_norm;
            }
            // TODO : need residual check
            if (print) {
                printf("\tnewton step %d, rhs = %.4e, soln = %.4e\n", inewton, full_resid_norm,
                       soln_norm);
            }


            if (abs(full_resid_norm) < (abs_tol + rel_tol * init_res)) {
                break;
            }
        }  // end of newton loop

        // write out solution
        if (write_vtk) {
            auto h_vars = fine_vars.createHostVec();
            std::stringstream outputFile;
            outputFile << outputPrefix << iload << ".vtk";
            printToVTK<Assembler, HostVec<T>>(fine_assembler, h_vars, outputFile.str());
        }

    }  // end of load factor loop
    

    CHECK_CUDA(cudaDeviceSynchronize());
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = end1 - start1;
    int ndof = cycle_type == "K" ? kmg->grids[0].N : mg->grids[0].N;
    double total = startup_time.count() + solve_time.count();
    double mem_MB = is_kcycle ? kmg->get_memory_usage_mb() : mg->get_memory_usage_mb();
    printf("wingbox GMG solve, ndof %d : startup time %.2e, solve time %.2e, total %.2e, with mem(MB) %.2e\n", ndof, startup_time.count(), solve_time.count(), total, mem_MB);
    if (write_vtk) {
        printf("\n--------------------\nWARNING : VTK writes could affect timing!\n--------------------\n");
    }

    if (is_kcycle) {
        // double check with true resid nrm
        T resid_nrm = kmg->grids[0].getResidNorm();
        printf("init resid_nrm = %.2e => final resid_nrm = %.2e\n", init_resid_nrm, resid_nrm);

        // print some of the data of host residual
        int *d_perm = kmg->grids[0].d_perm;
        auto h_soln = kmg->grids[0].d_vars.createPermuteVec(6, d_perm).createHostVec();
        printToVTK<Assembler,HostVec<T>>(kmg->grids[0].assembler, h_soln, "out/aob_wing_mg.vtk");
    } else {
        // double check with true resid nrm
        T resid_nrm = mg->grids[0].getResidNorm();
        printf("init resid_nrm = %.2e => final resid_nrm = %.2e\n", init_resid_nrm, resid_nrm);

        // print some of the data of host residual
        int *d_perm = mg->grids[0].d_perm;
        auto h_soln = mg->grids[0].d_vars.createPermuteVec(6, d_perm).createHostVec();
        printToVTK<Assembler,HostVec<T>>(mg->grids[0].assembler, h_soln, "out/aob_wing_mg.vtk");
    }
}

template <typename T, class Assembler>
void solve_nonlinear_direct(MPI_Comm &comm, int level, double SR) {
  
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

  // TODO : set this in from optimized design from AOB case

  // BSR factorization
  auto& bsr_data = assembler.getBsrData();
  double fillin = 10.0;  // 10.0
  bool print = true;
  bsr_data.AMD_reordering();
  bsr_data.compute_full_LU_pattern(fillin, print);
  assembler.moveBsrDataToDevice();

  // get the loads
  int nvars = assembler.get_num_vars();
  int nnodes = assembler.get_num_nodes();
  HostVec<T> h_loads(nvars);
  int level_exp = 1 << level;
  double load_mag = 40.0 * (16.0 / level_exp / level_exp); // 4x the linear loads (so get more geomNL deflections)
  double SR1 = (300.0 / SR);
  double SR3 = SR1 * SR1 * SR1;
  load_mag *= SR3;
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

  // solve the linear system
  // newton solve
    int num_load_factors = 10, num_newton = 10;
    T min_load_factor = 1.0 / (num_load_factors + 1), max_load_factor = 1.0, abs_tol = 1e-8,
        rel_tol = 1e-8;
    auto solve_func = CUSPARSE::direct_LU_solve<T>;
    std::string outputPrefix = "out/wing_nl_";
    bool write_vtk = true;

    const bool fast_assembly = true;
    // const bool fast_assembly = false;
    newton_solve<T, BsrMat<DeviceVec<T>>, DeviceVec<T>, Assembler, fast_assembly>(
        solve_func, kmat, loads, soln, assembler, res, rhs, vars,
        num_load_factors, min_load_factor, max_load_factor, num_newton, abs_tol,
        rel_tol, outputPrefix, print, write_vtk);

  CHECK_CUDA(cudaDeviceSynchronize());
  auto end1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> solve_time = end1 - start1;
  std::chrono::duration<double> total_time = end1 - start0;

  size_t bytes_per_double = sizeof(double);
  double mem_mb = static_cast<double>(bytes_per_double) * static_cast<double>(bsr_data.nnzb) * 36.0 / 1024.0 / 1024.0;
  printf("Newton-Raphson, direct LU solves on #dof %d, uses memory(MB) %.2e\n", nvars, mem_mb);
  printf("\tassembly %.2e and ovr startup %.2e, solve time %.2e and total time %.2e (sec)\n", assemb_time.count(), startup_time.count(), solve_time.count(), total_time.count());
  if (write_vtk) {
        printf("\n--------------------\nWARNING : VTK writes could affect timing!\n--------------------\n");
    }

  // print some of the data of host residual
  auto h_soln = soln.createHostVec();
  printToVTK<Assembler, HostVec<T>>(assembler, h_soln, "out/aob_nl_direct_L" + std::to_string(level) + ".vtk");

  // free data
  assembler.free();
  h_loads.free();
  kmat.free();
  soln.free();
  res.free();
  vars.free();
  h_soln.free();
}

template <typename T, class Assembler>
void gatekeeper_method(bool is_multigrid, MPI_Comm &comm, int level, double SR, int nsmooth, int ninnercyc, std::string cycle_type) {
    if (is_multigrid) {
        solve_nonlinear_multigrid<T, Assembler>(comm, level, SR, nsmooth, ninnercyc, cycle_type);
    } else {
        solve_nonlinear_direct<T, Assembler>(comm, level, SR);
    }
}

int main(int argc, char **argv) {

    // Intialize MPI and declare communicator
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    // DEFAULTS
    int level = 2; // level mesh to solve.. level 4 also a good starting setting (big case)
    bool is_multigrid = true;
    // bool is_debug = false;
    double SR = 300.0;
    int nsmooth = 2; // may need more here (esp for MITC elements, but CFI can use less)
    int ninnercyc = 2; // inner V-cycles to precond K-cycle
    std::string cycle_type = "K"; // "V", "F", "W", "K"
    std::string elem_type = "CFI4"; // 'MITC4', 'CFI4', 'CFI9'

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
        using Basis = LagrangeQuadBasis<T, Quad, 2>;
        using Assembler = MITCShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
        gatekeeper_method<T, Assembler>(is_multigrid, comm, level, SR, nsmooth, ninnercyc, cycle_type);
    } else if (elem_type == "CFI4") {
        using Basis = ChebyshevQuadBasis<T, Quad, 1>;
        using Assembler = FullyIntegratedShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
        gatekeeper_method<T, Assembler>(is_multigrid, comm, level, SR, nsmooth, ninnercyc, cycle_type);
    } else if (elem_type == "CFI9") {
        using Basis = ChebyshevQuadBasis<T, Quad, 2>;
        using Assembler = FullyIntegratedShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
        gatekeeper_method<T, Assembler>(is_multigrid, comm, level, SR, nsmooth, ninnercyc, cycle_type);
    } else {
        printf("ERROR : didn't run anything, elem type not in available types (see main function)\n");
    }

    MPI_Finalize();
    return 0;
};
