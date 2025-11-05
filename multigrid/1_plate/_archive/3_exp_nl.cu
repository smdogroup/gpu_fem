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
#include "multigrid/smoothers/mc_smooth1.h"
#include "multigrid/prolongation/structured.h"
#include "multigrid/solvers/gmg.h"
#include <string>
#include <chrono>

// new multigrid imports for K-cycles, etc.
#include "multigrid/solvers/solve_utils.h"
#include "multigrid/solvers/direct/cusp_directLU.h"
#include "multigrid/solvers/krylov/bsr_pcg.h"
#include "multigrid/solvers/multilevel/kcycle.h"
#include "multigrid/solvers/multilevel/twolevel.h"

/* 
experimental scripts for faster nonlinear multigrid solves
*/


// NOTE : weird BCs might be slowing down the multigrid + nonlinear solver conv here (we get weird nonlinear direct convergence initially as well, something we don't see for wings)

void to_lowercase(char *str) {
    for (; *str; ++str) {
        *str = std::tolower(*str);
    }
}

template <typename T, class Assembler>
void multigrid_solve(int nxe, double SR, int nsmooth, int ninnercyc, std::string cycle_type, T load_mag = 5.0e7) {
    // geometric multigrid method here..
    // need to make a number of grids..
    using Basis = typename Assembler::Basis;
    using Physics = typename Assembler::Phys;
    const SCALER scaler  = LINE_SEARCH;
    using Smoother = MulticolorGSSmoother_V1<Assembler>;
    using Prolongation = StructuredProlongation<Assembler, PLATE>;
    using GRID = SingleGrid<Assembler, Prolongation, Smoother, scaler>;
    using CoarseSolver = CusparseMGDirectLU<T, Assembler>;
    using MG = GeometricMultigridSolver<GRID, CoarseSolver>;

    // for K-cycles
    using KrylovSolve = PCGSolver<T, GRID>;
    using TwoLevelSolve = MultigridTwoLevelSolver<GRID>;
    using KMG = MultilevelKcycleSolver<GRID, CoarseSolver, TwoLevelSolve, KrylovSolve>;

    CHECK_CUDA(cudaDeviceSynchronize());
    auto start0 = std::chrono::high_resolution_clock::now();
    
    MG *mg;
    KMG *kmg;

    bool is_kcycle = cycle_type == "K";
    if (is_kcycle) {
        kmg = new KMG();
    } else {
        mg = new MG();
    }

    // get nxe_min for not exactly power of 2 case
    int pre_nxe_min = nxe > 32 ? 32 : 4;
    int nxe_min = pre_nxe_min;
    for (int c_nxe = nxe; c_nxe >= pre_nxe_min; c_nxe /= 2) {
        nxe_min = c_nxe;
    }

    // make each grid
    for (int c_nxe = nxe; c_nxe >= nxe_min; c_nxe /= 2) {
        // make the assembler
        int c_nye = c_nxe;
        double Lx = 1.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 1.0 / SR, rho = 2500, ys = 350e6;
        int nxe_per_comp = c_nxe / 4, nye_per_comp = c_nye/4; // for now (should have 25 grids)
        auto assembler = createPlateAssembler<Assembler>(c_nxe, c_nye, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp);
        double Q = load_mag / (c_nxe+1) / (c_nye + 1);
        Q *= (100.0 / SR) * (100.0 / SR) * (100.0 / SR);
        // T *my_loads = getPlatePointLoad<T, Physics>(c_nxe, c_nye, Lx, Ly, Q);
        T *my_loads = getPlateLoads<T, Physics>(c_nxe, c_nye, Lx, Ly, Q);
        // double in_plane_frac = 0.3;
        // T *my_loads = getPlateNonlinearLoads<T, Physics>(c_nxe, c_nye, Lx, Ly, Q, in_plane_frac);
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
        T omega = 1.5; // for GS-SOR
        // T omega = 0.7; // under-relaxed for better NL conv?
        auto smoother = new Smoother(assembler, kmat, h_color_rowp, omega);
        auto prolongation = new Prolongation(assembler);
        auto grid = GRID(assembler, prolongation, smoother, kmat, loads);
        
        if (is_kcycle) {
            kmg->grids.push_back(grid);
        } else {
            mg->grids.push_back(grid);
            if (full_LU) mg->coarse_solver = new CoarseSolver(assembler, kmat);
        }
    }

    // register the coarse assemblers to the prolongations..
    if (is_kcycle) {
        kmg->template init_prolongations<Basis>();
    } else {
        mg->template init_prolongations<Basis>();
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    auto end0 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> startup_time = end0 - start0;

    T init_resid_nrm = is_kcycle ? kmg->grids[0].getResidNorm() : mg->grids[0].getResidNorm();

    CHECK_CUDA(cudaDeviceSynchronize());
    auto start1 = std::chrono::high_resolution_clock::now();

    int pre_smooth = nsmooth, post_smooth = nsmooth; // need a little extra smoothing on cylinder (compare to plate).. (cause of curvature I think..)
    // bool print = true;
    bool print = false;
    T omega2 = 1.5;
    T atol = 1e-6, rtol = 1e-6;
    bool double_smooth = true; // twice as many smoothing steps at lower levels (similar cost, better conv?)

    int n_cycles = 500; // max # cycles
    int print_freq = 3;

    if (is_kcycle) {
        int n_krylov = 500;
        kmg->init_outer_solver(nsmooth, ninnercyc, n_krylov, omega2, atol, rtol, print_freq, print, double_smooth);    
    }

    std::vector<GRID>& grids = kmg->grids;

    // ---------------------------------------------------
    // 1) demo restrict fine to coarse soln

    // // first solve on fine grid (with initial linear defect)
    // kmg->solve();

    // // // now pass soln down to the coarse grid
    // grids[1].restrict_soln(grids[0].d_soln);

    // int *d_perm = kmg->grids[0].d_perm;
    // auto h_soln = kmg->grids[0].d_soln.createPermuteVec(6, d_perm).createHostVec();
    // printToVTK<Assembler,HostVec<T>>(kmg->grids[0].assembler, h_soln, "out/plate_lin0.vtk");

    // int *d_perm1 = kmg->grids[1].d_perm;
    // auto h_soln1 = kmg->grids[1].d_soln.createPermuteVec(6, d_perm1).createHostVec();
    // printToVTK<Assembler,HostVec<T>>(kmg->grids[1].assembler, h_soln1, "out/plate_lin1.vtk");

    // -----------------------------------------------------------
    // 2) actually try Newton-mg solve here (this is just V1, later versions may use FMG cycle so less extra work needs to be done on fine grids)
    //     i.e. you can do most of hte nonlinear solves to get in basin of attraction on coarser grids first.. (then nonlinear fine grid at end only, or some FMG cycle)

    // fine grid states
    auto& fine_assembler = grids[0].assembler;
    auto fine_soln = fine_assembler.createVarsVec();
    auto fine_res = fine_assembler.createVarsVec();
    auto fine_rhs = fine_assembler.createVarsVec();
    auto fine_loads = fine_assembler.createVarsVec();
    auto fine_vars = fine_assembler.createVarsVec();
    auto& fine_kmat = grids[0].Kmat;
    bool perm_out = true; // get fine grid loads
    grids[0].getDefect(fine_loads, perm_out);

    // first restrict the defect down to the coarsest grid
    int numLevels = is_kcycle ? kmg->getNumLevels() : mg->getNumLevels();
    for (int ilevel = 1; ilevel < numLevels; ilevel++) {
        grids[ilevel].restrict_defect(grids[ilevel-1].d_defect);
    }
    auto& coarse_assembler = grids[numLevels-1].assembler;
    auto coarse_soln = coarse_assembler.createVarsVec();
    auto coarse_res = coarse_assembler.createVarsVec();
    auto coarse_rhs = coarse_assembler.createVarsVec();
    auto coarse_loads = coarse_assembler.createVarsVec();
    auto coarse_vars = coarse_assembler.createVarsVec();
    auto& coarse_kmat = grids[numLevels-1].Kmat;
    grids[numLevels-1].getDefect(coarse_loads, perm_out);

    // now solve with Newton raphson on the coarse grid
    int num_load_factors = 50, num_newton = 10;
    T min_load_factor = 1.0 / (num_load_factors - 1), max_load_factor = 1.0, abs_tol = 1e-8,
        rel_tol = 1e-8;
    auto solve_func = CUSPARSE::direct_LU_solve<T>;
    std::string outputPrefix = "out/plate_nl_mg_";
    bool write_vtk = false;
    const bool fast_assembly = true;
    // const bool fast_assembly = false;
    bool coarse_print = true;
    printf("\n-------------------------\n");
    printf("Coarse grid Newton-Raphson load-step solve\n");
    printf("\n-------------------------\n");
    newton_solve<T, BsrMat<DeviceVec<T>>, DeviceVec<T>, Assembler, fast_assembly>(
        solve_func, coarse_kmat, coarse_loads, coarse_soln, coarse_assembler, coarse_res, coarse_rhs, coarse_vars,
        num_load_factors, min_load_factor, max_load_factor, num_newton, abs_tol,
        rel_tol, outputPrefix, coarse_print, write_vtk);
    printf("\n-------------------------\n");
    printf("Finished coarse grid Newton-Raphson load-step solve\n");
    printf("\n-------------------------\n");

    // now prolongate solution back up to fine grid (from coarse grid newton-raphson)
    grids[numLevels-1].setSolution(coarse_soln, perm_out);
    for (int ilevel = numLevels-1; ilevel >= 1; ilevel--) {
        grids[ilevel-1].prolongation->prolongate(grids[ilevel].d_soln, grids[ilevel-1].d_soln);
        
        // apply bcs to prolong
        grids[ilevel-1].d_soln.permuteData(6, grids[ilevel-1].d_perm);  // better way to do this later?
        grids[ilevel-1].assembler.apply_bcs(grids[ilevel-1].d_soln);
        grids[ilevel-1].d_soln.permuteData(6, grids[ilevel-1].d_iperm);  // better way to do this later?
    }
    grids[0].getSolution(fine_vars, perm_out);

    // need some preliminary nonlinear smoothing..
    // ---------------------------------------------

    // or some FAS or FMG iterations..
    

    // // print the coarse and fine grid solutions (Debug)
    // // ------------------------------------------------------
    // int *d_perm = kmg->grids[0].d_perm;
    // auto h_soln = kmg->grids[0].d_soln.createPermuteVec(6, d_perm).createHostVec();
    // printToVTK<Assembler,HostVec<T>>(kmg->grids[0].assembler, h_soln, "out/plate_nlf.vtk");

    // int *d_perm1 = kmg->grids[numLevels-1].d_perm;
    // auto h_soln1 = kmg->grids[numLevels-1].d_soln.createPermuteVec(6, d_perm1).createHostVec();
    // printToVTK<Assembler,HostVec<T>>(coarse_assembler, h_soln1, "out/plate_nlc.vtk");
    // return;
    

    // for (int iload = 0; iload < num_load_factors; iload++) {
    //     T load_factor =
    //         min_load_factor + (max_load_factor - min_load_factor) * iload / (num_load_factors - 1);

    //     T init_res = 1e50;
    //     if (print) {
    //         printf("load step %d / %d : load factor %.4e\n", iload, num_load_factors, load_factor);
    //     }
    T load_factor = 1.0; // no load factor steps at first..
    T init_res = 1e50;
    int iload = 0;
    printf("Newton-MG fine iterations\n");

    for (int inewton = 0; inewton < num_newton; inewton++) {

        // update the fine grid stiffness matrix and residual
        fine_assembler.set_variables(fine_vars);
        fine_assembler.add_jacobian_fast(fine_kmat);
        fine_assembler.add_residual_fast(fine_res);
        fine_assembler.apply_bcs(fine_res);
        fine_assembler.apply_bcs(fine_kmat);

        // pass current states to coarse grids and update their assemblies
        grids[0].setStateVars(fine_vars); // set vars into finest grid (so we can pass this down to coarser grids)
        kmg->update_coarse_grid_states(); // restrict state variables to coarse grids
        kmg->update_coarse_grid_jacobians(); // compute coarse grid NL stiffness matrices
        kmg->update_after_assembly(); // updates any dependent matrices like Dinv

        // if (iload == 1 && inewton == 1) {

        //     break;
        //     // // show the solution on the coarse grid
        //     // int *d_perm = kmg->grids[0].d_perm;
        //     // auto h_soln = kmg->grids[0].d_soln.createPermuteVec(6, d_perm).createHostVec();
        //     // printToVTK<Assembler,HostVec<T>>(kmg->grids[0].assembler, h_soln, "out/plate_nl_debug0.vtk");

        //     // int *d_perm1 = kmg->grids[1].d_perm;
        //     // auto h_soln1 = kmg->grids[1].d_soln.createPermuteVec(6, d_perm1).createHostVec();
        //     // printToVTK<Assembler,HostVec<T>>(kmg->grids[1].assembler, h_soln1, "out/plate_nl_debug1.vtk");
        // }

        // compute the new RHS for load factor schemes (on fine grid)
        fine_rhs.zeroValues();
        CUBLAS::axpy(load_factor, fine_loads, fine_rhs);
        CUBLAS::axpy(-1.0, fine_res, fine_rhs);
        fine_assembler.apply_bcs(fine_rhs);
        double rhs_norm = CUBLAS::get_vec_norm(fine_rhs);
        grids[0].setDefect(fine_rhs, perm_out);
        grids[0].zeroSolution();

        // solve the linear system using GMG solver for soln = u - u0 (and update variables)
        kmg->solve();
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

    // }  // end of load factor loop

}

template <typename T, class Assembler>
void solve_direct(int nxe, double SR, T load_mag = 5.0e7) {

    using Basis = typename Assembler::Basis;
    using Physics = typename Assembler::Phys;

    int nye = nxe;
    double Lx = 1.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 0.005, rho = 2500, ys = 350e6;
    int nxe_per_comp = nxe / 4, nye_per_comp = nye/4; // for now (should have 25 grids)
    auto assembler = createPlateAssembler<Assembler>(nxe, nye, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp);

    // BSR factorization
    auto& bsr_data = assembler.getBsrData();
    double fillin = 10.0;  // 10.0
    bool print = true;
    bsr_data.AMD_reordering();
    bsr_data.compute_full_LU_pattern(fillin, print);
    assembler.moveBsrDataToDevice();

    // get plate loads
    double Q = load_mag / (nxe + 1) / (nxe + 1);
    Q *= (100.0 / SR) * (100.0 / SR) * (100.0 / SR);
    // T *my_loads = getPlatePointLoad<T, Physics>(c_nxe, c_nye, Lx, Ly, Q);
    T *my_loads = getPlateLoads<T, Physics>(nxe, nye, Lx, Ly, Q);

    // double Q = 1.0e5;
    // T *my_loads = getPlatePointLoad<T, Physics>(nxe, nye, Lx, Ly, Q);
    // double in_plane_frac = 0.3;
    // T *my_loads = getPlateNonlinearLoads<T, Physics>(nxe, nye, Lx, Ly, Q, in_plane_frac);
    // T *my_loads = getPlateLoads<T, Physics>(nxe, nye, Lx, Ly, Q);
    auto loads = assembler.createVarsVec(my_loads);
    assembler.apply_bcs(loads);

    // setup kmat and initial vecs
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto soln = assembler.createVarsVec();
    auto res = assembler.createVarsVec();
    auto rhs = assembler.createVarsVec();
    auto vars = assembler.createVarsVec();

    // newton solve
    int num_load_factors = 50, num_newton = 10;
    T min_load_factor = 1.0 / (num_load_factors - 1), max_load_factor = 1.0, abs_tol = 1e-8,
        rel_tol = 1e-8;
    auto solve_func = CUSPARSE::direct_LU_solve<T>;
    std::string outputPrefix = "out/plate_";
    bool write_vtk = true;

    const bool fast_assembly = true;
    // const bool fast_assembly = false;
    newton_solve<T, BsrMat<DeviceVec<T>>, DeviceVec<T>, Assembler, fast_assembly>(
        solve_func, kmat, loads, soln, assembler, res, rhs, vars,
        num_load_factors, min_load_factor, max_load_factor, num_newton, abs_tol,
        rel_tol, outputPrefix, print, write_vtk);

    // print some of the data of host residual
    auto h_soln = soln.createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_soln, "out/plate_nl.vtk");

}

template <typename T, class Assembler>
void gatekeeper_method(bool is_multigrid, int nxe, double SR, int nsmooth, int ninnercyc, std::string cycle_type, T load_mag = 5.0e7) {
    if (is_multigrid) {
        multigrid_solve<T, Assembler>(nxe, SR, nsmooth, ninnercyc, cycle_type, load_mag);
    } else {
        solve_direct<T, Assembler>(nxe, SR, load_mag);
    }
}

int main(int argc, char **argv) {
    // input ----------
    bool is_multigrid = true;
    int nxe = 128; // default value (three grids)
    double SR = 100.0; // default
    int n_vcycles = 50;
    double load_mag = 5.0e7;

    int nsmooth = 2; // typically faster right now
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
        } else if (strcmp(arg, "--nxe") == 0) {
            if (i + 1 < argc) {
                nxe = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --nxe\n";
                return 1;
            }
        }  else if (strcmp(arg, "--sr") == 0) {
            if (i + 1 < argc) {
                SR = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --SR\n";
                return 1;
            }
        } else if (strcmp(arg, "--load") == 0) {
            if (i + 1 < argc) {
                load_mag = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --load\n";
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
            std::cerr << "Usage: " << argv[0] << " [direct/mg] [--nxe value] [--SR value] [--cycle char] [--nsmooth int] [--ninnercyc int]" << std::endl;
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

    printf("plate mesh with geomNL %s elements, nxe %d and SR %.2e\n------------\n", elem_type.c_str(), nxe, SR);
    if (elem_type == "MITC4") {
        using Basis = LagrangeQuadBasis<T, Quad, 2>;
        using Assembler = MITCShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
        gatekeeper_method<T, Assembler>(is_multigrid, nxe, SR, nsmooth, ninnercyc, cycle_type, load_mag);
    } else if (elem_type == "CFI4") {
        using Basis = ChebyshevQuadBasis<T, Quad, 1>;
        using Assembler = FullyIntegratedShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
        gatekeeper_method<T, Assembler>(is_multigrid, nxe, SR, nsmooth, ninnercyc, cycle_type, load_mag);
    } else if (elem_type == "CFI9") {
        using Basis = ChebyshevQuadBasis<T, Quad, 2>;
        using Assembler = FullyIntegratedShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
        gatekeeper_method<T, Assembler>(is_multigrid, nxe, SR, nsmooth, ninnercyc, cycle_type, load_mag);
    } else {
        printf("ERROR : didn't run anything, elem type not in available types (see main function)\n");
    }
    

    return 0;

    
}