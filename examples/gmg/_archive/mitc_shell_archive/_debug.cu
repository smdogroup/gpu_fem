// general gpu_fem imports
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"

// shell imports
#include "assembler.h"
#include "element/shell/shell_elem_group.h"
#include "element/shell/physics/isotropic_shell.h"

// local multigrid imports
#include "include/grid.h"
#include "include/fea.h"
#include "include/mg.h"
#include <string>

/* command line args:
    [steps/mg] [--nxe int] [--SR float] [--nvcyc int] [--omega float] [--min_nxe int]
    * nxe must be power of 2

    examples:
    ./_debug.out steps --nxe 2048 --SR 100.0 --nvcyc 30   to run individual steps of mg solve on 2048 x 2048 elem grid with slenderness ratio 100 and 30 v-cycles
    ./_debug.out mg --nxe 2048 --SR 10.0    to run full multigrid solve on 2048 x 2048 elem grid with slenderness ratio 10
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

void multigrid_plate_debug(int nxe, double SR) {
    // geometric multigrid method, debug individual steps on single grid here..

    using T = double;   
    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = ShellQuadBasis<T, Quad, 2>;
    using Geo = Basis::Geo;
    constexpr bool has_ref_axis = false;
    constexpr bool is_nonlinear = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;
    using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
    using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;

    // multigrid objects
    // const SMOOTHER smoother = MULTICOLOR_GS;
    const SMOOTHER smoother = LEXIGRAPHIC_GS;

    // using Prolongation = StructuredProlongation<PLATE>;
    using Prolongation = StructuredProlongation<CYLINDER>;
    using GRID = ShellGrid<Assembler, Prolongation, smoother>;

    int nye = nxe;
    double Lx = 1.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 1.0 / SR, rho = 2500, ys = 350e6;
    int nxe_per_comp = nxe / 4, nye_per_comp = nye/4; // for now (should have 25 grids)
    auto assembler = createPlateAssembler<Assembler>(nxe, nye, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp);
    double Q = 1.0; // load magnitude
    T *my_loads = getPlateLoads<T, Physics>(nxe, nye, Lx, Ly, Q);

    // make the shell grid
    GRID *grid = GRID::buildFromAssembler(assembler, my_loads);

    // make a second grid here
    auto coarse_assembler = createPlateAssembler<Assembler>(nxe / 2, nxe / 2, Lx, Ly, E, nu, thick, rho, ys, 
        nxe_per_comp / 2, nye_per_comp / 2);
    T *my_coarse_loads = getPlateLoads<T, Physics>(nxe / 2, nye / 2, Lx, Ly, Q);

    // make the shell grid
    bool full_LU = true; // only use full LU pattern on coarse grid..
    GRID *coarse_grid = GRID::buildFromAssembler(coarse_assembler, my_coarse_loads, full_LU);

    // solve on coarse grid first..
    coarse_grid->direct_solve();
    auto h_coarse_soln = coarse_grid->d_soln.createPermuteVec(6, coarse_grid->Kmat.getPerm()).createHostVec();
    printToVTK<Assembler,HostVec<T>>(coarse_assembler, h_coarse_soln, "out/plate_coarse_direct.vtk");

    auto h_coarse_soln2 = coarse_grid->d_soln.createPermuteVec(6, coarse_grid->Kmat.getPerm()).createHostVec();
    printToVTK<Assembler,HostVec<T>>(coarse_assembler, h_coarse_soln2, "out/plate_coarse_direct2.vtk");

    // DEBUG
    // int *h_c_iperm = DeviceVec<int>(coarse_grid->nnodes, coarse_grid->d_iperm).createHostVec().getPtr();
    // printf("h_ coarse iperm: ");
    // printVec<int>(coarse_grid->nnodes, h_c_iperm);
    // return;

    // fine defect here..
    auto h_fine_defect = grid->d_defect.createPermuteVec(6, grid->Kmat.getPerm()).createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_fine_defect, "out/plate_fine_defect_0.vtk");

    // try prolongation
    grid->prolongate(coarse_grid->d_iperm, coarse_grid->d_soln);

    // print some of the data of host residual
    auto h_soln = grid->d_soln.createPermuteVec(6, grid->Kmat.getPerm()).createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_soln, "out/plate_cf_soln.vtk");

    // fine defect here..
    auto h_fine_defect1 = grid->d_defect.createPermuteVec(6, grid->Kmat.getPerm()).createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_fine_defect1, "out/plate_fine_defect_1.vtk");

    // // does printing soln again change it?
    // auto h_soln3 = grid->d_soln.createPermuteVec(6, grid->Kmat.getPerm()).createHostVec();
    // printToVTK<Assembler,HostVec<T>>(assembler, h_soln3, "out/plate_cf_soln2.vtk");

    // try defect restriction
    coarse_grid->restrict_defect(grid->nelems, grid->d_iperm,
                        grid->d_defect);

    // print some of the data of host residual
    auto h_coarse_defect = coarse_grid->d_defect.createPermuteVec(6, coarse_grid->Kmat.getPerm()).createHostVec();
    printToVTK<Assembler,HostVec<T>>(coarse_assembler, h_coarse_defect, "out/plate_fc_defect.vtk");

    // // try doing multicolor block-GS iterations here (precursor to doing multgrid first)
    // int n_iters = 3;
    int n_iters = 10;
    // int n_iters = 1000;
    bool print = true;
    int print_freq = 1;
    T omega = 1.0; // TODO : may need somewhat damping for higher SR?
    // T omega = 0.7; // only seem to need damping for very small DOF (for full solve, still smoothes otherwise)
    grid->multicolorBlockGaussSeidel_slow(n_iters, print, print_freq, omega);

    // print some of the data of host residual
    auto h_soln2 = grid->d_soln.createPermuteVec(6, grid->Kmat.getPerm()).createHostVec();
    printToVTK<Assembler,HostVec<T>>(assembler, h_soln2, "out/plate_mg.vtk");
}

void multigrid_full_solve(int nxe, double SR, int n_vcycles, double omega, int min_nxe) {
    // geometric multigrid method, running the full solve with VTK, printouts and more specific debugs
    // throughout

    using T = double;   
    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = ShellQuadBasis<T, Quad, 2>;
    using Geo = Basis::Geo;
    constexpr bool has_ref_axis = false;
    constexpr bool is_nonlinear = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;
    using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
    using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;

    // multigrid objects
    const SMOOTHER smoother = MULTICOLOR_GS;
    // const SMOOTHER smoother = LEXIGRAPHIC_GS;

    // using Prolongation = StructuredProlongation<PLATE>;
    using Prolongation = StructuredProlongation<CYLINDER>;
    using GRID = ShellGrid<Assembler, Prolongation, smoother>;
    using MG = ShellMultigrid<GRID>;

    // define min nxe size..
    // int min_nxe = 4;
    // int min_nxe = nxe / 2; // for two-grid

    // get num levels real quick..
    int n_levels = 0, nxe_copy = nxe;
    while (nxe_copy >= min_nxe) {
        nxe_copy /= 2;
        ++n_levels;
    }
    printf("nxe %d, n_levels %d\n", nxe, n_levels);
    // return;

    // printf("pre assemblers\n");
    auto mg = MG();
    // printf("post assemblers + MG init\n");

    // int run_case = 1; // plate
    int run_case = 2; // cylinder

    // make each grid
    int c_nxe = nxe;
    for (int i_level = 0; i_level < n_levels; i_level++) {
        Assembler assembler;
        T *my_loads;

        if (run_case == 1) {
            // make the assembler
            int c_nye = c_nxe;
            double Lx = 1.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 1.0 / SR, rho = 2500, ys = 350e6;
            int nxe_per_comp = c_nxe / 4, nye_per_comp = c_nye/4; // for now (should have 25 grids)
            // printf("assembler pre-vec\n");
            assembler = createPlateAssembler<Assembler>(c_nxe, c_nye, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp);
            // printf("assembler post-vec\n");
            double Q = 1.0; // load magnitude
            my_loads = getPlateLoads<T, Physics>(c_nxe, c_nye, Lx, Ly, Q);
        } else if (run_case == 2) {
            int c_nhe = c_nxe;
            double L = 1.0, R = 0.5, thick = L / SR;
            double E = 70e9, nu = 0.3;
            // double rho = 2500, ys = 350e6;
            bool imperfection = false; // option for geom imperfection
            int imp_x = 1, imp_hoop = 1; // no imperfection this input doesn't matter rn..
            assembler = createCylinderAssembler<Assembler>(c_nxe, c_nhe, L, R, E, nu, thick, imperfection, imp_x, imp_hoop);

            // get the loads
            constexpr bool compressive = false;
            double Q = 1.0; // load magnitude
            my_loads = getCylinderLoads<T, Physics, compressive>(c_nxe, c_nhe, L, R, Q);
        }
        

        bool full_LU = i_level == n_levels - 1; // only do full LU pattern on coarsest grid
        printf("making grid nxe %d, with full LU ?= %d\n", c_nxe, (int)full_LU);

        // make the grid
        // printf("making grid with nxe %d\n", c_nxe);
        bool reorder = smoother == MULTICOLOR_GS;

        auto grid = *GRID::buildFromAssembler(assembler, my_loads, full_LU, reorder);
        mg.grids.push_back(grid); // add new grid
        c_nxe /= 2;
    }
    // GRID *grids = mg.grids.data();

    printf("starting v cycle solve\n");
    // init defect nrm
    T init_defect_nrm = mg.grids[0].getDefectNorm();
    printf("V-cycles: ||init_defect|| = %.2e\n", init_defect_nrm);

    // int n_levels = mg.getNumLevels();
    // if (print) printf("n_levels %d\n", n_levels);
    T atol = 1e-6, rtol = 1e-6;
    // int pre_smooth = 1, post_smooth = 1;
    int pre_smooth = 2, post_smooth = 2;
    bool print = true;
    bool write = true;
    // bool write = false;
    
    bool debug = true;
    // bool debug = false; 

    // T omega = 1.0;
    // T omega = 0.8;

    if (write) {
        auto h_fine_defectn1 = mg.grids[0].d_defect.createPermuteVec(6, mg.grids[0].Kmat.getPerm()).createHostVec();
        printToVTK<Assembler,HostVec<T>>(mg.grids[0].assembler, h_fine_defectn1, "out/0_plate_fine_defect0.vtk");
    }
    int n_steps = n_vcycles;
    T fin_defect_nrm = init_defect_nrm;

    for (int i_vcycle = 0; i_vcycle < n_vcycles; i_vcycle++) {
        // printf("V cycle step %d\n", i_vcycle);
        std::string file_suffix = "_" + time_string(i_vcycle) + ".vtk";

        // auto h_fine_defectn1 = mg.grids[0].d_defect.createPermuteVec(6, mg.grids[0].Kmat.getIPerm()).createHostVec();
        // printToVTK<Assembler,HostVec<T>>(assemblers[0], h_fine_defectn1, "out/0_plate_fine_defect0.vtk");

        // go down each level smoothing and restricting until lowest level
        for (int i_level = 0; i_level < n_levels; i_level++) {

            std::string file_prefix = "out/level" + std::to_string(i_level) + "_";

            // if not last  (pre-smooth)
            if (i_level < n_levels - 1) {
                // if (print) printf("\tlevel %d pre-smooth\n", i_level);

                // prelim defect
                if (write) {
                    auto h_fine_defect00 = mg.grids[i_level].d_defect.createPermuteVec(6, mg.grids[i_level].Kmat.getPerm()).createHostVec();
                    T xpts_shift[3] = {1.5 * i_level, 0.0, 0.0};
                    printToVTKDEBUG<Assembler,HostVec<T>>(mg.grids[i_level].assembler, h_fine_defect00, file_prefix + "pre1_start" + file_suffix, xpts_shift);
                }

                // pre-smooth; TODO : do fast version later.. but let's demo with slow version
                // first
                printf("GS on level %d\n", i_level);
                mg.grids[i_level].smoothDefect(pre_smooth, print, pre_smooth - 1, omega);

                if (write) {
                    auto h_fine_defect00 = mg.grids[i_level].d_defect.createPermuteVec(6, mg.grids[i_level].Kmat.getPerm()).createHostVec();
                    T xpts_shift[3] = {1.5 * i_level, 0.0, 1.5};
                    printToVTKDEBUG<Assembler,HostVec<T>>(mg.grids[i_level].assembler, h_fine_defect00, file_prefix + "pre2_smooth" + file_suffix, xpts_shift);
                }

                // restrict defect
                printf("restrict defect from level %d => %d\n", i_level, i_level + 1);
                mg.grids[i_level + 1].restrict_defect(
                    mg.grids[i_level].nelems, mg.grids[i_level].d_iperm,
                    mg.grids[i_level].d_defect);
                CHECK_CUDA(cudaDeviceSynchronize()); // needed to make this work right?

                if (write) {
                    auto h_fine_defect00 = mg.grids[i_level+1].d_defect.createPermuteVec(6, mg.grids[i_level+1].Kmat.getPerm()).createHostVec();
                    T xpts_shift[3] = {1.5 * i_level, 0.0, 3.0};
                    printToVTKDEBUG<Assembler,HostVec<T>>(mg.grids[i_level+1].assembler, h_fine_defect00, file_prefix + "pre3_restrict" + file_suffix, xpts_shift);
                }

            } else {
                if (print) printf("\t--level %d full-solve\n", i_level);

                // prelim defect
                if (write) {
                    auto h_fine_defect00 = mg.grids[i_level].d_defect.createPermuteVec(6, mg.grids[i_level].Kmat.getPerm()).createHostVec();
                    T xpts_shift[3] = {1.5 * i_level, 0.0, 0.0};
                    printToVTKDEBUG<Assembler,HostVec<T>>(mg.grids[i_level].assembler, h_fine_defect00, file_prefix + "full_pre1_start" + file_suffix, xpts_shift);
                }

                // coarsest grid full solve
                mg.grids[i_level].direct_solve(false); // false for don't print

                // prelim soln
                if (write) {
                    auto h_soln = mg.grids[i_level].d_soln.createPermuteVec(6, mg.grids[i_level].Kmat.getPerm()).createHostVec();
                    T xpts_shift[3] = {1.5 * i_level, 0.0, 1.5};
                    printToVTKDEBUG<Assembler,HostVec<T>>(mg.grids[i_level].assembler, h_soln, file_prefix + "full_pre2_soln" + file_suffix, xpts_shift);
                }
            }
        }

        // now go back up the hierarchy
        for (int i_level = n_levels - 2; i_level >= 0; i_level--) {

            std::string file_prefix = "out/level" + std::to_string(i_level) + "_";

            // prelim defect
            if (write) {
                auto h_soln = mg.grids[i_level].d_defect.createPermuteVec(6, mg.grids[i_level].Kmat.getPerm()).createHostVec();
                T xpts_shift[3] = {1.5 * i_level, -1.5, 0.0};
                printToVTKDEBUG<Assembler,HostVec<T>>(mg.grids[i_level].assembler, h_soln, file_prefix + "post1_defect" + file_suffix, xpts_shift);
            }

            // get coarse-fine correction from coarser grid to this grid
            mg.grids[i_level].prolongate(mg.grids[i_level + 1].d_iperm, mg.grids[i_level + 1].d_soln, debug, file_prefix, file_suffix);
            // if (print) printf("\tlevel %d post-smooth\n", i_level);

            CHECK_CUDA(cudaDeviceSynchronize()); // TODO : needed to make this work right?, adding write statements improved conv..

            if (write) {
                auto h_soln = mg.grids[i_level].d_soln.createPermuteVec(6, mg.grids[i_level].Kmat.getPerm()).createHostVec();
                T xpts_shift[3] = {1.5 * i_level, -1.5, 6.0};
                printToVTKDEBUG<Assembler,HostVec<T>>(mg.grids[i_level].assembler, h_soln, file_prefix + "post5_soln" + file_suffix, xpts_shift);
            }

            // post-smooth
            bool rev_colors = true; // rev colors only on post not pre-smooth?
            // bool rev_colors = false;
            mg.grids[i_level].smoothDefect(post_smooth, print, post_smooth - 1, omega, rev_colors); 

            if (write) {
                auto h_defect = mg.grids[i_level].d_defect.createPermuteVec(6, mg.grids[i_level].Kmat.getPerm()).createHostVec();
                T xpts_shift[3] = {1.5 * i_level, -1.5, 7.5};
                printToVTKDEBUG<Assembler,HostVec<T>>(mg.grids[i_level].assembler, h_defect, file_prefix + "post6_postsmooth_defect" + file_suffix, xpts_shift);
            }
        }

        // compute fine grid defect of V-cycle
        T defect_nrm = mg.grids[0].getDefectNorm();
        printf("v-cycle step %d, ||defect|| = %.3e\n", i_vcycle, defect_nrm);
        fin_defect_nrm = defect_nrm;

        if (defect_nrm < atol + rtol * init_defect_nrm) {
            printf("V-cycle GMG converged in %d steps\n", i_vcycle + 1);
            n_steps = i_vcycle + 1;
            break;
        }
    }
    printf("done with v-cycle solve, conv %.2e to %.2e ||defect|| in %d steps\n", init_defect_nrm, fin_defect_nrm, n_steps);


    // print some of the data of host residual
    // int *d_iperm = mg.grids[0].Kmat.getIPerm();
    // auto h_soln = mg.grids[0].d_soln.createPermuteVec(6, d_iperm).createHostVec();
    // printToVTK<Assembler,HostVec<T>>(assemblers[0], h_soln, "out/plate_mg.vtk");
}

int main(int argc, char **argv) {
    // input ----------
    bool is_full_mg = true; // default
    int nxe = 128; // default value
    double SR = 50.0; // default
    int n_vcycles = 50; // default
    int min_nxe = 4; // default
    double omega = 1.0; // default (but go lower sometimes), maybe also try gauss seidel SOR

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        char* arg = argv[i];
        to_lowercase(arg);

        if (strcmp(arg, "steps") == 0) {
            is_full_mg = false;
        } else if (strcmp(arg, "mg") == 0) {
            is_full_mg = true;
        } else if (strcmp(arg, "--nxe") == 0) {
            if (i + 1 < argc) {
                nxe = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --nxe\n";
                return 1;
            }
        } else if (strcmp(arg, "--nvcyc") == 0) {
            if (i + 1 < argc) {
                n_vcycles = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --nvcyc\n";
                return 1;
            }
        } else if (strcmp(arg, "--min_nxe") == 0) {
            if (i + 1 < argc) {
                min_nxe = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --min_nxe\n";
                return 1;
            }
        } else if (strcmp(arg, "--sr") == 0) {
            if (i + 1 < argc) {
                SR = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --SR\n";
                return 1;
            }
        } else if (strcmp(arg, "--omega") == 0) {
            if (i + 1 < argc) {
                omega = std::atof(argv[++i]);
            } else {
                std::cerr << "Missing value for --omega\n";
                return 1;
            }
        } else {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            std::cerr << "Usage: " << argv[0] << " [direct/mg] [--nxe value] [--SR value]" << std::endl;
            return 1;
        }
    }

    // done reading arts, now run stuff
    if (is_full_mg) {
        multigrid_full_solve(nxe, SR, n_vcycles, omega, min_nxe);
    } else {
        multigrid_plate_debug(nxe, SR);
    }

    return 0;

    
}