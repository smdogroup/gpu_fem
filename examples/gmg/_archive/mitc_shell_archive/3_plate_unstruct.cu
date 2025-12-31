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
#include "multigrid/grid.h"
#include "multigrid/fea.h"
#include "multigrid/mg.h"
#include <string>
#include <chrono>

/* command line args:
    [--nxe int] [--SR float] [--nvcyc int]
    * nxe must be power of 2

    examples:
    ./1_plate.out direct --nxe 2048 --SR 100.0    to run direct plate solve on 2048 x 2048 elem grid with slenderness ratio 100
    ./1_plate.out mg --nxe 2048 --SR 100.0    to run geometric multigrid plate solve on 2048 x 2048 elem grid with slenderness ratio 100
*/

void to_lowercase(char *str) {
    for (; *str; ++str) {
        *str = std::tolower(*str);
    }
}

void multigrid_plate_solve(int nxe, double SR, int n_vcycles) {
    // geometric multigrid method here..
    // need to make a number of grids..

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
    const SMOOTHER smoother = MULTICOLOR_GS_FAST;
    // const SMOOTHER smoother = LEXIGRAPHIC_GS;

    // using Prolongation = StructuredProlongation<PLATE>;
    using Prolongation = UnstructuredProlongation<Basis>;

    using GRID = ShellGrid<Assembler, Prolongation, smoother>;
    using MG = ShellMultigrid<GRID>;

    auto start0 = std::chrono::high_resolution_clock::now();
    auto mg = MG();

    int nxe_min = nxe > 32 ? 32 : 4;
    // int nxe_min = nxe / 2; // two level
    // int nxe_min = 4;
    // int nxe_min = 16;

    // make each grid
    for (int c_nxe = nxe; c_nxe >= nxe_min; c_nxe /= 2) {
        // make the assembler
        int c_nye = c_nxe;
        double Lx = 1.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 1.0 / SR, rho = 2500, ys = 350e6;
        int nxe_per_comp = c_nxe / 4, nye_per_comp = c_nye/4; // for now (should have 25 grids)
        auto assembler = createPlateAssembler<Assembler>(c_nxe, c_nye, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp);
        double Q = 1.0; // load magnitude
        T *my_loads = getPlateLoads<T, Physics>(c_nxe, c_nye, Lx, Ly, Q);
        printf("making grid with nxe %d\n", c_nxe);

        // make the grid
        bool full_LU = c_nxe == nxe_min; // smallest grid is direct solve
        bool reorder;
        if (smoother == LEXIGRAPHIC_GS) {
            reorder = false;
        } else if (smoother == MULTICOLOR_GS || smoother == MULTICOLOR_GS_FAST || smoother == MULTICOLOR_GS_FAST2) {
            reorder = true;
        }
        auto grid = *GRID::buildFromAssembler(assembler, my_loads, full_LU, reorder);
        mg.grids.push_back(grid); // add new grid
    }

    if (!Prolongation::structured) {
        mg.template init_unstructured<Basis>();
        // return; // TEMP DEBUG
    }

    // // -------------------------------
    // TEMP DEBUG unstructured
    // int *d_perm1 = mg.grids[0].d_perm;

    // mg.grids[1].direct_solve(false);
    // mg.grids[0].prolongate(mg.grids[1].d_iperm, mg.grids[1].d_soln);
    // auto h_soln1 = mg.grids[0].d_temp_vec.createPermuteVec(6, d_perm1).createHostVec();
    // printToVTK<Assembler,HostVec<T>>(mg.grids[0].assembler, h_soln1, "out/plate_mg_cf.vtk");

    // // plot orig fine defect
    // auto h_fdef = mg.grids[0].d_defect.createPermuteVec(6, d_perm1).createHostVec();
    // printToVTK<Assembler,HostVec<T>>(mg.grids[0].assembler, h_fdef, "out/plate_mg_fine_defect.vtk");

    // // now try restrict defect
    // mg.grids[1].restrict_defect(
    //                     mg.grids[0].nelems, mg.grids[0].d_iperm, mg.grids[0].d_defect);
    // int *d_perm2 = mg.grids[1].d_perm;
    // auto h_def2 = mg.grids[1].d_defect.createPermuteVec(6, d_perm2).createHostVec();
    // printToVTK<Assembler,HostVec<T>>(mg.grids[1].assembler, h_def2, "out/plate_mg_restrict.vtk");
    // return;
    // // -------------------------------

    auto end0 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> startup_time = end0 - start0;

    T init_resid_nrm = mg.grids[0].getResidNorm();

    auto start1 = std::chrono::high_resolution_clock::now();
    printf("starting v cycle solve\n");
    int pre_smooth = 1, post_smooth = 1;
    // int pre_smooth = 2, post_smooth = 2;
    // int pre_smooth = 4, post_smooth = 4;
    // bool print = false;
    bool print = false;
    T atol = 1e-6, rtol = 1e-6;
    T omega = 1.0;
    bool double_smooth = true; // false
    mg.vcycle_solve(pre_smooth, post_smooth, n_vcycles, print, atol, rtol, omega, double_smooth);
    printf("done with v-cycle solve\n");

    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = end1 - start1;
    int ndof = mg.grids[0].N;
    double total = startup_time.count() + solve_time.count();
    printf("plate GMG solve, ndof %d : startup time %.2e, solve time %.2e, total %.2e\n", ndof, startup_time.count(), solve_time.count(), total);

    // double check with true resid nrm
    T resid_nrm = mg.grids[0].getResidNorm();
    printf("init resid_nrm = %.2e => final resid_nrm = %.2e\n", init_resid_nrm, resid_nrm);

    // print some of the data of host residual
    int *d_perm = mg.grids[0].d_perm;
    auto h_soln = mg.grids[0].d_soln.createPermuteVec(6, d_perm).createHostVec();
    printToVTK<Assembler,HostVec<T>>(mg.grids[0].assembler, h_soln, "out/plate_mg.vtk");
}

int main(int argc, char **argv) {
    // input ----------
    int nxe = 256; // default value
    double SR = 100.0; // default
    int n_vcycles = 50;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        char* arg = argv[i];
        to_lowercase(arg);

        if (strcmp(arg, "--nxe") == 0) {
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
        } else if (strcmp(arg, "--nvcyc") == 0) {
            if (i + 1 < argc) {
                n_vcycles = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --nvcyc\n";
                return 1;
            }
        } else {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            std::cerr << "Usage: " << argv[0] << " [direct/mg] [--nxe value] [--SR value]" << std::endl;
            return 1;
        }
    }

    // done reading arts, now run stuff
    multigrid_plate_solve(nxe, SR, n_vcycles);

    return 0;

    
}