
#include "linalg/_linalg.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"
#include "solvers/_solvers.h"

// shell imports
#include "assembler.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/shell_elem_group.h"

// local multigrid imports
#include "multigrid/grid.h"
#include "multigrid/fea.h"
// #include "multigrid/solvers/gmg.h"
#include <string>
#include <chrono>

// new multigrid imports for more object-oriented and hierarchical
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

void solve_linear_pcg_kcycle_gmg(MPI_Comm &comm, int level, double SR, int nsmooth, int ncycles) {
    // geometric multigrid method here..
    // need to make a number of grids..
    // level gives the finest level here..

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

    // old smoothers
    const SMOOTHER smoother = MULTICOLOR_GS_FAST2_JUNCTION;
    // const SCALER scaler = NONE; // relying on Krylov cycles to scale prolong now..
    const SCALER scaler = LINE_SEARCH; // inner V-cycle still need these updates

    const bool is_bsr = true; // need this one if want to smooth prolongation
    // const bool is_bsr = false; // no difference in intra-nodal (default old working prolong)
    using Prolongation = UnstructuredProlongation<Basis, is_bsr>; 
    using GRID = ShellGrid<Assembler, Prolongation, smoother, scaler>;

    // multigrid solver hierarhcy classes (new)
    using DirectSolve = CusparseMGDirectLU<GRID>;
    using KrylovSolve = PCGSolver<T, GRID>;
    using TwoLevelSolve = MultigridTwoLevelSolver<GRID>;
    using MG = MultilevelKcycleSolver<GRID, DirectSolve, TwoLevelSolve, KrylovSolve>;

    auto start0 = std::chrono::high_resolution_clock::now();
    auto mg = MG();
    GRID *fine_grid;

    // make each wing multigrid object..
    for (int i = level; i >= 0; i--) {

        // read the ESP/CAPS => nastran mesh for TACS
        TACSMeshLoader mesh_loader{comm};
        std::string fname = "meshes/aob_wing_L" + std::to_string(i) + ".bdf";
        mesh_loader.scanBDFFile(fname.c_str());
        double E = 70e9, nu = 0.3, thick = 2.0 / SR;  // material & thick properties (start thicker first try)
        // double E = 70e9, nu = 0.3, thick = 1.0;  // material & thick properties (start thicker first try)
        // double E = 70e9, nu = 0.3, thick = 0.01;  // material & thick properties (start thicker first try)
        // double E = 70e9, nu = 0.3, thick = 0.005;  // material & thick properties

        printf("making assembler+GMG for mesh '%s'\n", fname.c_str());
        
        // create the TACS Assembler from the mesh loader
        auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));

        // create the loads (really only needed on finer mesh.. TBD how to setup nonlinear case..)
        int nvars = assembler.get_num_vars();
        int nnodes = assembler.get_num_nodes();
        HostVec<T> h_loads(nvars);
        double load_mag = 10.0;
        double *my_loads = h_loads.getPtr();
        for (int inode = 0; inode < nnodes; inode++) {
            my_loads[6 * inode + 2] = load_mag;
        }

        // TODO : run optimized design from AOB case

        // make the grid
        bool full_LU = i == 0; // smallest grid is direct solve
        bool reorder = true;
        // printf("reorder %d\n", reorder);
        auto grid = *GRID::buildFromAssembler(assembler, my_loads, full_LU, reorder);
        mg.grids.push_back(grid); // add new grid
    }

    if (!Prolongation::structured) {
        printf("begin unstructured map\n");
        // int ELEM_MAX = 4; // for plate, cylinder
        int ELEM_MAX = 10; // for wingbox esp near rib, spar, OML junctions
        mg.template init_unstructured<Basis>(ELEM_MAX);
        printf("done with init unstructured\n");
        // return; // TEMP DEBUG
    }

    // now create the solvers hierarchy (mostly internally to MG class, but have to give it some options here..
    T omega = 1.5, rtol = 1e-6, atol = 1e-6;
    bool symmetric = false;
    // int print_freq = 5;
    int print_freq = 1;

    // main settings here.. with several options
    // ----------------------------------------------------

    // choice #1:
    // bool just_outer_krylov = false;
    bool just_outer_krylov = true; // good results.. want to try false setting too though

    int nvcyc_inner, nvcyc_outer, nkcyc_inner, nkcyc_outer;

    if (just_outer_krylov) {
        nvcyc_inner = 1, nvcyc_outer = ncycles, nkcyc_inner = 2, nkcyc_outer = 200;
    } else {
        // nvcyc_inner = 1, nvcyc_outer = ncycles, nkcyc_inner = 2, nkcyc_outer = 200;
        nvcyc_inner = ncycles, nvcyc_outer = ncycles, nkcyc_inner = ncycles, nkcyc_outer = 200; // cascades some (exponential inc num direct solves by levels) - high smoothing?
        // TODO : exploring other settings..
    }
    
    // create the kcycle multigrid object
    // ----------------------------------

    // apply settings
    auto inner_subspaceOptions = SolverOptions(omega, nsmooth, nvcyc_inner);
    auto outer_subspaceOptions = SolverOptions(omega, nsmooth, nvcyc_outer);
    auto innerKrylovOptions = SolverOptions(omega, 0, nkcyc_inner);
    auto outerKrylovOptions = SolverOptions(omega, 0, nkcyc_outer, symmetric, atol, rtol, print_freq);
    outerKrylovOptions.print = true;

    // if need more debug, can turn some other print statements on..
    // innerKrylovOptions.print = true;
    // innerKrylovOptions.debug = true;

    // NOTE : if scaler = NONE, need outer krylov on every level
    // otherwise if scaler = LINE_SEARCH, you can do krylov only on outer level
    // if (!just_outer_krylov && scaler == LINE_SEARCH) {
    //     printf("this is very inefficient, if doing krylov on every level, no need for line searches too. Double check if that is true though..\n");
    // }
    if (just_outer_krylov && scaler == NONE) {
        printf("warning should use scaler == LINE_SEARCH for inner V-cycles to solve right if only using krylov on outer level\n");
        return;
    }

    mg.init_solvers(inner_subspaceOptions, outer_subspaceOptions, innerKrylovOptions, outerKrylovOptions, just_outer_krylov);

    auto end0 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> startup_time = end0 - start0;

    T init_resid_nrm = mg.grids[0].getResidNorm();

    CHECK_CUDA(cudaDeviceSynchronize());
    auto start1 = std::chrono::high_resolution_clock::now();
    
    mg.solve(); // solves on previously defined internal defect and soln (there is also another version of the method that takes vectors in)

    CHECK_CUDA(cudaDeviceSynchronize());
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = end1 - start1;
    int ndof = mg.grids[0].N;
    double total = startup_time.count() + solve_time.count();
    double mem_MB = mg.get_memory_usage_mb();
    printf("wingbox PCG K-cycle with GMG precond solve:\n");
    printf("\tndof %d : startup time %.2e, solve time %.2e, total %.2e, with mem(MB) %.2e\n", ndof, startup_time.count(), solve_time.count(), total, mem_MB);

    // double check with true resid nrm
    T resid_nrm = mg.grids[0].getResidNorm();
    printf("init resid_nrm = %.2e => final resid_nrm = %.2e\n", init_resid_nrm, resid_nrm);

    // print some of the data of host residual
    int *d_perm = mg.grids[0].d_perm;
    auto h_soln = mg.grids[0].d_soln.createPermuteVec(6, d_perm).createHostVec();
    printToVTK<Assembler,HostVec<T>>(mg.grids[0].assembler, h_soln, "out/aob_wing_mg.vtk");
}

int main(int argc, char **argv) {

    // Intialize MPI and declare communicator
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    // DEFAULTS
    bool is_gmres = false;
    int level = 3; // level mesh to solve..
    double SR = 50.0;
    // int nsmooth = 4;
    int nsmooth = 6; // typically faster right now
    int ncycles = 4; // how many V-cycles for preconditioner..

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        char* arg = argv[i];
        to_lowercase(arg);

        if (strcmp(arg, "--sr") == 0) {
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
        } else if (strcmp(arg, "--nsmooth") == 0) {
            if (i + 1 < argc) {
                nsmooth = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --nsmooth\n";
                return 1;
            }
        } else if (strcmp(arg, "--ncycles") == 0) {
            if (i + 1 < argc) {
                ncycles = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing value for --nsmooth\n";
                return 1;
            }
        } else {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            std::cerr << "Usage: " << argv[0] << " [direct/mg] [--level int] [--SR double] [--nsmooth int]" << std::endl;
            return 1;
        }
    }

    if (is_gmres) {
        // solve_linear_gmres_kcycle_gmg(comm, level, SR, nsmooth, ncycles);
        return 0;
    } else {
        solve_linear_pcg_kcycle_gmg(comm, level, SR, nsmooth, ncycles);
    }
    

    MPI_Finalize();
    return 0;
};
