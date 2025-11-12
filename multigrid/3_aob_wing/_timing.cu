// general gpu_fem imports

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


void to_lowercase(char *str) {
    for (; *str; ++str) {
        *str = std::tolower(*str);
    }
}

template <typename T, class Assembler>
void time_wing_multigrid(MPI_Comm &comm, int level, double SR, int nsmooth, int ninnercyc, std::string cycle_type, bool print_all_times = false) {
    /* timing / profiling of the wing multigrid case*/

    using Basis = typename Assembler::Basis;
    using Physics = typename Assembler::Phys;
    using Data = typename Physics::Data;
    using Smoother = MulticolorGSSmoother_V1<Assembler>;
    // const bool is_bsr = true; // need this one if want to smooth prolongation
    const bool is_bsr = false; // no difference in intra-nodal (default old working prolong)
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
        CHECK_CUDA(cudaDeviceSynchronize());
        auto start_00 = std::chrono::high_resolution_clock::now();
        TACSMeshLoader mesh_loader{comm};
        std::string fname = "meshes/aob_wing_L" + std::to_string(i) + ".bdf";
        printf("making assembler+GMG times for mesh '%s': \n", fname.c_str());
        mesh_loader.scanBDFFile(fname.c_str());
        double E = 70e9, nu = 0.3, thick = 2.0 / SR;  // material & thick properties (start thicker first try)
        CHECK_CUDA(cudaDeviceSynchronize());
        auto end_00 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> read_mesh_time = end_00 - start_00;
        if (print_all_times) printf("\tread mesh %.2e, ", read_mesh_time.count());

        // create the wing assembler
        CHECK_CUDA(cudaDeviceSynchronize());
        auto start_01 = std::chrono::high_resolution_clock::now();
        auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));
        CHECK_CUDA(cudaDeviceSynchronize());
        auto end_01 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> create_assemb_time = end_01 - start_01;
        if (print_all_times) printf("create assembler %.2e\n", create_assemb_time.count());

        // do the reordering
        
        auto &bsr_data = assembler.getBsrData();
        int num_colors = 0, *_color_rowp;
        bool coarsest_grid = i == 0;
        if (!coarsest_grid) {
            
            CHECK_CUDA(cudaDeviceSynchronize());
            auto start_02 = std::chrono::high_resolution_clock::now();
            WingboxMultiColoring<Assembler>::apply_coloring(assembler, bsr_data, num_colors, _color_rowp);
            CHECK_CUDA(cudaDeviceSynchronize());
            auto end_02 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> reorder_time = end_02 - start_02;
            if (print_all_times) printf("\tmulticolor reorder %.2e, ", reorder_time.count());

            CHECK_CUDA(cudaDeviceSynchronize());
            auto start_03 = std::chrono::high_resolution_clock::now();
            bsr_data.compute_nofill_pattern();
            CHECK_CUDA(cudaDeviceSynchronize());
            auto end_03 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> nofill_time = end_03 - start_03;
            if (print_all_times) printf("nofill pattern %.2e, ", nofill_time.count());
            
        } else {

            CHECK_CUDA(cudaDeviceSynchronize());
            auto start_02 = std::chrono::high_resolution_clock::now();
            bsr_data.AMD_reordering();
            CHECK_CUDA(cudaDeviceSynchronize());
            auto end_02 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> reorder_time = end_02 - start_02;
            if (print_all_times) printf("\tAMD reorder %.2e, ", reorder_time.count());

            CHECK_CUDA(cudaDeviceSynchronize());
            auto start_03 = std::chrono::high_resolution_clock::now();
            bsr_data.compute_full_LU_pattern(10.0, false);
            CHECK_CUDA(cudaDeviceSynchronize());
            auto end_03 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> full_LU_time = end_03 - start_03;
            if (print_all_times) printf("full LU pattern %.2e, ", full_LU_time.count());
        }      
        

        // move bsr data to device
        CHECK_CUDA(cudaDeviceSynchronize());
        auto start_04 = std::chrono::high_resolution_clock::now();
        auto h_color_rowp = HostVec<int>(num_colors + 1, _color_rowp);
        assembler.moveBsrDataToDevice();
        CHECK_CUDA(cudaDeviceSynchronize());
        auto end_04 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> move_bsr_time = end_04 - start_04;
        if (print_all_times) printf("move bsrData to device %2e\n", move_bsr_time.count());

        // compute laods and misc
        CHECK_CUDA(cudaDeviceSynchronize());
        auto start_05 = std::chrono::high_resolution_clock::now();
        int nvars = assembler.get_num_vars();
        int nnodes = assembler.get_num_nodes();
        HostVec<T> h_loads(nvars);
        double load_mag = 10.0;
        double *my_loads = h_loads.getPtr();
        for (int inode = 0; inode < nnodes; inode++) {
            my_loads[6 * inode + 2] = load_mag;
        }
        auto loads = assembler.createVarsVec(my_loads);
        assembler.apply_bcs(loads);
        auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
        auto res = assembler.createVarsVec();
        CHECK_CUDA(cudaDeviceSynchronize());
        auto end_05 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> loads_and_misc_time = end_05 - start_05;
        
        // assemble kmat time (TODO : put new methods in here)
        CHECK_CUDA(cudaDeviceSynchronize());
        auto start_06 = std::chrono::high_resolution_clock::now();
        // assembler.add_jacobian(res, kmat);
        assembler.add_jacobian_fast(kmat);
        // assembler.apply_bcs(res);
        assembler.apply_bcs(kmat); // apply bcs should be very small part.. nothing
        CHECK_CUDA(cudaDeviceSynchronize());
        auto end_06 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> assemb_kmat_time = end_06 - start_06;
        printf("\tassemble kmat %.2e, ", assemb_kmat_time.count());
        if (print_all_times) printf("loads + misc %.2e\n", loads_and_misc_time.count());
        if (!print_all_times) printf("\b\b\n");

        // create the smoother
        CHECK_CUDA(cudaDeviceSynchronize());
        auto start_07 = std::chrono::high_resolution_clock::now();
        T omega = 1.5; // for GS-SOR
        auto smoother = new Smoother(assembler, kmat, h_color_rowp, omega);
        CHECK_CUDA(cudaDeviceSynchronize());
        auto end_07 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> create_smoother_time = end_07 - start_07;
        if (print_all_times) printf("\tcreate smoother %.2e, ", create_smoother_time.count());

        // create prolongation
        CHECK_CUDA(cudaDeviceSynchronize());
        auto start_08 = std::chrono::high_resolution_clock::now();
        int ELEM_MAX = 10; // num nearby elements of each fine node for nz pattern construction
        // int ELEM_MAX = 4;
        auto prolongation = new Prolongation(assembler, ELEM_MAX);
        CHECK_CUDA(cudaDeviceSynchronize());
        auto end_08 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> create_prolong_time = end_08 - start_08;
        if (print_all_times) printf("create prolong (P1) %.2e, ", create_prolong_time.count());

        // create grid
        CHECK_CUDA(cudaDeviceSynchronize());
        auto start_09 = std::chrono::high_resolution_clock::now();
        auto grid = GRID(assembler, prolongation, smoother, kmat, loads);
        CHECK_CUDA(cudaDeviceSynchronize());
        auto end_09 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> create_grid_time = end_09 - start_09;
        if (print_all_times) printf("create grid %.2e\n", create_grid_time.count());

        if (is_kcycle) {
            kmg->grids.push_back(grid);
        } else {
            mg->grids.push_back(grid);
            if (coarsest_grid) {
                CHECK_CUDA(cudaDeviceSynchronize());
                auto start_10 = std::chrono::high_resolution_clock::now();
                mg->coarse_solver = new CoarseSolver(assembler, kmat);
                CHECK_CUDA(cudaDeviceSynchronize());
                auto end_10 = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> create_coarse_solver = end_10 - start_10;
                if (print_all_times) printf("\tcreate coarse solver %.2e\n", create_coarse_solver.count());

            }
        }

        auto end_loc = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> grid_loop_time = end_loc - start_00;
        printf("\ttotal grid loop time %.2e\n", grid_loop_time.count());
    }

    // register the coarse assemblers to the prolongations..
    CHECK_CUDA(cudaDeviceSynchronize());
    auto start_11 = std::chrono::high_resolution_clock::now();
    if (is_kcycle) {
        kmg->template init_prolongations<Basis>();
    } else {
        mg->template init_prolongations<Basis>();
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end_11 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> init_prolong_time = end_11 - start_11;
    printf("total create prolong = %.2e\n", init_prolong_time.count());

    T init_resid_nrm = is_kcycle ? kmg->grids[0].getResidNorm() : mg->grids[0].getResidNorm();

    CHECK_CUDA(cudaDeviceSynchronize());
    auto start1 = std::chrono::high_resolution_clock::now();
    int pre_smooth = nsmooth, post_smooth = nsmooth;
    // best was V(4,4) before
    // bool print = false;
    bool print = true;
    T atol = 1e-6, rtol = 1e-6;
    T omega2 = 1.5; // really is set up there
    int n_cycles = 200;
    if (SR > 100.0) n_cycles = 1000;
    bool time = false;
    bool symmetric = false;
    int print_freq = 5;
    bool double_smooth = true; // true tends to be slightly faster sometimes


    if (is_kcycle) {
        CHECK_CUDA(cudaDeviceSynchronize());
        auto start_12 = std::chrono::high_resolution_clock::now();
        int n_krylov = 500;
        kmg->init_outer_solver(nsmooth, ninnercyc, n_krylov, omega2, atol, rtol, print_freq, print, double_smooth);    
        CHECK_CUDA(cudaDeviceSynchronize());
        auto end_12 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> kmg_init = end_12 - start_12;
        printf("create kmg time = %.2e\n", kmg_init.count());
    }

    auto end0 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> startup_time = end0 - start0;

    
    // fastest is K-cycle usually
    printf("\nstarting %s cycle solve\n", cycle_type.c_str());
    if (cycle_type == "K") {
        kmg->solve();
    } else if (cycle_type == "V") {
        mg->vcycle_solve(0, pre_smooth, post_smooth, n_cycles, print, atol, rtol, double_smooth, print_freq, symmetric, time); //(good option)
    } else if (cycle_type == "W") {
        mg->wcycle_solve(0, pre_smooth, post_smooth, n_cycles, print, atol, rtol);
    } else if (cycle_type == "F") {
        mg->fcycle_solve(0, pre_smooth, post_smooth, n_cycles, print, atol, rtol, double_smooth, print_freq, symmetric, time); // also decent
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solve_time = end1 - start1;
    int ndof = cycle_type == "K" ? kmg->grids[0].N : mg->grids[0].N;
    double total = startup_time.count() + solve_time.count();
    double mem_MB = is_kcycle ? kmg->get_memory_usage_mb() : mg->get_memory_usage_mb();
    printf("wingbox GMG solve, ndof %d : startup time %.2e, solve time %.2e, total %.2e, with mem(MB) %.2e\n", ndof, startup_time.count(), solve_time.count(), total, mem_MB);

    if (is_kcycle) {
        // double check with true resid nrm
        T resid_nrm = kmg->grids[0].getResidNorm();
        printf("init resid_nrm = %.2e => final resid_nrm = %.2e\n", init_resid_nrm, resid_nrm);

        // print some of the data of host residual
        int *d_perm = kmg->grids[0].d_perm;
        auto h_soln = kmg->grids[0].d_soln.createPermuteVec(6, d_perm).createHostVec();
        printToVTK<Assembler,HostVec<T>>(kmg->grids[0].assembler, h_soln, "out/aob_wing_mg.vtk");
    } else {
        // double check with true resid nrm
        T resid_nrm = mg->grids[0].getResidNorm();
        printf("init resid_nrm = %.2e => final resid_nrm = %.2e\n", init_resid_nrm, resid_nrm);

        // print some of the data of host residual
        int *d_perm = mg->grids[0].d_perm;
        auto h_soln = mg->grids[0].d_soln.createPermuteVec(6, d_perm).createHostVec();
        printToVTK<Assembler,HostVec<T>>(mg->grids[0].assembler, h_soln, "out/aob_wing_mg.vtk");
    }
}

int main(int argc, char **argv) {
    // Intialize MPI and declare communicator
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    // DEFAULTS
    int level = 4; // level mesh to solve..
    double SR = 300.0;
    int nsmooth = 2; // typically faster right now
    int ninnercyc = 2; // inner V-cycles to precond K-cycle
    std::string cycle_type = "K"; // "V", "F", "W", "K"
    std::string elem_type = "MITC4"; // 'MITC4', 'CFI4', 'CFI9'

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
    constexpr bool is_nonlinear = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;

    // // for faster debugging and compile time, just uncomment this and comment below section out (only compiles a single element)
    // printf("AOB mesh with MITC4 elements, level %d and SR %.2e\n------------\n", level, SR);
    // using Basis = LagrangeQuadBasis<T, Quad, 2>;
    // using Assembler = MITCShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
    // time_wing_multigrid<T, Assembler>(comm, level, SR, nsmooth, ninnercyc, cycle_type);

    // for faster debugging, just uncomment this and comment below section out
    printf("AOB mesh with CFI4 elements, level %d and SR %.2e\n------------\n", level, SR);
    using Basis = ChebyshevQuadBasis<T, Quad, 1>;
    using Assembler = FullyIntegratedShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
    time_wing_multigrid<T, Assembler>(comm, level, SR, nsmooth, ninnercyc, cycle_type);

    // printf("AOB level %d mesh with %s elements, and SR %.2e\n------------\n", level, elem_type.c_str(), SR);
    // if (elem_type == "MITC4") {
    //     using Basis = LagrangeQuadBasis<T, Quad, 2>;
    //     using Assembler = MITCShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
    //     time_wing_multigrid<T, Assembler>(comm, level, SR, nsmooth, ninnercyc, cycle_type);
    // } else if (elem_type == "CFI4") {
    //     using Basis = ChebyshevQuadBasis<T, Quad, 1>;
    //     using Assembler = FullyIntegratedShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
    //     time_wing_multigrid<T, Assembler>(comm, level, SR, nsmooth, ninnercyc, cycle_type);
    // } else if (elem_type == "CFI9") {
    //     using Basis = ChebyshevQuadBasis<T, Quad, 2>;
    //     using Assembler = FullyIntegratedShellAssembler<T, Director, Basis, Physics, VecType, BsrMat>;
    //     time_wing_multigrid<T, Assembler>(comm, level, SR, nsmooth, ninnercyc, cycle_type);
    // } else {
    //     printf("ERROR : didn't run anything, elem type not in available types (see main function)\n");
    // }

    MPI_Finalize();
    return 0;

    
}