// general gpu_fem imports
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"
#include <chrono>

// shell imports
#include "assembler.h"
#include "element/shell/basis/lagrange_basis.h"
#include "element/shell/director/linear_rotation.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/mitc_shell.h"

// local multigrid imports
// #include "multigrid/grid.h"
#include "multigrid/prolongation/structured.h"
#include "multigrid/solvers/direct/cusp_directLU.h"
#include "multigrid/fea.h"
#include "multigrid/smoothers/mc_smooth1.h"
#include <string>
#include <chrono>

void to_lowercase(char *str) {
    for (; *str; ++str) {
        *str = std::tolower(*str);
    }
}

enum CLEANUP : short {
    SMOOTHER,
    DIRECT_SOLVE,
    STRUCT_PROLONGATION,
};

void test_cleanup(int nxe, double SR = 100.0) {
    // geometric multigrid method here..
    // need to make a number of grids..

    using T = double;   
    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = LagrangeQuadBasis<T, Quad, 1>;
    using Geo = Basis::Geo;
    constexpr bool has_ref_axis = false;
    constexpr bool is_nonlinear = false;
    using Data = ShellIsotropicData<T, has_ref_axis>;
    using Physics = IsotropicShell<T, Data, is_nonlinear>;
    using Assembler = MITCShellAssembler<T, Director, Basis, Physics, DeviceVec, BsrMat>;

    // direct solver class
    using DirectSolver = CusparseMGDirectLU<T, Assembler>;
    using Smoother = MulticolorGSSmoother_V1<Assembler>;
    using Prolongation = StructuredProlongation<Assembler, PLATE>;

    CHECK_CUDA(cudaDeviceSynchronize());
    auto start0 = std::chrono::high_resolution_clock::now();

    // make a single grid
    double Lx = 1.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 1.0 / SR, rho = 2500, ys = 350e6;
    int nxe_per_comp = nxe / 4, nye_per_comp = nxe/4; // for now (should have 25 grids)
    auto assembler = createPlateAssembler<Assembler>(nxe, nxe, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp);
    double Q = 1.0; // load magnitude
    T *my_loads = getPlateLoads<T, Physics>(nxe, nxe, Lx, Ly, Q);
    printf("making grid with nxe %d\n", nxe);

    auto &bsr_data = assembler.getBsrData();
    int num_colors, *_color_rowp;

    /* multicolor reordering */
    bsr_data.multicolor_reordering(num_colors, _color_rowp);
    bsr_data.compute_nofill_pattern();
    auto h_color_rowp = HostVec<int>(num_colors + 1, _color_rowp);
    assembler.moveBsrDataToDevice();

    // bsr_data.AMD_reordering();
    // bsr_data.compute_full_LU_pattern(10.0, false);
    // assembler.moveBsrDataToDevice();
    
    // make loads and assemble kmat
    auto loads = assembler.createVarsVec(my_loads);
    assembler.apply_bcs(loads);
    auto kmat = createBsrMat<Assembler, VecType<T>>(assembler);
    auto res = assembler.createVarsVec();
    auto soln = assembler.createVarsVec();
    int N = res.getSize();

    // assemble the kmat
    auto start1 = std::chrono::high_resolution_clock::now();
    assembler.add_jacobian(res, kmat);
    CHECK_CUDA(cudaDeviceSynchronize());
    assembler.apply_bcs(kmat);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> assembly_time = end1 - start1;
    printf("\tassemble kmat time %.2e\n", assembly_time.count());

    // const CLEANUP cleanup = SMOOTHER;
    // const CLEANUP cleanup = DIRECT_SOLVE;
    const CLEANUP cleanup = STRUCT_PROLONGATION;

    /* 1) test the smoother */
    // -----------------------------------------------------
    
    if constexpr (cleanup == SMOOTHER) {
        // T omega = 1.5;
        T omega = 0.7;
        auto smoother = Smoother(assembler, kmat, h_color_rowp, omega);
        auto defect = assembler.createVarsVec();

        // copy loads to defect and permute it
        loads.copyValuesTo(defect);
        int block_dim = bsr_data.block_dim, *d_iperm = kmat.getIPerm();
        defect.permuteData(block_dim, d_iperm);

        // now try and do smoothing solve
        int n_iters = 30, print_freq = 1;
        bool print = true;
        smoother.smoothDefect(defect, soln, n_iters, print, print_freq);

        int *d_perm = kmat.getPerm();
        defect.permuteData(block_dim, d_perm);
        auto h_defect = defect.createHostVec();
        printToVTK<Assembler,HostVec<T>>(assembler, h_defect, "out/plate_smooth.vtk");
    }
    

    /* 2) test the direct solver base class */
    // --------------------------------------------------

    if constexpr (cleanup == DIRECT_SOLVE) {
        // try and do direct solve
        printf("direct solve v0\n");
        CUSPARSE::direct_LU_solve(kmat, loads, soln);
        auto h_soln = soln.createHostVec();
        printToVTK<Assembler,HostVec<T>>(assembler, h_soln, "out/plate_direct0.vtk");
        // return; // temp debug

        // permute the loads to right order (of direct solver)
        int block_dim = bsr_data.block_dim;
        int *d_iperm = kmat.getIPerm();
        loads.permuteData(block_dim, d_iperm);

        // make the direct solver
        printf("make direct solver v1\n");
        auto solver = DirectSolver(assembler, kmat);
        printf("perform direct solver v1\n");
        solver.solve(loads, soln);

        // write the fine prolong to VTK
        int *d_perm = kmat.getPerm();
        soln.permuteData(block_dim, d_perm);
        auto h_soln2 = soln.createHostVec();
        printToVTK<Assembler,HostVec<T>>(assembler, h_soln2, "out/plate_direct1.vtk");
    }

    /* 3) test the structured prolongation */

    if constexpr (cleanup == STRUCT_PROLONGATION) {

        // make a coarse assembler as well
        printf("make coarse assembler\n");
        auto coarse_assembler = createPlateAssembler<Assembler>(nxe / 2, nxe / 2, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp / 2, nye_per_comp / 2);
        T *coarse_loads_ptr = getPlateLoads<T, Physics>(nxe / 2, nxe / 2, Lx, Ly, Q);
        auto &coarse_bsr_data = coarse_assembler.getBsrData();
        coarse_bsr_data.AMD_reordering();
        coarse_bsr_data.compute_full_LU_pattern(10.0, false);
        coarse_assembler.moveBsrDataToDevice();
        auto coarse_loads = coarse_assembler.createVarsVec(coarse_loads_ptr);
        coarse_assembler.apply_bcs(coarse_loads);
        auto coarse_kmat = createBsrMat<Assembler, VecType<T>>(coarse_assembler);
        auto coarse_res = coarse_assembler.createVarsVec();
        auto coarse_soln = coarse_assembler.createVarsVec();
        coarse_assembler.add_jacobian(coarse_res, coarse_kmat);
        coarse_assembler.apply_bcs(coarse_kmat);
        printf("\tdone making & assembling coarse problem\n");

        // solve the coarse problem
        printf("make the coarse direct solver\n");
        auto coarse_solver = DirectSolver(coarse_assembler, coarse_kmat);
        int block_dim = coarse_bsr_data.block_dim;
        int *coarse_d_iperm = coarse_kmat.getIPerm();
        coarse_loads.permuteData(block_dim, coarse_d_iperm);
        coarse_solver.solve(coarse_loads, coarse_soln);
        printf("\tdone with coarse direct solve\n");

        // now make the prolongator
        printf("make the struct prolongation\n");
        auto prolongation = Prolongation(assembler);
        prolongation.init_coarse_data(coarse_assembler);
        printf("\tdone making the struct prolongation\n");

        // prolong to the fine mesh and write the soln update
        printf("perform coarse fine prolong\n");
        prolongation.prolongate(coarse_soln, soln);
        printf("\tdone with coarse fine prolong\n");

        // write the coarse solve to VTK
        int *coarse_d_perm = coarse_kmat.getPerm();
        coarse_soln.permuteData(block_dim, coarse_d_perm);
        auto h_coarse_soln = coarse_soln.createHostVec();
        printToVTK<Assembler,HostVec<T>>(coarse_assembler, h_coarse_soln, "out/plate_prolong_c.vtk");

        // write the fine prolong to VTK
        int *d_perm = kmat.getPerm();
        soln.permuteData(block_dim, d_perm);
        auto h_soln = soln.createHostVec();
        printToVTK<Assembler,HostVec<T>>(assembler, h_soln, "out/plate_prolong_f.vtk");
    } // end of struct prolong cleanup

}


int main(int argc, char **argv) {
    // input ----------
    int nxe = 256; // default value

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
        } else {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            std::cerr << "Usage: " << argv[0] << " [direct/mg] [--nxe value] [--SR value] [--cycle char] [--nsmooth int] [--ninnercyc int]" << std::endl;
            return 1;
        }
    }

    // done reading arts, now run stuff
    test_cleanup(nxe);

    return 0;

    
}