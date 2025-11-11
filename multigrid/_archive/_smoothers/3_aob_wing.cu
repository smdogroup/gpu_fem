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

template <typename T, class Assembler, class Grid>
void plotSolution(Grid *grid, DeviceVec<T> vec, std::string filename) {
    // shortcut for the many plot solutions in here
    auto h_soln = vec.createPermuteVec(6, grid->Kmat.getPerm()).createHostVec();
    printToVTK<Assembler,HostVec<T>>(grid->assembler, h_soln, filename);
}

template <SMOOTHER smoother>
void multigrid_solve(MPI_Comm comm, double SR, int n_iters, double omega, std::string smooth_name) {
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
    using Prolongation = UnstructuredProlongationFast<Basis>;
    const SCALER scaler = NONE; // don't need line search for one prolong (just changes magnitude)
    // const SCALER scaler = LINE_SEARCH;
    using GRID = ShellGrid<Assembler, Prolongation, smoother, scaler>;
    // using MG = ShellMultigrid<GRID>;

    // make the fine grid
    TACSMeshLoader mesh_loader{comm};
    std::string fname = "../4_aob_wing/meshes/aob_wing_L2.bdf";
    mesh_loader.scanBDFFile(fname.c_str());
    double E = 70e9, nu = 0.3, thick = 2.0 / SR;
    auto fine_assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick));
    int nvarsf = fine_assembler.get_num_vars();
    int nnodesf = fine_assembler.get_num_nodes();
    HostVec<T> h_loads(nvarsf);
    double load_mag = 10.0;
    double *fine_loads = h_loads.getPtr();
    for (int inode = 0; inode < nnodesf; inode++) {
        fine_loads[6 * inode + 2] = load_mag;
    }
    bool full_LU = false, reorder = true;
    GRID *fine_grid = GRID::buildFromAssembler(fine_assembler, fine_loads, full_LU, reorder);

    // return;

    // make a second grid here
    TACSMeshLoader mesh_loader2{comm};
    fname = "../4_aob_wing/meshes/aob_wing_L1.bdf";
    mesh_loader2.scanBDFFile(fname.c_str());
    auto coarse_assembler = Assembler::createFromBDF(mesh_loader2, Data(E, nu, thick));
    int nvarsc = coarse_assembler.get_num_vars();
    int nnodesc = coarse_assembler.get_num_nodes();
    HostVec<T> h_loads2(nvarsc);
    double *coarse_loads = h_loads2.getPtr();
    for (int inode = 0; inode < nnodesc; inode++) {
        coarse_loads[6 * inode + 2] = load_mag;
    }
    full_LU = true; // only use full LU pattern on coarse grid..
    // printf("making coarse assembler\n");
    GRID *coarse_grid = GRID::buildFromAssembler(coarse_assembler, coarse_loads, full_LU);

    // init unstructured prolongation
    if (!Prolongation::structured) {
        int ELEM_MAX = 10;
        fine_grid->template init_unstructured_grid_maps<Basis>(*coarse_grid, ELEM_MAX);
    }

    // solve on the coarse grid first
    // printf("coarse grid direct solve\n");
    coarse_grid->direct_solve();
    // printf("plot coarse solution\n");
    plotSolution<T, Assembler, GRID>(coarse_grid, coarse_grid->d_soln, "out/1_aob_coarse_soln.vtk");
    // printf("done plot coarse solution\n");

    // zero the fine defect first then prolong and show init fine defect (with high freq error)
    fine_grid->zeroDefect();
    fine_grid->prolongate(coarse_grid->d_iperm, coarse_grid->d_soln);
    plotSolution<T, Assembler, GRID>(fine_grid, fine_grid->d_defect, "out/2_aob_fine_defect.vtk");

    // now use the smoother to try and reduce high freq error
    bool print = true;
    int print_freq = 1;
    // printf("fine grid smooth defect\n");
    fine_grid->smoothDefect(n_iters, print, print_freq, omega, true);
    // printf("fine grid write soln\n");
    std::stringstream outputFile;
    outputFile << "out/3_aob_" << smooth_name << "_omega_" << std::to_string(omega) << ".vtk";
    plotSolution<T, Assembler, GRID>(fine_grid, fine_grid->d_defect, outputFile.str());
    // printf("done and free\n");

    fine_grid->free();
    coarse_grid->free();
}

int main() {
    // multigrid objects

    double SR = 1.0;
    // double SR = 10.0;
    // int n_iters = 5;
    // int n_iters = 10;
    int n_iters = 30;

    MPI_Init(NULL, NULL);
    MPI_Comm comm = MPI_COMM_WORLD;

    // special and new one here..
    // double omega = 0.85;
    // multigrid_plate_solve<MULTICOLOR_GS_FAST2_JUNCTION>(comm, SR, n_iters, omega, "MULTICOLOR_GS_FAST2_JUNCTION");

    double omega = 1.5;
    // double omega = 0.7;
    // multigrid_solve<DAMPED_JACOBI>(comm, SR, n_iters, omega, "DAMPED_JACOBI"); // diverges
    multigrid_solve<LEXIGRAPHIC_GS>(comm, SR, n_iters, omega, "LEXIGRAPHIC_GS");
    multigrid_solve<MULTICOLOR_GS_FAST2>(comm, SR, n_iters, omega, "MULTICOLOR_GS_FAST2");
    multigrid_solve<MULTICOLOR_GS_FAST2_JUNCTION>(comm, SR, n_iters, omega, "MULTICOLOR_GS_FAST2_JUNCTION");

    // omega = 1.0;
    // multigrid_plate_solve<DAMPED_JACOBI>(comm, SR, n_iters, omega, "DAMPED_JACOBI");
    // multigrid_plate_solve<LEXIGRAPHIC_GS>(comm, SR, n_iters, omega, "LEXIGRAPHIC_GS");
    // multigrid_plate_solve<MULTICOLOR_GS_FAST2>(comm, SR, n_iters, omega, "MULTICOLOR_GS_FAST2");

    // omega = 1.5;
    // multigrid_plate_solve<DAMPED_JACOBI>(comm, SR, n_iters, omega, "DAMPED_JACOBI");
    // multigrid_plate_solve<LEXIGRAPHIC_GS>(comm, SR, n_iters, omega, "LEXIGRAPHIC_GS");
    // multigrid_plate_solve<MULTICOLOR_GS_FAST2>(comm, SR, n_iters, omega, "MULTICOLOR_GS_FAST2");
};