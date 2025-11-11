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
void multigrid_plate_solve(int nxe, double SR, int n_iters, double omega, std::string smooth_name) {
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
    using Prolongation = StructuredProlongation<CYLINDER>;
    const SCALER scaler = NONE; // don't need line search for one prolong (just changes magnitude)
    // const SCALER scaler = LINE_SEARCH;
    using GRID = ShellGrid<Assembler, Prolongation, smoother, scaler>;
    // using MG = ShellMultigrid<GRID>;

    // make the fine grid
    int nye = nxe;
    double L = 1.0, R = 0.5, thick = L / SR;
    double E = 70e9, nu = 0.3;
    // double rho = 2500, ys = 350e6;
    bool imperfection = false; // option for geom imperfection
    int imp_x = 1, imp_hoop = 1; // no imperfection this input doesn't matter rn..
    auto fine_assembler = createCylinderAssembler<Assembler>(nxe, nye, L, R, E, nu, thick, imperfection, imp_x, imp_hoop);
    constexpr bool compressive = false;
    const int load_case = 3; // petal and chirp load
    double Q = 1.0; // load magnitude
    T *fine_loads = getCylinderLoads<T, Physics, load_case>(nxe, nye, L, R, Q);
    // printf("making fine grid object\n");
    GRID *fine_grid = GRID::buildFromAssembler(fine_assembler, fine_loads);

    // make a second grid here
    // printf("making coarse grid assembler\n");
    auto coarse_assembler = createCylinderAssembler<Assembler>(nxe / 2, nye / 2, L, R, E, nu, thick, imperfection, imp_x, imp_hoop);
    T *my_coarse_loads = getCylinderLoads<T, Physics, load_case>(nxe / 2, nye / 2, L, R, Q);
    bool full_LU = true; // only use full LU pattern on coarse grid..
    // printf("making coarse assembler\n");
    GRID *coarse_grid = GRID::buildFromAssembler(coarse_assembler, my_coarse_loads, full_LU);

    // solve on the coarse grid first
    // printf("coarse grid direct solve\n");
    coarse_grid->direct_solve();
    // printf("plot coarse solution\n");
    plotSolution<T, Assembler, GRID>(coarse_grid, coarse_grid->d_soln, "out/1_cylinder_coarse_soln.vtk");
    // printf("done plot coarse solution\n");

    // zero the fine defect first then prolong and show init fine defect (with high freq error)
    fine_grid->zeroDefect();
    fine_grid->prolongate(coarse_grid->d_iperm, coarse_grid->d_soln);
    plotSolution<T, Assembler, GRID>(fine_grid, fine_grid->d_defect, "out/2_cylinder_fine_defect.vtk");

    // now use the smoother to try and reduce high freq error
    bool print = true;
    int print_freq = 1;
    // printf("fine grid smooth defect\n");
    fine_grid->smoothDefect(n_iters, print, print_freq, omega, true);
    // printf("fine grid write soln\n");
    std::stringstream outputFile;
    outputFile << "out/3_cylinder_" << smooth_name << "_omega_" << std::to_string(omega) << ".vtk";
    plotSolution<T, Assembler, GRID>(fine_grid, fine_grid->d_defect, outputFile.str());
    // printf("done and free\n");

    fine_grid->free();
    coarse_grid->free();
}

int main() {
    // multigrid objects

    int nxe = 128;
    double SR = 10.0;
    // int n_iters = 5;
    int n_iters = 10;

    double omega = 0.7;
    multigrid_plate_solve<DAMPED_JACOBI>(nxe, SR, n_iters, omega, "DAMPED_JACOBI");
    multigrid_plate_solve<LEXIGRAPHIC_GS>(nxe, SR, n_iters, omega, "LEXIGRAPHIC_GS");
    multigrid_plate_solve<MULTICOLOR_GS_FAST2>(nxe, SR, n_iters, omega, "MULTICOLOR_GS_FAST2");

    // omega = 1.0;
    // multigrid_plate_solve<DAMPED_JACOBI>(nxe, SR, n_iters, omega, "DAMPED_JACOBI");
    // multigrid_plate_solve<LEXIGRAPHIC_GS>(nxe, SR, n_iters, omega, "LEXIGRAPHIC_GS");
    // multigrid_plate_solve<MULTICOLOR_GS_FAST2>(nxe, SR, n_iters, omega, "MULTICOLOR_GS_FAST2");

    omega = 1.5;
    multigrid_plate_solve<DAMPED_JACOBI>(nxe, SR, n_iters, omega, "DAMPED_JACOBI");
    multigrid_plate_solve<LEXIGRAPHIC_GS>(nxe, SR, n_iters, omega, "LEXIGRAPHIC_GS");
    multigrid_plate_solve<MULTICOLOR_GS_FAST2>(nxe, SR, n_iters, omega, "MULTICOLOR_GS_FAST2");
    
};