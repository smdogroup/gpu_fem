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
void plotSolution(Grid *grid, DeviceVec<T> vec, std::string filename, bool perm = true) {
    // shortcut for the many plot solutions in here
    if (perm) {
        auto h_soln = vec.createPermuteVec(6, grid->Kmat.getPerm()).createHostVec();
        printToVTK<Assembler,HostVec<T>>(grid->assembler, h_soln, filename);
    } else {
        auto h_soln = vec.createHostVec();
        printToVTK<Assembler,HostVec<T>>(grid->assembler, h_soln, filename);
    }
}

void multigrid_junction_solve(MPI_Comm comm, double SR, int n_iters, double omega, std::string smooth_name) {
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
    const SMOOTHER smoother = MULTICOLOR_GS_FAST2_JUNCTION;
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

    // beginning of a new smoother
    // ---------------------------

    // plot the previous coloring (check)
    int *h_perm_v1 = DeviceVec<int>(fine_grid->nnodes, fine_grid->d_perm).createHostVec().getPtr();
    auto h_color_rowp = fine_grid->h_color_rowp;
    int _num_colors_v1 = h_color_rowp.getSize();
    printf("_num_colors_v1: %d\n", _num_colors_v1);
    int *_color_rowp_v1 = h_color_rowp.getPtr();
    T *colors_v1 = new T[fine_grid->N];
    memset(colors_v1, 0.0, fine_grid->N * sizeof(T));
    for (int icolor = 0; icolor < _num_colors_v1; icolor++) {
        for (int jp = _color_rowp_v1[icolor]; jp < _color_rowp_v1[icolor + 1]; jp++) {
            int inode = jp;
            // int perm_inode = h_perm_v1[inode];
            for (int idof = 0; idof < 6; idof++) {
                colors_v1[6 * inode + idof] = icolor; // keep it in permuted form cause plotSolution does un-permute
            }
        }
    }
    DeviceVec<T> d_colors_v1(fine_grid->N);
    cudaMemcpy(d_colors_v1.getPtr(), colors_v1, fine_grid->N * sizeof(T), cudaMemcpyHostToDevice);
    plotSolution<T, Assembler, GRID>(fine_grid, d_colors_v1, "out/_aob_colors1.vtk");


    // 1) compute new colored order with face, edge, corner hierarchy of the coloring
    // we're going a step further than interior vs junction nodes (so we can do supernodes better)
    // because corner supernodes will use edges + face nodes

    int _nnodes = fine_assembler.get_num_nodes();
    int *nodal_num_comps, *node_geom_ind;
    fine_grid->get_nodal_geom_indices(fine_assembler, nodal_num_comps, node_geom_ind);

    // plot all 6 DOF instead of just nodal value so we can write to VTK
    HostVec<T> h_nodal_num_comps(fine_grid->N), h_node_geom_ind(fine_grid->N);
    for (int i = 0; i < fine_grid->N; i++) {
        int inode = i / 6;
        h_nodal_num_comps[i] = nodal_num_comps[inode];
        h_node_geom_ind[i] = node_geom_ind[inode];
    }
    auto d_nodal_num_comps = h_nodal_num_comps.createDeviceVec();
    auto d_node_geom_ind = h_node_geom_ind.createDeviceVec();
    plotSolution<T, Assembler, GRID>(fine_grid, d_nodal_num_comps, "out/_aob_nodal_num_comps.vtk", false);
    plotSolution<T, Assembler, GRID>(fine_grid, d_node_geom_ind, "out/_aob_node_geom_ind.vtk", false);

    // 2) now develop color ordering based on num_comps and node_geom_ind and supernodes..
    // ------------------------------------------------------------------------------------
    // a) interior nodes are colored with no supernodes (colors 1-4) or block_dim = 6
    // b) edge nodes with 2 comps have 3 group-supernodes or block_dim = 18
    // c) edge nodes with 3 comps have 4 group-supernodes or block_dim = 24
    // d) corner nodes with 3 comps have 7 group-supernodes combining edge + interior or block_dim = 42
    // e) corner nodes with 5 comps have 10 group-supernodes or block_dim = 60


    // auto &bsr_data = fine_assembler.getBsrData();
    // bsr_data.multicolor_junction_reordering(is_interior_node, num_colors,
    //                                         _color_rowp);
    // printf("num_colors %d, color_rowp: ", num_colors);
    // printVec<int>(num_colors + 1, _color_rowp);
    

    // write smoother output
    // ---------------------
    bool print = true;
    int print_freq = 1;
    // printf("fine grid smooth defect\n");
    // fine_grid->smoothDefect(n_iters, print, print_freq, omega, true);
    std::stringstream outputFile;
    outputFile << "out/3_aob_" << smooth_name << "_omega_" << std::to_string(omega) << ".vtk";
    plotSolution<T, Assembler, GRID>(fine_grid, fine_grid->d_defect, outputFile.str());

    fine_grid->free();
    coarse_grid->free();
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

    // special and new one here.. custom supernodes on junction
    double omega = 0.7;
    // double omega = 1.5;
    multigrid_junction_solve(comm, SR, n_iters, omega, "MC_GS_SUPERNODE_JUNCTION");

    // omega = 1.5;
    // multigrid_solve<LEXIGRAPHIC_GS>(comm, SR, n_iters, omega, "LEXIGRAPHIC_GS");
    // multigrid_solve<MULTICOLOR_GS_FAST2>(comm, SR, n_iters, omega, "MULTICOLOR_GS_FAST2");
    // multigrid_solve<MULTICOLOR_GS_FAST2_JUNCTION>(comm, SR, n_iters, omega, "MULTICOLOR_GS_FAST2_JUNCTION");
};