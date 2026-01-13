
#include "linalg/_linalg.h"
#include "mesh/TACSMeshLoader.h"
#include "mesh/vtk_writer.h"
#include "solvers/_solvers.h"

// shell imports
#include "assembler.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/shell_elem_group.h"

// local multigrid imports
#include "include/grid.h"
#include "include/fea.h"
#include "include/mg.h"
#include <string>
#include <chrono>

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

void solve_linear_multigrid_v1(MPI_Comm &comm, int level) {
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

    // multigrid objects
    // const SMOOTHER smoother = MULTICOLOR_GS;
    const SMOOTHER smoother = MULTICOLOR_GS_FAST;
    // const SMOOTHER smoother = LEXIGRAPHIC_GS;

    // using previous prolong
    using Prolongation = UnstructuredProlongation<Basis>; // this appears to actually be a litle faster..
    // using Prolongation = UnstructuredProlongationFast<Basis>;

    using GRID = ShellGrid<Assembler, Prolongation, smoother>;
    using MG = ShellMultigrid<GRID>;

    auto start0 = std::chrono::high_resolution_clock::now();
    auto mg = MG();
    // std::vector<GRID> direct_grids;

    // make each wing multigrid object.. (highest mesh level is finest, this is flipped from MG object's convention)
    for (int i = level; i >= 0; i--) {

        // read the ESP/CAPS => nastran mesh for TACS
        TACSMeshLoader mesh_loader{comm};
        std::string fname = "meshes/naca_wing_L" + std::to_string(i) + ".bdf";
        mesh_loader.scanBDFFile(fname.c_str());
        double E = 70e9, nu = 0.3, thick = 1.0;  // material & thick properties (start thicker first try)
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

        // set reasonable design variables (optional, otherwise const thick..)
        int ndvs = assembler.get_num_dvs(); // 32 components
        // TODO : make thinner later

        // internal struct and skin/OML thicknesses
        T its_thick = 0.1, skin_thick = 1.0;
        // T its_thick = 0.01, skin_thick = 0.1;
        // T its_thick = 0.001, skin_thick = 0.01;

        bool is_int_struct[32] = {1, 1, 0, 1,   0, 0, 0, 1,   1, 1, 0, 1,   0, 0, 0, 1,
            1, 0, 0, 1,   0, 0, 1, 0,   0, 1, 0, 0,   1, 0, 0, 1 };
        T *h_dvs_ptr = new T[32];
        for (int j = 0; j < 32; j++) {
            if (is_int_struct[j]) {
                h_dvs_ptr[j] = its_thick;
            } else {
                h_dvs_ptr[j] = skin_thick;
            }
        }
        auto h_dvs = HostVec<T>(32, h_dvs_ptr);
        auto global_dvs = h_dvs.createDeviceVec();
        assembler.set_design_variables(global_dvs);

        // make the grid
        bool full_LU = i == 0; // smallest grid is direct solve
        bool reorder;
        if (smoother == LEXIGRAPHIC_GS) {
            reorder = false;
        } else if (smoother == MULTICOLOR_GS || smoother == MULTICOLOR_GS_FAST) {
            reorder = true;
        }
        auto grid = *GRID::buildFromAssembler(assembler, my_loads, full_LU, reorder);
        mg.grids.push_back(grid); // add new grid
    }

    if (!Prolongation::structured) {
        // int ELEM_MAX = 4; // for plate, cylinder
        int ELEM_MAX = 10; // for wingbox esp near rib, spar, OML junctions
        mg.template init_unstructured<Basis>(ELEM_MAX);
        // printf("done with init unstructured\n");
        // return; // TEMP DEBUG
    }

    // do prolongation..
    mg.grids[1].direct_solve(false);
    mg.grids[0].prolongate(mg.grids[1].d_iperm, mg.grids[1].d_soln);
    int *d_perm = mg.grids[0].d_perm;
    auto h_soln = mg.grids[0].d_soln.createPermuteVec(6, d_perm).createHostVec();
    printToVTK<Assembler,HostVec<T>>(mg.grids[0].assembler, h_soln, "out/wing_prolong_v1.vtk");

    auto h_def1 = mg.grids[0].d_defect.createPermuteVec(6, d_perm).createHostVec();
    printToVTK<Assembler,HostVec<T>>(mg.grids[0].assembler, h_def1, "out/wing_def_v1.vtk");

    // do restriction of defect
    mg.grids[1].restrict_defect(
                        mg.grids[0].nelems, mg.grids[0].d_iperm, mg.grids[0].d_defect);
    int *d_perm2 = mg.grids[1].d_perm;
    auto h_def2 = mg.grids[1].d_defect.createPermuteVec(6, d_perm2).createHostVec();
    printToVTK<Assembler,HostVec<T>>(mg.grids[1].assembler, h_def2, "out/wing_restr_v1.vtk");
}

void solve_linear_multigrid_v2(MPI_Comm &comm, int level) {
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

    // multigrid objects
    // const SMOOTHER smoother = MULTICOLOR_GS;
    const SMOOTHER smoother = MULTICOLOR_GS_FAST;
    // const SMOOTHER smoother = LEXIGRAPHIC_GS;

    // using Prolongation = UnstructuredProlongation<Basis>;
    using Prolongation = UnstructuredProlongationFast<Basis>;

    using GRID = ShellGrid<Assembler, Prolongation, smoother>;
    using MG = ShellMultigrid<GRID>;

    auto start0 = std::chrono::high_resolution_clock::now();
    auto mg = MG();
    // std::vector<GRID> direct_grids;

    // make each wing multigrid object.. (highest mesh level is finest, this is flipped from MG object's convention)
    for (int i = level; i >= 0; i--) {

        // read the ESP/CAPS => nastran mesh for TACS
        TACSMeshLoader mesh_loader{comm};
        std::string fname = "meshes/naca_wing_L" + std::to_string(i) + ".bdf";
        mesh_loader.scanBDFFile(fname.c_str());
        double E = 70e9, nu = 0.3, thick = 1.0;  // material & thick properties (start thicker first try)
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

        // set reasonable design variables (optional, otherwise const thick..)
        int ndvs = assembler.get_num_dvs(); // 32 components
        // TODO : make thinner later

        // internal struct and skin/OML thicknesses
        T its_thick = 0.1, skin_thick = 1.0;
        // T its_thick = 0.01, skin_thick = 0.1;
        // T its_thick = 0.001, skin_thick = 0.01;

        bool is_int_struct[32] = {1, 1, 0, 1,   0, 0, 0, 1,   1, 1, 0, 1,   0, 0, 0, 1,
            1, 0, 0, 1,   0, 0, 1, 0,   0, 1, 0, 0,   1, 0, 0, 1 };
        T *h_dvs_ptr = new T[32];
        for (int j = 0; j < 32; j++) {
            if (is_int_struct[j]) {
                h_dvs_ptr[j] = its_thick;
            } else {
                h_dvs_ptr[j] = skin_thick;
            }
        }
        auto h_dvs = HostVec<T>(32, h_dvs_ptr);
        auto global_dvs = h_dvs.createDeviceVec();
        assembler.set_design_variables(global_dvs);

        // make the grid
        bool full_LU = i == 0; // smallest grid is direct solve
        bool reorder;
        if (smoother == LEXIGRAPHIC_GS) {
            reorder = false;
        } else if (smoother == MULTICOLOR_GS || smoother == MULTICOLOR_GS_FAST) {
            reorder = true;
        }
        auto grid = *GRID::buildFromAssembler(assembler, my_loads, full_LU, reorder);
        mg.grids.push_back(grid); // add new grid
    }

    if (!Prolongation::structured) {
        // int ELEM_MAX = 4; // for plate, cylinder
        int ELEM_MAX = 10; // for wingbox esp near rib, spar, OML junctions
        mg.template init_unstructured<Basis>(ELEM_MAX);
        // printf("done with init unstructured\n");
        // return; // TEMP DEBUG
    }

    // do prolongation..
    mg.grids[1].direct_solve(false);
    mg.grids[0].prolongate(mg.grids[1].d_iperm, mg.grids[1].d_soln);
    int *d_perm = mg.grids[0].d_perm;
    auto h_soln = mg.grids[0].d_soln.createPermuteVec(6, d_perm).createHostVec();
    printToVTK<Assembler,HostVec<T>>(mg.grids[0].assembler, h_soln, "out/wing_prolong_v2.vtk");

    auto h_def1 = mg.grids[0].d_defect.createPermuteVec(6, d_perm).createHostVec();
    printToVTK<Assembler,HostVec<T>>(mg.grids[0].assembler, h_def1, "out/wing_def_v2.vtk");

    // do restriction of defect
    mg.grids[1].restrict_defect(
                        mg.grids[0].nelems, mg.grids[0].d_iperm, mg.grids[0].d_defect);
    int *d_perm2 = mg.grids[1].d_perm;
    auto h_def2 = mg.grids[1].d_defect.createPermuteVec(6, d_perm2).createHostVec();
    printToVTK<Assembler,HostVec<T>>(mg.grids[1].assembler, h_def2, "out/wing_restr_v2.vtk");
}

int main(int argc, char **argv) {

    // Intialize MPI and declare communicator
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    solve_linear_multigrid_v1(comm, 1);
    solve_linear_multigrid_v2(comm, 1);

    MPI_Finalize();
    return 0;
};
