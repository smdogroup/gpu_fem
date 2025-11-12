#include <iostream>
#include <sstream>

#include "chrono"
#include "coupled/_coupled.h"
#include "linalg/_linalg.h"
#include "mesh/TACSMeshLoader.h"
#include "solvers/_solvers.h"

// shell imports
#include "assembler.h"
#include "element/shell/basis/lagrange_basis.h"
#include "element/shell/director/linear_rotation.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/mitc_shell.h"

// multigrid imports
#include "multigrid/fea.h"
#include "multigrid/grid.h"
#include "multigrid/interface.h"
#include "multigrid/solvers/gmg.h"

// copied and modified from ../uCRM/_src/optim.h (uCRM optimization example)

class TacsGpuMultigridSolver {
   public:
    using T = double;
    // FEM typedefs
    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = LagrangeQuadBasis<T, Quad, 2>;
    using Data = ShellIsotropicData<T, false>;
    using Physics = IsotropicShell<T, Data, false>;
    using ElemGroup = MITCShellElementGroup<T, Director, Basis, Physics>;
    using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;

    // multigrid objects
    using Prolongation = UnstructuredProlongation<Basis, false>;
    using GRID = ShellGrid<Assembler, Prolongation, MULTICOLOR_GS_FAST2_JUNCTION, LINE_SEARCH>;
    using MG = GeometricMultigridSolver<GRID>;
    using StructSolver = TacsMGInterface<T, Assembler, MG>;

    // functions
    using DMass = Mass<T, DeviceVec>;
    using DKSFail = KSFailure<T, DeviceVec>;

    TacsGpuMultigridSolver(double rhoKS = 100.0, double safety_factor = 1.5,
                           double load_mag = 100.0, double omega = 1.5, int mesh_level = 3,
                           double SR = 50.0) {
        // init MPI comm
        MPI_Init(NULL, NULL);
        MPI_Comm comm = MPI_COMM_WORLD;

        // 1) Build mesh & assembler & multigrids
        auto mg = MG();

        // object's convention)
        for (int i = mesh_level; i >= 0; i--) {
            // read the ESP/CAPS => nastran mesh for TACS
            TACSMeshLoader mesh_loader{comm};
            std::string fname = "meshes/aob_wing_L" + std::to_string(i) + ".bdf";
            mesh_loader.scanBDFFile(fname.c_str());
            double E = 70e9, nu = 0.3,
                   thick = 2.0 / SR;  // material & thick properties (start thicker first try)
            double rho = 2500, ys = 350e6;

            printf("making assembler+GMG for mesh '%s'\n", fname.c_str());

            // create the TACS Assembler from the mesh loader
            auto assembler = Assembler::createFromBDF(mesh_loader, Data(E, nu, thick, rho, ys));

            // create the loads (really only needed on finer mesh.. TBD how to setup nonlinear
            // case..)
            int nvars = assembler.get_num_vars();
            int nnodes = assembler.get_num_nodes();
            HostVec<T> h_loads(nvars);
            double load_mag = 10.0;
            double *my_loads = h_loads.getPtr();
            for (int inode = 0; inode < nnodes; inode++) {
                my_loads[6 * inode + 2] = load_mag;
            }

            // make the grid
            bool full_LU = i == 0;  // smallest grid is direct solve
            bool reorder = true;    // to do multicolor
            // printf("reorder %d\n", reorder);
            auto grid = *GRID::buildFromAssembler(assembler, my_loads, full_LU, reorder);
            mg.grids.push_back(grid);  // add new grid
        }

        if (!Prolongation::structured) {
            printf("begin unstructured map\n");
            // int ELEM_MAX = 4; // for plate, cylinder
            int ELEM_MAX = 10;  // for wingbox esp near rib, spar, OML junctions
            mg.template init_unstructured<Basis>(ELEM_MAX);
            printf("done with init unstructured\n");
            // return; // TEMP DEBUG
        }

        // now make the solver interface
        bool print = true;
        solver = std::make_unique<StructSolver>(mg, print);
        T atol = 1e-6, rtol = 1e-6;
        // int pre_smooth = 2, post_smooth = 2;
        // int pre_smooth = 4, post_smooth = 4;
        int pre_smooth = 6, post_smooth = 6;
        int n_cycles = 300, print_freq = 10;
        // bool double_smooth = false;  // actually don't need this on the wing maybe
        bool double_smooth = true;
        solver->set_mg_solver_settings(rtol, atol, n_cycles, pre_smooth, post_smooth, print_freq,
                                       double_smooth, omega);

        // get struct loads on finest grid
        auto fine_grid = mg.grids[0];
        d_loads = DeviceVec<T>(fine_grid.N);
        mg.grids[0].getDefect(d_loads);

        // initialize any vecs needed at this level
        auto &assembler = mg.grids[0].assembler;
        nvars = assembler.get_num_vars();
        int nn = assembler.get_num_nodes();
        soln = DeviceVec<T>(nvars);
        ndvs = assembler.get_num_dvs();
        d_dvs = DeviceVec<T>(ndvs, /*initial=*/0.02);

        // 5) Functions
        mass = std::make_unique<DMass>();
        ksfail = std::make_unique<DKSFail>(rhoKS, safety_factor);

        dvs_changed = true;
        first_solve = true;
    }

    void set_design_variables(const std::vector<T> &dvs) {
        /* check if dvs changed before running new analysis (make sure this works right) */
        dvs_changed = (dvs.size() != prev_dvs.size());
        if (!dvs_changed) {
            for (int i = 0; i < dvs.size(); i++) {
                if (dvs[i] != prev_dvs[i]) {
                    dvs_changed = true;
                    break;
                }
            }
        }

        // if (first_solve) {clear
        //     dvs_changed = true;
        //     first_solve = false;
        // }
        dvs_changed = true;  // debug

        if (dvs_changed) {
            prev_dvs = dvs;
            CHECK_CUDA(
                cudaMemcpy(d_dvs.getPtr(), dvs.data(), ndvs * sizeof(T), cudaMemcpyHostToDevice));
            solver->set_design_variables(d_dvs);
        }
    }

    int get_num_vars() const { return nvars; }
    int get_num_dvs() const { return ndvs; }
    void writeSolution(const std::string &filename) const { solver->writeSoln(filename); }

    void solve() {
        if (dvs_changed) {
            printf("design changed, new solve\n");

            // debugging
            // auto h_loads = d_loads.createHostVec();
            // printf("h_loads:");
            // printVec<T>(h_loads.getSize(), h_loads.getPtr());

            solver->solve(d_loads);
            solver->copy_solution_out(soln);
        } else {
            // reload old state
            printf("design didn't change, reload vals\n");
            solver->copy_solution_in(soln);
        }
    }

    T evalFunction(const std::string &name) {
        if (name == "mass")
            return solver->evalFunction(*mass);
        else if (name == "ksfailure")
            return solver->evalFunction(*ksfail);
        throw std::invalid_argument("Unknown func");
    }

    // fills host array of length ndvs
    void evalFunctionSens(const std::string &name, T *out_h_sens) {
        double *dptr;
        if (name == "mass") {
            solver->solve_adjoint(*mass);
            dptr = mass->dv_sens.getPtr();
        } else if (name == "ksfailure") {
            solver->solve_adjoint(*ksfail);
            dptr = ksfail->dv_sens.getPtr();
        } else {
            throw std::invalid_argument("Unknown func");
        }
        CHECK_CUDA(cudaMemcpy(out_h_sens, dptr, ndvs * sizeof(T), cudaMemcpyDeviceToHost));
    }

    void free() {
        solver->free();
        // assembler->free();
        d_loads.free();
        d_dvs.free();
        // tear down MPI if *we* initialized it
        int mpi_inited = 0;
        MPI_Finalized(&mpi_inited);
        if (!mpi_inited) {
            MPI_Finalize();
        }
    }

   private:
    // std::unique_ptr<Assembler> assembler;
    std::unique_ptr<StructSolver> solver;
    std::unique_ptr<DMass> mass;
    std::unique_ptr<DKSFail> ksfail;

    int ndvs = 0, nvars = 0;
    DeviceVec<T> d_loads, d_dvs;
    std::vector<T> prev_dvs;
    bool dvs_changed;
    bool first_solve;
    DeviceVec<T> soln;
};