#include <iostream>
#include <sstream>

#include "chrono"
#include "coupled/_coupled.h"
#include "linalg/_linalg.h"
#include "solvers/_solvers.h"
#include "_plate_utils.h"

// shell imports
#include "assembler.h"
#include "element/shell/physics/isotropic_shell.h"
#include "element/shell/shell_elem_group.h"

// copied and modified from ../uCRM/_src/optim.h (uCRM optimization example)

class TACSGPUSolver {
   public:
    using T = double;
    // FEM typedefs
    using Quad = QuadLinearQuadrature<T>;
    using Director = LinearizedRotation<T>;
    using Basis = ShellQuadBasis<T, Quad, 2>;
    using Data = ShellIsotropicData<T, false>;
    using Physics = IsotropicShell<T, Data, false>;
    using ElemGroup = ShellElementGroup<T, Director, Basis, Physics>;
    using Assembler = ElementAssembler<T, ElemGroup, VecType, BsrMat>;
    using StructSolver = TacsLinearStatic<T, Assembler>;
    using DMass = Mass<T, DeviceVec>;
    using DKSFail = KSFailure<T, DeviceVec>;

    TACSGPUSolver(double rhoKS = 100.0, double safety_factor = 1.5, double load_mag = 100.0,
        int nxe = 100, int nx_comp = 5, int ny_comp = 5) {
        // 1) Build mesh & assembler
        assert(nxe % nx_comp == 0); // evenly divisible by number of elems_per_comp
        int nye = nxe;
        assert(nye % ny_comp == 0);
        int nxe_per_comp = nxe / nx_comp, nye_per_comp = nye / ny_comp;
        double Lx = 2.0, Ly = 1.0, E = 70e9, nu = 0.3, thick = 0.005, rho = 2500, ys = 250e6;
        
        Assembler local_asm = createPlateAssembler<Assembler>(nxe, nye, Lx, Ly, E, nu, thick, rho, ys, nxe_per_comp, nye_per_comp);
        // factor & move to GPU
        {
            auto &bsr = local_asm.getBsrData();
            bsr.AMD_reordering();
            bsr.compute_full_LU_pattern(10.0, true);
        }
        local_asm.moveBsrDataToDevice();
        assembler = std::make_unique<Assembler>(std::move(local_asm));

        // 2) Build loads
        nvars = assembler->get_num_vars();
        int nn = assembler->get_num_nodes();
        T *my_loads = getPlateLoads<T, Physics>(nxe, nye, Lx, Ly, load_mag);
        d_loads = assembler->createVarsVec(my_loads);
        assembler->apply_bcs(d_loads);


        soln = DeviceVec<T>(nvars);

        // 3) Design vars
        ndvs = assembler->get_num_dvs();
        d_dvs = DeviceVec<T>(ndvs, /*initial=*/0.02);

        // 4) Create solver
        auto kmat = createBsrMat<Assembler, VecType<T>>(*assembler);
        solver = std::make_unique<StructSolver>(*assembler, kmat, CUSPARSE::direct_LU_solve<T>,
                                                /*print=*/true);

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

        // if (first_solve) {
        //     dvs_changed = true;
        //     first_solve = false;
        // }
        dvs_changed = true; // debug

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
        assembler->free();
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
    std::unique_ptr<Assembler> assembler;
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