// generic Kcycle solver
#pragma once

#include <iostream>
#include <sstream>
#include <string>

#include "../solve_utils.h"

template <class GRID, class CoarseSolver, class SubspaceSolver, class KrylovSolver>
class MultilevelKcycleSolver {
   public:
    using T = double;

    MultilevelKcycleSolver() = default;

    template <class Basis>
    void init_prolongations() {
        /* pass in coarse assembler data for each prolongation operator */
        // 0 is the finest grid, nlevels-1 is the coarsest grid here
        printf("prolong assembly => \n\t");
        for (int ilevel = 0; ilevel < getNumLevels() - 1; ilevel++) {
            printf("/ level %d ", ilevel);
            grids[ilevel].prolongation->init_coarse_data(grids[ilevel + 1].assembler);
            grids[ilevel + 1].restriction =
                grids[ilevel].prolongation;  // copy prolong to restriction on coarser grid
            if (ilevel % 3 == 0 && ilevel > 0) printf(" /\n\t");
        }
        printf(" /\n");
        // do matrix smoothing (if not possible depending upon prolong type, it will be skipped
        // inside) if (grids[0].smooth_matrix_iters > 0) {
        //     printf("attempting to smooth matrices\n");
        // }
        // for (int ilevel = 0; ilevel < getNumLevels() - 1; ilevel++) {
        //     grids[ilevel].smoothMatrix(grids[ilevel].smooth_matrix_iters);
        // }
    }

    int getNumLevels() { return grids.size(); }

    double get_memory_usage_mb() {
        // get total memory usage of each Kmat across all grids
        double total_mem = 0.0;
        for (int ilevel = 0; ilevel < getNumLevels(); ilevel++) {
            total_mem += grids[ilevel].get_memory_usage_mb();
        }
        return total_mem;
    }

    int get_num_iterations() { return outer_solver->get_num_iterations(); }
    int get_num_lin_solves() { return num_lin_solves; }
    void factor() {}

    bool solve(DeviceVec<T> rhs, DeviceVec<T> soln, bool check_conv = true) {
        num_lin_solves++;
        return outer_solver->solve(rhs, soln, check_conv);
    }

    void update_after_assembly(DeviceVec<T> &vars) {
        // update states and do assemblies of coarse grids first
        bool perm = true;
        grids[0].setStateVars(vars, perm);  // set vars into fine grid & perm (VIS to SOLVE order)
        _update_coarse_grid_states();       // restrict state vars to coarse grid+assemblers
        _update_coarse_grid_jacobians();    // compute coarse grid NL stiffness matrices

        /* update matrices after new assembly */
        for (int ilevel = 0; ilevel < getNumLevels(); ilevel++) {
            grids[ilevel].update_after_assembly();
        }
        for (int ilevel = 0; ilevel < getNumLevels() - 1; ilevel++) {
            grids[ilevel].smoothMatrix(grids[ilevel].smooth_matrix_iters);
        }
        if (coarse_solver) coarse_solver->update_after_assembly(vars);
    }

    template <class Assembler>
    void debug_assembly() {
        /* debug and check that coarser grids, etc. have reasonable states */

        for (int ilevel = 0; ilevel < getNumLevels(); ilevel++) {
            int block_dim = grids[ilevel].block_dim;
            grids[ilevel].d_vars.permuteData(block_dim,
                                             grids[ilevel].d_perm);  // from SOLVE to VIS order
            auto h_vars = grids[ilevel].d_vars.createHostVec();
            grids[ilevel].d_vars.permuteData(block_dim, grids[ilevel].d_iperm);  // and undo perm
            std::stringstream outputFile;
            outputFile << "out/wing_debug_level_" << ilevel << ".vtk";
            printToVTK<Assembler, HostVec<T>>(grids[ilevel].assembler, h_vars, outputFile.str());
        }
    }

    void set_abs_tol(T rtol) { outer_solver->set_abs_tol(rtol); }
    void set_rel_tol(T rtol) { outer_solver->set_rel_tol(rtol); }
    void set_cycle_type(const std::string &cycle) {
        // set on the actual MG subspace preconditioner (the one that uses V/W/F)
        if (outer_solver) outer_solver->set_cycle_type(cycle);

        // set on everything else you happen to store (fine if redundant)
        for (BaseSolver *s : solvers) {
            if (s) s->set_cycle_type(cycle);
        }

        // // optional: also set on coarse_solver if it exists and might be another MG object
        // if (coarse_solver) coarse_solver->set_cycle_type(cycle);
    }

    void set_design_variables(DeviceVec<T> dvs) {
        for (int ilevel = 0; ilevel < getNumLevels(); ilevel++) {
            grids[ilevel].assembler.set_design_variables(dvs);
        }
    }

    void set_print(bool print) { outer_solver->set_print(print); }

    void init_outer_solver(cublasHandle_t &cublasHandle, cusparseHandle_t &cusparseHandle,
                           int n_smooth, int n_cycles, int n_krylov, T omega = 1.0, T atol = 1e-6,
                           T rtol = 1e-6, int print_freq = 1, bool print = true,
                           bool double_smooth = false) {
        // initialize objects, so we just do K-cycle on outer level

        int nvcyc_inner = 1, nvcyc_outer = n_cycles, nkcyc_inner = 2, nkcyc_outer = n_krylov;
        bool just_outer_krylov = true;

        // create the kcycle multigrid object
        // ----------------------------------

        // apply settings
        auto inner_subspaceOptions = SolverOptions(omega, n_smooth, nvcyc_inner);
        auto outer_subspaceOptions = SolverOptions(omega, n_smooth, nvcyc_outer);
        auto innerKrylovOptions = SolverOptions(omega, 0, nkcyc_inner);
        // bool symmetric = false;
        auto outerKrylovOptions =
            SolverOptions(omega, 0, nkcyc_outer, false, atol, rtol, print_freq);
        outerKrylovOptions.print = print;

        init_solvers(cublasHandle, cusparseHandle, inner_subspaceOptions, outer_subspaceOptions,
                     innerKrylovOptions, outerKrylovOptions, just_outer_krylov, double_smooth);
    }

    // create the hierarchy of solvers bottom-up
    void init_solvers(cublasHandle_t &cublasHandle, cusparseHandle_t &cusparseHandle,
                      SolverOptions inner_subspace_options, SolverOptions outer_subspace_options,
                      SolverOptions inner_krylov_options, SolverOptions outer_krylov_options,
                      bool just_outer_krylov = false, bool double_smooth = false) {
        // after you've created each grid, call this method to make the solver hierarchy (though you
        // can do this yourself if you wanted to)
        int nlevels = getNumLevels();
        int isolver = 0;
        for (int ilevel = nlevels - 1; ilevel >= 0; ilevel--) {
            if (ilevel == nlevels - 1) {
                // create coarse grid direct solver
                coarse_solver = new CoarseSolver(cublasHandle, cusparseHandle,
                                                 grids[ilevel].assembler, grids[ilevel].Kmat);
                solvers.push_back(coarse_solver);
            } else {
                bool is_coarse_direct =
                    ilevel == nlevels - 2;  // means level below is the coarse direct solver
                if (just_outer_krylov) {
                    // first make the subspace solver
                    auto subspace_options =
                        ilevel == 0 ? outer_subspace_options : inner_subspace_options;
                    auto copy_options = SolverOptions(subspace_options);
                    if (double_smooth) copy_options.nsmooth *= (1 << ilevel);

                    BaseSolver *subspace_solver = new SubspaceSolver(
                        cublasHandle, cusparseHandle, &grids[ilevel], &grids[ilevel + 1],
                        solvers[isolver - 1], copy_options, is_coarse_direct);
                    if (ilevel != 0) solvers.push_back(subspace_solver);

                    // then make the krylov solver at this level with the subspace solver as
                    // preconditioner
                    if (ilevel == 0) {
                        auto krylov_options =
                            ilevel == 0 ? outer_krylov_options : inner_krylov_options;
                        BaseSolver *krylov_solver =
                            new KrylovSolver(cublasHandle, cusparseHandle, &grids[ilevel],
                                             subspace_solver, krylov_options, ilevel);
                        solvers.push_back(krylov_solver);
                        outer_solver = krylov_solver;
                    }

                } else {
                    // all levels have krylov
                    // first make the subspace solver
                    auto subspace_options =
                        ilevel == 0 ? outer_subspace_options : inner_subspace_options;
                    auto copy_options = SolverOptions(subspace_options);
                    if (double_smooth) copy_options.nsmooth *= (1 << ilevel);
                    BaseSolver *subspace_solver = new SubspaceSolver(
                        cublasHandle, cusparseHandle, &grids[ilevel], &grids[ilevel + 1],
                        solvers[isolver - 1], copy_options, is_coarse_direct);

                    // then make the krylov solver at this level with the subspace solver as
                    // preconditioner
                    auto krylov_options = ilevel == 0 ? outer_krylov_options : inner_krylov_options;
                    BaseSolver *krylov_solver =
                        new KrylovSolver(cublasHandle, cusparseHandle, &grids[ilevel],
                                         subspace_solver, krylov_options, ilevel);
                    solvers.push_back(krylov_solver);
                    if (ilevel == 0) {
                        outer_solver = krylov_solver;
                    }
                }
            }
            isolver++;
        }
    }

    bool solve() { return this->solve(grids[0].d_defect, grids[0].d_soln); }

    void free() {
        if (is_free) return;
        is_free = true;  // now it's freed

        int n_levels = getNumLevels();
        for (int ilevel = 0; ilevel < n_levels; ilevel++) {
            grids[ilevel].free();
            solvers[ilevel]->free();
        }
        outer_solver->free();
    }

    // private:
    BaseSolver *outer_solver;
    BaseSolver *coarse_solver;
    std::vector<BaseSolver *> solvers;  // stored coarse to fine
    std::vector<GRID> grids;            // stored fine to coarse

   private:
    bool is_free = false;
    int num_lin_solves = 0;

    // helper methods for updating after jacobian update
    void _update_coarse_grid_states() {
        /* update all state variables from fine grid down to coarser grids (for nonlinear solutions)
         */
        for (int ilevel = 0; ilevel < getNumLevels() - 1; ilevel++) {
            grids[ilevel + 1].restrict_soln(grids[ilevel].d_vars);
        }
    }

    void _update_coarse_grid_jacobians() {
        /* update all state variables from fine grid down to coarser grids (for nonlinear solutions)
         */
        for (int ilevel = 1; ilevel < getNumLevels(); ilevel++) {
            grids[ilevel].assembler.add_jacobian_fast(grids[ilevel].Kmat);
            grids[ilevel].assembler.apply_bcs(grids[ilevel].Kmat);
        }
    }
};