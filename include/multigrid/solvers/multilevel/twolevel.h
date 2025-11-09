#pragma once

#include "../solve_utils.h"

template <class GRID, bool full_approx_scheme = false>
class MultigridTwoLevelSolver : public BaseSolver {
   public:
    // generic multigrid cycle subpsace solver (can be V-cycle, W-cycle, etc.)
    // only considers two levels, but can be nested with other solvers

    MultigridTwoLevelSolver(cublasHandle_t &cublasHandle_, cusparseHandle_t &cusparseHandle_,
                            GRID *fine_grid, GRID *coarse_grid, BaseSolver *coarse_solver_,
                            SolverOptions options)
        : fine_grid(fine_grid),
          coarse_grid(coarse_grid),
          coarse_solver(coarse_solver_),
          options(options) {}

    // nothing
    void update_after_assembly(DeviceVec<T> &vars) {}

    void set_print(bool print) { options.print = print; }
    void set_abs_tol(T atol) { options.atol = atol; }
    void set_rel_tol(T rtol) { options.rtol = rtol; }
    int get_num_iterations() { return n_steps; }

    bool solve(DeviceVec<T> rhs, DeviceVec<T> soln, bool check_conv = false) {
        // printf("in subpsace solve\n");

        // approximate fine grid solve using V-cycle and coarse direct solves
        bool is_perm = false;
        fine_grid->setDefect(rhs, is_perm);
        fine_grid->zeroSolution();

        T init_defect_nrm;
        if (check_conv || options.print) init_defect_nrm = fine_grid->getDefectNorm();
        bool converged = false;

        for (int icycle = 0; icycle < options.ncycles; icycle++) {
            // printf("icycle %d / %d\n", icycle, options.ncycles);

            n_steps = icycle + 1;

            // presmooth and restrict
            fine_grid->smoothDefect(options.nsmooth, options.debug, options.nsmooth - 1);
            if constexpr (full_approx_scheme) {
                coarse_grid->restrict_loads(fine_grid->d_rhs);
            } else {
                coarse_grid->restrict_defect(fine_grid->d_defect);
            }

            // coarse grid solve
            coarse_solver->solve(coarse_grid->d_defect, coarse_grid->d_soln);

            // prolongate and postsmooth
            fine_grid->prolongate(coarse_grid->d_soln);
            fine_grid->smoothDefect(options.nsmooth, options.debug, options.nsmooth - 1);

            // check convergence if flag on
            if (check_conv) {
                T defect_nrm = fine_grid->getDefectNorm();
                if (options.print && icycle % options.print_freq == 0)
                    printf("v-cycle step %d, ||defect|| = %.3e\n", icycle, defect_nrm);

                if (defect_nrm < options.atol + options.rtol * init_defect_nrm) {
                    if (options.print) {
                        printf(
                            "V-cycle DirectLU-GMG converged in %d steps from %.2e defect nrm to "
                            "%.2e\n",
                            icycle + 1, init_defect_nrm, defect_nrm);
                    }
                    converged = true;
                    break;
                }
            }
        }  // end of cycles

        fine_grid->getSolution(soln, is_perm);
        if (check_conv) {
            return !converged;
        } else {
            return false;  // no fail
        }
    }

    void free() {
        if (fine_grid) fine_grid->free();
        if (coarse_grid) coarse_grid->free();
        if (coarse_solver) coarse_solver->free();
    }

    SolverOptions options;

   private:
    GRID *fine_grid, *coarse_grid;
    BaseSolver *coarse_solver;
    int n_steps = 0;
    // right now the directLU solve code is stored on the coarse grid.. may pull it off of there
    // when I cleanup the code (and put it here)
};