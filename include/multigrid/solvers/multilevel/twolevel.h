#pragma once

#include <string>

#include "../solve_utils.h"

template <class GRID, bool full_approx_scheme = false>
class MultigridTwoLevelSolver : public BaseSolver {
   public:
    // generic multigrid cycle subpsace solver (can be V-cycle, W-cycle, etc.)
    // only considers two levels, but can be nested with other solvers

    MultigridTwoLevelSolver(cublasHandle_t &cublasHandle_, cusparseHandle_t &cusparseHandle_,
                            GRID *fine_grid, GRID *coarse_grid, BaseSolver *coarse_solver_,
                            SolverOptions options, bool is_coarse_direct_ = false)
        : fine_grid(fine_grid),
          coarse_grid(coarse_grid),
          coarse_solver(coarse_solver_),
          is_coarse_direct(is_coarse_direct_),
          options(options) {
        cycle = "V";  // default cycle type
    }

    // nothing
    void update_after_assembly(DeviceVec<T> &vars) {}

    void set_print(bool print) { options.print = print; }
    void set_abs_tol(T atol) { options.atol = atol; }
    void set_rel_tol(T rtol) { options.rtol = rtol; }
    int get_num_iterations() { return n_steps; }
    void factor() {}
    void set_cycle_type(std::string cycle_) {
        // printf("setting cycle type to %s\n", cycle_.c_str());
        cycle = cycle_;
    }

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
            if (is_coarse_direct || cycle == "V") {
                // then only only one coarse solve
                coarse_solver->solve(coarse_grid->d_defect, coarse_grid->d_soln);
            } else if (cycle == "W") {
                // set inner cycle to 'W' and do two inner W-cycle solves (defn of W-cycle)
                coarse_solver->set_cycle_type("W");
                coarse_solver->solve(coarse_grid->d_defect, coarse_grid->d_soln);
                coarse_solver->solve(coarse_grid->d_defect, coarse_grid->d_soln);
            } else if (cycle == "F" || cycle == "Fsym") {
                // do inner F-cycle and then V-cycle (defn of F-cycle)
                coarse_solver->set_cycle_type("F");
                coarse_solver->solve(coarse_grid->d_defect, coarse_grid->d_soln);
                coarse_solver->set_cycle_type("V");
                coarse_solver->solve(coarse_grid->d_defect, coarse_grid->d_soln);
                if (cycle == "Fsym") {
                    // symmetric modification to make it work with PCG still (not usual F-cycle)
                    coarse_solver->set_cycle_type("F");
                    coarse_solver->solve(coarse_grid->d_defect, coarse_grid->d_soln);
                }
            } else {
                printf("ERROR: cycle type not valid option 'V', 'W', 'F'\n");
            }

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
                            "V-cycle DirectLU-GMG converged in %d steps from %.2e defect nrm "
                            "to "
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

    std::string cycle;  // 'V', 'W' or 'F'
    bool is_coarse_direct;
    // right now the directLU solve code is stored on the coarse grid.. may pull it off of there
    // when I cleanup the code (and put it here)
};