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
                            SolverOptions options, bool is_coarse_direct_ = false, int level_ = 0)
        : fine_grid(fine_grid),
          coarse_grid(coarse_grid),
          coarse_solver(coarse_solver_),
          is_coarse_direct(is_coarse_direct_),
          level(level_),
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
        bool is_perm = false;

        int fine_level = level;
        int coarse_level = level + 1;

        fine_grid->setDefect(rhs, is_perm);
        fine_grid->zeroSolution();

        T init_defect_nrm = fine_grid->getDefectNorm();
        T converged_nrm = options.atol + options.rtol * init_defect_nrm;
        T final_defect_nrm = init_defect_nrm;

        bool converged = false;

        if (options.print && level == 0) {
            printf("\n================ TWO-LEVEL GMG START ================\n");
            printf("initial defect norm = %.6e\n", init_defect_nrm);
        }
        if (options.print) {
            printf("level %d MG start with ncyc = %d\n", level, options.ncycles);
        }

        for (int icycle = 0; icycle < options.ncycles; icycle++) {
            n_steps = icycle + 1;

            if (options.print && level == 0) {
                printf("\n===================================================\n");
                printf("V-cycle step %d\n", icycle);
                printf("===================================================\n");

                printf("[level %d] BEFORE pre-smooth\n", fine_level);
                printf("    defect norm = %.6e\n", fine_grid->d_defect.norm());
                printf("    soln   norm = %.6e\n", fine_grid->d_soln.norm());
            }

            // --------------------------------------------
            // presmooth
            // --------------------------------------------

            fine_grid->smoothDefect(options.nsmooth, options.debug, options.nsmooth - 1);

            if (options.print) {
                printf("[level %d] AFTER pre-smooth\n", fine_level);
                printf("    defect norm = %.6e\n", fine_grid->d_defect.norm());
                printf("    soln   norm = %.6e\n", fine_grid->d_soln.norm());
            }

            // --------------------------------------------
            // restrict
            // --------------------------------------------

            if constexpr (full_approx_scheme) {
                coarse_grid->restrict_loads(fine_grid->d_rhs);
            } else {
                coarse_grid->restrict_defect(fine_grid->d_defect);
            }

            if (options.print) {
                printf("[level %d -> %d] AFTER restrict\n", fine_level, coarse_level);
                printf("    defect norm = %.6e\n", coarse_grid->d_defect.norm());
                printf("    soln   norm = %.6e\n", coarse_grid->d_soln.norm());
            }

            // --------------------------------------------
            // coarse solve
            // --------------------------------------------

            if (options.print) {
                printf("[level %d] BEFORE pre-smooth\n", coarse_level);
                printf("    defect norm = %.6e\n", coarse_grid->d_defect.norm());
                printf("    soln norm = %.6e\n", coarse_grid->d_soln.norm());
            }

            if (is_coarse_direct || cycle == "V") {
                coarse_solver->solve(coarse_grid->d_defect, coarse_grid->d_soln);

            } else if (cycle == "W") {
                coarse_solver->set_cycle_type("W");

                coarse_solver->solve(coarse_grid->d_defect, coarse_grid->d_soln);
                coarse_solver->solve(coarse_grid->d_defect, coarse_grid->d_soln);

            } else if (cycle == "F" || cycle == "Fsym") {
                coarse_solver->set_cycle_type("F");
                coarse_solver->solve(coarse_grid->d_defect, coarse_grid->d_soln);

                coarse_solver->set_cycle_type("V");
                coarse_solver->solve(coarse_grid->d_defect, coarse_grid->d_soln);

                if (cycle == "Fsym") {
                    coarse_solver->set_cycle_type("F");
                    coarse_solver->solve(coarse_grid->d_defect, coarse_grid->d_soln);
                }

            } else {
                printf("ERROR: cycle type not valid option 'V', 'W', 'F'\n");
            }

            if (options.print) {
                printf("[level %d] AFTER post-smooth\n", coarse_level);
                printf("    defect norm = %.6e\n", coarse_grid->d_defect.norm());
                printf("    soln norm = %.6e\n", coarse_grid->d_soln.norm());
            }

            // --------------------------------------------
            // prolongation
            // --------------------------------------------

            if (options.print) {
                printf("[level %d] BEFORE prolongation\n", fine_level);
                printf("    defect norm = %.6e\n", fine_grid->d_defect.norm());
                printf("    soln   norm = %.6e\n", fine_grid->d_soln.norm());
            }

            fine_grid->prolongate(coarse_grid->d_soln);

            if (options.print) {
                printf("[level %d] AFTER prolongation\n", fine_level);
                printf("    defect norm = %.6e\n", fine_grid->d_defect.norm());
                printf("    soln   norm = %.6e\n", fine_grid->d_soln.norm());
            }

            // --------------------------------------------
            // postsmooth
            // --------------------------------------------

            fine_grid->smoothDefect(options.nsmooth, options.debug, options.nsmooth - 1);

            if (options.print) {
                printf("[level %d] AFTER post-smooth\n", fine_level);
                printf("    defect norm = %.6e\n", fine_grid->d_defect.norm());
                printf("    soln   norm = %.6e\n", fine_grid->d_soln.norm());
            }

            // --------------------------------------------
            // convergence check
            // --------------------------------------------

            T defect_nrm = fine_grid->getDefectNorm();
            final_defect_nrm = defect_nrm;

            if (options.print && level == 0) {
                printf("\n[V-cycle %d] fine defect norm = %.6e\n", icycle, defect_nrm);
            }

            if (check_conv && options.print && icycle % options.print_freq == 0 && level == 0) {
                printf("V-cycle step %d, ||defect|| = %.3e\n", icycle, defect_nrm);
            }

            if (check_conv && defect_nrm < converged_nrm) {
                if (options.print && level == 0) {
                    printf("\nGMG converged in %d V-cycles\n", icycle + 1);
                    printf("final defect norm = %.6e\n", defect_nrm);
                    printf("target norm       = %.6e\n", converged_nrm);
                }

                converged = true;
                break;
            }
        }

        fine_grid->getSolution(soln, is_perm);

        if (options.print && level == 0) {
            printf("\n================ TWO-LEVEL GMG END ==================\n");
            printf("final fine solution norm = %.6e\n", soln.norm());
            printf("final fine defect norm   = %.6e\n", final_defect_nrm);
            printf("=====================================================\n\n");
        }

        if (check_conv) {
            return !converged;
        } else {
            return false;
        }
    }

    void free() {
        if (fine_grid) fine_grid->free();
        if (coarse_grid) coarse_grid->free();
        if (coarse_solver) coarse_solver->free();
    }

    SolverOptions options;
    int level = 0;

   private:
    GRID *fine_grid, *coarse_grid;
    BaseSolver *coarse_solver;
    int n_steps = 0;

    std::string cycle;  // 'V', 'W' or 'F'
    bool is_coarse_direct;
    // right now the directLU solve code is stored on the coarse grid.. may pull it off of there
    // when I cleanup the code (and put it here)
};