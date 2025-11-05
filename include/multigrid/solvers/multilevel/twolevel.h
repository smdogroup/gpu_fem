#pragma once

#include "../solve_utils.h"

template <class GRID, bool is_nonlinear = false>
class MultigridTwoLevelSolver : public BaseSolver {
public:
    // generic multigrid cycle subpsace solver (can be V-cycle, W-cycle, etc.)
    // only considers two levels, but can be nested with other solvers

    MultigridTwoLevelSolver(GRID *fine_grid, GRID *coarse_grid, BaseSolver *coarse_solver_, SolverOptions options) : 
        fine_grid(fine_grid), coarse_grid(coarse_grid), coarse_solver(coarse_solver_), options(options) { }

    // nothing
    void update_after_assembly(DeviceVec<T> &vars) {}

    void set_abs_tol(T atol) {options.atol = atol;}
    void set_rel_tol(T rtol) {options.rtol = rtol;}

    void solve(DeviceVec<T> rhs, DeviceVec<T> soln, bool check_conv = false) {
        // printf("in subpsace solve\n");

        // approximate fine grid solve using V-cycle and coarse direct solves
        bool is_perm = false;
        fine_grid->setDefect(rhs, is_perm);
        fine_grid->zeroSolution();

        T init_defect_nrm;
        if (check_conv || options.print) init_defect_nrm = fine_grid->getDefectNorm();

        for (int icycle = 0; icycle < options.ncycles; icycle++) {

            // printf("icycle %d / %d\n", icycle, options.ncycles);

            // presmooth and restrict
            fine_grid->smoothDefect(options.nsmooth, options.debug, options.nsmooth-1);
            if constexpr (is_nonlinear) {
                coarse_grid->restrict_loads(fine_grid->d_defect); // d_rhs
            } else {
                coarse_grid->restrict_defect(fine_grid->d_defect);
            }
            

            // coarse grid solve
            coarse_solver->solve(coarse_grid->d_defect, coarse_grid->d_soln);

            // prolongate and postsmooth
            fine_grid->prolongate(coarse_grid->d_soln);
            fine_grid->smoothDefect(options.nsmooth, options.debug, options.nsmooth-1);

            // check convergence if flag on
            if (check_conv || options.print) {
                T defect_nrm = fine_grid->getDefectNorm();
                if (icycle % options.print_freq == 0)
                    printf("v-cycle step %d, ||defect|| = %.3e\n", icycle, defect_nrm);

                if (check_conv && defect_nrm < options.atol + options.rtol * init_defect_nrm) {
                    printf("V-cycle DirectLU-GMG converged in %d steps from %.2e defect nrm to %.2e\n",
                        icycle + 1, init_defect_nrm, defect_nrm);
                    break;
                }
            }
        } // end of cycles

        fine_grid->getSolution(soln, is_perm);
    }

    void free() {
        // TODO
        return;
    }

    SolverOptions options;

private:
    GRID *fine_grid, *coarse_grid;
    BaseSolver *coarse_solver;
    // right now the directLU solve code is stored on the coarse grid.. may pull it off of there when I cleanup the code (and put it here)
};