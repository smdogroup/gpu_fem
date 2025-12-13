#include "pde.h"

enum Solver { DAMPED_JACOBI, GAUSS_SEIDEL };

template <typename T, Solver solver_type>
class PoissonGeometricMultigrid {
    /* 2d poisson geomtric multigrid solver class */
    using GRID = PoissonSolver<T>;

   public:
    PoissonGeometricMultigrid(int nxe_) : nxe(nxe_) {
        // number of levels
        int log2_nxe = 0, nxe_copy = nxe;
        while (nxe_copy >>= 1) ++log2_nxe;
        n_levels = log2_nxe - 1;

        // check nxe is power of 2 (for good geom multigrid property, convenience)
        int nxe_power_2 = 1 << log2_nxe;
        assert(nxe_power_2 == nxe);

        // default damped jacobi setting
        setupDampedJacobi(2.0 / 3.0);

        // construct grids
        initGrids();
    }

    void initGrids() {
        grids = new GRID[n_levels];
        int c_nxe = nxe;
        bool red_black_order = solver_type == GAUSS_SEIDEL;
        for (int ilevel = 0; ilevel < n_levels; ilevel++, c_nxe /= 2) {
            printf("level %d, making poisson solver with nxe %d elems\n", ilevel, c_nxe);
            grids[ilevel] = GRID(c_nxe, red_black_order);
        }
    }

    void setupDampedJacobi(T omega = 2.0 / 3.0) {
        // TODO : later could set V-cycle vs different solve types?
        setup = true;
        jacobi_omega = omega;
    }

    void vcycle_solve(int pre_smooth, int post_smooth, int n_vcycles = 100, bool print = false,
                      int inner_solve_iters = 100, T atol = 1e-6, T rtol = 1e-6) {
        // init defect nrm
        T init_defect_nrm = grids[0].getDefectNorm();
        printf("init defect nrm, ||defect|| = %.2e\n", init_defect_nrm);

        for (int i_vcycle = 0; i_vcycle < n_vcycles; i_vcycle++) {
            // printf("V cycle step %d\n", i_vcycle);

            // go down each level smoothing and restricting until lowest level
            for (int i_level = 0; i_level < n_levels; i_level++) {
                // if not last  (pre-smooth)
                if (i_level < n_levels - 1) {
                    if (print) printf("\tlevel %d pre-smooth\n", i_level);

                    // pre-smooth
                    if (solver_type == DAMPED_JACOBI) {
                        grids[i_level].dampedJacobiDefect(pre_smooth, jacobi_omega, print,
                                                          pre_smooth - 1);
                    } else {  // GAUSS_SEIDEL
                        grids[i_level].redBlackGaussSeidelDefect(pre_smooth, print, pre_smooth - 1);
                    }

                    // restrict defect
                    grids[i_level + 1].restrict_defect(grids[i_level].d_perm,
                                                       grids[i_level].d_defect);
                } else {
                    if (print) printf("\tlevel %d full-solve\n", i_level);

                    // coarsest grid full solve (so prob want inner_solve_iters fairly high.. but
                    // since coarse grid should work usually)
                    if (solver_type == DAMPED_JACOBI) {
                        grids[i_level].dampedJacobiDefect(inner_solve_iters, jacobi_omega, print,
                                                          inner_solve_iters - 1);
                    } else {  // GAUSS_SEIDEL
                        grids[i_level].redBlackGaussSeidelDefect(inner_solve_iters, print,
                                                                 inner_solve_iters - 1);
                    }
                }
            }

            // now go back up the hierarchy
            for (int i_level = n_levels - 2; i_level >= 0; i_level--) {
                // get coarse-fine correction from coarser grid to this grid
                grids[i_level].prolongate(grids[i_level + 1].d_iperm, grids[i_level + 1].d_soln);

                if (print) printf("\tlevel %d post-smooth\n", i_level);

                // post-smooth
                if (solver_type == DAMPED_JACOBI) {
                    grids[i_level].dampedJacobiDefect(post_smooth, jacobi_omega, print,
                                                      post_smooth - 1);
                } else {  // GAUSS_SEIDEL
                    grids[i_level].redBlackGaussSeidelDefect(post_smooth, print, post_smooth - 1);
                }
            }

            // compute fine grid defect of V-cycle
            T defect_nrm = grids[0].getDefectNorm();
            printf("v-cycle step %d, ||defect|| = %.3e\n", i_vcycle, defect_nrm);

            if (defect_nrm < atol + rtol * init_defect_nrm) {
                printf("V-cycle GMG converged in %d steps\n", i_vcycle + 1);
                break;
            }
        }
    }

    void free() {
        for (int ilevel = 0; ilevel < n_levels; ilevel++) {
            grids[ilevel].free();
        }
    }

    int nxe, n_levels;
    bool setup;
    GRID *grids;

    // jacobi settings
    T jacobi_omega;
    int jacobi_inner_solve_iters;
};