#include "include/pde.h"

int main() {
    // make poisson solver and try and run it..

    using T = float; // double
    using GRID = PoissonSolver<T>;

    // set finest grid size (must be power of 2 for this case..)
    // int nxe = 8192; // fails too large problem for 3060 Ti GPU
    int nxe = 4096; // solves in 3 seconds the 16M problem on my local 3060 Ti GPU
    // int nxe = 2048;
    // int nxe = 1024;
    // int nxe = 256;
    // int nxe = 32;

    int log2_nxe = 0, nxe_copy = nxe;
    while (nxe_copy >>= 1) ++log2_nxe;
    int n_levels = log2_nxe - 1;
    printf("nxe %d, log2(nxe) %d, n_levels %d\n", nxe, log2_nxe, n_levels);

    bool red_black_order = false; // just lexigraphic or natural ordering for the damped jacobi

    GRID *grids = new GRID[n_levels]; 
    int c_nxe = nxe;
    for (int ilevel = 0; ilevel < n_levels; ilevel++, c_nxe /= 2) {
        printf("level %d, making poisson solver with nxe %d elems\n", ilevel, c_nxe);
        grids[ilevel] = GRID(c_nxe, red_black_order);
    }

    /* DEBUG before full V-cycles */
    // T grid1_def_nrm = grids[n_levels-2].getDefectNorm();

    // // test restriction on coarsest grid
    // grids[n_levels-1].restrict_defect(grids[n_levels-2].d_defect);
    // // cudaDeviceSynchronize();

    // T *h_coarse_defect0 = grids[n_levels-1].d_defect.createHostVec().getPtr();
    // printf("coarsest grid defect: ");
    // printVec<T>(grids[n_levels-1].N, h_coarse_defect0);

    // T grid2_def_nrm = grids[n_levels-1].getDefectNorm(); 

    // printf("grid 1 |defect| = %.2e => grid 2 |defect| = %.2e\n", grid1_def_nrm, grid2_def_nrm);

    // return 0;
    /* end of DEBUG section */

    // now try multigrid V-cycle solves here

    // /* try solve here.. */
    // int pre_smooth = 1, post_smooth = 1;
    // int pre_smooth = 3, post_smooth = 3; 
    int pre_smooth = 5, post_smooth = 5; // more smoothing steps just because it's damped jacobi (which is weaker but more parallel)
    // int pre_smooth = 30, post_smooth = 30;
    T omega = 2.0 / 3.0;

    // bool print = true;
    bool print = false;

    int n_vcycles = 100;
    // int n_vcycles = 1;

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
                grids[i_level].dampedJacobiDefect(pre_smooth, omega, print, pre_smooth - 1);

                // restrict defect
                grids[i_level + 1].restrict_defect(grids[i_level].d_perm, grids[i_level].d_defect);
            } else {
                if (print) printf("\tlevel %d full-solve\n", i_level);

                // print the defect here..
                // T *h_coarse_defect = grids[i_level].d_defect.createHostVec().getPtr();
                // printf("h_coarse_defect: ");
                // printVec<T>(grids[i_level].N, h_coarse_defect);

                // full-solve on last grid (of current defect)
                grids[i_level].dampedJacobiDefect(100, omega, print, 99);
            }
        }

        // now go back up the hierarchy
        for (int i_level = n_levels - 2; i_level >= 0; i_level--) {
            // get coarse-fine correction from coarser grid to this grid
            grids[i_level].prolongate(grids[i_level + 1].d_iperm, grids[i_level + 1].d_soln);

            if (print) printf("\tlevel %d post-smooth\n", i_level);

            // post-smooth
            grids[i_level].dampedJacobiDefect(post_smooth, omega, print, post_smooth - 1);
        }

        // compute fine grid defect of V-cycle
        T defect_nrm = grids[0].getDefectNorm();
        printf("v-cycle step %d, ||defect|| = %.3e\n", i_vcycle, defect_nrm);

        if (defect_nrm < 1e-6 * init_defect_nrm) {
            printf("V-cycle GMG converged in %d steps\n", i_vcycle + 1);
            break;
        }

    }

    // free
    for (int ilevel = 0; ilevel < n_levels; ilevel++) {
        grids[ilevel].free();
    }

    return 0;
};