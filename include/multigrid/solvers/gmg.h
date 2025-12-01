#pragma once

// original multilevel Vcycle solver
// V1 I should say..
#include "string"

template <class GRID, class CoarseSolver>
class GeometricMultigridSolver {
    /* a standard geomtric multigrid solver class */
    using T = double;

   public:
    GeometricMultigridSolver() = default;

    template <class Basis>
    void init_prolongations() {
        /* pass in coarse assembler data for each prolongation operator */
        // 0 is the finest grid, nlevels-1 is the coarsest grid here
        printf("create prolongation nz pattern\n");
        for (int ilevel = 0; ilevel < getNumLevels() - 1; ilevel++) {
            grids[ilevel].prolongation->init_coarse_data(grids[ilevel + 1].assembler);
            grids[ilevel + 1].restriction =
                grids[ilevel].prolongation;  // copy prolong to restriction on coarser grid
        }
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

    void set_design_variables(DeviceVec<T> dvs) {
        for (int ilevel = 0; ilevel < getNumLevels(); ilevel++) {
            grids[ilevel].assembler.set_design_variables(dvs);
        }
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
        if (coarse_solver) coarse_solver->update_after_assembly(vars);
    }

    void smooth_matrix() {
        for (int ilevel = 0; ilevel < getNumLevels() - 1; ilevel++) {
            grids[ilevel].smoothMatrix(grids[ilevel].smooth_matrix_iters);
        }
    }

    void vcycle_solve(int starting_level, int pre_smooth, int post_smooth, int n_vcycles = 100,
                      bool print = false, T atol = 1e-6, T rtol = 1e-6, bool double_smooth = false,
                      int print_freq = 1, bool time = false, bool debug = false) {
        // init defect nrm
        T init_defect_nrm = grids[0].getDefectNorm();
        if (print) printf("V-cycles: ||init_defect|| = %.2e\n", init_defect_nrm);
        T fin_defect_nrm = init_defect_nrm;
        int n_steps = n_vcycles;
        int n_levels = getNumLevels();

        for (int i_vcycle = 0; i_vcycle < n_vcycles; i_vcycle++) {
            /* restrict + pre-smooth down to one before coarse level */
            // ----------------------------------------------------------------

            for (int i_level = starting_level; i_level < n_levels - 1; i_level++) {
                if (time) CHECK_CUDA(cudaDeviceSynchronize());
                auto pre_smooth_time = std::chrono::high_resolution_clock::now();

                int inner_pre_smooth = pre_smooth * (double_smooth ? 1 << i_level : 1);
                grids[i_level].smoothDefect(inner_pre_smooth, debug, inner_pre_smooth - 1);

                if (time) {
                    CHECK_CUDA(cudaDeviceSynchronize());
                    auto post_smooth_time = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> smooth_time1 = post_smooth_time - pre_smooth_time;
                    printf("\tsmoothDefect[%d]-down in %.2e sec\n", i_level, smooth_time1.count());
                }
                auto pre_restr_time = std::chrono::high_resolution_clock::now();

                // restrict defect
                grids[i_level + 1].restrict_defect(grids[i_level].d_defect);

                if (time) {
                    CHECK_CUDA(cudaDeviceSynchronize());
                    auto post_restr_time = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> restr_time = post_restr_time - pre_restr_time;
                    printf("\trestr-defect[%d] in %.2e sec\n", i_level, restr_time.count());
                }
            }

            /* coarse solve */
            // -----------------------------------------------------------

            if (debug) printf("\t--level %d full-solve\n", n_levels - 1);
            if (time) CHECK_CUDA(cudaDeviceSynchronize());
            auto pre_direct_time = std::chrono::high_resolution_clock::now();

            // coarsest grid full solve
            coarse_solver->solve(grids[n_levels - 1].d_defect, grids[n_levels - 1].d_soln);

            if (time) {
                CHECK_CUDA(cudaDeviceSynchronize());
                auto post_direct_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> direct_time = post_direct_time - pre_direct_time;
                printf("\tdirect-solve[%d] in in %.2e sec\n", n_levels - 1, direct_time.count());
            }

            /* prolongations + post-smooths back up the levels */
            // ----------------------------------------------------------------------------

            for (int i_level = n_levels - 2; i_level >= starting_level; i_level--) {
                if (time) CHECK_CUDA(cudaDeviceSynchronize());
                auto pre_prolong_time = std::chrono::high_resolution_clock::now();

                // get coarse-fine correction from coarser grid to this grid
                grids[i_level].prolongate(grids[i_level + 1].d_soln);

                if (time) {
                    CHECK_CUDA(cudaDeviceSynchronize());
                    auto post_prolong_time = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> prolong_time =
                        post_prolong_time - pre_prolong_time;
                    printf("\tprolong-soln[%d] up in in %.2e sec\n", i_level, prolong_time.count());
                }
                auto pre_smooth_up_time = std::chrono::high_resolution_clock::now();

                // post-smooth
                int inner_post_smooth = post_smooth * (double_smooth ? 1 << i_level : 1);
                grids[i_level].smoothDefect(inner_post_smooth, debug, inner_post_smooth - 1);

                if (time) {
                    CHECK_CUDA(cudaDeviceSynchronize());
                    auto post_smooth_up_time = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> smooth_up_time =
                        post_smooth_up_time - pre_smooth_up_time;
                    printf("\tpost-smooth[%d] up in in %.2e sec\n", i_level,
                           smooth_up_time.count());
                }
            }

            /* compute fine grid defect of V-cycle */
            // -----------------------------------------------------

            T defect_nrm = grids[starting_level].getDefectNorm();
            fin_defect_nrm = defect_nrm;
            if (i_vcycle % print_freq == 0 && print)
                printf("v-cycle step %d, ||defect|| = %.3e\n", i_vcycle, defect_nrm);
            if (time) CHECK_CUDA(cudaDeviceSynchronize());

            if (defect_nrm < atol + rtol * init_defect_nrm && print) {
                printf("V-cycle GMG converged in %d steps to defect nrm %.2e from init_nrm %.2e\n",
                       i_vcycle + 1, defect_nrm, init_defect_nrm);
                n_steps = i_vcycle + 1;
                break;
            }
        }

        if (print)
            printf(
                "done with v-cycle solve from start level %d, conv %.2e to %.2e ||defect|| in %d "
                "steps\n",
                starting_level, init_defect_nrm, fin_defect_nrm, n_steps);
    }

    void fcycle_solve(int starting_level, int pre_smooth, int post_smooth, int n_fcycles = 100,
                      bool print = false, T atol = 1e-6, T rtol = 1e-6, bool double_smooth = false,
                      int print_freq = 1, bool time = false, bool debug = false) {
        // init defect nrm
        T init_defect_nrm = grids[0].getDefectNorm();
        if (print) printf("F-cycles: ||init_defect|| = %.2e\n", init_defect_nrm);
        T fin_defect_nrm = init_defect_nrm;
        int n_steps = n_fcycles;
        int n_levels = getNumLevels();
        int i_level = starting_level;

        for (int i_cycle = 0; i_cycle < n_fcycles; i_cycle++) {
            // pre-smooth and restrict
            if (i_level < n_levels - 1) {
                grids[i_level].smoothDefect(pre_smooth, debug, pre_smooth - 1);
                grids[i_level + 1].restrict_defect(grids[i_level].d_defect);
            }

            // call inner F-cycle 1 time
            if (i_level < n_levels - 1) {
                int pre_smooth_l = double_smooth ? pre_smooth * 2 : pre_smooth;
                int post_smooth_l = double_smooth ? post_smooth * 2 : post_smooth;

                fcycle_solve(i_level + 1, pre_smooth_l, post_smooth_l, 1, false, atol, rtol,
                             double_smooth);
            } else {
                coarse_solver->solve(grids[n_levels - 1].d_defect, grids[n_levels - 1].d_soln);
            }

            // post-smooth and restrict
            if (i_level < n_levels - 1) {
                grids[i_level].prolongate(grids[i_level + 1].d_soln);
                grids[i_level].smoothDefect(post_smooth, debug, post_smooth - 1);
            }

            // then call V-cycle at this level (one iteration)
            vcycle_solve(i_level, pre_smooth, post_smooth, 1, false, atol, rtol, double_smooth);

            // compute fine grid defect of V-cycle
            T defect_nrm = grids[i_level].getDefectNorm();
            fin_defect_nrm = defect_nrm;
            if (i_cycle % print_freq == 0 && print)
                printf("F-cycle step %d, ||defect|| = %.3e\n", i_cycle, defect_nrm);
            if (time) CHECK_CUDA(cudaDeviceSynchronize());

            if (defect_nrm < atol + rtol * init_defect_nrm && print) {
                printf("F-cycle GMG converged in %d steps to defect nrm %.2e from init_nrm %.2e\n",
                       i_cycle + 1, defect_nrm, init_defect_nrm);
                n_steps = i_cycle + 1;
                break;
            }
        }

        if (print)
            printf(
                "done with F-cycle solve from start level %d, conv %.2e to %.2e ||defect|| in %d "
                "steps\n",
                starting_level, init_defect_nrm, fin_defect_nrm, n_steps);
    }

    void wcycle_solve(int i_level, int pre_smooth, int post_smooth, int n_wcycles = 100,
                      bool print = false, T atol = 1e-6, T rtol = 1e-6) {
        /* W-cycle may have greater performance than V-cycle? what about also F-cycle? */

        bool is_outer_call = n_wcycles > 2 && i_level == 0;
        T init_defect_nrm, fin_defect_nrm;
        if (is_outer_call) {  // only for outer call..
            // init defect nrm
            init_defect_nrm = grids[0].getDefectNorm();
            printf("W-cycles: ||init_defect|| = %.2e\n", init_defect_nrm);
            fin_defect_nrm = init_defect_nrm;
        }
        int n_steps = n_wcycles;
        int n_levels = getNumLevels();

        for (int i_wcycle = 0; i_wcycle < n_wcycles; i_wcycle++) {
            // pre-smooth and restrict if not last level
            if (i_level < n_levels - 1) {
                grids[i_level].smoothDefect(pre_smooth, print, pre_smooth - 1);
                grids[i_level + 1].restrict_defect(grids[i_level].d_defect);
            }

            // solve this level or call lower level W-cycle (recursively)
            if (i_level < n_levels - 1) {
                // recursive call to lower level (2 inner W-cycles each step..)
                wcycle_solve(i_level + 1, pre_smooth, post_smooth, 2, print, atol, rtol);
            } else {
                // coarsest grid full solve
                if (print) printf("\t--level %d full-solve\n", i_level);
                coarse_solver->solve(grids[i_level].d_defect, grids[i_level].d_soln);
            }

            // now post-smooth and back up the hierarchy (if not direct solve inner level)
            if (i_level < n_levels - 1) {
                grids[i_level].prolongate(grids[i_level + 1].d_soln);
                grids[i_level].smoothDefect(post_smooth, print, post_smooth - 1);
            }

            // compute fine grid defect of V-cycle (outer call only)++
            T defect_nrm;
            if (is_outer_call) {
                defect_nrm = grids[i_level].getDefectNorm();
                fin_defect_nrm = defect_nrm;
                printf("W-cycle step %d, ||defect|| = %.3e\n", i_wcycle, defect_nrm);
            }

            if (defect_nrm < atol + rtol * init_defect_nrm && is_outer_call) {
                printf("W-cycle GMG converged in %d steps to defect nrm %.2e from init_nrm %.2e\n",
                       i_wcycle + 1, defect_nrm, init_defect_nrm);
                n_steps = i_wcycle + 1;
                break;
            }
        }

        if (is_outer_call) {
            printf("done with W-cycle solve, conv %.2e to %.2e ||defect|| in %d steps\n",
                   init_defect_nrm, fin_defect_nrm, n_steps);
        }
    }  // end of wcycle_solve method

    void free() {
        for (int ilevel = 0; ilevel < getNumLevels(); ilevel++) {
            grids[ilevel].free();
        }
        if (coarse_solver) coarse_solver->free();
    }

    // TEMPORARY DEBUG ROUTINES
    // =======================================================

    template <class Assembler>
    void _print_vtk_debug(int icycle, int ilevel, std::string quantity, std::string filename) {
        std::stringstream outputFile;
        // printf("print to VTK debug %d cycle, %d level, %s filename\n", icycle, ilevel,
        //        filename.c_str());
        outputFile << filename << "_level_" << ilevel << "_cycle" << icycle << ".vtk";
        if (quantity == "soln") {
            grids[ilevel].d_soln.permuteData(grids[ilevel].block_dim, grids[ilevel].d_perm);
            auto h_soln = grids[ilevel].d_soln.createHostVec();
            grids[ilevel].d_soln.permuteData(grids[ilevel].block_dim, grids[ilevel].d_iperm);
            printToVTK<Assembler, HostVec<T>>(grids[ilevel].assembler, h_soln, outputFile.str());
        } else if (quantity == "defect") {
            grids[ilevel].d_defect.permuteData(grids[ilevel].block_dim, grids[ilevel].d_perm);
            auto h_defect = grids[ilevel].d_defect.createHostVec();
            grids[ilevel].d_defect.permuteData(grids[ilevel].block_dim, grids[ilevel].d_iperm);
            printToVTK<Assembler, HostVec<T>>(grids[ilevel].assembler, h_defect, outputFile.str());
        } else if (quantity == "ddefect") {
            auto d_vec_temp2 = DeviceVec<T>(grids[ilevel].N, grids[ilevel].d_temp2);
            T a = grids[ilevel].omega;  // what we're subtracting from current defect..
            CHECK_CUBLAS(cublasDscal(grids[ilevel].cublasHandle, grids[ilevel].N, &a,
                                     d_vec_temp2.getPtr(), 1));
            d_vec_temp2.permuteData(grids[ilevel].block_dim, grids[ilevel].d_perm);
            auto h_ddefect = d_vec_temp2.createHostVec();
            printToVTK<Assembler, HostVec<T>>(grids[ilevel].assembler, h_ddefect, outputFile.str());
        }
        // printf("\tend print to VTK debug\n");
    }

    template <class Assembler>
    void debug_vcycle_solve(CoarseSolver *fine_LU_solver, BsrData &fine_LU_bsr_data,
                            int starting_level, int pre_smooth, int post_smooth,
                            int n_vcycles = 100, bool print = false, T atol = 1e-6, T rtol = 1e-6,
                            bool double_smooth = false, int print_freq = 1) {
        /* debug Vcycle solve to do VTK writing (esp to help debug NL problems) */

        // make some temporary vecs
        auto fine_soln = DeviceVec<T>(grids[0].N);

        // init defect nrm
        T init_defect_nrm = grids[0].getDefectNorm();
        if (print) printf("V-cycles: ||init_defect|| = %.2e\n", init_defect_nrm);
        T fin_defect_nrm = init_defect_nrm;
        int n_steps = n_vcycles;
        int n_levels = getNumLevels();
        // std::string quantity, filename;
        printf("MAKE SURE out/debug/ is a valid directory for DEBUG VCYCLE solve\n");

        for (int i_vcycle = 0; i_vcycle < n_vcycles; i_vcycle++) {
            /* restrict and pre-smooth */
            // -----------------------------------------------------------

            for (int i_level = starting_level; i_level < n_levels - 1; i_level++) {
                _print_vtk_debug<Assembler>(i_vcycle, i_level, "soln", "out/debug/step1_soln");
                _print_vtk_debug<Assembler>(i_vcycle, i_level, "defect", "out/debug/step2_def");

                int inner_pre_smooth = pre_smooth * (double_smooth ? 1 << i_level : 1);
                grids[i_level].smoothDefect(inner_pre_smooth, false, inner_pre_smooth - 1);

                _print_vtk_debug<Assembler>(i_vcycle, i_level, "defect", "out/debug/step3_smdef");

                // restrict defect
                grids[i_level + 1].restrict_defect(grids[i_level].d_defect);

                _print_vtk_debug<Assembler>(i_vcycle, i_level + 1, "defect",
                                            "out/debug/step4_resdef");
            }

            /* coarse solve */
            // -----------------------------------------------------------

            coarse_solver->solve(grids[n_levels - 1].d_defect, grids[n_levels - 1].d_soln);
            // _print_vtk_debug<Assembler>(i_vcycle, n_levels - 1, "defect",
            // "out/debug/step4_crsrhs");
            _print_vtk_debug<Assembler>(i_vcycle, n_levels - 1, "soln", "out/debug/step5_crssol");

            /* prolongations + post-smooths back up the levels */
            // ----------------------------------------------------------------------------

            for (int i_level = n_levels - 2; i_level >= starting_level; i_level--) {
                // compare exact solve of current defect to the coarse prolong update
                // need to get exact solution of current defect, with direct solver (to compare with
                // coarse-fine prediction)
                if (i_level == 0) {
                    // need to do several perms here
                    grids[i_level].d_defect.permuteData(
                        grids[i_level].block_dim, grids[i_level].d_perm);  // MC solve to VIS order
                    grids[i_level].d_defect.permuteData(
                        grids[i_level].block_dim, fine_LU_bsr_data.iperm);  // VIS to LU solve order
                    fine_LU_solver->solve(grids[i_level].d_defect, fine_soln);
                    grids[i_level].d_defect.permuteData(
                        grids[i_level].block_dim, fine_LU_bsr_data.perm);  // LU solve to VIS order
                    grids[i_level].d_defect.permuteData(
                        grids[i_level].block_dim, grids[i_level].d_iperm);  // VIS to MC solve order
                    fine_soln.permuteData(grids[i_level].block_dim,
                                          fine_LU_bsr_data.perm);  // LU solve to VIS order
                }

                // get coarse-fine correction from coarser grid to this grid
                grids[i_level].prolongate(grids[i_level + 1].d_soln);
                _print_vtk_debug<Assembler>(i_vcycle, i_level, "soln", "out/debug/step6_prosol");
                printf("level %d, omega = %.6e\n", i_level, grids[i_level].omega);
                _print_vtk_debug<Assembler>(i_vcycle, i_level, "ddefect",
                                            "out/debug/step7_proddef");  // change in defect
                _print_vtk_debug<Assembler>(i_vcycle, i_level, "defect", "out/debug/step8_prodef");

                // now printout comparison of the coarse solve and fine solve exact
                if (i_level == 0) {
                    std::stringstream outputFile0, outputFile;

                    outputFile0 << "out/debug/step9_cfsoln"
                                << "_level_" << i_level << "_cycle" << i_vcycle << ".vtk";
                    auto d_vec_temp = DeviceVec<T>(grids[i_level].N, grids[i_level].d_temp);
                    T a = grids[i_level].omega;
                    CHECK_CUBLAS(cublasDscal(grids[i_level].cublasHandle, grids[i_level].N, &a,
                                             d_vec_temp.getPtr(), 1));
                    d_vec_temp.permuteData(grids[i_level].block_dim, grids[i_level].d_perm);
                    auto h_dtemp = d_vec_temp.createHostVec();
                    printToVTK<Assembler, HostVec<T>>(grids[i_level].assembler, h_dtemp,
                                                      outputFile0.str());

                    outputFile << "out/debug/step10_LUsoln"
                               << "_level_" << i_level << "_cycle" << i_vcycle << ".vtk";
                    auto h_soln = fine_soln.createHostVec();
                    printToVTK<Assembler, HostVec<T>>(grids[i_level].assembler, h_soln,
                                                      outputFile.str());
                }

                // post-smooth
                int inner_post_smooth = post_smooth * (double_smooth ? 1 << i_level : 1);
                grids[i_level].smoothDefect(inner_post_smooth, false, inner_post_smooth - 1);

                // _print_vtk_debug<Assembler>(i_vcycle, i_level, "soln", "out/debug/step9_smsol");
                _print_vtk_debug<Assembler>(i_vcycle, i_level, "defect", "out/debug/step11_smdef");
            }

            /* compute fine grid defect of V-cycle */
            // -----------------------------------------------------

            T defect_nrm = grids[starting_level].getDefectNorm();
            fin_defect_nrm = defect_nrm;
            if (i_vcycle % print_freq == 0 && print)
                printf("DEBUG : v-cycle step %d, ||defect|| = %.3e\n", i_vcycle, defect_nrm);
            if (time) CHECK_CUDA(cudaDeviceSynchronize());

            if (defect_nrm < atol + rtol * init_defect_nrm && print) {
                printf(
                    "DEBUG :V-cycle GMG converged in %d steps to defect nrm %.2e from init_nrm "
                    "%.2e\n",
                    i_vcycle + 1, defect_nrm, init_defect_nrm);
                n_steps = i_vcycle + 1;
                break;
            }
        }

        if (print)
            printf(
                "DEBUG :done with v-cycle solve from start level %d, conv %.2e to %.2e ||defect|| "
                "in %d "
                "steps\n",
                starting_level, init_defect_nrm, fin_defect_nrm, n_steps);
    }

    void getCoarseFineStep(int starting_level, int pre_smooth, int post_smooth,
                           DeviceVec<T> &i_defect, DeviceVec<T> &ism_defect, DeviceVec<T> &cf_soln,
                           DeviceVec<T> &ch_defect, DeviceVec<T> &fsm_defect, bool smooth = true,
                           bool double_smooth = true) {
        /* run a single coarse-fine step, getting the orig defect, disp and proposed defect update
         * for debugging */

        // assumes fine grid defect already been set (or continuation of previous step)

        // init defect nrm
        T init_defect_nrm = grids[0].getDefectNorm();
        int n_levels = getNumLevels();
        printf("inside getCFstep\n");

        /* restrict and pre-smooth */
        // -----------------------------------------------------------

        // copy out current defect
        grids[starting_level].d_defect.copyValuesTo(i_defect);
        i_defect.permuteData(6, grids[starting_level].d_perm);

        for (int i_level = starting_level; i_level < n_levels - 1; i_level++) {
            printf("level %d pre-smooth\n", i_level);
            if (smooth) {
                int inner_pre_smooth = pre_smooth * (double_smooth ? 1 << i_level : 1);
                grids[i_level].smoothDefect(inner_pre_smooth, false, inner_pre_smooth - 1);
            }

            // copy out pre-smoothed defect
            if (i_level == starting_level) {
                grids[starting_level].d_defect.copyValuesTo(ism_defect);
                ism_defect.permuteData(6, grids[starting_level].d_perm);
            }

            // restrict defect
            printf("level %d restr-defect\n", i_level);
            grids[i_level + 1].restrict_defect(grids[i_level].d_defect);
        }

        /* coarse solve */
        // -----------------------------------------------------------

        printf("coarse solve\n");
        coarse_solver->solve(grids[n_levels - 1].d_defect, grids[n_levels - 1].d_soln);

        /* prolongations + post-smooths back up the levels */
        // ----------------------------------------------------------------------------

        for (int i_level = n_levels - 2; i_level >= starting_level; i_level--) {
            // get coarse-fine correction from coarser grid to this grid
            printf("level %d prolongate\n", i_level);
            grids[i_level].prolongate(grids[i_level + 1].d_soln);

            // copy out cf soln, and proposed change in defect
            if (i_level == starting_level) {
                printf("level %d cf, omega = %.6e\n", i_level, grids[i_level].omega);
                grids[starting_level].d_soln.copyValuesTo(cf_soln);
                cf_soln.permuteData(6, grids[starting_level].d_perm);

                cudaMemcpy(ch_defect.getPtr(), grids[starting_level].d_temp2,
                           grids[starting_level].N * sizeof(T), cudaMemcpyDeviceToDevice);
                T a = grids[starting_level].omega;  // what we're subtracting from current defect..
                CHECK_CUBLAS(cublasDscal(grids[starting_level].cublasHandle,
                                         grids[starting_level].N, &a, ch_defect.getPtr(), 1));

                ch_defect.permuteData(6, grids[starting_level].d_perm);
            }

            // post-smooth
            printf("level %d post-smooth\n", i_level);
            int inner_post_smooth = post_smooth * (double_smooth ? 1 << i_level : 1);
            grids[i_level].smoothDefect(inner_post_smooth, false, inner_post_smooth - 1);
        }

        // copy out final post-smoothed defect
        printf("final copy out defect from getCFStep\n");
        grids[starting_level].d_defect.copyValuesTo(fsm_defect);
        fsm_defect.permuteData(6, grids[starting_level].d_perm);
    }

    int nxe;  //, n_levels;
    bool setup;
    std::vector<GRID> grids;
    CoarseSolver *coarse_solver;

   private:
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