// GPU implementation of Ali Gray's inexact Newton solver (with Eisenstat-Walker method from CPU
// TACS) implemented by Sean Engelstad, Nov 4th 2025

#include <functional>
#include <iostream>
#include <sstream>
#include <string>

#include "mesh/vtk_writer.h"
#include "newton.h"
#include "solvers/linear_static/_utils.h"

template <typename T, class Mat, class Vec, class Assembler, class Solver,
          bool DO_LINE_SEARCH = true>
class InexactNewtonSolver {
   public:
    InexactNewtonSolver(cublasHandle_t &cublasHandle_, Assembler &assembler_, Mat &kmat_,
                        Vec &loads_, Solver *linear_solver_, T initLinSolveRtol_ = 1e-1,
                        T linSolveAtol_ = 1e-8, T minLinSolveTol_ = 1e-6, T maxLinSolveTol_ = 0.25,
                        T restart_dlam_ = 1e-2)
        : assembler(assembler_),
          kmat(kmat_),
          loads(loads_),
          linear_solver(linear_solver_),
          cublasHandle(cublasHandle_) {
        // EW exponent
        omega = 0.5 * (1.0 + sqrt(5));  // golden ratio
        nvars = assembler.get_num_vars();
        auto bsr_data = kmat.getBsrData();
        block_dim = bsr_data.block_dim;
        d_iperm = bsr_data.iperm;
        d_perm = bsr_data.perm;

        linear_solver->set_abs_tol(linSolveAtol);

        initLinSolveRtol = initLinSolveRtol_;
        linSolveAtol = linSolveAtol_;
        minLinSolveTol = minLinSolveTol_;
        maxLinSolveTol = maxLinSolveTol_;
        restart_dlam = restart_dlam_;

        // make res, soln, temp vecs
        res = assembler.createVarsVec();
        soln = assembler.createVarsVec();
        temp = assembler.createVarsVec();
        vars = assembler.createVarsVec();
        update = assembler.createVarsVec();
    }

    bool solve(T lambda, T rtol, T atol, Vec &state) {
        /* main function to call */

        // copy state from outer solver aka u0, and startup
        state.copyValuesTo(vars);
        assembler.set_variables(vars);
        T init_res_nrm = computeResidual(lambda);
        T prev_res_nrm = init_res_nrm;

        T linSolveRtol = initLinSolveRtol;

        bool converged = false;
        bool fatalFailure = false;

        // for (int inewton = 0; inewton < 5; inewton++) {
        for (int inewton = 0; inewton < 40; inewton++) {
            inewton_iters = inewton + 1;

            // res and convergence check
            T res_nrm = computeResidual(lambda);
            converged = checkConvergence(res_nrm, rtol, atol, init_res_nrm);
            int solver_iterations = linear_solver->get_num_iterations();
            if (inewton == 0) {
                printf("\tinewton 0 => resid %.5e\n", res_nrm);
            } else {
                printf("\tinewton %d => resid %.5e, #l-search %d, #solve-iters %d, lrtol %.2e\n",
                       inewton, res_nrm, line_search_iters, solver_iterations, linSolveRtol);
            }

            if (converged) break;     // return success
            if (fatalFailure) break;  // return failure

            // update jacobian, TODO : could add Ali's delay redo pc here, not doing that in my work
            // though
            updateJacobian();

            // Eisenstat-Walker method to update linear solve atol (to prevent over-solving)
            // ------------------------------------------------------------

            // except with predictor, doesn't always dec so keep inewton > 0 or inewton > 1
            // condition instead
            if (inewton > 1) {
                // don't check immediately, it almost always inc on first newton step so don't adapt
                // if (inewton > 1) {
                T zeta = std::pow(res_nrm / prev_res_nrm, omega);
                T zeta_star = std::pow(linSolveRtol, omega);
                // Ali has slight mistake here I think where he changes the atol not rtol in lin
                // solve
                linSolveRtol = zeta_star < 0.1 ? zeta : max(zeta, zeta_star);
                linSolveRtol =
                    std::clamp(linSolveRtol, minLinSolveTol, maxLinSolveTol);  // clip the rtol
            }
            linear_solver->set_rel_tol(linSolveRtol);

            // do an iterative linear solve here
            // ---------------------------------

            // NOTE : res and update are held in VIS (visualization) order
            // in this class, while the linear solver expects everything
            // in solve perm/order (so we permute to and from that)
            update.zeroValues();
            res.permuteData(block_dim,
                            d_iperm);                                // res from VIS => SOLVE order
            fatalFailure = linear_solver->solve(res, update, true);  // check_conv = true
            num_lin_solves++;
            update.permuteData(block_dim,
                               d_perm);  // update from SOLVE => VIS order

            if (fatalFailure) {
                failedRtol = linSolveRtol;
                continue;
            }

            // flip sign of update since rhs should have really been -res
            T a = -1.0;
            CHECK_CUBLAS(cublasDscal(cublasHandle, nvars, &a, update.getPtr(), 1));

            // do energy line search and apply update
            // ---------------------------------------
            T alpha = 1.0;
            if constexpr (DO_LINE_SEARCH) {
                alpha = energyLineSearch(lambda);
            }
            CHECK_CUBLAS(
                cublasDaxpy(cublasHandle, nvars, &alpha, update.getPtr(), 1, vars.getPtr(), 1));
            assembler.set_variables(vars);
            prev_res_nrm = res_nrm;

            // DEBUG prints here
            // ========================================
            // printf("\t\tlinsolveRelTol %.6e, alpha %.4e\n",
            // linSolveRtol, alpha); T update_nrm;
            // CHECK_CUBLAS(cublasDnrm2(cublasHandle, nvars,
            // update.getPtr(), 1, &update_nrm)); T vars_nrm;
            // CHECK_CUBLAS(cublasDnrm2(cublasHandle, nvars,
            // vars.getPtr(), 1, &vars_nrm)); printf("\t\tupdate nrm %.8e,
            // vars nrm %.8e\n", update_nrm, vars_nrm);
        }

        // now copy solution out
        vars.copyValuesTo(state);

        return converged;
    }

    T computeResidual(T &lambda) {
        /* compute r(u) = fint(u) - lambda * loads */
        assembler.add_residual_fast(res);
        T a = -lambda;  // loads assumed to be in VIS order here
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, nvars, &a, loads.getPtr(), 1, res.getPtr(), 1));
        assembler.apply_bcs(res);

        // then compute residual norm also
        T res_norm;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, nvars, res.getPtr(), 1, &res_norm));
        // printf("resid nrm %.8e\n", res_norm);
        return res_norm;
    }

    T getResidual(T &lambda, DeviceVec<T> d_res_out) {
        /* get residual vector and norm */
        T res_nrm = computeResidual(lambda);
        res.copyValuesTo(d_res_out);  // in vis order
        return res_nrm;
    }

    bool checkConvergence(T resid_nrm, T rtol, T atol, T init_resid_nrm) {
        // printf("\t\tresid nrm %.5e, rtol %.5e, atol %.5e, init_resid_nrm %.5e\n", resid_nrm,
        // rtol,
        //        atol, init_resid_nrm);
        return resid_nrm < (rtol * init_resid_nrm + atol);
    }

    void updateJacobian() {
        // TODO : could add Ali's delay preconditioner here, not gonna do that yet, GPU assembly
        // very fast
        assembler.add_jacobian_fast(kmat);
        assembler.apply_bcs(kmat);

        // then update solver if need be (such as ILU factoring or multigrid coarse grid assemblies)
        linear_solver->update_after_assembly(vars);
    }

    int get_num_newton_steps() {
        return inewton_iters;  // return how many newton steps used by solver (for this newton
                               // solve)
    }

    int get_num_lin_solves() {
        return num_lin_solves;  // how many total linear solves across all newton solves of
                                // continuation
    }

    T _dotProduct(Vec &vec1, Vec &vec2) {
        // helper GPU dot product method
        T out;
        CHECK_CUBLAS(cublasDdot(cublasHandle, nvars, vec1.getPtr(), 1, vec2.getPtr(), 1, &out));
        return out;
    }

    T energyLineSearch(T &_lambda) {
        // do energy line search on f(alpha) = du^T r(u0 + alpha * du) objective function

        // store u0 in temp for easy reset of state
        vars.copyValuesTo(temp);
        T alpha_old = 0.0;
        T f0 = _energyObjective(alpha_old, _lambda);  // f0 is merit function at alpha = 0

        // line search prelim settings
        T MU = 1e-4;    // expected decrase of line search
        T alpha = 1.0;  // starting value of alpha
        T fold = f0;
        T alpha_new = alpha;

        for (int isearch = 0; isearch < 25; isearch++) {
            line_search_iters = isearch + 1;  // record the num line search iterations done

            T fnew = _energyObjective(alpha, _lambda);
            T fred = abs(fnew / fold);

            // printf("\t\tline search %d => alpha=%.4e, fold=%.4e, fnew=%.4e\n", isearch, alpha,
            // fold, fnew);

            // can exit if already decreased
            if (fred <= (1.0 - MU * min(alpha, 1.0))) return alpha;

            T alpha_min = isearch == 0 ? 0.9 : 1e-2;
            if (fnew == fold) {
                alpha_new = alpha + 1e-2;
            } else {
                // lin interp alpha to fnew == 0?
                alpha_new = alpha - fnew * (alpha - alpha_old) / (fnew - fold);
                alpha_new = std::clamp(alpha_new, alpha_min, 2.0);
            }
            // clip mag of alpha increase to 0.5 max
            T dalpha = alpha_new - alpha;
            if (isearch > 0 && abs(dalpha) > 0.5) {
                T sign_step = dalpha > 0.0 ? 1.0 : -1.0;
                alpha_new = alpha + sign_step * 0.5;
            }

            // update quantities for next line search iteration
            alpha = alpha_new;
            alpha_old = alpha;
            fold = fnew;
        }
        return alpha;
    }

    T _energyObjective(T &alpha, T &_lambda) {
        /* compute energy line search obj func f(alpha) = du^T r(u0 + alpha * du) */

        // update state and compute new residual
        T a = alpha;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, nvars, &a, update.getPtr(), 1, vars.getPtr(), 1));
        assembler.set_variables(vars);
        computeResidual(_lambda);

        // compute the inner product <du, res>
        T objective = _dotProduct(update, res);

        // reset state to original
        temp.copyValuesTo(vars);  // temp is holding u0 (see start of line search call)
        assembler.set_variables(temp);
        return objective;
    }

    bool compute_optimal_restart(const T init_step, const T max_lambda, DeviceVec<T> &state,
                                 T &lambda, T &dlambda) {
        /* compute optimal energy min restart from Ali's paper */
        //    optLoadScale = (Fe^T dUi + Fi^T dUe) / (-2 Fe^T dUe); where dUi = Kinv * Fi, dUe =
        //    Kinv*Fe
        // first compute the norms of state vec and ext force
        printf("compute optimal restart\n");
        T state_nrm;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, nvars, state.getPtr(), 1, &state_nrm));
        T Fe_nrm;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, nvars, loads.getPtr(), 1, &Fe_nrm));
        Fe_nrm *= lambda;
        printf("\t state_nrm %.4e, Fe_nrm %.4e\n", state_nrm, Fe_nrm);

        bool restart_design = false;
        if (state_nrm > 0.0 and Fe_nrm > 0.0) {
            /* compute the energy inner products about current state + external loads */
            T FeUe, FeUi, FiUe;
            _get_energy_inner_products(lambda, FeUe, FeUi, FiUe);

            /* now predict the optimal energy load scale */
            T opt_load_scale = (FeUi + FiUe) / (-2.0 * FeUe);
            opt_load_scale *= -1.0;  // opposite sign because Ali defines fext differently here

            // then determine if it is reasonable and if we should restart
            if (opt_load_scale > (2.0 * max_lambda) || (opt_load_scale < 0.0)) {
                // if opt load scale more than double max load scale we're aiming for, or if
                // negative then it's more efficient to just solve from zero displacements
                // (structure changed too much)
                state.zeroValues();  // zero state vars (to output)
                state.copyValuesTo(vars);
                assembler.set_variables(vars);
                T ratio = opt_load_scale / max_lambda;
                printf(
                    "New Design - reset LAM=0: opt_load_scale %.4e > 2 or <0 so restart to zero "
                    "disps\n",
                    ratio);
                opt_load_scale = init_step;
            } else if (abs(opt_load_scale - max_lambda) < restart_dlam) {
                // opt load scale is so close to target load scale
                //  that it's more efficient to just solve to target load scale
                //  useful when optimization is near complete + design changes small
                T ratio = opt_load_scale / max_lambda;
                printf("New Design - set LAM_INIT=1.0: since opt_load_scale %.4e close enough\n",
                       ratio);
                opt_load_scale = max_lambda;
                restart_design = true;
            } else {
                // otherwise, we have a reasonable optimal load scale
                // now in case opt_load_scale is small, take max with suggested init_step
                //  as that will likely get faster solve
                T opt_load_scale0 = opt_load_scale;
                opt_load_scale = max(opt_load_scale, init_step);
                if (opt_load_scale0 < init_step) {
                    printf(
                        "New Design - set LAM_INIT=%.4e, as |1-energy_opt_lam|=%.4e < init step "
                        "size "
                        "%.4e\n",
                        opt_load_scale, opt_load_scale0, init_step);
                } else {
                    printf("New Design - set LAM_INIT=%.4e\n", opt_load_scale);
                }
                restart_design = true;
                // if opt load scale larger than max load scale, we need to load downwards,
                //  so change the sign of load stepping
                if (opt_load_scale > max_lambda) {
                    dlambda *= -1;
                    printf("\tLAM_INIT=%.4e > 1 so reversed load increments\n");
                }
            }  // end of opt load scale sanity checks

            // write to output the new load factor (even if it gets reset to init step)
            lambda = opt_load_scale;

        }  // end of nonzero state check

        return restart_design;
    }

    void _get_energy_inner_products(const T lambda, T &FeUe, T &FeUi, T &FiUe) {
        /* now compute energy inner products */
        // FeUe, FeUi and FiUi are outputs of this method

        // update jacobian in order to do linear solves
        // keeps states from last time
        updateJacobian();
        // TODO : setting for this here?
        linear_solver->set_rel_tol(1e-6);

        // 1) compute dUi = Kinv * Fi
        // where fint in res (temporarily)
        assembler.add_residual_fast(res);  // automatically zeros res
        assembler.apply_bcs(res);
        // then do linear solve dUi = Kinv * Fi
        res.permuteData(block_dim, d_iperm);  // res from VIS => SOLVE order
        linear_solver->solve(res, update, true);
        // update kept in solve order for inner product
        res.zeroValues();
        T a = 1.0;  // don't think this should be lambda here (just want based on full external
                    // loads lam = 1 final)
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, nvars, &a, loads.getPtr(), 1, res.getPtr(), 1));
        assembler.apply_bcs(res);  // res thus in VIS order here
        // convert res from VIS to solve order
        res.permuteData(block_dim, d_iperm);  // res from VIS => SOLVE order
        // compute Fe^T * dUi inner product; aka <res, update> inner product
        CHECK_CUBLAS(cublasDdot(cublasHandle, nvars, res.getPtr(), 1, update.getPtr(), 1, &FeUi));

        // 2) compute dUe = Kinv * Fe
        // keep res = Fe in solve order from last block
        linear_solver->solve(res, update, true);
        // keep update (aka Ue) in SOLVE order
        // now do inner product <Fe, dUe>
        CHECK_CUBLAS(cublasDdot(cublasHandle, nvars, res.getPtr(), 1, update.getPtr(), 1, &FeUe));

        // 3) compute another inner product FiUe = <Fi, dUe> (keeping dUe in update in SOLVE order)
        // assemble fint back into res
        assembler.add_residual_fast(res);  // automatically zeros res
        assembler.apply_bcs(res);
        res.permuteData(block_dim, d_iperm);  // res from VIS => SOLVE order
        // now take <Fi, dUe> dot product in SOLVE order
        CHECK_CUBLAS(cublasDdot(cublasHandle, nvars, res.getPtr(), 1, update.getPtr(), 1, &FiUe));
    }

    void debug_solve(T lambda, T rtol, T atol, Vec &state, Vec &resOut) {  // , Vec &resOut
        /* debug solve once we find a failed state (for debugging conv) */

        // copy state from outer solver aka u0, and startup
        state.copyValuesTo(vars);
        assembler.set_variables(vars);
        T res_nrm = computeResidual(lambda);
        printf("DEBUG SOLVE at lambda %.6e, res_nrm %.6e\n", lambda, res_nrm);
        updateJacobian();

        // write residual to the output
        res.copyValuesTo(resOut);

        // check the states in each level of coarser grid
        linear_solver->template debug_assembly<Assembler>();
        // printf("DONE WITH DEBUG ASSEMBLY (DEBUG)\n");

        linear_solver->set_rel_tol(failedRtol);  // set to same as what failed here
        printf("setting lin solve to failed rtol %.4e\n", failedRtol);

        // run linear solve (with debug flag on?)
        printf("calling linear solver in DEBUG SOLVE\n");
        update.zeroValues();
        res.permuteData(block_dim,
                        d_iperm);  // iperm and perm cause solvers operate in solver ordering
        linear_solver->solve(res, update);
        update.permuteData(block_dim, d_perm);
        printf("\tdone calling linear solver in DEBUG SOLVE\n");
    }

    void free() {
        loads.free();
        res.free();
        soln.free();
        temp.free();
        vars.free();
        update.free();
    }

   private:
    // main / most important states
    Assembler assembler;
    Mat kmat;
    Vec loads, res, soln, temp, vars, update;
    Solver *linear_solver;

    T initLinSolveRtol, linSolveAtol;
    T failedRtol;
    T minLinSolveTol, maxLinSolveTol;
    T restart_dlam;

    // helper states
    T omega;
    int nvars, block_dim, *d_perm, *d_iperm;
    cublasHandle_t &cublasHandle;

    int line_search_iters = 0;
    int inewton_iters = 0;
    int num_lin_solves = 0;
};
