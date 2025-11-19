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
                        T linSolveAtol_ = 1e-8, T minLinSolveTol_ = 1e-6, T maxLinSolveTol_ = 0.25)
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

            // except with predictor, doesn't always dec so keep inewton > 0 condition instead
            if (inewton > 0) {
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
                            d_iperm);  // res from VIS => SOLVE order
            fatalFailure = linear_solver->solve(res, update, true); // check_conv = true
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
        T a = -lambda;
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
        return inewton_iters;  // return how many newton steps used by solver
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

    // helper states
    T omega;
    int nvars, block_dim, *d_perm, *d_iperm;
    cublasHandle_t &cublasHandle;

    int line_search_iters = 0;
    int inewton_iters = 0;
};