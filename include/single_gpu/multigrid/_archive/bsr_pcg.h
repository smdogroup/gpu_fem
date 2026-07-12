#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#ifdef USE_GPU
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

#include "../solve_utils.h"

template <typename T, class GRID>
class PCGSolver : public BaseSolver {
   public:
    PCGSolver(cublasHandle_t &cublasHandle_, cusparseHandle_t &cusparseHandle_, GRID *grid_,
              BaseSolver *pc_, SolverOptions options, int ilevel_ = -1, int N_ = 0)
        : grid(grid_),
          pc(pc_),
          options(options),
          ilevel(ilevel_),
          cublasHandle(cublasHandle_),
          cusparseHandle(cusparseHandle_) {
        // Get matrix and initialize temporary data for the PCG solve.
        mat = grid->Kmat;

        auto bsr_data = mat.getBsrData();

        mb = bsr_data.nnodes;
        nnzb = bsr_data.nnzb;
        block_dim = bsr_data.block_dim;

        d_rowp = bsr_data.rowp;
        d_cols = bsr_data.cols;
        iperm = bsr_data.iperm;
        d_vals = mat.getPtr();

        cublasHandle = grid->cublasHandle;
        cusparseHandle = grid->cusparseHandle;

        if (N_ == 0) {
            N = grid->N;
        } else {
            N = N_;
        }

        d_rhs = DeviceVec<T>(N).getPtr();
        d_x = DeviceVec<T>(N).getPtr();

        // Description of the K matrix.
        descrK = 0;

        CHECK_CUSPARSE(cusparseCreateMatDescr(&descrK));
        CHECK_CUSPARSE(cusparseSetMatType(descrK, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descrK, CUSPARSE_INDEX_BASE_ZERO));

        // Temporary vectors.
        d_resid_vec = DeviceVec<T>(N);
        d_resid = d_resid_vec.getPtr();

        d_p = DeviceVec<T>(N).getPtr();
        d_w = DeviceVec<T>(N).getPtr();

        d_z_vec = DeviceVec<T>(N);
        d_z = d_z_vec.getPtr();
    }

    void update_after_assembly(DeviceVec<T> &vars) {
        bool perm = true;

        grid->setStateVars(vars, perm);
        grid->update_after_assembly();

        if (pc) {
            pc->update_after_assembly(vars);
        }
    }

    void factor() {}

    void set_print(bool print) { options.print = print; }

    void set_abs_tol(T atol) { options.atol = atol; }

    void set_rel_tol(T rtol) { options.rtol = rtol; }

    void set_cycle_type(std::string cycle_) {
        if (pc) {
            pc->set_cycle_type(cycle_);
        }
    }

    T getResidualNorm(DeviceVec<T> rhs_in, DeviceVec<T> soln_in) {
        // Compute r = b - Ax.
        CHECK_CUDA(cudaMemcpy(d_resid, rhs_in.getPtr(), N * sizeof(T), cudaMemcpyDeviceToDevice));

        T a_local = -1.0;
        T b_local = 1.0;

        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a_local,
                                      descrK, d_vals, d_rowp, d_cols, block_dim, soln_in.getPtr(),
                                      &b_local, d_resid));

        T resid_norm;

        CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &resid_norm));

        return resid_norm;
    }

    /*
     * Estimate the projected total solve time using the logarithmic residual
     * convergence rate over the final 10% of completed iterations.
     *
     * The stored data satisfy:
     *
     *     residual_history[i] = residual after iteration i + 1
     *     elapsed_history[i]  = elapsed time after iteration i + 1
     */
    double estimate_total_solve_time(int completed_iterations, T current_resid_norm, T conv_tol,
                                     double elapsed_sec,
                                     const std::vector<double> &residual_history,
                                     const std::vector<double> &elapsed_history) const {
        if (completed_iterations <= 0 || elapsed_sec <= 0.0 || residual_history.size() < 2 ||
            residual_history.size() != elapsed_history.size()) {
            return -1.0;
        }

        const int history_size = static_cast<int>(residual_history.size());

        /*
         * Use approximately the final 10% of completed iterations.
         *
         * For example, after 500 iterations this compares the state near
         * iteration 450 with the state at iteration 500.
         */
        const int window = std::max(
            1, static_cast<int>(std::ceil(0.10 * static_cast<double>(completed_iterations))));

        const int end_index = history_size - 1;
        const int start_index = std::max(0, end_index - window);

        const double r_start = residual_history[start_index];
        const double r_current = std::abs(static_cast<double>(current_resid_norm));
        const double r_target = std::abs(static_cast<double>(conv_tol));

        const double t_start = elapsed_history[start_index];
        const double t_current = elapsed_history[end_index];
        const double local_elapsed_sec = t_current - t_start;

        if (!std::isfinite(r_start) || !std::isfinite(r_current) || !std::isfinite(r_target) ||
            !std::isfinite(t_start) || !std::isfinite(t_current) ||
            !std::isfinite(local_elapsed_sec) || r_start <= 0.0 || r_current <= 0.0 ||
            r_target <= 0.0 || local_elapsed_sec <= 0.0) {
            return -1.0;
        }

        if (r_current <= r_target) {
            return elapsed_sec;
        }

        /*
         * The residual must have decreased over the selected local window.
         */
        if (r_current >= r_start) {
            return -1.0;
        }

        /*
         * Local logarithmic residual convergence rate in 1/second:
         *
         *             log(r_current / r_start)
         * rate = ----------------------------------
         *              t_current - t_start
         */
        const double local_log_rate = std::log(r_current / r_start) / local_elapsed_sec;

        if (!std::isfinite(local_log_rate) || local_log_rate >= 0.0) {
            return -1.0;
        }

        /*
         * Time required to proceed from the current residual to the target:
         *
         *                        log(r_target / r_current)
         * additional_time = --------------------------------
         *                              local_log_rate
         *
         * Both values are negative during convergence, yielding positive time.
         */
        double additional_sec = std::log(r_target / r_current) / local_log_rate;

        if (!std::isfinite(additional_sec)) {
            return -1.0;
        }

        additional_sec = std::max(0.0, additional_sec);

        /*
         * Return the projected total time measured from the beginning of the
         * solve, not merely the additional remaining time.
         */
        return elapsed_sec + additional_sec;
    }

    void print_failure_time_estimate(int completed_iterations, T current_resid_norm, T conv_tol,
                                     double elapsed_sec,
                                     const std::vector<double> &residual_history,
                                     const std::vector<double> &elapsed_history) const {
        const double estimated_total_sec =
            estimate_total_solve_time(completed_iterations, current_resid_norm, conv_tol,
                                      elapsed_sec, residual_history, elapsed_history);

        const int print_level = ilevel < 0 ? 0 : ilevel;

        if (estimated_total_sec >= 0.0 && std::isfinite(estimated_total_sec)) {
            std::printf(
                "L%d-PCG did not converge after %d iterations, "
                "resid = %.9e.\n"
                "Estimated total solve time: %.2fs\n",
                print_level, completed_iterations, static_cast<double>(current_resid_norm),
                estimated_total_sec);
        } else {
            std::printf(
                "L%d-PCG did not converge after %d iterations, "
                "resid = %.9e.\n"
                "Estimated total solve time unavailable.\n",
                print_level, completed_iterations, static_cast<double>(current_resid_norm));
        }

        std::fflush(stdout);
    }

    bool solve(DeviceVec<T> rhs_in, DeviceVec<T> soln_out, bool check_conv = false) {
        /*
         * Assumes rhs_in and soln_out use the solver permutation rather than
         * natural ordering.
         *
         * Performs a full K-cycle with left-preconditioned flexible PCG.
         * The convergence check reports the true residual.
         */

        CHECK_CUDA(cudaMemcpy(d_rhs, rhs_in.getPtr(), N * sizeof(T), cudaMemcpyDeviceToDevice));

        CHECK_CUDA(cudaMemset(d_x, 0, N * sizeof(T)));

        // Since x starts at zero, r0 = b.
        CHECK_CUDA(cudaMemcpy(d_resid, d_rhs, N * sizeof(T), cudaMemcpyDeviceToDevice));

        n_steps = 0;

        T init_resid_norm = 0.0;
        T conv_tol = 0.0;

        if (check_conv) {
            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &init_resid_norm));

            conv_tol = options.atol + init_resid_norm * options.rtol;

            if (options.print && ilevel == 0) {
                std::printf("L0-PCG init_resid = %.8e, target = %.3e\n",
                            static_cast<double>(init_resid_norm), static_cast<double>(conv_tol));
            }

            if (options.print && ilevel != 0) {
                std::printf("\tL%d-PCG init_resid %.2e, target = %.3e\n", ilevel,
                            static_cast<double>(init_resid_norm), static_cast<double>(conv_tol));
            }
        }

        /*
         * Synchronize before starting the timer so setup work queued before
         * the solve is not incorrectly included in the elapsed time.
         */
        CHECK_CUDA(cudaDeviceSynchronize());

        const auto solve_start = std::chrono::high_resolution_clock::now();

        std::vector<double> residual_history;
        std::vector<double> elapsed_history;

        if (check_conv) {
            residual_history.reserve(options.ncycles);
            elapsed_history.reserve(options.ncycles);
        }

        T rho_prev_local = 0.0;
        T rho_local = 0.0;

        bool converged = false;
        T final_resid_norm = init_resid_norm;

        for (int j = 0; j < options.ncycles; j++) {
            /*
             * 1) Apply the preconditioner:
             *
             *     Mz = r
             */
            pc->solve(d_resid_vec, d_z_vec);

            n_steps = j + 1;

            /*
             * 2) Compute rho and update the search direction.
             */
            CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_resid, 1, d_z, 1, &rho_local));

            if (j == 0) {
                CHECK_CUDA(cudaMemcpy(d_p, d_z, N * sizeof(T), cudaMemcpyDeviceToDevice));
            } else {
                const T beta = rho_local / rho_prev_local;

                T a_local = beta;

                CHECK_CUBLAS(cublasDscal(cublasHandle, N, &a_local, d_p, 1));

                a_local = 1.0;

                CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a_local, d_z, 1, d_p, 1));
            }

            rho_prev_local = rho_local;

            /*
             * 3) Matrix-vector product:
             *
             *     w = Ap
             */
            T a_local = 1.0;
            T b_local = 0.0;

            CHECK_CUSPARSE(cusparseDbsrmv(
                cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb,
                nnzb, &a_local, descrK, d_vals, d_rowp, d_cols, block_dim, d_p, &b_local, d_w));

            /*
             * 4) Update x and r.
             */
            T wp0;

            CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_w, 1, d_p, 1, &wp0));

            const T alpha_local = rho_local / wp0;

            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &alpha_local, d_p, 1, d_x, 1));

            a_local = -alpha_local;

            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a_local, d_w, 1, d_resid, 1));

            if (check_conv) {
                T resid_norm;

                CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &resid_norm));

                final_resid_norm = resid_norm;

                /*
                 * The norm reduction provides the host with the residual.
                 * Synchronize explicitly before recording the timestamp so
                 * elapsed time represents completed GPU work.
                 */
                CHECK_CUDA(cudaDeviceSynchronize());

                const auto now = std::chrono::high_resolution_clock::now();

                const double elapsed_sec = std::chrono::duration<double>(now - solve_start).count();

                residual_history.push_back(std::abs(static_cast<double>(resid_norm)));

                elapsed_history.push_back(elapsed_sec);

                if (j % options.print_freq == 0 && options.print) {
                    if (ilevel == 0) {
                        std::printf("L0-PCG [%d] = %.8e\n", j, static_cast<double>(resid_norm));
                    } else {
                        std::printf("\tL%d-PCG [%d] = %.8e\n", ilevel, j,
                                    static_cast<double>(resid_norm));
                    }
                }

                if (std::abs(resid_norm) < conv_tol) {
                    converged = true;

                    if (options.print) {
                        if (ilevel == 0) {
                            std::printf(
                                "\nL0-PCG converged in %d iterations "
                                "to %.9e resid\n",
                                j + 1, static_cast<double>(resid_norm));
                        } else {
                            std::printf(
                                "\nL%d-PCG converged in %d iterations "
                                "to %.9e resid\n",
                                ilevel, j + 1, static_cast<double>(resid_norm));
                        }
                    }

                    break;
                }
            }
        }

        /*
         * Print a projected total solve time only when:
         *
         *   1. convergence was being checked,
         *   2. the solver exhausted the requested iterations,
         *   3. printing is enabled.
         */
        if (check_conv && !converged && options.print) {
            CHECK_CUDA(cudaDeviceSynchronize());

            const auto solve_end = std::chrono::high_resolution_clock::now();

            const double elapsed_sec =
                std::chrono::duration<double>(solve_end - solve_start).count();

            print_failure_time_estimate(n_steps, final_resid_norm, conv_tol, elapsed_sec,
                                        residual_history, elapsed_history);

            std::printf("\n");
        }

        // Debug check.
        if (options.debug) {
            CHECK_CUDA(cudaMemcpy(d_resid, d_rhs, N * sizeof(T), cudaMemcpyDeviceToDevice));

            T resid_norm1;

            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &resid_norm1));

            T a_local = -1.0;
            T b_local = 1.0;

            CHECK_CUSPARSE(cusparseDbsrmv(
                cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb,
                nnzb, &a_local, descrK, d_vals, d_rowp, d_cols, block_dim, d_x, &b_local, d_resid));

            T resid_norm;

            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &resid_norm));

            T x_nrm;

            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_x, 1, &x_nrm));

            std::printf(
                "debug: L%d resid_nrm1 %.8e and v2 %.8e, "
                "with d_x norm %.2e\n",
                ilevel, static_cast<double>(resid_norm1), static_cast<double>(resid_norm),
                static_cast<double>(x_nrm));
        }

        CHECK_CUDA(cudaMemcpy(soln_out.getPtr(), d_x, N * sizeof(T), cudaMemcpyDeviceToDevice));

        /*
         * Preserve the original return convention:
         *
         *     false = converged
         *     true  = failed to converge
         *
         * When check_conv is false, converged remains false and this returns
         * true, matching the original implementation.
         */
        return !converged;
    }

    int get_num_iterations() { return n_steps; }

    void free() {
        if (is_free) {
            return;
        }

        is_free = true;

        if (grid) {
            grid->free();
        }

        d_resid_vec.free();

        if (d_x) {
            cudaFree(d_x);
            d_x = nullptr;
        }

        if (d_rhs) {
            cudaFree(d_rhs);
            d_rhs = nullptr;
        }

        if (d_p) {
            cudaFree(d_p);
            d_p = nullptr;
        }

        if (d_w) {
            cudaFree(d_w);
            d_w = nullptr;
        }

        if (d_z) {
            cudaFree(d_z);
            d_z = nullptr;
        }

        d_z_vec.free();

        if (descrK) {
            cusparseDestroyMatDescr(descrK);
            descrK = nullptr;
        }
    }

    void test_mult(DeviceVec<T> vec_in, DeviceVec<T> out) {
        T a_local = 1.0;
        T b_local = 0.0;

        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a_local,
                                      descrK, d_vals, d_rowp, d_cols, block_dim, vec_in.getPtr(),
                                      &b_local, out.getPtr()));
    }

    void test_precond(DeviceVec<T> vec_in, DeviceVec<T> out) { pc->solve(vec_in, out); }

    GRID *grid = nullptr;
    BaseSolver *pc = nullptr;

    SolverOptions options;
    int ilevel = -1;

   private:
    // Main matrix and linear-system data.
    BsrMat<DeviceVec<T>> mat;

    int N = 0;
    int mb = 0;
    int nb = 0;
    int nnzb = 0;
    int block_dim = 0;

    int *d_rowp = nullptr;
    int *d_cols = nullptr;
    int *iperm = nullptr;

    T *d_vals = nullptr;

    T *d_rhs = nullptr;
    T *d_x = nullptr;
    T *d_resid = nullptr;

    DeviceVec<T> d_resid_vec;

    int n_steps = 0;
    bool is_free = false;

    // cuSPARSE and cuBLAS handles.
    cusparseHandle_t &cusparseHandle;
    cublasHandle_t &cublasHandle;

    // Description of K.
    cusparseMatDescr_t descrK = nullptr;

    // Temporary vectors for PCG.
    DeviceVec<T> d_z_vec;

    T *d_p = nullptr;
    T *d_w = nullptr;
    T *d_z = nullptr;

    // Temporary scalars retained for compatibility with existing code.
    T rho = 0.0;
    T rho_prev = 0.0;
    T a = 0.0;
    T alpha = 0.0;
    T b = 0.0;
};