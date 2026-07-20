#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef USE_GPU
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

#include "../solve_utils.h"

template <typename T, class GRID, int N_SUBSPACE = 50>
class GMRESSolver : public BaseSolver {
   public:
    GMRESSolver(cublasHandle_t &cublasHandle_, cusparseHandle_t &cusparseHandle_, GRID *grid_,
                BaseSolver *pc_, SolverOptions options, int N_ = 0, int MAX_ITER_ = 100)
        : grid(grid_),
          pc(pc_),
          options(options),
          cublasHandle(cublasHandle_),
          cusparseHandle(cusparseHandle_) {
        mat = grid->Kmat;
        MAX_ITER = MAX_ITER_;
        printf("MAX_ITER = %d\n", MAX_ITER);

        auto bsr_data = mat.getBsrData();

        mb = bsr_data.nnodes;
        nnzb = bsr_data.nnzb;
        block_dim = bsr_data.block_dim;

        d_rowp = bsr_data.rowp;
        d_cols = bsr_data.cols;
        iperm = bsr_data.iperm;
        d_vals = mat.getPtr();

        if (!pc) {
            std::printf(
                "\nWARNING: GMRES solver was constructed with no "
                "preconditioner\n\n");
        }

        if (N_ == 0) {
            N = grid->N;
        } else {
            N = N_;
        }

        d_x = DeviceVec<T>(N).getPtr();

        descrK = 0;

        CHECK_CUSPARSE(cusparseCreateMatDescr(&descrK));
        CHECK_CUSPARSE(cusparseSetMatType(descrK, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descrK, CUSPARSE_INDEX_BASE_ZERO));

        d_tmp_vec = DeviceVec<T>(N);
        d_tmp = d_tmp_vec.getPtr();

        d_tmp2_vec = DeviceVec<T>(N);
        d_tmp2 = d_tmp2_vec.getPtr();

        d_xR_vec = DeviceVec<T>(N);
        d_xR = d_xR_vec.getPtr();

        d_Vmat = DeviceVec<T>((N_SUBSPACE + 1) * N * sizeof(T)).getPtr();

        d_resid_vec = DeviceVec<T>(N);
        d_resid = d_resid_vec.getPtr();

        d_w = DeviceVec<T>(N).getPtr();

        d_z_vec = DeviceVec<T>(N);
        d_z = d_z_vec.getPtr();

        d_Hred = DeviceVec<T>((N_SUBSPACE + 1) * (N_SUBSPACE + 1)).getPtr();

        d_gred = DeviceVec<T>(N_SUBSPACE + 1).getPtr();

        h_y = new T[N_SUBSPACE + 1];

        Hred = new T[(N_SUBSPACE + 1) * (N_SUBSPACE + 1)];
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

    void set_cycle_type(std::string cycle_) {}

    T getResidualNorm(DeviceVec<T> rhs_in, DeviceVec<T> soln_in) {
        CHECK_CUDA(cudaMemcpy(d_resid, rhs_in.getPtr(), N * sizeof(T), cudaMemcpyDeviceToDevice));

        T a = -1.0;
        T b = 1.0;

        CHECK_CUSPARSE(cusparseDbsrmv(
            cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb,
            &a, descrK, d_vals, d_rowp, d_cols, block_dim, soln_in.getPtr(), &b, d_resid));

        T resid_norm;

        CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &resid_norm));

        return resid_norm;
    }

    /*
     * Estimate the projected total solve time using the logarithmic
     * convergence rate over the final 10% of completed GMRES iterations.
     *
     * The residual history contains the rotated GMRES residual estimate
     * abs(g[j + 1]) from every inner iteration.
     *
     *     local_rate =
     *         log(r_current / r_start) /
     *         (t_current - t_start)
     *
     *     additional_time =
     *         log(r_target / r_current) /
     *         local_rate
     *
     *     estimated_total_time =
     *         elapsed_time + additional_time
     */
    double estimate_total_solve_time(int completed_iterations, T current_resid, T conv_tol,
                                     double elapsed_sec,
                                     const std::vector<double> &residual_history,
                                     const std::vector<double> &elapsed_history) const {
        if (completed_iterations <= 0 || elapsed_sec <= 0.0 || residual_history.size() < 2 ||
            residual_history.size() != elapsed_history.size()) {
            return -1.0;
        }

        const int history_size = static_cast<int>(residual_history.size());

        const int window = std::max(
            1, static_cast<int>(std::ceil(0.10 * static_cast<double>(completed_iterations))));

        const int end_index = history_size - 1;
        const int start_index = std::max(0, end_index - window);

        const double r_start = residual_history[start_index];

        const double r_current = std::abs(static_cast<double>(current_resid));

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

        if (r_current >= r_start) {
            return -1.0;
        }

        const double local_log_rate = std::log(r_current / r_start) / local_elapsed_sec;

        if (!std::isfinite(local_log_rate) || local_log_rate >= 0.0) {
            return -1.0;
        }

        double additional_sec = std::log(r_target / r_current) / local_log_rate;

        if (!std::isfinite(additional_sec)) {
            return -1.0;
        }

        additional_sec = std::max(0.0, additional_sec);

        return elapsed_sec + additional_sec;
    }

    void print_failure_time_estimate(int completed_iterations, T current_resid, T conv_tol,
                                     double elapsed_sec,
                                     const std::vector<double> &residual_history,
                                     const std::vector<double> &elapsed_history) const {
        const double estimated_total_sec =
            estimate_total_solve_time(completed_iterations, current_resid, conv_tol, elapsed_sec,
                                      residual_history, elapsed_history);

        if (estimated_total_sec >= 0.0 && std::isfinite(estimated_total_sec)) {
            std::printf(
                "GMRES did not converge after %d iterations. "
                "Estimated total solve time: %.2fs\n",
                completed_iterations, estimated_total_sec);
        } else {
            std::printf(
                "GMRES did not converge after %d iterations. "
                "Estimated total solve time unavailable.\n",
                completed_iterations);
        }

        std::fflush(stdout);
    }

    bool solve(DeviceVec<T> rhs_in, DeviceVec<T> soln_out, bool check_conv = false) {
        T a;
        T b;

        total_iter = 0;

        bool converged = false;

        T init_beta;

        CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, rhs_in.getPtr(), 1, &init_beta));

        const T conv_tol = options.atol + init_beta * options.rtol;

        CHECK_CUDA(cudaMemset(d_x, 0, N * sizeof(T)));

        /*
         * Synchronize before starting the timer so previously queued GPU work
         * is not included in this solve's elapsed time.
         */
        CHECK_CUDA(cudaDeviceSynchronize());

        const auto solve_start = std::chrono::high_resolution_clock::now();

        std::vector<double> residual_history;
        std::vector<double> elapsed_history;

        residual_history.reserve(MAX_ITER);
        elapsed_history.reserve(MAX_ITER);

        T latest_gmres_resid = init_beta;

        /*
         * Preserve the original restart count behavior.
         *
         * This performs floor(MAX_ITER / N_SUBSPACE) complete restart cycles.
         */
        const int num_outer_iterations = MAX_ITER / N_SUBSPACE;
        // printf("MAX_ITER = %d, N_SUBSPACE = %d\n");

        for (int iouter = 0; iouter < num_outer_iterations; iouter++) {
            int jj = N_SUBSPACE - 1;

            CHECK_CUDA(cudaMemset(d_Vmat, 0, (N_SUBSPACE + 1) * N * sizeof(T)));

            std::memset(g, 0, (N_SUBSPACE + 1) * sizeof(T));

            std::memset(cs, 0, N_SUBSPACE * sizeof(T));

            std::memset(ss, 0, N_SUBSPACE * sizeof(T));

            std::memset(H, 0, (N_SUBSPACE + 1) * N_SUBSPACE * sizeof(T));

            /*
             * Compute the true residual:
             *
             *     r = rhs - A*x
             */
            CHECK_CUDA(
                cudaMemcpy(d_resid, rhs_in.getPtr(), N * sizeof(T), cudaMemcpyDeviceToDevice));

            a = -1.0;
            b = 1.0;

            CHECK_CUSPARSE(cusparseDbsrmv(
                cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb,
                nnzb, &a, descrK, d_vals, d_rowp, d_cols, block_dim, d_x, &b, d_resid));

            T init_true_resid;

            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &init_true_resid));

            T beta;

            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &beta));

            if (options.debug) {
                CHECK_CUDA(cudaDeviceSynchronize());
            }

            if (options.print) {
                std::printf(
                    "GMRES init resid = true %.9e, "
                    "precond %.9e\n",
                    static_cast<double>(init_true_resid), static_cast<double>(beta));
            }

            g[0] = beta;

            /*
             * Handle an exactly zero residual at the start of a restart.
             */
            if (std::abs(beta) <= conv_tol) {
                converged = true;
                latest_gmres_resid = beta;
                break;
            }

            /*
             * v0 = r0 / beta
             */
            a = 1.0 / beta;

            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_resid, 1, &d_Vmat[0], 1));

            for (int j = 0; j < N_SUBSPACE; j++) {
                /*
                 * Apply the right preconditioner:
                 *
                 *     tmp2 = M^-1 * v_j
                 */
                CHECK_CUDA(
                    cudaMemcpy(d_tmp, &d_Vmat[j * N], N * sizeof(T), cudaMemcpyDeviceToDevice));

                if (pc) {
                    pc->solve(d_tmp_vec, d_tmp2_vec);
                } else {
                    CHECK_CUDA(cudaMemcpy(d_tmp2, d_tmp, N * sizeof(T), cudaMemcpyDeviceToDevice));
                }

                /*
                 * w = A * tmp2
                 */
                a = 1.0;
                b = 0.0;

                CHECK_CUSPARSE(cusparseDbsrmv(
                    cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, mb,
                    mb, nnzb, &a, descrK, d_vals, d_rowp, d_cols, block_dim, d_tmp2, &b, d_w));

                /*
                 * Arnoldi orthogonalization.
                 */
                for (int i = 0; i < j + 1; i++) {
                    CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_w, 1, &d_Vmat[i * N], 1,
                                            &H[N_SUBSPACE * i + j]));

                    a = -H[N_SUBSPACE * i + j];

                    CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, &d_Vmat[i * N], 1, d_w, 1));
                }

                CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_w, 1, &H[N_SUBSPACE * (j + 1) + j]));

                a = 1.0 / H[N_SUBSPACE * (j + 1) + j];

                CHECK_CUBLAS(cublasDcopy(cublasHandle, N, d_w, 1, &d_Vmat[(j + 1) * N], 1));

                CHECK_CUBLAS(cublasDscal(cublasHandle, N, &a, &d_Vmat[(j + 1) * N], 1));

                /*
                 * Apply previous Givens rotations.
                 */
                for (int i = 0; i < j; i++) {
                    T temp = H[i * N_SUBSPACE + j];

                    H[N_SUBSPACE * i + j] =
                        cs[i] * H[N_SUBSPACE * i + j] + ss[i] * H[N_SUBSPACE * (i + 1) + j];

                    H[N_SUBSPACE * (i + 1) + j] =
                        -ss[i] * temp + cs[i] * H[N_SUBSPACE * (i + 1) + j];
                }

                T hx = H[N_SUBSPACE * j + j];

                T hy = H[N_SUBSPACE * (j + 1) + j];

                T r = std::hypot(hx, hy);

                cs[j] = hx / r;
                ss[j] = hy / r;

                T g_temp = g[j];

                g[j] *= cs[j];
                g[j + 1] = -ss[j] * g_temp;

                H[N_SUBSPACE * j + j] = r;
                H[N_SUBSPACE * (j + 1) + j] = 0.0;

                /*
                 * This inner iteration is now complete.
                 */
                total_iter++;

                latest_gmres_resid = std::abs(g[j + 1]);

                /*
                 * Record completed GPU work before taking the timestamp.
                 * The cuBLAS norm/dot operations generally synchronize through
                 * scalar host results, but the explicit synchronization keeps
                 * timing semantics unambiguous.
                 */
                CHECK_CUDA(cudaDeviceSynchronize());

                const auto now = std::chrono::high_resolution_clock::now();

                const double elapsed_sec = std::chrono::duration<double>(now - solve_start).count();

                residual_history.push_back(std::abs(static_cast<double>(latest_gmres_resid)));

                elapsed_history.push_back(elapsed_sec);

                if (options.print && ((total_iter - 1) % options.print_freq == 0)) {
                    std::printf("GMRES iter %d : resid %.9e\n", total_iter - 1,
                                static_cast<double>(latest_gmres_resid));
                }

                if (check_conv && latest_gmres_resid < conv_tol) {
                    jj = j;
                    converged = true;
                    break;
                }
            }

            /*
             * Solve the reduced Hessenberg system.
             */
            std::memset(Hred, 0, (N_SUBSPACE + 1) * (N_SUBSPACE + 1) * sizeof(T));

            for (int i = 0; i < jj + 1; i++) {
                for (int j = 0; j < jj + 1; j++) {
                    Hred[(jj + 1) * i + j] = H[N_SUBSPACE * j + i];
                }
            }

            CHECK_CUDA(
                cudaMemcpy(d_Hred, Hred, (jj + 1) * (jj + 1) * sizeof(T), cudaMemcpyHostToDevice));

            CHECK_CUDA(cudaMemcpy(d_gred, g, (jj + 1) * sizeof(T), cudaMemcpyHostToDevice));

            CHECK_CUBLAS(cublasDtrsv(cublasHandle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                                     CUBLAS_DIAG_NON_UNIT, jj + 1, d_Hred, jj + 1, d_gred, 1));

            CHECK_CUDA(cudaMemcpy(h_y, d_gred, (jj + 1) * sizeof(T), cudaMemcpyDeviceToHost));

            CHECK_CUDA(cudaMemset(d_xR, 0, N * sizeof(T)));

            for (int j = 0; j < jj + 1; j++) {
                a = h_y[j];

                CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, &d_Vmat[j * N], 1, d_xR, 1));
            }

            /*
             * Apply the right preconditioner to the accumulated Krylov-space
             * update.
             */
            if (pc) {
                pc->solve(d_xR_vec, d_tmp_vec);
            } else {
                CHECK_CUDA(cudaMemcpy(d_tmp, d_xR, N * sizeof(T), cudaMemcpyDeviceToDevice));
            }

            CHECK_CUDA(cudaMemcpy(d_xR, d_tmp, N * sizeof(T), cudaMemcpyDeviceToDevice));

            /*
             * x = x0 + xR
             */
            a = 1.0;

            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_xR, 1, d_x, 1));

            if (converged) {
                break;
            }
        }

        /*
         * Compute the final true residual.
         */
        CHECK_CUDA(cudaMemcpy(d_resid, rhs_in.getPtr(), N * sizeof(T), cudaMemcpyDeviceToDevice));

        a = -1.0;
        b = 1.0;

        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a, descrK,
                                      d_vals, d_rowp, d_cols, block_dim, d_x, &b, d_resid));

        T final_resid;

        CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &final_resid));

        CHECK_CUDA(cudaDeviceSynchronize());

        const auto solve_end = std::chrono::high_resolution_clock::now();

        const double elapsed_sec = std::chrono::duration<double>(solve_end - solve_start).count();

        /*
         * Use the final true residual for the printed result.
         *
         * For the time projection, the most recent GMRES residual estimate is
         * used because the history is also composed of GMRES residual
         * estimates. This keeps the logarithmic rate calculation consistent.
         */
        if (options.print) {
            if (converged) {
                std::printf(
                    "GMRES converged to %.4e resid in %d "
                    "iterations\n",
                    static_cast<double>(final_resid), total_iter);
            } else {
                std::printf(
                    "GMRES did NOT CONVERGE with %.4e resid "
                    "in %d iterations\n",
                    static_cast<double>(final_resid), total_iter);

                print_failure_time_estimate(total_iter, latest_gmres_resid, conv_tol, elapsed_sec,
                                            residual_history, elapsed_history);

                std::printf("\n");
            }
        }

        CHECK_CUDA(cudaMemcpy(soln_out.getPtr(), d_x, N * sizeof(T), cudaMemcpyDeviceToDevice));

        CHECK_CUDA(cudaDeviceSynchronize());

        return !converged;
    }

    int get_num_iterations() { return total_iter; }

    void free() {
        if (is_free) {
            return;
        }

        is_free = true;

        if (grid) {
            grid->free();
        }

        d_resid_vec.free();
        d_z_vec.free();
        d_tmp_vec.free();
        d_tmp2_vec.free();
        d_xR_vec.free();

        if (d_x) {
            cudaFree(d_x);
            d_x = nullptr;
        }

        if (d_w) {
            cudaFree(d_w);
            d_w = nullptr;
        }

        /*
         * d_tmp, d_tmp2, d_z, d_resid, and d_xR are owned by their
         * corresponding DeviceVec objects and are released above.
         *
         * d_Vmat, d_Hred, and d_gred were obtained from temporary DeviceVec
         * objects in the original constructor pattern. Their ownership depends
         * on DeviceVec::getPtr() behavior, so this preserves the original
         * allocation/free convention rather than changing ownership here.
         */

        if (descrK) {
            cusparseDestroyMatDescr(descrK);
            descrK = nullptr;
        }

        delete[] h_y;
        h_y = nullptr;

        delete[] Hred;
        Hred = nullptr;
    }

    GRID *grid = nullptr;
    BaseSolver *pc = nullptr;

    SolverOptions options;
    int ilevel = -1;

   private:
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

    T *d_x = nullptr;
    T *d_resid = nullptr;

    DeviceVec<T> d_resid_vec;

    int total_iter = 0;
    bool is_free = false;

    void *pBuffer = nullptr;

    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_USE_LEVEL;

    const cusparseSolvePolicy_t policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;

    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE;

    const cusparseOperation_t trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;

    const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;

    cusparseHandle_t &cusparseHandle;
    cublasHandle_t &cublasHandle;

    cusparseMatDescr_t descrK = nullptr;

    DeviceVec<T> d_z_vec;
    DeviceVec<T> d_tmp_vec;
    DeviceVec<T> d_tmp2_vec;
    DeviceVec<T> d_xR_vec;

    T *d_tmp = nullptr;
    T *d_tmp2 = nullptr;
    T *d_w = nullptr;
    T *d_z = nullptr;
    T *d_xR = nullptr;

    T *d_Hred = nullptr;
    T *d_gred = nullptr;
    T *h_y = nullptr;
    T *d_Vmat = nullptr;
    T *Hred = nullptr;

    cusparseMatDescr_t descr_L = nullptr;
    cusparseMatDescr_t descr_U = nullptr;

    bsrsv2Info_t info_L = nullptr;
    bsrsv2Info_t info_U = nullptr;

    T g[N_SUBSPACE + 1];
    T cs[N_SUBSPACE];
    T ss[N_SUBSPACE];

    T H[(N_SUBSPACE + 1) * N_SUBSPACE];

    int MAX_ITER = 0;
};