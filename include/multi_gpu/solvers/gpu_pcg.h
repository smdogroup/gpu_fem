#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "matvec/gpumat.h"
#include "matvec/gpuvec.h"
#include "utils/multigpu_context.h"

template <typename T, class Partitioner, class PRECOND>
class GPU_PCG {
   public:
    using Vec = GPUvec<T, Partitioner>;
    using Mat = GPUbsrmat<T, Partitioner>;

    GPU_PCG(MultiGPUContext *ctx_, const Partitioner *part_, Mat *A_, PRECOND *M_, int N_,
            int block_dim_ = 6, const char *precond_name_ = "Unspecified")
        : ctx(ctx_),
          part(part_),
          A(A_),
          M(M_),
          N(N_),
          block_dim(block_dim_),
          precond_name(precond_name_) {
        resid = new Vec(ctx, part, block_dim);
        w = new Vec(ctx, part, block_dim);
        p = new Vec(ctx, part, block_dim);
        z = new Vec(ctx, part, block_dim);
    }

    void free() {
        resid->free();
        w->free();
        p->free();
        z->free();
    }

    double residual_progress_fraction(T resid_norm, T init_resid_norm, T conv_tol,
                                      bool finished = false) {
        if (finished) {
            return 1.0;
        }

        double r0 = std::abs(double(init_resid_norm));
        double r = std::abs(double(resid_norm));
        double rt = std::abs(double(conv_tol));

        if (r0 <= 0.0 || rt <= 0.0) {
            return 1.0;
        }

        if (r <= rt) {
            return 1.0;
        }

        if (r >= r0) {
            return 0.0;
        }

        double denom = std::log(r0) - std::log(rt);

        if (std::abs(denom) <= 1e-300) {
            return 1.0;
        }

        double frac = (std::log(r0) - std::log(r)) / denom;

        if (frac < 0.0) {
            frac = 0.0;
        }

        if (frac > 1.0) {
            frac = 1.0;
        }

        return frac;
    }

    double predicted_seconds_left(int iter, T resid_norm, T init_resid_norm, T conv_tol,
                                  double elapsed_sec) {
        if (iter <= 0 || elapsed_sec <= 0.0) {
            return -1.0;
        }

        double frac = residual_progress_fraction(resid_norm, init_resid_norm, conv_tol);

        if (frac <= 1e-12) {
            return -1.0;
        }

        if (frac >= 1.0) {
            return 0.0;
        }

        return elapsed_sec * (1.0 - frac) / frac;
    }

    /*
     * Estimate the total solve time after failure using the logarithmic
     * residual decay rate measured over the final 10% of iterations.
     *
     * Local convergence rate:
     *
     *     rate = log(r_current / r_start) / (t_current - t_start)
     *
     * Estimated additional time:
     *
     *     additional_time = log(r_target / r_current) / rate
     *
     * Estimated total solve time:
     *
     *     total_time = elapsed_time + additional_time
     */
    void print_failure_time_estimate(int iter, T resid_norm, T conv_tol, double elapsed_sec,
                                     const std::vector<double> &residual_history,
                                     const std::vector<double> &elapsed_history) {
        double estimated_total_sec = -1.0;

        if (iter > 0 && elapsed_sec > 0.0 && residual_history.size() >= 2 &&
            residual_history.size() == elapsed_history.size()) {
            int window = std::max(1, int(std::ceil(0.10 * double(iter))));

            int end_index = int(residual_history.size()) - 1;
            int start_index = std::max(0, end_index - window);

            double r_start = residual_history[start_index];
            double r_current = std::abs(double(resid_norm));
            double r_target = std::abs(double(conv_tol));

            double t_start = elapsed_history[start_index];
            double t_current = elapsed_history[end_index];
            double local_elapsed_sec = t_current - t_start;

            if (std::isfinite(r_start) && std::isfinite(r_current) && std::isfinite(r_target) &&
                std::isfinite(t_start) && std::isfinite(t_current) &&
                std::isfinite(local_elapsed_sec) && r_start > 0.0 && r_current > 0.0 &&
                r_target > 0.0 && local_elapsed_sec > 0.0 && r_current < r_start) {
                double local_log_rate = std::log(r_current / r_start) / local_elapsed_sec;

                if (std::isfinite(local_log_rate) && local_log_rate < 0.0) {
                    double additional_sec = std::log(r_target / r_current) / local_log_rate;

                    if (std::isfinite(additional_sec)) {
                        additional_sec = std::max(0.0, additional_sec);
                        estimated_total_sec = elapsed_sec + additional_sec;
                    }
                }
            }
        }

        if (estimated_total_sec >= 0.0 && std::isfinite(estimated_total_sec)) {
            std::printf(
                "PCG did not converge after %d iterations. "
                "Estimated total solve time: %.2fs\n",
                iter, estimated_total_sec);
        } else {
            std::printf(
                "PCG did not converge after %d iterations. "
                "Estimated total solve time unavailable.\n",
                iter);
        }

        std::fflush(stdout);
    }

    void print_progress_bar(int iter, int max_iter, T resid_norm, T init_resid_norm, T conv_tol,
                            double elapsed_sec, bool finished = false) {
        const int width = 36;

        double frac = residual_progress_fraction(resid_norm, init_resid_norm, conv_tol, finished);

        int filled = int(frac * width);

        const char *green = "\033[42m";
        const char *gray = "\033[100m";
        const char *reset = "\033[0m";

        double rel = 0.0;

        if (std::abs(double(init_resid_norm)) > 0.0) {
            rel = std::abs(double(resid_norm) / double(init_resid_norm));
        }

        double eta =
            predicted_seconds_left(iter, resid_norm, init_resid_norm, conv_tol, elapsed_sec);

        // Keep the cursor parked on the blank spacer line below the progress bar.
        std::printf("\033[1A");
        std::printf("\r\033[K");

        std::printf("%4d/%d [", iter, max_iter);

        std::printf("%s", green);

        for (int i = 0; i < filled; i++) {
            std::printf(" ");
        }

        std::printf("%s", gray);

        for (int i = filled; i < width; i++) {
            std::printf(" ");
        }

        std::printf("%s", reset);

        if (finished) {
            std::printf("] %.2fs", elapsed_sec);
        } else if (eta >= 0.0) {
            std::printf("] %.2fs", eta);
        } else {
            std::printf("] ---");
        }

        std::printf(" - resid: %.8e - tol: %.3e - rel: %.3e", double(resid_norm), double(conv_tol),
                    rel);

        // Return to the blank spacer line below the progress bar.
        std::printf("\n");
        std::printf("\r\033[K");

        std::fflush(stdout);
    }

    int solve(Vec *rhs, Vec *x, int max_iter = 500, T abs_tol = 1e-8, T rel_tol = 1e-8,
              int print_freq = 50, bool can_print = true, bool use_progress_bar = true) {
        T a = 0.0;
        T b = 0.0;

        ctx->sync();

        auto start = std::chrono::high_resolution_clock::now();

        rhs->copyTo(resid);

        a = -1.0;
        b = 1.0;
        A->mult(a, x, b, resid);

        T init_resid_norm = resid->norm();
        T conv_tol = abs_tol + init_resid_norm * rel_tol;

        if (can_print && use_progress_bar) {
            std::printf("\nPCG with Precond=%s, ndof=%d\n", precond_name, N);

            // Create the blank spacer line where the cursor will live.
            std::printf("\n");

            print_progress_bar(0, max_iter, init_resid_norm, init_resid_norm, conv_tol, 0.0);
        }

        if (can_print && !use_progress_bar) {
            std::printf("\nPCG with Precond=%s, ndof=%d\n", precond_name, N);

            std::printf("PCG init_resid = %.8e, target = %.3e\n", double(init_resid_norm),
                        double(conv_tol));
        }

        M->solve(resid, z);
        z->copyTo(p);

        bool converged = false;
        int total_iter = 0;
        T final_resid_norm = init_resid_norm;

        std::vector<double> residual_history;
        std::vector<double> elapsed_history;

        residual_history.reserve(max_iter);
        elapsed_history.reserve(max_iter);

        for (int j = 0; j < max_iter; j++) {
            a = 1.0;
            b = 0.0;
            A->mult(a, p, b, w);

            T rz0 = resid->dotProd(z);
            T wp0 = w->dotProd(p);
            T alpha = rz0 / wp0;

            x->axpy(alpha, p);

            a = -alpha;
            resid->axpy(a, w);

            M->solve(resid, z);

            T rz1 = resid->dotProd(z);
            T beta = rz1 / rz0;

            p->scale(beta);

            a = 1.0;
            p->axpy(a, z);

            T resid_norm = resid->norm();

            final_resid_norm = resid_norm;
            total_iter = j + 1;

            auto now = std::chrono::high_resolution_clock::now();

            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - start);

            double elapsed_sec = elapsed.count() / 1e6;

            residual_history.push_back(std::abs(double(resid_norm)));
            elapsed_history.push_back(elapsed_sec);

            if (can_print && use_progress_bar && (j % print_freq == 0 || j == max_iter - 1)) {
                print_progress_bar(j + 1, max_iter, resid_norm, init_resid_norm, conv_tol,
                                   elapsed_sec);
            }

            if (can_print && !use_progress_bar && j % print_freq == 0) {
                std::printf("PCG [%d] = %.8e, target = %.3e\n", j, double(resid_norm),
                            double(conv_tol));
            }

            if (std::abs(resid_norm) < conv_tol) {
                converged = true;

                auto finish_now = std::chrono::high_resolution_clock::now();

                auto finish_elapsed =
                    std::chrono::duration_cast<std::chrono::microseconds>(finish_now - start);

                double finish_elapsed_sec = finish_elapsed.count() / 1e6;

                if (can_print && use_progress_bar) {
                    print_progress_bar(j + 1, max_iter, resid_norm, init_resid_norm, conv_tol,
                                       finish_elapsed_sec, true);

                    std::printf("\n");
                }

                if (can_print && !use_progress_bar) {
                    std::printf("\tPCG converged in %d iterations to %.9e resid\n", j + 1,
                                double(resid_norm));
                }

                break;
            }
        }

        if (can_print && use_progress_bar && !converged) {
            auto now = std::chrono::high_resolution_clock::now();

            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - start);

            double elapsed_sec = elapsed.count() / 1e6;

            /*
             * The cursor is already on the blank spacer line directly below
             * the final progress-bar printout.
             */
            print_failure_time_estimate(total_iter, final_resid_norm, conv_tol, elapsed_sec,
                                        residual_history, elapsed_history);

            std::printf("\n");
        }

        ctx->sync();

        if (can_print && !use_progress_bar) {
            if (converged) {
                std::printf("\tPCG converged in %d iterations to %.9e resid\n", total_iter,
                            double(final_resid_norm));
            } else {
                auto now = std::chrono::high_resolution_clock::now();

                auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - start);

                double elapsed_sec = elapsed.count() / 1e6;

                std::printf(
                    "\tPCG failed to converge after %d iterations, "
                    "resid = %.9e\n",
                    total_iter, double(final_resid_norm));

                print_failure_time_estimate(total_iter, final_resid_norm, conv_tol, elapsed_sec,
                                            residual_history, elapsed_history);
            }
        }

        return converged ? total_iter : -total_iter;
    }

    MultiGPUContext *ctx = nullptr;
    const Partitioner *part = nullptr;
    Mat *A = nullptr;
    PRECOND *M = nullptr;

    int N = 0;
    int block_dim = 0;

    const char *precond_name = "Unspecified";

    Vec *resid = nullptr;
    Vec *w = nullptr;
    Vec *p = nullptr;
    Vec *z = nullptr;
};