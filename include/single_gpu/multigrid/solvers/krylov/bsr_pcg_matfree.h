#pragma once

#ifdef USE_GPU
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

#include "../solve_utils.h"

template <typename T, class Operator>
class MatrixFreePCGSolver : public BaseSolver {
   public:
    MatrixFreePCGSolver(cublasHandle_t &cublasHandle_, Operator *op_, BaseSolver *pc_,
                        SolverOptions options_, int N_, int ilevel_ = -1)
        : op(op_),
          pc(pc_),
          options(options_),
          N(N_),
          ilevel(ilevel_),
          cublasHandle(cublasHandle_),
          r_vec(N_),
          z_vec(N_),
          p_vec(N_),
          w_vec(N_),
          x_vec(N_),
          rhs_vec(N_) {}

    void update_after_assembly(DeviceVec<T> &vars) override {
        // matrix-free operator usually does not need anything here
        // unless the preconditioner does
        if (pc) {
            // printf("pc update after assembly in krylov\n");
            pc->update_after_assembly(vars);
            // printf("\tdone with pc update after assembly in krylov\n");
        }
    }

    void factor() override {
        // nothing for Krylov itself
    }

    void set_print(bool print) override { options.print = print; }
    void set_abs_tol(T atol) override { options.atol = atol; }
    void set_rel_tol(T rtol) override { options.rtol = rtol; }
    void set_cycle_type(std::string cycle_) override {
        if (pc) pc->set_cycle_type(cycle_);
    }

    int get_num_iterations() override { return n_steps; }

    T getResidualNorm(DeviceVec<T> rhs_in, DeviceVec<T> soln_in) {
        // r = rhs - A x
        CHECK_CUDA(
            cudaMemcpy(r_vec.getPtr(), rhs_in.getPtr(), N * sizeof(T), cudaMemcpyDeviceToDevice));

        op->mat_vec(soln_in, w_vec);

        T minus_one = -1.0;
        CHECK_CUBLAS(
            cublasDaxpy(cublasHandle, N, &minus_one, w_vec.getPtr(), 1, r_vec.getPtr(), 1));

        T resid_norm = 0.0;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, r_vec.getPtr(), 1, &resid_norm));
        return resid_norm;
    }

    bool solve(DeviceVec<T> rhs_in, DeviceVec<T> soln_out, bool check_conv = false) override {
        // x = 0
        x_vec.zeroValues();

        // rhs cache
        CHECK_CUDA(
            cudaMemcpy(rhs_vec.getPtr(), rhs_in.getPtr(), N * sizeof(T), cudaMemcpyDeviceToDevice));

        // r = rhs - A*x = rhs initially since x=0
        CHECK_CUDA(
            cudaMemcpy(r_vec.getPtr(), rhs_vec.getPtr(), N * sizeof(T), cudaMemcpyDeviceToDevice));

        n_steps = 0;
        bool converged = false;

        T init_resid_norm = 0.0;
        if (check_conv) {
            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, r_vec.getPtr(), 1, &init_resid_norm));
            if (options.print) {
                if (ilevel == 0)
                    printf("TopLEVEL MF-PCG init_resid = %.8e\n", init_resid_norm);
                else
                    printf("\tL%d-MF-PCG init_resid = %.8e\n", ilevel, init_resid_norm);
            }
        }

        T rho_prev = 0.0;

        for (int j = 0; j < options.ncycles; j++) {
            // z = M^{-1} r
            if (pc) {
                pc->solve(r_vec, z_vec);
            } else {
                CHECK_CUDA(cudaMemcpy(z_vec.getPtr(), r_vec.getPtr(), N * sizeof(T),
                                      cudaMemcpyDeviceToDevice));
            }

            T rho = 0.0;
            CHECK_CUBLAS(cublasDdot(cublasHandle, N, r_vec.getPtr(), 1, z_vec.getPtr(), 1, &rho));

            if (j == 0) {
                CHECK_CUDA(cudaMemcpy(p_vec.getPtr(), z_vec.getPtr(), N * sizeof(T),
                                      cudaMemcpyDeviceToDevice));
            } else {
                T beta = rho / rho_prev;

                CHECK_CUBLAS(cublasDscal(cublasHandle, N, &beta, p_vec.getPtr(), 1));

                T one = 1.0;
                CHECK_CUBLAS(
                    cublasDaxpy(cublasHandle, N, &one, z_vec.getPtr(), 1, p_vec.getPtr(), 1));
            }

            rho_prev = rho;

            // w = A p
            op->mat_vec(p_vec, w_vec);

            T pAp = 0.0;
            CHECK_CUBLAS(cublasDdot(cublasHandle, N, p_vec.getPtr(), 1, w_vec.getPtr(), 1, &pAp));

            T alpha = rho / pAp;

            // x += alpha p
            CHECK_CUBLAS(
                cublasDaxpy(cublasHandle, N, &alpha, p_vec.getPtr(), 1, x_vec.getPtr(), 1));

            // r -= alpha w
            T neg_alpha = -alpha;
            CHECK_CUBLAS(
                cublasDaxpy(cublasHandle, N, &neg_alpha, w_vec.getPtr(), 1, r_vec.getPtr(), 1));

            n_steps = j + 1;

            if (check_conv) {
                T resid_norm = 0.0;
                CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, r_vec.getPtr(), 1, &resid_norm));

                if (options.print && (j % options.print_freq == 0)) {
                    if (ilevel == 0)
                        printf("MF-PCG [%d] = %.8e\n", j, resid_norm);
                    else
                        printf("\tL%d-MF-PCG [%d] = %.8e\n", ilevel, j, resid_norm);
                }

                if (abs(resid_norm) < (options.atol + options.rtol * init_resid_norm)) {
                    converged = true;
                    if (options.print) {
                        if (ilevel == 0)
                            printf("MF-PCG converged in %d iterations to %.9e resid\n", j + 1,
                                   resid_norm);
                        else
                            printf("\tL%d-MF-PCG converged in %d iterations to %.9e resid\n",
                                   ilevel, j + 1, resid_norm);
                    }
                    break;
                }
            }
        }

        if (options.debug) {
            // true residual check: r = rhs - A*x
            CHECK_CUDA(cudaMemcpy(r_vec.getPtr(), rhs_vec.getPtr(), N * sizeof(T),
                                  cudaMemcpyDeviceToDevice));
            op->mat_vec(x_vec, w_vec);
            T minus_one = -1.0;
            CHECK_CUBLAS(
                cublasDaxpy(cublasHandle, N, &minus_one, w_vec.getPtr(), 1, r_vec.getPtr(), 1));

            T resid_norm = 0.0;
            T x_norm = 0.0;
            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, r_vec.getPtr(), 1, &resid_norm));
            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, x_vec.getPtr(), 1, &x_norm));
            printf("debug: MF-PCG true residual %.8e, x norm %.8e\n", resid_norm, x_norm);
        }

        CHECK_CUDA(
            cudaMemcpy(soln_out.getPtr(), x_vec.getPtr(), N * sizeof(T), cudaMemcpyDeviceToDevice));

        return !converged;
    }

    void free() override {
        if (is_free) return;
        is_free = true;
        r_vec.free();
        z_vec.free();
        p_vec.free();
        w_vec.free();
        x_vec.free();
        rhs_vec.free();
    }
    int get_num_dof() { return N; }

   private:
    Operator *op;
    BaseSolver *pc;
    SolverOptions options;
    int N;
    int ilevel;
    int n_steps = 0;
    bool is_free = false;

    cublasHandle_t &cublasHandle;

    DeviceVec<T> r_vec, z_vec, p_vec, w_vec, x_vec, rhs_vec;
};