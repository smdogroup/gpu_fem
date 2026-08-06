#pragma once

#ifdef USE_GPU
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif
#include "../solve_utils.h"

template <typename T, class Operator, class GRID, int N_SUBSPACE = 50>
class MatrixFreeGMRESSolver : public BaseSolver {
   public:
    MatrixFreeGMRESSolver(cublasHandle_t &cublasHandle_, cusparseHandle_t &cusparseHandle_,
                          GRID *grid_, Operator *op_, BaseSolver *pc_, SolverOptions options,
                          int MAX_ITER_ = 200, int N_ = 0)
        : grid(grid_),
          op(op_),
          pc(pc_),
          options(options),
          cublasHandle(cublasHandle_),
          cusparseHandle(cusparseHandle_) {
        // get matrix and init other temp data for PCG solve
        mat = grid->Kmat;
        // soln = grid->d_soln;
        // rhs = grid->d_rhs;
        MAX_ITER = MAX_ITER_;

        auto bsr_data = mat.getBsrData();
        mb = bsr_data.nnodes;
        nnzb = bsr_data.nnzb;
        block_dim = bsr_data.block_dim;
        d_rowp = bsr_data.rowp;
        d_cols = bsr_data.cols;
        iperm = bsr_data.iperm;
        d_vals = mat.getPtr();

        if (!pc) printf("\nWARNING : GMRES solver was constructed with no preconditioner\n\n");

        // cublasHandle = grid->cublasHandle;
        // cusparseHandle = grid->cusparseHandle;

        if (N_ == 0) {
            N = grid->N;
        } else {
            N = N_;
        }
        d_x_vec = DeviceVec<T>(N);
        d_x = d_x_vec.getPtr();  // needs to be separate vec than soln in grid

        // printf("PCG Krylov solver made with options ncycl %d and print %d, with problem size
        // %d\n", options.ncycles, options.print, N);

        // description of the K matrix
        descrK = 0;
        CHECK_CUSPARSE(cusparseCreateMatDescr(&descrK));
        CHECK_CUSPARSE(cusparseSetMatType(descrK, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descrK, CUSPARSE_INDEX_BASE_ZERO));

        // make temp vecs
        d_tmp_vec = DeviceVec<T>(N);
        d_tmp = d_tmp_vec.getPtr();
        d_tmp2_vec = DeviceVec<T>(N);
        d_tmp2 = d_tmp2_vec.getPtr();
        d_xR_vec = DeviceVec<T>(N);
        d_xR = d_xR_vec.getPtr();
        d_Vmat = DeviceVec<T>((N_SUBSPACE + 1) * N * sizeof(T)).getPtr();

        d_resid_vec = DeviceVec<T>(N);
        d_resid = d_resid_vec.getPtr();
        d_w_vec = DeviceVec<T>(N);
        d_w = d_w_vec.getPtr();
        d_z_vec = DeviceVec<T>(N);
        d_z = d_z_vec.getPtr();

        d_Hred = DeviceVec<T>((N_SUBSPACE + 1) * (N_SUBSPACE + 1)).getPtr();
        d_gred = DeviceVec<T>((N_SUBSPACE + 1)).getPtr();
        h_y = new T[N_SUBSPACE + 1];
        Hred = new T[(N_SUBSPACE + 1) * (N_SUBSPACE + 1)];
    }

    // nothing
    void update_after_assembly(DeviceVec<T> &vars) {
        bool perm = true;
        grid->setStateVars(vars, perm);
        grid->update_after_assembly();
        if (pc) pc->update_after_assembly(vars);
    }
    void factor() {}

    void set_print(bool print) { options.print = print; }
    void set_abs_tol(T atol) { options.atol = atol; }
    void set_rel_tol(T rtol) { options.rtol = rtol; }
    void set_cycle_type(std::string cycle_) {}

    T getResidualNorm(DeviceVec<T> rhs_in, DeviceVec<T> soln_in) {
        // compute r_0 = b - Ax
        CHECK_CUDA(cudaMemcpy(d_resid, rhs_in.getPtr(), N * sizeof(T), cudaMemcpyDeviceToDevice));
        op->mat_vec(soln_in, d_tmp_vec);
        T a = -1.0;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_tmp_vec.getPtr(), 1, d_resid, 1));

        T resid_norm;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &resid_norm));
        return resid_norm;
    }

    bool solve(DeviceVec<T> rhs_in, DeviceVec<T> soln_out, bool check_conv = false) {
        // assumes rhs_in and soln_out are in permutation for solve (not natural order)
        // performs full K-cycle with left-precond flexible PCG (note this shows true resid even
        // though it is left precond, unlike GMRES)!
        T a, b;
        total_iter = 0;
        bool converged = false;

        T init_beta;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, rhs_in.getPtr(), 1, &init_beta));

        // zero prev data on new call
        CHECK_CUDA(cudaMemset(d_x, 0.0, N * sizeof(T)));

        for (int iouter = 0; iouter < MAX_ITER / N_SUBSPACE; iouter++) {
            int jj = N_SUBSPACE - 1;

            // reset some states for new subspace
            CHECK_CUDA(cudaMemset(d_Vmat, 0.0, (N_SUBSPACE + 1) * N * sizeof(T)));
            memset(g, 0.0, (N_SUBSPACE + 1) * sizeof(T));
            memset(cs, 0.0, N_SUBSPACE * sizeof(T));
            memset(ss, 0.0, N_SUBSPACE * sizeof(T));
            memset(H, 0.0, (N_SUBSPACE + 1) * N_SUBSPACE * sizeof(T));

            // compute resid = rhs - A * x
            CHECK_CUDA(
                cudaMemcpy(d_resid, rhs_in.getPtr(), N * sizeof(T), cudaMemcpyDeviceToDevice));
            op->mat_vec(d_x_vec, d_tmp_vec);
            a = -1.0;
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_x_vec.getPtr(), 1, d_resid, 1));

            // get initial rhs
            T init_true_resid;
            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &init_true_resid));

            // GMRES initial residual
            // assumes here d_X is 0 initially => so r0 = b - Ax
            T beta;
            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &beta));
            if (options.debug) CHECK_CUDA(cudaDeviceSynchronize());
            if (options.print)
                printf("GMRES init resid = true %.9e, precond %.9e\n", init_true_resid, beta);
            g[0] = beta;

            // set v0 = r0 / beta (unit vec)
            a = 1.0 / beta;
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_resid, 1, &d_Vmat[0], 1));

            // then begin main GMRES iteration loop!
            for (int j = 0; j < N_SUBSPACE; j++, total_iter++) {
                /* right preconditioner */
                // U^-1 L^-1 * vj => d_tmp2 precond solve here
                CHECK_CUDA(
                    cudaMemcpy(d_tmp, &d_Vmat[j * N], N * sizeof(T), cudaMemcpyDeviceToDevice));
                if (pc) {
                    pc->solve(d_tmp_vec, d_tmp2_vec);
                } else {
                    // otherwise no preconditioner
                    cudaMemcpy(d_tmp2, d_tmp, N * sizeof(T), cudaMemcpyDeviceToDevice);
                }

                // w = A * vj + 0 * w
                // BSR matrix multiply here MV
                d_w_vec.zeroValues();
                op->mat_vec(d_tmp2_vec, d_w_vec);

                // now update householder matrix
                for (int i = 0; i < j + 1; i++) {
                    // compute w -= <w, vi> * vi
                    CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_w, 1, &d_Vmat[i * N], 1,
                                            &H[N_SUBSPACE * i + j]));
                    // if (options.debug) printf("H[%d,%d] = %.9e\n", i, j, H[N_SUBSPACE * i + j]);
                    a = -H[N_SUBSPACE * i + j];
                    CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, &d_Vmat[i * N], 1, d_w, 1));
                }

                // norm of w => H_{j+1,j}
                T nrm_w;
                CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_w, 1, &H[N_SUBSPACE * (j + 1) + j]));

                // v_{j+1} column unit vec = w / H_{j+1,j}
                a = 1.0 / H[N_SUBSPACE * (j + 1) + j];
                CHECK_CUBLAS(cublasDcopy(cublasHandle, N, d_w, 1, &d_Vmat[(j + 1) * N], 1));
                CHECK_CUBLAS(cublasDscal(cublasHandle, N, &a, &d_Vmat[(j + 1) * N], 1));

                // then givens rotations to elim householder matrix
                for (int i = 0; i < j; i++) {
                    T temp = H[i * N_SUBSPACE + j];
                    H[N_SUBSPACE * i + j] =
                        cs[i] * H[N_SUBSPACE * i + j] + ss[i] * H[N_SUBSPACE * (i + 1) + j];
                    H[N_SUBSPACE * (i + 1) + j] =
                        -ss[i] * temp + cs[i] * H[N_SUBSPACE * (i + 1) + j];
                }

                T hx = H[N_SUBSPACE * j + j];        // + 1e-12;
                T hy = H[N_SUBSPACE * (j + 1) + j];  // + 1e-12;
                T r = hypot(hx, hy);                 // always non-negative
                cs[j] = hx / r;
                ss[j] = hy / r;

                T g_temp = g[j];
                g[j] *= cs[j];
                g[j + 1] = -ss[j] * g_temp;

                // printf("GMRES iter %d : resid %.9e\n", j, nrm_w);
                if (options.print && (j % options.print_freq == 0))
                    printf("GMRES iter %d : resid %.9e\n", j, abs(g[j + 1]));

                H[N_SUBSPACE * j + j] = r;
                H[N_SUBSPACE * (j + 1) + j] = 0.0;

                if (check_conv && abs(g[j + 1]) < (options.atol + init_beta * options.rtol)) {
                    // if (options.print)
                    //     printf("GMRES converged in %d iterations to %.9e resid\n", j + 1, g[j +
                    //     1]);
                    jj = j;
                    converged = true;
                    break;
                }
            }  // end of inner loop

            // now solve Hessenberg triangular system
            // only up to size jj+1 x jj+1 where we exited on iteration jj
            memset(Hred, 0.0, (N_SUBSPACE + 1) * (N_SUBSPACE + 1) * sizeof(T));
            for (int i = 0; i < jj + 1; i++) {
                for (int j = 0; j < jj + 1; j++) {
                    // in-place transpose to be compatible with column-major cublasDtrsv later on
                    Hred[(jj + 1) * i + j] = H[N_SUBSPACE * j + i];
                }
            }

            // TODO : switch this part to solve on GPU..
            CHECK_CUDA(
                cudaMemcpy(d_Hred, Hred, (jj + 1) * (jj + 1) * sizeof(T), cudaMemcpyHostToDevice));

            // also create gred vector on the device
            CHECK_CUDA(cudaMemcpy(d_gred, g, (jj + 1) * sizeof(T), cudaMemcpyHostToDevice));

            // now solve Hessenberg system H * y = g
            CHECK_CUBLAS(cublasDtrsv(cublasHandle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                                     CUBLAS_DIAG_NON_UNIT, jj + 1, d_Hred, jj + 1, d_gred, 1));
            CHECK_CUDA(cudaMemcpy(h_y, d_gred, (jj + 1) * sizeof(T), cudaMemcpyDeviceToHost));

            // now compute matrix product xR = V * y (the preconditioned solution first)
            CHECK_CUDA(cudaMemset(d_xR, 0.0, N * sizeof(T)));
            for (int j = 0; j < jj + 1; j++) {
                a = h_y[j];
                // printf("h_y[%d] = %.4e\n", j, a);
                CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, &d_Vmat[j * N], 1, d_xR, 1));
            }

            // then compute xR = M^-1 xR (un-preconditions it back to x space)
            if (pc) {
                pc->solve(d_xR_vec, d_tmp_vec);
            } else {
                // no preconditioner
                cudaMemcpy(d_tmp, d_xR, N * sizeof(T), cudaMemcpyDeviceToDevice);
            }

            CHECK_CUDA(cudaMemcpy(d_xR, d_tmp, N * sizeof(T), cudaMemcpyDeviceToDevice));

            // then update x = x_0 + xR
            a = 1.0;
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_xR, 1, d_x, 1));

            if (converged) break;
        }  // end of outer iterations

        // check final residual
        // --------------------

        // compute resid = rhs - A * x again
        CHECK_CUDA(cudaMemcpy(d_resid, rhs_in.getPtr(), N * sizeof(T), cudaMemcpyDeviceToDevice));
        op->mat_vec(d_x_vec, d_tmp_vec);
        a = -1.0;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_x_vec.getPtr(), 1, d_resid, 1));

        T final_resid;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &final_resid));

        if (options.print) {
            if (converged) {
                printf("GMRES converged to %.4e resid in %d iterations\n", final_resid, total_iter);
            } else {
                printf("GMRES did NOT CONVERGE with %.4e resid in %d iterations\n", final_resid,
                       total_iter);
            }
        }

        // cudaMemcpy(soln_out.getPtr(), d_x, N * sizeof(T), cudaMemcpyDeviceToDevice);
        CHECK_CUDA(cudaMemcpy(soln_out.getPtr(), d_x, N * sizeof(T), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaDeviceSynchronize());

        return !converged;
    }

    int get_num_iterations() { return total_iter; }

    void free() {
        if (is_free) return;
        is_free = true;  // now it's freed

        if (grid) grid->free();
        d_resid_vec.free();
        if (d_x) cudaFree(d_x);
        if (d_tmp) cudaFree(d_tmp);
        if (d_w) cudaFree(d_w);
        if (d_z) cudaFree(d_z);
        d_z_vec.free();
    }

    GRID *grid;
    Operator *op;
    BaseSolver *pc;
    SolverOptions options;
    int ilevel;

   private:
    // main matrix and linear system data
    BsrMat<DeviceVec<T>> mat;
    // DeviceVec<T> soln, rhs;
    int N, mb, nb, nnzb, block_dim;
    int *d_rowp, *d_cols, *iperm;
    DeviceVec<T> d_x_vec;
    T *d_vals;
    T *d_x, *d_resid;
    DeviceVec<T> d_resid_vec;
    int total_iter = 0;

    bool is_free = false;

    void *pBuffer = 0;
    // level scheduling makes it more parallel
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                                policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    // const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_USE_LEVEL,

    // tried changing both policy L and U to be USE_LEVEL not really a change
    // policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL,
    // policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE,
                              trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;

    // cusparse and cublas handles
    cusparseHandle_t &cusparseHandle;
    cublasHandle_t &cublasHandle;

    // description of K matrix
    cusparseMatDescr_t descrK;

    // temp vecs for GMRES algorithm
    DeviceVec<T> d_w_vec;
    DeviceVec<T> d_z_vec, d_tmp_vec, d_tmp2_vec, d_xR_vec;
    T *d_tmp, *d_tmp2, *d_w, *d_z, *d_xR;
    T *d_Hred, *d_gred;
    T *h_y;
    T *d_Vmat;
    T *Hred;

    cusparseMatDescr_t descr_L = 0, descr_U = 0;
    bsrsv2Info_t info_L = 0, info_U = 0;

    // GMRES temp data
    T g[N_SUBSPACE + 1], cs[N_SUBSPACE], ss[N_SUBSPACE];
    T H[(N_SUBSPACE + 1) * (N_SUBSPACE)];
    int MAX_ITER;
};