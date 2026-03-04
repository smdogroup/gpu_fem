#pragma once

#ifdef USE_GPU
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif
#include "../solve_utils.h"

template <typename T, class GRID>
class PCGSolver : public BaseSolver {
   public:
    PCGSolver(cublasHandle_t &cublasHandle_, cusparseHandle_t &cusparseHandle_, GRID *grid_,
              BaseSolver *pc_, SolverOptions options, int ilevel_ = -1)
        : grid(grid_),
          pc(pc_),
          options(options),
          ilevel(ilevel_),
          cublasHandle(cublasHandle_),
          cusparseHandle(cusparseHandle_) {
        // get matrix and init other temp data for PCG solve
        mat = grid->Kmat;
        // soln = grid->d_soln;
        // rhs = grid->d_rhs;

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

        N = grid->N;
        d_rhs = DeviceVec<T>(N).getPtr();
        d_x = DeviceVec<T>(N).getPtr();  // needs to be separate vec than soln in grid

        // printf("PCG Krylov solver made with options ncycl %d and print %d, with problem size
        // %d\n", options.ncycles, options.print, N);

        // description of the K matrix
        descrK = 0;
        CHECK_CUSPARSE(cusparseCreateMatDescr(&descrK));
        CHECK_CUSPARSE(cusparseSetMatType(descrK, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descrK, CUSPARSE_INDEX_BASE_ZERO));

        // make temp vecs
        d_resid_vec = DeviceVec<T>(N);
        d_resid = d_resid_vec.getPtr();
        d_p = DeviceVec<T>(N).getPtr();
        d_w = DeviceVec<T>(N).getPtr();
        d_z_vec = DeviceVec<T>(N);
        d_z = d_z_vec.getPtr();
    }

    // nothing
    // void update_after_assembly(DeviceVec<T> &vars) {}
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
    void set_cycle_type(std::string cycle_) { pc->set_cycle_type(cycle_); }

    T getResidualNorm(DeviceVec<T> rhs_in, DeviceVec<T> soln_in) {
        // compute r_0 = b - Ax
        CHECK_CUDA(cudaMemcpy(d_resid, rhs_in.getPtr(), N * sizeof(T), cudaMemcpyDeviceToDevice));
        T a = -1.0, b = 1.0;
        CHECK_CUSPARSE(cusparseDbsrmv(
            cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb,
            &a, descrK, d_vals, d_rowp, d_cols, block_dim, soln_in.getPtr(), &b, d_resid));

        T resid_norm;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &resid_norm));
        return resid_norm;
    }

    bool solve(DeviceVec<T> rhs_in, DeviceVec<T> soln_out, bool check_conv = false) {
        // assumes rhs_in and soln_out are in permutation for solve (not natural order)
        // performs full K-cycle with left-precond flexible PCG (note this shows true resid even
        // though it is left precond, unlike GMRES)!

        // copy rhs from method into internal rhs and set soln to zero cause this is like a defect
        // solve
        cudaMemcpy(d_rhs, rhs_in.getPtr(), N * sizeof(T), cudaMemcpyDeviceToDevice);
        cudaMemset(d_x, 0.0, N * sizeof(T));  // re-zero the solution

        // compute r_0 = b - Ax
        CHECK_CUDA(cudaMemcpy(d_resid, d_rhs, N * sizeof(T), cudaMemcpyDeviceToDevice));
        // T a = -1.0, b = 1.0;
        // CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
        //                               CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a, descrK,
        //                               d_vals, d_rowp, d_cols, block_dim, d_x, &b, d_resid));

        n_steps = 0;

        // compute |r_0|
        T init_resid_norm;
        if (check_conv) {
            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &init_resid_norm));
            if (options.print && ilevel == 0) printf("L0-PCG init_resid = %.8e\n", init_resid_norm);
            if (options.print && ilevel != 0)
                printf("\tL%d-PCG init_resid %.2e\n", ilevel, init_resid_norm);
        }

        T rho_prev, rho;  // coefficients that we need to remember
        bool converged = false;

        // inner loop
        for (int j = 0; j < options.ncycles; j++) {
            /* inner 1) solve Mz = r for z (precond) */
            // ----------------------------------------
            pc->solve(d_resid_vec, d_z_vec);

            n_steps = j + 1;  // record the num steps

            /* 2) compute dot products, and p vec */
            // -------------------------------------

            // if fletcher-reeves method
            CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_resid, 1, d_z, 1, &rho));

            if (j == 0) {
                // first iteration, p = z
                cudaMemcpy(d_p, d_z, N * sizeof(T), cudaMemcpyDeviceToDevice);
            } else {
                // compute beta
                T beta = rho / rho_prev;

                // p_new = z + beta * p in two steps
                a = beta;  // p *= beta scalar
                CHECK_CUBLAS(cublasDscal(cublasHandle, N, &a, d_p, 1));
                a = 1.0;  // p += z
                CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_z, 1, d_p, 1));
            }

            // store rho for next iteration (prev), only used in this part
            rho_prev = rho;

            /* 3) compute w = A * p mat-vec product */
            // ----------------------------------------

            // w = A * p
            a = 1.0, b = 0.0;
            CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a,
                                          descrK, d_vals, d_rowp, d_cols, block_dim, d_p, &b, d_w));

            /* 4) update x and r using dot products */
            // ---------------------------------------

            // alpha = <r,z> / <w,p> = rho / <w,p>
            T wp0;
            CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_w, 1, d_p, 1, &wp0));
            T alpha = rho / wp0;

            // x += alpha * p
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &alpha, d_p, 1, d_x, 1));

            // r -= alpha * w
            a = -alpha;
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_w, 1, d_resid, 1));

            // // copy z into zprev (for polak-riberre formula)
            // cudaMemcpy(d_zprev, d_z, N * sizeof(T), cudaMemcpyDeviceToDevice);

            // check for convergence
            if (check_conv) {
                T resid_norm;
                CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &resid_norm));

                if (j % options.print_freq == 0 && options.print) {
                    if (ilevel == 0) printf("L0-PCG [%d] = %.8e\n", j, resid_norm);
                    if (ilevel != 0) printf("\tL%d-PCG [%d] = %.8e\n", ilevel, j, resid_norm);
                }

                if (abs(resid_norm) < (options.atol + init_resid_norm * options.rtol)) {
                    converged = true;
                    if (options.print) {
                        printf("\nL0-PCG converged in %d iterations to %.9e resid\n", j + 1,
                               resid_norm);
                    }
                    break;
                }
            }
        }

        // debug check
        if (options.debug) {
            CHECK_CUDA(cudaMemcpy(d_resid, d_rhs, N * sizeof(T), cudaMemcpyDeviceToDevice));
            T resid_norm1;
            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &resid_norm1));
            a = -1.0, b = 1.0;
            CHECK_CUSPARSE(cusparseDbsrmv(
                cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb,
                nnzb, &a, descrK, d_vals, d_rowp, d_cols, block_dim, d_x, &b, d_resid));

            T resid_norm;
            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &resid_norm));

            T x_nrm;
            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_x, 1, &x_nrm));
            printf("debug: L%d resid_nrm1 %.8e and v2 %.8e, with d_x norm %.2e\n", ilevel,
                   resid_norm1, resid_norm, x_nrm);
        }

        // copy internal soln to external solution of the solve method
        cudaMemcpy(soln_out.getPtr(), d_x, N * sizeof(T), cudaMemcpyDeviceToDevice);

        // DEBUG
        // printf("\t\t#solve-iters %d, fail %d\n", n_steps, !converged);

        return !converged;
    }

    int get_num_iterations() { return n_steps; }

    void free() {
        if (is_free) return;
        is_free = true;  // now it's freed

        if (grid) grid->free();
        d_resid_vec.free();
        if (d_x) cudaFree(d_x);
        if (d_rhs) cudaFree(d_rhs);
        if (d_p) cudaFree(d_p);
        if (d_w) cudaFree(d_w);
        if (d_z) cudaFree(d_z);
        d_z_vec.free();
    }

    GRID *grid;
    BaseSolver *pc;
    SolverOptions options;
    int ilevel;

   private:
    // main matrix and linear system data
    BsrMat<DeviceVec<T>> mat;
    // DeviceVec<T> soln, rhs;
    int N, mb, nb, nnzb, block_dim;
    int *d_rowp, *d_cols, *iperm;
    T *d_vals;
    T *d_rhs, *d_x, *d_resid;
    DeviceVec<T> d_resid_vec;
    int n_steps = 0;

    bool is_free = false;

    // cusparse and cublas handles
    cusparseHandle_t &cusparseHandle;
    cublasHandle_t &cublasHandle;

    // description of K matrix
    cusparseMatDescr_t descrK;

    // temp vecs for PCG algorithm
    DeviceVec<T> d_z_vec;
    T *d_p, *d_w, *d_z;

    // temp scalars for PCG
    T rho, rho_prev, a, alpha, b;
};