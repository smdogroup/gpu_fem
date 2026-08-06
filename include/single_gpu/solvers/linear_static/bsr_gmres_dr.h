#pragma once

/* deflated GMRES linear solver for BSR matrices in CuSparse on the GPU, Sean Engelstad */

#include <assert.h>
#include <cblas.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <lapacke.h>

#include <iostream>

#include "_cusparse_utils.h"
#include "_utils.h"
#include "chrono"
#include "cublas_v2.h"
#include "cuda_utils.h"

/*
References the following papers:
// 1) GMRES deflated restart from "GMRES with adaptively deflated restarting and its performance on
an
//        electromagnetic cavity problem" at
https://www.sciencedirect.com/science/article/pii/S0168927411000705
// 2) thick restarting from "Thick-Restart Lanczos Method for Large
//        Symmetric Eigenvalue problems" at
https://epubs.siam.org/doi/epdf/10.1137/S0895479898334605

For deflated GMRES, you first perform m iterations of standard GMRES with:

    A * V_m = V_{m+1} * H_{m+1,m}         with A (NxN), V_m (Nxm), V_{m+1} (Nxm+1), H_{m+1,m}

(m+1xm) Then solve the update x = x_0 + V_m * y, with y an mx1 vector with: H_{m+1,m} * y = beta *
e1 Here beta = |r_0| the init residual and this can be shown to be equivalent to min || b - Ax || in
the Krylov subspace, as Vm * (beta * e1) = r0 term. The system is solved using Givens rotations to
make H_{m+1,m} upper triangular, with the right hand side becoming a g vector after Givens
rotations. H_{m+1,m} * y = g

After the standard GMRES solve of m iterations, the deflated GMRES strategy seeks to keep the lowest
eigenvalue modes in A * x = lambda * x of A, which will help prevent stalling in the next restart.
The eigenvalue problem can be solved in the Krylov subspace basis as follows:

   (H_{m+1,m}^T H_{m+1,m}) Z = Lambda * H_{m,m}^T Z

Where we take only the first Z_k deflated reduced eigenvectors.
The full eigenvectors are then Phi_k = V_m * Z_k. Common choices for the subspace size m and
deflation size k are (m,k) = (30,10). When we do the restart, the Arnoldi matrix is updated with a
k+1 x k+1 nonzero block from the deflation eigenvectors. Namely, this part is:

    A * V_k' = V_{k+1}' * H_{k+1,k}'

with the primed or new bases the eigenvectors with:

    V_k' = Phi_k = V_m * Z_k and V_{k+1}'= [V_k', v_{m+1}]

    H_{k+1,k}' = [Z_k^T H_{m,m} Z_k; h_{m+1,m} e_m^T Z_k]

Then, the new Arnoldi system continues filling in the k+2 to m+1 H_{m+1,m}'
matrix and adding new search directions to V_m' starting from k+2 entry. Namely, the new full system
is also, A * V_m' = V_{m+1}' * H_{m+1,m}' The update on x = x_0 + V_m' * y minimizes the residual
||b - Ax|| = || r_0 - A * dx|| like before except now the k eigenvectors may be in the same
direction as r_0 (as they are l.c. of previous search directions). The min norm update is then given
by || V_{m+1}' * V_{m+1}'^T * (r0 - A * dx) || like before, with now
    || r0 - A * V_m' * y || = || V_{m+1}' * (V_{m+1}'^T r0 - V_{m+1}'^T A * V_m' y) || = || c -
H_{m+1,m} y || So the reduced basis update y is given by:

    H_{m+1,m}' * y = c,    with c = V_{m+1}'^T r_0 = [0_{k-1}; <w, r0>, 0_{m-k}]

And Givens rotations can still be applied to H_{m+1,m}' and c to give an upper triangular system:
H_{m+1,m}'' * y = g' With final update x = x_0 + V_m' * y and then repeat. The deflated restart is
supposed to be especially good at reducing solve time by less orthogonalization work for especially
stiff linear systems. Since last m+1,m entry is zero after final Givens rotation, becomes an mxm
triangular system.
*/

namespace CUSPARSE {

template <typename T, bool use_precond = true>
void GMRES_DR_solve(BsrMat<DeviceVec<T>> &mat, DeviceVec<T> &rhs, DeviceVec<T> &soln,
                    int subspace_size = 30, int deflation_size = 10, int max_iter = 300,
                    T abs_tol = 1e-8, T rel_tol = 1e-8, bool can_print = false,
                    bool debug = false) {
    /* deflated GMRES solver with right precond and MGS only */

    static_assert(std::is_same<T, double>::value,
                  "Only double precision is written in our code for cuSparse Deflated GMRES");

    auto rhs_perm = inv_permute_rhs<BsrMat<DeviceVec<T>>, DeviceVec<T>>(mat, rhs);

    if (can_print) {
        printf("Deflated GMRES ILU solve\n");
    }
    auto start = std::chrono::high_resolution_clock::now();

    // copy important inputs for Bsr structure out of BsrMat
    BsrData bsr_data = mat.getBsrData();
    int mb = bsr_data.nnodes;
    int nnzb = bsr_data.nnzb;
    int block_dim = bsr_data.block_dim;
    index_t *d_rowp = bsr_data.rowp;
    index_t *d_cols = bsr_data.cols;
    int *iperm = bsr_data.iperm;
    int N = soln.getSize();
    T *d_rhs = rhs_perm.getPtr();
    T *d_x = soln.getPtr();

    // permute data in soln if guess is not zero
    soln.permuteData(block_dim, iperm);

    // note this changes the mat data to be LU (but that's the whole point
    // of LU solve is for repeated linear solves we now just do triangular
    // solves)
    T *d_vals = mat.getPtr();

    // also make a temporary array for the preconditioner values
    T *d_vals_ILU0 = DeviceVec<T>(mat.get_nnz()).getPtr();
    // ILU0 equiv to ILU(k) if sparsity pattern has ILU(k)
    CHECK_CUDA(
        cudaMemcpy(d_vals_ILU0, d_vals, mat.get_nnz() * sizeof(T), cudaMemcpyDeviceToDevice));

    // make temp vecs
    T *d_tmp = DeviceVec<T>(soln.getSize()).getPtr();
    T *d_tmp2 = DeviceVec<T>(soln.getSize()).getPtr();
    T *d_resid = DeviceVec<T>(soln.getSize()).getPtr();
    T *d_w = DeviceVec<T>(soln.getSize()).getPtr();

    // create initial cusparse and cublas handles --------------

    /* Create CUBLAS context */
    cublasHandle_t cublasHandle = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    /* Create CUSPARSE context */
    cusparseHandle_t cusparseHandle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

    // create the matrix BSR object
    // -----------------------------

    /* Description of the A matrix */
    cusparseMatDescr_t descrA = 0;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
    CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

    // create ILU(0) preconditioner
    // -----------------------------
    // [equiv to ILU(k) precondioner if ILU(k) sparsity pattern used in BsrData object]

    // init objects for LU factorization and LU solve
    cusparseMatDescr_t descr_L = 0, descr_U = 0;
    bsrsv2Info_t info_L = 0, info_U = 0;
    void *pBuffer = 0;
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE,
                              trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;

    T a = 1.0, b = 0.0;
    // perform the symbolic and numeric factorization of LU on given sparsity pattern
    CUSPARSE::perform_ilu0_factorization(cusparseHandle, descr_L, descr_U, info_L, info_U, &pBuffer,
                                         mb, nnzb, block_dim, d_vals_ILU0, d_rowp, d_cols, trans_L,
                                         trans_U, policy_L, policy_U, dir);

    // main GMRES solve
    // ----------------
    int m = subspace_size, k = deflation_size;

    // Hessenberg linear system data (solved on host because so small and can be done separately on
    // host without any copying needed)
    T *H = new T[(m + 1) * m];  // hessenberg matrix
    T *g = new T[m + 1];        // hessenberg RHS
    T *y = new T[m];            // solution of Hessenberg system for update x = x0 + Vm * y

    memset(H, 0.0, (m + 1) * m * sizeof(T));

    // temp data for the deflation eigenvalue problem
    T *Htmp = new T[m * m];   // temp hessenberg for products
    T *Htmp2 = new T[m * m];  // temp hessenberg for products
    T *wr = new T[m], *wi = new T[m],
      *vr = new T[m * m];                 // eigenvalues + eigenvecs in reduced space
    T *Z = DeviceVec<T>(k * m).getPtr();  // device vec of reduced eigenvectors
    T beta, init_resid;

    // Krylov subspace and full space eigvecs
    T *d_V = DeviceVec<T>((m + 1) * N).getPtr();  // Kryov search directions
    T *d_Phi = DeviceVec<T>(k * N).getPtr();

    bool converged = false;
    int total_iter = 0;
    int nrestarts = max_iter / subspace_size;
    if (debug) printf("nrestarts = %d\n", nrestarts);

    for (int irestart = 0; irestart < nrestarts; irestart++) {
        // reset RHS to zero
        memset(g, 0.0, (m + 1) * sizeof(T));

        if (irestart == 0) {
            /* compute initial residual */
            // compute r0 = b - A * x
            CHECK_CUDA(cudaMemcpy(d_resid, d_rhs, N * sizeof(T), cudaMemcpyDeviceToDevice));
            a = 1.0, b = 0.0;
            CHECK_CUSPARSE(cusparseDbsrmv(
                cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb,
                nnzb, &a, descrA, d_vals, d_rowp, d_cols, block_dim, d_x, &b, d_tmp));
            a = -1.0;
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_tmp, 1, d_resid, 1));

            /* compute first w = r0 / |r0| */
            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &beta));
            CHECK_CUDA(cudaMemcpy(d_w, d_resid, N * sizeof(T), cudaMemcpyDeviceToDevice));
            T inrm_w = 1.0 / beta;
            CHECK_CUBLAS(cublasDscal(cublasHandle, N, &inrm_w, d_w, 1));
            // put w as v_0
            CHECK_CUDA(cudaMemcpy(d_V, d_w, N * sizeof(T), cudaMemcpyDeviceToDevice));

            printf("init resid %.8e\n", beta);

            // RHS for first iteration is just g[0]
            g[0] = beta;
            init_resid = beta;  // for convergence checks with rel_tol
        } else {
            // g[k] = <w, r0>, rest of g was zeroed out, this comes from g = V^T r0 and dense part
            // is zero, past k is zero bc no V there yet and will be orthog to it
            CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_w, 1, d_resid, 1, &beta));
            g[k] = beta;
        }

        int start = irestart == 0 ? 0 : k;

        // then begin main GMRES iteration loop!
        for (int j = start; j < m; j++, total_iter++) {
            // compute w = A * v
            // -----------------
            CHECK_CUDA(cudaMemcpy(d_w, &d_V[j * N], N * sizeof(T), cudaMemcpyDeviceToDevice));

            if constexpr (use_precond) {
                // first compute U^-1 L^-1 vj => tmp
                a = 1.0;
                CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                    cusparseHandle, dir, trans_L, mb, nnzb, &a, descr_L, d_vals_ILU0, d_rowp,
                    d_cols, block_dim, info_L, d_w, d_tmp, policy_L, pBuffer));
                CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                    cusparseHandle, dir, trans_U, mb, nnzb, &a, descr_U, d_vals_ILU0, d_rowp,
                    d_cols, block_dim, info_U, d_tmp, d_w, policy_U, pBuffer));
            }

            // then compute A * tmp => w, so in full w = A * M^-1 * vj
            a = 1.0, b = 0.0;
            CHECK_CUSPARSE(cusparseDbsrmv(
                cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb,
                nnzb, &a, descrA, d_vals, d_rowp, d_cols, block_dim, d_w, &b, d_tmp));
            CHECK_CUDA(cudaMemcpy(d_w, d_tmp, N * sizeof(T), cudaMemcpyDeviceToDevice));

            /* modified GS orthogonalization of w against prev vi */
            for (int i = 0; i < j + 1; i++) {
                T *vi = &d_V[i * N];
                // compute H_{i,j} = <vi, w>
                CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_w, 1, vi, 1, &H[m * i + j]));
                // if (debug) printf("H[%d,%d] = %.9e\n", i, j, H[m * i + j]);

                // w -= H_{i,j} * vi
                a = -H[m * i + j];
                CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, vi, 1, d_w, 1));
            }

            // compute H_{j+1,j} = || w ||
            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_w, 1, &H[m * (j + 1) + j]));

            // normalize w
            T nrm_w = H[m * (j + 1) + j];
            T inrm_w = 1.0 / nrm_w;
            CHECK_CUBLAS(cublasDscal(cublasHandle, N, &inrm_w, d_w, 1));

            // copy w into v_{j+1}
            CHECK_CUDA(cudaMemcpy(&d_V[(j + 1) * N], d_w, N * sizeof(T), cudaMemcpyDeviceToDevice));
        }  // end of inner loop for Krylov subspace

        // copy H into Htmp (since it LU factors in place and will mess up later computations)
        memcpy(Htmp, H, m * m * sizeof(T));

        // see cuda_examples/gmres/_gmres_util.py : gmres_dr() method
        // for some reason solving H[m x m] * y = g[m x 1] works better than least-squares.. revisit
        // later because supposed to be a least-squares solve of m+1 x m dimension
        int *ipiv = (int *)malloc(m * sizeof(int));
        int info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, m, 1, Htmp, m, ipiv, g, 1);
        // overwrites g to solution y in place
        memcpy(y, g, m * sizeof(T));

        // compute update x = x0 + V * y
        // -----------------------------

        CHECK_CUDA(cudaMemset(d_tmp, 0.0, N * sizeof(T)));
        for (int j = 0; j < m; j++) {
            a = y[j];
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, &d_V[j * N], 1, d_tmp, 1));
        }

        if constexpr (use_precond) {
            // now compute matrix product xR = V * y (the preconditioned solution first)

            // compute U^-1 L^-1 * xR => x
            a = 1.0;
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_L, mb, nnzb, &a,
                                                 descr_L, d_vals_ILU0, d_rowp, d_cols, block_dim,
                                                 info_L, d_tmp, d_tmp2, policy_L, pBuffer));
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, mb, nnzb, &a,
                                                 descr_U, d_vals_ILU0, d_rowp, d_cols, block_dim,
                                                 info_U, d_tmp2, d_tmp, policy_U, pBuffer));
        }
        a = 1.0;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_tmp, 1, d_x, 1));

        /* check residual and convergence here, exit if done */
        // compute r0 = b - A * x
        CHECK_CUDA(cudaMemcpy(d_resid, d_rhs, N * sizeof(T), cudaMemcpyDeviceToDevice));
        a = 1.0, b = 0.0;
        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a, descrA,
                                      d_vals, d_rowp, d_cols, block_dim, d_x, &b, d_tmp));
        a = -1.0;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_tmp, 1, d_resid, 1));
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &beta));
        printf("GMRES-DR [%d]: resid %.8e\n", total_iter, beta);
        if (beta < (abs_tol + init_resid * rel_tol)) {
            printf("GMRES-DR converged in %d iters to %.8e resid\n", total_iter, beta);
            break;
        }

        // compute Ritz eigenvectors for deflated GMRES
        // --------------------------------------------

        // solve H * z = lambda * z regular eigenvalue problem
        // more numerically stable than least-squares eig value Hbar^T * Hbar * z = lam * H^T * z
        int info2 = LAPACKE_dgeev(
            LAPACK_ROW_MAJOR, 'N', 'V', m, H, m, wr, wi, nullptr, m, vr, m  // right eigenvectors
        );

        /* load vr into Z_k host to device (col-major) */
        CHECK_CUDA(cudaMemcpy(Z, vr, k * m * sizeof(T), cudaMemcpyHostToDevice));

        // now compute Phi = V_{m} * Z_k where Z_k is an mxk matrix and then Phi is {Nxk}
        // here cublasDgemv takes in mat that is stored in column-major format (which this is)
        a = 1.0, b = 0.0;  // on device here
        CHECK_CUBLAS(cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, N, m, k, &a, d_V, N, Z, m,
                                 &b, d_Phi, N));

        // printf("here3\n");
        // printf("VR:");
        // printVec<T>(m * k, VR);

        /* compute new Krylov basis */
        // V[:,:k] = Phik [k x N size]
        CHECK_CUDA(cudaMemcpy(d_V, d_Phi, k * N * sizeof(T), cudaMemcpyDeviceToDevice));
        // V[:,k] = V[:,m]
        CHECK_CUDA(cudaMemcpy(&d_V[k * N], &d_V[m * N], N * sizeof(T), cudaMemcpyDeviceToDevice));
        // zero out V[:,k+1:]
        CHECK_CUDA(cudaMemset(&d_V[(k + 1) * N], 0.0, (m - k) * N * sizeof(T)));

        /* compute new Hessenberg matrix */

        // first copy H into Htmp
        T beta2 = H[m * m + m - 1];  // beta = H[m,m-1] last entry
        // compute Z_k^T H_{m,m} Z_k for restarted Hessenberg (last k+1 row is zero)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, k, m, m, 1.0, vr, m, H, m, 0.0, Htmp,
                    m);  // flip transposes on vr aka Z_k because it is col major
        // then Htmp * Zk again respecting row and col major
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, k, k, m, 1.0, Htmp, m, vr, m, 0.0,
                    Htmp2, k);

        // zero out H matrix
        memset(H, 0.0, (m + 1) * m * sizeof(T));

        // copy Htmp2 as Hk => H[:k,:k], since Htmp2 stored as kxk above in buffer
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                H[m * i + j] = Htmp2[k * i + j];
            }
        }

        // finally copy out beta * Zk[m,:] into H[k,:k] part, with vr equiv to Zk on host
        for (int j = 0; j < k; j++) {
            // recall vr is col-major and this goes over columns
            H[m * k + j] = beta2 * vr[k * m + j];
        }
    }  // end of outer restart loop

    // report final #iters and resid
    if (can_print) printf("GMRES-DR converged to %.4e resid in %d iterations\n", beta, total_iter);

    // cleanup and inverse permute the solution for exit
    // -------------------------------------------------

    // now also inverse permute the soln data
    permute_soln<BsrMat<DeviceVec<T>>, DeviceVec<T>>(mat, soln);

    // free resources
    cudaFree(pBuffer);
    cusparseDestroyMatDescr(descr_L);
    cusparseDestroyMatDescr(descr_U);
    cusparseDestroyBsrsv2Info(info_L);
    cusparseDestroyBsrsv2Info(info_U);
    cusparseDestroy(cusparseHandle);

    // free resources on host
    free(H);
    free(g);
    free(Htmp);
    free(Htmp2);
    free(y);

    // TODO : still missing a few free / delete[] statements

    CHECK_CUDA(cudaFree(d_resid));
    CHECK_CUDA(cudaFree(d_w));
    CHECK_CUDA(cudaFree(d_tmp));
    CHECK_CUDA(cudaFree(Z));

    // print timing data
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    double dt = duration.count() / 1e6;
    if (can_print) {
        printf("\tfinished in %.4e sec\n", dt);
    }
}

};  // namespace CUSPARSE
