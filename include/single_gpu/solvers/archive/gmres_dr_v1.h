#pragma once

/* deflated GMRES linear solver for BSR matrices in CuSparse on the GPU, Sean Engelstad */

#include <assert.h>
#include <cblas.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <lapacke.h>

#include <iostream>

#include "../../cuda_utils.h"
#include "_cusparse_utils.h"
#include "_utils.h"
#include "chrono"
#include "cublas_v2.h"

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

    H_{m+1,m}' * y = c,    with c = V_{m+1}'^T r_0 = [V_{k}'^T r_0; 0, 0_{m-k}]

And Givens rotations can still be applied to H_{m+1,m}' and c to give an upper triangular system:
H_{m+1,m}'' * y = g' With final update x = x_0 + V_m' * y and then repeat. The deflated restart is
supposed to be especially good at reducing solve time by less orthogonalization work for especially
stiff linear systems. Since last m+1,m entry is zero after final Givens rotation, becomes an mxm
triangular system.
*/

namespace CUSPARSE {

template <typename T, bool right = false, bool modifiedGS = true, bool use_precond = true>
void GMRES_DR_solve(BsrMat<DeviceVec<T>> &mat, DeviceVec<T> &rhs, DeviceVec<T> &soln,
                    int subspace_size = 30, int deflation_size = 10, int max_iter = 300,
                    T abs_tol = 1e-8, T rel_tol = 1e-8, bool can_print = false, bool debug = false,
                    int print_freq = 10) {
    static_assert(std::is_same<T, double>::value,
                  "Only double precision is written in our code for cuSparse Deflated GMRES");

    auto rhs_perm = inv_permute_rhs<BsrMat<DeviceVec<T>>, DeviceVec<T>>(mat, rhs);

    // which type of preconditioners
    constexpr bool left_precond = use_precond && !right;
    constexpr bool right_precond = use_precond && right;

    // if (can_print) {
    //     printf("begin cusparse GMRES solve\n");
    // }
    auto start = std::chrono::high_resolution_clock::now();

    // copy important inputs for Bsr structure out of BsrMat
    // TODO : was trying to make some of these const but didn't accept it in
    // final solve
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
    T *d_xR = DeviceVec<T>(soln.getSize()).getPtr();  // for right preconditioning

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
    // tried changing both policy L and U to be USE_LEVEL not really a change
    // policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL,
    // policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
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
    T *g = new T[m + 1];        // hessenberg RHS (Givens rotated)
    T *cs = new T[m];           // Givens cosines
    T *ss = new T[m];           // Givens sines
    T *y = new T[m];            // solution of Hessenberg system for update x = x0 + Vm * y
    // final Hessenberg system will become mxm (but need extra storage for final Givens application
    // on m+1 row) H matrix stored with typical row-major format so that mxm matrix is just first
    // m^2 values

    memset(H, 0.0, (m + 1) * m * sizeof(T));

    // temp data for the deflation eigenvalue problem
    T *HTH = new T[m * m];  // temporary matrix to hold H^T H matrix of eig problem
    T *HT = new T[m * m];   // temp matrix for H^T in eig problem
    T *alphaR = new T[m], *alphaI = new T[m], *betav = new T[m], *VL = new T[m * m],
      *VR = new T[m * m];
    T *Z = DeviceVec<T>(k * m).getPtr();  // device vec of reduced eigenvectors

    // dense Krylov subspace vectors stored on device however
    T *d_V = DeviceVec<T>((m + 1) * N).getPtr();  // Kryov search directions
    // temporary storage for deflation eigenvecs (quickly stored back in d_V)
    T *d_Phi = DeviceVec<T>(k * N).getPtr();

    bool converged = false;
    int total_iter = 0;
    int nrestarts = max_iter / subspace_size;
    if (debug) printf("nrestarts = %d\n", nrestarts);

    for (int irestart = 0; irestart < nrestarts; irestart++) {
        // compute r0 = b - A * x
        CHECK_CUDA(cudaMemcpy(d_resid, d_rhs, N * sizeof(T), cudaMemcpyDeviceToDevice));

        int mm = m;

        // compute xR0 = L * U * x0 for preconditioned initial guess (if right precond)
        if constexpr (right_precond) {
            a = 1.0, b = 0.0;
            CHECK_CUSPARSE(cusparseDbsrmv(
                cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb,
                nnzb, &a, descr_U, d_vals_ILU0, d_rowp, d_cols, block_dim, d_x, &b, d_tmp));
            a = 1.0, b = 0.0;
            CHECK_CUSPARSE(cusparseDbsrmv(
                cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb,
                nnzb, &a, descr_L, d_vals_ILU0, d_rowp, d_cols, block_dim, d_tmp, &b, d_xR));
        }

        // then subtract Ax from rhs
        a = 1.0, b = 0.0;
        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a, descrA,
                                      d_vals, d_rowp, d_cols, block_dim, d_x, &b, d_tmp));
        a = -1.0;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_tmp, 1, d_resid, 1));

        // get initial residual beta = || r_0 ||, preconditioned if left precond
        T beta;
        if constexpr (left_precond) {
            // if left precond, compute M^-1 r0 = U^-1 L^-1 r0
            a = 1.0;
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_L, mb, nnzb, &a,
                                                 descr_L, d_vals_ILU0, d_rowp, d_cols, block_dim,
                                                 info_L, d_resid, d_tmp, policy_L, pBuffer));
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, mb, nnzb, &a,
                                                 descr_U, d_vals_ILU0, d_rowp, d_cols, block_dim,
                                                 info_U, d_tmp, d_resid, policy_U, pBuffer));
        }
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &beta));

        // reset Hessenberg RHS g vec
        memset(g, 0.0, (m + 1) * sizeof(T));

        // zero out V past k for restart cases (for the r0 and m-k new vecs)
        if (irestart > 0) CHECK_CUDA(cudaMemset(&d_V[k * N], 0.0, (m - k + 1) * N * sizeof(T)));

        // report initial residual
        if (can_print) printf("GMRES init resid = %.9e\n", beta);

        // standard vs deflated restart section
        if (irestart == 0) {
            // standard restart, only for first startup
            // set v0 = r0 / beta and initial g[0] = beta for RHS of Hessenberg system
            a = 1.0 / beta;
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_resid, 1, &d_V[0], 1));

            // set initial residual in Hessenberg RHS
            g[0] = beta;
        } else {
            // deflated restart
            // compute k many initial g vectors of RHS since previous eigenvectors may be in same
            // direction as r0. Namely g0 = V_{m+1}'^T r0, or first k values g[0:k] = V_k'^T * r0
            // where V_k' are the eigenvectors of previous subspace
            for (int i = 0; i < k; i++) {
                // store in g[i] = <v_i, r0>
                CHECK_CUBLAS(cublasDdot(cublasHandle, N, &d_V[i * N], 1, d_resid, 1, &g[i]));
                printf("restart g[%d] = %.4e\n", i, g[i]);
            }

            // also set v_k (0-based) to r0 / beta and then orthogonalize it
            a = 1.0 / beta;
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_resid, 1, &d_V[k * N], 1));

            // modified GS to orthogonalize r0 against previous search directions (prob unnecessary)
            for (int i = 0; i < k; i++) {
                // compute Htmp = <vi, vk>
                T tmp;
                CHECK_CUBLAS(cublasDdot(cublasHandle, N, &d_V[k * N], 1, &d_V[i * N], 1, &tmp));

                // vk -= H_{i,j} * vi
                a = -tmp;
                CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, &d_V[i * N], 1, &d_V[k * N], 1));
            }

            // now normalize it again vk /= |vk|
            T nrm;
            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &nrm));
            T inrm = 1.0 / nrm;
            CHECK_CUBLAS(cublasDscal(cublasHandle, N, &inrm, &d_V[k * N], 1));

            // also set g_k = v_k^T r_0
            g[k] = beta;
        }

        // start at k (0-based) if restart, less new iterations with restarted cases
        int start = irestart == 0 ? 0 : k;

        // then begin main GMRES iteration loop!
        for (int j = start; j < m; j++, total_iter++) {
            // current subspace size if exit early
            mm = j;

            // compute w = A * v
            // -----------------
            if constexpr (right_precond) {
                // first compute U^-1 L^-1 vj => tmp
                a = 1.0;
                CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                    cusparseHandle, dir, trans_L, mb, nnzb, &a, descr_L, d_vals_ILU0, d_rowp,
                    d_cols, block_dim, info_L, &d_V[j * N], d_tmp, policy_L, pBuffer));
                CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                    cusparseHandle, dir, trans_U, mb, nnzb, &a, descr_U, d_vals_ILU0, d_rowp,
                    d_cols, block_dim, info_U, d_tmp, d_tmp2, policy_U, pBuffer));

                // then compute A * tmp => w, so in full w = A * M^-1 * vj
                a = 1.0, b = 0.0;
                CHECK_CUSPARSE(cusparseDbsrmv(
                    cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, mb,
                    mb, nnzb, &a, descrA, d_vals, d_rowp, d_cols, block_dim, d_tmp2, &b, d_w));
            }

            if constexpr (left_precond) {
                // compute A * vj => w
                a = 1.0, b = 0.0;
                CHECK_CUSPARSE(cusparseDbsrmv(
                    cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, mb,
                    mb, nnzb, &a, descrA, d_vals, d_rowp, d_cols, block_dim, &d_V[j * N], &b, d_w));

                // compute U^-1 * L^-1 * w => w
                a = 1.0;
                CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                    cusparseHandle, dir, trans_L, mb, nnzb, &a, descr_L, d_vals_ILU0, d_rowp,
                    d_cols, block_dim, info_L, d_w, d_tmp, policy_L, pBuffer));
                CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                    cusparseHandle, dir, trans_U, mb, nnzb, &a, descr_U, d_vals_ILU0, d_rowp,
                    d_cols, block_dim, info_U, d_tmp, d_w, policy_U, pBuffer));
            }

            // orthogonalize w against previous search directions (Gram-Schmidt)
            // -----------------------------------------------------------------

            // TODO : could add householder and modified GS with reorthogonalization here too
            if constexpr (modifiedGS) {
                // modified GS is more numerically stable but more flops than classical
                for (int i = 0; i < j + 1; i++) {
                    // compute H_{i,j} = <vi, w>
                    CHECK_CUBLAS(
                        cublasDdot(cublasHandle, N, d_w, 1, &d_V[i * N], 1, &H[m * i + j]));
                    if (debug) printf("H[%d,%d] = %.9e\n", i, j, H[m * i + j]);

                    // w -= H_{i,j} * vi
                    a = -H[m * i + j];
                    CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, &d_V[i * N], 1, d_w, 1));
                }
            } else {
                // TODO : could implement classical GS like in regular GMRES, but then I need
                // H on the device.. prob not worth it (classical GS was slower because orthog less
                // accurate, despite less flops)
                printf("classical GS not implemented in deflated GMRES yet..\n");
            }

            // compute H_{j+1,j} = || w ||
            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_w, 1, &H[m * (j + 1) + j]));

            // compute v_{j+1} = w / || w ||
            a = 1.0 / H[m * (j + 1) + j];  // fine to add in because V mat zeroed out each restart
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_w, 1, &d_V[(j + 1) * N], 1));

            // apply Givens rotations to Hessenberg matrix
            for (int i = 0; i < j; i++) {
                T temp = H[m * i + j];
                H[m * i + j] = cs[i] * H[m * i + j] + ss[i] * H[m * (i + 1) + j];
                H[m * (i + 1) + j] = -ss[i] * temp + cs[i] * H[m * (i + 1) + j];
            }

            // apply Givens rotations to Hessenberg RHS
            T H1 = H[m * j + j], H2 = H[m * (j + 1) + j];
            cs[j] = H1 / sqrt(H1 * H1 + H2 * H2);
            ss[j] = cs[j] * H2 / H1;
            g[j + 1] = -ss[j] * g[j];
            g[j] *= cs[j];

            // compute new entries in the Hessenberg matrix for vj
            H[m * j + j] = cs[j] * H[m * j + j] + ss[j] * H[m * (j + 1) + j];
            H[m * (j + 1) + j] = 0.0;

            // printf("H:");
            // printVec<T>(m * m, H);

            // printf("GMRES iter %d : resid %.9e\n", j, nrm_w);
            if (can_print && (j % print_freq == 0))
                printf("GMRES iter %d : resid %.9e\n", j, abs(g[j + 1]));

            if (debug) printf("j=%d, g[j]=%.9e, g[j+1]=%.9e\n", j, g[j], g[j + 1]);

            if (abs(g[j + 1]) < (abs_tol + beta * rel_tol)) {
                if (can_print)
                    printf("GMRES converged in %d iterations to %.9e resid\n", j + 1, g[j + 1]);
                converged = true;
                break;
            }
        }  // end of inner loop for Krylov subspace

        // now solve Householder triangular system H * y = g of size mm x mm
        //     (where mm might be less than m if exit early)
        for (int i = mm - 1; i >= 0; i--) {
            T num = g[i];
            // subtract previous solved y terms
            for (int j = i + 1; j < mm; j++) {
                num -= H[m * i + j] * y[j];
            }
            y[i] = num / H[m * i + i];
        }

        // printf("y:");
        // printVec<T>(mm, y);

        // compute update x = x0 + V * y
        // -----------------------------

        if constexpr (left_precond) {
            for (int j = 0; j < mm; j++) {
                a = y[j];
                CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, &d_V[j * N], 1, d_x, 1));
            }
        }

        if constexpr (right_precond) {
            // now compute matrix product xR = V * y (the preconditioned solution first)
            for (int j = 0; j < mm; j++) {
                a = y[j];
                CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, &d_V[j * N], 1, d_xR, 1));
            }

            // compute U^-1 L^-1 * xR => x
            a = 1.0;
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_L, mb, nnzb, &a,
                                                 descr_L, d_vals_ILU0, d_rowp, d_cols, block_dim,
                                                 info_L, d_xR, d_tmp, policy_L, pBuffer));
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, mb, nnzb, &a,
                                                 descr_U, d_vals_ILU0, d_rowp, d_cols, block_dim,
                                                 info_U, d_tmp, d_x, policy_U, pBuffer));
        }

        // exits here if out of steps or if converged
        if (converged || !(total_iter < max_iter)) break;

        // compute Ritz eigenvectors for deflated GMRES
        // --------------------------------------------

        // copy H into HT of size mm
        for (int i = 0; i < mm; i++) {
            for (int j = 0; j < mm; j++) {
                HT[mm * i + j] = H[m * j + i];
            }
        }

        // compute H^T H matrix of size m since it didn't converge
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, m, m, 1.0, H, m, H, m, 0.0, HTH, m);

        // solve gen eigenvalue problem (HT * H) * Z = lambda * HT * Z
        // int info;
        int info = LAPACKE_dggev(LAPACK_ROW_MAJOR, 'N', 'V', m, HTH, m, HT, m, alphaR, alphaI,
                                 betav, VL, m, VR, m);
        if (info != 0) {
            printf("Eigenvalue solve failed with info = %d\n", info);
        }

        // printf("VR0:");
        // printVec<T>(m * k, VR);

        // normalize first k Ritz vectors in VR (stored in column major format)
        for (int i = 0; i < k; i++) {
            T inrm = 1.0 / (cblas_dnrm2(m, &VR[i * m], 1) + 1e-14);
            cblas_dscal(m, inrm, &VR[i * m], 1);
        }

        // load VR => Z_k host to device (column major)
        CHECK_CUDA(cudaMemcpy(Z, VR, k * m * sizeof(T), cudaMemcpyHostToDevice));

        // printf("here2\n");

        // now compute Phi = V_{m} * Z_k where Z_k is an mxk matrix and then Phi is {Nxk}
        // here cublasDgemv takes in mat that is stored in column-major format (which this is)
        a = 1.0, b = 0.0;  // on device here
        CHECK_CUBLAS(cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, N, m, k, &a, d_V, N, Z, m,
                                 &b, d_Phi, N));

        // printf("here3\n");
        // printf("VR:");
        // printVec<T>(m * k, VR);

        // then copy Phi into d_V again
        CHECK_CUDA(cudaMemcpy(d_V, d_Phi, k * m * sizeof(T), cudaMemcpyDeviceToDevice));

        // also compute new starting H matrix on host
        // compute Z_k^T H_{m,m} Z_k for restarted Hessenberg (last k+1 row is zero)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, k, m, m, 1.0, VR, m, H, m, 0.0, HTH,
                    m);  // flip transposes on Z because it is col major
        // then HTH_temp * Zk again respecting row and col major
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, k, k, m, 1.0, HTH, m, VR, m, 0.0, HT,
                    k);

        // printf("here4\n");

        // zero out H matrix
        memset(H, 0.0, (m + 1) * m * sizeof(T));

        // copy kxk part stored in HT to full rank m+1 x m matrix in H
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                H[m * i + j] = HT[k * i + j];
            }
        }

        // printf("H[k x k]:");
        // printVec<T>(k * k, HT);

        // the k+1th row in restart h_{m+1,m} * em^T Z_k should be zero
        // the new Hessenberg RHS c = V_{m+1}'^T r_0 is computed in restart (see start of restart
        // code)

    }  // end of outer iterations

    // check final residual
    // --------------------

    // compute final residual r(x) = b - Ax
    CHECK_CUDA(cudaMemcpy(d_resid, d_rhs, N * sizeof(T), cudaMemcpyDeviceToDevice));
    a = 1.0, b = 0.0;
    CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a, descrA,
                                  d_vals, d_rowp, d_cols, block_dim, d_x, &b, d_tmp));
    a = -1.0;
    CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_tmp, 1, d_resid, 1));
    T final_resid;
    CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &final_resid));

    // now apply preconditioner to the residual
    if constexpr (left_precond) {
        // zero vec_tmp
        CHECK_CUDA(cudaMemset(d_tmp, 0.0, N * sizeof(T)));

        // ILU solve U^-1 L^-1 * b
        // L^-1 * b => tmp
        a = 1.0;
        CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_L, mb, nnzb, &a, descr_L,
                                             d_vals_ILU0, d_rowp, d_cols, block_dim, info_L,
                                             d_resid, d_tmp, policy_L, pBuffer));

        // U^-1 * tmp => into rhs
        CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, mb, nnzb, &a, descr_U,
                                             d_vals_ILU0, d_rowp, d_cols, block_dim, info_U, d_tmp,
                                             d_resid, policy_U, pBuffer));
    }

    T final_precond_resid;
    CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &final_precond_resid));

    if (can_print)
        printf("GMRES converged to %.4e resid, %.4e precond resid in %d iterations\n", final_resid,
               final_precond_resid, total_iter);

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
    free(cs);
    free(ss);
    free(HT);
    free(HTH);
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
