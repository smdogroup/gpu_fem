#pragma once

#include <assert.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include <chrono>
#include <iostream>

namespace CUSPARSE {

template <typename T, bool precond = true>
void HGMRES_solve(BsrMat<DeviceVec<T>> &mat, DeviceVec<T> &rhs, DeviceVec<T> &soln, int _m = 100,
                  int max_iter = 500, T abs_tol = 1e-8, T rel_tol = 1e-8, bool can_print = false,
                  bool debug = false, int print_freq = 10, T eta_precond = 0.0) {
    /* householder orthog, right preconditioned GMRES solver, more numerically stable than MGS GMRES
            based off working python implementation in `cuda_examples/gmres/_gmres_util.py :
            gmres_householder() of p. 160 Saad, Sparse Linear Systems
     */

    static_assert(std::is_same<T, double>::value,
                  "Only double precision is written in our code for cuSparse Householder GMRES");

    auto start = std::chrono::high_resolution_clock::now();

    // apply permutations to incoming vecs (rhs and initial soln x0)
    auto rhs_perm = inv_permute_rhs<BsrMat<DeviceVec<T>>, DeviceVec<T>>(mat, rhs);

    printf("here0\n");

    // setup GMRES storage
    // -------------------
    BsrData bsr_data = mat.getBsrData();
    int mb = bsr_data.nnodes;
    int nnzb = bsr_data.nnzb;
    int block_dim = bsr_data.block_dim;
    index_t *d_rowp = bsr_data.rowp;
    index_t *d_cols = bsr_data.cols;
    int *iperm = bsr_data.iperm;
    int N = soln.getSize();
    int m = min(_m, bsr_data.nnodes);  // subspace size
    T *d_rhs = rhs_perm.getPtr();
    T *d_x = soln.getPtr();
    T *d_vals = mat.getPtr();
    T *d_vals_ILU0 = DeviceVec<T>(mat.get_nnz()).getPtr();
    if constexpr (precond) {
        CHECK_CUDA(
            cudaMemcpy(d_vals_ILU0, d_vals, mat.get_nnz() * sizeof(T), cudaMemcpyDeviceToDevice));
    }
    T *d_tmp = DeviceVec<T>(N).getPtr();
    T *d_tmp2 = DeviceVec<T>(N).getPtr();
    T *d_resid = DeviceVec<T>(N).getPtr();
    T *d_h = DeviceVec<T>(N).getPtr();
    T *d_v = DeviceVec<T>(N).getPtr();
    T *d_z = DeviceVec<T>(N).getPtr();
    T one = 1.0;
    T *d_one = HostVec<T>(1, &one).createDeviceVec().getPtr();

    printf("here0.1\n");

    // permute soln with nodal reordering
    soln.permuteData(block_dim, iperm);

    // // just add diagonal to precond only?, temp debug
    // int ndiag = mb * block_dim;
    // dim3 block(32);
    // int nblocks = (ndiag + block.x - 1) / block.x;
    // dim3 grid(nblocks);
    // // T eta = 1e7;
    // // T eta = 1e-3;
    // // add_mat_diag_kernel<T><<<grid, block>>>(mb, block_dim, d_rowp, d_cols, d_vals_ILU0, eta);
    // mult_diag_kernel<T><<<grid, block>>>(mb, block_dim, d_rowp, d_cols, d_vals_ILU0,
    // eta_precond);

    // Hessenberg system and householder vec storage
    T *g = new T[m + 1];
    T *cs = new T[m];
    T *ss = new T[m];
    T *y = new T[m];
    T *HT = new T[(m + 1) * m];
    // zero out Hessenberg matrix and RHS
    memset(HT, 0.0, (m + 1) * m * sizeof(T));
    memset(g, 0.0, (m + 1) * sizeof(T));

    // storing HT this time, so each col of H consecutive now
    T *d_Wmat;  // householder vecs
    CHECK_CUDA(cudaMalloc((void **)&d_Wmat, (m + 1) * N * sizeof(T)));

    printf("here0.2\n");

    // scalars for CuSparse and CuBlas operations
    T a = 1.0, b = 0.0;

    // create initial cusparse and cublas handles
    // ------------------------------------------

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

    if constexpr (precond) {
        // perform the symbolic and numeric factorization of LU on given sparsity pattern
        if (debug) CHECK_CUDA(cudaDeviceSynchronize());
        auto start_factor = std::chrono::high_resolution_clock::now();

        CUSPARSE::perform_ilu0_factorization(cusparseHandle, descr_L, descr_U, info_L, info_U,
                                             &pBuffer, mb, nnzb, block_dim, d_vals_ILU0, d_rowp,
                                             d_cols, trans_L, trans_U, policy_L, policy_U, dir);

        if (debug) CHECK_CUDA(cudaDeviceSynchronize());
        auto end_factor = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> factor_time = end_factor - start_factor;
        if (can_print && debug) printf("ILU(0) factor time in %.4e sec\n", factor_time.count());
    }

    printf("here1\n");

    // GMRES restart and inner loops begin
    // -----------------------------------

    bool converged = false;
    int total_iter = 0;

    // double triang_time = 0.0, SpMV_time = 0.0, GS_time = 0.0;
    // printf("max_iter %d, m %d\n", max_iter, m);
    int nrestarts = max_iter / m;
    // printf("nrestarts %d\n", nrestarts);

    for (int irestart = 0; irestart < nrestarts; irestart++) {
        // store final subspace size
        int mm = m;

        // compute initial residual r0
        CHECK_CUDA(cudaMemcpy(d_resid, d_rhs, N * sizeof(T), cudaMemcpyDeviceToDevice));
        a = 1.0, b = 0.0;
        if (debug) CHECK_CUDA(cudaDeviceSynchronize());
        auto start_mult = std::chrono::high_resolution_clock::now();
        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a, descrA,
                                      d_vals, d_rowp, d_cols, block_dim, d_x, &b, d_tmp));
        if (debug) CHECK_CUDA(cudaDeviceSynchronize());
        auto end_mult = std::chrono::high_resolution_clock::now();
        a = -1.0;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_tmp, 1, d_resid, 1));
        T beta;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &beta));
        // computed as <h_{-1}, e1>
        // g[0] = -beta;  // RHS orig beta * e1, will get Givens rotationed though

        // copy resid into z
        CHECK_CUDA(cudaMemcpy(d_z, d_resid, N * sizeof(T), cudaMemcpyDeviceToDevice));

        // if (debug) {
        //     printf("----------------\n");
        //     printf("r0:");
        //     auto h_resid = DeviceVec<T>(30, d_resid).createHostVec();
        //     printVec<T>(30, h_resid.getPtr());
        // }

        if (can_print) printf("HGMRES init resid = %.9e\n", beta);
        std::chrono::duration<double> spmv_time = end_mult - start_mult;
        // SpMV_time += spmv_time.count();
        // if (can_print && debug) {
        //     printf("\tSpMV on GPU in %.4e sec\n", spmv_time.count());
        // }

        // printf("here2\n");

        // begin main GMRES Householder loop (one startup loop, diff than MGS method)
        for (int j = 0; j < m + 1; j++, total_iter++) {
            /* compute Householder unit vector w_j */
            T *d_w = &d_Wmat[j * N];  // hold w_j

            T sigma;  // sigma = norm2(z[j:])
            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N - j, &d_z[j], 1, &sigma));

            T zsub0;  // zsub0 = z[0] copied to host
            CHECK_CUDA(cudaMemcpy(&zsub0, &d_z[j], sizeof(T), cudaMemcpyDeviceToHost));
            T sz0 = zsub0 >= 0.0 ? 1.0 : -1.0;
            T alpha = -sz0 * sigma;

            // zero w tmp vec
            CHECK_CUDA(cudaMemset(d_w, 0.0, N * sizeof(T)));

            // printf("here3\n");

            // copy u = vsub - alpha * e1 into the w[j:] (from jth entry on so first j values 0)
            a = 1.0;
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N - j, &a, &d_z[j], 1, &d_w[j], 1));
            a = -alpha;
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, 1, &a, d_one, 1, &d_w[j], 1));

            // if (debug) {
            //     printf("----------------\n");
            //     printf("alpha %.4e, sigma %.4e, sz0 %.4e\n", alpha, sigma, sz0);
            //     printf("w[:,%d]: ", j);
            //     auto h_w = DeviceVec<T>(30, d_w).createHostVec();
            //     printVec<T>(30, h_w.getPtr());
            //     printf("z[:,%d]: ", j);
            //     auto h_z = DeviceVec<T>(30, d_z).createHostVec();
            //     printVec<T>(30, h_z.getPtr());
            // }

            // normalize w[j:] on to make it a unit vec (or just all of w)
            T nrm_w;
            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_w, 1, &nrm_w));
            T inrm_w = 1.0 / nrm_w;
            CHECK_CUBLAS(cublasDscal(cublasHandle, N, &inrm_w, d_w, 1));

            // if (debug) {
            //     printf("w[:,%d]': ", j);
            //     auto h_w = DeviceVec<T>(30, d_w).createHostVec();
            //     printVec<T>(30, h_w.getPtr());
            // }

            // printf("here4\n");

            /* compute Hessenberg vector h_{j-1} */

            // h_{j-1} = P_j * z where P_j = I - 2 w_j w_j^T is a reflector
            //   or in short h_{j-1} = z - 2 <w_j, z> * w_j
            CHECK_CUDA(cudaMemcpy(d_h, d_z, N * sizeof(T), cudaMemcpyDeviceToDevice));
            T wz_dot;
            CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_w, 1, d_z, 1, &wz_dot));
            // if (debug) printf("wz_dot %.4e\n", wz_dot);
            a = -2.0 * wz_dot;
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_w, 1, d_h, 1));

            // copy only up to 0:j or first j+1 vals of d_h into host H matrix (for col j-1)
            //  remember first one is startup iteration, should be ok to do device to host for now
            //  as long as small subspace size? TBD
            if (j == 0) {
                // g[0] = <e1, h_{-1}>, see Saad p. 160
                CHECK_CUBLAS(cublasDdot(cublasHandle, 1, d_one, 1, d_h, 1, &g[0]));
            } else {
                CHECK_CUDA(
                    cudaMemcpy(&HT[m * (j - 1)], d_h, (j + 1) * sizeof(T), cudaMemcpyDeviceToHost));

                // if (debug) {
                //     printf("H[:,%d]: ", j - 1);
                //     printVec<T>(j + 1, &HT[m * (j - 1)]);
                // }

                /* Givens rotations for Hessenberg matrix and RHS */

                // see cuda_examples/gmres/_gmres_utils.py : gmres_householder method Givens
                // section, applies previous Givens rotations
                for (int i = 0; i < j - 1; i++) {
                    T temp = HT[m * (j - 1) + i];
                    HT[m * (j - 1) + i] =
                        cs[i] * HT[m * (j - 1) + i] + ss[i] * HT[m * (j - 1) + (i + 1)];
                    HT[m * (j - 1) + i + 1] =
                        -ss[i] * temp + cs[i] * HT[m * (j - 1) + (i + 1)];  // correct
                    // HT[m * (j - 1) + i + 1] = -ss[i] * temp + ss[i] * HT[m * (j - 1) + (i + 1)];
                    // // incorrect
                }

                // if (debug) {
                //     printf("H[:,%d] p2: ", j - 1);
                //     printVec<T>(j + 1, &HT[m * (j - 1)]);
                //     printf("g:");
                //     printVec<T>(j + 1, g);
                // }

                T hx = HT[m * (j - 1) + j - 1];  // + 1e-12;
                T hy = HT[m * (j - 1) + j];      // + 1e-12;
                // cs[j - 1] = hx / sqrt(hx * hx + hy * hy);
                // ss[j - 1] = cs[j - 1] * hy / hx;
                T r = hypot(hx, hy);  // always non-negative
                cs[j - 1] = hx / r;
                ss[j - 1] = hy / r;
                // if (debug)
                // printf("hx %.4e, hy %.4e, c %.4e, s %.4e\n", hx, hy, cs[j - 1], ss[j - 1]);

                // if (debug) printf("g[%d] pre Givens = %.4e\n", j - 1, g[j - 1]);
                // if (debug) printf("g[%d] pre Givens = %.4e\n", j, g[j]);

                // old style
                g[j] = -ss[j - 1] * g[j - 1];
                g[j - 1] *= cs[j - 1];

                HT[m * (j - 1) + (j - 1)] = r;
                HT[m * (j - 1) + j] = 0.0;

                // if (debug) printf("g[%d] = %.4e\n", j - 1, g[j - 1]);
                // if (debug) printf("g[%d] = %.4e\n", j, g[j]);

                if (can_print && (j % print_freq == 0)) printf("HGMRES [%d] = %.9e\n", j, g[j]);

                // check convergence of householder GMRES using Givens rotations of RHS
                if (abs(g[j]) < (abs_tol + beta * rel_tol)) {
                    if (can_print)
                        printf("HGMRES converged in %d iterations to %.9e resid\n", j + 1, g[j]);
                    mm = j;
                    converged = true;
                    break;
                }  // end of conv if statement
            }      // end of Givens section

            /* apply Reflector backward orthogonalization to get v */
            // v = P1 * ... * Pj * ej
            // let ej stored in d_v device array
            CHECK_CUDA(cudaMemset(d_v, 0.0, N * sizeof(T)));
            CHECK_CUDA(cudaMemcpy(&d_v[j], d_one, sizeof(T), cudaMemcpyDeviceToDevice));

            // perform reflector orthog first Pj then ... then P1
            for (int i = j; i >= 0; i--) {
                T *d_wi = &d_Wmat[i * N];
                T wv_dot;
                CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_wi, 1, d_v, 1, &wv_dot));
                a = -2.0 * wv_dot;
                CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_wi, 1, d_v, 1));
            }

            // if (debug) {
            //     printf("v: ");
            //     auto h_v = DeviceVec<T>(30, d_v).createHostVec();
            //     printVec<T>(30, h_v.getPtr());
            // }

            /* compute new Arnoldi vec and forward orthog */
            if (j <= m) {
                // z = Pj * ... * P1 * A * M^-1 * v

                /* first A * M^-1 * v into z */
                if constexpr (precond) {
                    a = 1.0;
                    CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                        cusparseHandle, dir, trans_L, mb, nnzb, &a, descr_L, d_vals_ILU0, d_rowp,
                        d_cols, block_dim, info_L, d_v, d_tmp, policy_L, pBuffer));
                    CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                        cusparseHandle, dir, trans_U, mb, nnzb, &a, descr_U, d_vals_ILU0, d_rowp,
                        d_cols, block_dim, info_U, d_tmp, d_tmp2, policy_U, pBuffer));
                    a = 1.0, b = 0.0;
                    CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb,
                                                  &a, descrA, d_vals, d_rowp, d_cols, block_dim,
                                                  d_tmp2, &b, d_z));
                } else {
                    a = 1.0, b = 0.0;
                    CHECK_CUSPARSE(cusparseDbsrmv(
                        cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        mb, mb, nnzb, &a, descrA, d_vals, d_rowp, d_cols, block_dim, d_v, &b, d_z));
                }

                // if (debug) {
                //     printf("z2: ");
                //     auto h_z = DeviceVec<T>(30, d_z).createHostVec();
                //     printVec<T>(30, h_z.getPtr());
                // }

                /* then forward Householder reflectors */
                for (int i = 0; i < j + 1; i++) {
                    T *d_wi = &d_Wmat[i * N];
                    T wz_dot;
                    CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_wi, 1, d_z, 1, &wz_dot));
                    a = -2.0 * wz_dot;
                    CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_wi, 1, d_z, 1));
                }

                // if (debug) {
                //     printf("z3: ");
                //     auto h_z = DeviceVec<T>(30, d_z).createHostVec();
                //     printVec<T>(30, h_z.getPtr());
                // }
            }
        }  // end of inner subspace loop

        /* solve the Hessenberg system which is triangular thanks to the Givens rotations */

        // it has a size mm x mm where mm was the final iteration on exit of inner loop
        //      mm-1 to 0 inclusive is full size mm
        for (int i = mm - 1; i >= 0; i--) {
            T num = g[i];
            // subtract previous solved y terms
            for (int j = i + 1; j < mm; j++) {
                // numerator -= H[i,j] * y[j]
                num -= HT[m * j + i] * y[j];
            }
            y[i] = num / HT[m * i + i];
        }

        /* update solution z = Pi * (y[i] * ei + z) iteratively */
        // first set z to zero (full size N)
        CHECK_CUDA(cudaMemset(d_z, 0.0, N * sizeof(T)));
        for (int i = mm - 1; i >= 0; i--) {
            // z += y[i] * ei (just add to that entry)
            a = y[i];
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, 1, &a, d_one, 1, &d_z[i], 1));

            // z -= 2 * <z, wi> * wi (reflector Pi applied to z)
            T *d_wi = &d_Wmat[i * N];
            T wz_dot;
            CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_wi, 1, d_z, 1, &wz_dot));
            a = -2.0 * wz_dot;
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_wi, 1, d_z, 1));
        }

        // if (debug) {
        //     printf("--------------\n");
        //     printf("d_z before precond: ");
        //     auto h_z = DeviceVec<T>(30, d_z).createHostVec();
        //     printVec<T>(30, h_z.getPtr());
        // }

        if constexpr (precond) {
            /* apply right precond z = M^-1 z */
            a = 1.0;
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_L, mb, nnzb, &a,
                                                 descr_L, d_vals_ILU0, d_rowp, d_cols, block_dim,
                                                 info_L, d_z, d_tmp, policy_L, pBuffer));
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, mb, nnzb, &a,
                                                 descr_U, d_vals_ILU0, d_rowp, d_cols, block_dim,
                                                 info_U, d_tmp, d_z, policy_U, pBuffer));
        }

        /* final soln update x += z aka x += M^-1 * Vmm * ymm */
        a = 1.0;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_z, 1, d_x, 1));

    }  // end of outer restart loop

    /* debug check final residual (should match final Givens rotation) */
    if (can_print && debug) {
        // copy rhs into resid again
        CHECK_CUDA(cudaMemcpy(d_resid, d_rhs, N * sizeof(T), cudaMemcpyDeviceToDevice));

        // then subtract Ax from b
        a = 1.0, b = 0.0;
        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, mb, mb, nnzb, &a, descrA,
                                      d_vals, d_rowp, d_cols, block_dim, d_x, &b, d_tmp));
        // resid -= A * x
        a = -1.0;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_tmp, 1, d_resid, 1));

        T final_resid;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_resid, 1, &final_resid));

        if (can_print)
            printf("HGMRES converged to %.9e resid in %d iterations\n", final_resid, total_iter);
    }

    /* cleanup GMRES data and inv permute soln .. TODO : later persist some data in repeated solves?
     * maybe with class or struct? */

    // now also inverse permute the soln data
    permute_soln<BsrMat<DeviceVec<T>>, DeviceVec<T>>(mat, soln);

    // free resources
    cudaFree(pBuffer);
    cusparseDestroyMatDescr(descr_L);
    cusparseDestroyMatDescr(descr_U);
    cusparseDestroyBsrsv2Info(info_L);
    cusparseDestroyBsrsv2Info(info_U);
    cusparseDestroy(cusparseHandle);

    // TODO : still missing a few free / delete[] statements
    CHECK_CUDA(cudaFree(d_resid));
    CHECK_CUDA(cudaFree(d_Wmat));
    CHECK_CUDA(cudaFree(d_tmp));
    CHECK_CUDA(cudaFree(d_tmp2));
    CHECK_CUDA(cudaFree(d_z));
    CHECK_CUDA(cudaFree(d_v));
    CHECK_CUDA(cudaFree(d_h));
    CHECK_CUDA(cudaFree(d_vals_ILU0));

    // print timing data
    CHECK_CUDA(cudaDeviceSynchronize());
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    double dt = duration.count() / 1e6;
    if (can_print) {
        printf("\tfinished in %.4e sec\n", dt);
    }
}  // end of HGMRES solve
};  // namespace CUSPARSE