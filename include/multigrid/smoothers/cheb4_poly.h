#pragma once
#include "../solvers/solve_utils.h"
#include "_smoothers.cuh"
#include "linalg/vec.h"

// in order to estimate CG-Lanczos coefficients
#include <vector>

#include "lapacke.h"

/* fourth order chebyshev polynomial smoother*/
// based on this paper, https://arxiv.org/pdf/2202.08830
//  "Optimal Polynomial Smoothers for Multigrid V-cycles" by James Lottes
// commonly used as a multigrid smoother (and his paper gives optimal coefficients for 2-level and
// multilevel smoothing)
//  can also be used as preconditioner for GMRES iterations
// we use L1-jacobi, with absolute value block-element row-sums (D_{L1})_{ii} = \sum_j |A_{ij}| with
// i and j representing nodes (not entries inside nodes)
//    this since L1-jacobi doesn't require eigenvalue estimates (always has rho(D_{L1}^{-1} A) <= 1)
//    and thus is commonly used in hypre and SA-AMG methods for tentative prolongator

template <class Assembler, bool L1_JACOBI = false>
class ChebyshevPolynomialSmoother : public BaseSolver {
   public:
    using T = typename Assembler::T;

    ChebyshevPolynomialSmoother(cublasHandle_t &cublasHandle_, cusparseHandle_t &cusparseHandle_,
                                Assembler &assembler_, BsrMat<DeviceVec<T>> Kmat_, T omega_ = 1.0,
                                int ORDER_ = 4, int n_solve_steps_ = 1, bool debug_ = false)
        : cublasHandle(cublasHandle_), cusparseHandle(cusparseHandle_) {
        Kmat = Kmat_;
        block_dim = assembler_.getBsrData().block_dim;
        N = assembler_.get_num_vars();
        nnodes = N / block_dim;
        assembler = assembler_;
        omega = omega_;
        ORDER = ORDER_;
        n_solve_steps = n_solve_steps_;  // only used for it as a preconditioner (not MG smoother),
                                         // but this arg is ignored anyways (fix later)

        spectral_radius = 1.0;  // default spectral radius (no adjustment until solved)
        CG_LANCZOS = false;     // by default we don't do CG Lanczos updates
        debug = debug_;

        // get data out of kmat
        auto d_kmat_bsr_data = Kmat.getBsrData();
        d_kmat_vals = Kmat.getVec().getPtr();
        d_kmat_rowp = d_kmat_bsr_data.rowp;
        d_kmat_rows = d_kmat_bsr_data.rows;
        d_kmat_cols = d_kmat_bsr_data.cols;
        kmat_nnzb = d_kmat_bsr_data.nnzb;

        initCuda();
        const bool startup = true;
        buildDiagInvMat<startup>();
    }

    void setup_cg_lanczos(DeviceVec<T> loads, int N_LANCZOS_ = 10) {
        /* setup cg lanczos in order to do spectral radius estimates for more robustness */
        CG_LANCZOS = true;
        // assumes loads are in solve order here
        d_lanczos_loads_vec = DeviceVec<T>(N);
        cudaMemcpy(d_lanczos_loads_vec.getPtr(), loads.getPtr(), N * sizeof(T),
                   cudaMemcpyDeviceToDevice);
        N_LANCZOS = N_LANCZOS_;  // number of steps for lanczos spectral radius estimate
        alpha_vals = new T[N_LANCZOS];
        beta_vals = new T[N_LANCZOS];
        delta_vals = new T[N_LANCZOS];
        eta_vals = new T[N_LANCZOS];

        // then compute spectral radius
        compute_spectral_radius();
    }

    bool solve(DeviceVec<T> rhs, DeviceVec<T> soln, bool check_conv = false) {
        /* solve method for the smoother if it is used as a preconditioner instead */

        // setup rhs and soln with init guess of 0
        cudaMemcpy(d_rhs, rhs.getPtr(), N * sizeof(T), cudaMemcpyDeviceToDevice);
        cudaMemset(d_inner_soln, 0.0, N * sizeof(T));  // re-zero the solution

        // call smoother on the defect=rhs and solution pair
        this->smoothDefect(d_rhs_vec, d_inner_soln_vec, n_solve_steps);

        // copy internal soln to external solution of the solve method
        cudaMemcpy(soln.getPtr(), d_inner_soln, N * sizeof(T), cudaMemcpyDeviceToDevice);

        return false;  // fail = False
    }

    void update_after_assembly(DeviceVec<T> &vars) {
        const bool startup = false;
        buildDiagInvMat<startup>();
        if (CG_LANCZOS)
            compute_spectral_radius();  // for more robustness update rho(Dinv*A) aka max eigenvalue
                                        // estimate
    }

    void set_abs_tol(T atol) {}
    void set_rel_tol(T atol) {}
    int get_num_iterations() { return 0; }
    void set_print(bool print) {}
    void free() {}  // TBD on this one

    void initCuda() {
        // init some util vecs
        d_temp_vec = DeviceVec<T>(N);
        d_temp = d_temp_vec.getPtr();
        d_temp2 = DeviceVec<T>(N).getPtr();
        d_resid = DeviceVec<T>(N).getPtr();
        d_z = DeviceVec<T>(N).getPtr();
        d_zprev = DeviceVec<T>(N).getPtr();

        // for linear solver / precond use
        d_rhs_vec = DeviceVec<T>(N);
        d_rhs = d_rhs_vec.getPtr();
        d_inner_soln_vec = DeviceVec<T>(N);
        d_inner_soln = d_inner_soln_vec.getPtr();

        // make mat handles for SpMV
        CHECK_CUSPARSE(cusparseCreateMatDescr(&descrKmat));
        CHECK_CUSPARSE(cusparseSetMatType(descrKmat, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descrKmat, CUSPARSE_INDEX_BASE_ZERO));

        CHECK_CUSPARSE(cusparseCreateMatDescr(&descrDinvMat));
        CHECK_CUSPARSE(cusparseSetMatType(descrDinvMat, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descrDinvMat, CUSPARSE_INDEX_BASE_ZERO));
    }

    template <bool startup = true>
    void buildDiagInvMat() {
        // first need to construct rowp and cols for diagonal (fairly easy)

        // startup section
        int ndiag_vals = block_dim * block_dim * nnodes;
        if constexpr (startup) {
            int *h_diag_rowp = new int[nnodes + 1];
            diag_inv_nnzb = nnodes;
            int *h_diag_cols = new int[nnodes];
            h_diag_rowp[0] = 0;

            for (int i = 0; i < nnodes; i++) {
                h_diag_rowp[i + 1] = i + 1;
                h_diag_cols[i] = i;
            }

            // on host, get the pointer locations in Kmat of the block diag entries..
            int *h_kmat_rowp = DeviceVec<int>(nnodes + 1, d_kmat_rowp).createHostVec().getPtr();
            int *h_kmat_cols = DeviceVec<int>(kmat_nnzb, d_kmat_cols).createHostVec().getPtr();

            // now copy to device
            d_diag_rowp = HostVec<int>(nnodes + 1, h_diag_rowp).createDeviceVec().getPtr();
            d_diag_cols = HostVec<int>(nnodes, h_diag_cols).createDeviceVec().getPtr();

            // create the bsr data object on device
            d_diag_bsr_data = BsrData(nnodes, block_dim, diag_inv_nnzb, d_diag_rowp, d_diag_cols,
                                      nullptr, nullptr, false);
            delete[] h_diag_rowp;
            delete[] h_diag_cols;

            // now allocate DeviceVec for the values
            d_diag_vals = DeviceVec<T>(ndiag_vals);
            d_diag_LU_vals = d_diag_vals.getPtr();  // just copy these pointers..

            int *h_kmat_diagp = new int[nnodes];
            for (int block_row = 0; block_row < nnodes; block_row++) {
                for (int jp = h_kmat_rowp[block_row]; jp < h_kmat_rowp[block_row + 1]; jp++) {
                    int block_col = h_kmat_cols[jp];
                    // printf("row %d, col %d\n", block_row, block_col);
                    if (block_row == block_col) {
                        h_kmat_diagp[block_row] = jp;
                    }
                }
            }

            d_kmat_diagp = HostVec<int>(nnodes, h_kmat_diagp).createDeviceVec().getPtr();

            delete[] h_kmat_rowp;
            delete[] h_kmat_cols;

            if constexpr (L1_JACOBI) {
                d_block_norms = DeviceVec<T>(kmat_nnzb).getPtr();
            }
        }  // end of startup

        // regular jacobi preconditioner
        //  zero previous values (to get new Dinv, in case optimization or nonlinear problem)
        d_diag_vals.zeroValues();  // this is vector for the opinter d_diag_LU_vals (confusing, can
                                   // fix later
        k_copyBlockDiagFromBsrMat<T><<<(ndiag_vals + 31) / 32, 32>>>(
            nnodes, block_dim, d_kmat_diagp, d_kmat_vals, d_diag_LU_vals);

        // we compute the L1 diagonal matrix off-diag terms
        if constexpr (L1_JACOBI) {
            // get block norms
            printf("trying to compute block norms\n");
            k_computeBlockL1Norms<T><<<(kmat_nnzb + 31) / 32, 32>>>(
                kmat_nnzb, block_dim, d_kmat_rows, d_kmat_cols, d_kmat_vals, d_block_norms);
            CHECK_CUDA(cudaDeviceSynchronize());
            printf("\tdone computing block norms\n");

            // then add in ||A_{ij}||_1 for off-block diags (diagonal part i==js just 0 cause we
            // d_block_norms has zero for those blocks)
            printf("trying to add block norms to D mat\n");
            k_accumulateBlockL1ToDiag<T><<<(kmat_nnzb + 31) / 32, 32>>>(
                kmat_nnzb, block_dim, d_kmat_rows, d_block_norms, d_diag_LU_vals);
            CHECK_CUDA(cudaDeviceSynchronize());
            printf("\tdone adding block norms to D mat\n");
        }

        // ilu0 factoriation
        if constexpr (startup) {
            // create M matrix object (for full numeric factorization)
            cusparseCreateMatDescr(&descr_M);
            cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ZERO);
            cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL);
            cusparseCreateBsrilu02Info(&info_M);

            // init L matrix objects (for triangular solve)
            cusparseCreateMatDescr(&descr_L);
            cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO);
            cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
            cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
            cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT);
            cusparseCreateBsrsv2Info(&info_L);

            // init U matrix objects (for triangular solve)
            cusparseCreateMatDescr(&descr_U);
            cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO);
            cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL);
            cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER);
            cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT);
            cusparseCreateBsrsv2Info(&info_U);

            // symbolic and numeric factorizations
            CHECK_CUSPARSE(cusparseDbsrilu02_bufferSize(
                cusparseHandle, dir, nnodes, diag_inv_nnzb, descr_M, d_diag_LU_vals, d_diag_rowp,
                d_diag_cols, block_dim, info_M, &pBufferSize_M));
            CHECK_CUSPARSE(cusparseDbsrsv2_bufferSize(
                cusparseHandle, dir, trans_L, nnodes, diag_inv_nnzb, descr_L, d_diag_LU_vals,
                d_diag_rowp, d_diag_cols, block_dim, info_L, &pBufferSize_L));
            CHECK_CUSPARSE(cusparseDbsrsv2_bufferSize(
                cusparseHandle, dir, trans_U, nnodes, diag_inv_nnzb, descr_U, d_diag_LU_vals,
                d_diag_rowp, d_diag_cols, block_dim, info_U, &pBufferSize_U));
            pBufferSize = std::max({pBufferSize_M, pBufferSize_L, pBufferSize_U});
            // cudaMalloc((void **)&pBuffer, pBufferSize);
            cudaMalloc(&pBuffer, pBufferSize);

            // perform ILU symbolic factorization on L
            CHECK_CUSPARSE(cusparseDbsrilu02_analysis(
                cusparseHandle, dir, nnodes, diag_inv_nnzb, descr_M, d_diag_LU_vals, d_diag_rowp,
                d_diag_cols, block_dim, info_M, policy_M, pBuffer));
            status = cusparseXbsrilu02_zeroPivot(cusparseHandle, info_M, &structural_zero);
            if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
                printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
            }

            // analyze sparsity patern of L for efficient triangular solves
            CHECK_CUSPARSE(cusparseDbsrsv2_analysis(
                cusparseHandle, dir, trans_L, nnodes, diag_inv_nnzb, descr_L, d_diag_LU_vals,
                d_diag_rowp, d_diag_cols, block_dim, info_L, policy_L, pBuffer));
            CHECK_CUDA(cudaDeviceSynchronize());

            // analyze sparsity pattern of U for efficient triangular solves
            CHECK_CUSPARSE(cusparseDbsrsv2_analysis(
                cusparseHandle, dir, trans_U, nnodes, diag_inv_nnzb, descr_U, d_diag_LU_vals,
                d_diag_rowp, d_diag_cols, block_dim, info_U, policy_U, pBuffer));
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        // perform ILU numeric factorization (with M policy)
        CHECK_CUSPARSE(cusparseDbsrilu02(cusparseHandle, dir, nnodes, diag_inv_nnzb, descr_M,
                                         d_diag_LU_vals, d_diag_rowp, d_diag_cols, block_dim,
                                         info_M, policy_M, pBuffer));
        CHECK_CUDA(cudaDeviceSynchronize());
        status = cusparseXbsrilu02_zeroPivot(cusparseHandle, info_M, &numerical_zero);
        if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
            printf("block U(%d,%d) is not invertible\n", numerical_zero, numerical_zero);
        }

        // then on each nodal block of D matrix, cusparse computes LU factorization
        // CUSPARSE::perform_ilu0_factorization(cusparseHandle, descr_L, descr_U, info_L, info_U,
        //                                      &pBuffer, nnodes, diag_inv_nnzb, block_dim,
        //                                      d_diag_LU_vals, d_diag_rowp, d_diag_cols, trans_L,
        //                                      trans_U, policy_L, policy_U, dir);

        // now compute Dinv linear operator from LU triang solves (so don't need triang solves in
        // main solve), costs 6 triang solves of D^-1 = U^-1 L^-1

        // startup part of Dinv linear operator
        if constexpr (startup) {
            d_dinv_vals = DeviceVec<T>(ndiag_vals);
        }

        // apply e1 through e6 (each dof per node for shell if 6 dof per node case)
        // to get effective matrix.. need six temp vectors..
        for (int i = 0; i < block_dim; i++) {
            // set d_temp to ei (one of e1 through e6 per block)
            cudaMemset(d_temp, 0.0, N * sizeof(T));
            dim3 block(32);
            dim3 grid((nnodes + 31) / 32);
            k_setBlockUnitVec<T><<<grid, block>>>(nnodes, block_dim, i, d_temp);

            // now compute D^-1 through U^-1 L^-1 triang solves and copy result into d_temp2
            const double alpha = 1.0;
            CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                cusparseHandle, dir, trans_L, nnodes, nnodes, &alpha, descr_L, d_diag_LU_vals,
                d_diag_rowp, d_diag_cols, block_dim, info_L, d_temp, d_resid, policy_L,
                pBuffer));  // prob only need U^-1 part for block diag.. TBD

            CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                cusparseHandle, dir, trans_U, nnodes, nnodes, &alpha, descr_U, d_diag_LU_vals,
                d_diag_rowp, d_diag_cols, block_dim, info_U, d_resid, d_temp2, policy_U, pBuffer));

            // now copy temp2 into columns of new operator
            dim3 grid2((N + 31) / 32);
            k_setLUinv_operator<T>
                <<<grid2, block>>>(nnodes, block_dim, i, d_temp2, d_dinv_vals.getPtr());
        }  // this works!

        if constexpr (startup) {
            D_LU_mat = BsrMat<DeviceVec<T>>(d_diag_bsr_data, d_dinv_vals);
        }
    }

    void smoothDefect(DeviceVec<T> d_defect, DeviceVec<T> d_soln, int n_iters, bool print = false,
                      int print_freq = 10) {
        /* apply chebyshev smoother a certain number of times to solve the system */

        for (int iter = 0; iter < n_iters; iter++) {
            // number of smoothing steps

            // reset z and zprev to zero (cause new smooth solve her)
            cudaMemset(d_z, 0.0, N * sizeof(T));
            cudaMemset(d_zprev, 0.0, N * sizeof(T));

            // NOTE : I've rewritten the recursion with d = b - A*x_k the defect
            //   and z_k = delta(x_k) helping to update the defect (as standard MG smoother would)
            //   still only does one A*() mat-vec product and one Dinv*() mat-vec product each

            // iteration starts by first computing z_1 so k=1 (as z_0 = 0)
            for (int k = 1; k < ORDER + 1;
                 k++) {  // order = 4 is default fourth-order chebyshev from template parameter
                // then compute D_{L1}^{-1} * residual (the l1-jacobi preconditioner) into the temp
                // vec
                //   where D_{L1}^{-1} was LU factored and then computed as a linear operator so we
                //   can do mat-vec mult here!
                // default spectral radius is 1.0 when CG Lanczos not set
                T a = omega / spectral_radius,
                  b = 0.0;  // b = 0 so adds to replace d_temp (with scalar of omega,
                            // should be omega = 1.0 by default in D_{L1} jacobi)
                CHECK_CUSPARSE(cusparseDbsrmv(
                    cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    nnodes, nnodes, diag_inv_nnzb, &a, descrDinvMat, d_dinv_vals.getPtr(),
                    d_diag_rowp, d_diag_cols, block_dim, d_defect.getPtr(), &b, d_temp));

                // then compute the recursion of zprev into z
                //  first re-zero new z
                cudaMemset(d_z, 0.0, N * sizeof(T));
                //  then add old z into it with the prescribed scalar
                a = (2.0 * k - 3.0) / (2.0 * k + 1.0);
                CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_zprev, 1, d_z, 1));
                // then add preconditioned residual into it too
                a = (8.0 * k - 4.0) / (2.0 * k + 1.0);
                CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_temp, 1, d_z, 1));

                // then copy d_z into the previous value (for next iteration)
                CHECK_CUDA(cudaMemcpy(d_zprev, d_z, N * sizeof(T), cudaMemcpyDeviceToDevice));

                // and finally update the solution using the current d_z vector
                a = 1.0;
                CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_z, 1, d_soln.getPtr(), 1));

                // update the defect by z_k = delta(x_k), so defect -= A * z_k
                a = -1.0, b = 1.0;
                CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                              CUSPARSE_OPERATION_NON_TRANSPOSE, nnodes, nnodes,
                                              kmat_nnzb, &a, descrKmat, d_kmat_vals, d_kmat_rowp,
                                              d_kmat_cols, block_dim, d_z, &b, d_defect.getPtr()));

            }  // end of chebyshev recursion
        }

    }  // end of smoothDefect function

    /* CG-lanczos spectral radius section */
    void compute_spectral_radius() {
        // temporarily rename some temp vecs/pointers for CG style coefficients
        T *d_x = d_inner_soln;
        T *d_p = d_temp;
        T *d_w = d_temp2;
        // lastly d_z already covered

        /* first run n_lanczos steps of CG (with only jacobi preconditioner) */
        // code reused from PCG (since don't want duplicate memory by extra PCG object, and
        // BaseSolver makes it so I can't easily call it as jacobi precond) I also don't have the
        // grid object to easily make PCG, anyways could generalize / cleanup later, just get this
        // working for now
        cudaMemset(d_x, 0.0, N * sizeof(T));
        cudaMemcpy(d_resid, d_lanczos_loads_vec.getPtr(), N * sizeof(T), cudaMemcpyDeviceToDevice);
        T rho_prev, rho;  // coefficients that we need to remember
        // inner loop
        for (int j = 0; j < N_LANCZOS; j++) {
            // compute z = Dinv*r
            T a = 1.0, b = 0.0;
            CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE, nnodes, nnodes,
                                          diag_inv_nnzb, &a, descrDinvMat, d_dinv_vals.getPtr(),
                                          d_diag_rowp, d_diag_cols, block_dim, d_resid, &b, d_z));
            // compute dot products, rho = <r, z>
            CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_resid, 1, d_z, 1, &rho));
            if (j == 0) {
                // first iteration, p := z
                cudaMemcpy(d_p, d_z, N * sizeof(T), cudaMemcpyDeviceToDevice);
            } else {
                // compute beta and record it
                beta_vals[j - 1] = rho / rho_prev;
                // p_new = z + beta * p in two steps
                a = beta_vals[j - 1];  // p *= beta scalar
                CHECK_CUBLAS(cublasDscal(cublasHandle, N, &a, d_p, 1));
                a = 1.0;  // p += z
                CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_z, 1, d_p, 1));
            }
            // store rho for next iteration (prev), only used in this part
            rho_prev = rho;
            // compute w = A * p
            a = 1.0, b = 0.0;
            CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE, nnodes, nnodes,
                                          kmat_nnzb, &a, descrKmat, d_kmat_vals, d_kmat_rowp,
                                          d_kmat_cols, block_dim, d_p, &b, d_w));
            // compute alpha = <r,z> / <w,p> = rho / <w,p>
            T wp0;
            CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_w, 1, d_p, 1, &wp0));
            alpha_vals[j] = rho / wp0;
            // x += alpha * p
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &alpha_vals[j], d_p, 1, d_x, 1));
            // r -= alpha * w
            a = -alpha_vals[j];
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_w, 1, d_resid, 1));
        }
        // then record the last CG coefficient
        // z = Dinv*r
        T a = 1.0, b = 0.0;
        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, nnodes, nnodes,
                                      diag_inv_nnzb, &a, descrDinvMat, d_dinv_vals.getPtr(),
                                      d_diag_rowp, d_diag_cols, block_dim, d_resid, &b, d_z));
        // compute rho = <r, z>
        CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_resid, 1, d_z, 1, &rho));
        // compute last beta
        beta_vals[N_LANCZOS - 1] = rho / rho_prev;

        /* now compute equivalent lanczos coefficients */
        for (int j = 0; j < N_LANCZOS; j++) {
            delta_vals[j] = (j == 0) ? (1.0 / alpha_vals[j])
                                     : (1.0 / alpha_vals[j] + beta_vals[j - 1] / alpha_vals[j - 1]);
            eta_vals[j] = sqrt(beta_vals[j]) / alpha_vals[j];
        }

        /* now get spectral radius from LAPACKe small tridiag matrix eigval solve on the host */
        int info = LAPACKE_dstev(LAPACK_ROW_MAJOR,  // matrices stored row-major in C++
                                 'N',               // compute eigenvalues only
                                 N_LANCZOS,
                                 delta_vals,  // diagonal
                                 eta_vals,    // off-diagonal
                                 nullptr,     // no eigenvectors
                                 N_LANCZOS);
        // max eigenvalue (as it overwrites eigvals into delta_vals in-place)
        T max_eigval = delta_vals[0];
        for (int i = 1; i < N_LANCZOS; i++) {
            if (delta_vals[i] > max_eigval) max_eigval = delta_vals[i];
        }
        // and set this as spectral radius estimate (recommend omega = 0.9 or something so we are
        // consrevative)
        spectral_radius = max_eigval;
        // print current max spectral radius for DEBUG
        if (debug) printf("spectral radius %.8e\n", spectral_radius);
    }

    /* prolong matrix-smoothing area (AMG) */

    void smoothMatrix(int n_iters, BsrMat<DeviceVec<T>> *prolong_mat, BsrMat<DeviceVec<T>> *Z_mat,
                      BsrMat<DeviceVec<T>> *Zprev_mat, int nnzb_prod, int *d_P_prodblocks,
                      int *d_K_prodblocks, int *d_Z_prodblocks) {
        // smooth the prolongation matrix using Kmat and Dinv mat, Z_mat is temp matrix for
        // smoothing process
        // TODO : add option if we want to do fewer smoothing steps (if using higher-order)?

        // store vals for matrices and other useful pointers
        T *d_P_vals = prolong_mat->getPtr();
        T *d_Z_vals = Z_mat->getPtr();
        T *d_Zprev_vals = Zprev_mat->getPtr();
        auto prolong_bsr_data = prolong_mat->getBsrData();
        int P_nnzb = prolong_bsr_data.nnzb;
        int *d_P_rows = prolong_bsr_data.rows;
        int *d_P_cols = prolong_bsr_data.cols;

        if constexpr (Assembler::Phys::vars_per_node > 6) {
            printf(
                "WARNING: vpn > 6, smooth matrix exited. Devel to get this to work, change block "
                "sizes for HR element\n");
            return;
        }

        for (int iter = 0; iter < n_iters; iter++) {
            // number of smoothing steps

            // reset z and zprev to zero (cause new smooth solve her)
            Z_mat->zeroValues();
            Zprev_mat->zeroValues();

            // iteration starts by first computing z_1 so k=1 (as z_0 = 0)
            for (int k = 1; k < ORDER + 1; k++) {
                // compute -omega/rho(Dinv*A) * beta_k * A*P into Z first (scaled prolong defect
                // matrix)
                Z_mat->zeroValues();
                dim3 PKP_block(216), PKP_grid(nnzb_prod);
                T beta_k = (8.0 * k - 4.0) / (2.0 * k + 1.0);
                T a = -omega / spectral_radius * beta_k;
                k_compute_mat_mat_prod<T><<<PKP_grid, PKP_block>>>(
                    nnzb_prod, block_dim, a, d_K_prodblocks, d_P_prodblocks, d_Z_prodblocks,
                    d_kmat_vals, d_P_vals, d_Z_vals);

                // compute Dinv*Z into Z in-place (equiv to Dinv*scale*A*P => Z)
                dim3 DP_block(216), DP_grid(P_nnzb);
                k_compute_Dinv_P_mmprod<T><<<DP_grid, DP_block>>>(
                    P_nnzb, block_dim, d_dinv_vals.getPtr(), d_P_rows, d_P_vals);

                // add alpha_k * Zprev into Z
                dim3 add_block(64);
                T alpha_k = (2.0 * k - 3.0) / (2.0 * k + 1.0);
                k_add_colored_submat_PFP<T>
                    <<<DP_grid, add_block>>>(P_nnzb, block_dim, alpha_k, 0, d_Zprev_vals, d_Z_vals);

                // do orthogonal projector on Z (only really needed for coarse-grid galerkin AMG,
                // not smooth GMG) if constexpr (do_orthog_projector) {
                //     dim3 OP_block(32), OP_grid(nnodes_fine);
                //     d_SU_vals.zeroValues();
                //     // compute SU matrix
                //     k_orthog_projector_computeSU<T><<<OP_grid, OP_block>>>(nnodes_fine,
                //     block_dim, d_Bc,
                //         d_free_dof_ptr, d_PF_rowp, d_PF_cols, d_PF_vals, d_SU_vals.getPtr());

                //     // remove rigid-body row-sums
                //     k_orthog_projector_removeRowSums<T><<<OP_grid, OP_block>>>(nnodes_fine,
                //     block_dim, d_Bc, d_free_dof_ptr, d_PF_rowp, d_PF_cols, d_SU_vals.getPtr(),
                //     d_UTUinv_vals.getPtr(), d_PF_vals);
                // }

                // add Z into P (the prolongation update)
                T scale = 1.0;
                k_add_colored_submat_PFP<T>
                    <<<DP_grid, add_block>>>(P_nnzb, block_dim, scale, 0, d_Z_vals, d_P_vals);

                // now copy Z into Zprev
                Z_mat->copyValuesTo(*Zprev_mat);

            }  // end of chebyshev recursion
        }

    }  // end of smoothMatrix function

    // data
    Assembler assembler;
    int N, nelems, block_dim, nnodes;
    BsrMat<DeviceVec<T>> Kmat, D_LU_mat;  // can't get Dinv_mat directly at moment
    DeviceVec<T> d_temp_vec, d_rhs_vec, d_inner_soln_vec;
    T *d_temp, *d_temp2, *d_resid;
    T *d_rhs, *d_inner_soln;
    T *d_z, *d_zprev;
    const int *d_elem_conn;
    HostVec<int> h_color_rowp;
    int n_solve_steps;

    // turn off private during debugging
   private:  // private data for cusparse and cublas
    // ----------------------------------------------------

    /* CG-Lanczos data */
    bool CG_LANCZOS;
    DeviceVec<T> d_lanczos_loads_vec;
    int N_LANCZOS;
    T spectral_radius = 1.0;
    T *alpha_vals, *beta_vals;  // cg coefficients
    T *delta_vals, *eta_vals;   // lanczos coefficients

    /* main smoother data */
    // smoother settings
    int ORDER = 4;
    T omega = 1.0;
    bool symmetric = false;

    // private data
    cublasHandle_t &cublasHandle;
    cusparseHandle_t &cusparseHandle;
    cusparseMatDescr_t descrKmat = 0, descrDinvMat = 0;
    size_t bufferSizeMV;
    void *buffer_MV = nullptr;

    // color rowp and nnzb pointers data for row-slicing
    int *h_color_submat_nnzb;
    int **d_color_submat_rowp, **d_color_submat_rows, **d_color_submat_cols;
    T **d_color_submat_vals;

    // for diag inv mat
    int diag_inv_nnzb, *d_diag_rowp, *d_diag_cols;
    int *d_piv, *d_info;
    DeviceVec<T> d_diag_vals;
    T *d_diag_LU_vals;
    T **d_diag_LU_batch_ptr, **d_temp_batch_ptr;
    bool build_lu_inv_operator;
    int *d_kmat_diagp;
    BsrData d_diag_bsr_data;
    DeviceVec<T> d_dinv_vals;
    T *d_block_norms;
    bool debug;

    // for kmat
    int kmat_nnzb, *d_kmat_rowp, *d_kmat_rows, *d_kmat_cols;
    T *d_kmat_vals, *d_kmat_lu_vals;

    // CUSPARSE triang solve for Dinv as diag LU
    cusparseMatDescr_t descr_L = 0, descr_U = 0;
    bsrsv2Info_t info_L = 0, info_U = 0;
    void *pBuffer = 0;
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                                policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE,
                              trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;

    // and simiarly for Kmat a few differences
    bool full_LU;  // full LU only for coarsest mesh
    cusparseMatDescr_t descr_kmat_L = 0, descr_kmat_U = 0;
    bsrsv2Info_t info_kmat_L = 0, info_kmat_U = 0;
    void *kmat_pBuffer = 0;

    // more objects for ilu0 factorization
    cusparseMatDescr_t descr_M = 0;
    bsrilu02Info_t info_M = 0;
    int pBufferSize_M, pBufferSize_L, pBufferSize_U, pBufferSize;
    int structural_zero, numerical_zero;
    const cusparseSolvePolicy_t policy_M =
        CUSPARSE_SOLVE_POLICY_USE_LEVEL;  // CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    cusparseStatus_t status;
};
