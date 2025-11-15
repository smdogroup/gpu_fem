#pragma once
#include "../solvers/solve_utils.h"
#include "_smoothers.cuh"
#include "linalg/vec.h"

/* fourth order chebyshev polynomial smoother*/
// based on this paper, https://arxiv.org/pdf/2202.08830
//  "Optimal Polynomial Smoothers for Multigrid V-cycles" by James Lottes
// commonly used as a multigrid smoother (and his paper gives optimal coefficients for 2-level and multilevel smoothing)
//  can also be used as preconditioner for GMRES iterations
// we use L1-jacobi, with absolute value block-element row-sums (D_{L1})_{ii} = \sum_j |A_{ij}| with i and j representing nodes (not entries inside nodes)
//    this since L1-jacobi doesn't require eigenvalue estimates (always has rho(D_{L1}^{-1} A) <= 1) and thus is commonly used in hypre
//    and SA-AMG methods for tentative prolongator

template <class Assembler, int order = 4>
class ChebyshevPolynomialSmoother : public BaseSolver {
   public:
    using T = typename Assembler::T;

    ChebyshevPolynomialSmoother(cublasHandle_t &cublasHandle_, cusparseHandle_t &cusparseHandle_,
                            Assembler &assembler_, BsrMat<DeviceVec<T>> Kmat_,
                            T omega_ = 1.0, int n_solve_steps_ = 1)
        : cublasHandle(cublasHandle_), cusparseHandle(cusparseHandle_) {
        Kmat = Kmat_;
        block_dim = 6;
        N = assembler_.get_num_vars();
        nnodes = N / 6;
        assembler = assembler_;
        omega = omega_;
        n_solve_steps = n_solve_steps_;  // only used for it as a preconditioner (not MG smoother), but this arg is ignored anyways (fix later)

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
            d_diag_bsr_data = BsrData(nnodes, 6, diag_inv_nnzb, d_diag_rowp, d_diag_cols, nullptr,
                                      nullptr, false);
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
        }  // end of startup

        // we compute the L1 diagonal matrix with absolute value elem-wise row-sums
        // first zero the previous diag values (as we're gonna add into it)
        d_diag_vals.zeroValues(); // this is vector for the opinter d_diag_LU_vals (confusing, can fix later
        int n_mat_vals = block_dim * block_dim * kmat_nnzb;
        dim3 block(32);
        int ntblocks = (n_mat_vals + 31) / 32; // num thread blocks
        dim3 grid(ntblocks);
        printf("compute L1 block diags with nnodes %d and kmat_nnzb %d\n", nnodes, kmat_nnzb);
        k_computeL1BlockDiags<T><<<grid, block>>>(kmat_nnzb, block_dim, d_kmat_rows, d_kmat_vals, d_diag_LU_vals);
        CHECK_CUDA(cudaDeviceSynchronize());
        printf("\tdone with compute L1 block diags\n");

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
        // n_iters should really be 1 though?
        // the kind of hte smoother is really how much it should be iterating (so I'm gonna ignore n_iters here, ignored arg)
        // but it is a required arg of base class or other smoother calls, so leaving it

        // reset z and zprev to zero (cause new smooth solve her)
        cudaMemset(d_z, 0.0, N * sizeof(T));
        cudaMemset(d_zprev, 0.0, N * sizeof(T));

        // iteration starts by first computing z_1 so k=1 (as z_0 = 0)
        for (int k = 1; k < order + 1; k++) { // order = 4 is default fourth-order chebyshev from template parameter
            // compute the residual r = b - A*x_{k-1}
            //  first copy b = d_defect into r the residual
            CHECK_CUDA(cudaMemcpy(d_resid, d_defect.getPtr(), N * sizeof(T), cudaMemcpyDeviceToDevice));
            // hten subtract A * x_{k-1} from r
            T a = -1.0, b = 1.0;
            CHECK_CUSPARSE(cusparseDbsrmv(
                cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, nnodes,
                nnodes, kmat_nnzb, &a, descrKmat, d_kmat_vals, d_kmat_rowp, d_kmat_cols,
                block_dim, d_soln.getPtr(), &b, d_resid));

            // then compute D_{L1}^{-1} * residual (the l1-jacobi preconditioner) into the temp vec
            //   where D_{L1}^{-1} was LU factored and then computed as a linear operator so we can do mat-vec mult here!
            a = omega, b = 0.0; // b = 0 so adds to replace d_temp (with scalar of omega, should be omega = 1.0 by default in D_{L1} jacobi)
            CHECK_CUSPARSE(cusparseDbsrmv(
                cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, nnodes,
                nnodes, diag_inv_nnzb, &a, descrDinvMat, d_dinv_vals.getPtr(), d_diag_rowp, d_diag_cols,
                block_dim, d_resid, &b, d_temp));

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
        } // end of chebyshev recursion
    } // end of smoothDefect function

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
    //    private:  // private data for cusparse and cublas
    // ----------------------------------------------------

    // smoother settings
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