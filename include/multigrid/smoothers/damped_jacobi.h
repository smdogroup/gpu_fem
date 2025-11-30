#pragma once
#include "../solvers/solve_utils.h"
#include "linalg/vec.h"

template <class Assembler>
class DampedJacobiSmoother : public BaseSolver {
    /* a lexigraphic gauss seidel smoother */
    using T = typename Assembler::T;

   public:
    DampedJacobiSmoother(cublasHandle_t &cublasHandle_, cusparseHandle_t &cusparseHandle_,
                         Assembler &assembler_, BsrMat<DeviceVec<T>> Kmat_, T omega_ = 0.7,
                         int nsmooth_ = 10)
        : cublasHandle(cublasHandle_), cusparseHandle(cusparseHandle_) {
        Kmat = Kmat_;
        block_dim = assembler.getBsrData().block_dim;
        N = assembler_.get_num_vars();
        nnodes = N / block_dim;
        assembler = assembler_;

        nsmooth = nsmooth_;
        omegaJac = omega_;

        // get data out of kmat
        auto d_kmat_bsr_data = Kmat.getBsrData();
        d_kmat_vals = Kmat.getVec().getPtr();
        d_kmat_rowp = d_kmat_bsr_data.rowp;
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
        this->smoothDefect(d_rhs_vec, d_inner_soln_vec, nsmooth, false, 10, omegaJac);

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
        d_rhs_vec = DeviceVec<T>(N);
        d_rhs = d_rhs_vec.getPtr();
        d_inner_soln_vec = DeviceVec<T>(N);
        d_inner_soln = d_inner_soln_vec.getPtr();

        // init some util vecs
        d_temp_vec = DeviceVec<T>(N);
        d_temp = d_temp_vec.getPtr();
        d_temp2 = DeviceVec<T>(N).getPtr();

        d_resid_vec = DeviceVec<T>(N);
        d_resid = d_resid_vec.getPtr();

        // make mat handles for SpMV
        CHECK_CUSPARSE(cusparseCreateMatDescr(&descrKmat));
        CHECK_CUSPARSE(cusparseSetMatType(descrKmat, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descrKmat, CUSPARSE_INDEX_BASE_ZERO));
    }

    template <bool startup = true>
    void buildDiagInvMat() {
        // first need to construct rowp and cols for diagonal (fairly easy)

        // startup section
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
            int ndiag_vals = block_dim * block_dim * nnodes;
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
        }

        // call the kernel to copy out diag vals first
        int ndiag_vals = block_dim * block_dim * nnodes;
        dim3 block(32);
        int nblocks = (ndiag_vals + 31) / 32;
        dim3 grid(nblocks);
        k_copyBlockDiagFromBsrMat<T>
            <<<grid, block>>>(nnodes, block_dim, d_kmat_diagp, d_kmat_vals, d_diag_LU_vals);

        // then on each nodal block of D matrix, cusparse computes LU factorization
        CUSPARSE::perform_ilu0_factorization(cusparseHandle, descr_L, descr_U, info_L, info_U,
                                             &pBuffer, nnodes, diag_inv_nnzb, block_dim,
                                             d_diag_LU_vals, d_diag_rowp, d_diag_cols, trans_L,
                                             trans_U, policy_L, policy_U, dir);

        // now compute Dinv linear operator from LU triang solves (so don't need triang solves in
        // main solve), costs 6 triang solves of D^-1 = U^-1 L^-1
        bool build_lu_inv_operator = true;
        if (build_lu_inv_operator) {
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

                CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, nnodes, nnodes,
                                                     &alpha, descr_U, d_diag_LU_vals, d_diag_rowp,
                                                     d_diag_cols, block_dim, info_U, d_resid,
                                                     d_temp2, policy_U, pBuffer));

                // now copy temp2 into columns of new operator
                dim3 grid2((N + 31) / 32);
                k_setLUinv_operator<T>
                    <<<grid2, block>>>(nnodes, block_dim, i, d_temp2, d_dinv_vals.getPtr());
            }  // this works!

            if constexpr (startup) {
                D_LU_mat = BsrMat<DeviceVec<T>>(d_diag_bsr_data, d_dinv_vals);
            }

        } else {
            if constexpr (startup) {
                D_LU_mat = BsrMat<DeviceVec<T>>(d_diag_bsr_data, d_diag_vals);
            }
        }  // end of Dinv linear operator if block
    }

    void smoothDefect(DeviceVec<T> d_defect, DeviceVec<T> d_soln, int n_iters, bool print = false,
                      int print_freq = 10, T omega = 0.8) {
        /* damped jacobi smoothing */

        for (int iter = 0; iter < n_iters; iter++) {
            // compute Dinv * defect => soln update (aka temp)
            // -----------------------------------------------
            T a = 1.0, b = 0.0;
            // note in this case d_diag_LU_vals_color refers to Dinv form of LU factors on each
            // nodal block
            CHECK_CUSPARSE(cusparseDbsrmv(  // NOTE just uses descrKmat cause would be the same as
                                            // descrDinv (convenience)
                cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, nnodes,
                nnodes, nnodes, &a, descrKmat, d_diag_LU_vals, d_diag_rowp, d_diag_cols, block_dim,
                d_defect.getPtr(), &b, d_temp));

            // -----------------------------------------------------------------
            // soln and defect update
            a = omega;
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_temp, 1, d_soln.getPtr(), 1));

            a = -omega, b = 1.0;  // so that defect := defect - mat*vec
            CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE, nnodes, nnodes,
                                          kmat_nnzb, &a, descrKmat, d_kmat_vals, d_kmat_rowp,
                                          d_kmat_cols, block_dim, d_temp, &b, d_defect.getPtr()));

            /* report progress of defect nrm if printing.. */
            T defect_nrm;
            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_defect.getPtr(), 1, &defect_nrm));
            if (print && iter % print_freq == 0)
                printf("\tDJacobi %d/%d : ||defect|| = %.4e\n", iter + 1, n_iters, defect_nrm);

            // --------------------------------------------------------------------------------
        }  // next block-GS iteration
    }

    // standard matrix data
    Assembler assembler;
    int N, nelems, block_dim, nnodes;
    BsrMat<DeviceVec<T>> Kmat, D_LU_mat;  // can't get Dinv_mat directly at moment
    DeviceVec<T> d_temp_vec, d_resid_vec;
    T *d_temp, *d_resid, *d_temp2;

    T omegaJac;
    int nsmooth;

    DeviceVec<T> d_rhs_vec, d_inner_soln_vec;
    T *d_rhs, *d_inner_soln;

    // CUSPARSE and cublas data
    cusparseHandle_t &cusparseHandle;
    cublasHandle_t &cublasHandle;

    // for kmat
    int kmat_nnzb, *d_kmat_rowp, *d_kmat_cols;
    T *d_kmat_vals, *d_kmat_lu_vals;
    cusparseMatDescr_t descrKmat = 0;
    size_t bufferSizeMV;
    void *buffer_MV = nullptr;

    cusparseMatDescr_t descr_L = 0, descr_U = 0;
    bsrsv2Info_t info_L = 0, info_U = 0;
    void *pBuffer = 0;
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                                policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE,
                              trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;

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
};