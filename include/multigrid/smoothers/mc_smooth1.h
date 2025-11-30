#pragma once
#include "../solvers/solve_utils.h"
#include "_smoothers.cuh"
#include "linalg/vec.h"

template <class Assembler>
class MulticolorGSSmoother_V1 : public BaseSolver {
    /* a multicolor submat gauss seidel smoother */
   public:
    using T = typename Assembler::T;

    // MulticolorGSSmoother_V1() = default;

    MulticolorGSSmoother_V1(cublasHandle_t &cublasHandle_, cusparseHandle_t &cusparseHandle_,
                            Assembler &assembler_, BsrMat<DeviceVec<T>> Kmat_,
                            HostVec<int> h_color_rowp_, T omega_ = 1.0, bool symmetric_ = false,
                            int n_solve_steps_ = 4)
        : cublasHandle(cublasHandle_), cusparseHandle(cusparseHandle_) {
        Kmat = Kmat_;
        h_color_rowp = h_color_rowp_;
        block_dim = assembler_.getBsrData().block_dim;
        N = assembler_.get_num_vars();
        nnodes = N / block_dim;
        assembler = assembler_;
        omega = omega_;
        symmetric = symmetric_;
        n_solve_steps = n_solve_steps_;  // only used for it as a preconditioner (not MG smoother)

        // get data out of kmat
        auto d_kmat_bsr_data = Kmat.getBsrData();
        d_kmat_vals = Kmat.getVec().getPtr();
        d_kmat_rowp = d_kmat_bsr_data.rowp;
        d_kmat_cols = d_kmat_bsr_data.cols;
        kmat_nnzb = d_kmat_bsr_data.nnzb;

        initCuda();
        const bool startup = true;
        buildDiagInvMat<startup>();
        buildTransposeColorMatrices<startup>();
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
        buildTransposeColorMatrices<startup>();
    }

    void set_abs_tol(T atol) {}
    void set_rel_tol(T atol) {}
    int get_num_iterations() { return 0; }
    void set_print(bool print) {}
    void free() {}  // TBD on this one

    void initCuda() {
        // // init handles
        // CHECK_CUBLAS(cublasCreate(&cublasHandle));
        // CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

        // init some util vecs
        d_temp_vec = DeviceVec<T>(N);
        d_temp = d_temp_vec.getPtr();
        d_temp2 = DeviceVec<T>(N).getPtr();
        d_resid = DeviceVec<T>(N).getPtr();

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
        }  // end of startup

        // call the kernel to copy out diag vals first
        int ndiag_vals = block_dim * block_dim * nnodes;
        dim3 block(32);
        int nblocks = (ndiag_vals + 31) / 32;
        dim3 grid(nblocks);
        k_copyBlockDiagFromBsrMat<T>
            <<<grid, block>>>(nnodes, block_dim, d_kmat_diagp, d_kmat_vals, d_diag_LU_vals);

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

    template <bool startup = true>
    void buildTransposeColorMatrices() {
        // build color transpose matrices (one for each color)
        // only needed for fastest MC-BGS method..

        // compute the sparsity patterns of each col-subcolor matrix N x N_c (as N x N_c matrix)
        // stored in one data structure (basicaly each nodal block is stored in different order,
        // with a few extra pointers than the Kmat) kernels for some of these steps (initialization
        // / assembly) may not be fully optimized yet..

        int num_colors = h_color_rowp.getSize() - 1;
        int *color_rowp = h_color_rowp.getPtr();  // says which rows in d_kmat_rowp are each
        int block_dim2 = block_dim * block_dim;

        if constexpr (startup) {  // on startup block (not reassembly)
            int **h_color_submat_rowp = new int *[num_colors];
            int **h_color_submat_rows = new int *[num_colors];
            int **h_color_submat_cols = new int *[num_colors];
            h_color_submat_nnzb = new int[num_colors];  // what is the nnzb for each submat

            // copy kmat pointers to host
            int *h_kmat_rowp = DeviceVec<int>(nnodes + 1, d_kmat_rowp).createHostVec().getPtr();
            int *h_kmat_cols = DeviceVec<int>(kmat_nnzb, d_kmat_cols).createHostVec().getPtr();

            // get sparsity for each color-sliced matrix
            for (int icolor = 0; icolor < num_colors; icolor++) {
                // temp debug
                // printf("color %d, ", icolor);

                int start_node = color_rowp[icolor], end_node = color_rowp[icolor + 1];
                // int ncols = end_node - start_node;
                // int mb = nnodes, nb = ncols; // dimensions of column sub-matrix

                // construct a rowp and cols for each sub-matrix
                h_color_submat_rowp[icolor] = new int[nnodes + 1];
                int *_rowp = h_color_submat_rowp[icolor];
                _rowp[0] = 0;
                for (int i = 0; i < nnodes; i++) {
                    int _row_ct = 0;
                    for (int jp = h_kmat_rowp[i]; jp < h_kmat_rowp[i + 1]; jp++) {
                        int j = h_kmat_cols[jp];
                        if (start_node <= j && j < end_node) {
                            _row_ct++;
                        }
                    }
                    _rowp[i + 1] = _rowp[i] + _row_ct;
                }

                int _nnzb = _rowp[nnodes];
                h_color_submat_rows[icolor] = new int[_nnzb];
                int *_rows = h_color_submat_rows[icolor];
                for (int i = 0; i < nnodes; i++) {
                    for (int jp = _rowp[i]; jp < _rowp[i + 1]; jp++) {
                        _rows[jp] = i;
                    }
                }

                h_color_submat_cols[icolor] = new int[_nnzb];
                int *_cols = h_color_submat_cols[icolor];
                int *_next = new int[nnodes];  // help for inserting matrix
                memcpy(_next, _rowp, nnodes * sizeof(int));
                for (int i = 0; i < nnodes; i++) {
                    for (int jp = h_kmat_rowp[i]; jp < h_kmat_rowp[i + 1]; jp++) {
                        int j = h_kmat_cols[jp];
                        if (start_node <= j && j < end_node) {
                            int j2 = j - start_node;
                            _cols[_next[i]++] = j2;
                        }
                    }
                }
                delete[] _next;
                h_color_submat_nnzb[icolor] = _nnzb;
            }
            d_color_submat_rowp = new int *[num_colors];
            d_color_submat_rows = new int *[num_colors];
            d_color_submat_cols = new int *[num_colors];
            d_color_submat_vals = new T *[num_colors];

            for (int icolor = 0; icolor < num_colors; icolor++) {
                cudaMalloc((void **)&d_color_submat_rowp[icolor], (nnodes + 1) * sizeof(int));
                int submat_nnzb = h_color_submat_nnzb[icolor];
                cudaMalloc((void **)&d_color_submat_rows[icolor], submat_nnzb * sizeof(int));
                cudaMalloc((void **)&d_color_submat_cols[icolor], submat_nnzb * sizeof(int));
                cudaMalloc((void **)&d_color_submat_vals[icolor],
                           block_dim2 * submat_nnzb * sizeof(T));

                // copy data to device
                cudaMemcpy(d_color_submat_rowp[icolor], h_color_submat_rowp[icolor],
                           (nnodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(d_color_submat_rows[icolor], h_color_submat_rows[icolor],
                           submat_nnzb * sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(d_color_submat_cols[icolor], h_color_submat_cols[icolor],
                           submat_nnzb * sizeof(int), cudaMemcpyHostToDevice);
            }
        }  // end of startup block

        // re-assembly part.. need to copy values from new Kmat into submatrices for each color
        // better way is maybe a transpose mat-vec product on row-sliced data?... but that may not
        // be as fast as you think?
        for (int icolor = 0; icolor < num_colors; icolor++) {
            // submat comes out of T **d_submat_vals then by color..
            int submat_nnzb = h_color_submat_nnzb[icolor];
            int start_col = color_rowp[icolor];
            // int submat_nnz = submat_nnzb * block_dim2;

            dim3 block(block_dim2);
            dim3 grid(submat_nnzb);

            // TODO : figure out good strategy to optimize this kernel later..
            k_copy_color_submat<T><<<grid, block>>>(
                nnodes, submat_nnzb, start_col, block_dim, d_color_submat_rows[icolor],
                d_color_submat_cols[icolor], d_kmat_rowp, d_kmat_cols, d_kmat_vals,
                d_color_submat_vals[icolor]);
        }
    }

    void smoothDefect(DeviceVec<T> d_defect, DeviceVec<T> d_soln, int n_iters, bool print = false,
                      int print_freq = 10) {
        /* first fast version of the smoother using color submatrices */

        int num_colors = h_color_rowp.getSize() - 1;
        int *color_rowp = h_color_rowp.getPtr();
        // printf("mc BGS-fast with # colors = %d\n", num_colors);

        bool time_debug = false;
        // bool time_debug = true;
        if (time_debug) printf("\t\tncolors = %d, #iters %d MC-BGS\n", num_colors, n_iters);
        print_freq = max(print_freq, 1);  // so not zero

        int m = symmetric ? 2 * n_iters : n_iters;

        for (int iter = 0; iter < m; iter++) {
            for (int _icolor = 0; _icolor < num_colors; _icolor++) {
                // -------------------------------------------------------------
                // prelim block (getting color sub-vectors ready)

                if (time_debug) CHECK_CUDA(cudaDeviceSynchronize());
                auto prelim_time = std::chrono::high_resolution_clock::now();
                // int _icolor2 = (_icolor + iter) % num_colors;  // permute order as you go
                bool rev_colors = symmetric ? iter % 2 == 1 : false;
                int icolor = rev_colors ? num_colors - 1 - _icolor : _icolor;

                // get active rows / cols for this color
                int start = color_rowp[icolor], end = color_rowp[icolor + 1];
                int nblock_rows_color = end - start;
                int nrows_color = nblock_rows_color * block_dim;
                T *d_defect_color = &d_defect.getPtr()[block_dim * start];
                cudaMemset(d_temp, 0.0, N * sizeof(T));  // holds dx_color
                T *d_temp_color = &d_temp[block_dim * start];
                int block_dim2 = block_dim * block_dim;
                int diag_inv_nnzb_color = nblock_rows_color;
                T *d_dinv_vals_color = &d_dinv_vals.getPtr()[start * block_dim2];  // from LU factor

                if (time_debug) {
                    CHECK_CUDA(cudaDeviceSynchronize());
                    auto end_prelim_time = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> full_prelim_time = end_prelim_time - prelim_time;
                    printf("\t\tprelim time on iter %d,color %d in %.2e sec\n", iter, icolor,
                           full_prelim_time.count());
                }

                // --------------------------------------------------------------
                // apply Dinv * vec on each color sub-vector

                auto start_Dinv_LU_tmie = std::chrono::high_resolution_clock::now();
                // use Dinv linear operator built from LU factor of D diag matrix to apply Dinv *
                // vec on each color (new method)
                T a = 1.0, b = 0.0;
                // note in this case d_diag_LU_vals_color refers to Dinv form of LU factors on each
                // nodal block
                CHECK_CUSPARSE(cusparseDbsrmv(  // NOTE just uses descrKmat cause would be the same
                                                // as descrDinv (convenience)
                    cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    nblock_rows_color, nblock_rows_color, diag_inv_nnzb_color, &a, descrKmat,
                    d_dinv_vals_color, d_diag_rowp, d_diag_cols, block_dim, d_defect_color, &b,
                    d_temp_color));

                // timing part
                if (time_debug) {
                    CHECK_CUDA(cudaDeviceSynchronize());
                    auto end_Dinv_LU_time = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> Dinv_LU_time =
                        end_Dinv_LU_time - start_Dinv_LU_tmie;
                    printf("\t\tDinv LU time on iter %d,color %d in %.2e sec\n", iter, icolor,
                           Dinv_LU_time.count());
                }

                // -----------------------------------------------------------------
                // color soln update => defect update for each color

                auto start_Bsrmv_time = std::chrono::high_resolution_clock::now();

                // 2) update soln x_color += dx_color
                T *d_soln_color = &d_soln.getPtr()[block_dim * start];
                a = omega;
                CHECK_CUBLAS(
                    cublasDaxpy(cublasHandle, nrows_color, &a, d_temp_color, 1, d_soln_color, 1));

                // // print submat vals here here..
                // T *h_submat_vals = new T[100];
                // cudaMemcpy(h_submat_vals, d_color_submat_vals[icolor], 100 * sizeof(T),
                // cudaMemcpyDeviceToHost); printf("h_submat_vals: "); printVec<T>(100,
                // h_submat_vals);

                // get submat size, to do submat-vec product A[:,color] * dx[color]
                int start_bcol = h_color_rowp[icolor], end_bcol = h_color_rowp[icolor + 1];
                int mb = nnodes, nb = end_bcol - start_bcol;
                int submat_nnzb = h_color_submat_nnzb[icolor];
                a = -omega, b = 1.0;  // so that defect := defect - mat*vec
                CHECK_CUSPARSE(cusparseDbsrmv(
                    cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, mb,
                    nb, submat_nnzb, &a, descrKmat, d_color_submat_vals[icolor],
                    d_color_submat_rowp[icolor], d_color_submat_cols[icolor], block_dim,
                    d_temp_color, &b, d_defect.getPtr()));

                if (time_debug) {
                    CHECK_CUDA(cudaDeviceSynchronize());
                    auto end_Bsrmv_time = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> bsrmv_time = end_Bsrmv_time - start_Bsrmv_time;
                    printf("\t\tbsrmv time on iter %d,color %d in %.2e sec\n", iter, icolor,
                           bsrmv_time.count());
                }

                // -------------------------------------------------------------------------------------
            }  // next color iteration

            /* report progress of defect nrm if printing.. */
            T defect_nrm;
            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_defect.getPtr(), 1, &defect_nrm));
            if (print && iter % print_freq == 0)
                printf("\tMC-BGS %d/%d : ||defect|| = %.4e\n", iter + 1, n_iters, defect_nrm);

            // --------------------------------------------------------------------------------
        }  // next block-GS iteration
    }

    void smoothMatrix(BsrMat<DeviceVec<T>> *prolong_mat, BsrMat<DeviceVec<T>> *Z_mat,
                      BsrMat<DeviceVec<T>> *Zprev_mat, int n_iters, int nnzb_prod,
                      int *d_P_prodBlocks, int *d_K_prodblocks, int *d_Z_prodblocks) {
        printf("WARNING: MC-Smoother doesn't smooth the matrix yet\n");
    }

    // data
    Assembler assembler;
    int N, nelems, block_dim, nnodes;
    BsrMat<DeviceVec<T>> Kmat, D_LU_mat;  // can't get Dinv_mat directly at moment
    DeviceVec<T> d_temp_vec, d_rhs_vec, d_inner_soln_vec;
    T *d_temp, *d_temp2, *d_resid;
    T *d_rhs, *d_inner_soln;
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
    int kmat_nnzb, *d_kmat_rowp, *d_kmat_cols;
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