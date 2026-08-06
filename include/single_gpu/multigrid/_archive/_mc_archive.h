void multicolorBlockGaussSeidel_slow(int n_iters, bool print = false, int print_freq = 10,
                                         T omega = 1.0, bool symmetric = false) {
        // slower version of do multicolor BSRmat block gauss-seidel on the defect
        // slower in the sense that it uses full mat-vec and full triang solves (does work right)
        // would like a faster version with color slicing next..

        // T init_defect_nrm;
        // CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_defect.getPtr(), 1, &init_defect_nrm));
        // if (print) printf("Multicolor Block-GS init defect nrm = %.4e\n", init_defect_nrm);

        int num_colors = h_color_rowp.getSize() - 1;
        int *color_rowp = h_color_rowp.getPtr();

        for (int iter = 0; iter < n_iters; iter++) {
            for (int _icolor = 0; _icolor < num_colors; _icolor++) {
                int _icolor2 = (_icolor + iter) % num_colors;  // permute order as you go
                int icolor = symmetric ? num_colors - 1 - _icolor2 : _icolor2;

                // get active rows / cols for this color
                int start = color_rowp[icolor], end = color_rowp[icolor + 1];
                int nblock_rows_color = end - start;
                int nrows_color = nblock_rows_color * block_dim;
                T *d_defect_color = &d_defect.getPtr()[block_dim * start];
                cudaMemset(d_temp, 0.0, N * sizeof(T));  // holds dx_color
                T *d_temp_color = &d_temp[block_dim * start];
                T *d_temp_color2 = &d_temp2[block_dim * start];
                cudaMemset(d_temp2, 0.0, N * sizeof(T));  // DEBUG
                cudaMemcpy(d_temp_color2, d_defect_color, nblock_rows_color * block_dim * sizeof(T),
                           cudaMemcpyDeviceToDevice);

                T a = 1.0, b = 0.0;
                const double alpha = 1.0;
                CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                    cusparseHandle, dir, trans_L, nnodes, diag_inv_nnzb, &alpha, descr_L,
                    d_diag_LU_vals, d_diag_rowp, d_diag_cols, block_dim, info_L, d_temp2, d_resid,
                    policy_L, pBuffer));  // prob only need U^-1 part for block diag.. TBD

                CHECK_CUSPARSE(cusparseDbsrsv2_solve(cusparseHandle, dir, trans_U, nnodes,
                                                     diag_inv_nnzb, &alpha, descr_U, d_diag_LU_vals,
                                                     d_diag_rowp, d_diag_cols, block_dim, info_U,
                                                     d_resid, d_temp, policy_U, pBuffer));

                // 2) update soln x_color += dx_color
                T *d_soln_color = &d_soln.getPtr()[block_dim * start];
                a = omega;
                CHECK_CUBLAS(
                    cublasDaxpy(cublasHandle, nrows_color, &a, d_temp_color, 1, d_soln_color, 1));

                a = -omega,
                b = 1.0;  // so that defect := defect - mat*vec
                CHECK_CUSPARSE(cusparseDbsrmv(
                    cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    nnodes, nnodes, kmat_nnzb, &a, descrKmat, d_kmat_vals, d_kmat_rowp, d_kmat_cols,
                    block_dim, d_temp, &b, d_defect.getPtr()));

            }  // next color iteration

            // printf("iter %d, done with color iterations\n", iter);

            /* report progress of defect nrm if printing.. */
            T defect_nrm;
            CHECK_CUBLAS(cublasDnrm2(cublasHandle, N, d_defect.getPtr(), 1, &defect_nrm));
            if (print && iter % print_freq == 0)
                printf("\tMC-BGS %d/%d : ||defect|| = %.4e\n", iter + 1, n_iters, defect_nrm);

        }  // next block-GS iteration
    }

    void multicolorBlockGaussSeidel_fast(int n_iters, bool print = false, int print_freq = 10,
                                         T omega = 1.0, bool symmetric = false) {
        // faster version

        int num_colors = h_color_rowp.getSize() - 1;
        int *color_rowp = h_color_rowp.getPtr();
        // printf("mc BGS-fast with # colors = %d\n", num_colors);

        bool time_debug = false;
        // bool time_debug = true;
        if (time_debug) printf("\t\tncolors = %d, #iters %d MC-BGS\n", num_colors, n_iters);

        for (int iter = 0; iter < n_iters; iter++) {
            for (int _icolor = 0; _icolor < num_colors; _icolor++) {
                // -------------------------------------------------------------
                // prelim block (getting color sub-vectors ready)

                if (time_debug) CHECK_CUDA(cudaDeviceSynchronize());
                auto prelim_time = std::chrono::high_resolution_clock::now();

                int _icolor2 = (_icolor + iter) % num_colors;  // permute order as you go
                int icolor = symmetric ? num_colors - 1 - _icolor2 : _icolor2;

                // get active rows / cols for this color
                int start = color_rowp[icolor], end = color_rowp[icolor + 1];
                int nblock_rows_color = end - start;
                int nrows_color = nblock_rows_color * block_dim;
                T *d_defect_color = &d_defect.getPtr()[block_dim * start];
                cudaMemset(d_temp, 0.0, N * sizeof(T));  // holds dx_color
                T *d_temp_color = &d_temp[block_dim * start];
                int block_dim2 = block_dim * block_dim;
                int diag_inv_nnzb_color = nblock_rows_color;
                T *d_diag_LU_vals_color = &d_diag_LU_vals[start * block_dim2];

                if (time_debug) {
                    CHECK_CUDA(cudaDeviceSynchronize());
                    auto end_prelim_time = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> full_prelim_time = end_prelim_time - prelim_time;
                    printf("\t\tprelim time on iter %d,color %d in %.2e sec\n", iter, icolor,
                           full_prelim_time.count());
                }
                auto start_Dinv_LU_tmie = std::chrono::high_resolution_clock::now();

                // --------------------------------------------------------------
                // apply Dinv * vec on each color sub-vecotr
                // use LU triang solves to apply Dinv * vec on each color (old method)
                T a = 1.0, b = 0.0;
                const double alpha = 1.0;
                CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                    cusparseHandle, dir, trans_L, nblock_rows_color, diag_inv_nnzb_color, &alpha,
                    descr_L, d_diag_LU_vals_color, d_diag_rowp, d_diag_cols, block_dim, info_L,
                    d_defect_color, d_resid, policy_L,
                    pBuffer));  // prob only need U^-1 part for block diag.. TBD

                CHECK_CUSPARSE(cusparseDbsrsv2_solve(
                    cusparseHandle, dir, trans_U, nblock_rows_color, diag_inv_nnzb_color, &alpha,
                    descr_U, d_diag_LU_vals_color, d_diag_rowp, d_diag_cols, block_dim, info_U,
                    d_resid, d_temp_color, policy_U, pBuffer));

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

                a = -omega, b = 1.0;  // so that defect := defect - mat*vec
                // this does full mat-vec product, so much slower..
                CHECK_CUSPARSE(cusparseDbsrmv(
                    cusparseHandle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    nnodes, nnodes, kmat_nnzb, &a, descrKmat, d_kmat_vals, d_kmat_rowp, d_kmat_cols,
                    block_dim, d_temp, &b, d_defect.getPtr()));

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