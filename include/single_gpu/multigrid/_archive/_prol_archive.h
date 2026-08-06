void prolongate_debug(int *d_coarse_iperm, DeviceVec<T> coarse_soln_in,
                          std::string file_prefix = "", std::string file_suffix = "",
                          int n_smooth = 0, T y_offset = -1.5) {
        // call main prolongate
        if (n_smooth == 0) {
            prolongate(d_coarse_iperm, coarse_soln_in);
        } else {
            smoothed_prolongate(d_coarse_iperm, coarse_soln_in, n_smooth);
        }

        // DEBUG : write out the cf update, defect update and before and after defects
        auto h_cf_update = d_temp_vec.createPermuteVec(6, d_perm).createHostVec();
        T xpts_shift[3] = {0.0, y_offset, 1.5};
        printToVTKDEBUG<Assembler, HostVec<T>>(
            assembler, h_cf_update, file_prefix + "post2_cf_soln" + file_suffix, xpts_shift);

        auto h_cf_loads = DeviceVec<T>(N, d_temp2).createPermuteVec(6, d_perm).createHostVec();
        T xpts_shift2[3] = {0.0, y_offset, 3.0};
        printToVTKDEBUG<Assembler, HostVec<T>>(
            assembler, h_cf_loads, file_prefix + "post3_cf_loads" + file_suffix, xpts_shift2);

        auto h_defect2 = d_defect.createPermuteVec(6, d_perm).createHostVec();
        T xpts_shift3[3] = {0.0, y_offset, 4.5};
        printToVTKDEBUG<Assembler, HostVec<T>>(
            assembler, h_defect2, file_prefix + "post4_cf_fin_defect" + file_suffix, xpts_shift3);
    }

    void smoothed_prolongate(int *d_coarse_iperm, DeviceVec<T> coarse_soln_in, int n_smooth = 0) {
        // prolongate from coarser grid to this fine grid (with a smoothing step, more for
        // debugging) if really need this (like an AMG hybrid step here), you should smooth
        // prolongation matrix itself first..
        cudaMemset(d_temp, 0.0, N * sizeof(T));

        if constexpr (Prolongation::structured) {
            Prolongation::prolongate(nelems, d_coarse_iperm, d_iperm, coarse_soln_in, d_temp_vec,
                                     d_weights);
        } else {
            if constexpr (Prolongation::assembly) {
                // permute coarse soln in
                coarse_soln_in.permuteData(block_dim, d_coarse_iperm);
                Prolongation::prolongate(cusparseHandle, descrP, P_mat, coarse_soln_in, d_temp_vec);
            } else {
                // slower version
                Prolongation::prolongate(nnodes, d_coarse_conn, d_n2e_ptr, d_n2e_elems, d_n2e_xis,
                                         d_coarse_iperm, d_iperm, coarse_soln_in, d_temp_vec);
            }
        }
        // CHECK_CUDA(cudaDeviceSynchronize());

        // zero bcs of coarse-fine prolong
        d_temp_vec.permuteData(block_dim, d_perm);  // better way to do this later?
        assembler.apply_bcs(d_temp_vec);
        d_temp_vec.permuteData(block_dim, d_iperm);

        // copy out the old defect into resid temporarily
        CHECK_CUDA(cudaMemcpy(d_resid, d_defect.getPtr(), N * sizeof(T), cudaMemcpyDeviceToDevice));

        // compute contributing defect and copy in temporarily replacing d_defect
        T a = -1.0, b = 0.0;  // -K * d_temp + 0 * d_defect => d_defect
        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, nnodes, nnodes, kmat_nnzb,
                                      &a, descrKmat, d_kmat_vals, d_kmat_rowp, d_kmat_cols,
                                      block_dim, d_temp, &b, d_defect.getPtr()));
        CHECK_CUDA(cudaMemcpy(d_soln.getPtr(), d_temp, N * sizeof(T), cudaMemcpyDeviceToDevice));

        // before we coarse-fine rescale with 1DOF min energy the update.. let's do a smoothing of
        // the update
        multicolorBlockGaussSeidel_fast(n_smooth);

        // now copy back to d_temp and d_temp2 and old d_defect
        // CHECK_CUDA(cudaMemcpy(d_temp2, d_defect.getPtr(), N * sizeof(T),
        // cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(d_temp, d_soln.getPtr(), N * sizeof(T), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(d_defect.getPtr(), d_resid, N * sizeof(T), cudaMemcpyDeviceToDevice));

        // rescale coarse-fine using 1DOF min energy step
        // since FEA restrict and prolong operations are not energy minimally scaled
        // if u = u0 + omega * s, with s the proposed d_temp or du here (or line search)
        // then min energy omega from 1DOF galerkin is omega = <s, defect> / <s, Ks>
        // so need 2 dot prods, one SpMV, see 'multigrid/_python_demos/4_gmg_shell/1_mg.py' also
        T sT_defect;
        CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_defect.getPtr(), 1, d_temp, 1, &sT_defect));

        a = 1.0, b = 0.0;  // K * d_temp + 0 * d_temp2 => d_temp2
        CHECK_CUSPARSE(cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, nnodes, nnodes, kmat_nnzb,
                                      &a, descrKmat, d_kmat_vals, d_kmat_rowp, d_kmat_cols,
                                      block_dim, d_temp, &b, d_temp2));

        T sT_Ks;
        CHECK_CUBLAS(cublasDdot(cublasHandle, N, d_temp2, 1, d_temp, 1, &sT_Ks));
        T omega = sT_defect / sT_Ks;
        // if (debug) printf("omega = %.2e\n", omega);

        // now add coarse-fine dx into soln and update defect (with u = u0 + omega * d_temp)
        a = omega;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_temp, 1, d_soln.getPtr(), 1));
        a = -omega;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, N, &a, d_temp2, 1, d_defect.getPtr(), 1));
    }