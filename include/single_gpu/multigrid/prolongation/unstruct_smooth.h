#pragma once

#include <set>
#include <unordered_set>

#include "_unstruct_utils.h"
#include "_unstructured.cuh"

template <class Assembler_, class Basis_, class Smoother, bool KMAT_FILLIN = true,
          bool SEPARATE_PT_STORAGE = true>
class UnstructuredSmoothProlongation {
   public:
    using T = double;
    using Assembler = Assembler_;
    using Basis = Basis_;

    /*
    difference in this and the regular unstructured prolongation, is the sparsity pattern is A*P0
    insetad of just P0 (includes one fillin step) this way it will smooth better */

    static constexpr bool structured = false;
    static constexpr bool assembly = true;
    static constexpr bool is_bsr = true;  // uses full BSR matrix (no CSR)
    static constexpr bool smoothed = true;

    UnstructuredSmoothProlongation(cusparseHandle_t &cusparseHandle_, Assembler &fine_assembler_,
                                   Smoother *smoother_, int ELEM_MAX_ = 10, int nsmooth_iters_ = 2)
        : handle(cusparseHandle_) {
        fine_assembler = fine_assembler_;
        ELEM_MAX = ELEM_MAX_;
        smoother = smoother_;
        nsmooth_iters = nsmooth_iters_;

        // init some data from fine assembler, and other startup
        block_dim = fine_assembler.getBsrData().block_dim;
        h_fine_conn = fine_assembler.getConn().createHostVec().getPtr();

        // other startup
        descr_P = 0;
        CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_P));
        CHECK_CUSPARSE(cusparseSetMatType(descr_P, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_P, CUSPARSE_INDEX_BASE_ZERO));
        descr_PT = 0;
        CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_PT));
        CHECK_CUSPARSE(cusparseSetMatType(descr_PT, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_PT, CUSPARSE_INDEX_BASE_ZERO));
    }

    // nothing (though smoother needs to do matrix-smoothing in some cases)
    void update_after_assembly(DeviceVec<T> &vars) {
        assemble_matrices();  // reassemble matrices
        smoother->update_after_assembly(vars);
        smoothMatrix();
        update_after_smooth();
    }

    void smoothMatrix() {
        smoother->smoothMatrix(nsmooth_iters, prolong_mat, Z_mat, Zprev_mat, nnzb_prod,
                               d_K_prodBlocks, d_P_prodBlocks, d_Z_prodBlocks);
    }

    void init_coarse_data(Assembler &coarse_assembler_) {
        coarse_assembler = coarse_assembler_;
        d_coarse_iperm = coarse_assembler.getBsrData().iperm;
        // printf("\tDEBUG: construct nz pattern\n");
        construct_nz_pattern();
        // matrix fillin here P0 => A*P0
        // printf("\tDEBUG: apply kmat fillin\n");
        apply_kmat_fillin();
        // printf("\tDEBUG: assemble matrices\n");
        // printf("assemble matrices\n");
        assemble_matrices();

        d_fine_bcs = fine_assembler.getBCs();
        d_coarse_bcs = coarse_assembler.getBCs();

        // if constexpr (KMAT_FILLIN) {
        // allocate extra Z matrix storage and Z mat (for smoothing updates)
        auto d_Z_vec = DeviceVec<T>(P_nnzb * block_dim2);
        d_Z_vals = d_Z_vec.getPtr();
        Z_mat = new BsrMat<DeviceVec<T>>(P_bsr_data, d_Z_vec);

        // TODO : temporarily we use an extra Zprev_mat for matrix smoothing
        //   for less mem storage, could later remove this extra matrix and just do -Dinv*A*P
        //   into Z in one step
        auto d_Zprev_vec = DeviceVec<T>(P_nnzb * block_dim2);
        d_Zprev_vals = d_Zprev_vec.getPtr();
        Zprev_mat = new BsrMat<DeviceVec<T>>(P_bsr_data, d_Zprev_vec);

        // printf("\tDEBUG: compute matmat nz pattern\n");
        compute_matmat_prod_nz_pattern();
        // }

        // apply bcs to it now
        // printf("apply bcs\n");
        // const bool ones_on_diag = true;
        const bool ones_on_diag = false;  // just zero out completely for prolong matrix
        prolong_mat->template apply_bc_rows<ones_on_diag>(d_fine_bcs);
        prolong_mat->template apply_bc_cols<ones_on_diag>(d_coarse_bcs);
        // printf("\tapply bcs done\n");

        // printf("smooth matrix\n");
        smoothMatrix();
        // printf("unstruct smooth\n");
        update_after_smooth();
    }

    void construct_nz_pattern() {
        // call the utils method to get nz pattern and do any other steps
        prolong_mat = nullptr;
        BsrMat<DeviceVec<T>> *restrict_mat = nullptr;
        d_coarse_conn = nullptr, d_n2e_ptr = nullptr, d_n2e_elems = nullptr, d_n2e_xis = nullptr;
        const bool include_restrict = SEPARATE_PT_STORAGE;  // don't get PT matrix also if false
        init_unstructured_grid_maps<T, Assembler, Basis, is_bsr, include_restrict>(
            fine_assembler, coarse_assembler, prolong_mat, restrict_mat, d_coarse_conn, d_n2e_ptr,
            d_n2e_elems, d_n2e_xis, ELEM_MAX);

        h_coarse_conn = coarse_assembler.getConn().createHostVec().getPtr();

        // then get some data out of this (for more readable code later)
        P_bsr_data = prolong_mat->getBsrData();
        d_P_vals = prolong_mat->getPtr();
        d_P_rowp = P_bsr_data.rowp, d_P_rows = P_bsr_data.rows, d_P_cols = P_bsr_data.cols;
        P_nnzb = P_bsr_data.nnzb;
        nnodes_fine = P_bsr_data.nnodes;
        d_fine_iperm = P_bsr_data.iperm;

        if constexpr (SEPARATE_PT_STORAGE) {
            PT_bsr_data = restrict_mat->getBsrData();
            d_PT_vals = restrict_mat->getPtr();
            d_PT_rowp = PT_bsr_data.rowp, d_PT_rows = PT_bsr_data.rows,
            d_PT_cols = PT_bsr_data.cols;
            PT_nnzb = PT_bsr_data.nnzb;
            d_coarse_iperm = PT_bsr_data.iperm;
        }

        // optional d_weights
        N_fine = nnodes_fine * block_dim;
        nnodes_coarse = coarse_assembler.get_num_nodes();
        N_coarse = nnodes_coarse * block_dim;
        d_coarse_weights = DeviceVec<T>(N_coarse).getPtr();
        d_fine_ones = DeviceVec<T>(N_fine).getPtr();
        k_vec_set<T><<<(N_fine + 31) / 32, 32>>>(N_fine, 1.0, d_fine_ones);
    }

    void compute_matmat_prod_nz_pattern() {
        // get pointers
        auto kmat_bsr_data = fine_assembler.getBsrData();
        int *h_kmat_rowp =
            DeviceVec<int>(nnodes_fine + 1, kmat_bsr_data.rowp).createHostVec().getPtr();
        int *h_kmat_cols =
            DeviceVec<int>(kmat_bsr_data.nnzb, kmat_bsr_data.cols).createHostVec().getPtr();

        nnzb_prod = 0;
        for (int i = 0; i < nnodes_fine; i++) {
            for (int jp = h_P_rowp[i]; jp < h_P_rowp[i + 1]; jp++) {
                int j = h_P_cols[jp];  // (P_F)_{ij} output
                // now inner loop k for K_{ik} * P_{kj}
                for (int kp = h_kmat_rowp[i]; kp < h_kmat_rowp[i + 1]; kp++) {
                    int k = h_kmat_cols[kp];

                    // check P_{kj} nz
                    bool nz_Pkj = false;  // now also use PF = -K*P sparsity for P cause we add K*P
                                          // fillin (for better prolong)
                    for (int jp2 = h_P_rowp[k]; jp2 < h_P_rowp[k + 1]; jp2++) {
                        int j2 = h_P_cols[jp2];
                        if (j2 == j) {
                            nz_Pkj = true;
                        }
                    }
                    if (!nz_Pkj) continue;
                    // otherwise, we do have a valid nz product here
                    nnzb_prod++;
                }
            }
        }
        // now allocate the block indices of the product
        int *h_PF_blocks = new int[nnzb_prod];
        int *h_K_blocks = new int[nnzb_prod];
        int *h_P_blocks = new int[nnzb_prod];
        memset(h_PF_blocks, 0, nnzb_prod * sizeof(int));
        memset(h_K_blocks, 0, nnzb_prod * sizeof(int));
        memset(h_P_blocks, 0, nnzb_prod * sizeof(int));
        int inz_prod = 0;
        for (int i = 0; i < nnodes_fine; i++) {
            for (int jp = h_P_rowp[i]; jp < h_P_rowp[i + 1]; jp++) {
                int j = h_P_cols[jp];  // (P_F)_{ij} output
                // now inner loop k for K_{ik} * P_{kj}
                for (int kp = h_kmat_rowp[i]; kp < h_kmat_rowp[i + 1]; kp++) {
                    int k = h_kmat_cols[kp];

                    // check P_{kj} nz
                    bool nz_Pkj = false;
                    int _jp2 = -1;
                    for (int jp2 = h_P_rowp[k]; jp2 < h_P_rowp[k + 1]; jp2++) {
                        int j2 = h_P_cols[jp2];
                        if (j2 == j) {
                            nz_Pkj = true;
                            _jp2 = jp2;
                        }
                    }
                    if (!nz_Pkj) continue;
                    // otherwise, we do have a valid nz product here
                    h_PF_blocks[inz_prod] = jp;
                    h_K_blocks[inz_prod] = kp;
                    h_P_blocks[inz_prod] = _jp2;
                    inz_prod++;
                }
            }
        }
        // now allocate onto the device
        d_Z_prodBlocks = HostVec<int>(nnzb_prod, h_PF_blocks).createDeviceVec().getPtr();
        d_K_prodBlocks = HostVec<int>(nnzb_prod, h_K_blocks).createDeviceVec().getPtr();
        d_P_prodBlocks = HostVec<int>(nnzb_prod, h_P_blocks).createDeviceVec().getPtr();

        // printf("DEBUG: PF_nnzb = %d, nnzb_prod %d\n", P_nnzb, nnzb_prod);
    }

    void apply_kmat_fillin() {
        // get matrix filin P0 => A*P0, done on host and then update the device pointers (only once)

        // get host pointers (only once) of init sparsities
        auto kmat_bsr_data = fine_assembler.getBsrData();
        int *h_kmat_rowp =
            DeviceVec<int>(nnodes_fine + 1, kmat_bsr_data.rowp).createHostVec().getPtr();
        int *h_kmat_cols =
            DeviceVec<int>(kmat_bsr_data.nnzb, kmat_bsr_data.cols).createHostVec().getPtr();
        int h_kmat_nnzb = kmat_bsr_data.nnzb;
        int *h_P_rowp0 = DeviceVec<int>(nnodes_fine + 1, d_P_rowp).createHostVec().getPtr();
        int *h_P_cols0 = DeviceVec<int>(P_bsr_data.nnzb, d_P_cols).createHostVec().getPtr();

        int AP_nnzb, *h_AP_cols, *h_AP_rows, *h_AP_rowp;

        // build new sparsity
        // need permutations here?
        if constexpr (KMAT_FILLIN) {
            // delete previous nofill P0 device pointers
            cudaFree(d_P_rowp);
            cudaFree(d_P_rows);
            cudaFree(d_P_cols);
            // ok cause this is only done on startup

            h_AP_rowp = new int[nnodes_fine + 1];
            memset(h_AP_rowp, 0, (nnodes_fine + 1) * sizeof(int));
            for (int i = 0; i < nnodes_fine; i++) {
                h_AP_rowp[i + 1] = h_AP_rowp[i];
                std::unordered_set<int> unique_cols;
                for (int kp = h_kmat_rowp[i]; kp < h_kmat_rowp[i + 1]; kp++) {
                    int k = h_kmat_cols[kp];
                    for (int jp = h_P_rowp0[k]; jp < h_P_rowp0[k + 1]; jp++) {
                        int j = h_P_cols0[jp];
                        unique_cols.insert(j);
                    }
                }
                h_AP_rowp[i + 1] += unique_cols.size();
            }
            // printVec<int>(50, PF_rowp);
            AP_nnzb = h_AP_rowp[nnodes_fine];
            h_AP_cols = new int[AP_nnzb];
            h_AP_rows = new int[AP_nnzb];
            int iinz = 0;
            for (int i = 0; i < nnodes_fine; i++) {
                std::set<int> ordered_unique_cols;
                for (int kp = h_kmat_rowp[i]; kp < h_kmat_rowp[i + 1]; kp++) {
                    int k = h_kmat_cols[kp];
                    for (int jp = h_P_rowp0[k]; jp < h_P_rowp0[k + 1]; jp++) {
                        int j = h_P_cols0[jp];
                        ordered_unique_cols.insert(j);
                    }
                }
                for (int col : ordered_unique_cols) {
                    h_AP_cols[iinz] = col;
                    h_AP_rows[iinz] = i;
                    iinz++;
                }
            }

            // copy final fillin sparsity to host (for use in mat-mat prod sparsity)
            h_P_rowp = h_AP_rowp;
            h_P_cols = h_AP_cols;

            // now store the new fillin P mat sparsity
            d_P_rowp = HostVec<int>(nnodes_fine + 1, h_AP_rowp).createDeviceVec().getPtr();
            d_P_rows = HostVec<int>(AP_nnzb, h_AP_rows).createDeviceVec().getPtr();
            d_P_cols = HostVec<int>(AP_nnzb, h_AP_cols).createDeviceVec().getPtr();
        } else {
            cudaFree(d_P_rowp);
            cudaFree(d_P_rows);
            cudaFree(d_P_cols);

            // no kmat fillin
            h_AP_rowp = new int[nnodes_fine + 1];
            for (int i = 0; i < nnodes_fine + 1; i++) {
                h_AP_rowp[i] = h_P_rowp0[i];
            }
            AP_nnzb = h_P_rowp0[nnodes_fine];

            // printf("AP_nnzb %d\n", AP_nnzb);
            h_AP_cols = new int[AP_nnzb];
            h_AP_rows = new int[AP_nnzb];
            for (int i = 0; i < nnodes_fine; i++) {
                for (int jp = h_P_rowp0[i]; jp < h_P_rowp0[i + 1]; jp++) {
                    h_AP_rows[jp] = i;
                    h_AP_cols[jp] = h_P_cols0[jp];
                }
            }
            h_P_rowp = h_AP_rowp;
            h_P_cols = h_AP_cols;

            d_P_rowp = HostVec<int>(nnodes_fine + 1, h_AP_rowp).createDeviceVec().getPtr();
            d_P_rows = HostVec<int>(AP_nnzb, h_AP_rows).createDeviceVec().getPtr();
            d_P_cols = HostVec<int>(AP_nnzb, h_AP_cols).createDeviceVec().getPtr();
        }

        // and free + allocate new matrix vals
        prolong_mat->free();  // frees up device values
        block_dim2 = block_dim * block_dim;
        P_nnzb = AP_nnzb;
        auto d_P_vals_vec = DeviceVec<T>(AP_nnzb * block_dim2);
        d_P_vals = d_P_vals_vec.getPtr();

        // and update the P_bsr_data also
        int *d_fine_perm = P_bsr_data.perm;

        bool host = true;
        auto h_P_bsr_data =
            BsrData(nnodes_fine, block_dim, AP_nnzb, h_AP_rowp, h_AP_cols, nullptr, nullptr, host);
        h_P_bsr_data.rows = h_AP_rows;
        // need to add fc_elem map and fine and coarse elem conn in order to get correct elem ind
        // maps
        h_P_bsr_data.nelems = fine_assembler.get_num_elements();
        h_P_bsr_data.elem_conn = h_fine_conn;
        h_P_bsr_data.cols_elem_conn = h_coarse_conn;
        // h_P_bsr_data.rc_elem_map = h_fc_elem_map;
        h_P_bsr_data.rc_elem_map = nullptr;
        int nodes_per_elem = 4;
        h_P_bsr_data.nodes_per_elem = nodes_per_elem;
        h_P_bsr_data.n_eim = h_P_bsr_data.nelems * nodes_per_elem * nodes_per_elem;

        auto c_bsr_data = coarse_assembler.getBsrData();
        int c_nnodes = coarse_assembler.get_num_nodes();
        int *h_cperm = DeviceVec<int>(c_nnodes, c_bsr_data.perm).createHostVec().getPtr();
        int *h_ciperm = DeviceVec<int>(c_nnodes, c_bsr_data.iperm).createHostVec().getPtr();
        h_P_bsr_data.c_perm = h_cperm;
        h_P_bsr_data.c_iperm = h_ciperm;
        h_P_bsr_data.mb = nnodes_fine;
        h_P_bsr_data.nb = nnodes_coarse;

        P_bsr_data = h_P_bsr_data.createDeviceBsrData();
        // printf("\tdone make P bsr data for FC-matrix\n");
        P_bsr_data.mb = nnodes_fine, P_bsr_data.nb = nnodes_coarse;
        P_bsr_data.rows = d_P_rows;  // need rows

        prolong_mat = new BsrMat<DeviceVec<T>>(P_bsr_data, d_P_vals_vec);

        if constexpr (SEPARATE_PT_STORAGE) {
            // get transpose maps here..
            cudaFree(d_PT_vals);
            PT_nnzb = P_nnzb;
            d_PT_vals = DeviceVec<T>(AP_nnzb * block_dim2).getPtr();
            d_PT_rowp = P_bsr_data.tr_rowp;
            d_PT_cols = P_bsr_data.tr_cols;
            // d_PT_rows = P_bsr_data.tr_rows;
        }
    }

    void assemble_matrices() {
        /* assemble the P matrix values from baseline interp */

        // assemble P mat
        cudaMemset(d_P_vals, 0.0, P_nnzb * block_dim2 * sizeof(T));
        dim3 block(32);
        dim3 grid((nnodes_fine + 31) / 32);
        k_prolong_mat_assembly<T, Basis, is_bsr>
            <<<grid, block>>>(d_coarse_iperm, d_coarse_conn, d_n2e_ptr, d_n2e_elems, d_n2e_xis,
                              nnodes_fine, d_fine_iperm, d_P_rowp, d_P_cols, block_dim, d_P_vals);

        // assemble PT mat
        // if constexpr (SEPARATE_PT_STORAGE) {
        //     k_restrict_mat_assembly<T, Basis, is_bsr><<<grid, block>>>(
        //         d_coarse_iperm, d_coarse_conn, d_n2e_ptr, d_n2e_elems, d_n2e_xis, nnodes_fine,
        //         d_fine_iperm, d_PT_rowp, d_PT_cols, red_block_dim, d_PT_vals, d_coarse_weights);
        // }
    }

    void update_after_smooth() {
        // compute coarse weights for nonlinear problems as P^T * ones => weights (row-sums of P^T
        // coarse nodes)
        cudaMemset(d_coarse_weights, 0.0, N_coarse * sizeof(T));
        int nprods = P_nnzb * block_dim2;
        dim3 block0(32), grid0((nprods + 31) / 32);
        // computes row-sums
        k_bsrmv_transpose<T><<<grid0, block0>>>(P_nnzb, block_dim, d_P_rows, d_P_cols, d_P_vals,
                                                d_fine_ones, d_coarse_weights);

        if constexpr (SEPARATE_PT_STORAGE) {
            // copy values from P to PT
            int *d_block_P_to_PT_map = P_bsr_data.tr_block_map;
            k_copy_P_to_PT<T>
                <<<grid0, block0>>>(P_nnzb, block_dim, d_block_P_to_PT_map, d_P_vals, d_PT_vals);

            int *h_block_P_to_PT_map =
                DeviceVec<int>(P_nnzb, d_block_P_to_PT_map).createHostVec().getPtr();
            printf("h_block_P_to_PT_map: ");
            printVec<int>(100, h_block_P_to_PT_map);
        }
    }

    void prolongate(DeviceVec<T> perm_coarse_soln_in, DeviceVec<T> perm_dx_fine) {
        // get important data & vecs out
        // printf("DEBUG: run prolong\n");

        // now do cusparse Bsrmv.. for P @ coarse_soln => dx_fine (permuted nodes order)
        T a = 1.0, b = 0.0;
        int mb = P_bsr_data.mb, nb = P_bsr_data.nb;
        // printf("prolongate with block_dim %d, mb %d, nb %d, P_nnzb %d\n", block_dim, mb, nb,
        // P_nnzb);
        CHECK_CUSPARSE(cusparseDbsrmv(handle, CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, mb, nb, P_nnzb, &a, descr_P,
                                      d_P_vals, d_P_rowp, d_P_cols, block_dim,
                                      perm_coarse_soln_in.getPtr(), &b, perm_dx_fine.getPtr()));

        // CHECK_CUDA(cudaDeviceSynchronize());
        // printf("\tDEBUG: done prolong\n");
    }

    template <bool normalize = false>
    void restrict_vec(DeviceVec<T> fine_vec_in, DeviceVec<T> coarse_vec_out) {
        /* either restricts defect or solution from fine to coarse */
        // printf("\tDEBUG: begin restrict with P_nnzb %d, PT_nnzb %d\n", P_nnzb, PT_nnzb);
        if constexpr (SEPARATE_PT_STORAGE) {
            T a = 1.0, b = 0.0;
            int mb = PT_bsr_data.mb, nb = PT_bsr_data.nb;
            // printf("mb %d, nb %d\n", mb, nb);
            CHECK_CUSPARSE(cusparseDbsrmv(handle, CUSPARSE_DIRECTION_ROW,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE, mb, nb, PT_nnzb, &a,
                                          descr_PT, d_PT_vals, d_PT_rowp, d_PT_cols, block_dim,
                                          fine_vec_in.getPtr(), &b, coarse_vec_out.getPtr()));
        } else {
            coarse_vec_out.zeroValues();  // zero before add new result
            int nprods = P_nnzb * block_dim2;
            dim3 block0(32), grid0((nprods + 31) / 32);
            // my own custom kernel here
            k_bsrmv_transpose<T><<<grid0, block0>>>(P_nnzb, block_dim, d_P_rows, d_P_cols, d_P_vals,
                                                    fine_vec_in.getPtr(), coarse_vec_out.getPtr());
        }

        // CHECK_CUDA(cudaDeviceSynchronize());
        // printf("\tDEBUG: done restrict\n");

        // NORMALIZE section, only for restricting the solution (not defects)
        if constexpr (normalize) {
            // now divide the coarse vec by coarse weights to normalize
            dim3 block(32);
            int nblock = (N_coarse + 31) / 32;
            dim3 grid(nblock);
            k_vec_normalize2<T>
                <<<grid, block>>>(N_coarse, coarse_vec_out.getPtr(), d_coarse_weights);
        }
    }

    // public
    // Zmat is temp matrix for smoothing
    BsrMat<DeviceVec<T>> *prolong_mat, *Z_mat, *Zprev_mat;

    // pointers for mat-mat smoothing
    int *h_P_rowp, *h_P_cols;  // final fillin version A*P sparsity
    int *d_P_prodBlocks, *d_K_prodBlocks, *d_Z_prodBlocks;
    int nnzb_prod;

    T *d_Zprev_vals;

   private:
    Assembler fine_assembler, coarse_assembler;
    int ELEM_MAX;  // the max number of nearest neighbor elements for NZ pattern construction
    Smoother *smoother;

    cusparseHandle_t &handle;
    BsrData P_bsr_data, PT_bsr_data;
    T *d_P_vals, *d_Z_vals, *d_PT_vals;
    cusparseMatDescr_t descr_P, descr_PT;
    int *d_P_rowp, *d_P_rows, *d_P_cols;
    int *d_PT_rowp, *d_PT_rows, *d_PT_cols;
    int P_nnzb, PT_nnzb;
    T *d_coarse_weights;
    T *d_fine_ones;

    int *d_coarse_conn, *d_n2e_ptr, *d_n2e_elems;
    T *d_n2e_xis;
    int *d_coarse_iperm, *d_fine_iperm;
    int nnodes_fine, block_dim, block_dim2, N_fine;
    int nnodes_coarse, N_coarse;
    DeviceVec<int> d_coarse_bcs, d_fine_bcs;
    int nsmooth_iters;
    int *h_coarse_conn, *h_fine_conn;
};