// make new class for locking prolongation here (for plate/cylinder)

#pragma once

#include <set>
#include <unordered_set>

#include "multigrid/prolongation/_unstruct_utils.h"
#include "multigrid/prolongation/_unstructured.cuh"

template <class Assembler_, class Basis_, bool kmat_fillin = true>
class LockingAwareUnstructuredProlongation {
   public:
    using T = double;
    using Assembler = Assembler_;
    using Basis = Basis_;

    /*
    difference in this and the regular unstructured prolongation, is the sparsity pattern is A*P0
    insetad of just P0 (includes one fillin step) this way it will smooth better */

    static constexpr bool structured = true;
    static constexpr bool assembly = true;
    static constexpr bool is_bsr = true;  // uses full BSR matrix (no CSR)
    // static constexpr bool smoothed = true;
    // technically is smoothed but put false for now, so doesn't try to call missing smoothMatrix
    // API, TODO is fix this here
    static constexpr bool smoothed = false;

    LockingAwareUnstructuredProlongation(cusparseHandle_t &cusparseHandle_,
                                         Assembler &fine_assembler_, int ELEM_MAX_ = 10)
        : handle(cusparseHandle_) {
        fine_assembler = fine_assembler_;
        ELEM_MAX = ELEM_MAX_;
        has_init_coarse = false;

        // init some data from fine assembler, and other startup
        block_dim = fine_assembler.getBsrData().block_dim;
        h_fine_conn = fine_assembler.getConn().createHostVec().getPtr();

        // other startup
        descr_P = 0;
        CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_P));
        CHECK_CUSPARSE(cusparseSetMatType(descr_P, CUSPARSE_MATRIX_TYPE_GENERAL));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_P, CUSPARSE_INDEX_BASE_ZERO));
    }

    // nothing (though smoother needs to do matrix-smoothing in some cases)
    void update_after_assembly(DeviceVec<T> &d_vars) {
        // TODO : need to fix this for nonlinear smooth matrix
        // assemble_matrices(); // reassemble matrices?
    }

    void init_coarse_data(Assembler &coarse_assembler_) {
        // main call, TBD (see call below done customly for now)
    }

    void init_coarse_data_manual(Assembler &coarse_assembler_, int *h_fc_elem_map_) {
        // TBD : make this use main API above later..
        if (has_init_coarse) return;
        has_init_coarse = true;

        coarse_assembler = coarse_assembler_;
        h_fc_elem_map = h_fc_elem_map_;
        d_coarse_iperm = coarse_assembler.getBsrData().iperm;
        // printf("\tDEBUG: construct nz pattern\n");
        construct_nz_pattern();
        if constexpr (kmat_fillin) {
            // matrix fillin here P0 => A*P0
            // printf("\tDEBUG: apply kmat fillin\n");
            apply_kmat_fillin();
        }
        // printf("\tDEBUG: assemble matrices\n");
        assemble_matrices();

        if constexpr (kmat_fillin) {
            // allocate extra Z matrix storage and Z mat (for smoothing updates)
            auto d_Z_vec = DeviceVec<T>(P_nnzb * block_dim2);
            d_Z_vals = d_Z_vec.getPtr();
            Z_mat = new BsrMat<DeviceVec<T>>(P_bsr_data, d_Z_vec);

            auto d_RHS_vec = DeviceVec<T>(P_nnzb * block_dim2);
            d_RHS_vals = d_RHS_vec.getPtr();
            RHS_mat = new BsrMat<DeviceVec<T>>(P_bsr_data, d_RHS_vec);

            // printf("\tDEBUG: compute matmat nz pattern\n");
            compute_matmat_prod_nz_pattern();
        }
    }

    void construct_nz_pattern() {
        // call the utils method to get nz pattern and do any other steps
        prolong_mat = nullptr;
        BsrMat<DeviceVec<T>> *restrict_mat = nullptr;
        d_coarse_conn = nullptr, d_n2e_ptr = nullptr, d_n2e_elems = nullptr, d_n2e_xis = nullptr;
        const bool include_restrict = false;  // don't get PT matrix also
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

        // printf("P0 rowp: ");
        // printVec<int>(nnodes_fine + 1, h_P_rowp0);
        // printf("P0 cols: ");
        // printVec<int>(P_bsr_data.nnzb, h_P_cols0);

        // delete previous nofill P0 device pointers
        cudaFree(d_P_rowp);
        cudaFree(d_P_rows);
        cudaFree(d_P_cols);
        // ok cause this is only done on startup

        // build new sparsity
        // need permutations here?
        int *h_AP_rowp = new int[nnodes_fine + 1];
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
        int AP_nnzb = h_AP_rowp[nnodes_fine];
        int *h_AP_cols = new int[AP_nnzb];
        int *h_AP_rows = new int[AP_nnzb];
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

        // and free + allocate new matrix vals
        prolong_mat->free();  // frees up device values
        block_dim2 = block_dim * block_dim;
        P_nnzb = AP_nnzb;
        auto d_P_vals_vec = DeviceVec<T>(AP_nnzb * block_dim2);
        d_P_vals = d_P_vals_vec.getPtr();

        // and update the P_bsr_data also
        int *d_fine_perm = P_bsr_data.perm;
        // need to create on host for it to create elem ind map and other things
        bool host = true;
        auto h_P_bsr_data =
            BsrData(nnodes_fine, block_dim, AP_nnzb, h_AP_rowp, h_AP_cols, nullptr, nullptr, host);
        h_P_bsr_data.rows = h_AP_rows;
        // need to add fc_elem map and fine and coarse elem conn in order to get correct elem ind
        // maps
        h_P_bsr_data.nelems = fine_assembler.get_num_elements();
        h_P_bsr_data.elem_conn = h_fine_conn;
        h_P_bsr_data.cols_elem_conn = h_coarse_conn;
        h_P_bsr_data.rc_elem_map = h_fc_elem_map;
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

        // printf("make P bsr data for FC-matrix\n");
        P_bsr_data = h_P_bsr_data.createDeviceBsrData();
        // printf("\tdone make P bsr data for FC-matrix\n");

        // P_bsr_data = BsrData(nnodes_fine, block_dim, AP_nnzb, d_P_rowp, d_P_cols, d_fine_perm,
        //                      d_fine_iperm, false);
        P_bsr_data.mb = nnodes_fine, P_bsr_data.nb = nnodes_coarse;
        P_bsr_data.rows = d_P_rows;  // need rows
        prolong_mat = new BsrMat<DeviceVec<T>>(P_bsr_data, d_P_vals_vec);
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
    }

    void prolongate(DeviceVec<T> perm_coarse_soln_in, DeviceVec<T> perm_dx_fine) {
        // get important data & vecs out

        // now do cusparse Bsrmv.. for P @ coarse_soln => dx_fine (permuted nodes order)
        T a = 1.0, b = 0.0;
        int mb = P_bsr_data.mb, nb = P_bsr_data.nb;
        // printf("prolongate with block_dim %d, mb %d, nb %d, P_nnzb %d\n", block_dim, mb, nb,
        // P_nnzb);
        CHECK_CUSPARSE(cusparseDbsrmv(handle, CUSPARSE_DIRECTION_ROW,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, mb, nb, P_nnzb, &a, descr_P,
                                      d_P_vals, d_P_rowp, d_P_cols, block_dim,
                                      perm_coarse_soln_in.getPtr(), &b, perm_dx_fine.getPtr()));
    }

    template <bool normalize = false>
    void restrict_vec(DeviceVec<T> fine_vec_in, DeviceVec<T> coarse_vec_out) {
        /* either restricts defect or solution from fine to coarse */
        coarse_vec_out.zeroValues();  // zero before add new result
        int nprods = P_nnzb * block_dim2;
        dim3 block0(32), grid0((nprods + 31) / 32);
        k_bsrmv_transpose<T><<<grid0, block0>>>(P_nnzb, block_dim, d_P_rows, d_P_cols, d_P_vals,
                                                fine_vec_in.getPtr(), coarse_vec_out.getPtr());

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
    // BsrMat<DeviceVec<T>> *prolong_mat, *Z_mat, *Zprev_mat;
    BsrMat<DeviceVec<T>> *prolong_mat, *Z_mat, *RHS_mat;
    BsrMat<DeviceVec<T>> *Zprev_mat;  // unused Zprev now

    // pointers for mat-mat smoothing
    int *h_P_rowp, *h_P_cols;  // final fillin version A*P sparsity
    int *d_P_prodBlocks, *d_K_prodBlocks, *d_Z_prodBlocks;
    int nnzb_prod;

    // T *d_Zprev_vals;
    T *d_RHS_vals;

   private:
    Assembler fine_assembler, coarse_assembler;
    int ELEM_MAX;  // the max number of nearest neighbor elements for NZ pattern construction

    cusparseHandle_t &handle;
    BsrData P_bsr_data;
    T *d_P_vals, *d_Z_vals;
    cusparseMatDescr_t descr_P;
    int *d_P_rowp, *d_P_rows, *d_P_cols;
    int P_nnzb;
    T *d_coarse_weights;
    T *d_fine_ones;

    int *d_coarse_conn, *d_n2e_ptr, *d_n2e_elems;
    int *h_fc_elem_map, *h_coarse_conn, *h_fine_conn;
    T *d_n2e_xis;
    int *d_coarse_iperm, *d_fine_iperm;
    int nnodes_fine, block_dim, block_dim2, N_fine;
    int nnodes_coarse, N_coarse;

    bool has_init_coarse;
};