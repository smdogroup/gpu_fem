#pragma once
#include <algorithm>
#include <random>
#include <vector>

#include "../utils.h"
#include "chrono"
#include "cuda_utils.h"
#include "reordering_utils.h"
#include "stdlib.h"
#include "vec.h"

typedef int index_t;  // for sparse utils

// dependency to smdogroup/sparse-utils repo
#include "sparse_utils/sparse_symbolic.h"
#include "sparse_utils/sparse_utils.h"

class BsrData {
    /* holds bsr sparsity pattern */
   public:
    BsrData() = default;
    __HOST__ BsrData(const int nelems, const int nnodes, const int &nodes_per_elem,
                     const int &block_dim, const int32_t *conn)
        : nelems(nelems),
          nnodes(nnodes),
          nodes_per_elem(nodes_per_elem),
          elem_conn(conn),
          block_dim(block_dim),
          elem_ind_map(nullptr),
          tr_rowp(nullptr),
          tr_cols(nullptr),
          tr_block_map(nullptr),
          host(true) {
        _get_row_col_ptrs_sparse();

        // make a nominal ordering (no permutation)
        n_eim = nelems * nodes_per_elem * nodes_per_elem;
        perm = new int32_t[nnodes];
        iperm = new int32_t[nnodes];
        for (int inode = 0; inode < nnodes; inode++) {
            perm[inode] = inode;
            iperm[inode] = inode;
        }
    }

    __HOST__ BsrData(const int mb, const int block_dim, const int nnzb, index_t *rowp,
                     index_t *cols, int *perm = nullptr, int *iperm = nullptr, bool host = true)
        : nnzb(nnzb),
          nnodes(mb),
          nodes_per_elem(nodes_per_elem),
          rowp(rowp),
          cols(cols),
          elem_conn(nullptr),
          block_dim(block_dim),
          elem_ind_map(nullptr),
          tr_rowp(nullptr),
          tr_cols(nullptr),
          tr_block_map(nullptr),
          host(host) {
        if (!perm || !iperm) {
            this->perm = new int32_t[nnodes];
            this->iperm = new int32_t[nnodes];
            for (int inode = 0; inode < nnodes; inode++) {
                this->perm[inode] = inode;
                this->iperm[inode] = inode;
            }
        } else {
            this->perm = perm;
            this->iperm = iperm;
        }
    }

    /* length of values array */
    __HOST_DEVICE__ int32_t getNumValues() const { return nnzb * block_dim * block_dim; }

    __HOST__ void _get_row_col_ptrs_sparse() {
        /* get the rowp, cols from the element connectivity using sparse utils */
        auto su_mat = SparseUtils::BSRMatFromConnectivityCUDA<double, 1>(nelems, nnodes,
                                                                         nodes_per_elem, elem_conn);
        nnzb = su_mat->nnz;
        rowp = su_mat->rowp;
        cols = su_mat->cols;
    }

    __HOST__ static int getBandWidth(const int &nnodes, const int &nnzb, int *rowp, int *cols) {
        /* compute bandwidth for reordering and sparsity pattern changes (esp. q-ordering)
            bandwidth = max_{a_ij neq 0} |i - j|   ( so max off-diagonal distance
            within sparsity pattern) */
        int bandwidth = 0;
        for (int i = 0; i < nnodes; i++) {
            for (int jp = rowp[i]; jp < rowp[i + 1]; jp++) {
                int j = cols[jp];
                int diff = abs(i - j);
                if (diff > bandwidth) {
                    bandwidth = diff;
                }
            }
        }
        return bandwidth;
    }

    SparseUtils::BSRMat<double, 1, 1> getSparseUtilsMat() {
        // deep copy rowp, cols
        int *c_rowp = HostVec<int>(nnodes + 1, rowp).copyVec().getPtr();
        int *c_cols = HostVec<int>(nnzb, cols).copyVec().getPtr();

        auto A = SparseUtils::BSRMat<double, 1, 1>(nnodes, nnodes, nnzb, rowp, cols, nullptr);

        int *iperm_copy = HostVec<int>(nnodes, iperm).copyVec().getPtr();
        int *perm_copy = HostVec<int>(nnodes, perm).copyVec().getPtr();
        A.perm = perm_copy;  // don't swap
        A.iperm = iperm_copy;
        return A;
    }

    __HOST__ void compute_nofill_pattern() {
        /* compute the rowp, cols after a permutation or reordering with no fillin */

        auto A = getSparseUtilsMat();
        auto A2 = SparseUtils::BSRMatApplyPerm<double, 1>(A, perm, iperm);
        // auto A2 = SparseUtils::BSRMatApplyPerm<double, 1>(A, iperm, perm);
        // if (rowp) delete[] rowp;
        // if (cols) delete[] cols;

        // get permuted no-fill pattern for Kmat
        nnzb = A2->rowp[nnodes];
        rowp = A2->rowp;
        cols = A2->cols;
    }

    __HOST__ void compute_full_LU_pattern(double fill_factor = 10.0, bool print = false) {
        /* compute the full LU pattern with a permutation */
        auto start = std::chrono::high_resolution_clock::now();
        if (print) {
            printf("begin full LU symbolic factorization::\n");
            printf("\tnnzb = %d\n", nnzb);
        }
        auto su_mat = getSparseUtilsMat();

        auto su_mat2 = SparseUtils::BSRMatReorderSymbolicCUDA<double, 1>(su_mat, fill_factor);
        // SparseUtils::BSRMatReorderFactorSymbolic<double, 1>(su_mat, perm, fill_factor);
        // SparseUtils::BSRMatReorderFactorSymbolic<double, 1>(su_mat, iperm, fill_factor);

        nnzb = su_mat2->nnz;
        rowp = su_mat2->rowp;
        cols = su_mat2->cols;

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        double dt = duration.count() / 1e6;
        if (print) {
            printf("\tfinished full LU symbolic factorization in %.4e sec\n", dt);
        }
    }

    __HOST__ void compute_ILUk_pattern(int levFill, double fill_factor = 10.0, bool print = true) {
        /* compute the full ILU(k) fill pattern including reorderings, levFill is k.
            TACS also returns a **_levs pointer (should we do that here too?) */

        compute_nofill_pattern();

        int *levels;
        computeILUk(nnodes, nnzb, rowp, cols, levFill, fill_factor, &levels, print);
        nnzb = rowp[nnodes];
    }

    __HOST__ void AMD_reordering() {
        /* computes the sparsity pattern for a full LU for direct LU solve
           with AMD reordering, consider making enum for reordering types here? */

        // compute fillin
        double fill_factor = 5.0;
        auto su_mat = SparseUtils::BSRMat<double, 1, 1>(nnodes, nnodes, nnzb, rowp, cols, nullptr);
        auto su_mat2 = BSRMatAMDFactorSymbolicCUDA(su_mat, fill_factor);
        // printf("done with BSRMatAMDFactorSymbolicCUDA\n");

        // TODO : not the most efficient since it includes factorization in above too (can fix later
        // if need be) perm = su_mat2->perm; iperm = su_mat2->iperm;
        perm = su_mat2->perm;
        iperm = su_mat2->iperm;
        // perm = su_mat2->iperm;  // my reordering definition is flipped from sparse utils
        // iperm = su_mat2->perm;
    }

    __HOST__ void RCM_reordering(int num_rcm_iters = 1) {
        /* reverse cuthill McKee reordering to reduce matrix bandwidth and
           # nz in the sparsity pattern */

        int *_new_perm = new int[nnodes];
        // int num_rcm_iters = 3;  // don't think we need to change this
        int root_node = 0;
        TacsComputeRCMOrder(nnodes, rowp, cols, _new_perm, root_node, num_rcm_iters);
        // flip the new permutation
        for (int k = 0; k < nnodes; k++) {
            perm[_new_perm[k]] = k;  // flipped from TACS
            iperm[k] = _new_perm[k];
        }
        if (_new_perm) delete[] _new_perm;
    }

    __HOST__ void qorder_reordering(double p_factor, int rcm_iters = 5, bool print = true) {
        /*  qordering combines RCM reordering to reduce bandwidth with random reordering
                to reduce chain lengths in ILU factorization for more stable ILU decomp numerically
                this also should improve GMRES convergence
            #rows for random = 1/pfactor * bandwidth so lower pfactor is more random
        */

        // keep orig rowp, cols
        int *orig_rowp = new int[nnodes + 1];
        int *orig_cols = new int[nnzb];
        for (int i = 0; i < nnodes + 1; i++) {
            orig_rowp[i] = rowp[i];
        }
        for (int i = 0; i < nnzb; i++) {
            orig_cols[i] = cols[i];
        }

        // first we perform RCM reordering to lower bandwidth
        int bandwidth_0 = getBandWidth(nnodes, nnzb, rowp, cols);
        RCM_reordering(rcm_iters);
        compute_nofill_pattern();
        int bandwidth_1 = getBandWidth(nnodes, nnzb, rowp, cols);
        if (print)
            printf("prelim RCM reordering reduces bandwidth from %d to %d\n", bandwidth_0,
                   bandwidth_1);

        // then we perform random reordering to reduce chain lengths
        int prune_width = (int)(1.0 / p_factor * bandwidth_1);
        if (print)
            printf("qordering with init bandwidth %d and prune width %d\n", bandwidth_1,
                   prune_width);
        int num_prunes = (nnodes + prune_width - 1) / prune_width;
        std::random_device rd;  // random number generator
        std::mt19937 g(rd());
        // since iperm is used for sparsity change now, qperm modifies that
        std::vector<int> q_perm(perm, perm + nnodes);
        for (int iprune = 0; iprune < num_prunes; iprune++) {
            int lower = prune_width * iprune;
            int upper = std::min(lower + prune_width, nnodes);
            std::shuffle(q_perm.begin() + lower, q_perm.begin() + upper, g);
        }

        // also reset the rowp, cols to original (was changed after reordering computation to check
        // bandwidth)
        if (rowp) delete[] rowp;
        if (cols) delete[] cols;
        rowp = orig_rowp;
        cols = orig_cols;

        // update final permutation and iperm (deep copy)
        for (int i = 0; i < nnodes; i++) {
            perm[i] = q_perm[i];
            iperm[q_perm[i]] = i;

            // try swapping perm, iperm
            // iperm[i] = q_perm[i];
            // perm[q_perm[i]] = i;
        }
    }

    __HOST__ void _compute_symbolic_maps_for_gpu() {
        /* computes symbolic maps such as elem_ind_map and transpose mappings
           for efficient kernel processing on GPU */

        if (elem_ind_map) delete[] elem_ind_map;
        if (tr_rowp) delete[] tr_rowp;
        if (tr_cols) delete[] tr_cols;
        if (tr_block_map) delete[] tr_block_map;

        // printf("before elem ind map\n");

        // 1) compute the elem ind map -------------
        /* elem_ind_map is nelems x 4 x 4 array for nodes_per_elem = 4
           it's useful in telling how to add each block matrix to global

           we use iperm map : old row => new row here since the connectivity is on
           unreordered nodes and the rowp, cols are for a reordered matrix
           for more details on reordering, see tests/reordering/README.md
           */
        int nodes_per_elem2 = nodes_per_elem * nodes_per_elem;
        elem_ind_map = new index_t[nelems * nodes_per_elem2]();
        for (int ielem = 0; ielem < nelems; ielem++) {
            const int32_t *local_conn = &elem_conn[nodes_per_elem * ielem];
            for (int block_row = 0; block_row < nodes_per_elem; block_row++) {
                int32_t _global_block_row = local_conn[block_row];
                // iperm map : old row to new row
                int32_t global_block_row = iperm[_global_block_row];
                int col_istart = rowp[global_block_row];
                int col_iend = rowp[global_block_row + 1];
                for (int block_col = 0; block_col < nodes_per_elem; block_col++) {
                    int32_t _global_block_col = local_conn[block_col];
                    // iperm map : old col to new col
                    int32_t global_block_col = iperm[_global_block_col];
                    // get matching ind in cols for the global_block_col
                    for (int i = col_istart; i < col_iend; i++) {
                        if (cols[i] == global_block_col) {
                            // add this component of block kelem matrix into
                            // elem_ind_map
                            int block_ind = nodes_per_elem * block_row + block_col;
                            // if (i == 0) {
                            //     printf(
                            //         "elem %d, glob_ind %d, elem_block_ind %d, glob_block_row %d,
                            //         " "glob_block_col %d\n", ielem, i, block_ind,
                            //         global_block_row, global_block_col);
                            // }
                            elem_ind_map[nodes_per_elem2 * ielem + block_ind] = i;
                            break;
                        }
                    }
                }
            }
        }

        // printf("elem_ind_map2:");
        // printVec<int>(nelems * nodes_per_elem2, elem_ind_map);

        // 2) compute the transpose matrix maps --------------
        /* difficult to apply bcs to cols BSR with column-major format, but easy to apply to rows
           so compute here the the sparsity maps of the transpose matrix */

        // get transpose rowp, cols of BSR matrix
        auto su_mat = SparseUtils::BSRMat<double, 1, 1>(nnodes, nnodes, nnzb, rowp, cols, nullptr);
        auto su_mat_transpose = SparseUtils::BSRMatMakeTransposeSymbolic(su_mat);
        tr_rowp = su_mat_transpose->rowp;
        tr_cols = su_mat_transpose->cols;

        // also compute a transpose block map map
        // from transpose_block_ind => orig_block_ind (of the values arrays)
        tr_block_map = new int32_t[nnzb];
        int32_t *temp_col_local_block_ind_ctr = new int32_t[nnodes];
        memset(temp_col_local_block_ind_ctr, 0, nnodes * sizeof(int32_t));
        for (int block_row = 0; block_row < nnodes; block_row++) {
            for (int orig_block_ind = rowp[block_row]; orig_block_ind < rowp[block_row + 1];
                 orig_block_ind++) {
                int32_t block_col = cols[orig_block_ind];
                int32_t orig_val_ind = orig_block_ind;
                int32_t transpose_val_ind =
                    tr_rowp[block_col] + temp_col_local_block_ind_ctr[block_col];
                temp_col_local_block_ind_ctr[block_col]++;  // increment number of block cols in
                                                            // this col
                tr_block_map[transpose_val_ind] = orig_val_ind;
            }
        }

        delete[] temp_col_local_block_ind_ctr;
    }

    __HOST__ BsrData createHostBsrData() {
        /* create new BsrData object with all Device sparsity data copied to the Host */
        BsrData new_bsr;  // bypass main constructor so we don't end up making host data
        new_bsr.nnzb = this->nnzb;
        new_bsr.nodes_per_elem = this->nodes_per_elem;
        new_bsr.block_dim = this->block_dim;
        new_bsr.nelems = this->nelems;
        new_bsr.nnodes = this->nnodes;
        new_bsr.host = true;

        int *h_elem_conn = nullptr;
#ifdef USE_GPU
        h_elem_conn = new int[nodes_per_elem * nelems];
        CHECK_CUDA(cudaMemcpy(h_elem_conn, elem_conn, nodes_per_elem * nelems * sizeof(int),
                              cudaMemcpyDeviceToHost));
#endif

        // create HostVec wrapper objects in CUDA, transfer to device and get ptr for new object
        DeviceVec<int> d_rowp(nnodes + 1, rowp), d_cols(nnzb, cols), d_perm(nnodes, perm),
            d_iperm(nnodes, iperm), d_elem_ind_map(n_eim, elem_ind_map),
            d_tr_rowp(nnodes + 1, tr_rowp), d_tr_cols(nnzb, tr_cols),
            d_tr_block_map(nnzb, tr_block_map);
        new_bsr.rowp = d_rowp.createHostVec().getPtr();
        new_bsr.cols = d_cols.createHostVec().getPtr();
        new_bsr.perm = d_perm.createHostVec().getPtr();
        new_bsr.iperm = d_iperm.createHostVec().getPtr();
        new_bsr.elem_ind_map = d_elem_ind_map.createHostVec().getPtr();
        new_bsr.tr_rowp = d_tr_rowp.createHostVec().getPtr();
        new_bsr.tr_cols = d_tr_cols.createHostVec().getPtr();
        new_bsr.tr_block_map = d_tr_block_map.createHostVec().getPtr();
        new_bsr.elem_conn = h_elem_conn;
        new_bsr.nelems = nelems;
        new_bsr.nodes_per_elem = nodes_per_elem;
        new_bsr.n_eim = n_eim;

        return new_bsr;
    }

    __HOST__ BsrData createDeviceBsrData() {
        /* create new BsrData object with all host sparsity data copied to the device */

        // before transferring to device, compute the symbolic maps for the GPU
        // call here instead of at the end of each symbolic factorization change
        _compute_symbolic_maps_for_gpu();

        int *d_elem_conn = nullptr;
#ifdef USE_GPU
        CHECK_CUDA(cudaMalloc((void **)&d_elem_conn, nodes_per_elem * nelems * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(d_elem_conn, elem_conn, nodes_per_elem * nelems * sizeof(int),
                              cudaMemcpyHostToDevice));
#endif

        BsrData new_bsr;  // bypass main constructor so we don't end up making host data
        new_bsr.nnzb = this->nnzb;
        new_bsr.nodes_per_elem = this->nodes_per_elem;
        new_bsr.block_dim = this->block_dim;
        new_bsr.nelems = this->nelems;
        new_bsr.nnodes = this->nnodes;
        new_bsr.host = false;

        // create HostVec wrapper objects in CUDA, transfer to device and get ptr for new object
        HostVec<int> h_rowp(nnodes + 1, rowp), h_cols(nnzb, cols), h_perm(nnodes, perm),
            h_iperm(nnodes, iperm), h_elem_ind_map(n_eim, elem_ind_map),
            h_tr_rowp(nnodes + 1, tr_rowp), h_tr_cols(nnzb, tr_cols),
            h_tr_block_map(nnzb, tr_block_map);
        new_bsr.rowp = h_rowp.createDeviceVec().getPtr();
        new_bsr.cols = h_cols.createDeviceVec().getPtr();
        new_bsr.perm = h_perm.createDeviceVec().getPtr();
        new_bsr.iperm = h_iperm.createDeviceVec().getPtr();
        new_bsr.elem_ind_map = h_elem_ind_map.createDeviceVec().getPtr();
        new_bsr.tr_rowp = h_tr_rowp.createDeviceVec().getPtr();
        new_bsr.tr_cols = h_tr_cols.createDeviceVec().getPtr();
        new_bsr.tr_block_map = h_tr_block_map.createDeviceVec().getPtr();
        new_bsr.elem_conn = d_elem_conn;
        new_bsr.nelems = nelems;
        new_bsr.nodes_per_elem = nodes_per_elem;
        new_bsr.n_eim = n_eim;

        return new_bsr;
    }

    __HOST__ void free() {
        /* cleanup / delete data after use*/
        if (!this->host) {
#ifdef USE_GPU
            // delete data on device
            if (rowp) cudaFree(rowp);
            if (cols) cudaFree(cols);
            if (perm) cudaFree(perm);
            if (iperm) cudaFree(iperm);
            if (elem_ind_map) cudaFree(elem_ind_map);
            if (tr_rowp) cudaFree(tr_rowp);
            if (tr_cols) cudaFree(tr_cols);
            if (tr_block_map) cudaFree(tr_block_map);
#endif  // USE_GPU
        } else {
            // delete data on host
            if (rowp) delete[] rowp;
            if (cols) delete[] cols;
            if (perm) delete[] perm;
            if (iperm) delete[] iperm;
            if (elem_ind_map) delete[] elem_ind_map;
            if (tr_rowp) delete[] tr_rowp;
            if (tr_cols) delete[] tr_cols;
            if (tr_block_map) delete[] tr_block_map;
        }
    }

    // object data, kept public =---------------------------
    int32_t nnzb, nelems, nnodes, nodes_per_elem, block_dim, n_eim;
    const int32_t *elem_conn;
    int32_t *rowp, *cols, *elem_ind_map;
    int32_t *perm, *iperm;
    int32_t *tr_rowp, *tr_cols, *tr_block_map;
    bool host;
};