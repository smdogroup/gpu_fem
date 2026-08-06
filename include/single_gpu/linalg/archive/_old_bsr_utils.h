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

// typedef std::size_t index_t;

// in order to work with cusparse index_t has to be int
typedef int index_t;

// dependency to smdogroup/sparse-utils repo
#include "sparse_utils/sparse_symbolic.h"
#include "sparse_utils/sparse_utils.h"

// pre declaration for readability and use in the BsrData class below
__HOST__ void get_row_col_ptrs(const int &nelems, const int &nnodes, const int32_t *conn,
                               const int &nodes_per_elem, int &nnzb, index_t *&rowPtr,
                               index_t *&colPtr);

__HOST__ void sparse_utils_fillin(const int &nnodes, int &nnzb, index_t *&rowPtr, index_t *&colPtr,
                                  double fill_factor, const bool print = false);

__HOST__ void sparse_utils_reordered_fillin(const int &nnodes, int &nnzb, index_t *&rowPtr,
                                            index_t *&colPtr, index_t *&perm, index_t *&iperm,
                                            double fill_factor, const bool print = false);

__HOST__ void get_row_col_ptrs_sparse(int nelems, int nnodes, int nodes_per_elem,
                                      const int32_t *conn, int &nnzb, index_t *&rowPtr,
                                      index_t *&colPtr);

__HOST__
void get_elem_ind_map(const int &nelems, const int &nnodes, const int32_t *conn,
                      const int &nodes_per_elem, const int &nnzb, index_t *&rowPtr,
                      index_t *&colPtr, index_t *&perm, index_t *&elemIndMap);

class BsrData {
   public:
    BsrData() = default;
    __HOST__ BsrData(const int nelems, const int nnodes, const int &nodes_per_elem,
                     const int &block_dim, const int32_t *conn)
        : nelems(nelems),
          nnodes(nnodes),
          nodes_per_elem(nodes_per_elem),
          conn(conn),
          block_dim(block_dim),
          elemIndMap(nullptr),
          transpose_rowPtr(nullptr),
          transpose_colPtr(nullptr),
          transpose_block_map(nullptr),
          host(true) {
        make_nominal_ordering();
        // get_row_col_ptrs(nelems, nnodes, conn, nodes_per_elem, nnzb, rowPtr,
        //                  colPtr);
        get_row_col_ptrs_sparse(nelems, nnodes, nodes_per_elem, conn, nnzb, rowPtr, colPtr);
    }

    __HOST__ BsrData(const int nnodes, const int block_dim, const int nnzb, index_t *origRowPtr,
                     index_t *origColPtr, int *perm = nullptr, int *iperm = nullptr)
        : nnodes(nnodes),
          block_dim(block_dim),
          nnzb(nnzb),
          rowPtr(origRowPtr),
          conn(nullptr),
          colPtr(origColPtr),
          elemIndMap(nullptr),
          transpose_rowPtr(nullptr),
          transpose_colPtr(nullptr),
          transpose_block_map(nullptr),
          perm(perm),
          iperm(iperm),
          host(true) {
        if (!perm) {
            make_nominal_ordering();
        }
    }

    __HOST__ void post_reordering_update(int nnzb_, int *rowPtr_, int *colPtr_, int *perm_,
                                         int *iperm_, bool print = true) {
        // pass in reordered sparsity
        nnzb = nnzb_;
        rowPtr = rowPtr_;
        colPtr = colPtr_;
        perm = perm_;
        iperm = iperm_;

        // now update assembly information and transpose maps
        if (conn != nullptr) {
            get_elem_ind_map(nelems, nnodes, this->conn, nodes_per_elem, nnzb, rowPtr, colPtr, perm,
                             this->elemIndMap);
        } else {
            if (print) {
                printf("no elem conn, provided so skipping get_elem_ind_map");
            }
        }
        get_transpose_bsr_data();
    }

    __HOST__ void symbolic_factorization(double fill_factor = 10.0, const bool print = false) {
        // do symbolic factorization for fillin
        auto start = std::chrono::high_resolution_clock::now();
        if (print) {
            printf("begin symbolic factorization::\n");
            printf("\tnnzb = %d\n", nnzb);
        }
        // sparse_utils_fillin(nnodes, nnzb, rowPtr, colPtr, fill_factor,
        // print);
        sparse_utils_reordered_fillin(nnodes, nnzb, rowPtr, colPtr, perm, iperm, fill_factor,
                                      print);

        // if (print) {
        //     printf("\t1/3 done with sparse utils fillin\n");
        // }

        if (conn != nullptr) {
            get_elem_ind_map(nelems, nnodes, this->conn, nodes_per_elem, nnzb, rowPtr, colPtr, perm,
                             this->elemIndMap);
            // if (print) {
            //     printf("\t2/3 done with elem ind map\n");
            // }
        } else {
            if (print) {
                printf("no elem conn, provided so skipping get_elem_ind_map");
            }
        }

        get_transpose_bsr_data();
        if (print) {
            printf("\t3/3 done getting transpose map\n");
        }

        // print timing data
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        double dt = duration.count() / 1e6;
        if (print) {
            printf("\tfinished symbolic factorization in %.4e sec\n", dt);
        }
    }

    __HOST__ void make_nominal_ordering() {
        // original ordering (no permutation yet just 0 to nnodes-1)
        perm = new int32_t[nnodes];
        iperm = new int32_t[nnodes];
        for (int inode = 0; inode < nnodes; inode++) {
            perm[inode] = inode;
            iperm[inode] = inode;
        }
    }

    __HOST__ void get_transpose_bsr_data() {
        // get symbolic rowPtr, colPtr locations for transpose matrix using
        // sparse_utils.h

        // printf("starting get transpose bsr data\n");

        // make wrapper object of BSRMat in sparse_utils.h
        auto su_mat =
            SparseUtils::BSRMat<double, 1, 1>(nnodes, nnodes, nnzb, rowPtr, colPtr, nullptr);

        // perform transpose symbolic operation
        // for transpose row and col Ptrs
        auto su_mat_transpose = SparseUtils::BSRMatMakeTransposeSymbolic(su_mat);
        transpose_rowPtr = su_mat_transpose->rowp;
        transpose_colPtr = su_mat_transpose->cols;

        // printf("")

        // also compute a transpose block map map
        // from transpose_block_ind => orig_block_ind (of the values arrays)
        transpose_block_map = new index_t[nnzb];

        index_t *temp_col_local_block_ind_ctr = new index_t[nnodes];
        memset(temp_col_local_block_ind_ctr, 0, nnodes * sizeof(index_t));
        for (int block_row = 0; block_row < nnodes; block_row++) {
            for (int orig_block_ind = rowPtr[block_row]; orig_block_ind < rowPtr[block_row + 1];
                 orig_block_ind++) {
                index_t block_col = colPtr[orig_block_ind];

                // get data for transpose block map
                index_t orig_val_ind = orig_block_ind;
                index_t transpose_val_ind =
                    transpose_rowPtr[block_col] + temp_col_local_block_ind_ctr[block_col];
                // increment the number of block col in this col already used
                temp_col_local_block_ind_ctr[block_col]++;
                // store the map between transpose block val ind to
                // original val ind
                transpose_block_map[transpose_val_ind] = orig_val_ind;
            }
        }

        // printf("rowPtr: ");
        // printVec<index_t>(nnodes + 1, rowPtr);
        // printf("colPtr: ");
        // printVec<index_t>(nnzb, colPtr);

        // printf("tr_rowPtr: ");
        // printVec<index_t>(nnodes + 1, transpose_rowPtr);
        // printf("tr_colPtr: ");
        // printVec<index_t>(nnzb, transpose_colPtr);
        // printf("transpose_block_map: ");
        // printVec<index_t>(nnzb, transpose_block_map);

        delete[] temp_col_local_block_ind_ctr;
    }

    __HOST__ BsrData createDeviceBsrData() {  // deep copy each array onto the
        BsrData new_bsr;
        new_bsr.nnzb = this->nnzb;
        new_bsr.nodes_per_elem = this->nodes_per_elem;
        new_bsr.block_dim = this->block_dim;
        new_bsr.nelems = this->nelems;
        new_bsr.nnodes = this->nnodes;
        new_bsr.host = false;

        // make host vec copies of these pointers
        // first the essentials
        HostVec<index_t> h_rowPtr(this->nnodes + 1, this->rowPtr);
        auto d_rowPtr = h_rowPtr.createDeviceVec();
        new_bsr.rowPtr = d_rowPtr.getPtr();

        HostVec<index_t> h_colPtr(nnzb, this->colPtr);
        auto d_colPtr = h_colPtr.createDeviceVec();
        new_bsr.colPtr = d_colPtr.getPtr();

        // also send permutations to device
        HostVec<index_t> h_perm(this->nnodes, this->perm);
        auto d_perm = h_perm.createDeviceVec();
        new_bsr.perm = d_perm.getPtr();

        HostVec<index_t> h_iperm(this->nnodes, this->iperm);
        auto d_iperm = h_iperm.createDeviceVec();
        new_bsr.iperm = d_iperm.getPtr();

        // elem ind map
        int nodes_per_elem2 = this->nodes_per_elem * this->nodes_per_elem;
        int n_elemIndMap = this->nelems * nodes_per_elem2;
        if (this->elemIndMap) {
            HostVec<index_t> h_elemIndMap(n_elemIndMap, this->elemIndMap);
            auto d_elemIndMap = h_elemIndMap.createDeviceVec();
            new_bsr.elemIndMap = d_elemIndMap.getPtr();
        }

        // // debug
        // printf("h_rowPtr: ");
        // printVec<int>(new_bsr.nnodes + 1, h_rowPtr.getPtr());
        // printf("\n");
        // DeviceVec<int> d_rowPtr2(new_bsr.nnodes + 1, new_bsr.rowPtr);
        // auto h_rowPtr2 = d_rowPtr2.createHostVec();
        // printf("h_rowPtr2: ");
        // printVec<int>(new_bsr.nnodes, h_rowPtr2.getPtr());
        // printf("\n");

        // TODO : also send tranpose data to device
        if (this->transpose_rowPtr) {
            HostVec<index_t> h_transpose_rowPtr(this->nnodes + 1, this->transpose_rowPtr);
            auto d_transpose_rowPtr = h_transpose_rowPtr.createDeviceVec();
            new_bsr.transpose_rowPtr = d_transpose_rowPtr.getPtr();
        }

        if (this->transpose_colPtr) {
            HostVec<index_t> h_transpose_colPtr(nnzb, this->transpose_colPtr);
            auto d_transpose_colPtr = h_transpose_colPtr.createDeviceVec();
            new_bsr.transpose_colPtr = d_transpose_colPtr.getPtr();
        }

        if (this->transpose_block_map) {
            HostVec<index_t> h_transpose_block_map(nnzb, this->transpose_block_map);
            auto d_transpose_block_map = h_transpose_block_map.createDeviceVec();
            new_bsr.transpose_block_map = d_transpose_block_map.getPtr();
        }

        return new_bsr;
    }

    __HOST__ BsrData createHostBsrData() {  // deep copy each array onto the
        BsrData new_bsr;
        new_bsr.nnzb = this->nnzb;
        new_bsr.nodes_per_elem = this->nodes_per_elem;
        new_bsr.block_dim = this->block_dim;
        new_bsr.nelems = this->nelems;
        new_bsr.nnodes = this->nnodes;

        // make host vec copies of these pointers
        DeviceVec<index_t> d_rowPtr(this->nnodes + 1, this->rowPtr);
        DeviceVec<index_t> d_colPtr(nnzb, this->colPtr);
        int nodes_per_elem2 = this->nodes_per_elem * this->nodes_per_elem;
        int n_elemIndMap = this->nelems * nodes_per_elem2;
        DeviceVec<index_t> d_elemIndMap(n_elemIndMap, this->elemIndMap);

        // now use the Vec utils to create host pointers of them
        auto h_rowPtr = d_rowPtr.createHostVec();
        auto h_colPtr = d_colPtr.createHostVec();
        auto h_elemIndMap = d_elemIndMap.createHostVec();

        // now copy to the new BSR structure
        new_bsr.rowPtr = h_rowPtr.getPtr();
        new_bsr.colPtr = h_colPtr.getPtr();
        new_bsr.elemIndMap = h_elemIndMap.getPtr();

        // TODO : also send tranpose data to device
        DeviceVec<index_t> d_transpose_rowPtr(this->nnodes + 1, this->transpose_rowPtr);
        auto h_transpose_rowPtr = d_transpose_rowPtr.createHostVec();
        new_bsr.transpose_rowPtr = h_transpose_rowPtr.getPtr();

        DeviceVec<index_t> d_transpose_colPtr(nnzb, this->transpose_colPtr);
        auto h_transpose_colPtr = d_transpose_colPtr.createHostVec();
        new_bsr.transpose_colPtr = h_transpose_colPtr.getPtr();

        DeviceVec<index_t> d_transpose_block_map(nnzb, this->transpose_block_map);
        auto h_transpose_block_map = d_transpose_block_map.createHostVec();
        new_bsr.transpose_block_map = h_transpose_block_map.getPtr();

        // also send permutations to device
        if (this->perm) {
            DeviceVec<index_t> d_perm(this->nnodes, this->perm);
            auto h_perm = d_perm.createHostVec();
            new_bsr.perm = h_perm.getPtr();
        }

        if (this->iperm) {
            DeviceVec<index_t> d_iperm(this->nnodes, this->iperm);
            auto h_iperm = d_iperm.createHostVec();
            new_bsr.iperm = h_iperm.getPtr();
        }
        return new_bsr;
    }

    __HOST_DEVICE__ int32_t getNumValues() const {
        // length of values array
        return nnzb * block_dim * block_dim;
    }

    __HOST__ void free() {
        if (!this->host) {
            if (this->rowPtr) {
                cudaFree(this->rowPtr);
            }
            if (this->colPtr) {
                cudaFree(this->colPtr);
            }
            if (this->elemIndMap) {
                cudaFree(this->elemIndMap);
            }
            if (this->perm) {
                cudaFree(this->perm);
            }
            if (this->iperm) {
                cudaFree(this->iperm);
            }
            if (this->transpose_rowPtr) {
                cudaFree(this->transpose_rowPtr);
            }
            if (this->transpose_colPtr) {
                cudaFree(this->transpose_colPtr);
            }
            if (this->transpose_block_map) {
                cudaFree(this->transpose_block_map);
            }
        } else {  // on the device
            if (this->rowPtr) {
                delete[] rowPtr;
            }
            if (this->colPtr) {
                delete[] colPtr;
            }
            if (this->elemIndMap) {
                delete[] elemIndMap;
            }
            if (this->perm) {
                delete[] perm;
            }
            if (this->iperm) {
                delete[] iperm;
            }
            if (this->transpose_rowPtr) {
                delete[] transpose_rowPtr;
            }
            if (this->transpose_colPtr) {
                delete[] transpose_colPtr;
            }
            if (this->transpose_block_map) {
                delete[] transpose_block_map;
            }
        }
    }

    // private: (keep public for now?)
    int32_t nnzb;  // num nonzero blocks (in full matrix)
    int32_t nelems, nnodes;
    int32_t nodes_per_elem;  // kelem : nodes_per_elem^2 dense matrix
    int32_t block_dim;       // equiv to vars_per_node (each block is 6x6)
    const int32_t *conn;     // element connectivity
    index_t *rowPtr, *colPtr, *elemIndMap;
    index_t *perm, *iperm;  // reorderings of nodes for lower fillin
    index_t *transpose_rowPtr, *transpose_colPtr, *transpose_block_map;
    bool host;
};

// main utils
// ---------------------------------------------------

__HOST_DEVICE__ bool node_in_elem_conn(const int &nodes_per_elem, const int inode,
                                       const int *elem_conn) {
    for (int i = 0; i < nodes_per_elem; i++) {
        if (elem_conn[i] == inode) {
            return true;
        }
    }
    return false;
}

__HOST__ void get_row_col_ptrs_sparse(int nelems, int nnodes, int nodes_per_elem,
                                      const int32_t *conn, int &nnzb, index_t *&rowPtr,
                                      index_t *&colPtr) {
    auto su_mat =
        SparseUtils::BSRMatFromConnectivityCUDA<double, 1>(nelems, nnodes, nodes_per_elem, conn);

    nnzb = su_mat->nnz;
    rowPtr = su_mat->rowp;
    colPtr = su_mat->cols;
}

__HOST__ void get_row_col_ptrs(
    const int &nelems, const int &nnodes, const int *conn, const int &nodes_per_elem,
    int &nnzb,         // num nonzero blocks
    index_t *&rowPtr,  // array of len nnodes+1 for how many cols in each row
    index_t *&colPtr   // array of len nnzb of the column indices for each block
) {
    // could launch a kernel to do this somewhat in parallel?

    // Need to speed this up here..

    nnzb = 0;
    std::vector<index_t> _rowPtr(nnodes + 1, 0);
    std::vector<index_t> _colPtr;

    // loop over each block row checking nz values
    for (int inode = 0; inode < nnodes; inode++) {
        std::vector<index_t> temp;
        for (int ielem = 0; ielem < nelems; ielem++) {
            const int *elem_conn = &conn[ielem * nodes_per_elem];
            if (node_in_elem_conn(nodes_per_elem, inode, elem_conn)) {
                for (int in = 0; in < nodes_per_elem; in++) {
                    temp.push_back(elem_conn[in]);
                }
            }
        }

        // first sort the vector
        std::sort(temp.begin(), temp.end());

        // remove duplicates
        auto last = std::unique(temp.begin(), temp.end());
        temp.erase(last, temp.end());

        // add this to _colPtr
        _colPtr.insert(_colPtr.end(), temp.begin(), temp.end());

        // add num non zeros to nnzb for this row, also update
        nnzb += temp.size();
        _rowPtr[inode + 1] = nnzb;
    }

    // copy data to output pointers (deep copy)
    rowPtr = new index_t[nnodes + 1];
    std::copy(_rowPtr.begin(), _rowPtr.end(), rowPtr);
    colPtr = new index_t[nnzb];
    std::copy(_colPtr.begin(), _colPtr.end(), colPtr);
}

__HOST__ void sparse_utils_fillin(const int &nnodes, int &nnzb, index_t *&rowPtr, index_t *&colPtr,
                                  double fill_factor, const bool print) {
    // fillin without reordering (results in more nonzeros, no permutations)

    std::vector<index_t> _rowPtr(nnodes + 1, 0);
    std::vector<index_t> _colPtr(index_t(fill_factor * nnzb));
    int nnzb_old = nnzb;

    nnzb = SparseUtils::CSRFactorSymbolic(nnodes, rowPtr, colPtr, _rowPtr, _colPtr);
    if (print) {
        printf(
            "\tsymbolic factorization with fill_factor %.2f from nnzb %d to "
            "%d NZ\n",
            fill_factor, nnzb_old, nnzb);
    }

    // resize this after running the symbolic factorization
    // TODO : can use less than this full nnzb in the case of preconditioning or
    // incomplete ILU?
    _colPtr.resize(nnzb);

    std::copy(_rowPtr.begin(), _rowPtr.end(), rowPtr);
    colPtr = new index_t[nnzb];
    std::copy(_colPtr.begin(), _colPtr.end(), colPtr);

    // printf("rowPtr: ");
    // printVec<index_t>(nnodes + 1, rowPtr);
    // printf("colPtr: ");
    // printVec<index_t>(nnzb, colPtr);
}

// AMD reordered fillin
__HOST__ void sparse_utils_reordered_fillin(const int &nnodes, int &nnzb, index_t *&rowPtr,
                                            index_t *&colPtr, index_t *&perm, index_t *&iperm,
                                            double fill_factor, const bool print) {
    // printf("pre-fillin: rowPtr: ");
    // printVec<int>(nnodes + 1, rowPtr);
    // printf("pre-fillin: colPtr: ");
    // printVec<int>(nnzb, colPtr);

    auto su_mat = SparseUtils::BSRMat<double, 1, 1>(nnodes, nnodes, nnzb, rowPtr, colPtr, nullptr);

    int nnzb_old = nnzb;

    // does fillin on CSR matrix after computing AMD reordering that reduces num
    // nonzeros for fillin during factorization
    // auto su_mat2 = BSRMatAMDFactorSymbolic(su_mat, fill_factor);
    auto su_mat2 = BSRMatAMDFactorSymbolicCUDA(su_mat, fill_factor);

    // delete previous pointers here
    if (rowPtr) {
        delete[] rowPtr;
    }
    if (colPtr) {
        delete[] colPtr;
    }
    if (perm) {
        delete[] perm;
    }
    if (iperm) {
        delete[] iperm;
    }

    // double check that the pointers we are copying are in long-term memory
    // (not some vector arrays hopefully)
    nnzb = su_mat2->nnz;
    rowPtr = su_mat2->rowp;
    colPtr = su_mat2->cols;

    // get reverse perm as iperm, iperm as perm from convention in sparse-utils
    // is flipped from my definition of reordering matrix.. (realized this after
    // code was written)
    perm = su_mat2->iperm;
    iperm = su_mat2->perm;

    // printf("post-fillin: rowPtr: ");
    // printVec<int>(nnodes + 1, rowPtr);
    // printf("post-fillin: colPtr: ");
    // printVec<int>(nnzb, colPtr);

    // printf("perm: ");
    // printVec<int32_t>(nnodes, perm);

    if (print) {
        printf(
            "\tsymbolic factorization with fill_factor %.2f from nnzb %d to "
            "%d NZ\n",
            fill_factor, nnzb_old, nnzb);
    }
}

// // RCM (reverse cuthill mcgee) reordered fillin
// __HOST__ void RCM_reordered_fillin(const int &nnodes, int &nnzb,
//                                    index_t *&rowPtr, index_t *&colPtr,
//                                    index_t *&perm, index_t *&iperm,
//                                    const bool print) {

//     // original fillin
//     int nnzb_pre_fillin = nnzb;
//     // DEBUG (get nnzb and then bandwidth)
//     double fill_factor = 10.0;
//     sparse_utils_fillin(nnodes, nnzb,
//         rowPtr, colPtr, fill_factor, print);
//     // TODO : need to be careful and not change orig row, colPtr here?
//     int orig_bandwidth = getBandWidth(nnodes, nnzb, rowPtr, colPtr);

//     // call RCM reordering
//     int *perm;
//     int root = 0; // why do we need this input?
//     int n_rcm_vars = 10; // what is this?
//     int rvars = TacsComputeRCMOrder(nnodes, rowPtr, colPtr, perm, root,
//     n_rcm_iters);

//     // Debugging, compute new bandwidth with permutation
//     // apply new permutation using sparse utils
//     // (TODO : make separate routine for this, see
//     BSRMatAMDFactorSymbolicCUDA above)

//     // compute new rowPtr, colPtr (TODO : needs to be in temp rowPtr, colPtr
//     probably for debug bandwidth check) sparse_utils_fillin(nnodes, nnzb,
//         rowPtr, colPtr, fill_factor, print);
//     int new_bandwidth = getBandWidth(nnodes, nnzb, rowPtr, colPtr);
//     if (print) {
//         printf("\tsymbolic factorization with fill_factor %.2f from nnzb %d
//         to "
//                "%d NZ\n",
//                fill_factor, nnzb_old, nnzb);
//     }

//     // TODO : need to get ILU(k) here instead? not full fillin?
//     // Dr. K says ILU(k) code is in TACS in BSCRMat class
// }

// __HOST__ void qordering(int fill_level, double prune_factor, int const int
// nnodes, const int nnzb,
//     int* rowPtr, int *colPtr, bool print) {
//     // first apply RCM reordering
//     int *perm;
//     int root = 0; // why do we need this input?
//     int n_rcm_vars = 10; // what is this?
//     int rvars = TacsComputeRCMOrder(nnodes, rowPtr, colPtr, perm, root,
//     n_rcm_iters);

//     // need to get bandwidth from RCM
//     // TODO Make this method that takes arbitrary perm and then does fillin
//     // and it also returns bandwidth in the temp row, colPtr arrays in there
//     int bandwidth = sparse_utils_perm_fillin(nnodes, nnzb, rowPtr, colPtr,
//     fill_factor, print);

//     // now modify the perm with random reorderings of the prune width (see
//     q-ordering paper) int prune_width = (int)(1.0/prune_factor * bandwidth);
//     int num_prunes = (nnodes + prune_width - 1) / prune_width;
//     // random number generator
//     std::random_device rd;
//     std::mt19937 g(rd());

//     std::vector<int> perm2; // TODO : define it relative to perm
//     for (int iprune = 0; iprune < num_prunes; iprune++) {
//         // TODO : how to best apply extra permutation on top of current one?
//         int lower = prune_width * iprune;
//         int upper = max(lower + prune_width, nnodes);
//         std::shuffle(perm2.begin() + lower, perm2.begin() + upper, g)
//     }
//     // TODO : apply this new perm2 to the old one maybe?

//     // TODO : also compute iperm?

//     // now compute symbolic ILU(k) factorization.. so that CUDA ILU(0)
//     actually returns ILU(k) later on
//     // use fill_level input here
//     // TODO : should ILU(k) be a separate step? Probably, break it out of
//     this routine

//     // TODO : visualize matrices at some of these steps so I check if I'm
//     doing it right..
// }

__HOST__ int getBandWidth(const int &nnodes, const int &nnzb, int *rowPtr, int *colPtr) {
    // need to compute bandwidth for q-ordering
    // bandwidth = max_{a_ij neq 0} |i - j|   ( so max off-diagonal distance
    // within sparsity pattern)
    int bandwidth = 0;
    for (int i = 0; i < nnodes; i++) {
        for (int jp = rowPtr[i]; jp < rowPtr[i + 1]; jp++) {
            int j = colPtr[jp];
            int diff = abs(i - j);
            if (diff > bandwidth) {
                bandwidth = diff;
            }
        }
    }
    return bandwidth;
}

// *!
//   Compute the ILU(levFill) preconditioner

//   fill == The expected degree of fill-in after the computation
//   levs == The level set of the entry
// */
__HOST__ void computeILUk(int nnodes, int nnzb, int *&rowPtr, int *&colPtr, int levFill,
                          double fill, int **_levs) {
    int nrows = nnodes;  // Record the number of rows/columns
    int ncols = nnodes;

    // Number of non-zeros in the original matrix
    int mat_size = nnzb;
    int size = 0;
    int max_size = (int)(fill * mat_size);  // The maximum size - for now

    int *cols = new int[max_size];
    int *levs = new int[max_size];  // The level of fill of an entry
    int *rowp = new int[nrows + 1];
    int *diag = new int[nrows];

    // Fill in the first entries
    rowp[0] = 0;

    // Allocate space for the temporary row info
    int *rlevs = new int[ncols];
    int *rcols = new int[ncols];

    for (int i = 0; i < nrows; i++) {
        int nr = 0;  // Number of entries in the current row

        // Add the matrix elements to the current row of the matrix.
        // These new elements are sorted.
        int diag_flag = 0;
        for (int j = rowPtr[i]; j < rowPtr[i + 1]; j++) {
            if (colPtr[j] == i) {
                diag_flag = 1;
            }
            rcols[nr] = colPtr[j];
            rlevs[nr] = 0;
            nr++;
        }

        // No diagonal element associated with row i, add one!
        if (!diag_flag) {
            nr = TacsMergeSortedArrays(nr, rcols, 1, &i);
        }

        // Now, perform the symbolic factorization -- this generates new entries
        int j = 0;
        for (; rcols[j] < i; j++) {  // For entries in this row, before the diagonal
            int clev = rlevs[j];     // the level of fill for this entry

            int p = j + 1;                   // The index into rcols
            int k_end = rowp[rcols[j] + 1];  // the end of row number cols[j]

            // Start with the first entry after the diagonal in row, cols[j]
            // k is the index into cols for row cols[j]
            for (int k = diag[rcols[j]] + 1; k < k_end; k++) {
                // Increment p to an entry where we may have cols[k] == rcols[p]
                while (p < nr && rcols[p] < cols[k]) {
                    p++;
                }

                // The element already exists, check if it has a lower level of
                // fill and update the fill level if necessary
                if (p < nr && rcols[p] == cols[k]) {
                    if (rlevs[p] > (clev + levs[k] + 1)) {
                        rlevs[p] = clev + levs[k] + 1;
                    }
                } else if ((clev + levs[k] + 1) <= levFill) {
                    // The element does not exist but should since the level of
                    // fill is low enough. Insert the new entry into the list,
                    // but keep the list sorted
                    for (int n = nr; n > p; n--) {
                        rlevs[n] = rlevs[n - 1];
                        rcols[n] = rcols[n - 1];
                    }

                    rlevs[p] = clev + levs[k] + 1;
                    rcols[p] = cols[k];
                    nr++;
                }
            }
        }

        // Check if the size will be exceeded by adding the new elements
        if (size + nr > max_size) {
            int mat_ext = (int)((fill - 1.0) * mat_size);
            if (nr > mat_ext) {
                mat_ext = nr;
            }
            max_size = max_size + mat_ext;
            TacsExtendArray(&cols, size, max_size);
            TacsExtendArray(&levs, size, max_size);
        }

        // Now, put the new entries into the cols/levs arrays
        for (int k = 0; k < nr; k++) {
            cols[size] = rcols[k];
            levs[size] = rlevs[k];
            size++;
        }

        rowp[i + 1] = size;
        diag[i] = j + rowp[i];
    }

    // Clip the cols array to the correct size
    if (max_size > size) {
        TacsExtendArray(&cols, size, size);
    }

    if (rowPtr[nrows] > 0) {
        int rank = 0;
        // MPI_Comm_rank(comm, &rank);
        printf(
            "[%d] BCSRMat: ILU(%d) Input fill ratio %4.2f, actual "
            "fill ratio: %4.2f, nnz(ILU) = %d\n",
            rank, levFill, fill, (1.0 * rowp[nrows]) / rowPtr[nrows], rowp[nrows]);
    }

    delete[] rcols;
    delete[] rlevs;

    // Store the rowp/cols and diag arrays
    // data->rowp = rowp;
    // data->cols = cols;
    // data->diag = diag;
    rowPtr = rowp;
    colPtr = cols;
    // need to store separate diag too? it's only integers, I don't think I need
    // it here..

    *_levs = levs;
}

__HOST__ void get_elem_ind_map(
    const int &nelems, const int &nnodes, const int32_t *conn, const int &nodes_per_elem,
    const int &nnzb,   // num nonzero blocks
    index_t *&rowPtr,  // array of len nnodes+1 for how many cols in each row
    index_t *&colPtr,  // array of len nnzb of the column indices for each block
    index_t *&perm, index_t *&elemIndMap) {
    int nodes_per_elem2 = nodes_per_elem * nodes_per_elem;

    // determine where each global_node node of this elem
    // should be added into the ind of colPtr as map

    // printf("rowPtr: ");
    // printVec<int>(nnodes + 1, rowPtr);
    // printf("colPtr: ");
    // printVec<int>(nnzb, colPtr);

    // printf("in elem_ind_map: perm\n");
    // printVec<int32_t>(9, perm);

    elemIndMap = new index_t[nelems * nodes_per_elem2]();
    // elemIndMap is nelems x 4 x 4 array for nodes_per_elem = 4
    // shows how to add each block matrix into global
    for (int ielem = 0; ielem < nelems; ielem++) {
        const int32_t *local_conn = &conn[nodes_per_elem * ielem];

        // TODO : might be error here due to sorting the local conn..
        // std::vector<int32_t> sorted_conn;
        // for (int lnode = 0; lnode < nodes_per_elem; lnode++) {
        //     sorted_conn.push_back(local_conn[lnode]);
        // }
        // std::sort(sorted_conn.begin(), sorted_conn.end());

        // loop over each block row
        for (int n = 0; n < nodes_per_elem; n++) {
            // int global_node_row = sorted_conn[n];
            int32_t _global_node_row = local_conn[n];
            int32_t global_node_row = perm[_global_node_row];
            // get the block col range of colPtr for this block row
            int col_istart = rowPtr[global_node_row];
            int col_iend = rowPtr[global_node_row + 1];

            // loop over each block col
            for (int m = 0; m < nodes_per_elem; m++) {
                // int global_node_col = sorted_conn[m];
                int32_t _global_node_col = local_conn[m];
                int32_t global_node_col = perm[_global_node_col];

                // find the matching indices in colPtr for the global_node_col
                // of this elem_conn
                for (int i = col_istart; i < col_iend; i++) {
                    if (colPtr[i] == global_node_col) {
                        // add this component of block kelem matrix into
                        // elem_ind_map
                        int nm = nodes_per_elem * n + m;
                        elemIndMap[nodes_per_elem2 * ielem + nm] = i;
                        break;
                    }
                }
            }
        }
    }
}

// nice reference utility (may not use this)
// template <typename T, int block_dim, int nodes_per_elem, int dof_per_elem>
// void assemble_sparse_bsr_matrix(
//     const int &nelems, const int &nnodes, const int &vars_per_node,
//     const int *conn,
//     int &nnzb,        // num nonzero blocks
//     int *&rowPtr,     // array of len nnodes+1 for how many cols in each row
//     int *&colPtr,     // array of len nnzb of the column indices for each
//     block int *&elemIndMap, // map of rowPtr, colPtr assembly locations for
//     kelem int &nvalues,     // length of values array = block_dim^2 * nnzb T
//     *&values) {
//     // get nnzb, rowPtr, colPtr
//     get_row_col_ptrs<nodes_per_elem>(nelems, nnodes, conn, nnzb, rowPtr,
//                                      colPtr);

//     // get elemIndMap to know how to add into the values array
//     get_elem_ind_map<T, nodes_per_elem>(nelems, nnodes, conn, nnzb, rowPtr,
//                                         colPtr, elemIndMap);

//     // create values array (T*& because we can't initialize it outside this
//     // method so
//     //     pointer itself changes, normally T* is enough but not here)
//     nvalues = block_dim * block_dim * nnzb;
//     int nnz_per_block = block_dim * block_dim;
//     int blocks_per_elem = nodes_per_elem * nodes_per_elem;
//     values = new T[nnz_per_block * nnzb];

//     // now add each kelem into values array as part of assembly process
//     for (int ielem = 0; ielem < nelems; ielem++) {
//         T kelem[dof_per_elem * dof_per_elem];
//         get_fake_kelem<T, dof_per_elem>(0.0, kelem);

//         // now use elemIndxMap to add into values
//         for (int elem_block = 0; elem_block < blocks_per_elem; elem_block++)
//         {
//             int istart = nnz_per_block *
//                          elemIndMap[blocks_per_elem * ielem + elem_block];
//             T *val = &values[istart];
//             int elem_block_row = elem_block / nodes_per_elem;
//             int elem_block_col = elem_block % nodes_per_elem;
//             for (int inz = 0; inz < nnz_per_block; inz++) {
//                 int inner_row = inz / block_dim;
//                 int row = vars_per_node * elem_block_row + inner_row;
//                 int inner_col = inz % block_dim;
//                 int col = vars_per_node * elem_block_col + inner_col;

//                 val[inz] += kelem[dof_per_elem * row + col];
//             }
//         }
//     }
// }
