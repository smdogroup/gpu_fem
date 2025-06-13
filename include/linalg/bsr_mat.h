#pragma once

#include "bsr_data.h"
#include "vec.h"
#ifdef USE_GPU
#include "../cuda_utils.h"
#include "bsr_mat.cuh"
#endif  // USE_GPU

template <class Vec_>
class BsrMat {
   public:
    using T = typename Vec_::type;
    using Vec = Vec_;
    BsrMat() = default;
#ifdef USE_GPU
    static constexpr dim3 bcs_block = dim3(32);
    static constexpr dim3 nodes_block = dim3(32);
#endif  // USE_GPU

    __HOST_DEVICE__ BsrMat(const BsrData &bsr_data, Vec &values)
        : bsr_data(bsr_data), values(values) {}
    __HOST__ BsrMat(const BsrData &bsr_data) : bsr_data(bsr_data) {
        int nvalues = bsr_data.getNumValues();
        values = Vec(nvalues);
    }

    __HOST__ void zeroValues() { values.zeroValues(); }
    __HOST_DEVICE__ BsrData getBsrData() const { return bsr_data; }
    __HOST_DEVICE__ int get_nnz() { return bsr_data.getNumValues(); }
    __HOST_DEVICE__ Vec getVec() { return values; }
    __HOST__ HostVec<T> createHostVec() { return values.createHostVec(); }
    __HOST_DEVICE__ T *getPtr() { return values.getPtr(); }
    __HOST_DEVICE__ const T *getPtr() const { return values.getPtr(); }
    __HOST_DEVICE__ int *getPerm() { return bsr_data.perm; }
    __HOST_DEVICE__ int *getIPerm() { return bsr_data.iperm; }
    __HOST_DEVICE__ int getBlockDim() { return bsr_data.block_dim; }
    __HOST_DEVICE__ int *getRowPtr() { return bsr_data.rowp; }
    __HOST_DEVICE__ int *getColPtr() { return bsr_data.cols; }

    __HOST__ void apply_bcs(DeviceVec<int> bcs) {
        /* apply bcs to the matrix values (rows + cols) */
        int nbcs = bcs.getSize();
        const index_t *rowPtr = bsr_data.rowp, *colPtr = bsr_data.cols, *iperm = bsr_data.iperm,
                      *tr_rowp = bsr_data.tr_rowp, *tr_cols = bsr_data.tr_cols,
                      *tr_block_map = bsr_data.tr_block_map;
        int nnodes = bsr_data.nnodes, nodes_per_elem = bsr_data.nodes_per_elem,
            block_dim = bsr_data.block_dim;
        int nnz_per_block = block_dim * block_dim,
            blocks_per_elem = nodes_per_elem * nodes_per_elem;
        T *valPtr = values.getPtr();

#ifdef USE_GPU
        dim3 block = bcs_block;
        int nblocks = (nbcs + block.x - 1) / block.x;
        dim3 grid(nblocks);

        // launch two kernels asynchronously
        apply_mat_bcs_rows_kernel<T, DeviceVec><<<grid, block>>>(
            bcs, rowPtr, colPtr, iperm, nnodes, valPtr, blocks_per_elem, nnz_per_block, block_dim);

        apply_mat_bcs_cols_kernel<T, DeviceVec>
            <<<grid, block>>>(bcs, tr_rowp, tr_cols, tr_block_map, iperm, nnodes, valPtr,
                              blocks_per_elem, nnz_per_block, block_dim);
        CHECK_CUDA(cudaDeviceSynchronize());
#endif  // USE_GPU
    }

    __HOST__ void add_diag_nugget(T eta) {
        /* apply bcs to the matrix values (rows + cols) */
        const index_t *rowPtr = bsr_data.rowp, *colPtr = bsr_data.cols;
        int nnodes = bsr_data.nnodes, block_dim = bsr_data.block_dim;
        int ndiag = bsr_data.nnodes * bsr_data.block_dim;
        T *valPtr = values.getPtr();

#ifdef USE_GPU
        dim3 block(32);
        int nblocks = (ndiag + block.x - 1) / block.x;
        dim3 grid(nblocks);

        // adds eta * I to the diag where eta > 0 is a scalar
        add_mat_diag_kernel<T><<<grid, block>>>(nnodes, block_dim, rowPtr, colPtr, valPtr, eta);
        CHECK_CUDA(cudaDeviceSynchronize());
#endif  // USE_GPU
    }

    __HOST__ void mult_diag_nugget(T eta) {
        /* apply bcs to the matrix values (rows + cols) */
        const index_t *rowPtr = bsr_data.rowp, *colPtr = bsr_data.cols;
        int nnodes = bsr_data.nnodes, block_dim = bsr_data.block_dim;
        int ndiag = bsr_data.nnodes * bsr_data.block_dim;
        T *valPtr = values.getPtr();

#ifdef USE_GPU
        dim3 block(32);
        int nblocks = (ndiag + block.x - 1) / block.x;
        dim3 grid(nblocks);

        // adds eta * I to the diag where eta > 0 is a scalar
        mult_diag_kernel<T><<<grid, block>>>(nnodes, block_dim, rowPtr, colPtr, valPtr, eta);
        CHECK_CUDA(cudaDeviceSynchronize());
#endif  // USE_GPU
    }

    void copyValuesTo(BsrMat<Vec> mat) {
        /* copy values to another matrix object */

        const index_t *rowp = bsr_data.rowp, *cols = bsr_data.cols;
        int *t_rowp = mat.getRowPtr(), *t_cols = mat.getColPtr();
        T *vals = values.getPtr(), *t_vals = mat.getPtr();
        int block_dim = bsr_data.block_dim, nnodes = bsr_data.nnodes;
        int block_dim2 = block_dim * block_dim;

#ifndef USE_GPU
        // CPU version
        for (int irow = 0; irow < nnodes; irow++) {
            int t_jp = t_rowp[irow];
            int t_jp_max = t_rowp[irow + 1];
            for (int jp = rowp[irow]; jp < rowp[irow + 1]; jp++) {
                int bcol = cols[jp];
                // have other mat catch up to the same column
                for (; t_cols[t_jp] < bcol; t_jp++) {
                }
                // want to put a debug check that same column # here?
                // copy entire block into here
                for (int inz = 0; inz < block_dim2; inz++) {
                    t_vals[block_dim2 * t_jp + inz] = vals[block_dim2 * jp + inz];
                }
            }
        }
#endif

#ifdef USE_GPU
        // GPU version
        dim3 block = nodes_block;
        int nblocks = (nnodes + block.x - 1) / block.x;
        dim3 grid(nblocks);

        copy_mat_values_kernel<T>
            <<<grid, block>>>(nnodes, block_dim, rowp, cols, vals, t_rowp, t_cols, t_vals);
        CHECK_CUDA(cudaDeviceSynchronize());
#endif
    }

#ifdef USE_GPU
    __DEVICE__
    void addElementMatrixValuesFromShared(const bool active_thread, const int start,
                                          const int stride, const T scale, const int ielem,
                                          const int dof_per_node, const int nodes_per_elem,
                                          const int32_t *elem_conn, const T *shared_elem_mat) {
        /* add element stiffness matrices from shared memory to global data */

        int dof_per_elem = dof_per_node * nodes_per_elem, block_dim = bsr_data.block_dim,
            blocks_per_elem = nodes_per_elem * nodes_per_elem;
        int nnz_per_block = bsr_data.block_dim * bsr_data.block_dim;
        const index_t *elem_ind_map = bsr_data.elem_ind_map;
        const index_t *loc_elem_ind_map = &elem_ind_map[blocks_per_elem * ielem];
        T *valPtr = values.getPtr();

        for (int elem_block = start; elem_block < blocks_per_elem; elem_block += stride) {
            int glob_block_ind = loc_elem_ind_map[elem_block];
            int istart = nnz_per_block * glob_block_ind;
            T *val = &valPtr[istart];
            int elem_block_row = elem_block / nodes_per_elem;
            int elem_block_col = elem_block % nodes_per_elem;

            // loop over each nz in each block of kelem
            for (int inz = 0; inz < nnz_per_block; inz++) {
                int local_row = inz / block_dim;
                int local_col = inz % block_dim;
                int erow = block_dim * elem_block_row + local_row;
                int ecol = block_dim * elem_block_col + local_col;

                atomicAdd(&val[inz], scale * shared_elem_mat[dof_per_elem * erow + ecol]);
            }
        }
    }

    __DEVICE__
    void addElementMatRow(const bool active_thread, const int elem_block_row, const int inner_row,
                          const int ielem, const int start, const int stride,
                          const int dof_per_node, const int nodes_per_elem,
                          const int32_t *elem_conn, const T *elem_data) {
        /* add row from elem stiffness matrix (adding by row is coalesced, while col is not) */

        // int dof_per_elem = dof_per_node * nodes_per_elem,
        int block_dim = bsr_data.block_dim, blocks_per_elem = nodes_per_elem * nodes_per_elem;
        int nnz_per_block = bsr_data.block_dim * bsr_data.block_dim;
        const index_t *elem_ind_map = bsr_data.elem_ind_map;
        const index_t *loc_elem_ind_map = &elem_ind_map[blocks_per_elem * ielem];
        T *valPtr = values.getPtr();

        for (int elem_block_col = start; elem_block_col < nodes_per_elem; elem_block_col++) {
            int elem_block = nodes_per_elem * elem_block_row + elem_block_col;
            int glob_block_ind = loc_elem_ind_map[elem_block];
            int istart = nnz_per_block * glob_block_ind + block_dim * inner_row;
            T *val = &valPtr[istart];

            for (int idof = 0; idof < dof_per_node; idof++) {
                atomicAdd(&val[idof], elem_data[block_dim * elem_block_col + idof]);
            }
        }
    }
#endif  // USE_GPU

    template <typename I>
    __HOST_DEVICE__ T &operator[](const I i) {
        return values[i];
    }
    template <typename I>
    __HOST_DEVICE__ const T &operator[](const I i) const {
        return values[i];
    }

    void free() {
        // use free not destructor as no pass by ref allowed in kernels and often leads to
        // unintended destructor calls cannot free bsr_data
        values.free();
    }

    // host code
    __HOST__ void apply_bcs(HostVec<int> bcs);
    __HOST_DEVICE__ void addElementMatrixValues(const T scale, const int ielem,
                                                const int dof_per_node, const int nodes_per_elem,
                                                const int32_t *elem_conn, const T *elem_mat);

   private:
    BsrData bsr_data;  // was const before
    Vec values;
};

template <class Assembler, class Vec>
__HOST__ BsrMat<Vec> createBsrMat(Assembler &assembler) {
    return BsrMat<Vec>(assembler.getBsrData());
}

template <class Vec>
__HOST__ void BsrMat<Vec>::apply_bcs(HostVec<int> bcs) {
    // some prelim values needed for both cases
    int nbcs = bcs.getSize();
    const index_t *rowPtr = bsr_data.rowp;
    const index_t *colPtr = bsr_data.cols;
    const index_t *perm = bsr_data.perm;
    int nnodes = bsr_data.nnodes;
    T *valPtr = values.getPtr();

    int blocks_per_elem = bsr_data.nodes_per_elem * bsr_data.nodes_per_elem;
    int nnz_per_block = bsr_data.block_dim * bsr_data.block_dim;
    int block_dim = bsr_data.block_dim;

    // loop over each bc
    for (int ibc = 0; ibc < nbcs; ibc++) {
        // zero out the bc rows
        int _glob_row = bcs[ibc];
        int glob_row = perm[_glob_row];  // the bc dof
        int bc_temp = glob_row;
        int inner_row = glob_row % block_dim;  // the local dof constrained in this node
        int block_row = glob_row / block_dim;  // equiv to bc node

        // set bc row to zero
        for (int col_ptr_ind = rowPtr[block_row]; col_ptr_ind < rowPtr[block_row + 1];
             col_ptr_ind++) {
            T *val = &valPtr[nnz_per_block * col_ptr_ind];

            int block_col = colPtr[col_ptr_ind];
            for (int inner_col = 0; inner_col < block_dim; inner_col++) {
                int inz = block_dim * inner_row + inner_col;  // nz entry in block
                int glob_col = block_col * block_dim + inner_col;
                // ternary operation will be more friendly on the GPU (even
                // though this is CPU here)
                val[inz] = (glob_row == glob_col) ? 1.0 : 0.0;
            }
        }

        // TODO : try adding bc to column how too, but will want to speed
        // this up eventually or just use CPU for debug for now set bc row
        // to zero
        for (int block_row2 = 0; block_row2 < nnodes; block_row2++) {
            for (int col_ptr_ind = rowPtr[block_row2]; col_ptr_ind < rowPtr[block_row2 + 1];
                 col_ptr_ind++) {
                T *val = &valPtr[nnz_per_block * col_ptr_ind];

                int block_col = colPtr[col_ptr_ind];
                for (int inz = 0; inz < block_dim * block_dim; inz++) {
                    int inner_row2 = inz / block_dim;
                    int inner_col = inz % block_dim;
                    int glob_row2 = block_row2 * block_dim + inner_row2;

                    int glob_col = block_col * block_dim + inner_col;

                    if (glob_col != bc_temp)
                        continue;  // only apply bcs to column here so need
                                   // matching column

                    // ternary operation will be more friendly on the
                    // GPU (even though this is CPU here)
                    val[inz] = (glob_row2 == glob_col) ? 1.0 : 0.0;
                }
                // val += nnz_per_block;
            }
        }

        // TODO : for bc column want K^T rowPtr, colPtr map probably
        // to do it without if statements (compute map on CPU assembler
        // init) NOTE : zero out columns only needed for nonzero BCs
    }
}

template <class Vec>
__HOST_DEVICE__ void BsrMat<Vec>::addElementMatrixValues(const T scale, const int ielem,
                                                         const int dof_per_node,
                                                         const int nodes_per_elem,
                                                         const int32_t *elem_conn,
                                                         const T *elem_mat) {
    /* only works for no reordering right now*/

    // similar method to vec.h or Vec.addElementValues but here for matrix
    // and here for Bsr format
    int dof_per_elem = dof_per_node * nodes_per_elem;
    int blocks_per_elem = bsr_data.nodes_per_elem * bsr_data.nodes_per_elem;
    int nnz_per_block = bsr_data.block_dim * bsr_data.block_dim;
    int block_dim = bsr_data.block_dim;
    const index_t *elem_ind_map = bsr_data.elem_ind_map;
    T *valPtr = values.getPtr();

    // loop over each of the blocks in the kelem
    for (int elem_block = 0; elem_block < blocks_per_elem; elem_block++) {
        // perm already applied to elem_ind_map
        int istart = nnz_per_block * elem_ind_map[blocks_per_elem * ielem + elem_block];
        T *val = &valPtr[istart];
        int block_row = elem_block / nodes_per_elem;
        int block_col = elem_block % nodes_per_elem;

        // loop over each nz in each block of kelem
        for (int inz = 0; inz < nnz_per_block; inz++) {
            int inner_row = inz / block_dim;
            int inner_col = inz % block_dim;
            int row = block_dim * block_row + inner_row;
            int col = block_dim * block_col + inner_col;

            val[inz] += scale * elem_mat[dof_per_elem * row + col];
        }
    }
}