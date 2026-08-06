#pragma once
#include "bsr_data.h"

template <class Vec_>
class CsrMat {
   public:
    using T = typename Vec_::type;
    using Vec = Vec_;
    CsrMat() = default;

    __HOST_DEVICE__ BsrData getBsrData() const { return this->csr_data; }
    __HOST_DEVICE__ BsrData getCsrData() const { return this->csr_data; }
    __HOST_DEVICE__ T *getPtr() { return this->values.getPtr(); }
    __HOST_DEVICE__ int get_nnz() { return csr_data.nnzb; }
    __HOST_DEVICE__ int *getPerm() { return csr_data.perm; }
    __HOST_DEVICE__ int *getIPerm() { return csr_data.iperm; }
    __HOST_DEVICE__ int getBlockDim() { return csr_data.block_dim; }
    __HOST__ void zeroValues() { values.zeroValues(); }

    // BsrData is technically a CsrData here with block_dim = 1

    __HOST__ CsrMat(BsrMat<Vec> &mat) {
        /* switch from a BSR to CSR matrix */
        const BsrData &bsr_data = mat.getBsrData();

        // copy rowp, cols off of device
        int *h_rowp = DeviceVec<int>(bsr_data.nnodes + 1, bsr_data.rowp).createHostVec().getPtr();
        int *h_cols = DeviceVec<int>(bsr_data.nnzb, bsr_data.cols).createHostVec().getPtr();
        int block_dim = bsr_data.block_dim;
        int block_dim2 = block_dim * block_dim;
        T *h_vals = mat.getVec().createHostVec().getPtr();

        // printf("nbrows %d, nnzb %d\n", bsr_data.nnodes, bsr_data.nnzb);
        // printf("h_rowp:");
        // printVec<int>(bsr_data.nnodes + 1, h_rowp);
        // printf("h_cols:");
        // printVec<int>(bsr_data.nnzb, h_cols);
        // printf("h_vals:");
        // printVec<T>(bsr_data.nnzb, h_vals);

        // create new rowp, cols and values
        std::vector<int> new_rowp, new_cols;
        new_rowp.push_back(0);
        std::vector<T> new_vals;
        int nrows = bsr_data.nnodes * block_dim;
        int new_nnz = 0;
        for (int glob_row = 0; glob_row < nrows; glob_row++) {
            int brow = glob_row / block_dim;     // block row
            int inn_row = glob_row % block_dim;  // inner row inside block

            for (int jp = h_rowp[brow]; jp < h_rowp[brow + 1]; jp++) {
                int bcol = h_cols[jp];  // block column
                for (int inn_col = 0; inn_col < block_dim; inn_col++) {
                    int glob_col = block_dim * bcol + inn_col;  // CSR col
                    new_nnz += 1;
                    new_cols.push_back(glob_col);
                    int inz = block_dim * inn_row + inn_col;  // nz in block out of 36 usually
                    new_vals.push_back(h_vals[block_dim2 * jp + inz]);
                }
            }
            new_rowp.push_back(new_nnz);
        }

        // printf("here2, nrows %d, new_nnz %d\n", nrows, new_nnz);

        // printf("nrows %d, nnz %d\n", nrows, new_nnz);
        // printf("new_rowp:");
        // printVec<int>(nrows, new_rowp.data());
        // printf("new_cols:");
        // printVec<int>(new_nnz, new_cols.data());
        // printf("new_vals:");
        // printVec<T>(new_nnz, new_vals.data());

        int *new_hrowp = new int[nrows + 1];
        std::copy(new_rowp.begin(), new_rowp.end(), new_hrowp);
        int *new_hcols = new int[new_nnz];
        std::copy(new_cols.begin(), new_cols.end(), new_hcols);
        T *new_hvals = new T[new_nnz];
        std::copy(new_vals.begin(), new_vals.end(), new_hvals);

        // Allocate device memory & deep copy
#ifdef USE_GPU
        int *d_rowp = nullptr;
        int *d_cols = nullptr;

        cudaMalloc(&d_rowp, sizeof(int) * (nrows + 1));
        cudaMemcpy(d_rowp, new_hrowp, sizeof(int) * (nrows + 1), cudaMemcpyHostToDevice);

        cudaMalloc(&d_cols, sizeof(int) * new_nnz);
        cudaMemcpy(d_cols, new_hcols, sizeof(int) * new_nnz, cudaMemcpyHostToDevice);

        T *d_vals = nullptr;
        cudaMalloc(&d_vals, sizeof(T) * new_nnz);
        cudaMemcpy(d_vals, new_hvals, sizeof(T) * new_nnz, cudaMemcpyHostToDevice);
        this->values = DeviceVec<T>(new_nnz, d_vals);

// deep copy values
#endif

        delete[] new_hrowp;
        delete[] new_hcols;
        delete[] new_hvals;

        // make new BsrData object for the CSR matrix, still copy block_dim from Bsr for later perm
        // and iperm calls (even though CSR kind of has like block_dim = 1, block_dim will be used
        // for vec perm)
        this->csr_data = BsrData(nrows, block_dim, new_nnz, d_rowp, d_cols, bsr_data.perm,
                                 bsr_data.iperm, false);

        // printf("here3, nrows %d, new_nnz %d\n", this->bsr_data.nnodes, this->bsr_data.nnzb);
    }

   private:
    BsrData csr_data;  // was const before
    Vec values;
};

// TODO : class CsrData inherits from BsrData