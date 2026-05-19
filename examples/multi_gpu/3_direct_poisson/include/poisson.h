#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#include "linalg/bsr_data.h"

template <typename T>
void genLaplaceCSR(int *rowp, int *cols, T *vals, int N, int nz, T *rhs) {
    int n = (int)std::sqrt((double)N);
    assert(n * n == N);

    printf("laplace dimension = %d\n", n);

    int idx = 0;
    for (int i = 0; i < N; i++) {
        int ix = i % n;
        int iy = i / n;

        rowp[i] = idx;

        if (iy > 0) {
            vals[idx] = 1.0;
            cols[idx] = i - n;
            idx++;
        } else {
            rhs[i] -= 1.0;
        }

        if (ix > 0) {
            vals[idx] = 1.0;
            cols[idx] = i - 1;
            idx++;
        } else {
            rhs[i] -= 0.0;
        }

        vals[idx] = -4.0;
        cols[idx] = i;
        idx++;

        if (ix < n - 1) {
            vals[idx] = 1.0;
            cols[idx] = i + 1;
            idx++;
        } else {
            rhs[i] -= 0.0;
        }

        if (iy < n - 1) {
            vals[idx] = 1.0;
            cols[idx] = i + n;
            idx++;
        } else {
            rhs[i] -= 0.0;
        }
    }

    rowp[N] = idx;
    assert(idx == nz);
}

template <typename T>
void genBlockDiagPoissonCSR(int *rowp, int *cols, T *vals, int nnode, int block_dim, T *rhs) {
    const int n = (int)std::sqrt((double)nnode);
    assert(n * n == nnode);

    const int N = nnode * block_dim;

    printf("laplace dimension = %d, block_dim = %d, scalar N = %d\n", n, block_dim, N);

    int idx = 0;

    for (int inode = 0; inode < nnode; inode++) {
        const int ix = inode % n;
        const int iy = inode / n;

        for (int d = 0; d < block_dim; d++) {
            const int row = block_dim * inode + d;
            rowp[row] = idx;

            if (iy > 0) {
                cols[idx] = block_dim * (inode - n) + d;
                vals[idx] = 1.0;
                idx++;
            } else {
                rhs[row] -= 1.0;
            }

            if (ix > 0) {
                cols[idx] = block_dim * (inode - 1) + d;
                vals[idx] = 1.0;
                idx++;
            }

            cols[idx] = row;
            vals[idx] = -4.0;
            idx++;

            if (ix < n - 1) {
                cols[idx] = block_dim * (inode + 1) + d;
                vals[idx] = 1.0;
                idx++;
            }

            if (iy < n - 1) {
                cols[idx] = block_dim * (inode + n) + d;
                vals[idx] = 1.0;
                idx++;
            }
        }
    }

    rowp[N] = idx;
}

template <typename T>
void CSRtoBSR(int block_dim, int N, const int *csr_rowp, const int *csr_cols, const T *csr_vals,
              int **bsr_rowp, int **bsr_cols, T **bsr_vals, int *nnzb) {
    assert(N % block_dim == 0);

    const int nbrows = N / block_dim;
    const int block_dim2 = block_dim * block_dim;

    std::vector<int> rowp;
    std::vector<int> cols;
    std::vector<T> vals;

    rowp.reserve(nbrows + 1);
    rowp.push_back(0);

    for (int brow = 0; brow < nbrows; brow++) {
        std::vector<int> loc_bcols;

        for (int inner_row = 0; inner_row < block_dim; inner_row++) {
            const int row = block_dim * brow + inner_row;
            for (int j = csr_rowp[row]; j < csr_rowp[row + 1]; j++) {
                loc_bcols.push_back(csr_cols[j] / block_dim);
            }
        }

        std::sort(loc_bcols.begin(), loc_bcols.end());
        loc_bcols.erase(std::unique(loc_bcols.begin(), loc_bcols.end()), loc_bcols.end());

        const int nloc_bcols = (int)loc_bcols.size();
        std::vector<T> loc_bvals(block_dim2 * nloc_bcols, T(0));

        for (int inner_row = 0; inner_row < block_dim; inner_row++) {
            const int row = block_dim * brow + inner_row;
            for (int j = csr_rowp[row]; j < csr_rowp[row + 1]; j++) {
                const int col = csr_cols[j];
                const int bcol = col / block_dim;
                const int inner_col = col % block_dim;

                auto it = std::lower_bound(loc_bcols.begin(), loc_bcols.end(), bcol);
                assert(it != loc_bcols.end() && *it == bcol);

                const int loc_bcol_ind = (int)(it - loc_bcols.begin());
                const int ind = block_dim2 * loc_bcol_ind + block_dim * inner_row + inner_col;
                loc_bvals[ind] = csr_vals[j];
            }
        }

        for (int bcol : loc_bcols) {
            cols.push_back(bcol);
        }
        for (T v : loc_bvals) {
            vals.push_back(v);
        }

        rowp.push_back((int)cols.size());
    }

    *nnzb = (int)cols.size();
    *bsr_rowp = new int[rowp.size()];
    *bsr_cols = new int[cols.size()];
    *bsr_vals = new T[vals.size()];

    std::copy(rowp.begin(), rowp.end(), *bsr_rowp);
    std::copy(cols.begin(), cols.end(), *bsr_cols);
    std::copy(vals.begin(), vals.end(), *bsr_vals);
}

template <typename T>
void BSRtoCSR(int block_dim, int N, int nnzb, const int *bsr_rowp, const int *bsr_cols,
              const T *bsr_vals, int **csr_rowp, int **csr_cols, T **csr_vals, int *nz) {
    assert(N % block_dim == 0);

    const int nbrows = N / block_dim;
    const int block_dim2 = block_dim * block_dim;

    std::vector<int> rowp;
    std::vector<int> cols;
    std::vector<T> vals;

    rowp.reserve(N + 1);
    rowp.push_back(0);

    for (int row = 0; row < N; row++) {
        const int brow = row / block_dim;
        const int inner_row = row % block_dim;

        for (int jp = bsr_rowp[brow]; jp < bsr_rowp[brow + 1]; jp++) {
            const int bcol = bsr_cols[jp];

            for (int inner_col = 0; inner_col < block_dim; inner_col++) {
                const int col = block_dim * bcol + inner_col;
                const int val_ind = block_dim2 * jp + block_dim * inner_row + inner_col;

                cols.push_back(col);
                vals.push_back(bsr_vals[val_ind]);
            }
        }

        rowp.push_back((int)cols.size());
    }

    *nz = (int)cols.size();
    assert(*nz == nnzb * block_dim2);

    *csr_rowp = new int[rowp.size()];
    *csr_cols = new int[cols.size()];
    *csr_vals = new T[vals.size()];

    std::copy(rowp.begin(), rowp.end(), *csr_rowp);
    std::copy(cols.begin(), cols.end(), *csr_cols);
    std::copy(vals.begin(), vals.end(), *csr_vals);
}

template <typename T>
void convertBSRtoLUpattern(int block_dim, int N, int nnzb, const int *bsr_rowp, const int *bsr_cols,
                           const T *bsr_vals, int **perm, int **iperm, int **rowp, int **cols,
                           T **vals, int *lu_nnzb) {
    assert(N % block_dim == 0);

    const int nnodes = N / block_dim;
    const int block_dim2 = block_dim * block_dim;

    BsrData bsr_data(nnodes, block_dim, nnzb, const_cast<int *>(bsr_rowp),
                     const_cast<int *>(bsr_cols), nullptr, nullptr, true);

    bsr_data.AMD_reordering();
    bsr_data.compute_full_LU_pattern();

    *lu_nnzb = bsr_data.rowp[nnodes];

    *perm = new int[nnodes];
    *iperm = new int[nnodes];
    *rowp = new int[nnodes + 1];
    *cols = new int[*lu_nnzb];
    *vals = new T[block_dim2 * (*lu_nnzb)];

    for (int i = 0; i <= nnodes; i++) {
        (*rowp)[i] = bsr_data.rowp[i];
    }

    for (int i = 0; i < nnodes; i++) {
        (*perm)[i] = bsr_data.perm[i];    // permuted row -> original row
        (*iperm)[i] = bsr_data.iperm[i];  // original row -> permuted row
    }

    for (int j = 0; j < *lu_nnzb; j++) {
        (*cols)[j] = bsr_data.cols[j];
    }

    std::fill(*vals, *vals + block_dim2 * (*lu_nnzb), T(0));

    // Copy original values into the permuted full-LU sparsity pattern.
    // Explicit fill-in entries remain numerical zero but are present structurally.
    for (int prow = 0; prow < nnodes; prow++) {
        const int row = bsr_data.perm[prow];

        for (int jp = bsr_data.rowp[prow]; jp < bsr_data.rowp[prow + 1]; jp++) {
            const int pcol = bsr_data.cols[jp];
            const int col = bsr_data.perm[pcol];

            int old_jp = -1;
            for (int jp2 = bsr_rowp[row]; jp2 < bsr_rowp[row + 1]; jp2++) {
                if (bsr_cols[jp2] == col) {
                    old_jp = jp2;
                    break;
                }
            }

            if (old_jp >= 0) {
                for (int ii = 0; ii < block_dim2; ii++) {
                    (*vals)[block_dim2 * jp + ii] = bsr_vals[block_dim2 * old_jp + ii];
                }
            }
        }
    }
}

template <typename T>
void permute_block_vec(int N, int block_dim, const int *perm, const T *x_orig, T *x_perm) {
    const int nnodes = N / block_dim;
    for (int prow = 0; prow < nnodes; prow++) {
        const int row = perm[prow];
        for (int d = 0; d < block_dim; d++) {
            x_perm[block_dim * prow + d] = x_orig[block_dim * row + d];
        }
    }
}

template <typename T>
void unpermute_block_vec(int N, int block_dim, const int *perm, const T *x_perm, T *x_orig) {
    const int nnodes = N / block_dim;
    for (int prow = 0; prow < nnodes; prow++) {
        const int row = perm[prow];
        for (int d = 0; d < block_dim; d++) {
            x_orig[block_dim * row + d] = x_perm[block_dim * prow + d];
        }
    }
}
