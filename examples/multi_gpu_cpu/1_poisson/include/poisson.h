#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// -----------------------------------------------------------------------------
// Threaded Poisson helper routines.
//
// Keeps the original API names where useful and also adds *_threaded variants.
// These are host-only assembly/conversion utilities. GPU work remains in the
// matvec driver.
// -----------------------------------------------------------------------------

template <typename T>
void genLaplaceCSR_threaded(int *rowp, int *cols, T *vals, int N, int nz, T *rhs) {
    // Second-order Laplace operator on a square n x n grid, n^2 = N.
    int n = (int)std::sqrt((double)N);
    assert(n * n == N);

    std::vector<int> row_nnz(N);

#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        int ix = i % n;
        int iy = i / n;

        int ct = 1;  // center
        if (iy > 0) ct++;
        if (ix > 0) ct++;
        if (ix < n - 1) ct++;
        if (iy < n - 1) ct++;
        row_nnz[i] = ct;
    }

    rowp[0] = 0;
    for (int i = 0; i < N; i++) {
        rowp[i + 1] = rowp[i] + row_nnz[i];
    }

    assert(rowp[N] == nz);

#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        int ix = i % n;
        int iy = i / n;
        int idx = rowp[i];

        rhs[i] = 0.0;

        // up
        if (iy > 0) {
            vals[idx] = (T)1.0;
            cols[idx] = i - n;
            idx++;
        } else {
            rhs[i] -= (T)1.0;
        }

        // left
        if (ix > 0) {
            vals[idx] = (T)1.0;
            cols[idx] = i - 1;
            idx++;
        } else {
            rhs[i] -= (T)0.0;
        }

        // center
        vals[idx] = (T)-4.0;
        cols[idx] = i;
        idx++;

        // right
        if (ix < n - 1) {
            vals[idx] = (T)1.0;
            cols[idx] = i + 1;
            idx++;
        } else {
            rhs[i] -= (T)0.0;
        }

        // down
        if (iy < n - 1) {
            vals[idx] = (T)1.0;
            cols[idx] = i + n;
            idx++;
        } else {
            rhs[i] -= (T)0.0;
        }
    }
}

template <typename T>
void genLaplaceCSR(int *rowp, int *cols, T *vals, int N, int nz, T *rhs) {
    genLaplaceCSR_threaded<T>(rowp, cols, vals, N, nz, rhs);
}

template <typename T>
void getDiagValsBSR_threaded(const int *rowp, const int *cols, const T *vals, int block_dim,
                             int nnodes, T *diag_vals) {
    int block_dim2 = block_dim * block_dim;

#pragma omp parallel for
    for (int i = 0; i < nnodes; i++) {
        for (int jp = rowp[i]; jp < rowp[i + 1]; jp++) {
            int j = cols[jp];
            if (i == j) {
                for (int ii = 0; ii < block_dim2; ii++) {
                    diag_vals[block_dim2 * i + ii] = vals[block_dim2 * jp + ii];
                }
                break;
            }
        }
    }
}

template <typename T>
void getDiagValsBSR(int *rowp, int *cols, T *vals, int block_dim, int nnodes, T *diag_vals) {
    getDiagValsBSR_threaded<T>(rowp, cols, vals, block_dim, nnodes, diag_vals);
}

template <typename T>
void CSRtoBSR_threaded(int block_dim, int N, const int *csr_rowp, const int *csr_cols,
                       const T *csr_vals, int **bsr_rowp, int **bsr_cols, T **bsr_vals, int *nnzb) {
    assert(block_dim == 2);  // preserves original limitation
    assert(N % block_dim == 0);

    const int nnodes = N / block_dim;
    const int block_dim2 = block_dim * block_dim;

    std::vector<int> row_counts(nnodes, 0);

    // Pass 1: determine unique block columns per block row. For block_dim = 2 and
    // 5-point scalar Laplacian, a tiny local fixed buffer is enough and much
    // cheaper than creating vectors in every row.
#pragma omp parallel for
    for (int brow = 0; brow < nnodes; brow++) {
        int loc_bcols[16];
        int nloc = 0;

        for (int inner_row = 0; inner_row < block_dim; inner_row++) {
            int row = block_dim * brow + inner_row;
            for (int j = csr_rowp[row]; j < csr_rowp[row + 1]; j++) {
                int bcol = csr_cols[j] / block_dim;
                bool exists = false;
                for (int k = 0; k < nloc; k++) {
                    if (loc_bcols[k] == bcol) {
                        exists = true;
                        break;
                    }
                }
                if (!exists) {
                    assert(nloc < 16);
                    loc_bcols[nloc++] = bcol;
                }
            }
        }

        std::sort(loc_bcols, loc_bcols + nloc);
        row_counts[brow] = nloc;
    }

    std::vector<int> rowp(nnodes + 1, 0);
    for (int i = 0; i < nnodes; i++) {
        rowp[i + 1] = rowp[i] + row_counts[i];
    }

    int total_nnzb = rowp[nnodes];
    std::vector<int> cols(total_nnzb, -1);
    std::vector<T> vals(total_nnzb * block_dim2, (T)0.0);

    // Pass 2: fill block columns and block values.
#pragma omp parallel for
    for (int brow = 0; brow < nnodes; brow++) {
        int loc_bcols[16];
        int nloc = 0;

        for (int inner_row = 0; inner_row < block_dim; inner_row++) {
            int row = block_dim * brow + inner_row;
            for (int j = csr_rowp[row]; j < csr_rowp[row + 1]; j++) {
                int bcol = csr_cols[j] / block_dim;
                bool exists = false;
                for (int k = 0; k < nloc; k++) {
                    if (loc_bcols[k] == bcol) {
                        exists = true;
                        break;
                    }
                }
                if (!exists) {
                    assert(nloc < 16);
                    loc_bcols[nloc++] = bcol;
                }
            }
        }

        std::sort(loc_bcols, loc_bcols + nloc);

        int row_start = rowp[brow];
        for (int k = 0; k < nloc; k++) {
            cols[row_start + k] = loc_bcols[k];
        }

        for (int inner_row = 0; inner_row < block_dim; inner_row++) {
            int row = block_dim * brow + inner_row;
            for (int j = csr_rowp[row]; j < csr_rowp[row + 1]; j++) {
                int col = csr_cols[j];
                int bcol = col / block_dim;
                int inner_col = col % block_dim;

                int loc_bcol_ind = -1;
                for (int k = 0; k < nloc; k++) {
                    if (bcol == loc_bcols[k]) {
                        loc_bcol_ind = k;
                        break;
                    }
                }

                assert(loc_bcol_ind >= 0);
                int bsr_block = row_start + loc_bcol_ind;
                int val_ind = block_dim2 * bsr_block + block_dim * inner_row + inner_col;
                vals[val_ind] = csr_vals[j];
            }
        }
    }

    *bsr_rowp = new int[rowp.size()];
    *bsr_cols = new int[cols.size()];
    *bsr_vals = new T[vals.size()];

#pragma omp parallel for
    for (int i = 0; i < (int)rowp.size(); i++) {
        (*bsr_rowp)[i] = rowp[i];
    }

#pragma omp parallel for
    for (int i = 0; i < (int)cols.size(); i++) {
        (*bsr_cols)[i] = cols[i];
    }

#pragma omp parallel
    for (int i = 0; i < (int)vals.size(); i++) {
        (*bsr_vals)[i] = vals[i];
    }

    *nnzb = total_nnzb;
}

template <typename T>
void CSRtoBSR(int block_dim, int N, int *csr_rowp, int *csr_cols, T *csr_vals, int **bsr_rowp,
              int **bsr_cols, T **bsr_vals, int *nnzb) {
    CSRtoBSR_threaded<T>(block_dim, N, csr_rowp, csr_cols, csr_vals, bsr_rowp, bsr_cols, bsr_vals,
                         nnzb);
}

template <typename T>
void BSRtoCSR_threaded(int block_dim, int N, int nnzb, const int *bsr_rowp, const int *bsr_cols,
                       const T *bsr_vals, int **csr_rowp, int **csr_cols, T **csr_vals, int *nz) {
    assert(N % block_dim == 0);

    const int nnodes = N / block_dim;
    const int block_dim2 = block_dim * block_dim;

    std::vector<int> row_counts(N, 0);

#pragma omp parallel for
    for (int row = 0; row < N; row++) {
        int brow = row / block_dim;
        row_counts[row] = (bsr_rowp[brow + 1] - bsr_rowp[brow]) * block_dim;
    }

    std::vector<int> rowp(N + 1, 0);
    for (int i = 0; i < N; i++) {
        rowp[i + 1] = rowp[i] + row_counts[i];
    }

    int total_nz = rowp[N];
    std::vector<int> cols(total_nz, -1);
    std::vector<T> vals(total_nz, (T)0.0);

#pragma omp parallel for
    for (int row = 0; row < N; row++) {
        int brow = row / block_dim;
        int inner_row = row % block_dim;
        int idx = rowp[row];

        for (int j = bsr_rowp[brow]; j < bsr_rowp[brow + 1]; j++) {
            int bcol = bsr_cols[j];
            for (int inner_col = 0; inner_col < block_dim; inner_col++) {
                int col = block_dim * bcol + inner_col;
                cols[idx] = col;
                vals[idx] = bsr_vals[block_dim2 * j + block_dim * inner_row + inner_col];
                idx++;
            }
        }
    }

    *nz = total_nz;
    *csr_rowp = new int[N + 1];
    *csr_cols = new int[total_nz];
    *csr_vals = new T[total_nz];

#pragma omp parallel for
    for (int i = 0; i < N + 1; i++) {
        (*csr_rowp)[i] = rowp[i];
    }

#pragma omp parallel for
    for (int i = 0; i < total_nz; i++) {
        (*csr_cols)[i] = cols[i];
        (*csr_vals)[i] = vals[i];
    }
}

template <typename T>
void BSRtoCSR(int block_dim, int N, int nnzb, int *bsr_rowp, int *bsr_cols, T *bsr_vals,
              int **csr_rowp, int **csr_cols, T **csr_vals, int *nz) {
    BSRtoCSR_threaded<T>(block_dim, N, nnzb, bsr_rowp, bsr_cols, bsr_vals, csr_rowp, csr_cols,
                         csr_vals, nz);
}