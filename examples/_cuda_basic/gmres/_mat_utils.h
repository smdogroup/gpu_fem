#include <cmath> 
#include <cassert>
#include <cstdio>
#include <vector>
#include <algorithm>

template <typename T>
void genLaplaceCSR(int *rowp, int *cols, T *vals, int N, int nz, T *rhs) {
    // second order laplace operator on a square domain with nxn nodes for n^2 = N dof
    // linear system based off CUDA samples github here.. created on host first
    // https://github.com/NVIDIA/cuda-samples/tree/master/Samples/4_CUDA_Libraries/conjugateGradientPrecond

    int n = (int) sqrt((double)N);
    assert(n*n==N);
    printf("laplace dimension = %d\n", n);
    int idx = 0;

    // loop over the rows
    for (int i = 0; i < N ; i++) {
        int ix = i % n;
        int iy = i / n;

        rowp[i] = idx;

        // up
        if (iy > 0)
        {
            vals[idx] = 1.0;
            cols[idx] = i - n;
            idx++;
        }
        else
        {
            rhs[i] -= 1.0;
        }

        // left
        if (ix > 0) {
            vals[idx] = 1.0;
            cols[idx] = i - 1;
            idx++;
        } else {
            rhs[i] -= 0.0;
        }

        // center
        vals[idx] = -4.0;
        cols[idx] = i;
        idx++;

        //right
        if (ix  < n - 1)
        {
            vals[idx] = 1.0;
            cols[idx] = i + 1;
            idx++;
        }
        else
        {
            rhs[i] -= 0.0;
        }

        // down
        if (iy  < n - 1)
        {
            vals[idx] = 1.0;
            cols[idx] = i + n;
            idx++;
        }
        else
        {
            rhs[i] -= 0.0;
        }
    }

    rowp[N] = idx;
}

template <typename T>
void printVec(const int N, const T *vec, int nl_freq = 4);

template <>
void printVec<int>(const int N, const int *vec, int nl_freq) {
    for (int i = 0; i < N; i++) {
        printf("%d,", vec[i]);
    }
    printf("\n");
}

template <>
void printVec<float>(const int N, const float *vec, int nl_freq) {
    int ct = 0;
    for (int i = 0; i < N; i++) {
        printf("%.9e,", vec[i]);
        ct += 1;
        if (ct % nl_freq == 0) printf("\n");
    }
    printf("\n");
}

template <>
void printVec<double>(const int N, const double *vec, int nl_freq) {
    int ct = 0;
    for (int i = 0; i < N; i++) {
        printf("%.9e,", vec[i]);
        ct += 1;
        if (ct % nl_freq == 0) printf("\n");
    }
    printf("\n");
}

template <typename T>
void CSRtoBSR(int block_dim, int N, int *csr_rowp, int *csr_cols, T *csr_vals,
    int **bsr_rowp, int **bsr_cols, T **bsr_vals, int *nnzb) {
    assert(block_dim == 2); // for now
    assert(N % block_dim == 0);
    std::vector<int> rowp(0), cols;
    std::vector<T> vals;
    int block_dim2 = block_dim * block_dim;
    int nvals = 0;
    rowp.push_back(0);

    for (int brow = 0; brow < N/2; brow++) {
        // get list of nonzero block cols
        std::vector<int> loc_bcols;
        for (int inner_row = 0; inner_row < 2; inner_row++) {
            int row = 2 * brow + inner_row;
            for (int j = csr_rowp[row]; j < csr_rowp[row+1]; j++) {
                // add any new loc_bcols
                int col = csr_cols[j];
                int bcol = col / 2;
                loc_bcols.push_back(bcol);
            }
        }

        // sort and uniquify the local block cols
        std::sort(loc_bcols.begin(), loc_bcols.end());
        auto last = std::unique(loc_bcols.begin(), loc_bcols.end());
        loc_bcols.erase(last, loc_bcols.end());

        // printf("pre sort and uniquify");
        // printVec<int>(loc_bcols.size(), loc_bcols.data());
        // printf("brow %d, loc bcols:", brow);
        // printVec<int>(loc_bcols.size(), loc_bcols.data());
        int nloc_bcols = loc_bcols.size();

        // then allocate the block data for this row
        int nbrow_vals = block_dim2*loc_bcols.size();
        // printf("nbrow_vals = %d\n", nbrow_vals);
        std::vector<T> loc_bvals(nbrow_vals);

        // now fill in the loc block data values
        for (int inner_row = 0; inner_row < 2; inner_row++) {
            int row = 2 * brow + inner_row;
            for (int j = csr_rowp[row]; j < csr_rowp[row+1]; j++) {
                // add any new loc_bcols
                int col = csr_cols[j];
                int bcol = col / 2;
                int inner_col = col % 2;

                // find the matching loc_bcol ind
                int loc_bcol_ind = -1;
                for (int k = 0; k < nbrow_vals; k++) {
                    if (bcol == loc_bcols[k]) {
                        loc_bcol_ind = k;
                        break;
                    }
                }

                int bind = block_dim2 * loc_bcol_ind;
                int ind = bind + 2 * inner_row + inner_col;
                // printf("ind %d\n", ind);
                loc_bvals[ind] = csr_vals[j];
            }
        }
        // printf("done with ")

        // printf("brow p2 %d, loc bdata:\n", brow);
        // printVec<T>(loc_bvals.size(), loc_bvals.data());

        // update rows, cols, vals now
        nvals += loc_bcols.size();
        rowp.push_back(nvals);
        for (int i = 0; i < nloc_bcols; i++) {
            cols.push_back(loc_bcols[i]);
        }
        for (int i = 0; i < loc_bvals.size(); i++) {
            vals.push_back(loc_bvals[i]);
        }
        // printf("rowp:");
        // printVec<int>(rowp.size(), rowp.data());
        // printf("cols:");
        // printVec<int>(cols.size(), cols.data());
        // printf("vals:");
        // printVec<T>(vals.size(), vals.data());
        // if (brow == 2) return;
    }

    // printf("rowp:");
    // printVec<int>(rowp.size(), rowp.data());
    // printf("cols:");
    // printVec<int>(cols.size(), cols.data());
    // printf("vals:");
    // printVec<T>(vals.size(), vals.data());

    // TODO : copy the new BSR data to the pointers (deep copy out of vectors)
    // so now on heap
    // printf("here1\n");
    *bsr_rowp = new int[rowp.size()];
    *bsr_cols = new int[cols.size()];
    *bsr_vals = new T[vals.size()];
    // printf("here2\n");

    for (int i = 0; i < rowp.size(); i++) {
        (*bsr_rowp)[i] = rowp[i];
    }
    for (int i = 0; i < cols.size(); i++) {
        (*bsr_cols)[i] = cols[i];
    }
    for (int i = 0; i < vals.size(); i++) {
        (*bsr_vals)[i] = vals[i];
    }
    *nnzb = cols.size();

    // printf("nnzb = %d\n", *nnzb);
    // printf("rowp.size = %ld\n", rowp.size());
    // printf("cols.size = %ld\n", cols.size());
    // printf("vals.size = %ld\n", vals.size());

    // printf("bsr_rowp 1:");
    // printVec<int>(N/2+1, *bsr_rowp);
    // printf("bsr_cols 1:");
    // printVec<int>(*nnzb, *bsr_cols);
    // printf("bsr_vals 1:");
    // printVec<T>((*nnzb) * 4, *bsr_vals);
}

template <typename T>
void BSRtoCSR(int block_dim, int N, int nnzb, int *bsr_rowp, int *bsr_cols, T *bsr_vals,
              int **csr_rowp, int **csr_cols, T **csr_vals, int *nz) {
    
    assert(N % 2 == 0);

    std::vector<int> rowp, cols;
    std::vector<T> vals;
    rowp.push_back(0);
    int nvals = 0;

    for (int row = 0; row < N; row++) {
        int brow = row / 2;
        int inner_row = row % 2;

        for (int j = bsr_rowp[brow]; j < bsr_rowp[brow+1]; j++) {
            int bcol = bsr_cols[j];
            for (int inner_col = 0; inner_col < 2; inner_col++) {
                int col = 2 * bcol + inner_col;
                cols.push_back(col);
                int val_ind = 4 * j + 2 * inner_row + inner_col;
                T val = bsr_vals[val_ind];
                vals.push_back(val);
                nvals += 1;
            }
        }

        rowp.push_back(nvals);
    }

    // printf("csr2 rows:");
    // printVec<int>(rowp.size(), rowp.data());
    // printf("csr2 cols:");
    // printVec<int>(cols.size(), cols.data());
    // printf("csr2 vals:");
    // printVec<T>(vals.size(), vals.data());

    // now deep copy out of it
    *nz = nvals;
    *csr_rowp = new int[N + 1];
    *csr_cols = new int[nvals];
    *csr_vals = new T[nvals];

    for (int i = 0; i < N+1; i++) {
        (*csr_rowp)[i] = rowp[i];
    }
    for (int i = 0; i < nvals; i++) {
        (*csr_cols)[i] = cols[i];
        (*csr_vals)[i] = vals[i];
    }
}