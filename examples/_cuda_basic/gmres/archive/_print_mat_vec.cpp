#include "../_mat_utils.h"

int main() {
    using T = double;
    int *rowp, *cols;
    int N = 16;
    int M = N;
    T *vals, *rhs, *x;
    int nz = 5 * N - 4 * (int)sqrt((double)N);

    // allocate rowp, cols on host
    rowp = (int*)malloc(sizeof(int) * (N + 1));
    cols = (int*)malloc(sizeof(int) * nz);
    vals = (T*)malloc(sizeof(T) * nz);
    x = (T*)malloc(sizeof(T) * N);
    rhs = (T*)malloc(sizeof(T) * N);

    // initialize data
    genLaplaceCSR<T>(rowp, cols, vals, N, nz, rhs); 

    // printVec<int>(N+1, rowp);
    // printf("\n");
    // printVec<int>(nz, cols);
    // printf("\n");
    // printVec<T>(nz, vals);
    // printf("\n");
    // printVec<T>(N, rhs);

    // debug BSR conversion
    int block_dim = 2, nnzb;
    int *bsr_rowp, *bsr_cols;
    T *bsr_vals;
    convertToBSR<T>(block_dim, N, rowp, cols, vals, &bsr_rowp, &bsr_cols, &bsr_vals, &nnzb);

    printf("nnzb = %d\n", nnzb);
    printf("bsr_rowp:");
    printVec<int>(N/2+1, bsr_rowp);
    printf("bsr_cols:");
    printVec<int>(nnzb, bsr_cols);
    printf("bsr_vals:");
    printVec<T>(nnzb * 4, bsr_vals);
    return 0;
}