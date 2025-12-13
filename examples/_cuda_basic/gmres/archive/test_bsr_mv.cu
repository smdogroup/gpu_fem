#include <iostream>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

#define CHECK_CUDA(call)                                                         \
    {                                                                            \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    }

#define CHECK_CUSPARSE(call)                                     \
    {                                                            \
        cusparseStatus_t err = call;                             \
        if (err != CUSPARSE_STATUS_SUCCESS) {                    \
            std::cerr << "CUSPARSE error: " << err << std::endl; \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    }

int main() {
    using T = double;

    // problem
    // A = [1, 0; 0, 2]
    // x = [3, 4]
    // y = A * x = [3, 8] (Demonstrate with BSR mv operation)

    // initialize host data
    int rowp[3] = {0, 1, 2};
    int cols[2] = {0, 1};
    T vals[2] = {1.0, 2.0};
    T x[2] = {3, 4};
    T y[2];

    int *d_rowp, *d_cols;
    T *d_vals, *d_x, *d_y;
    CHECK_CUDA(cudaMalloc((void **)&d_rowp, 3 * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&d_cols, 2 * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&d_vals, 2 * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void **)&d_x, 2 * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void **)&d_y, 2 * sizeof(T)));

    CHECK_CUDA(cudaMemcpy(d_rowp, rowp, 3 * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_cols, cols, 2 * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vals, vals, 2 * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, x, 2 * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, y, 2 * sizeof(T), cudaMemcpyHostToDevice));

    /* Description of the A matrix */
    cusparseMatDescr_t descrA = 0;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
    CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

    /* Create CUSPARSE context */
    cusparseHandle_t cusparseHandle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

    int mb = 2;
    int block_dim = 1;
    T a = 1.0, b = 0.0;
    int nnzb = 2;

    CHECK_CUSPARSE(cusparseDbsrmv(
        cusparseHandle, 
        CUSPARSE_DIRECTION_ROW,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        mb, mb, nnzb,
        &a, descrA,
        d_vals, d_rowp, d_cols,
        block_dim,
        d_x,
        &b,
        d_y
    ));

    // copy d_y back to host y vec
    T y_ref[2] = {3, 8};
    CHECK_CUDA(cudaMemcpy(&y[0], d_y, 2 * sizeof(T), cudaMemcpyDeviceToHost));
    printf("y:");
    for (int i = 0; i < 2; i++) {
        printf("%.4e,", y[i]);
    }
};