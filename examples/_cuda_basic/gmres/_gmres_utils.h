#include <iostream>
#include <cublas_v2.h>
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

#define CHECK_CUBLAS(func)                                                             \
    {                                                                                  \
        cublasStatus_t status = (func);                                                \
        if (status != CUBLAS_STATUS_SUCCESS) {                                         \
            printf("CUBLAS API failed at line %d with error: %d\n", __LINE__, status); \
            return EXIT_FAILURE;                                                       \
        }                                                                              \
    }

#define CHECK_CUSOLVER(func)                                                             \
    {                                                                                    \
        cusolverStatus_t status = (func);                                                \
        if (status != CUSOLVER_STATUS_SUCCESS) {                                         \
            printf("CUSOLVER API failed at line %d with error: %d\n", __LINE__, status); \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    }
