#pragma once

#include <iostream>

#ifdef USE_GPU
#define __SHARED__ __shared__
#define __HOST_DEVICE__ __host__ __device__
#define __HOST__ __host__
#define __DEVICE__ __device__
#define __GLOBAL__ __global__
#else
#define __SHARED__
#define __HOST_DEVICE__
#define __HOST__
#define __DEVICE__
#define __GLOBAL__
#endif

// A2D
#include "a2ddefs.h"

#ifdef USE_GPU
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

template <uint32_t xdim = 1, uint32_t ydim = 1, uint32_t zdim = 1,
          uint32_t max_registers_per_thread = 255, uint32_t elements_per_block = 1>
class ExecParameters {
   public:
};

#ifdef USE_GPU

// an atomic add for complex numbers, so we can do complex-step tests on GPU calls for unittests
__device__ inline void atomicAdd(A2D_complex_t<double>* addr, A2D_complex_t<double> val) {
    atomicAdd(reinterpret_cast<double*>(addr), val.real());
    atomicAdd(reinterpret_cast<double*>(addr) + 1, val.imag());
}

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
            exit(EXIT_FAILURE);                                                        \
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

void cuda_show_kernel_error() {
    auto err = cudaGetLastError();
    std::cout << "error code: " << err << "\n";
    std::cout << "error string: " << cudaGetErrorString(err) << "\n";
}
#endif