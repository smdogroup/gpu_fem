#pragma once

#include <complex>
#include <fstream>

#include "a2dcore.h"
#include "adscalar.h"
#include "cuda_utils.h"
#include "stdlib.h"

template <typename T>
__HOST_DEVICE__ void printVec(const int N, const T *vec);

template <>
__HOST_DEVICE__ void printVec<int>(const int N, const int *vec) {
    for (int i = 0; i < N; i++) {
        printf("%d,", vec[i]);
    }
    printf("\n");
}

template <>
__HOST_DEVICE__ void printVec<double>(const int N, const double *vec) {
    for (int i = 0; i < N; i++) {
        printf("%.5e,", vec[i]);
    }
    printf("\n");
}

template <>
__HOST_DEVICE__ void printVec<float>(const int N, const float *vec) {
    for (int i = 0; i < N; i++) {
        printf("%.5e,", vec[i]);
    }
    printf("\n");
}

template <>
__HOST_DEVICE__ void printVec<A2D::ADScalar<double, 1>>(const int N,
                                                        const A2D::ADScalar<double, 1> *vec) {
    for (int i = 0; i < N; i++) {
        printf("%.5e,", vec[i].value);
    }
    printf("\n\tderiv ");
    for (int i = 0; i < N; i++) {
        printf("%.5e,", vec[i].deriv[0]);
    }
    printf("\n");
}

// template <>
// __HOST_DEVICE__ void
// printVec<std::complex<double>>(const int N, const std::complex<double> *vec) {
//     for (int i = 0; i < N; i++) {
//         printf("%.5e,", A2D::RealPart(vec[i]));
//     }
//     printf("\n");
// }

__HOST_DEVICE__ bool is_unique(int N, int32_t *local_conn) {
    // check whether all values are different or not
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++) {
            if (local_conn[i] == local_conn[j]) {
                return false;
            }
        }
    }
    return true;
}

__HOST__ void make_unique_conn(int nstrides, int stride, int maxVal, int32_t *conn) {
    for (int ielem = 0; ielem < nstrides; ielem++) {
        int32_t *local_conn = &conn[stride * ielem];
        while (!is_unique(stride, local_conn)) {
            for (int iloc = 0; iloc < stride; iloc++) {
                local_conn[iloc] = rand() % maxVal;
            }
        }
    }
}

template <typename T>
void write_to_csv(size_t size, const T *array, const std::string &filename) {
    std::ofstream out(filename);
    if (!out) {
        throw std::ios_base::failure("Failed to open file for writing");
    }
    for (size_t i = 0; i < size; ++i) {
        out << array[i];
        if (i != size - 1) {
            out << ",";
        }
    }
    out << "\n";
    out.close();
}