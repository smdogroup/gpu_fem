#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include <iostream>
#include <thrust/device_vector.h>

template <typename T>
void printVec(const int N, const T *vec);

template <>
void printVec<int>(const int N, const int *vec) {
    for (int i = 0; i < N; i++) {
        printf("%d,", vec[i]);
    }
    printf("\n");
}

int main() {
    // demonstrate going from row counts to rowptr
    // int N = 32;
    int N = 1024 * 1024;

    int *h_rowCounts = new int[N];
    // random init
    for (int i = 0; i < N; i++) {
        h_rowCounts[i] = rand() % 10; // random integer from 0 to 10
    }

    // serial version for ref
    int *h_true_rowp = new int[N+1];
    int nvals = 0;
    for (int i = 1; i < N + 1; i++) {
        nvals += h_rowCounts[i-1];
        h_true_rowp[i] += nvals;
    }
    printf("nvals = %d\n", nvals);

    int n_print = min(32, N);
    printf("row counts: ");
    printVec<int>(n_print, h_rowCounts);
    printf("ref serial rowp: ");
    printVec<int>(n_print + 1, h_true_rowp);

    // TODO : call GPU kernel now
    int *d_rowCounts, *d_rowp;
    cudaMalloc((void **)&d_rowCounts, (N+1) * sizeof(int));
    cudaMalloc((void **)&d_rowp, (N+1) * sizeof(int));
    cudaMemcpy(d_rowCounts, h_rowCounts, N * sizeof(int), cudaMemcpyHostToDevice);

    // use thrust here
    thrust::device_vector<int> d_counts(N);
    thrust::copy_n(thrust::device_pointer_cast(d_rowCounts), N, d_counts.begin());
    thrust::device_vector<int> d_rowp2(N+1);
    d_rowp2[0] = 0;
    thrust::inclusive_scan(d_counts.begin(), d_counts.end(), d_rowp2.begin() + 1);

    cudaMemcpy(d_rowp, thrust::raw_pointer_cast(d_rowp2.data()), (N+1) * sizeof(int), cudaMemcpyDeviceToDevice);

    // now copy to host to check..
    int *h_rowp = new int[N+1];
    cudaMemcpy(h_rowp, d_rowp, (N+1) * sizeof(int), cudaMemcpyDeviceToHost);

    printf("thrust GPU rowp: ");
    printVec<int>(n_print + 1, h_rowp);

    return 0;
};