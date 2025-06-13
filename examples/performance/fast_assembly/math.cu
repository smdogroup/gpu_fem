#include "kernels.cuh"
#include "a2dcore.h"
#include <cuda_runtime.h>
#include <chrono>

/*
look at this paper to optimize stuff
https://pure.manchester.ac.uk/ws/portalfiles/portal/62970458/08214236.pdf
*/

int main() {
    // test_dotprod<<<1,1>>>();
    double *b;
    cudaMalloc((void **)&b, sizeof(double));
    cudaMemset(b, 0.0, sizeof(double));

    auto start0 = std::chrono::high_resolution_clock::now();
    int N = 1<<10;

    test_matinv<<<N,32>>>(b);
    // test_smallreg_kernel<<<N,32>>>(b);

    auto end0 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> mat_inv_time = end0 - start0;
    printf("elapsed time = %.4e\n", mat_inv_time.count());

    // test_matinv<<<1,1>>>(b);
    double *h_b = new double[1];
    cudaMemcpy(h_b, b, sizeof(double), cudaMemcpyDeviceToHost);
    printf("b = %.4e\n", h_b[0]);
}