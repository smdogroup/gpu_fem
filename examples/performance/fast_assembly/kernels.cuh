#pragma once
#include "efficient_math.h"
#include "a2dcore.h"

__global__ void test_dotprod() {
    using T = double;
    T a[3], b[3];
    for (int i = 0; i < 3; i++) {
        a[i] = 0.1234 + 0.5732 * i;
        b[i] = 2 * i;
    }
    T c = A2D::VecDotCore<T,3>(a,b);
}

__global__ void test_matinv(double *b) {
    using T = double;
    T A[9], Ainv[9];
    for (int i = 0; i < 9; i++) A[i] = 0.1 + 0.2 * i;
    for (int i = 0; i < 3; i++) A[3*i+i] += 5.0;
    // A2D::MatInvCore<T, 3>(A, Ainv);
    // MatInvNoInline<T>(A, Ainv);
    // MatInvInline<T>(A, Ainv);
    MatInvFast<T>(A, Ainv);
    // for (int i = 0; i < 9; i++) printf("Ainv[%d] = %.4e\n", i, Ainv[i]);

    T A_Ainv[9];
    for (int i = 0; i < 9; i++) A_Ainv[i] = Ainv[i];
    // // A2D::MatMatMultCore3x3<T>(A, Ainv, A_Ainv);
    // MatMultCoreInline<T>(A, Ainv, A_Ainv);

    for (int i = 0; i < 9; i++) atomicAdd(b, A_Ainv[i]);
}

__global__ void test_smallreg_kernel(double *b) {
    using T = double;

    // Declare matrix values directly (no A[9])
    T a00 = 5.1, a01 = 0.3, a02 = 0.5;
    T a10 = 0.7, a11 = 5.5, a12 = 0.9;
    T a20 = 1.1, a21 = 1.3, a22 = 5.9;

    // Compute determinant (fully fused)
    T det = a00 * (a11 * a22 - a12 * a21)
          - a01 * (a10 * a22 - a12 * a20)
          + a02 * (a10 * a21 - a11 * a20);
    T invdet = 1.0 / det;

    // Write inverse directly, no Ainv[9]
    T s = 0.0;
    s += (a11 * a22 - a12 * a21) * invdet;
    s += (a02 * a21 - a01 * a22) * invdet;
    s += (a01 * a12 - a02 * a11) * invdet;
    s += (a12 * a20 - a10 * a22) * invdet;
    s += (a00 * a22 - a02 * a20) * invdet;
    s += (a02 * a10 - a00 * a12) * invdet;
    s += (a10 * a21 - a11 * a20) * invdet;
    s += (a01 * a20 - a00 * a21) * invdet;
    s += (a00 * a11 - a01 * a10) * invdet;

    atomicAdd(b, s);
}