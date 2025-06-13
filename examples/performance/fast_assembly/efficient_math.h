#pragma once

template <typename T>
__device__ __noinline__ void MatInvNoInline(const T A[], T Ainv[]) {
    T det = (A[8] * (A[0] * A[4] - A[3] * A[1]) - A[7] * (A[0] * A[5] - A[3] * A[2]) +
             A[6] * (A[1] * A[5] - A[2] * A[4]));
    T detinv = 1.0 / det;

    Ainv[0] = (A[4] * A[8] - A[5] * A[7]) * detinv;
    Ainv[1] = -1.0 * (A[1] * A[8] - A[2] * A[7]) * detinv;
    Ainv[2] = (A[1] * A[5] - A[2] * A[4]) * detinv;

    Ainv[3] = -1.0 * (A[3] * A[8] - A[5] * A[6]) * detinv;
    Ainv[4] = (A[0] * A[8] - A[2] * A[6]) * detinv;
    Ainv[5] = -1.0 * (A[0] * A[5] - A[2] * A[3]) * detinv;

    Ainv[6] = (A[3] * A[7] - A[4] * A[6]) * detinv;
    Ainv[7] = -1.0 * (A[0] * A[7] - A[1] * A[6]) * detinv;
    Ainv[8] = (A[0] * A[4] - A[1] * A[3]) * detinv;
}

template <typename T>
__device__ __inline__ void MatInvInline(const T A[], T Ainv[]) {
    // default uses like 15-20 registers
    T det = (A[8] * (A[0] * A[4] - A[3] * A[1]) - A[7] * (A[0] * A[5] - A[3] * A[2]) +
             A[6] * (A[1] * A[5] - A[2] * A[4]));
    T detinv = 1.0 / det;

    Ainv[0] = (A[4] * A[8] - A[5] * A[7]) * detinv;
    Ainv[1] = -1.0 * (A[1] * A[8] - A[2] * A[7]) * detinv;
    Ainv[2] = (A[1] * A[5] - A[2] * A[4]) * detinv;

    Ainv[3] = -1.0 * (A[3] * A[8] - A[5] * A[6]) * detinv;
    Ainv[4] = (A[0] * A[8] - A[2] * A[6]) * detinv;
    Ainv[5] = -1.0 * (A[0] * A[5] - A[2] * A[3]) * detinv;

    Ainv[6] = (A[3] * A[7] - A[4] * A[6]) * detinv;
    Ainv[7] = -1.0 * (A[0] * A[7] - A[1] * A[6]) * detinv;
    Ainv[8] = (A[0] * A[4] - A[1] * A[3]) * detinv;
}

template <typename T>
__device__ __noinline__ void MatInvFast(const T A[], T Ainv[]) {
    // TODO: make this one faster (use fewer registers)
    // it's actually tricky to use a small amount of registers for this I think
    T r1, r2, r3, r4, r5;
    r1 = A[0], r2 = A[4], r3 = A[3], r4 = A[1];
    r1 *= r2, r3 *= r4;
    r1 -= r3;
    r2 = A[8];
    r1 *= r2;
    // now r1 holds the first minor det

    r2 = A[0], r3 = A[5], r4 = A[3], r5 = A[2];
    r2 *= r3;
    r4 *= r5;
    r2 -= r4;
    r3 = A[7];
    r2 *= r3;
    r1 -= r2;
    // now r1 is first - second minor det

    r2 = A[1], r3 = A[5], r4 = A[2], r5 = A[4];
    r2 *= r3;
    r4 *= r5;
    r2 -= r4;
    r3 = A[6];
    r2 *= r3;
    r1 += r2;
    // now r1 is the det

    r2 = r1, r3 = 1.0;
    r1 = r3 / r2;
    // now r1 is 1/det

    // entry 0
    r2 = A[4], r3 = A[8], r4 = A[5], r5 = A[7];
    r2 *= r3, r4 *= r5;
    r2 -= r4;
    r2 *= r1;
    Ainv[0] = r2;

    // entry 1
    r2 = A[1], r3 = A[8], r4 = A[2], r5 = A[7];
    r2 *= r3, r4 *= r5;
    r2 -= r4;
    r2 *= -r1;
    Ainv[1] = r2;

    // entry 2
    r2 = A[1], r3 = A[5], r4 = A[2], r5 = A[4];
    r2 *= r3, r4 *= r5;
    r2 -= r4;
    r2 *= r1;
    Ainv[2] = r2;

    // entry 3
    r2 = A[3], r3 = A[8], r4 = A[5], r5 = A[6];
    r2 *= r3, r4 *= r5;
    r2 -= r4;
    r2 *= -r1;
    Ainv[3] = r2;

    // entry 4
    r2 = A[0], r3 = A[8], r4 = A[2], r5 = A[6];
    r2 *= r3, r4 *= r5;
    r2 -= r4;
    r2 *= r1;
    Ainv[4] = r2;

    // entry 5
    r2 = A[0], r3 = A[5], r4 = A[2], r5 = A[3];
    r2 *= r3, r4 *= r5;
    r2 -= r4;
    r2 *= -r1;
    Ainv[5] = r2;

    // entry 6
    r2 = A[3], r3 = A[7], r4 = A[4], r5 = A[6];
    r2 *= r3, r4 *= r5;
    r2 -= r4;
    r2 *= r1;
    Ainv[6] = r2;

    // entry 7
    r2 = A[0], r3 = A[7], r4 = A[1], r5 = A[6];
    r2 *= r3, r4 *= r5;
    r2 -= r4;
    r2 *= -r1;
    Ainv[7] = r2;

    // entry 8
    r2 = A[0], r3 = A[4], r4 = A[1], r5 = A[3];
    r2 *= r3, r4 *= r5;
    r2 -= r4;
    r2 *= r1;
    Ainv[8] = r2;
}

template <typename T>
__device__ void MatMultCoreReg(const T A[], const T B[], T C[]) {
    // option 1
    C[0] = A[0] * B[0] + A[1] * B[3] + A[2] * B[6];
    C[1] = A[0] * B[1] + A[1] * B[4] + A[2] * B[7];
    C[2] = A[0] * B[2] + A[1] * B[5] + A[2] * B[8];
    C[3] = A[3] * B[0] + A[4] * B[3] + A[5] * B[6];
    C[4] = A[3] * B[1] + A[4] * B[4] + A[5] * B[7];
    C[5] = A[3] * B[2] + A[4] * B[5] + A[5] * B[8];
    C[6] = A[6] * B[0] + A[7] * B[3] + A[8] * B[6];
    C[7] = A[6] * B[1] + A[7] * B[4] + A[8] * B[7];
    C[8] = A[6] * B[2] + A[7] * B[5] + A[8] * B[8];
}

template <typename T>
__device__ __inline__ void MatMultCoreInline(const T A[], const T B[], T C[]) {
    // option 1
    C[0] = A[0] * B[0] + A[1] * B[3] + A[2] * B[6];
    C[1] = A[0] * B[1] + A[1] * B[4] + A[2] * B[7];
    C[2] = A[0] * B[2] + A[1] * B[5] + A[2] * B[8];
    C[3] = A[3] * B[0] + A[4] * B[3] + A[5] * B[6];
    C[4] = A[3] * B[1] + A[4] * B[4] + A[5] * B[7];
    C[5] = A[3] * B[2] + A[4] * B[5] + A[5] * B[8];
    C[6] = A[6] * B[0] + A[7] * B[3] + A[8] * B[6];
    C[7] = A[6] * B[1] + A[7] * B[4] + A[8] * B[7];
    C[8] = A[6] * B[2] + A[7] * B[5] + A[8] * B[8];
}

template <typename T>
__device__ __inline__ void MatMultCoreFast(const T A[], const T B[], T C[]) {
    // fast on GPUs, few registers

    C[0] = A[0] * B[0] + A[1] * B[3] + A[2] * B[6];
    C[1] = A[0] * B[1] + A[1] * B[4] + A[2] * B[7];
    C[2] = A[0] * B[2] + A[1] * B[5] + A[2] * B[8];
    C[3] = A[3] * B[0] + A[4] * B[3] + A[5] * B[6];
    C[4] = A[3] * B[1] + A[4] * B[4] + A[5] * B[7];
    C[5] = A[3] * B[2] + A[4] * B[5] + A[5] * B[8];
    C[6] = A[6] * B[0] + A[7] * B[3] + A[8] * B[6];
    C[7] = A[6] * B[1] + A[7] * B[4] + A[8] * B[7];
    C[8] = A[6] * B[2] + A[7] * B[5] + A[8] * B[8];
}
