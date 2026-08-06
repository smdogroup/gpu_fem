#pragma once
#ifdef USE_GPU
#include <cuda_runtime.h>
#endif
#include "a2dcore.h"
#include "cuda_utils.h"
#include "math.h"
#include "utils.h"

// perform SVD on 3x3 covariance matrix H
// need to implement myself since can't call from inside the __global__ other cusparse, cublas
// methods, etc. was going to try this one: jacobi SVD
// https://netlib.org/lapack/lawnspdf/lawn170.pdf which is fast for small matrices this is a nice
// one too, but a lot of steps: https://pages.cs.wisc.edu/~sifakis/papers/SVD_TR1690.pdf decided to
// use this exact solution for eigenvalues of a 3x3 matrix,
// https://dl.acm.org/doi/pdf/10.1145/355578.366316

template <typename T>
__HOST_DEVICE__ void eig3x3_cubic(const T A[9], T sigma[3], T VT[9]) {
    // 3x3 analytic eigenvalue solve using cubic formula style approach + cross product method for
    // eigenvectors this approach isn't giving the right eigenvectors though with cross product null
    // space method

    double pi = 3.141592653589723846;

    // exact eigenvalues of a 3x3 matrix from this source,
    // https://dl.acm.org/doi/pdf/10.1145/355578.366316 using cubic equation formula (Cardano's),
    // uses invariants => trace(A), det(A) and the squared one

    // m = trace(A)/3
    T m = (A[0] + A[4] + A[8]) / 3.0;

    // p = 1/6 * sum square elements of A-mI
    // q = 0.5 * det(A - mI)
    T p, q;
    {
        T AmI[9];
        for (int i = 0; i < 9; i++) {
            AmI[i] = A[i];
        }
        AmI[0] -= m;
        AmI[4] -= m;
        AmI[8] -= m;

        q = 0.5 * A2D::MatDetCore<T, 3>(AmI);

        p = 0.0;
        for (int i = 0; i < 9; i++) {
            p += AmI[i] * AmI[i];
        }
        p /= 6.0;
    }

    // printf("m %.4e p %.4e q %.4e\n", m, p, q);

    // we now have our three invariants m, p, q
    // we also need this angle from trigonomery phi
    T phi = atan(sqrt(p * p * p - q * q) / q) / 3.0;
    // ensures phi is in [0,pi] and ? symbol prevents warp divergence
    phi += phi < 0.0 ? pi : 0.0;

    // then the three eigenvalues of A are sigma^2 eigvals
    sigma[0] = m + 2 * sqrt(p) * cos(phi);
    sigma[1] = m - sqrt(p) * (cos(phi) + sqrt(3.0) * sin(phi));
    sigma[2] = m - sqrt(p) * (cos(phi) - sqrt(3.0) * sin(phi));

    // now that we have S diag matrix, how do we get V and U?
    // for V we can solve the system (A - sigma[i] * I) * vi = 0 for each S[i]
    // one cool way to find each vi is to take the cross product of the first two rows of Bi = A -
    // S[i] * I
    {  // scope block here
        for (int ieig = 0; ieig < 3; ieig++) {
            // compute B_i = A - lam_i *I
            T B[9];
            for (int i = 0; i < 9; i++) {
                B[i] = A[i];
            }
            B[0] -= sigma[ieig];
            B[4] -= sigma[ieig];
            B[8] -= sigma[ieig];

            // now take cross product of each pair of rows and add them to get null space vector
            T c[3], tmp[3];
            for (int i = 0; i < 3; i++) {
                c[i] = 0.0;
            }
            A2D::VecCrossCore<T>(&B[0], &B[3], tmp);
            A2D::VecAddCore<T, 3>(tmp, c);
            A2D::VecCrossCore<T>(&B[0], &B[6], tmp);
            A2D::VecAddCore<T, 3>(tmp, c);
            A2D::VecCrossCore<T>(&B[3], &B[6], tmp);
            A2D::VecAddCore<T, 3>(tmp, c);
            // normalize c
            T cnorm = sqrt(A2D::VecDotCore<T, 3>(c, c));  //  + 1e-12
            A2D::VecScaleCore<T, 3>(1.0 / cnorm, c, &VT[3 * ieig]);
        }

        // now need to apply Graham-Schmidt process on the V matrix
        // first v2 -= <v1,v2> v1
        T v12 = A2D::VecDotCore<T, 3>(&VT[0], &VT[3]);
        A2D::VecAddCore<T, 3>(-v12, &VT[0], &VT[3]);

        // normalize v2
        T norm2 = sqrt(A2D::VecDotCore<T, 3>(&VT[3], &VT[3])) + 1e-12;
        A2D::VecScaleCore<T, 3>(1.0 / norm2, &VT[3], &VT[3]);

        // then v3 -= <v1,v3> v1
        T v13 = A2D::VecDotCore<T, 3>(&VT[0], &VT[6]);
        A2D::VecAddCore<T, 3>(-v13, &VT[0], &VT[6]);

        // v3 -= <v2,v3> v2
        T v23 = A2D::VecDotCore<T, 3>(&VT[3], &VT[6]);
        A2D::VecAddCore<T, 3>(-v23, &VT[3], &VT[6]);

        // normalize v3
        T norm3 = sqrt(A2D::VecDotCore<T, 3>(&VT[6], &VT[6])) + 1e-12;
        A2D::VecScaleCore<T, 3>(1.0 / norm3, &VT[6], &VT[6]);
    }  // end of scope block for getting VT
}

template <typename T, int N>
__HOST_DEVICE__ void condSwap(bool swap, T vec1[], T vec2[]) {
    T tmp[N];
    for (int i = 0; i < N; i++) {
        tmp[i] = vec1[i];
        vec1[i] = swap ? vec2[i] : vec1[i];
        vec2[i] = swap ? tmp[i] : vec2[i];
    }
}

template <typename T, int Niter, bool smoothed = true>
__HOST_DEVICE__ void eig3x3_givens(T A[9], T sigma[3], T VT[9], double rhoKS = 100.0) {
    // https://pages.cs.wisc.edu/~sifakis/papers/SVD_TR1690.pdf
    T V[9], Q[9], tmp[9];
    // lower rhoKS more choppy load transfer
    // best is around 100 usually

    // initialize V to identify matrix
    for (int i = 0; i < 9; i++) {
        V[i] = 0.0;
    }
    for (int j = 0; j < 3; j++) {
        V[3 * j + j] = 1.0;
    }

    // printf("Ain: ");
    // printVec<T>(9, A);

    // printf("V: ");
    // printVec<T>(9, V);

    // modify A in place using givens rotations each time
    for (int i = 0; i < Niter; i++) {
        // cycle through each pair of (i,j) where i neq j cycle 3
        for (int cycle = 0; cycle < 3; cycle++) {
            // compute the new Q givens matrix
            int i0 = cycle, i1 = (cycle + 1) % 3, i2 = (cycle + 2) % 3;
            T a11 = A[3 * i0 + i0], a12 = A[3 * i0 + i1],
              a22 = A[3 * i1 + i1];  // assumes A sym here
            T omega = 1.0 / sqrt(a12 * a12 + (a11 - a22) * (a11 - a22));

            T s, c;
            bool b;
            T a_sum = a12 * a12 + (a11 - a22) * (a11 - a22);
            if constexpr (smoothed) {
                T w1 = exp(rhoKS * a12 * a12 / a_sum);
                T w2 = exp(rhoKS * (a11 - a22) * (a11 - a22) / a_sum);
                T sum = w1 + w2;
                s = (w2 * omega * a12 + w1 * sqrt(0.5)) / sum;
                c = (w2 * omega * (a11 - a22) + w1 * sqrt(0.5)) / sum;

                // b = a12 * a12 < (a11 - a22) * (a11 - a22);
                // T s2 = b ? omega * a12 : sqrt(0.5);
                // T c2 = b ? omega * (a11 - a22) : sqrt(0.5);
                // printf("a12*a12 %.4e\n", a12 * a12);
                // printf("(a11-a22)^2 %.4e\n", (a11 - a22) * (a11 - a22));
                // printf("s %.4e, c %.4e, s2 %.4e, c2 %.4e\n", s, c, s2, c2);

            } else {
                b = a12 * a12 < (a11 - a22) * (a11 - a22);
                s = b ? omega * a12 : sqrt(0.5);
                c = b ? omega * (a11 - a22) : sqrt(0.5);
            }

            // reset Q to zero
            for (int i = 0; i < 9; i++) {
                Q[i] = 0.0;
            }
            Q[3 * i0 + i0] = c;
            Q[3 * i0 + i1] = -s;
            Q[3 * i1 + i0] = s;
            Q[3 * i1 + i1] = c;
            Q[3 * i2 + i2] = 1.0;

            // printf("Q: ");
            // printVec<T>(9, Q);
            // if (cycle == 2) return;

            // update A and full V matrix
            // V = V * Q
            for (int i = 0; i < 9; i++) {
                tmp[i] = V[i];  // copy over to tmp
            }
            A2D::MatMatMultCore3x3<T>(tmp, Q, V);

            // A = Q.T @ A @ Q
            A2D::MatMatMultCore3x3<T, A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(Q, A, tmp);
            A2D::MatMatMultCore3x3<T>(tmp, Q, A);

            // after first phase check here
            // printf("Q=");
            // printVec<T>(9, Q);
            // printf("A=");
            // printVec<T>(9, A);
            // printf("V=");
            // printVec<T>(9, V);
            // if (cycle == 2) return;
        }
    }
    // printf("V=");
    // printVec<T>(9, V);

    // then transpose V into output VT
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            VT[3 * i + j] = V[3 * j + i];
        }
    }

    // also get the eigenvalues
    for (int i = 0; i < 3; i++) {
        sigma[i] = A[3 * i + i];
    }

    // now need a ? operator eigenvalue and eigvec sort in rows of VT
    for (int iouter = 0; iouter < 2; iouter++) {
        for (int i0 = 0; i0 < 2; i0++) {
            int i1 = i0 + 1;
            // now compare sigma[i0] to sigma[i1] with ? operator (for fast GPU performance)
            if constexpr (smoothed) {
                T sig_sum = sigma[i0] + sigma[i1];
                T w1 = exp(rhoKS * sigma[i1] / sig_sum);
                T w2 = exp(rhoKS * sigma[i0] / sig_sum);
                T w_sum = w1 + w2;
                T tmp2 = sigma[i0];
                sigma[i0] = (w1 * sigma[i1] + w2 * sigma[i0]) / w_sum;
                sigma[i1] = (w1 * tmp2 + w2 * sigma[i1]) / w_sum;

            } else {
                bool swap = sigma[i0] < sigma[i1];  // since want descending eigvals
                condSwap<T, 3>(swap, &VT[3 * i0], &VT[3 * i1]);

                // now also cond swap on sigma
                T tmp2 = sigma[i0];
                sigma[i0] = swap ? sigma[i1] : sigma[i0];
                sigma[i1] = swap ? tmp2 : sigma[i1];
            }
        }
    }
}

template <typename T, int N_iter, bool smoothed = true, bool can_swap = true>
__HOST_DEVICE__ void eig3x3_exact_givens(T A[9], T sigma[3], T VT[9], double rhoKS = 100.0) {
    // https://pages.cs.wisc.edu/~sifakis/papers/SVD_TR1690.pdf
    T V[9], Q[9], tmp[9];
    // lower rhoKS more choppy load transfer
    // best is around 100 usually

    // initialize V to identity matrix
    for (int i = 0; i < 9; i++) {
        V[i] = 0.0;
    }
    for (int j = 0; j < 3; j++) {
        V[3 * j + j] = 1.0;
    }

    // printf("Ain: ");
    // printVec<T>(9, A);

    // printf("V: ");
    // printVec<T>(9, V);

    // modify A in place using givens rotations each time
    for (int i = 0; i < N_iter; i++) {
        // cycle through each pair of (i,j) where i neq j cycle 3
        for (int cycle = 0; cycle < 3; cycle++) {
            // compute the new Q givens matrix
            int i0 = cycle, i1 = (cycle + 1) % 3, i2 = (cycle + 2) % 3;
            T a11 = A[3 * i0 + i0], a12 = A[3 * i0 + i1],
              a22 = A[3 * i1 + i1];  // assumes A sym here

            // one strategy with atan (may suffer from numerical instability if a11 near a22)
            // ------------------------
            T th = 0.5 * atan(2.0 * a12 / (a11 - a22 + 1e-14));
            T s = sin(th);
            T c = cos(th);

            // tried other more numerically stable Givens strategies, but above was still the best

            // reset Q to zero
            for (int i = 0; i < 9; i++) {
                Q[i] = 0.0;
            }
            Q[3 * i0 + i0] = c;
            Q[3 * i0 + i1] = -s;
            Q[3 * i1 + i0] = s;
            Q[3 * i1 + i1] = c;
            Q[3 * i2 + i2] = 1.0;

            // update A and full V matrix
            // V = V * Q
            for (int i = 0; i < 9; i++) {
                tmp[i] = V[i];  // copy over to tmp
            }
            A2D::MatMatMultCore3x3<T>(tmp, Q, V);

            // A = Q.T @ A @ Q
            A2D::MatMatMultCore3x3<T, A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(Q, A, tmp);
            A2D::MatMatMultCore3x3<T>(tmp, Q, A);
        }
    }

    // then transpose V into output VT
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            VT[3 * i + j] = V[3 * j + i];
        }
    }

    // also get the eigenvalues
    for (int i = 0; i < 3; i++) {
        sigma[i] = A[3 * i + i];
    }

    // now need a ? operator eigenvalue and eigvec sort in rows of VT
    if (can_swap) {
        for (int iouter = 0; iouter < 2; iouter++) {
            for (int i0 = 0; i0 < 2; i0++) {
                int i1 = i0 + 1;
                // now compare sigma[i0] to sigma[i1] with ? operator (for fast GPU performance)
                if constexpr (smoothed) {
                    T sig_sum = sigma[i0] + sigma[i1];
                    T w1 = exp(rhoKS * sigma[i1] / sig_sum);
                    T w2 = exp(rhoKS * sigma[i0] / sig_sum);
                    T w_sum = w1 + w2;
                    T tmp2 = sigma[i0];
                    sigma[i0] = (w1 * sigma[i1] + w2 * sigma[i0]) / w_sum;
                    sigma[i1] = (w1 * tmp2 + w2 * sigma[i1]) / w_sum;

                } else {
                    bool swap = sigma[i0] < sigma[i1];  // since want descending eigvals
                    condSwap<T, 3>(swap, &VT[3 * i0], &VT[3 * i1]);

                    // now also cond swap on sigma
                    T tmp2 = sigma[i0];
                    sigma[i0] = swap ? sigma[i1] : sigma[i0];
                    sigma[i1] = swap ? tmp2 : sigma[i1];
                }
            }
        }
    }
}

template <typename T>
__HOST_DEVICE__ void svd3x3_cubic(const T H[9], T sigma[3], T U[9], T VT[9],
                                  const bool print = false) {
    // so are given H a 3x3 matrix and we wish to find H = U * Sigma * V^T

    // now call the 3x3 eigenvalue problem on A = H^T H = V * Sigma^2 * V^T
    T A[9];
    A2D::MatMatMultCore3x3<T, A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(H, H, A);
    eig3x3_cubic<T>(A, sigma, VT);

    // now call the 3x3 eigenvalue problem on A = H H^T = U * Sigma^2 * U^T
    // technically this writes into sigma again, that's fine
    A2D::MatMatMultCore3x3<T, A2D::MatOp::NORMAL, A2D::MatOp::TRANSPOSE>(H, H, A);
    T UT[9];
    eig3x3_cubic<T>(A, sigma, UT);
    // un-transpose UT => U
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            U[3 * i + j] = UT[3 * j + i];
        }
    }

    // change sigma^0.5 => sigma because we had sigma^2 before
    for (int i = 0; i < 3; i++) {
        sigma[i] = sqrt(sigma[i]);
    }

    // now we have completed the SVD
}

template <typename T>
__HOST_DEVICE__ void svd3x3_givens(const T H[9], T sigma[3], T U[9], T VT[9],
                                   const bool print = false) {
    // so are given H a 3x3 matrix and we wish to find H = U * Sigma * V^T

    // now call the 3x3 eigenvalue problem on A = H^T H = V * Sigma^2 * V^T
    T A[9];
    A2D::MatMatMultCore3x3<T, A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(H, H, A);
    eig3x3_givens<T, 45>(A, sigma, VT);  // use 15 iterations of convergence

    // now call the 3x3 eigenvalue problem on A = H H^T = U * Sigma^2 * U^T
    // technically this writes into sigma again, that's fine
    A2D::MatMatMultCore3x3<T, A2D::MatOp::NORMAL, A2D::MatOp::TRANSPOSE>(H, H, A);
    T UT[9], sigma2[3];
    eig3x3_givens<T, 45>(A, sigma2, UT);  // use 15 iterations of convergence
    // un-transpose UT => U
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            U[3 * i + j] = UT[3 * j + i];
        }
    }

    // reorder U and VT to have sigma, sigma2 orders the same?

    // change sigma^0.5 => sigma because we had sigma^2 before
    for (int i = 0; i < 3; i++) {
        sigma[i] = sqrt(sigma[i]);
    }

    // now we have completed the SVD
}

template <typename T>
__HOST_DEVICE__ void _QR_3x3_decomp(T B[9], T U[9]) {
    for (int i = 0; i < 9; i++) {
        U[i] = 0.0;
    }
    for (int j = 0; j < 3; j++) {
        U[3 * j + j] = 1.0;
    }

    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < row; col++) {
            T aqq = B[3 * col + col];
            T apq = B[3 * row + col];
            T c = aqq / sqrt(aqq * aqq + apq * apq);
            T s = apq / sqrt(aqq * aqq + apq * apq);

            // construct givens rotation matrix
            T Q[9];
            for (int i = 0; i < 9; i++) {
                Q[i] = 0.0;
                int j = i % 3;
                Q[3 * j + j] = 1.0;
            }
            Q[3 * row + row] = c;
            Q[3 * col + col] = c;
            Q[3 * row + col] = s;
            Q[3 * col + row] = -s;

            // update the U matrix
            T tmp[9];
            for (int i = 0; i < 9; i++) {
                tmp[i] = B[i];
            }
            A2D::MatMatMultCore3x3<T, A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(Q, tmp, B);

            for (int i = 0; i < 9; i++) {
                tmp[i] = U[i];
            }
            A2D::MatMatMultCore3x3<T>(tmp, Q, U);
        }
    }
}

template <typename T, int N_iter = 15, bool exact_givens = true>
__HOST_DEVICE__ void svd3x3_QR(const T H[9], T sigma[3], T U[9], T VT[9], const bool print = false,
                               double rhoKS = 100.0) {
    // so are given H a 3x3 matrix and we wish to find H = U * Sigma * V^T

    // now call the 3x3 eigenvalue problem on A = H^T H = V * Sigma^2 * V^T
    T A[9];
    A2D::MatMatMultCore3x3<T, A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(H, H, A);
    if constexpr (exact_givens) {
        eig3x3_exact_givens<T, N_iter>(A, sigma, VT, rhoKS);
    } else {
        eig3x3_givens<T, N_iter>(A, sigma, VT, rhoKS);  // use 15 iterations of convergence
    }

    // now do QR decomposition on B = H * V to decomposite it B = QR similar to B=U*Sigma where U is
    // Q
    T B[9];
    A2D::MatMatMultCore3x3<T, A2D::MatOp::NORMAL, A2D::MatOp::TRANSPOSE>(H, VT, B);
    _QR_3x3_decomp<T>(B, U);

    // reorder U and VT to have sigma, sigma2 orders the same?

    // change sigma^0.5 => sigma because we had sigma^2 before
    for (int i = 0; i < 3; i++) {
        sigma[i] = sqrt(sigma[i]);
    }

    // now we have completed the SVD
}

template <typename T, bool exact_givens = true>
__HOST_DEVICE__ void computeRotation(const T H[9], T R[9], const bool print = false) {
    T sigma[3], U[9], VT[9];
    // svd3x3_cubic<T>(H, sigma, U, VT, print);
    // svd3x3_givens<T>(H, sigma, U, VT, print);

    // it's actually quite sensitive to this..
    double rhoKS = 100.0;
    // exact rotation has much more stable SVD jacobian
    // constexpr int N_iter = exact_givens ? 1 : 15;
    constexpr int N_iter = 15;
    // constexpr int N_iter = 5;

    // svd3x3_QR<T>(H, sigma, U, VT, print, rhoKS);
    svd3x3_QR<T, N_iter, exact_givens>(H, sigma, U, VT, print, rhoKS);

    // compute rotation matrix R = U * VT
    A2D::MatMatMultCore3x3<T>(U, VT, R);
}