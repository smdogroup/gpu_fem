#pragma once
#ifdef USE_GPU
#include <cuda_runtime.h>
#endif
#include "a2dcore.h"
#include "cuda_utils.h"
#include "math.h"

// perform SVD on 3x3 covariance matrix H
// need to implement myself since can't call from inside the __global__ other cusparse, cublas
// methods, etc. was going to try this one: jacobi SVD
// https://netlib.org/lapack/lawnspdf/lawn170.pdf which is fast for small matrices this is a nice
// one too, but a lot of steps: https://pages.cs.wisc.edu/~sifakis/papers/SVD_TR1690.pdf decided to
// use this exact solution for eigenvalues of a 3x3 matrix,
// https://dl.acm.org/doi/pdf/10.1145/355578.366316

// uncomment for debugging
// if (print) {
//     printf("rt-sigmas: [%.4e, %.4e, %.4e]\n", sigma[0], sigma[1], sigma[2]);
// }

// if (print) {
//     printf("VT-GS: [%.4e, %.4e, %.4e, %.4e, %.4e, %.4e, %.4e, %.4e, %.4e]\n", VT[0], VT[1],
//            VT[2], VT[3], VT[4], VT[5], VT[6], VT[7], VT[8]);
// }

// if (print) {
//     printf("U: [%.4e, %.4e, %.4e, %.4e, %.4e, %.4e, %.4e, %.4e, %.4e]\n", U[0], U[1], U[2],
//            U[3], U[4], U[5], U[6], U[7], U[8]);
// }

template <typename T>
__DEVICE__ void svd3x3(const T H[9], T sigma[3], T U[9], T VT[9], const bool print = false) {
    // TODO : test this routine on the host for debugging

    // so are given H a 3x3 matrix and we wish to find H = U * Sigma * V^T
    // we can find the singular values in the diagonal
    // from the 3x3 eigenvalue problem H^* H = V Sigma^2 V^T

    double pi = 3.141592653589723846;

    // let A = H^* H (note if complex I need to add cmplx conjugate's here for hermitian transpose)
    T A[9];
    A2D::MatMatMultCore3x3<T, A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(H, H, A);

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

    // printf("phi %.4e\n", phi);

    // then the three eigenvalues of A are sigma^2 eigvals
    sigma[0] = m + 2 * sqrt(p) * cos(phi);
    sigma[1] = m - sqrt(p) * (cos(phi) + sqrt(3.0) * sin(phi));
    sigma[2] = m - sqrt(p) * (cos(phi) - sqrt(3.0) * sin(phi));

    // if (print) {
    //     printf("sigmas: %.4e %.4e %.4e\n", sigma[0], sigma[1], sigma[2]);
    //     // for (int i = 0; i < 3; i++) {
    //     //     printf("sigma[%d] = %.4e\n", i, sigma[i]);
    //     // }
    // }

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

        // if (print) {
        //     printf("VT: %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e\n", VT[0], VT[1], VT[2],
        //     VT[3],
        //            VT[4], VT[5], VT[6], VT[7], VT[8]);
        // }

        if constexpr (std::is_same<T, A2D::ADScalar<double, 1>>::value) {
            if (print) {
                printf("m: (%.4e,%.4e)\n", m.value, m.deriv[0]);
                printf("p: (%.4e,%.4e)\n", p.value, p.deriv[0]);
                printf("q: (%.4e,%.4e)\n", q.value, q.deriv[0]);
                printf("phi: (%.4e,%.4e)\n", phi.value, phi.deriv[0]);
                for (int i = 0; i < 3; i++) {
                    printf("sigma[%d]: (%.4e,%.4e)\n", i, sigma[i].value, sigma[i].deriv[0]);
                }
                for (int i = 0; i < 9; i++) {
                    printf("VT[%d]: (%.4e,%.4e)\n", i, VT[i].value, VT[i].deriv[0]);
                }
            }
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

    if (print) {
        printf("VT-GS: [%.4e, %.4e, %.4e, %.4e, %.4e, %.4e, %.4e, %.4e, %.4e]\n", VT[0], VT[1],
               VT[2], VT[3], VT[4], VT[5], VT[6], VT[7], VT[8]);
    }

    // change sigma^0.5 => sigma because we had sigma^2 before
    for (int i = 0; i < 3; i++) {
        sigma[i] = sqrt(sigma[i]);
    }

    if (print) {
        printf("rt-sigmas: [%.4e, %.4e, %.4e]\n", sigma[0], sigma[1], sigma[2]);
    }

    {  // scope block for computing U
        // now we have sigma and VT in H = U * sigma * VT
        // U can be found by U = H * V * sigma^-1

        // first U (tmp) = H * VT^T = H * V
        A2D::MatMatMultCore3x3<T, A2D::MatOp::NORMAL, A2D::MatOp::TRANSPOSE>(H, VT, U);

        if (print) {
            printf("U1: [%.4e, %.4e, %.4e, %.4e, %.4e, %.4e, %.4e, %.4e, %.4e]\n", U[0], U[1], U[2],
                   U[3], U[4], U[5], U[6], U[7], U[8]);
        }

        // now find U = U (tmp) * sigma^-1 by scaling each column by 1.0/si (see if later I need
        // numerical stability for si near 0)
        for (int i = 0; i < 9; i++) {
            int irow = i / 3;
            int icol = i % 3;
            U[3 * irow + icol] /= sigma[icol];
        }
    }

    // if (print) {
    //     printf("U2: %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e\n", U[0], U[1], U[2], U[3],
    //     U[4],
    //            U[5], U[6], U[7], U[8]);
    // }

    // now we have completed the SVD
}

template <typename T>
__DEVICE__ void computeRotation(const T H[9], T R[9], const bool print = false) {
    T sigma[3], U[9], VT[9];
    svd3x3<T>(H, sigma, U, VT, print);

    // compute rotation matrix R = U * VT
    A2D::MatMatMultCore3x3<T>(U, VT, R);
}

template <typename T>
__DEVICE__ void computeRotation(const T H[9], T R[9], T S[9]) {
    T sigma[3], U[9], VT[9];
    svd3x3<T>(H, sigma, U, VT);

    // compute rotation matrix R = U * VT
    A2D::MatMatMultCore3x3<T>(U, VT, R);

    // compute symmetric matrix S = V * Sigma * VT
    T tmp[9];
    for (int i = 0; i < 3; i++) {
        A2D::VecScaleCore<T, 3>(sigma[i], &VT[3 * i], &tmp[3 * i]);
    }

    // S = V * tmp
    A2D::MatMatMultCore3x3<T, A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(VT, tmp, S);
}

template <typename T>
__DEVICE__ void svd_15x15_adjoint(T M[225], T adjoint[15], T rhs[15], T fa[3], T d[3], T R[9],
                                  T S[9]) {
    // the two adjoint eqns which become 15x15 are:
    // dL/dR = fA*d^T - X^T * S + R * Y = 0 [9 eqns]
    // dL/dS = XR + R^T X^T = 0             [6 eqns]
    // where R,S are known and X[9], Y[6] unknown adjoint variables as Y is sym

    // T M[225], T adjoint[15], T rhs[15] all in shared memory..

    // assemble the M matrix
    // first for R system
    // TODO : can parallelize this assembly some
    for (int eqn = 0; eqn < 9; eqn++) {
        int ieqn = eqn / 3, jeqn = eqn % 3;
        int jx = ieqn;  // bc of transpose

        // -X^T * S term
        for (int ix = 0; ix < 3; ix++) {
            int is = ix, js = jeqn;
            int xind = 3 * ix + jx;
            M[eqn * 15 + xind] = -S[3 * is + js];
        }
    }

    // the 9x6 R * Y flattened matrix for Y[6] unknown
    // Construct matrix A (9x6) using R[i] directly
    // A[0] = {R[0], R[1], R[2], 0, 0, 0};
    // A[1] = {R[0], R[1], R[2], R[1], R[2], 0};
    // A[2] = {R[0], R[1], R[2], R[2], 0, 0};
    // A[3] = {R[3], R[4], R[5], 0, 0, 0};
    // A[4] = {R[3], R[4], R[5], R[4], R[5], 0};
    // A[5] = {R[3], R[4], R[5], R[5], 0, 0};
    // A[6] = {R[6], R[7], R[8], 0, 0, 0};
    // A[7] = {R[6], R[7], R[8], R[7], R[8], 0};
    // A[8] = {R[6], R[7], R[8], R[8], 0, 0};

    // R * Y term
    int tri_rows[3] = {1, 1, 2};
    int tri_cols[3] = {3, 4, 3};
    for (int eqn = 0; eqn < 3; eqn++) {
        for (int j = 0; j < 3; j++) {
            // the main [0,1,2] [3,4,5], [6,7,8] section
            M[15 * eqn + 9 + j] = R[j];
            M[15 * (eqn + 3) + 9 + j] = R[j + 3];
            M[15 * (eqn + 6) + 9 + j] = R[j + 6];

            // the secondary triangular nz pattern
            M[15 * (3 * eqn + tri_rows[j]) + 9 + tri_cols[j]] = R[3 * eqn + tri_rows[j]];
        }
    }

    // XR + R^T X^T term
    // need to check whether this is correct
    int rowIndex = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            int idx = i * 3 + j;  // Linear index for 3x3 matrix R

            // Fill first 3 rows (equations)
            M[15 * (9 + rowIndex) + idx] = R[idx];  // XR part for first 3 rows
            if (i != j) {
                M[15 * (9 + rowIndex + 3) + idx] = R[idx];  // R^T X^T part for next 3 rows
            }
        }
        ++rowIndex;
    }

    // assemble the rhs
    memset(&rhs[0], 0.0, 9 * sizeof(T));
    for (int i = 0; i < 9; i++) {
        int row = i / 3, col = i % 3;
        rhs[i] = -fa[row] * d[col];  // outer product fA * d^T
    }

    // pass the system to
    _solve15x15(&M[0], &adjoint[0], &rhs[0]);

    //
}

template <typename T>
__DEVICE__ void _solve15x15(T *M, T *y, T *x, int num_aero_nodes) {
    // solve the 15x15 dense linear system M * x = y
    // for a single aero node

    // Perform Gaussian elimination (simplified version)
    for (int i = 0; i < 15; ++i) {
        // Each thread computes the pivot row
        if (threadIdx.x == i) {
            float pivot = M[i * 15 + i];
            // Normalize pivot row
            for (int j = 0; j < 15; ++j) {
                M[i * 15 + j] /= pivot;
            }
            y[i] /= pivot;
        }

        __syncthreads();  // Ensure that the pivot row is updated before proceeding

        // Eliminate below pivot
        for (int j = i + 1 + threadIdx.x; j < 15; j += blockDim.x) {
            if (j > i) {
                float factor = M[j * 15 + i];
                for (int k = 0; k < 15; ++k) {
                    M[j * 15 + k] -= factor * M[i * 15 + k];
                }
                y[j] -= factor * y[i];
            }
        }

        __syncthreads();  // Ensure elimination step is complete
    }

    // Back-substitution (starting from last row to first row)
    for (int i = 14; i >= 0; --i) {
        if (threadIdx.x == i) {
            // Each thread does its part of the back-substitution
            for (int j = i + 1; j < 15; ++j) {
                y[i] -= M[i * 15 + j] * y[j];
            }
        }
        __syncthreads();  // Synchronize before moving to the next row
    }

    // Write back the solution
    for (int i = threadIdx.x; i < 15; i += blockDim.x) {
        x[i] = y[i];
    }
}

// template <typename T>
// __HOST_DEVICE__ void computeRotation(const T H[9], T R[9], T S[9]) {
//     T sigma[3], U[9], VT[9];
//     // svd3x3_cubic<T>(H, sigma, U, VT);
//     // svd3x3_givens<T>(H, sigma, U, VT);
//     svd3x3_QR<T>(H, sigma, U, VT);

//     // compute rotation matrix R = U * VT
//     A2D::MatMatMultCore3x3<T>(U, VT, R);

//     // compute symmetric matrix S = V * Sigma * VT
//     T tmp[9];
//     for (int i = 0; i < 3; i++) {
//         A2D::VecScaleCore<T, 3>(sigma[i], &VT[3 * i], &tmp[3 * i]);
//     }

//     // S = V * tmp
//     A2D::MatMatMultCore3x3<T, A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(VT, tmp, S);
// }