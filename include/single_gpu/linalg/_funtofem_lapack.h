#ifndef FUNTOFEM_LAPACK_H
#define FUNTOFEM_LAPACK_H

/*
  This file contains the definitions of several LAPACK/BLAS functions.
*/

// #include "TransferScheme.h"

template <typename T>
T doublePart(T val) {
    return val;
    // TODO : go back later and extend this to complex properly
}

#ifdef __cplusplus
extern "C" {
#endif

namespace Funtofem {

#define LAPACKsyevd dsyevd_
#define LAPACKdgesvd dgesvd_
#define LAPACKzgesvd zgesvd_
#define LAPACKdgelss dgelss_
#define LAPACKzgelss zgelss_

#ifdef FUNTOFEM_USE_COMPLEX
#define LAPACKgetrf zgetrf_
#define LAPACKgetrs zgetrs_
#define BLASgemv zgemv_
#define BLASgemm zgemm_
#else
#define LAPACKgetrf dgetrf_
#define LAPACKgetrs dgetrs_
#define BLASgemv dgemv_
#define BLASgemm dgemm_
#endif

// // Compute an LU factorization of a matrix
// extern void LAPACKgetrf(int *m, int *n, double *a, int *lda, int *ipiv, int *info);

// // This routine solves a system of equations with a factored matrix
// extern void LAPACKgetrs(const char *c, int *n, int *nrhs, const double *a, int *lda,
//                         const int *ipiv, double *b, int *ldb, int *info);

// Compute the eigenvalues of a symmetric matrix
extern void LAPACKsyevd(const char *jobz, const char *uplo, int *N, double *A, int *lda, double *w,
                        double *work, int *lwork, int *iwork, int *liwork, int *info);

// // Compute the SVD decomposition of a matrix
// extern void LAPACKdgesvd(const char *jobu, const char *jobvt, int *m, int *n, double *a, int
// *lda,
//                          double *s, double *u, int *ldu, double *vt, int *ldvt, double *work,
//                          int *lwork, int *info);

// // Compute the complex SVD decomposition of a matrix
// extern void LAPACKzgesvd(const char *jobu, const char *jobvt, int *m, int *n, double *a, int
// *lda,
//                          double *s, double *u, int *ldu, double *vt, int *ldvt, double *work,
//                          int *lwork, double *rwork, int *info);

// // Level 2 BLAS routines
// // y = alpha * A * x + beta * y, for a general matrix
// extern void BLASgemv(const char *c, int *m, int *n, double *alpha, double *a, int *lda, double
// *x,
//                      int *incx, double *beta, double *y, int *incy);

// // Level 3 BLAS routines
// // C := alpha*op( A )*op( B ) + beta*C,
// extern void BLASgemm(const char *ta, const char *tb, int *m, int *n, int *k, double *alpha,
//                      double *a, int *lda, double *b, int *ldb, double *beta, double *c, int
//                      *ldc);

// // Solve an over or underdetermined system of equations
// extern void LAPACKdgelss(int *m, int *n, int *nrhs, double *a, int *lda, double *b, int *ldb,
//                          double *s, double *rcond, int *rank, double *work, int *lwork, int
//                          *info);

// extern void LAPACKzgelss(int *m, int *n, int *nrhs, double *a, int *lda, double *b, int *ldb,
//                          double *s, double *rcond, int *rank, double *work, int *lwork,
//                          double *rwork, int *info);

}  // namespace Funtofem

#ifdef __cplusplus
}
#endif

#endif  // FUNTOFEM_LAPACK_H