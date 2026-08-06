#ifndef UTILS_H
#define UTILS_H

#include <complex>

#include "a2ddefs.h"
#include "mpi.h"

class VTKInfo {
 public:
  static const int VTK_LINE = 3;
  static const int VTK_TRIANGLE = 5;
  static const int VTK_QUADRILATERAL = 9;
  static const int VTK_TETRAHEDRON = 10;
  static const int VTK_HEXAHEDRAL = 12;
  static const int VTK_PRISM = 13;
  static const int VTK_PYRAMID = 14;
};

template <typename T>
constexpr MPI_Datatype get_mpi_type() {
  if constexpr (std::is_same<T, int>::value) {
    return MPI_INT;
  } else if constexpr (std::is_same<T, float>::value) {
    return MPI_FLOAT;
  } else if constexpr (std::is_same<T, double>::value) {
    return MPI_DOUBLE;
  } else if constexpr (std::is_same<T, std::complex<float>>::value) {
    return MPI_C_COMPLEX;
  } else if constexpr (std::is_same<T, std::complex<double>>::value) {
    return MPI_C_DOUBLE_COMPLEX;
  } else {
    return MPI_DATATYPE_NULL;
  }
}

template <typename T>
bool operator>(const std::complex<T>& l, const std::complex<T>& r) {
  return (std::real(l) > std::real(r));
}

template <typename T>
bool operator>=(const std::complex<T>& l, const std::complex<T>& r) {
  return (std::real(l) >= std::real(r));
}

template <typename T>
bool operator<(const std::complex<T>& l, const std::complex<T>& r) {
  return (std::real(l) < std::real(r));
}

template <typename T>
bool operator<=(const std::complex<T>& l, const std::complex<T>& r) {
  return (std::real(l) <= std::real(r));
}

template <typename T, typename R,
          typename = std::enable_if_t<A2D::is_scalar_type<R>::value>>
bool operator>(const std::complex<T>& l, const R& r) {
  return (std::real(l) > r);
}

template <typename T, typename R,
          typename = std::enable_if_t<A2D::is_scalar_type<R>::value>>
bool operator>=(const std::complex<T>& l, const R& r) {
  return (std::real(l) >= r);
}

template <typename T, typename R,
          typename = std::enable_if_t<A2D::is_scalar_type<R>::value>>
bool operator<(const std::complex<T>& l, const R& r) {
  return (std::real(l) < r);
}

template <typename T, typename R,
          typename = std::enable_if_t<A2D::is_scalar_type<R>::value>>
bool operator<=(const std::complex<T>& l, const R& r) {
  return (std::real(l) <= r);
}

template <typename T, typename L,
          typename = std::enable_if_t<A2D::is_scalar_type<L>::value>>
bool operator>(const L& l, const std::complex<T>& r) {
  return (l > std::real(r));
}

template <typename T, typename L,
          typename = std::enable_if_t<A2D::is_scalar_type<L>::value>>
bool operator>=(const L& l, const std::complex<T>& r) {
  return (l >= std::real(r));
}

template <typename T, typename L,
          typename = std::enable_if_t<A2D::is_scalar_type<L>::value>>
bool operator<(const L& l, const std::complex<T>& r) {
  return (l < std::real(r));
}

template <typename T, typename L,
          typename = std::enable_if_t<A2D::is_scalar_type<L>::value>>
bool operator<=(const L& l, const std::complex<T>& r) {
  return (l <= std::real(r));
}

#include "adscalar.h"

namespace FEM {

template <typename T,
          typename = std::enable_if_t<A2D::is_scalar_type<T>::value>>
T fabs(const T& v) {
  if (v < 0.0) {
    return -v;
  } else {
    return v;
  }
}

template <typename T>
std::complex<T> fabs(const std::complex<T>& v) {
  if (std::real(v) < 0.0) {
    return -v;
  } else {
    return v;
  }
}

template <typename T, int N>
A2D::ADScalar<T, N> fabs(const A2D::ADScalar<T, N>& v) {
  if (v < 0.0) {
    return -1.0 * v;
  } else {
    return v;
  }
}

}  // namespace FEM

/*!
  Compute the inverse of a matrix.

  A == A row-major ordered matrix of n x n
  Ainv == Array of size n x n
*/
template <typename T>
int ComputeInverse(const int n, T* A, int* ipiv, T* Ainv) {
  int fail = 0;

  for (int k = 0; k < n - 1; k++) {
    int nk = n * k;

    // Find the maximum value and use it as the pivot
    int r = k;
    double maxv = FEM::fabs(A[nk + k]);
    for (int j = k + 1; j < n; j++) {
      double t = FEM::fabs(A[n * j + k]);
      if (t > maxv) {
        maxv = t;
        r = j;
      }
    }

    ipiv[k] = r;

    // If a swap is required, swap the rows
    if (r != k) {
      int nr = n * r;
      for (int j = 0; j < n; j++) {
        T t = A[nk + j];
        A[nk + j] = A[nr + j];
        A[nr + j] = t;
      }
    }

    if (A[nk + k] == 0.0) {
      fail = k + 1;
      return fail;
    }

    for (int i = k + 1; i < n; i++) {
      A[n * i + k] = A[n * i + k] / A[nk + k];
    }

    for (int i = k + 1; i < n; i++) {
      int ni = n * i;
      for (int j = k + 1; j < n; j++) {
        A[ni + j] -= A[ni + k] * A[nk + j];
      }
    }
  }

  // Now, compute the matrix-inverse
  for (int k = 0; k < n; k++) {
    int ip = k;
    for (int i = 0; i < n - 1; i++) {
      if (ip == ipiv[i]) {
        ip = i;
      } else if (ip == i) {
        ip = ipiv[i];
      }
    }

    for (int i = 0; i < ip; i++) {
      Ainv[n * i + k] = 0.0;
    }

    Ainv[n * ip + k] = 1.0;

    for (int i = ip + 1; i < n; i++) {
      int ni = n * i;
      Ainv[ni + k] = 0.0;
      for (int j = ip; j < i; j++) {
        Ainv[ni + k] -= A[ni + j] * Ainv[n * j + k];
      }
    }

    for (int i = n - 1; i >= 0; i--) {
      int ni = n * i;
      for (int j = i + 1; j < n; j++) {
        Ainv[ni + k] -= A[ni + j] * Ainv[n * j + k];
      }
      Ainv[ni + k] = Ainv[ni + k] / A[ni + i];
    }
  }

  return fail;
}

#endif  // UTILS_H
