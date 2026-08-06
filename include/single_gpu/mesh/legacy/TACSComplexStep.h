#pragma once
#include <complex>

// Define the real part function for the complex data type
inline double TacsRealPart(const std::complex<double>& c) { return real(c); }

// Define the imaginary part function for the complex data type
inline double TacsImagPart(const std::complex<double>& c) { return imag(c); }

// Dummy function for real part
inline double TacsRealPart(const double& r) { return r; }

// There are issues with noise in the complex version of std::arctan
// We use th following definition to avoid issues with cs
inline std::complex<double> atan(const std::complex<double>& c) {
  double cReal = TacsRealPart(c);
  double cImag = TacsImagPart(c);
  std::complex<double> val =
      atan(TacsRealPart(cReal)) +
      (1 / (1 + pow(cReal, 2))) * cImag * std::complex<double>(0.0, 1);
  return val;
}

// Compute the absolute value
#ifndef FABS_COMPLEX_IS_DEFINED  // prevent redefinition
#define FABS_COMPLEX_IS_DEFINED
inline std::complex<double> fabs(const std::complex<double>& c) {
  if (real(c) < 0.0) {
    return -c;
  }
  return c;
}
#endif  // FABS_COMPLEX_IS_DEFINED