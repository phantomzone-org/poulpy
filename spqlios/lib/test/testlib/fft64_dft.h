#ifndef SPQLIOS_FFT64_DFT_H
#define SPQLIOS_FFT64_DFT_H

#include "negacyclic_polynomial.h"
#include "reim4_elem.h"

class reim_fft64vec {
  std::vector<double> v;

 public:
  reim_fft64vec() = default;
  explicit reim_fft64vec(uint64_t n);
  reim_fft64vec(uint64_t n, const double* data);
  uint64_t nn() const;
  static reim_fft64vec zero(uint64_t n);
  /** random complex coefficients (unstructured) */
  static reim_fft64vec random(uint64_t n, double log2bound);
  /** random fft of a small int polynomial */
  static reim_fft64vec dft_random(uint64_t n, uint64_t log2bound);
  double* data();
  const double* data() const;
  void save_as(double* dest) const;
  reim4_elem get_blk(uint64_t blk) const;
  void set_blk(uint64_t blk, const reim4_elem& value);
};

reim_fft64vec operator+(const reim_fft64vec& a, const reim_fft64vec& b);
reim_fft64vec operator-(const reim_fft64vec& a, const reim_fft64vec& b);
reim_fft64vec operator*(const reim_fft64vec& a, const reim_fft64vec& b);
reim_fft64vec operator*(double coeff, const reim_fft64vec& v);
reim_fft64vec& operator+=(reim_fft64vec& a, const reim_fft64vec& b);
reim_fft64vec& operator-=(reim_fft64vec& a, const reim_fft64vec& b);

/** infty distance */
double infty_dist(const reim_fft64vec& a, const reim_fft64vec& b);

reim_fft64vec simple_fft64(const znx_i64& polynomial);
znx_i64 simple_rint_ifft64(const reim_fft64vec& fftvec);
rnx_f64 naive_ifft64(const reim_fft64vec& fftvec);
reim_fft64vec simple_fft64(const rnx_f64& polynomial);
rnx_f64 simple_ifft64(const reim_fft64vec& v);

#endif  // SPQLIOS_FFT64_DFT_H
