#ifndef SPQLIOS_FFT64_LAYOUTS_H
#define SPQLIOS_FFT64_LAYOUTS_H

#include "../../spqlios/arithmetic/vec_znx_arithmetic.h"
#include "fft64_dft.h"
#include "negacyclic_polynomial.h"
#include "reim4_elem.h"

/** @brief test layout for the VEC_ZNX_DFT */
struct fft64_vec_znx_dft_layout {
 public:
  const uint64_t nn;
  const uint64_t size;
  VEC_ZNX_DFT* const data;
  reim_vector_view view;
  /** @brief fill with random double values (unstructured) */
  void fill_random(double log2bound);
  /** @brief fill with random ffts of small int polynomials */
  void fill_dft_random(uint64_t log2bound);
  reim4_elem get(uint64_t idx, uint64_t blk) const;
  reim4_elem get_zext(uint64_t idx, uint64_t blk) const;
  void set(uint64_t idx, uint64_t blk, const reim4_elem& value);
  fft64_vec_znx_dft_layout(uint64_t n, uint64_t size);
  void fill_random_log2bound(uint64_t bits);
  void fill_dft_random_log2bound(uint64_t bits);
  double* get_addr(uint64_t idx);
  const double* get_addr(uint64_t idx) const;
  reim_fft64vec get_copy_zext(uint64_t idx) const;
  void set(uint64_t idx, const reim_fft64vec& value);
  thash content_hash() const;
  ~fft64_vec_znx_dft_layout();
};

/** @brief test layout for the VEC_ZNX_BIG */
class fft64_vec_znx_big_layout {
 public:
  const uint64_t nn;
  const uint64_t size;
  VEC_ZNX_BIG* const data;
  fft64_vec_znx_big_layout(uint64_t n, uint64_t size);
  void fill_random();
  znx_i64 get_copy(uint64_t index) const;
  znx_i64 get_copy_zext(uint64_t index) const;
  void set(uint64_t index, const znx_i64& value);
  thash content_hash() const;
  ~fft64_vec_znx_big_layout();
};

/** @brief test layout for the VMP_PMAT */
class fft64_vmp_pmat_layout {
 public:
  const uint64_t nn;
  const uint64_t nrows;
  const uint64_t ncols;
  VMP_PMAT* const data;
  fft64_vmp_pmat_layout(uint64_t n, uint64_t nrows, uint64_t ncols);
  double* get_addr(uint64_t row, uint64_t col, uint64_t blk) const;
  reim4_elem get(uint64_t row, uint64_t col, uint64_t blk) const;
  thash content_hash() const;
  reim4_elem get_zext(uint64_t row, uint64_t col, uint64_t blk) const;
  reim_fft64vec get_zext(uint64_t row, uint64_t col) const;
  void set(uint64_t row, uint64_t col, uint64_t blk, const reim4_elem& v) const;
  void set(uint64_t row, uint64_t col, const reim_fft64vec& value);
  /** @brief fill with random double values (unstructured) */
  void fill_random(double log2bound);
  /** @brief fill with random ffts of small int polynomials */
  void fill_dft_random(uint64_t log2bound);
  ~fft64_vmp_pmat_layout();
};

/** @brief test layout for the SVP_PPOL */
class fft64_svp_ppol_layout {
 public:
  const uint64_t nn;
  SVP_PPOL* const data;
  fft64_svp_ppol_layout(uint64_t n);
  thash content_hash() const;
  reim_fft64vec get_copy() const;
  void set(const reim_fft64vec&);
  /** @brief fill with random double values (unstructured) */
  void fill_random(double log2bound);
  /** @brief fill with random ffts of small int polynomials */
  void fill_dft_random(uint64_t log2bound);
  ~fft64_svp_ppol_layout();
};

/** @brief test layout for the CNV_PVEC_L */
class fft64_cnv_left_layout {
  const uint64_t nn;
  const uint64_t size;
  CNV_PVEC_L* const data;
  fft64_cnv_left_layout(uint64_t n, uint64_t size);
  reim4_elem get(uint64_t idx, uint64_t blk);
  thash content_hash() const;
  ~fft64_cnv_left_layout();
};

/** @brief test layout for the CNV_PVEC_R */
class fft64_cnv_right_layout {
  const uint64_t nn;
  const uint64_t size;
  CNV_PVEC_R* const data;
  fft64_cnv_right_layout(uint64_t n, uint64_t size);
  reim4_elem get(uint64_t idx, uint64_t blk);
  thash content_hash() const;
  ~fft64_cnv_right_layout();
};

#endif  // SPQLIOS_FFT64_LAYOUTS_H
