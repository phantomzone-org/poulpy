#ifndef SPQLIOS_EXT_VEC_RNX_LAYOUT_H
#define SPQLIOS_EXT_VEC_RNX_LAYOUT_H

#include "../../spqlios/arithmetic/vec_rnx_arithmetic.h"
#include "fft64_dft.h"
#include "negacyclic_polynomial.h"
#include "reim4_elem.h"
#include "test_commons.h"

/** @brief a test memory layout for rnx i64 polynomials vectors */
class rnx_vec_f64_layout {
  uint64_t n;
  uint64_t size;
  uint64_t slice;
  double* data_start;
  uint8_t* region;

 public:
  // NO-COPY structure
  rnx_vec_f64_layout(const rnx_vec_f64_layout&) = delete;
  void operator=(const rnx_vec_f64_layout&) = delete;
  rnx_vec_f64_layout(rnx_vec_f64_layout&&) = delete;
  void operator=(rnx_vec_f64_layout&&) = delete;
  /** @brief initialises a memory layout */
  rnx_vec_f64_layout(uint64_t n, uint64_t size, uint64_t slice);
  /** @brief destructor */
  ~rnx_vec_f64_layout();

  /** @brief get a copy of item index index (extended with zeros) */
  rnx_f64 get_copy_zext(uint64_t index) const;
  /** @brief get a copy of item index index (extended with zeros) */
  rnx_f64 get_copy(uint64_t index) const;
  /** @brief get a copy of item index index (extended with zeros) */
  reim_fft64vec get_dft_copy_zext(uint64_t index) const;
  /** @brief get a copy of item index index (extended with zeros) */
  reim_fft64vec get_dft_copy(uint64_t index) const;

  /** @brief get a copy of item index index (index<size) */
  void set(uint64_t index, const rnx_f64& elem);
  /** @brief fill with random values */
  void fill_random(double log2bound = 0);
  /** @brief raw pointer access */
  double* data();
  /** @brief raw pointer access (const version) */
  const double* data() const;
  /** @brief content hashcode */
  thash content_hash() const;
};

/** @brief test layout for the VMP_PMAT */
class fft64_rnx_vmp_pmat_layout {
 public:
  const uint64_t nn;
  const uint64_t nrows;
  const uint64_t ncols;
  RNX_VMP_PMAT* const data;
  fft64_rnx_vmp_pmat_layout(uint64_t n, uint64_t nrows, uint64_t ncols);
  double* get_addr(uint64_t row, uint64_t col, uint64_t blk) const;
  reim4_elem get(uint64_t row, uint64_t col, uint64_t blk) const;
  thash content_hash() const;
  reim4_elem get_zext(uint64_t row, uint64_t col, uint64_t blk) const;
  reim_fft64vec get_zext(uint64_t row, uint64_t col) const;
  void set(uint64_t row, uint64_t col, uint64_t blk, const reim4_elem& v) const;
  void set(uint64_t row, uint64_t col, const reim_fft64vec& value);
  /** @brief fill with random double values (unstructured) */
  void fill_random(double log2bound);
  ~fft64_rnx_vmp_pmat_layout();
};

/** @brief test layout for the SVP_PPOL */
class fft64_rnx_svp_ppol_layout {
 public:
  const uint64_t nn;
  RNX_SVP_PPOL* const data;
  fft64_rnx_svp_ppol_layout(uint64_t n);
  thash content_hash() const;
  reim_fft64vec get_copy() const;
  void set(const reim_fft64vec&);
  /** @brief fill with random double values (unstructured) */
  void fill_random(double log2bound);
  /** @brief fill with random ffts of small int polynomials */
  void fill_dft_random(uint64_t log2bound);
  ~fft64_rnx_svp_ppol_layout();
};
#endif  // SPQLIOS_EXT_VEC_RNX_LAYOUT_H
