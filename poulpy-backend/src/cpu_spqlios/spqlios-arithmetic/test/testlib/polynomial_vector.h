#ifndef SPQLIOS_POLYNOMIAL_VECTOR_H
#define SPQLIOS_POLYNOMIAL_VECTOR_H

#include "negacyclic_polynomial.h"
#include "test_commons.h"

/** @brief a test memory layout for znx i64 polynomials vectors */
class znx_vec_i64_layout {
  uint64_t n;
  uint64_t size;
  uint64_t slice;
  int64_t* data_start;
  uint8_t* region;

 public:
  // NO-COPY structure
  znx_vec_i64_layout(const znx_vec_i64_layout&) = delete;
  void operator=(const znx_vec_i64_layout&) = delete;
  znx_vec_i64_layout(znx_vec_i64_layout&&) = delete;
  void operator=(znx_vec_i64_layout&&) = delete;
  /** @brief initialises a memory layout */
  znx_vec_i64_layout(uint64_t n, uint64_t size, uint64_t slice);
  /** @brief destructor */
  ~znx_vec_i64_layout();

  /** @brief get a copy of item index index (extended with zeros) */
  znx_i64 get_copy_zext(uint64_t index) const;
  /** @brief get a copy of item index index (extended with zeros) */
  znx_i64 get_copy(uint64_t index) const;
  /** @brief get a copy of item index index (index<size) */
  void set(uint64_t index, const znx_i64& elem);
  /** @brief fill with random values */
  void fill_random(uint64_t bits = 63);
  /** @brief raw pointer access */
  int64_t* data();
  /** @brief raw pointer access (const version) */
  const int64_t* data() const;
  /** @brief content hashcode */
  __uint128_t content_hash() const;
};

#endif  // SPQLIOS_POLYNOMIAL_VECTOR_H
