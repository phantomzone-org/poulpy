#ifndef SPQLIOS_EXT_ZN_LAYOUTS_H
#define SPQLIOS_EXT_ZN_LAYOUTS_H

#include "../../spqlios/arithmetic/zn_arithmetic.h"
#include "test_commons.h"

class zn32_pmat_layout {
 public:
  const uint64_t nrows;
  const uint64_t ncols;
  ZN32_VMP_PMAT* const data;
  zn32_pmat_layout(uint64_t nrows, uint64_t ncols);

 private:
  int32_t* get_addr(uint64_t row, uint64_t col) const;

 public:
  int32_t get(uint64_t row, uint64_t col) const;
  int32_t get_zext(uint64_t row, uint64_t col) const;
  void set(uint64_t row, uint64_t col, int32_t value);
  void fill_random();
  thash content_hash() const;
  ~zn32_pmat_layout();
};

template <typename T>
std::vector<int32_t> vmp_product(const T* vec, uint64_t vec_size, uint64_t out_size, const zn32_pmat_layout& mat);

#endif  // SPQLIOS_EXT_ZN_LAYOUTS_H
