#include "zn_layouts.h"

zn32_pmat_layout::zn32_pmat_layout(uint64_t nrows, uint64_t ncols)
    : nrows(nrows),  //
      ncols(ncols),  //
      data((ZN32_VMP_PMAT*)malloc((nrows * ncols + 7) * sizeof(int32_t))) {}

zn32_pmat_layout::~zn32_pmat_layout() { free(data); }

int32_t* zn32_pmat_layout::get_addr(uint64_t row, uint64_t col) const {
  REQUIRE_DRAMATICALLY(row < nrows, "row overflow" << row << " / " << nrows);
  REQUIRE_DRAMATICALLY(col < ncols, "col overflow" << col << " / " << ncols);
  const uint64_t nblk = ncols >> 5;
  const uint64_t rem_ncols = ncols & 31;
  uint64_t blk = col >> 5;
  uint64_t col_rem = col & 31;
  if (blk < nblk) {
    // column is part of a full block
    return (int32_t*)data + blk * nrows * 32 + row * 32 + col_rem;
  } else {
    // column is part of the last block
    return (int32_t*)data + blk * nrows * 32 + row * rem_ncols + col_rem;
  }
}
int32_t zn32_pmat_layout::get(uint64_t row, uint64_t col) const { return *get_addr(row, col); }
int32_t zn32_pmat_layout::get_zext(uint64_t row, uint64_t col) const {
  if (row >= nrows || col >= ncols) return 0;
  return *get_addr(row, col);
}
void zn32_pmat_layout::set(uint64_t row, uint64_t col, int32_t value) { *get_addr(row, col) = value; }
void zn32_pmat_layout::fill_random() {
  int32_t* d = (int32_t*)data;
  for (uint64_t i = 0; i < nrows * ncols; ++i) d[i] = uniform_i64_bits(32);
}
thash zn32_pmat_layout::content_hash() const { return test_hash(data, nrows * ncols * sizeof(int32_t)); }

template <typename T>
std::vector<int32_t> vmp_product(const T* vec, uint64_t vec_size, uint64_t out_size, const zn32_pmat_layout& mat) {
  uint64_t rows = std::min(vec_size, mat.nrows);
  uint64_t cols = std::min(out_size, mat.ncols);
  std::vector<int32_t> res(out_size, 0);
  for (uint64_t j = 0; j < cols; ++j) {
    for (uint64_t i = 0; i < rows; ++i) {
      res[j] += vec[i] * mat.get(i, j);
    }
  }
  return res;
}

template std::vector<int32_t> vmp_product(const int8_t* vec, uint64_t vec_size, uint64_t out_size,
                                          const zn32_pmat_layout& mat);
template std::vector<int32_t> vmp_product(const int16_t* vec, uint64_t vec_size, uint64_t out_size,
                                          const zn32_pmat_layout& mat);
template std::vector<int32_t> vmp_product(const int32_t* vec, uint64_t vec_size, uint64_t out_size,
                                          const zn32_pmat_layout& mat);
