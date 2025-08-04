#include "vec_rnx_layout.h"

#include <cstring>

#include "../../spqlios/arithmetic/vec_rnx_arithmetic.h"

#ifdef VALGRIND_MEM_TESTS
#include "valgrind/memcheck.h"
#endif

#define CANARY_PADDING (1024)
#define GARBAGE_VALUE (242)

rnx_vec_f64_layout::rnx_vec_f64_layout(uint64_t n, uint64_t size, uint64_t slice) : n(n), size(size), slice(slice) {
  REQUIRE_DRAMATICALLY(is_pow2(n), "not a power of 2" << n);
  REQUIRE_DRAMATICALLY(slice >= n, "slice too small" << slice << " < " << n);
  this->region = (uint8_t*)malloc(size * slice * sizeof(int64_t) + 2 * CANARY_PADDING);
  this->data_start = (double*)(region + CANARY_PADDING);
  // ensure that any invalid value is kind-of garbage
  memset(region, GARBAGE_VALUE, size * slice * sizeof(int64_t) + 2 * CANARY_PADDING);
  // mark inter-slice memory as not accessible
#ifdef VALGRIND_MEM_TESTS
  VALGRIND_MAKE_MEM_NOACCESS(region, CANARY_PADDING);
  VALGRIND_MAKE_MEM_NOACCESS(region + size * slice * sizeof(int64_t) + CANARY_PADDING, CANARY_PADDING);
  for (uint64_t i = 0; i < size; ++i) {
    VALGRIND_MAKE_MEM_UNDEFINED(data_start + i * slice, n * sizeof(int64_t));
  }
  if (size != slice) {
    for (uint64_t i = 0; i < size; ++i) {
      VALGRIND_MAKE_MEM_NOACCESS(data_start + i * slice + n, (slice - n) * sizeof(int64_t));
    }
  }
#endif
}

rnx_vec_f64_layout::~rnx_vec_f64_layout() { free(region); }

rnx_f64 rnx_vec_f64_layout::get_copy_zext(uint64_t index) const {
  if (index < size) {
    return rnx_f64(n, data_start + index * slice);
  } else {
    return rnx_f64::zero(n);
  }
}

rnx_f64 rnx_vec_f64_layout::get_copy(uint64_t index) const {
  REQUIRE_DRAMATICALLY(index < size, "index overflow: " << index << " / " << size);
  return rnx_f64(n, data_start + index * slice);
}

reim_fft64vec rnx_vec_f64_layout::get_dft_copy_zext(uint64_t index) const {
  if (index < size) {
    return reim_fft64vec(n, data_start + index * slice);
  } else {
    return reim_fft64vec::zero(n);
  }
}

reim_fft64vec rnx_vec_f64_layout::get_dft_copy(uint64_t index) const {
  REQUIRE_DRAMATICALLY(index < size, "index overflow: " << index << " / " << size);
  return reim_fft64vec(n, data_start + index * slice);
}

void rnx_vec_f64_layout::set(uint64_t index, const rnx_f64& elem) {
  REQUIRE_DRAMATICALLY(index < size, "index overflow: " << index << " / " << size);
  REQUIRE_DRAMATICALLY(elem.nn() == n, "incompatible ring dimensions: " << elem.nn() << " / " << n);
  elem.save_as(data_start + index * slice);
}

double* rnx_vec_f64_layout::data() { return data_start; }
const double* rnx_vec_f64_layout::data() const { return data_start; }

void rnx_vec_f64_layout::fill_random(double log2bound) {
  for (uint64_t i = 0; i < size; ++i) {
    set(i, rnx_f64::random_log2bound(n, log2bound));
  }
}

thash rnx_vec_f64_layout::content_hash() const {
  test_hasher hasher;
  for (uint64_t i = 0; i < size; ++i) {
    hasher.update(data() + i * slice, n * sizeof(int64_t));
  }
  return hasher.hash();
}

fft64_rnx_vmp_pmat_layout::fft64_rnx_vmp_pmat_layout(uint64_t n, uint64_t nrows, uint64_t ncols)
    : nn(n),
      nrows(nrows),
      ncols(ncols),  //
      data((RNX_VMP_PMAT*)alloc64(nrows * ncols * nn * 8)) {}

double* fft64_rnx_vmp_pmat_layout::get_addr(uint64_t row, uint64_t col, uint64_t blk) const {
  REQUIRE_DRAMATICALLY(row < nrows, "row overflow: " << row << " / " << nrows);
  REQUIRE_DRAMATICALLY(col < ncols, "col overflow: " << col << " / " << ncols);
  REQUIRE_DRAMATICALLY(blk < nn / 8, "block overflow: " << blk << " / " << (nn / 8));
  double* d = (double*)data;
  if (col == (ncols - 1) && (ncols % 2 == 1)) {
    // special case: last column out of an odd column number
    return d + blk * nrows * ncols * 8  // major: blk
           + col * nrows * 8            // col == ncols-1
           + row * 8;
  } else {
    // general case: columns go by pair
    return d + blk * nrows * ncols * 8    // major: blk
           + (col / 2) * (2 * nrows) * 8  // second: col pair index
           + row * 2 * 8                  // third: row index
           + (col % 2) * 8;               // minor: col in colpair
  }
}

reim4_elem fft64_rnx_vmp_pmat_layout::get(uint64_t row, uint64_t col, uint64_t blk) const {
  return reim4_elem(get_addr(row, col, blk));
}
reim4_elem fft64_rnx_vmp_pmat_layout::get_zext(uint64_t row, uint64_t col, uint64_t blk) const {
  REQUIRE_DRAMATICALLY(blk < nn / 8, "block overflow: " << blk << " / " << (nn / 8));
  if (row < nrows && col < ncols) {
    return reim4_elem(get_addr(row, col, blk));
  } else {
    return reim4_elem::zero();
  }
}
void fft64_rnx_vmp_pmat_layout::set(uint64_t row, uint64_t col, uint64_t blk, const reim4_elem& value) const {
  value.save_as(get_addr(row, col, blk));
}

fft64_rnx_vmp_pmat_layout::~fft64_rnx_vmp_pmat_layout() { spqlios_free(data); }

reim_fft64vec fft64_rnx_vmp_pmat_layout::get_zext(uint64_t row, uint64_t col) const {
  if (row >= nrows || col >= ncols) {
    return reim_fft64vec::zero(nn);
  }
  if (nn < 8) {
    // the pmat is just col major
    double* addr = (double*)data + (row + col * nrows) * nn;
    return reim_fft64vec(nn, addr);
  }
  // otherwise, reconstruct it block by block
  reim_fft64vec res(nn);
  for (uint64_t blk = 0; blk < nn / 8; ++blk) {
    reim4_elem v = get(row, col, blk);
    res.set_blk(blk, v);
  }
  return res;
}
void fft64_rnx_vmp_pmat_layout::set(uint64_t row, uint64_t col, const reim_fft64vec& value) {
  REQUIRE_DRAMATICALLY(row < nrows, "row overflow: " << row << " / " << nrows);
  REQUIRE_DRAMATICALLY(col < ncols, "row overflow: " << col << " / " << ncols);
  if (nn < 8) {
    // the pmat is just col major
    double* addr = (double*)data + (row + col * nrows) * nn;
    value.save_as(addr);
    return;
  }
  // otherwise, reconstruct it block by block
  for (uint64_t blk = 0; blk < nn / 8; ++blk) {
    reim4_elem v = value.get_blk(blk);
    set(row, col, blk, v);
  }
}
void fft64_rnx_vmp_pmat_layout::fill_random(double log2bound) {
  for (uint64_t row = 0; row < nrows; ++row) {
    for (uint64_t col = 0; col < ncols; ++col) {
      set(row, col, reim_fft64vec::random(nn, log2bound));
    }
  }
}

fft64_rnx_svp_ppol_layout::fft64_rnx_svp_ppol_layout(uint64_t n)
    : nn(n),  //
      data((RNX_SVP_PPOL*)alloc64(nn * 8)) {}

reim_fft64vec fft64_rnx_svp_ppol_layout::get_copy() const { return reim_fft64vec(nn, (double*)data); }

void fft64_rnx_svp_ppol_layout::set(const reim_fft64vec& value) { value.save_as((double*)data); }

void fft64_rnx_svp_ppol_layout::fill_dft_random(uint64_t log2bound) { set(reim_fft64vec::dft_random(nn, log2bound)); }

void fft64_rnx_svp_ppol_layout::fill_random(double log2bound) { set(reim_fft64vec::random(nn, log2bound)); }

fft64_rnx_svp_ppol_layout::~fft64_rnx_svp_ppol_layout() { spqlios_free(data); }
thash fft64_rnx_svp_ppol_layout::content_hash() const { return test_hash(data, nn * sizeof(double)); }