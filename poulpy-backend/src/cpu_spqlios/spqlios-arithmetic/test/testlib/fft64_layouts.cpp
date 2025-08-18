#include "fft64_layouts.h"
#ifdef VALGRIND_MEM_TESTS
#include "valgrind/memcheck.h"
#endif

void* alloc64(uint64_t size) {
  static uint64_t _msk64 = -64;
  if (size == 0) return nullptr;
  uint64_t rsize = (size + 63) & _msk64;
  uint8_t* reps = (uint8_t*)spqlios_alloc(rsize);
  REQUIRE_DRAMATICALLY(reps != 0, "Out of memory");
#ifdef VALGRIND_MEM_TESTS
  VALGRIND_MAKE_MEM_NOACCESS(reps + size, rsize - size);
#endif
  return reps;
}

fft64_vec_znx_dft_layout::fft64_vec_znx_dft_layout(uint64_t n, uint64_t size)
    : nn(n),                                      //
      size(size),                                 //
      data((VEC_ZNX_DFT*)alloc64(n * size * 8)),  //
      view(n / 2, size, (double*)data) {}

fft64_vec_znx_dft_layout::~fft64_vec_znx_dft_layout() { spqlios_free(data); }

double* fft64_vec_znx_dft_layout::get_addr(uint64_t idx) {
  REQUIRE_DRAMATICALLY(idx < size, "index overflow " << idx << " / " << size);
  return ((double*)data) + idx * nn;
}
const double* fft64_vec_znx_dft_layout::get_addr(uint64_t idx) const {
  REQUIRE_DRAMATICALLY(idx < size, "index overflow " << idx << " / " << size);
  return ((double*)data) + idx * nn;
}
reim_fft64vec fft64_vec_znx_dft_layout::get_copy_zext(uint64_t idx) const {
  if (idx < size) {
    return reim_fft64vec(nn, get_addr(idx));
  } else {
    return reim_fft64vec::zero(nn);
  }
}
void fft64_vec_znx_dft_layout::fill_dft_random_log2bound(uint64_t bits) {
  for (uint64_t i = 0; i < size; ++i) {
    set(i, simple_fft64(znx_i64::random_log2bound(nn, bits)));
  }
}
void fft64_vec_znx_dft_layout::set(uint64_t idx, const reim_fft64vec& value) {
  REQUIRE_DRAMATICALLY(value.nn() == nn, "ring dimension mismatch");
  value.save_as(get_addr(idx));
}
thash fft64_vec_znx_dft_layout::content_hash() const { return test_hash(data, size * nn * sizeof(double)); }

reim4_elem fft64_vec_znx_dft_layout::get(uint64_t idx, uint64_t blk) const {
  REQUIRE_DRAMATICALLY(idx < size, "index overflow: " << idx << " / " << size);
  REQUIRE_DRAMATICALLY(blk < nn / 8, "blk overflow: " << blk << " / " << nn / 8);
  double* reim = ((double*)data) + idx * nn;
  return reim4_elem(reim + blk * 4, reim + nn / 2 + blk * 4);
}
reim4_elem fft64_vec_znx_dft_layout::get_zext(uint64_t idx, uint64_t blk) const {
  REQUIRE_DRAMATICALLY(blk < nn / 8, "blk overflow: " << blk << " / " << nn / 8);
  if (idx < size) {
    return get(idx, blk);
  } else {
    return reim4_elem::zero();
  }
}
void fft64_vec_znx_dft_layout::set(uint64_t idx, uint64_t blk, const reim4_elem& value) {
  REQUIRE_DRAMATICALLY(idx < size, "index overflow: " << idx << " / " << size);
  REQUIRE_DRAMATICALLY(blk < nn / 8, "blk overflow: " << blk << " / " << nn / 8);
  double* reim = ((double*)data) + idx * nn;
  value.save_re_im(reim + blk * 4, reim + nn / 2 + blk * 4);
}
void fft64_vec_znx_dft_layout::fill_random(double log2bound) {
  for (uint64_t i = 0; i < size; ++i) {
    set(i, reim_fft64vec::random(nn, log2bound));
  }
}
void fft64_vec_znx_dft_layout::fill_dft_random(uint64_t log2bound) {
  for (uint64_t i = 0; i < size; ++i) {
    set(i, reim_fft64vec::dft_random(nn, log2bound));
  }
}

fft64_vec_znx_big_layout::fft64_vec_znx_big_layout(uint64_t n, uint64_t size)
    : nn(n),       //
      size(size),  //
      data((VEC_ZNX_BIG*)alloc64(n * size * 8)) {}

znx_i64 fft64_vec_znx_big_layout::get_copy(uint64_t index) const {
  REQUIRE_DRAMATICALLY(index < size, "index overflow: " << index << " / " << size);
  return znx_i64(nn, ((int64_t*)data) + index * nn);
}
znx_i64 fft64_vec_znx_big_layout::get_copy_zext(uint64_t index) const {
  if (index < size) {
    return znx_i64(nn, ((int64_t*)data) + index * nn);
  } else {
    return znx_i64::zero(nn);
  }
}
void fft64_vec_znx_big_layout::set(uint64_t index, const znx_i64& value) {
  REQUIRE_DRAMATICALLY(index < size, "index overflow: " << index << " / " << size);
  value.save_as(((int64_t*)data) + index * nn);
}
void fft64_vec_znx_big_layout::fill_random() {
  for (uint64_t i = 0; i < size; ++i) {
    set(i, znx_i64::random_log2bound(nn, 1));
  }
}
fft64_vec_znx_big_layout::~fft64_vec_znx_big_layout() { spqlios_free(data); }

fft64_vmp_pmat_layout::fft64_vmp_pmat_layout(uint64_t n, uint64_t nrows, uint64_t ncols)
    : nn(n),
      nrows(nrows),
      ncols(ncols),  //
      data((VMP_PMAT*)alloc64(nrows * ncols * nn * 8)) {}

double* fft64_vmp_pmat_layout::get_addr(uint64_t row, uint64_t col, uint64_t blk) const {
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

reim4_elem fft64_vmp_pmat_layout::get(uint64_t row, uint64_t col, uint64_t blk) const {
  return reim4_elem(get_addr(row, col, blk));
}
reim4_elem fft64_vmp_pmat_layout::get_zext(uint64_t row, uint64_t col, uint64_t blk) const {
  REQUIRE_DRAMATICALLY(blk < nn / 8, "block overflow: " << blk << " / " << (nn / 8));
  if (row < nrows && col < ncols) {
    return reim4_elem(get_addr(row, col, blk));
  } else {
    return reim4_elem::zero();
  }
}
void fft64_vmp_pmat_layout::set(uint64_t row, uint64_t col, uint64_t blk, const reim4_elem& value) const {
  value.save_as(get_addr(row, col, blk));
}

fft64_vmp_pmat_layout::~fft64_vmp_pmat_layout() { spqlios_free(data); }

reim_fft64vec fft64_vmp_pmat_layout::get_zext(uint64_t row, uint64_t col) const {
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
void fft64_vmp_pmat_layout::set(uint64_t row, uint64_t col, const reim_fft64vec& value) {
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
void fft64_vmp_pmat_layout::fill_random(double log2bound) {
  for (uint64_t row = 0; row < nrows; ++row) {
    for (uint64_t col = 0; col < ncols; ++col) {
      set(row, col, reim_fft64vec::random(nn, log2bound));
    }
  }
}
void fft64_vmp_pmat_layout::fill_dft_random(uint64_t log2bound) {
  for (uint64_t row = 0; row < nrows; ++row) {
    for (uint64_t col = 0; col < ncols; ++col) {
      set(row, col, reim_fft64vec::dft_random(nn, log2bound));
    }
  }
}

fft64_svp_ppol_layout::fft64_svp_ppol_layout(uint64_t n)
    : nn(n),  //
      data((SVP_PPOL*)alloc64(nn * 8)) {}

reim_fft64vec fft64_svp_ppol_layout::get_copy() const { return reim_fft64vec(nn, (double*)data); }

void fft64_svp_ppol_layout::set(const reim_fft64vec& value) { value.save_as((double*)data); }

void fft64_svp_ppol_layout::fill_dft_random(uint64_t log2bound) { set(reim_fft64vec::dft_random(nn, log2bound)); }

void fft64_svp_ppol_layout::fill_random(double log2bound) { set(reim_fft64vec::random(nn, log2bound)); }

fft64_svp_ppol_layout::~fft64_svp_ppol_layout() { spqlios_free(data); }
thash fft64_svp_ppol_layout::content_hash() const { return test_hash(data, nn * sizeof(double)); }

fft64_cnv_left_layout::fft64_cnv_left_layout(uint64_t n, uint64_t size)
    : nn(n),  //
      size(size),
      data((CNV_PVEC_L*)alloc64(size * nn * 8)) {}

reim4_elem fft64_cnv_left_layout::get(uint64_t idx, uint64_t blk) {
  REQUIRE_DRAMATICALLY(idx < size, "idx overflow: " << idx << " / " << size);
  REQUIRE_DRAMATICALLY(blk < nn / 8, "block overflow: " << blk << " / " << (nn / 8));
  return reim4_elem(((double*)data) + blk * size + idx);
}

fft64_cnv_left_layout::~fft64_cnv_left_layout() { spqlios_free(data); }

fft64_cnv_right_layout::fft64_cnv_right_layout(uint64_t n, uint64_t size)
    : nn(n),  //
      size(size),
      data((CNV_PVEC_R*)alloc64(size * nn * 8)) {}

reim4_elem fft64_cnv_right_layout::get(uint64_t idx, uint64_t blk) {
  REQUIRE_DRAMATICALLY(idx < size, "idx overflow: " << idx << " / " << size);
  REQUIRE_DRAMATICALLY(blk < nn / 8, "block overflow: " << blk << " / " << (nn / 8));
  return reim4_elem(((double*)data) + blk * size + idx);
}

fft64_cnv_right_layout::~fft64_cnv_right_layout() { spqlios_free(data); }
