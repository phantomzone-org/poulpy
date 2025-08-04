#include "ntt120_layouts.h"

mod_q120x2::mod_q120x2() {}
mod_q120x2::mod_q120x2(const mod_q120& a, const mod_q120& b) {
  value[0] = a;
  value[1] = b;
}
mod_q120x2::mod_q120x2(q120x2b* addr) {
  uint64_t* p = (uint64_t*)addr;
  value[0] = mod_q120::from_q120b(p);
  value[1] = mod_q120::from_q120b(p + 4);
}

ntt120_vec_znx_dft_layout::ntt120_vec_znx_dft_layout(uint64_t n, uint64_t size)
    : nn(n),       //
      size(size),  //
      data((VEC_ZNX_DFT*)alloc64(n * size * 4 * sizeof(uint64_t))) {}

mod_q120x2 ntt120_vec_znx_dft_layout::get_copy_zext(uint64_t idx, uint64_t blk) {
  return mod_q120x2(get_blk(idx, blk));
}
q120x2b* ntt120_vec_znx_dft_layout::get_blk(uint64_t idx, uint64_t blk) {
  REQUIRE_DRAMATICALLY(idx < size, "idx overflow");
  REQUIRE_DRAMATICALLY(blk < nn / 2, "blk overflow");
  uint64_t* d = (uint64_t*)data;
  return (q120x2b*)(d + 4 * nn * idx + 8 * blk);
}
ntt120_vec_znx_dft_layout::~ntt120_vec_znx_dft_layout() { spqlios_free(data); }
q120_nttvec ntt120_vec_znx_dft_layout::get_copy_zext(uint64_t idx) {
  int64_t* d = (int64_t*)data;
  if (idx < size) {
    return q120_nttvec(nn, (q120b*)(d + idx * nn * 4));
  } else {
    return q120_nttvec::zero(nn);
  }
}
void ntt120_vec_znx_dft_layout::set(uint64_t idx, const q120_nttvec& value) {
  REQUIRE_DRAMATICALLY(idx < size, "index overflow: " << idx << " / " << size);
  q120b* dest_addr = (q120b*)((int64_t*)data + idx * nn * 4);
  value.save_as(dest_addr);
}
void ntt120_vec_znx_dft_layout::fill_random() {
  for (uint64_t i = 0; i < size; ++i) {
    set(i, q120_nttvec::random(nn));
  }
}
thash ntt120_vec_znx_dft_layout::content_hash() const { return test_hash(data, nn * size * 4 * sizeof(int64_t)); }
ntt120_vec_znx_big_layout::ntt120_vec_znx_big_layout(uint64_t n, uint64_t size)
    : nn(n),  //
      size(size),
      data((VEC_ZNX_BIG*)alloc64(n * size * sizeof(__int128_t))) {}

znx_i128 ntt120_vec_znx_big_layout::get_copy(uint64_t index) const { return znx_i128(nn, get_addr(index)); }
znx_i128 ntt120_vec_znx_big_layout::get_copy_zext(uint64_t index) const {
  if (index < size) {
    return znx_i128(nn, get_addr(index));
  } else {
    return znx_i128::zero(nn);
  }
}
__int128* ntt120_vec_znx_big_layout::get_addr(uint64_t index) const {
  REQUIRE_DRAMATICALLY(index < size, "index overflow: " << index << " / " << size);
  return (__int128_t*)data + index * nn;
}
void ntt120_vec_znx_big_layout::set(uint64_t index, const znx_i128& value) { value.save_as(get_addr(index)); }
ntt120_vec_znx_big_layout::~ntt120_vec_znx_big_layout() { spqlios_free(data); }
