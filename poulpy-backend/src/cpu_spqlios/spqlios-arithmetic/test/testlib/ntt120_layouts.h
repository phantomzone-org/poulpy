#ifndef SPQLIOS_NTT120_LAYOUTS_H
#define SPQLIOS_NTT120_LAYOUTS_H

#include "../../spqlios/arithmetic/vec_znx_arithmetic.h"
#include "mod_q120.h"
#include "negacyclic_polynomial.h"
#include "ntt120_dft.h"
#include "test_commons.h"

struct q120b_vector_view {};

struct mod_q120x2 {
  mod_q120 value[2];
  mod_q120x2();
  mod_q120x2(const mod_q120& a, const mod_q120& b);
  mod_q120x2(__int128_t value);
  explicit mod_q120x2(q120x2b* addr);
  explicit mod_q120x2(q120x2c* addr);
  void save_as(q120x2b* addr) const;
  void save_as(q120x2c* addr) const;
  static mod_q120x2 random();
};
mod_q120x2 operator+(const mod_q120x2& a, const mod_q120x2& b);
mod_q120x2 operator-(const mod_q120x2& a, const mod_q120x2& b);
mod_q120x2 operator*(const mod_q120x2& a, const mod_q120x2& b);
bool operator==(const mod_q120x2& a, const mod_q120x2& b);
bool operator!=(const mod_q120x2& a, const mod_q120x2& b);
mod_q120x2& operator+=(mod_q120x2& a, const mod_q120x2& b);
mod_q120x2& operator-=(mod_q120x2& a, const mod_q120x2& b);

/** @brief test layout for the VEC_ZNX_DFT */
struct ntt120_vec_znx_dft_layout {
  const uint64_t nn;
  const uint64_t size;
  VEC_ZNX_DFT* const data;
  ntt120_vec_znx_dft_layout(uint64_t n, uint64_t size);
  mod_q120x2 get_copy_zext(uint64_t idx, uint64_t blk);
  q120_nttvec get_copy_zext(uint64_t idx);
  void set(uint64_t idx, const q120_nttvec& v);
  q120x2b* get_blk(uint64_t idx, uint64_t blk);
  thash content_hash() const;
  void fill_random();
  ~ntt120_vec_znx_dft_layout();
};

/** @brief test layout for the VEC_ZNX_BIG */
class ntt120_vec_znx_big_layout {
 public:
  const uint64_t nn;
  const uint64_t size;
  VEC_ZNX_BIG* const data;
  ntt120_vec_znx_big_layout(uint64_t n, uint64_t size);

 private:
  __int128* get_addr(uint64_t index) const;

 public:
  znx_i128 get_copy(uint64_t index) const;
  znx_i128 get_copy_zext(uint64_t index) const;
  void set(uint64_t index, const znx_i128& value);
  ~ntt120_vec_znx_big_layout();
};

/** @brief test layout for the VMP_PMAT */
class ntt120_vmp_pmat_layout {
  const uint64_t nn;
  const uint64_t nrows;
  const uint64_t ncols;
  VMP_PMAT* const data;
  ntt120_vmp_pmat_layout(uint64_t n, uint64_t nrows, uint64_t ncols);
  mod_q120x2 get(uint64_t row, uint64_t col, uint64_t blk) const;
  ~ntt120_vmp_pmat_layout();
};

/** @brief test layout for the SVP_PPOL */
class ntt120_svp_ppol_layout {
  const uint64_t nn;
  SVP_PPOL* const data;
  ntt120_svp_ppol_layout(uint64_t n);
  ~ntt120_svp_ppol_layout();
};

/** @brief test layout for the CNV_PVEC_L */
class ntt120_cnv_left_layout {
  const uint64_t nn;
  const uint64_t size;
  CNV_PVEC_L* const data;
  ntt120_cnv_left_layout(uint64_t n, uint64_t size);
  mod_q120x2 get(uint64_t idx, uint64_t blk);
  ~ntt120_cnv_left_layout();
};

/** @brief test layout for the CNV_PVEC_R */
class ntt120_cnv_right_layout {
  const uint64_t nn;
  const uint64_t size;
  CNV_PVEC_R* const data;
  ntt120_cnv_right_layout(uint64_t n, uint64_t size);
  mod_q120x2 get(uint64_t idx, uint64_t blk);
  ~ntt120_cnv_right_layout();
};

#endif  // SPQLIOS_NTT120_LAYOUTS_H
