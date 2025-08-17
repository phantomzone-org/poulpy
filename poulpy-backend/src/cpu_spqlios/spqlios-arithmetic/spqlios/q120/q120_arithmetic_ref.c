#include <assert.h>
#include <math.h>
#include <stdint.h>

#include "q120_arithmetic.h"
#include "q120_arithmetic_private.h"
#include "q120_common.h"

#define MODQ(val, q) ((val) % (q))

double comp_bit_size_red(const uint64_t h, const uint64_t qs[4]) {
  assert(h < 128);
  double h_pow2_bs = 0;
  for (uint64_t k = 0; k < 4; ++k) {
    double t = log2((double)MODQ((__uint128_t)1 << h, qs[k]));
    if (t > h_pow2_bs) h_pow2_bs = t;
  }
  return h_pow2_bs;
}

double comp_bit_size_sum(const uint64_t n, const double* const bs) {
  double s = 0;
  for (uint64_t i = 0; i < n; ++i) {
    s += pow(2, bs[i]);
  }
  return log2(s);
}

void vec_mat1col_product_baa_precomp(q120_mat1col_product_baa_precomp* precomp) {
  uint64_t qs[4] = {Q1, Q2, Q3, Q4};

  double min_res_bs = 1000;
  uint64_t min_h = -1;

  double ell_bs = log2((double)MAX_ELL);
  for (uint64_t h = 1; h < 64; ++h) {
    double h_pow2_bs = comp_bit_size_red(h, qs);

    const double bs[] = {h + ell_bs, 64 - h + ell_bs + h_pow2_bs};
    const double res_bs = comp_bit_size_sum(2, bs);

    if (min_res_bs > res_bs) {
      min_res_bs = res_bs;
      min_h = h;
    }
  }

  assert(min_res_bs < 64);
  precomp->h = min_h;
  for (uint64_t k = 0; k < 4; ++k) {
    precomp->h_pow_red[k] = MODQ(UINT64_C(1) << precomp->h, qs[k]);
  }
#ifndef NDEBUG
  precomp->res_bit_size = min_res_bs;
#endif
  // printf("AA %lu %lf\n", min_h, min_res_bs);
}

EXPORT q120_mat1col_product_baa_precomp* q120_new_vec_mat1col_product_baa_precomp() {
  q120_mat1col_product_baa_precomp* res = malloc(sizeof(q120_mat1col_product_baa_precomp));
  vec_mat1col_product_baa_precomp(res);
  return res;
}

EXPORT void q120_delete_vec_mat1col_product_baa_precomp(q120_mat1col_product_baa_precomp* addr) { free(addr); }

EXPORT void q120_vec_mat1col_product_baa_ref(q120_mat1col_product_baa_precomp* precomp, const uint64_t ell,
                                             q120b* const res, const q120a* const x, const q120a* const y) {
  /**
   * Algorithm:
   *  - res = acc1 + acc2 . ((2^H) % Q)
   *  - acc1 is the sum of H LSB of products x[i].y[i]
   *  - acc2 is the sum of 64-H MSB of products x[i]].y[i]
   *  - for l < 10k acc1 will have H + log2(10000) and acc2 64 - H + log2(10000) bits
   *  - final sum has max(H, 64 - H + bit_size((2^H) % Q)) + log2(10000) + 1 bits
   */
  const uint64_t H = precomp->h;
  const uint64_t MASK = (UINT64_C(1) << H) - 1;

  uint64_t acc1[4] = {0, 0, 0, 0};  // accumulate H least significant bits of product
  uint64_t acc2[4] = {0, 0, 0, 0};  // accumulate 64 - H most significan bits of product

  const uint64_t* const x_ptr = (uint64_t*)x;
  const uint64_t* const y_ptr = (uint64_t*)y;

  for (uint64_t i = 0; i < 4 * ell; i += 4) {
    for (uint64_t j = 0; j < 4; ++j) {
      uint64_t t = x_ptr[i + j] * y_ptr[i + j];
      acc1[j] += t & MASK;
      acc2[j] += t >> H;
    }
  }

  uint64_t* const res_ptr = (uint64_t*)res;
  for (uint64_t j = 0; j < 4; ++j) {
    res_ptr[j] = acc1[j] + acc2[j] * precomp->h_pow_red[j];
    assert(log2(res_ptr[j]) < precomp->res_bit_size);
  }
}

void vec_mat1col_product_bbb_precomp(q120_mat1col_product_bbb_precomp* precomp) {
  uint64_t qs[4] = {Q1, Q2, Q3, Q4};

  double ell_bs = log2((double)MAX_ELL);
  double min_res_bs = 1000;
  uint64_t min_h = -1;

  const double s1_bs = 32 + ell_bs;
  const double s2_bs = 32 + ell_bs + log2(3);
  const double s3_bs = 32 + ell_bs + log2(3);
  const double s4_bs = 32 + ell_bs;
  for (uint64_t h = 16; h < 32; ++h) {
    const double s1l_bs = h;
    const double s1h_bs = (s1_bs - h) + comp_bit_size_red(h, qs);
    const double s2l_bs = h + comp_bit_size_red(32, qs);
    const double s2h_bs = (s2_bs - h) + comp_bit_size_red(32 + h, qs);
    const double s3l_bs = h + comp_bit_size_red(64, qs);
    const double s3h_bs = (s3_bs - h) + comp_bit_size_red(64 + h, qs);
    const double s4l_bs = h + comp_bit_size_red(96, qs);
    const double s4h_bs = (s4_bs - h) + comp_bit_size_red(96 + h, qs);

    const double bs[] = {s1l_bs, s1h_bs, s2l_bs, s2h_bs, s3l_bs, s3h_bs, s4l_bs, s4h_bs};
    const double res_bs = comp_bit_size_sum(8, bs);

    if (min_res_bs > res_bs) {
      min_res_bs = res_bs;
      min_h = h;
    }
  }

  assert(min_res_bs < 64);
  precomp->h = min_h;
  for (uint64_t k = 0; k < 4; ++k) {
    precomp->s1h_pow_red[k] = UINT64_C(1) << precomp->h;                                       // 2^24
    precomp->s2l_pow_red[k] = MODQ(UINT64_C(1) << 32, qs[k]);                                  // 2^32
    precomp->s2h_pow_red[k] = MODQ(precomp->s2l_pow_red[k] * precomp->s1h_pow_red[k], qs[k]);  // 2^(32+24)
    precomp->s3l_pow_red[k] = MODQ(precomp->s2l_pow_red[k] * precomp->s2l_pow_red[k], qs[k]);  // 2^64 = 2^(32+32)
    precomp->s3h_pow_red[k] = MODQ(precomp->s3l_pow_red[k] * precomp->s1h_pow_red[k], qs[k]);  // 2^(64+24)
    precomp->s4l_pow_red[k] = MODQ(precomp->s3l_pow_red[k] * precomp->s2l_pow_red[k], qs[k]);  // 2^96 = 2^(64+32)
    precomp->s4h_pow_red[k] = MODQ(precomp->s4l_pow_red[k] * precomp->s1h_pow_red[k], qs[k]);  // 2^(96+24)
  }
// printf("AA %lu %lf\n", min_h, min_res_bs);
#ifndef NDEBUG
  precomp->res_bit_size = min_res_bs;
#endif
}

EXPORT q120_mat1col_product_bbb_precomp* q120_new_vec_mat1col_product_bbb_precomp() {
  q120_mat1col_product_bbb_precomp* res = malloc(sizeof(q120_mat1col_product_bbb_precomp));
  vec_mat1col_product_bbb_precomp(res);
  return res;
}

EXPORT void q120_delete_vec_mat1col_product_bbb_precomp(q120_mat1col_product_bbb_precomp* addr) { free(addr); }

EXPORT void q120_vec_mat1col_product_bbb_ref(q120_mat1col_product_bbb_precomp* precomp, const uint64_t ell,
                                             q120b* const res, const q120b* const x, const q120b* const y) {
  /**
   * Algorithm:
   *  1. Split x_i and y_i in 2 32-bit parts and compute the cross-products:
   *    - x_i = xl_i + xh_i . 2^32
   *    - y_i = yl_i + yh_i . 2^32
   *    - A_i = xl_i . yl_i
   *    - B_i = xl_i . yh_i
   *    - C_i = xh_i . yl_i
   *    - D_i = xh_i . yh_i
   *    - we have x_i . y_i == A_i + (B_i + C_i) . 2^32 + D_i . 2^64
   *  2. Split A_i, B_i, C_i and D_i into 2 32-bit parts
   *    - A_i = Al_i + Ah_i . 2^32
   *    - B_i = Bl_i + Bh_i . 2^32
   *    - C_i = Cl_i + Ch_i . 2^32
   *    - D_i = Dl_i + Dh_i . 2^32
   *  3. Compute the sums:
   *    - S1 = \sum Al_i
   *    - S2 = \sum (Ah_i + Bl_i + Cl_i)
   *    - S3 = \sum (Bh_i + Ch_i + Dl_i)
   *    - S4 = \sum Dh_i
   *    - here S1, S4 have 32 + log2(ell) bits and S2, S3 have 32 + log2(ell) +
   *      log2(3) bits
   *    - for ell == 10000 S2, S3 have < 47 bits
   *  4. Split S1, S2, S3 and S4 in 2 24-bit parts (24 = ceil(47/2))
   *    - S1 = S1l + S1h . 2^24
   *    - S2 = S2l + S2h . 2^24
   *    - S3 = S3l + S3h . 2^24
   *    - S4 = S4l + S4h . 2^24
   *  5. Compute final result as:
   *    - \sum x_i . y_i = S1l + S1h . 2^24
   *                       + S2l . 2^32 + S2h . 2^(32+24)
   *                       + S3l . 2^64 + S3h . 2^(64 + 24)
   *                       + S4l . 2^96 + S4l . 2^(96+24)
   *    - here the powers of 2 are reduced modulo the primes Q before
   *      multiplications
   *    - the result will be on 24 + 3 + bit size of primes Q
   */
  const uint64_t H1 = 32;
  const uint64_t MASK1 = (UINT64_C(1) << H1) - 1;

  uint64_t s1[4] = {0, 0, 0, 0};
  uint64_t s2[4] = {0, 0, 0, 0};
  uint64_t s3[4] = {0, 0, 0, 0};
  uint64_t s4[4] = {0, 0, 0, 0};

  const uint64_t* const x_ptr = (uint64_t*)x;
  const uint64_t* const y_ptr = (uint64_t*)y;

  for (uint64_t i = 0; i < 4 * ell; i += 4) {
    for (uint64_t j = 0; j < 4; ++j) {
      const uint64_t xl = x_ptr[i + j] & MASK1;
      const uint64_t xh = x_ptr[i + j] >> H1;
      const uint64_t yl = y_ptr[i + j] & MASK1;
      const uint64_t yh = y_ptr[i + j] >> H1;

      const uint64_t a = xl * yl;
      const uint64_t al = a & MASK1;
      const uint64_t ah = a >> H1;

      const uint64_t b = xl * yh;
      const uint64_t bl = b & MASK1;
      const uint64_t bh = b >> H1;

      const uint64_t c = xh * yl;
      const uint64_t cl = c & MASK1;
      const uint64_t ch = c >> H1;

      const uint64_t d = xh * yh;
      const uint64_t dl = d & MASK1;
      const uint64_t dh = d >> H1;

      s1[j] += al;
      s2[j] += ah + bl + cl;
      s3[j] += bh + ch + dl;
      s4[j] += dh;
    }
  }

  const uint64_t H2 = precomp->h;
  const uint64_t MASK2 = (UINT64_C(1) << H2) - 1;

  uint64_t* const res_ptr = (uint64_t*)res;
  for (uint64_t j = 0; j < 4; ++j) {
    const uint64_t s1l = s1[j] & MASK2;
    const uint64_t s1h = s1[j] >> H2;
    const uint64_t s2l = s2[j] & MASK2;
    const uint64_t s2h = s2[j] >> H2;
    const uint64_t s3l = s3[j] & MASK2;
    const uint64_t s3h = s3[j] >> H2;
    const uint64_t s4l = s4[j] & MASK2;
    const uint64_t s4h = s4[j] >> H2;

    uint64_t t = s1l;
    t += s1h * precomp->s1h_pow_red[j];
    t += s2l * precomp->s2l_pow_red[j];
    t += s2h * precomp->s2h_pow_red[j];
    t += s3l * precomp->s3l_pow_red[j];
    t += s3h * precomp->s3h_pow_red[j];
    t += s4l * precomp->s4l_pow_red[j];
    t += s4h * precomp->s4h_pow_red[j];

    res_ptr[j] = t;
    assert(log2(res_ptr[j]) < precomp->res_bit_size);
  }
}

void vec_mat1col_product_bbc_precomp(q120_mat1col_product_bbc_precomp* precomp) {
  uint64_t qs[4] = {Q1, Q2, Q3, Q4};

  double min_res_bs = 1000;
  uint64_t min_h = -1;

  double pow2_32_bs = comp_bit_size_red(32, qs);

  double ell_bs = log2((double)MAX_ELL);
  double s1_bs = 32 + ell_bs;
  for (uint64_t h = 16; h < 32; ++h) {
    double s2l_bs = pow2_32_bs + h;
    double s2h_bs = s1_bs - h + comp_bit_size_red(32 + h, qs);

    const double bs[] = {s1_bs, s2l_bs, s2h_bs};
    const double res_bs = comp_bit_size_sum(3, bs);

    if (min_res_bs > res_bs) {
      min_res_bs = res_bs;
      min_h = h;
    }
  }

  assert(min_res_bs < 64);
  precomp->h = min_h;
  for (uint64_t k = 0; k < 4; ++k) {
    precomp->s2l_pow_red[k] = MODQ(UINT64_C(1) << 32, qs[k]);
    precomp->s2h_pow_red[k] = MODQ(UINT64_C(1) << (32 + precomp->h), qs[k]);
  }
#ifndef NDEBUG
  precomp->res_bit_size = min_res_bs;
#endif
  // printf("AA %lu %lf\n", min_h, min_res_bs);
}

EXPORT q120_mat1col_product_bbc_precomp* q120_new_vec_mat1col_product_bbc_precomp() {
  q120_mat1col_product_bbc_precomp* res = malloc(sizeof(q120_mat1col_product_bbc_precomp));
  vec_mat1col_product_bbc_precomp(res);
  return res;
}

EXPORT void q120_delete_vec_mat1col_product_bbc_precomp(q120_mat1col_product_bbc_precomp* addr) { free(addr); }

EXPORT void q120_vec_mat1col_product_bbc_ref_old(q120_mat1col_product_bbc_precomp* precomp, const uint64_t ell,
                                                 q120b* const res, const q120b* const x, const q120c* const y) {
  /**
   * Algorithm:
   *  0. We have
   *    - y0_i == y_i % Q and y1_i == (y_i . 2^32) % Q
   *  1. Split x_i in 2 32-bit parts and compute the cross-products:
   *    - x_i = xl_i + xh_i . 2^32
   *    - A_i = xl_i . y1_i
   *    - B_i = xh_i . y2_i
   *    - we have x_i . y_i == A_i + B_i
   *  2. Split A_i and B_i into 2 32-bit parts
   *    - A_i = Al_i + Ah_i . 2^32
   *    - B_i = Bl_i + Bh_i . 2^32
   *  3. Compute the sums:
   *    - S1 = \sum Al_i + Bl_i
   *    - S2 = \sum Ah_i + Bh_i
   *    - here S1 and S2 have 32 + log2(ell) bits
   *    - for ell == 10000 S1, S2 have < 46 bits
   *  4. Split S2 in 27-bit and 19-bit parts (27+19 == 46)
   *    - S2 = S2l + S2h . 2^27
   *  5. Compute final result as:
   *    - \sum x_i . y_i = S1 + S2l . 2^32 + S2h . 2^(32+27)
   *    - here the powers of 2 are reduced modulo the primes Q before
   *      multiplications
   *    - the result will be on < 52 bits
   */

  const uint64_t H1 = 32;
  const uint64_t MASK1 = (UINT64_C(1) << H1) - 1;

  uint64_t s1[4] = {0, 0, 0, 0};
  uint64_t s2[4] = {0, 0, 0, 0};

  const uint64_t* const x_ptr = (uint64_t*)x;
  const uint32_t* const y_ptr = (uint32_t*)y;

  for (uint64_t i = 0; i < 4 * ell; i += 4) {
    for (uint64_t j = 0; j < 4; ++j) {
      const uint64_t xl = x_ptr[i + j] & MASK1;
      const uint64_t xh = x_ptr[i + j] >> H1;
      const uint64_t y0 = y_ptr[2 * (i + j)];
      const uint64_t y1 = y_ptr[2 * (i + j) + 1];

      const uint64_t a = xl * y0;
      const uint64_t al = a & MASK1;
      const uint64_t ah = a >> H1;

      const uint64_t b = xh * y1;
      const uint64_t bl = b & MASK1;
      const uint64_t bh = b >> H1;

      s1[j] += al + bl;
      s2[j] += ah + bh;
    }
  }

  const uint64_t H2 = precomp->h;
  const uint64_t MASK2 = (UINT64_C(1) << H2) - 1;

  uint64_t* const res_ptr = (uint64_t*)res;
  for (uint64_t k = 0; k < 4; ++k) {
    const uint64_t s2l = s2[k] & MASK2;
    const uint64_t s2h = s2[k] >> H2;

    uint64_t t = s1[k];
    t += s2l * precomp->s2l_pow_red[k];
    t += s2h * precomp->s2h_pow_red[k];

    res_ptr[k] = t;
    assert(log2(res_ptr[k]) < precomp->res_bit_size);
  }
}

static __always_inline void accum_mul_q120_bc(uint64_t res[8],  //
                                              const uint32_t x_layb[8], const uint32_t y_layc[8]) {
  for (uint64_t i = 0; i < 4; ++i) {
    static const uint64_t MASK32 = 0xFFFFFFFFUL;
    uint64_t x_lo = x_layb[2 * i];
    uint64_t x_hi = x_layb[2 * i + 1];
    uint64_t y_lo = y_layc[2 * i];
    uint64_t y_hi = y_layc[2 * i + 1];
    uint64_t xy_lo = x_lo * y_lo;
    uint64_t xy_hi = x_hi * y_hi;
    res[2 * i] += (xy_lo & MASK32) + (xy_hi & MASK32);
    res[2 * i + 1] += (xy_lo >> 32) + (xy_hi >> 32);
  }
}

static __always_inline void accum_to_q120b(uint64_t res[4],  //
                                           const uint64_t s[8], const q120_mat1col_product_bbc_precomp* precomp) {
  const uint64_t H2 = precomp->h;
  const uint64_t MASK2 = (UINT64_C(1) << H2) - 1;
  for (uint64_t k = 0; k < 4; ++k) {
    const uint64_t s2l = s[2 * k + 1] & MASK2;
    const uint64_t s2h = s[2 * k + 1] >> H2;
    uint64_t t = s[2 * k];
    t += s2l * precomp->s2l_pow_red[k];
    t += s2h * precomp->s2h_pow_red[k];
    res[k] = t;
    assert(log2(res[k]) < precomp->res_bit_size);
  }
}

EXPORT void q120_vec_mat1col_product_bbc_ref(q120_mat1col_product_bbc_precomp* precomp, const uint64_t ell,
                                             q120b* const res, const q120b* const x, const q120c* const y) {
  uint64_t s[8] = {0, 0, 0, 0, 0, 0, 0, 0};

  const uint32_t(*const x_ptr)[8] = (const uint32_t(*const)[8])x;
  const uint32_t(*const y_ptr)[8] = (const uint32_t(*const)[8])y;

  for (uint64_t i = 0; i < ell; i++) {
    accum_mul_q120_bc(s, x_ptr[i], y_ptr[i]);
  }
  accum_to_q120b((uint64_t*)res, s, precomp);
}

EXPORT void q120x2_vec_mat1col_product_bbc_ref(q120_mat1col_product_bbc_precomp* precomp, const uint64_t ell,
                                               q120b* const res, const q120b* const x, const q120c* const y) {
  uint64_t s[2][16] = {0};

  const uint32_t(*const x_ptr)[2][8] = (const uint32_t(*const)[2][8])x;
  const uint32_t(*const y_ptr)[2][8] = (const uint32_t(*const)[2][8])y;
  uint64_t(*re)[4] = (uint64_t(*)[4])res;

  for (uint64_t i = 0; i < ell; i++) {
    accum_mul_q120_bc(s[0], x_ptr[i][0], y_ptr[i][0]);
    accum_mul_q120_bc(s[1], x_ptr[i][1], y_ptr[i][1]);
  }
  accum_to_q120b(re[0], s[0], precomp);
  accum_to_q120b(re[1], s[1], precomp);
}

EXPORT void q120x2_vec_mat2cols_product_bbc_ref(q120_mat1col_product_bbc_precomp* precomp, const uint64_t ell,
                                                q120b* const res, const q120b* const x, const q120c* const y) {
  uint64_t s[4][16] = {0};

  const uint32_t(*const x_ptr)[2][8] = (const uint32_t(*const)[2][8])x;
  const uint32_t(*const y_ptr)[4][8] = (const uint32_t(*const)[4][8])y;
  uint64_t(*re)[4] = (uint64_t(*)[4])res;

  for (uint64_t i = 0; i < ell; i++) {
    accum_mul_q120_bc(s[0], x_ptr[i][0], y_ptr[i][0]);
    accum_mul_q120_bc(s[1], x_ptr[i][1], y_ptr[i][1]);
    accum_mul_q120_bc(s[2], x_ptr[i][0], y_ptr[i][2]);
    accum_mul_q120_bc(s[3], x_ptr[i][1], y_ptr[i][3]);
  }
  accum_to_q120b(re[0], s[0], precomp);
  accum_to_q120b(re[1], s[1], precomp);
  accum_to_q120b(re[2], s[2], precomp);
  accum_to_q120b(re[3], s[3], precomp);
}

EXPORT void q120x2_extract_1blk_from_q120b_ref(uint64_t nn, uint64_t blk,
                                               q120x2b* const dst,     // 8 doubles
                                               const q120b* const src  // a q120b vector
) {
  const uint64_t* in = (uint64_t*)src;
  uint64_t* out = (uint64_t*)dst;
  for (uint64_t i = 0; i < 8; ++i) {
    out[i] = in[8 * blk + i];
  }
}

// function on layout c is the exact same as on layout b
#ifdef __APPLE__
#pragma weak q120x2_extract_1blk_from_q120c_ref = q120x2_extract_1blk_from_q120b_ref
#else
EXPORT void q120x2_extract_1blk_from_q120c_ref(uint64_t nn, uint64_t blk,
                                               q120x2c* const dst,     // 8 doubles
                                               const q120c* const src  // a q120c vector
                                               ) __attribute__((alias("q120x2_extract_1blk_from_q120b_ref")));
#endif

EXPORT void q120x2_extract_1blk_from_contiguous_q120b_ref(
    uint64_t nn, uint64_t nrows, uint64_t blk,
    q120x2b* const dst,     // nrows * 2 q120
    const q120b* const src  // a contiguous array of nrows q120b vectors
) {
  const uint64_t* in = (uint64_t*)src;
  uint64_t* out = (uint64_t*)dst;
  for (uint64_t row = 0; row < nrows; ++row) {
    for (uint64_t i = 0; i < 8; ++i) {
      out[i] = in[8 * blk + i];
    }
    in += 4 * nn;
    out += 8;
  }
}

EXPORT void q120x2b_save_1blk_to_q120b_ref(uint64_t nn, uint64_t blk,
                                           q120b* dest,        // 1 reim vector of length m
                                           const q120x2b* src  // 8 doubles
) {
  const uint64_t* in = (uint64_t*)src;
  uint64_t* out = (uint64_t*)dest;
  for (uint64_t i = 0; i < 8; ++i) {
    out[8 * blk + i] = in[i];
  }
}
