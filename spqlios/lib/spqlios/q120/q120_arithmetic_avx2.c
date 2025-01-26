#include <immintrin.h>

#include "q120_arithmetic.h"
#include "q120_arithmetic_private.h"

EXPORT void q120_vec_mat1col_product_baa_avx2(q120_mat1col_product_baa_precomp* precomp, const uint64_t ell,
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
  const __m256i MASK = _mm256_set1_epi64x((UINT64_C(1) << H) - 1);

  __m256i acc1 = _mm256_setzero_si256();
  __m256i acc2 = _mm256_setzero_si256();

  const __m256i* x_ptr = (__m256i*)x;
  const __m256i* y_ptr = (__m256i*)y;

  for (uint64_t i = 0; i < ell; ++i) {
    __m256i a = _mm256_loadu_si256(x_ptr);
    __m256i b = _mm256_loadu_si256(y_ptr);
    __m256i t = _mm256_mul_epu32(a, b);

    acc1 = _mm256_add_epi64(acc1, _mm256_and_si256(t, MASK));
    acc2 = _mm256_add_epi64(acc2, _mm256_srli_epi64(t, H));

    x_ptr++;
    y_ptr++;
  }

  const __m256i H_POW_RED = _mm256_loadu_si256((__m256i*)precomp->h_pow_red);

  __m256i t = _mm256_add_epi64(acc1, _mm256_mul_epu32(acc2, H_POW_RED));
  _mm256_storeu_si256((__m256i*)res, t);
}

EXPORT void q120_vec_mat1col_product_bbb_avx2(q120_mat1col_product_bbb_precomp* precomp, const uint64_t ell,
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
  const __m256i MASK1 = _mm256_set1_epi64x((UINT64_C(1) << H1) - 1);

  __m256i s1 = _mm256_setzero_si256();
  __m256i s2 = _mm256_setzero_si256();
  __m256i s3 = _mm256_setzero_si256();
  __m256i s4 = _mm256_setzero_si256();

  const __m256i* x_ptr = (__m256i*)x;
  const __m256i* y_ptr = (__m256i*)y;

  for (uint64_t i = 0; i < ell; ++i) {
    __m256i x = _mm256_loadu_si256(x_ptr);
    __m256i xl = _mm256_and_si256(x, MASK1);
    __m256i xh = _mm256_srli_epi64(x, H1);

    __m256i y = _mm256_loadu_si256(y_ptr);
    __m256i yl = _mm256_and_si256(y, MASK1);
    __m256i yh = _mm256_srli_epi64(y, H1);

    __m256i a = _mm256_mul_epu32(xl, yl);
    __m256i b = _mm256_mul_epu32(xl, yh);
    __m256i c = _mm256_mul_epu32(xh, yl);
    __m256i d = _mm256_mul_epu32(xh, yh);

    s1 = _mm256_add_epi64(s1, _mm256_and_si256(a, MASK1));

    s2 = _mm256_add_epi64(s2, _mm256_srli_epi64(a, H1));
    s2 = _mm256_add_epi64(s2, _mm256_and_si256(b, MASK1));
    s2 = _mm256_add_epi64(s2, _mm256_and_si256(c, MASK1));

    s3 = _mm256_add_epi64(s3, _mm256_srli_epi64(b, H1));
    s3 = _mm256_add_epi64(s3, _mm256_srli_epi64(c, H1));
    s3 = _mm256_add_epi64(s3, _mm256_and_si256(d, MASK1));

    s4 = _mm256_add_epi64(s4, _mm256_srli_epi64(d, H1));

    x_ptr++;
    y_ptr++;
  }

  const uint64_t H2 = precomp->h;
  const __m256i MASK2 = _mm256_set1_epi64x((UINT64_C(1) << H2) - 1);

  const __m256i S1H_POW_RED = _mm256_loadu_si256((__m256i*)precomp->s1h_pow_red);
  __m256i s1l = _mm256_and_si256(s1, MASK2);
  __m256i s1h = _mm256_srli_epi64(s1, H2);
  __m256i t = _mm256_add_epi64(s1l, _mm256_mul_epu32(s1h, S1H_POW_RED));

  const __m256i S2L_POW_RED = _mm256_loadu_si256((__m256i*)precomp->s2l_pow_red);
  const __m256i S2H_POW_RED = _mm256_loadu_si256((__m256i*)precomp->s2h_pow_red);
  __m256i s2l = _mm256_and_si256(s2, MASK2);
  __m256i s2h = _mm256_srli_epi64(s2, H2);
  t = _mm256_add_epi64(t, _mm256_mul_epu32(s2l, S2L_POW_RED));
  t = _mm256_add_epi64(t, _mm256_mul_epu32(s2h, S2H_POW_RED));

  const __m256i S3L_POW_RED = _mm256_loadu_si256((__m256i*)precomp->s3l_pow_red);
  const __m256i S3H_POW_RED = _mm256_loadu_si256((__m256i*)precomp->s3h_pow_red);
  __m256i s3l = _mm256_and_si256(s3, MASK2);
  __m256i s3h = _mm256_srli_epi64(s3, H2);
  t = _mm256_add_epi64(t, _mm256_mul_epu32(s3l, S3L_POW_RED));
  t = _mm256_add_epi64(t, _mm256_mul_epu32(s3h, S3H_POW_RED));

  const __m256i S4L_POW_RED = _mm256_loadu_si256((__m256i*)precomp->s4l_pow_red);
  const __m256i S4H_POW_RED = _mm256_loadu_si256((__m256i*)precomp->s4h_pow_red);
  __m256i s4l = _mm256_and_si256(s4, MASK2);
  __m256i s4h = _mm256_srli_epi64(s4, H2);
  t = _mm256_add_epi64(t, _mm256_mul_epu32(s4l, S4L_POW_RED));
  t = _mm256_add_epi64(t, _mm256_mul_epu32(s4h, S4H_POW_RED));

  _mm256_storeu_si256((__m256i*)res, t);
}

EXPORT void q120_vec_mat1col_product_bbc_avx2(q120_mat1col_product_bbc_precomp* precomp, const uint64_t ell,
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
  const __m256i MASK1 = _mm256_set1_epi64x((UINT64_C(1) << H1) - 1);

  __m256i s1 = _mm256_setzero_si256();
  __m256i s2 = _mm256_setzero_si256();

  const __m256i* x_ptr = (__m256i*)x;
  const __m256i* y_ptr = (__m256i*)y;

  for (uint64_t i = 0; i < ell; ++i) {
    __m256i x = _mm256_loadu_si256(x_ptr);
    __m256i xl = _mm256_and_si256(x, MASK1);
    __m256i xh = _mm256_srli_epi64(x, H1);

    __m256i y = _mm256_loadu_si256(y_ptr);
    __m256i y0 = _mm256_and_si256(y, MASK1);
    __m256i y1 = _mm256_srli_epi64(y, H1);

    __m256i a = _mm256_mul_epu32(xl, y0);
    __m256i b = _mm256_mul_epu32(xh, y1);

    s1 = _mm256_add_epi64(s1, _mm256_and_si256(a, MASK1));
    s1 = _mm256_add_epi64(s1, _mm256_and_si256(b, MASK1));

    s2 = _mm256_add_epi64(s2, _mm256_srli_epi64(a, H1));
    s2 = _mm256_add_epi64(s2, _mm256_srli_epi64(b, H1));

    x_ptr++;
    y_ptr++;
  }

  const uint64_t H2 = precomp->h;
  const __m256i MASK2 = _mm256_set1_epi64x((UINT64_C(1) << H2) - 1);

  __m256i t = s1;

  const __m256i S2L_POW_RED = _mm256_loadu_si256((__m256i*)precomp->s2l_pow_red);
  const __m256i S2H_POW_RED = _mm256_loadu_si256((__m256i*)precomp->s2h_pow_red);
  __m256i s2l = _mm256_and_si256(s2, MASK2);
  __m256i s2h = _mm256_srli_epi64(s2, H2);
  t = _mm256_add_epi64(t, _mm256_mul_epu32(s2l, S2L_POW_RED));
  t = _mm256_add_epi64(t, _mm256_mul_epu32(s2h, S2H_POW_RED));

  _mm256_storeu_si256((__m256i*)res, t);
}

/**
 * @deprecated keeping this one for history only.
 * There is a slight register starvation condition on the q120x2_vec_mat2cols
 * strategy below sounds better.
 */
EXPORT void q120x2_vec_mat2cols_product_bbc_avx2_old(q120_mat1col_product_bbc_precomp* precomp, const uint64_t ell,
                                                     q120b* const res, const q120b* const x, const q120c* const y) {
  __m256i s0 = _mm256_setzero_si256();  // col 1a
  __m256i s1 = _mm256_setzero_si256();
  __m256i s2 = _mm256_setzero_si256();  // col 1b
  __m256i s3 = _mm256_setzero_si256();
  __m256i s4 = _mm256_setzero_si256();  // col 2a
  __m256i s5 = _mm256_setzero_si256();
  __m256i s6 = _mm256_setzero_si256();  // col 2b
  __m256i s7 = _mm256_setzero_si256();
  __m256i s8, s9, s10, s11;
  __m256i s12, s13, s14, s15;

  const __m256i* x_ptr = (__m256i*)x;
  const __m256i* y_ptr = (__m256i*)y;
  __m256i* res_ptr = (__m256i*)res;
  for (uint64_t i = 0; i < ell; ++i) {
    s8 = _mm256_loadu_si256(x_ptr);
    s9 = _mm256_loadu_si256(x_ptr + 1);
    s10 = _mm256_srli_epi64(s8, 32);
    s11 = _mm256_srli_epi64(s9, 32);

    s12 = _mm256_loadu_si256(y_ptr);
    s13 = _mm256_loadu_si256(y_ptr + 1);
    s14 = _mm256_srli_epi64(s12, 32);
    s15 = _mm256_srli_epi64(s13, 32);

    s12 = _mm256_mul_epu32(s8, s12);   // -> s0,s1
    s13 = _mm256_mul_epu32(s9, s13);   // -> s2,s3
    s14 = _mm256_mul_epu32(s10, s14);  // -> s0,s1
    s15 = _mm256_mul_epu32(s11, s15);  // -> s2,s3

    s10 = _mm256_slli_epi64(s12, 32);  // -> s0
    s11 = _mm256_slli_epi64(s13, 32);  // -> s2
    s12 = _mm256_srli_epi64(s12, 32);  // -> s1
    s13 = _mm256_srli_epi64(s13, 32);  // -> s3
    s10 = _mm256_srli_epi64(s10, 32);  // -> s0
    s11 = _mm256_srli_epi64(s11, 32);  // -> s2

    s0 = _mm256_add_epi64(s0, s10);
    s1 = _mm256_add_epi64(s1, s12);
    s2 = _mm256_add_epi64(s2, s11);
    s3 = _mm256_add_epi64(s3, s13);

    s10 = _mm256_slli_epi64(s14, 32);  // -> s0
    s11 = _mm256_slli_epi64(s15, 32);  // -> s2
    s14 = _mm256_srli_epi64(s14, 32);  // -> s1
    s15 = _mm256_srli_epi64(s15, 32);  // -> s3
    s10 = _mm256_srli_epi64(s10, 32);  // -> s0
    s11 = _mm256_srli_epi64(s11, 32);  // -> s2

    s0 = _mm256_add_epi64(s0, s10);
    s1 = _mm256_add_epi64(s1, s14);
    s2 = _mm256_add_epi64(s2, s11);
    s3 = _mm256_add_epi64(s3, s15);

    // deal with the second column
    // s8,s9 are still in place!
    s10 = _mm256_srli_epi64(s8, 32);
    s11 = _mm256_srli_epi64(s9, 32);

    s12 = _mm256_loadu_si256(y_ptr + 2);
    s13 = _mm256_loadu_si256(y_ptr + 3);
    s14 = _mm256_srli_epi64(s12, 32);
    s15 = _mm256_srli_epi64(s13, 32);

    s12 = _mm256_mul_epu32(s8, s12);   // -> s4,s5
    s13 = _mm256_mul_epu32(s9, s13);   // -> s6,s7
    s14 = _mm256_mul_epu32(s10, s14);  // -> s4,s5
    s15 = _mm256_mul_epu32(s11, s15);  // -> s6,s7

    s10 = _mm256_slli_epi64(s12, 32);  // -> s4
    s11 = _mm256_slli_epi64(s13, 32);  // -> s6
    s12 = _mm256_srli_epi64(s12, 32);  // -> s5
    s13 = _mm256_srli_epi64(s13, 32);  // -> s7
    s10 = _mm256_srli_epi64(s10, 32);  // -> s4
    s11 = _mm256_srli_epi64(s11, 32);  // -> s6

    s4 = _mm256_add_epi64(s4, s10);
    s5 = _mm256_add_epi64(s5, s12);
    s6 = _mm256_add_epi64(s6, s11);
    s7 = _mm256_add_epi64(s7, s13);

    s10 = _mm256_slli_epi64(s14, 32);  // -> s4
    s11 = _mm256_slli_epi64(s15, 32);  // -> s6
    s14 = _mm256_srli_epi64(s14, 32);  // -> s5
    s15 = _mm256_srli_epi64(s15, 32);  // -> s7
    s10 = _mm256_srli_epi64(s10, 32);  // -> s4
    s11 = _mm256_srli_epi64(s11, 32);  // -> s6

    s4 = _mm256_add_epi64(s4, s10);
    s5 = _mm256_add_epi64(s5, s14);
    s6 = _mm256_add_epi64(s6, s11);
    s7 = _mm256_add_epi64(s7, s15);

    x_ptr += 2;
    y_ptr += 4;
  }
  // final reduction
  const uint64_t H2 = precomp->h;
  s8 = _mm256_set1_epi64x((UINT64_C(1) << H2) - 1);          // MASK2
  s9 = _mm256_loadu_si256((__m256i*)precomp->s2l_pow_red);   // S2L_POW_RED
  s10 = _mm256_loadu_si256((__m256i*)precomp->s2h_pow_red);  // S2H_POW_RED
  //--- s0,s1
  s11 = _mm256_and_si256(s1, s8);
  s12 = _mm256_srli_epi64(s1, H2);
  s13 = _mm256_mul_epu32(s11, s9);
  s14 = _mm256_mul_epu32(s12, s10);
  s0 = _mm256_add_epi64(s0, s13);
  s0 = _mm256_add_epi64(s0, s14);
  _mm256_storeu_si256(res_ptr + 0, s0);
  //--- s2,s3
  s11 = _mm256_and_si256(s3, s8);
  s12 = _mm256_srli_epi64(s3, H2);
  s13 = _mm256_mul_epu32(s11, s9);
  s14 = _mm256_mul_epu32(s12, s10);
  s2 = _mm256_add_epi64(s2, s13);
  s2 = _mm256_add_epi64(s2, s14);
  _mm256_storeu_si256(res_ptr + 1, s2);
  //--- s4,s5
  s11 = _mm256_and_si256(s5, s8);
  s12 = _mm256_srli_epi64(s5, H2);
  s13 = _mm256_mul_epu32(s11, s9);
  s14 = _mm256_mul_epu32(s12, s10);
  s4 = _mm256_add_epi64(s4, s13);
  s4 = _mm256_add_epi64(s4, s14);
  _mm256_storeu_si256(res_ptr + 2, s4);
  //--- s6,s7
  s11 = _mm256_and_si256(s7, s8);
  s12 = _mm256_srli_epi64(s7, H2);
  s13 = _mm256_mul_epu32(s11, s9);
  s14 = _mm256_mul_epu32(s12, s10);
  s6 = _mm256_add_epi64(s6, s13);
  s6 = _mm256_add_epi64(s6, s14);
  _mm256_storeu_si256(res_ptr + 3, s6);
}

EXPORT void q120x2_vec_mat2cols_product_bbc_avx2(q120_mat1col_product_bbc_precomp* precomp, const uint64_t ell,
                                                 q120b* const res, const q120b* const x, const q120c* const y) {
  __m256i s0 = _mm256_setzero_si256();  // col 1a
  __m256i s1 = _mm256_setzero_si256();
  __m256i s2 = _mm256_setzero_si256();  // col 1b
  __m256i s3 = _mm256_setzero_si256();
  __m256i s4 = _mm256_setzero_si256();  // col 2a
  __m256i s5 = _mm256_setzero_si256();
  __m256i s6 = _mm256_setzero_si256();  // col 2b
  __m256i s7 = _mm256_setzero_si256();
  __m256i s8, s9, s10, s11;
  __m256i s12, s13, s14, s15;

  s11 = _mm256_set1_epi64x(0xFFFFFFFFUL);
  const __m256i* x_ptr = (__m256i*)x;
  const __m256i* y_ptr = (__m256i*)y;
  __m256i* res_ptr = (__m256i*)res;
  for (uint64_t i = 0; i < ell; ++i) {
    // treat item a
    s8 = _mm256_loadu_si256(x_ptr);
    s9 = _mm256_srli_epi64(s8, 32);

    s12 = _mm256_loadu_si256(y_ptr);
    s13 = _mm256_loadu_si256(y_ptr + 2);
    s14 = _mm256_srli_epi64(s12, 32);
    s15 = _mm256_srli_epi64(s13, 32);

    s12 = _mm256_mul_epu32(s8, s12);  // c1a -> s0,s1
    s13 = _mm256_mul_epu32(s8, s13);  // c2a -> s4,s5
    s14 = _mm256_mul_epu32(s9, s14);  // c1a -> s0,s1
    s15 = _mm256_mul_epu32(s9, s15);  // c2a -> s4,s5

    s8 = _mm256_and_si256(s12, s11);   // -> s0
    s9 = _mm256_and_si256(s13, s11);   // -> s4
    s12 = _mm256_srli_epi64(s12, 32);  // -> s1
    s13 = _mm256_srli_epi64(s13, 32);  // -> s5
    s0 = _mm256_add_epi64(s0, s8);
    s1 = _mm256_add_epi64(s1, s12);
    s4 = _mm256_add_epi64(s4, s9);
    s5 = _mm256_add_epi64(s5, s13);

    s8 = _mm256_and_si256(s14, s11);   // -> s0
    s9 = _mm256_and_si256(s15, s11);   // -> s4
    s14 = _mm256_srli_epi64(s14, 32);  // -> s1
    s15 = _mm256_srli_epi64(s15, 32);  // -> s5
    s0 = _mm256_add_epi64(s0, s8);
    s1 = _mm256_add_epi64(s1, s14);
    s4 = _mm256_add_epi64(s4, s9);
    s5 = _mm256_add_epi64(s5, s15);

    // treat item b
    s8 = _mm256_loadu_si256(x_ptr + 1);
    s9 = _mm256_srli_epi64(s8, 32);

    s12 = _mm256_loadu_si256(y_ptr + 1);
    s13 = _mm256_loadu_si256(y_ptr + 3);
    s14 = _mm256_srli_epi64(s12, 32);
    s15 = _mm256_srli_epi64(s13, 32);

    s12 = _mm256_mul_epu32(s8, s12);  // c1b -> s2,s3
    s13 = _mm256_mul_epu32(s8, s13);  // c2b -> s6,s7
    s14 = _mm256_mul_epu32(s9, s14);  // c1b -> s2,s3
    s15 = _mm256_mul_epu32(s9, s15);  // c2b -> s6,s7

    s8 = _mm256_and_si256(s12, s11);   // -> s2
    s9 = _mm256_and_si256(s13, s11);   // -> s6
    s12 = _mm256_srli_epi64(s12, 32);  // -> s3
    s13 = _mm256_srli_epi64(s13, 32);  // -> s7
    s2 = _mm256_add_epi64(s2, s8);
    s3 = _mm256_add_epi64(s3, s12);
    s6 = _mm256_add_epi64(s6, s9);
    s7 = _mm256_add_epi64(s7, s13);

    s8 = _mm256_and_si256(s14, s11);   // -> s2
    s9 = _mm256_and_si256(s15, s11);   // -> s6
    s14 = _mm256_srli_epi64(s14, 32);  // -> s3
    s15 = _mm256_srli_epi64(s15, 32);  // -> s7
    s2 = _mm256_add_epi64(s2, s8);
    s3 = _mm256_add_epi64(s3, s14);
    s6 = _mm256_add_epi64(s6, s9);
    s7 = _mm256_add_epi64(s7, s15);

    x_ptr += 2;
    y_ptr += 4;
  }
  // final reduction
  const uint64_t H2 = precomp->h;
  s8 = _mm256_set1_epi64x((UINT64_C(1) << H2) - 1);          // MASK2
  s9 = _mm256_loadu_si256((__m256i*)precomp->s2l_pow_red);   // S2L_POW_RED
  s10 = _mm256_loadu_si256((__m256i*)precomp->s2h_pow_red);  // S2H_POW_RED
  //--- s0,s1
  s11 = _mm256_and_si256(s1, s8);
  s12 = _mm256_srli_epi64(s1, H2);
  s13 = _mm256_mul_epu32(s11, s9);
  s14 = _mm256_mul_epu32(s12, s10);
  s0 = _mm256_add_epi64(s0, s13);
  s0 = _mm256_add_epi64(s0, s14);
  _mm256_storeu_si256(res_ptr + 0, s0);
  //--- s2,s3
  s11 = _mm256_and_si256(s3, s8);
  s12 = _mm256_srli_epi64(s3, H2);
  s13 = _mm256_mul_epu32(s11, s9);
  s14 = _mm256_mul_epu32(s12, s10);
  s2 = _mm256_add_epi64(s2, s13);
  s2 = _mm256_add_epi64(s2, s14);
  _mm256_storeu_si256(res_ptr + 1, s2);
  //--- s4,s5
  s11 = _mm256_and_si256(s5, s8);
  s12 = _mm256_srli_epi64(s5, H2);
  s13 = _mm256_mul_epu32(s11, s9);
  s14 = _mm256_mul_epu32(s12, s10);
  s4 = _mm256_add_epi64(s4, s13);
  s4 = _mm256_add_epi64(s4, s14);
  _mm256_storeu_si256(res_ptr + 2, s4);
  //--- s6,s7
  s11 = _mm256_and_si256(s7, s8);
  s12 = _mm256_srli_epi64(s7, H2);
  s13 = _mm256_mul_epu32(s11, s9);
  s14 = _mm256_mul_epu32(s12, s10);
  s6 = _mm256_add_epi64(s6, s13);
  s6 = _mm256_add_epi64(s6, s14);
  _mm256_storeu_si256(res_ptr + 3, s6);
}

EXPORT void q120x2_vec_mat1col_product_bbc_avx2(q120_mat1col_product_bbc_precomp* precomp, const uint64_t ell,
                                                q120b* const res, const q120b* const x, const q120c* const y) {
  __m256i s0 = _mm256_setzero_si256();  // col 1a
  __m256i s1 = _mm256_setzero_si256();
  __m256i s2 = _mm256_setzero_si256();  // col 1b
  __m256i s3 = _mm256_setzero_si256();
  __m256i s4 = _mm256_set1_epi64x(0xFFFFFFFFUL);
  __m256i s8, s9, s10, s11;
  __m256i s12, s13, s14, s15;

  const __m256i* x_ptr = (__m256i*)x;
  const __m256i* y_ptr = (__m256i*)y;
  __m256i* res_ptr = (__m256i*)res;
  for (uint64_t i = 0; i < ell; ++i) {
    s8 = _mm256_loadu_si256(x_ptr);
    s9 = _mm256_loadu_si256(x_ptr + 1);
    s10 = _mm256_srli_epi64(s8, 32);
    s11 = _mm256_srli_epi64(s9, 32);

    s12 = _mm256_loadu_si256(y_ptr);
    s13 = _mm256_loadu_si256(y_ptr + 1);
    s14 = _mm256_srli_epi64(s12, 32);
    s15 = _mm256_srli_epi64(s13, 32);

    s12 = _mm256_mul_epu32(s8, s12);   // -> s0,s1
    s13 = _mm256_mul_epu32(s9, s13);   // -> s2,s3
    s14 = _mm256_mul_epu32(s10, s14);  // -> s0,s1
    s15 = _mm256_mul_epu32(s11, s15);  // -> s2,s3

    s8 = _mm256_and_si256(s12, s4);    // -> s0
    s9 = _mm256_and_si256(s13, s4);    // -> s2
    s10 = _mm256_and_si256(s14, s4);   // -> s0
    s11 = _mm256_and_si256(s15, s4);   // -> s2
    s12 = _mm256_srli_epi64(s12, 32);  // -> s1
    s13 = _mm256_srli_epi64(s13, 32);  // -> s3
    s14 = _mm256_srli_epi64(s14, 32);  // -> s1
    s15 = _mm256_srli_epi64(s15, 32);  // -> s3

    s0 = _mm256_add_epi64(s0, s8);
    s1 = _mm256_add_epi64(s1, s12);
    s2 = _mm256_add_epi64(s2, s9);
    s3 = _mm256_add_epi64(s3, s13);
    s0 = _mm256_add_epi64(s0, s10);
    s1 = _mm256_add_epi64(s1, s14);
    s2 = _mm256_add_epi64(s2, s11);
    s3 = _mm256_add_epi64(s3, s15);

    x_ptr += 2;
    y_ptr += 2;
  }
  // final reduction
  const uint64_t H2 = precomp->h;
  s8 = _mm256_set1_epi64x((UINT64_C(1) << H2) - 1);          // MASK2
  s9 = _mm256_loadu_si256((__m256i*)precomp->s2l_pow_red);   // S2L_POW_RED
  s10 = _mm256_loadu_si256((__m256i*)precomp->s2h_pow_red);  // S2H_POW_RED
  //--- s0,s1
  s11 = _mm256_and_si256(s1, s8);
  s12 = _mm256_srli_epi64(s1, H2);
  s13 = _mm256_mul_epu32(s11, s9);
  s14 = _mm256_mul_epu32(s12, s10);
  s0 = _mm256_add_epi64(s0, s13);
  s0 = _mm256_add_epi64(s0, s14);
  _mm256_storeu_si256(res_ptr + 0, s0);
  //--- s2,s3
  s11 = _mm256_and_si256(s3, s8);
  s12 = _mm256_srli_epi64(s3, H2);
  s13 = _mm256_mul_epu32(s11, s9);
  s14 = _mm256_mul_epu32(s12, s10);
  s2 = _mm256_add_epi64(s2, s13);
  s2 = _mm256_add_epi64(s2, s14);
  _mm256_storeu_si256(res_ptr + 1, s2);
}
