#include <immintrin.h>

#include "reim_fft_private.h"

void reim_from_znx64_bnd50_fma(const REIM_FROM_ZNX64_PRECOMP* precomp, void* r, const int64_t* x) {
  static const double EXPO = INT64_C(1) << 52;
  const int64_t ADD_CST = INT64_C(1) << 51;
  const double SUB_CST = INT64_C(3) << 51;

  const __m256d SUB_CST_4 = _mm256_set1_pd(SUB_CST);
  const __m256i ADD_CST_4 = _mm256_set1_epi64x(ADD_CST);
  const __m256d EXPO_4 = _mm256_set1_pd(EXPO);

  double(*out)[4] = (double(*)[4])r;
  __m256i* in = (__m256i*)x;
  __m256i* inend = (__m256i*)(x + (precomp->m << 1));
  do {
    // read the next value
    __m256i a = _mm256_loadu_si256(in);
    a = _mm256_add_epi64(a, ADD_CST_4);
    __m256d ad = _mm256_castsi256_pd(a);
    ad = _mm256_or_pd(ad, EXPO_4);
    ad = _mm256_sub_pd(ad, SUB_CST_4);
    // store the next value
    _mm256_storeu_pd(out[0], ad);
    ++out;
    ++in;
  } while (in < inend);
}

// version where the output norm can be as big as 2^63
void reim_to_znx64_avx2_bnd63_fma(const REIM_TO_ZNX64_PRECOMP* precomp, int64_t* r, const void* x) {
  static const uint64_t SIGN_MASK = 0x8000000000000000UL;
  static const uint64_t EXPO_MASK = 0x7FF0000000000000UL;
  static const uint64_t MANTISSA_MASK = 0x000FFFFFFFFFFFFFUL;
  static const uint64_t MANTISSA_MSB = 0x0010000000000000UL;
  const double divisor_bits = precomp->divisor * ((double)(INT64_C(1) << 52));
  const double offset = precomp->divisor / 2.;

  const __m256d SIGN_MASK_4 = _mm256_castsi256_pd(_mm256_set1_epi64x(SIGN_MASK));
  const __m256i EXPO_MASK_4 = _mm256_set1_epi64x(EXPO_MASK);
  const __m256i MANTISSA_MASK_4 = _mm256_set1_epi64x(MANTISSA_MASK);
  const __m256i MANTISSA_MSB_4 = _mm256_set1_epi64x(MANTISSA_MSB);
  const __m256d offset_4 = _mm256_set1_pd(offset);
  const __m256i divi_bits_4 = _mm256_castpd_si256(_mm256_set1_pd(divisor_bits));

  double(*in)[4] = (double(*)[4])x;
  __m256i* out = (__m256i*)r;
  __m256i* outend = (__m256i*)(r + (precomp->m << 1));
  do {
    // read the next value
    __m256d a = _mm256_loadu_pd(in[0]);
    // a += sign(a) * m/2
    __m256d asign = _mm256_and_pd(a, SIGN_MASK_4);
    a = _mm256_add_pd(a, _mm256_or_pd(asign, offset_4));
    // sign: either 0 or -1
    __m256i sign_mask = _mm256_castpd_si256(asign);
    sign_mask = _mm256_sub_epi64(_mm256_set1_epi64x(0), _mm256_srli_epi64(sign_mask, 63));
    // compute the exponents
    __m256i a0exp = _mm256_and_si256(_mm256_castpd_si256(a), EXPO_MASK_4);
    __m256i a0lsh = _mm256_sub_epi64(a0exp, divi_bits_4);
    __m256i a0rsh = _mm256_sub_epi64(divi_bits_4, a0exp);
    a0lsh = _mm256_srli_epi64(a0lsh, 52);
    a0rsh = _mm256_srli_epi64(a0rsh, 52);
    // compute the new mantissa
    __m256i a0pos = _mm256_and_si256(_mm256_castpd_si256(a), MANTISSA_MASK_4);
    a0pos = _mm256_or_si256(a0pos, MANTISSA_MSB_4);
    a0lsh = _mm256_sllv_epi64(a0pos, a0lsh);
    a0rsh = _mm256_srlv_epi64(a0pos, a0rsh);
    __m256i final = _mm256_or_si256(a0lsh, a0rsh);
    // negate if the sign was negative
    final = _mm256_xor_si256(final, sign_mask);
    final = _mm256_sub_epi64(final, sign_mask);
    // read the next value
    _mm256_storeu_si256(out, final);
    ++out;
    ++in;
  } while (out < outend);
}

// version where the output norm can be as big as 2^50
void reim_to_znx64_avx2_bnd50_fma(const REIM_TO_ZNX64_PRECOMP* precomp, int64_t* r, const void* x) {
  static const uint64_t MANTISSA_MASK = 0x000FFFFFFFFFFFFFUL;
  const int64_t SUB_CST = INT64_C(1) << 51;
  const double add_cst = precomp->divisor * ((double)(INT64_C(3) << 51));

  const __m256i SUB_CST_4 = _mm256_set1_epi64x(SUB_CST);
  const __m256d add_cst_4 = _mm256_set1_pd(add_cst);
  const __m256i MANTISSA_MASK_4 = _mm256_set1_epi64x(MANTISSA_MASK);

  double(*in)[4] = (double(*)[4])x;
  __m256i* out = (__m256i*)r;
  __m256i* outend = (__m256i*)(r + (precomp->m << 1));
  do {
    // read the next value
    __m256d a = _mm256_loadu_pd(in[0]);
    a = _mm256_add_pd(a, add_cst_4);
    __m256i ai = _mm256_castpd_si256(a);
    ai = _mm256_and_si256(ai, MANTISSA_MASK_4);
    ai = _mm256_sub_epi64(ai, SUB_CST_4);
    // store the next value
    _mm256_storeu_si256(out, ai);
    ++out;
    ++in;
  } while (out < outend);
}
