#include <immintrin.h>

#include "../commons_private.h"
#include "coeffs_arithmetic.h"

// res = a + b. dimension n must be a power of 2
EXPORT void znx_add_i64_avx(uint64_t nn, int64_t* res, const int64_t* a, const int64_t* b) {
  if (nn <= 2) {
    if (nn == 1) {
      res[0] = a[0] + b[0];
    } else {
      _mm_storeu_si128((__m128i*)res,                     //
                       _mm_add_epi64(                     //
                           _mm_loadu_si128((__m128i*)a),  //
                           _mm_loadu_si128((__m128i*)b)));
    }
  } else {
    const __m256i* aa = (__m256i*)a;
    const __m256i* bb = (__m256i*)b;
    __m256i* rr = (__m256i*)res;
    __m256i* const rrend = (__m256i*)(res + nn);
    do {
      _mm256_storeu_si256(rr,                          //
                          _mm256_add_epi64(            //
                              _mm256_loadu_si256(aa),  //
                              _mm256_loadu_si256(bb)));
      ++rr;
      ++aa;
      ++bb;
    } while (rr < rrend);
  }
}

// res = a - b. dimension n must be a power of 2
EXPORT void znx_sub_i64_avx(uint64_t nn, int64_t* res, const int64_t* a, const int64_t* b) {
  if (nn <= 2) {
    if (nn == 1) {
      res[0] = a[0] - b[0];
    } else {
      _mm_storeu_si128((__m128i*)res,                     //
                       _mm_sub_epi64(                     //
                           _mm_loadu_si128((__m128i*)a),  //
                           _mm_loadu_si128((__m128i*)b)));
    }
  } else {
    const __m256i* aa = (__m256i*)a;
    const __m256i* bb = (__m256i*)b;
    __m256i* rr = (__m256i*)res;
    __m256i* const rrend = (__m256i*)(res + nn);
    do {
      _mm256_storeu_si256(rr,                          //
                          _mm256_sub_epi64(            //
                              _mm256_loadu_si256(aa),  //
                              _mm256_loadu_si256(bb)));
      ++rr;
      ++aa;
      ++bb;
    } while (rr < rrend);
  }
}

EXPORT void znx_negate_i64_avx(uint64_t nn, int64_t* res, const int64_t* a) {
  if (nn <= 2) {
    if (nn == 1) {
      res[0] = -a[0];
    } else {
      _mm_storeu_si128((__m128i*)res,           //
                       _mm_sub_epi64(           //
                           _mm_set1_epi64x(0),  //
                           _mm_loadu_si128((__m128i*)a)));
    }
  } else {
    const __m256i* aa = (__m256i*)a;
    __m256i* rr = (__m256i*)res;
    __m256i* const rrend = (__m256i*)(res + nn);
    do {
      _mm256_storeu_si256(rr,                         //
                          _mm256_sub_epi64(           //
                              _mm256_set1_epi64x(0),  //
                              _mm256_loadu_si256(aa)));
      ++rr;
      ++aa;
    } while (rr < rrend);
  }
}

EXPORT void rnx_divide_by_m_avx(uint64_t n, double m, double* res, const double* a) {
  // TODO: see if there is a faster way of dividing by a power of 2?
  const double invm = 1. / m;
  if (n < 8) {
    switch (n) {
      case 1:
        *res = *a * invm;
        break;
      case 2:
        _mm_storeu_pd(res,                         //
                      _mm_mul_pd(_mm_loadu_pd(a),  //
                                 _mm_set1_pd(invm)));
        break;
      case 4:
        _mm256_storeu_pd(res,                               //
                         _mm256_mul_pd(_mm256_loadu_pd(a),  //
                                       _mm256_set1_pd(invm)));
        break;
      default:
        NOT_SUPPORTED();  // non-power of 2
    }
    return;
  }
  const __m256d invm256 = _mm256_set1_pd(invm);
  double* rr = res;
  const double* aa = a;
  const double* const aaend = a + n;
  do {
    _mm256_storeu_pd(rr,                                 //
                     _mm256_mul_pd(_mm256_loadu_pd(aa),  //
                                   invm256));
    _mm256_storeu_pd(rr + 4,                                 //
                     _mm256_mul_pd(_mm256_loadu_pd(aa + 4),  //
                                   invm256));
    rr += 8;
    aa += 8;
  } while (aa < aaend);
}
