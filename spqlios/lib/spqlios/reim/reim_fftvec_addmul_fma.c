#include <assert.h>
#include <immintrin.h>
#include <stdint.h>

#include "reim_fft_private.h"

EXPORT void reim_fftvec_addmul_fma(const REIM_FFTVEC_ADDMUL_PRECOMP* precomp, double* r, const double* a,
                                   const double* b) {
  const uint64_t m = precomp->m;
  double* rr_ptr = r;
  double* ri_ptr = r + m;
  const double* ar_ptr = a;
  const double* ai_ptr = a + m;
  const double* br_ptr = b;
  const double* bi_ptr = b + m;

  const double* const rend_ptr = ri_ptr;
  while (rr_ptr != rend_ptr) {
    __m256d rr = _mm256_loadu_pd(rr_ptr);
    __m256d ri = _mm256_loadu_pd(ri_ptr);
    const __m256d ar = _mm256_loadu_pd(ar_ptr);
    const __m256d ai = _mm256_loadu_pd(ai_ptr);
    const __m256d br = _mm256_loadu_pd(br_ptr);
    const __m256d bi = _mm256_loadu_pd(bi_ptr);

    rr = _mm256_fmsub_pd(ai, bi, rr);
    rr = _mm256_fmsub_pd(ar, br, rr);
    ri = _mm256_fmadd_pd(ar, bi, ri);
    ri = _mm256_fmadd_pd(ai, br, ri);

    _mm256_storeu_pd(rr_ptr, rr);
    _mm256_storeu_pd(ri_ptr, ri);

    rr_ptr += 4;
    ri_ptr += 4;
    ar_ptr += 4;
    ai_ptr += 4;
    br_ptr += 4;
    bi_ptr += 4;
  }
}

EXPORT void reim_fftvec_mul_fma(const REIM_FFTVEC_MUL_PRECOMP* precomp, double* r, const double* a, const double* b) {
  const uint64_t m = precomp->m;
  double* rr_ptr = r;
  double* ri_ptr = r + m;
  const double* ar_ptr = a;
  const double* ai_ptr = a + m;
  const double* br_ptr = b;
  const double* bi_ptr = b + m;

  const double* const rend_ptr = ri_ptr;
  while (rr_ptr != rend_ptr) {
    const __m256d ar = _mm256_loadu_pd(ar_ptr);
    const __m256d ai = _mm256_loadu_pd(ai_ptr);
    const __m256d br = _mm256_loadu_pd(br_ptr);
    const __m256d bi = _mm256_loadu_pd(bi_ptr);

    const __m256d t1 = _mm256_mul_pd(ai, bi);
    const __m256d t2 = _mm256_mul_pd(ar, bi);

    __m256d rr = _mm256_fmsub_pd(ar, br, t1);
    __m256d ri = _mm256_fmadd_pd(ai, br, t2);

    _mm256_storeu_pd(rr_ptr, rr);
    _mm256_storeu_pd(ri_ptr, ri);

    rr_ptr += 4;
    ri_ptr += 4;
    ar_ptr += 4;
    ai_ptr += 4;
    br_ptr += 4;
    bi_ptr += 4;
  }
}
