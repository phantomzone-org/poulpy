#include <assert.h>
#include <immintrin.h>
#include <stdint.h>

#include "reim4_fftvec_private.h"

EXPORT void reim4_fftvec_addmul_fma(const REIM4_FFTVEC_ADDMUL_PRECOMP* tables, double* r_ptr, const double* a_ptr,
                                    const double* b_ptr) {
  const double* const rend_ptr = r_ptr + (tables->m << 1);
  while (r_ptr != rend_ptr) {
    __m256d rr = _mm256_loadu_pd(r_ptr);
    __m256d ri = _mm256_loadu_pd(r_ptr + 4);
    const __m256d ar = _mm256_loadu_pd(a_ptr);
    const __m256d ai = _mm256_loadu_pd(a_ptr + 4);
    const __m256d br = _mm256_loadu_pd(b_ptr);
    const __m256d bi = _mm256_loadu_pd(b_ptr + 4);

    rr = _mm256_fmsub_pd(ai, bi, rr);
    rr = _mm256_fmsub_pd(ar, br, rr);
    ri = _mm256_fmadd_pd(ar, bi, ri);
    ri = _mm256_fmadd_pd(ai, br, ri);

    _mm256_storeu_pd(r_ptr, rr);
    _mm256_storeu_pd(r_ptr + 4, ri);

    r_ptr += 8;
    a_ptr += 8;
    b_ptr += 8;
  }
}

EXPORT void reim4_fftvec_mul_fma(const REIM4_FFTVEC_MUL_PRECOMP* tables, double* r_ptr, const double* a_ptr,
                                 const double* b_ptr) {
  const double* const rend_ptr = r_ptr + (tables->m << 1);
  while (r_ptr != rend_ptr) {
    const __m256d ar = _mm256_loadu_pd(a_ptr);
    const __m256d ai = _mm256_loadu_pd(a_ptr + 4);
    const __m256d br = _mm256_loadu_pd(b_ptr);
    const __m256d bi = _mm256_loadu_pd(b_ptr + 4);

    const __m256d t1 = _mm256_mul_pd(ai, bi);
    const __m256d t2 = _mm256_mul_pd(ar, bi);

    __m256d rr = _mm256_fmsub_pd(ar, br, t1);
    __m256d ri = _mm256_fmadd_pd(ai, br, t2);

    _mm256_storeu_pd(r_ptr, rr);
    _mm256_storeu_pd(r_ptr + 4, ri);

    r_ptr += 8;
    a_ptr += 8;
    b_ptr += 8;
  }
}
