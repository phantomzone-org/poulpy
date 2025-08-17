#include <assert.h>
#include <immintrin.h>
#include <stdint.h>

#include "reim4_fftvec_private.h"

EXPORT void reim4_from_cplx_fma(const REIM4_FROM_CPLX_PRECOMP* tables, double* r_ptr, const void* a) {
  const double* const rend_ptr = r_ptr + (tables->m << 1);

  const double* a_ptr = (double*)a;
  while (r_ptr != rend_ptr) {
    __m256d t1 = _mm256_loadu_pd(a_ptr);
    __m256d t2 = _mm256_loadu_pd(a_ptr + 4);

    _mm256_storeu_pd(r_ptr, _mm256_unpacklo_pd(t1, t2));
    _mm256_storeu_pd(r_ptr + 4, _mm256_unpackhi_pd(t1, t2));

    r_ptr += 8;
    a_ptr += 8;
  }
}

EXPORT void reim4_to_cplx_fma(const REIM4_TO_CPLX_PRECOMP* tables, void* r, const double* a_ptr) {
  const double* const aend_ptr = a_ptr + (tables->m << 1);
  double* r_ptr = (double*)r;

  while (a_ptr != aend_ptr) {
    __m256d t1 = _mm256_loadu_pd(a_ptr);
    __m256d t2 = _mm256_loadu_pd(a_ptr + 4);

    _mm256_storeu_pd(r_ptr, _mm256_unpacklo_pd(t1, t2));
    _mm256_storeu_pd(r_ptr + 4, _mm256_unpackhi_pd(t1, t2));

    r_ptr += 8;
    a_ptr += 8;
  }
}
