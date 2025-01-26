#include <errno.h>
#include <immintrin.h>
#include <string.h>

#include "../commons_private.h"
#include "reim_fft_internal.h"
#include "reim_fft_private.h"

typedef union {double d; uint64_t u;} dblui64_t;

EXPORT void reim_to_tnx_avx(const REIM_TO_TNX_PRECOMP* tables, double* r, const double* x) {
  const uint64_t n = tables->m << 1;
  const __m256d add_cst = _mm256_set1_pd(tables->add_cst);
  const __m256d mask_and = _mm256_castsi256_pd(_mm256_set1_epi64x(tables->mask_and));
  const __m256d mask_or = _mm256_castsi256_pd(_mm256_set1_epi64x(tables->mask_or));
  const __m256d sub_cst = _mm256_set1_pd(tables->sub_cst);
  __m256d reg0,reg1;
  for (uint64_t i=0; i<n; i+=8) {
    reg0 = _mm256_loadu_pd(x+i);
    reg1 = _mm256_loadu_pd(x+i+4);
    reg0 = _mm256_add_pd(reg0, add_cst);
    reg1 = _mm256_add_pd(reg1, add_cst);
    reg0 = _mm256_and_pd(reg0, mask_and);
    reg1 = _mm256_and_pd(reg1, mask_and);
    reg0 = _mm256_or_pd(reg0, mask_or);
    reg1 = _mm256_or_pd(reg1, mask_or);
    reg0 = _mm256_sub_pd(reg0, sub_cst);
    reg1 = _mm256_sub_pd(reg1, sub_cst);
    _mm256_storeu_pd(r+i, reg0);
    _mm256_storeu_pd(r+i+4, reg1);
  }
}
