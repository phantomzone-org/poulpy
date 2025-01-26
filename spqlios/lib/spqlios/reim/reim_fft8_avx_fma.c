#include <assert.h>
#include <immintrin.h>
#include <stdint.h>

#include "reim_fft_private.h"

__always_inline void reim_ctwiddle_avx_fma(__m256d* ra, __m256d* rb, __m256d* ia, __m256d* ib, const __m256d omre,
                                           const __m256d omim) {
  // rb * omre - ib * omim;
  __m256d rprod0 = _mm256_mul_pd(*ib, omim);
  rprod0 = _mm256_fmsub_pd(*rb, omre, rprod0);

  // rb * omim + ib * omre;
  __m256d iprod0 = _mm256_mul_pd(*rb, omim);
  iprod0 = _mm256_fmadd_pd(*ib, omre, iprod0);

  *rb = _mm256_sub_pd(*ra, rprod0);
  *ib = _mm256_sub_pd(*ia, iprod0);
  *ra = _mm256_add_pd(*ra, rprod0);
  *ia = _mm256_add_pd(*ia, iprod0);
}

EXPORT void reim_fft8_avx_fma(double* dre, double* dim, const void* ompv) {
  const double* omp = (const double*)ompv;

  __m256d ra0 = _mm256_loadu_pd(dre);
  __m256d ra4 = _mm256_loadu_pd(dre + 4);
  __m256d ia0 = _mm256_loadu_pd(dim);
  __m256d ia4 = _mm256_loadu_pd(dim + 4);

  // 1
  {
    // duplicate omegas in precomp?
    __m128d omri = _mm_loadu_pd(omp);
    __m256d omriri = _mm256_set_m128d(omri, omri);
    __m256d omi = _mm256_permute_pd(omriri, 15);
    __m256d omr = _mm256_permute_pd(omriri, 0);

    reim_ctwiddle_avx_fma(&ra0, &ra4, &ia0, &ia4, omr, omi);
  }

  // 2
  {
    const __m128d fft8neg = _mm_castsi128_pd(_mm_set_epi64x(UINT64_C(1) << 63, 0));
    __m128d omri = _mm_loadu_pd(omp + 2);             // r,i
    __m128d omrmi = _mm_xor_pd(omri, fft8neg);        // r,-i
    __m256d omrirmi = _mm256_set_m128d(omrmi, omri);  // r,i,r,-i
    __m256d omi = _mm256_permute_pd(omrirmi, 3);      // i,i,r,r
    __m256d omr = _mm256_permute_pd(omrirmi, 12);     // r,r,-i,-i

    __m256d rb = _mm256_permute2f128_pd(ra0, ra4, 0x31);
    __m256d ib = _mm256_permute2f128_pd(ia0, ia4, 0x31);
    __m256d ra = _mm256_permute2f128_pd(ra0, ra4, 0x20);
    __m256d ia = _mm256_permute2f128_pd(ia0, ia4, 0x20);

    reim_ctwiddle_avx_fma(&ra, &rb, &ia, &ib, omr, omi);

    ra0 = _mm256_permute2f128_pd(ra, rb, 0x20);
    ra4 = _mm256_permute2f128_pd(ra, rb, 0x31);
    ia0 = _mm256_permute2f128_pd(ia, ib, 0x20);
    ia4 = _mm256_permute2f128_pd(ia, ib, 0x31);
  }

  // 3
  {
    const __m256d fft8neg2 = _mm256_castsi256_pd(_mm256_set_epi64x(UINT64_C(1) << 63, UINT64_C(1) << 63, 0, 0));
    __m256d om = _mm256_loadu_pd(omp + 4);            // r0,r1,i0,i1
    __m256d omi = _mm256_permute2f128_pd(om, om, 1);  // i0,i1,r0,r1
    __m256d omr = _mm256_xor_pd(om, fft8neg2);        // r0,r1,-i0,-i1

    __m256d rb = _mm256_unpackhi_pd(ra0, ra4);
    __m256d ib = _mm256_unpackhi_pd(ia0, ia4);
    __m256d ra = _mm256_unpacklo_pd(ra0, ra4);
    __m256d ia = _mm256_unpacklo_pd(ia0, ia4);

    reim_ctwiddle_avx_fma(&ra, &rb, &ia, &ib, omr, omi);

    ra4 = _mm256_unpackhi_pd(ra, rb);
    ia4 = _mm256_unpackhi_pd(ia, ib);
    ra0 = _mm256_unpacklo_pd(ra, rb);
    ia0 = _mm256_unpacklo_pd(ia, ib);
  }

  // 4
  _mm256_storeu_pd(dre, ra0);
  _mm256_storeu_pd(dre + 4, ra4);
  _mm256_storeu_pd(dim, ia0);
  _mm256_storeu_pd(dim + 4, ia4);
}
