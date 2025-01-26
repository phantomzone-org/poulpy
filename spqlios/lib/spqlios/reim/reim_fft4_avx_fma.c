#include <assert.h>
#include <immintrin.h>
#include <stdint.h>

#include "reim_fft_private.h"

__always_inline void reim_ctwiddle_avx_fma(__m128d* ra, __m128d* rb, __m128d* ia, __m128d* ib, const __m128d omre,
                                           const __m128d omim) {
  // rb * omre - ib * omim;
  __m128d rprod0 = _mm_mul_pd(*ib, omim);
  rprod0 = _mm_fmsub_pd(*rb, omre, rprod0);

  // rb * omim + ib * omre;
  __m128d iprod0 = _mm_mul_pd(*rb, omim);
  iprod0 = _mm_fmadd_pd(*ib, omre, iprod0);

  *rb = _mm_sub_pd(*ra, rprod0);
  *ib = _mm_sub_pd(*ia, iprod0);
  *ra = _mm_add_pd(*ra, rprod0);
  *ia = _mm_add_pd(*ia, iprod0);
}

EXPORT void reim_fft4_avx_fma(double* dre, double* dim, const void* ompv) {
  const double* omp = (const double*)ompv;

  __m128d ra0 = _mm_loadu_pd(dre);
  __m128d ra2 = _mm_loadu_pd(dre + 2);
  __m128d ia0 = _mm_loadu_pd(dim);
  __m128d ia2 = _mm_loadu_pd(dim + 2);

  // 1
  {
    // duplicate omegas in precomp?
    __m128d om = _mm_loadu_pd(omp);
    __m128d omre = _mm_permute_pd(om, 0);
    __m128d omim = _mm_permute_pd(om, 3);

    reim_ctwiddle_avx_fma(&ra0, &ra2, &ia0, &ia2, omre, omim);
  }

  // 2
  {
    const __m128d fft4neg = _mm_castsi128_pd(_mm_set_epi64x(UINT64_C(1) << 63, 0));
    __m128d om = _mm_loadu_pd(omp + 2);      // om: r,i
    __m128d omim = _mm_permute_pd(om, 1);    // omim: i,r
    __m128d omre = _mm_xor_pd(om, fft4neg);  // omre: r,-i

    __m128d rb = _mm_unpackhi_pd(ra0, ra2);  // (r0, r1), (r2, r3) -> (r1, r3)
    __m128d ib = _mm_unpackhi_pd(ia0, ia2);  // (i0, i1), (i2, i3) -> (i1, i3)
    __m128d ra = _mm_unpacklo_pd(ra0, ra2);  // (r0, r1), (r2, r3) -> (r0, r2)
    __m128d ia = _mm_unpacklo_pd(ia0, ia2);  // (i0, i1), (i2, i3) -> (i0, i2)

    reim_ctwiddle_avx_fma(&ra, &rb, &ia, &ib, omre, omim);

    ra0 = _mm_unpacklo_pd(ra, rb);
    ia0 = _mm_unpacklo_pd(ia, ib);
    ra2 = _mm_unpackhi_pd(ra, rb);
    ia2 = _mm_unpackhi_pd(ia, ib);
  }

  // 4
  _mm_storeu_pd(dre, ra0);
  _mm_storeu_pd(dre + 2, ra2);
  _mm_storeu_pd(dim, ia0);
  _mm_storeu_pd(dim + 2, ia2);
}
