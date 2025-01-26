#include <immintrin.h>
#include <string.h>

#include "cplx_fft_internal.h"
#include "cplx_fft_private.h"

typedef double D4MEM[4];

/**
 * @brief complex fft via bfs strategy (for m between 2 and 8)
 * @param dat the data to run the algorithm on
 * @param omg precomputed tables (must have been filled with fill_omega)
 * @param m ring dimension of the FFT (modulo X^m-i)
 */
void cplx_fft_avx2_fma_bfs_2(D4MEM* dat, const D4MEM** omg, uint32_t m) {
  double* data = (double*)dat;
  int32_t _2nblock = m >> 1;  // = h in ref code
  D4MEM* const finaldd = (D4MEM*)(data + 2 * m);
  while (_2nblock >= 2) {
    int32_t nblock = _2nblock >> 1;  // =h/2 in ref code
    D4MEM* dd = (D4MEM*)data;
    do {
      const __m256d om = _mm256_load_pd(*omg[0]);
      const __m256d omim = _mm256_addsub_pd(_mm256_setzero_pd(), _mm256_unpackhi_pd(om, om));
      const __m256d omre = _mm256_unpacklo_pd(om, om);
      D4MEM* const ddend = (dd + nblock);
      D4MEM* ddmid = ddend;
      do {
        const __m256d b = _mm256_loadu_pd(ddmid[0]);
        const __m256d t1 = _mm256_mul_pd(b, omre);
        const __m256d barb = _mm256_shuffle_pd(b, b, 5);
        const __m256d t2 = _mm256_fmadd_pd(barb, omim, t1);
        const __m256d a = _mm256_loadu_pd(dd[0]);
        const __m256d newa = _mm256_add_pd(a, t2);
        const __m256d newb = _mm256_sub_pd(a, t2);
        _mm256_storeu_pd(dd[0], newa);
        _mm256_storeu_pd(ddmid[0], newb);
        dd += 1;
        ddmid += 1;
      } while (dd < ddend);
      dd += nblock;
      *omg += 1;
    } while (dd < finaldd);
    _2nblock >>= 1;
  }
  // last iteration when _2nblock == 1
  {
    D4MEM* dd = (D4MEM*)data;
    do {
      const __m256d om = _mm256_load_pd(*omg[0]);
      const __m256d omre = _mm256_unpacklo_pd(om, om);
      const __m256d omim = _mm256_unpackhi_pd(om, om);
      const __m256d ab = _mm256_loadu_pd(dd[0]);
      const __m256d bb = _mm256_permute4x64_pd(ab, 0b11101110);
      const __m256d bbbar = _mm256_permute4x64_pd(ab, 0b10111011);
      const __m256d t1 = _mm256_mul_pd(bbbar, omim);
      const __m256d t2 = _mm256_fmaddsub_pd(bb, omre, t1);
      const __m256d aa = _mm256_permute4x64_pd(ab, 0b01000100);
      const __m256d newab = _mm256_add_pd(aa, t2);
      _mm256_storeu_pd(dd[0], newab);
      dd += 1;
      *omg += 1;
    } while (dd < finaldd);
  }
}

__always_inline void cplx_twiddle_fft_avx2(int32_t h, D4MEM* data, const void* omg) {
  const __m256d om = _mm256_loadu_pd(omg);
  const __m256d omim = _mm256_unpackhi_pd(om, om);
  const __m256d omre = _mm256_unpacklo_pd(om, om);
  D4MEM* d0 = data;
  D4MEM* const ddend = d0 + (h>>1);
  D4MEM* d1 = ddend;
  do {
    const __m256d b = _mm256_loadu_pd(d1[0]);
    const __m256d barb = _mm256_shuffle_pd(b, b, 5);
    const __m256d t1 = _mm256_mul_pd(barb, omim);
    const __m256d t2 = _mm256_fmaddsub_pd(b, omre, t1);
    const __m256d a = _mm256_loadu_pd(d0[0]);
    const __m256d newa = _mm256_add_pd(a, t2);
    const __m256d newb = _mm256_sub_pd(a, t2);
    _mm256_storeu_pd(d0[0], newa);
    _mm256_storeu_pd(d1[0], newb);
    d0 += 1;
    d1 += 1;
  } while (d0 < ddend);
}

__always_inline void cplx_bitwiddle_fft_avx2(int32_t h, void* data, const void* powom) {
  const __m256d omx = _mm256_loadu_pd(powom);
  const __m256d oma = _mm256_permute2f128_pd(omx, omx, 0x00);
  const __m256d omb = _mm256_permute2f128_pd(omx, omx, 0x11);
  const __m256d omaim = _mm256_unpackhi_pd(oma, oma);
  const __m256d omare = _mm256_unpacklo_pd(oma, oma);
  const __m256d ombim = _mm256_unpackhi_pd(omb, omb);
  const __m256d ombre = _mm256_unpacklo_pd(omb, omb);
  D4MEM* d0 = (D4MEM*) data;
  D4MEM* const ddend = d0 + (h>>1);
  D4MEM* d1 = ddend;
  D4MEM* d2 = d0+h;
  D4MEM* d3 = d1+h;
  __m256d reg0,reg1,reg2,reg3,tmp0,tmp1;
  do {
    reg0 = _mm256_loadu_pd(d0[0]);
    reg1 = _mm256_loadu_pd(d1[0]);
    reg2 = _mm256_loadu_pd(d2[0]);
    reg3 = _mm256_loadu_pd(d3[0]);
    tmp0 = _mm256_shuffle_pd(reg2, reg2, 5);
    tmp1 = _mm256_shuffle_pd(reg3, reg3, 5);
    tmp0 = _mm256_mul_pd(tmp0, omaim);
    tmp1 = _mm256_mul_pd(tmp1, omaim);
    tmp0 = _mm256_fmaddsub_pd(reg2, omare, tmp0);
    tmp1 = _mm256_fmaddsub_pd(reg3, omare, tmp1);
    reg2 = _mm256_sub_pd(reg0, tmp0);
    reg3 = _mm256_sub_pd(reg1, tmp1);
    reg0 = _mm256_add_pd(reg0, tmp0);
    reg1 = _mm256_add_pd(reg1, tmp1);
    //--------------------------------------
    tmp0 = _mm256_shuffle_pd(reg1, reg1, 5);
    tmp1 = _mm256_shuffle_pd(reg3, reg3, 5);
    tmp0 = _mm256_mul_pd(tmp0, ombim);  //(r,i)
    tmp1 = _mm256_mul_pd(tmp1, ombre);  //(-i,r)
    tmp0 = _mm256_fmaddsub_pd(reg1, ombre, tmp0);
    tmp1 = _mm256_fmsubadd_pd(reg3, ombim, tmp1);
    reg1 = _mm256_sub_pd(reg0, tmp0);
    reg3 = _mm256_add_pd(reg2, tmp1);
    reg0 = _mm256_add_pd(reg0, tmp0);
    reg2 = _mm256_sub_pd(reg2, tmp1);
    /////
    _mm256_storeu_pd(d0[0], reg0);
    _mm256_storeu_pd(d1[0], reg1);
    _mm256_storeu_pd(d2[0], reg2);
    _mm256_storeu_pd(d3[0], reg3);
    d0 += 1;
    d1 += 1;
    d2 += 1;
    d3 += 1;
  } while (d0 < ddend);
}

/**
 * @brief complex fft via bfs strategy (for m >= 16)
 * @param dat the data to run the algorithm on
 * @param omg precomputed tables (must have been filled with fill_omega)
 * @param m ring dimension of the FFT (modulo X^m-i)
 */
void cplx_fft_avx2_fma_bfs_16(D4MEM* dat, const D4MEM** omg, uint32_t m) {
  double* data = (double*)dat;
  D4MEM* const finaldd = (D4MEM*)(data + 2 * m);
  uint32_t mm = m;
  uint32_t log2m = _mm_popcnt_u32(m-1); // log2(m)
  if (log2m % 2 == 1) {
    uint32_t h = mm>>1;
    cplx_twiddle_fft_avx2(h, dat, **omg);
    *omg += 1;
    mm >>= 1;
  }
  while(mm>16) {
    uint32_t h = mm/4;
    for (CPLX* d = (CPLX*) data; d < (CPLX*) finaldd; d += mm) {
      cplx_bitwiddle_fft_avx2(h, d, (CPLX*) *omg);
      *omg += 1;
    }
    mm=h;
  }
  {
    D4MEM* dd = (D4MEM*)data;
    do {
      cplx_fft16_avx_fma(dd, *omg);
      dd += 8;
      *omg += 4;
    } while (dd < finaldd);
    _mm256_zeroupper();
  }
  /*
  int32_t _2nblock = m >> 1;  // = h in ref code
  while (_2nblock >= 16) {
    int32_t nblock = _2nblock >> 1;  // =h/2 in ref code
    D4MEM* dd = (D4MEM*)data;
    do {
      const __m256d om = _mm256_load_pd(*omg[0]);
      const __m256d omim = _mm256_addsub_pd(_mm256_setzero_pd(), _mm256_unpackhi_pd(om, om));
      const __m256d omre = _mm256_unpacklo_pd(om, om);
      D4MEM* const ddend = (dd + nblock);
      D4MEM* ddmid = ddend;
      do {
        const __m256d b = _mm256_loadu_pd(ddmid[0]);
        const __m256d t1 = _mm256_mul_pd(b, omre);
        const __m256d barb = _mm256_shuffle_pd(b, b, 5);
        const __m256d t2 = _mm256_fmadd_pd(barb, omim, t1);
        const __m256d a = _mm256_loadu_pd(dd[0]);
        const __m256d newa = _mm256_add_pd(a, t2);
        const __m256d newb = _mm256_sub_pd(a, t2);
        _mm256_storeu_pd(dd[0], newa);
        _mm256_storeu_pd(ddmid[0], newb);
        dd += 1;
        ddmid += 1;
      } while (dd < ddend);
      dd += nblock;
      *omg += 1;
    } while (dd < finaldd);
    _2nblock >>= 1;
  }
  // last iteration when _2nblock == 8
  {
    D4MEM* dd = (D4MEM*)data;
    do {
      cplx_fft16_avx_fma(dd, *omg);
      dd += 8;
      *omg += 4;
    } while (dd < finaldd);
    _mm256_zeroupper();
  }
   */
}

/**
 * @brief complex fft via dfs recursion (for m >= 16)
 * @param dat the data to run the algorithm on
 * @param omg precomputed tables (must have been filled with fill_omega)
 * @param m ring dimension of the FFT (modulo X^m-i)
 */
void cplx_fft_avx2_fma_rec_16(D4MEM* dat, const D4MEM** omg, uint32_t m) {
  if (m <= 8) return cplx_fft_avx2_fma_bfs_2(dat, omg, m);
  if (m <= 2048) return cplx_fft_avx2_fma_bfs_16(dat, omg, m);
  double* data = (double*)dat;
  int32_t _2nblock = m >> 1;       // = h in ref code
  int32_t nblock = _2nblock >> 1;  // =h/2 in ref code
  D4MEM* dd = (D4MEM*)data;
  const __m256d om = _mm256_load_pd(*omg[0]);
  const __m256d omim = _mm256_addsub_pd(_mm256_setzero_pd(), _mm256_unpackhi_pd(om, om));
  const __m256d omre = _mm256_unpacklo_pd(om, om);
  D4MEM* const ddend = (dd + nblock);
  D4MEM* ddmid = ddend;
  do {
    const __m256d b = _mm256_loadu_pd(ddmid[0]);
    const __m256d t1 = _mm256_mul_pd(b, omre);
    const __m256d barb = _mm256_shuffle_pd(b, b, 5);
    const __m256d t2 = _mm256_fmadd_pd(barb, omim, t1);
    const __m256d a = _mm256_loadu_pd(dd[0]);
    const __m256d newa = _mm256_add_pd(a, t2);
    const __m256d newb = _mm256_sub_pd(a, t2);
    _mm256_storeu_pd(dd[0], newa);
    _mm256_storeu_pd(ddmid[0], newb);
    dd += 1;
    ddmid += 1;
  } while (dd < ddend);
  *omg += 1;
  cplx_fft_avx2_fma_rec_16(dat, omg, _2nblock);
  cplx_fft_avx2_fma_rec_16(ddend, omg, _2nblock);
}

/**
 * @brief complex fft via best strategy (for m>=1)
 * @param dat the data to run the algorithm on: m complex numbers
 * @param omg precomputed tables (must have been filled with fill_omega)
 * @param m ring dimension of the FFT (modulo X^m-i)
 */
EXPORT void cplx_fft_avx2_fma(const CPLX_FFT_PRECOMP* precomp, void* d) {
  const uint32_t m = precomp->m;
  const D4MEM* omg = (D4MEM*)precomp->powomegas;
  if (m <= 1) return;
  if (m <= 8) return cplx_fft_avx2_fma_bfs_2(d, &omg, m);
  if (m <= 2048) return cplx_fft_avx2_fma_bfs_16(d, &omg, m);
  cplx_fft_avx2_fma_rec_16(d, &omg, m);
}
