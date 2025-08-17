#include "immintrin.h"
#include "reim_fft_internal.h"
#include "reim_fft_private.h"

void reim_invtwiddle_ifft_ref(uint64_t h, double* re, double* im, double om[2]);
void reim_invbitwiddle_ifft_ref(uint64_t h, double* re, double* im, double om[4]);

__always_inline void reim_invtwiddle_ifft_avx2_fma(uint32_t h, double* re, double* im, double om[2]) {
  const __m128d omx = _mm_load_pd(om);
  const __m256d omra = _mm256_set_m128d(omx, omx);
  const __m256d omi = _mm256_unpackhi_pd(omra, omra);
  const __m256d omr = _mm256_unpacklo_pd(omra, omra);
  double* r0 = re;
  double* r1 = re + h;
  double* i0 = im;
  double* i1 = im + h;
  for (uint32_t i = 0; i < h; i += 4) {
    __m256d ur0 = _mm256_loadu_pd(r0 + i);
    __m256d ur1 = _mm256_loadu_pd(r1 + i);
    __m256d ui0 = _mm256_loadu_pd(i0 + i);
    __m256d ui1 = _mm256_loadu_pd(i1 + i);
    __m256d tra = _mm256_sub_pd(ur0, ur1);
    __m256d tia = _mm256_sub_pd(ui0, ui1);
    ur0 = _mm256_add_pd(ur0, ur1);
    ui0 = _mm256_add_pd(ui0, ui1);
    ur1 = _mm256_mul_pd(omi, tia);
    ui1 = _mm256_mul_pd(omi, tra);
    ur1 = _mm256_fmsub_pd(omr, tra, ur1);
    ui1 = _mm256_fmadd_pd(omr, tia, ui1);
    _mm256_storeu_pd(r0 + i, ur0);
    _mm256_storeu_pd(r1 + i, ur1);
    _mm256_storeu_pd(i0 + i, ui0);
    _mm256_storeu_pd(i1 + i, ui1);
  }
}

__always_inline void reim_invbitwiddle_ifft_avx2_fma(uint32_t h, double* re, double* im, double om[4]) {
  // reim_invbitwiddle_ifft_ref(h,re,im, om);
  double* const r0 = re;
  double* const r1 = re + h;
  double* const r2 = re + 2 * h;
  double* const r3 = re + 3 * h;
  double* const i0 = im;
  double* const i1 = im + h;
  double* const i2 = im + 2 * h;
  double* const i3 = im + 3 * h;
  const __m256d om0 = _mm256_loadu_pd(om);
  const __m256d omb = _mm256_permute2f128_pd(om0, om0, 0x11);
  const __m256d oma = _mm256_permute2f128_pd(om0, om0, 0x00);
  const __m256d omai = _mm256_unpackhi_pd(oma, oma);
  const __m256d omar = _mm256_unpacklo_pd(oma, oma);
  const __m256d ombi = _mm256_unpackhi_pd(omb, omb);
  const __m256d ombr = _mm256_unpacklo_pd(omb, omb);
  for (uint32_t i = 0; i < h; i += 4) {
    __m256d ur0 = _mm256_loadu_pd(r0 + i);
    __m256d ur1 = _mm256_loadu_pd(r1 + i);
    __m256d ur2 = _mm256_loadu_pd(r2 + i);
    __m256d ur3 = _mm256_loadu_pd(r3 + i);
    __m256d ui0 = _mm256_loadu_pd(i0 + i);
    __m256d ui1 = _mm256_loadu_pd(i1 + i);
    __m256d ui2 = _mm256_loadu_pd(i2 + i);
    __m256d ui3 = _mm256_loadu_pd(i3 + i);
    __m256d tra, trb, tia, tib;
    //------ twiddles 2
    tra = _mm256_sub_pd(ur0, ur1);
    trb = _mm256_sub_pd(ur2, ur3);
    tia = _mm256_sub_pd(ui0, ui1);
    tib = _mm256_sub_pd(ui2, ui3);
    ur0 = _mm256_add_pd(ur0, ur1);
    ur2 = _mm256_add_pd(ur2, ur3);
    ui0 = _mm256_add_pd(ui0, ui1);
    ui2 = _mm256_add_pd(ui2, ui3);
    ur1 = _mm256_mul_pd(omai, tia);
    ur3 = _mm256_mul_pd(omar, tib);
    ui1 = _mm256_mul_pd(omai, tra);
    ui3 = _mm256_mul_pd(omar, trb);
    ur1 = _mm256_fmsub_pd(omar, tra, ur1);
    ur3 = _mm256_fmadd_pd(omai, trb, ur3);
    ui1 = _mm256_fmadd_pd(omar, tia, ui1);
    ui3 = _mm256_fmsub_pd(omai, tib, ui3);
    //------ twiddles 1
    tra = _mm256_sub_pd(ur0, ur2);
    trb = _mm256_sub_pd(ur1, ur3);
    tia = _mm256_sub_pd(ui0, ui2);
    tib = _mm256_sub_pd(ui1, ui3);
    ur0 = _mm256_add_pd(ur0, ur2);
    ur1 = _mm256_add_pd(ur1, ur3);
    ui0 = _mm256_add_pd(ui0, ui2);
    ui1 = _mm256_add_pd(ui1, ui3);
    ur2 = _mm256_mul_pd(ombi, tia);
    ur3 = _mm256_mul_pd(ombi, tib);
    ui2 = _mm256_mul_pd(ombi, tra);
    ui3 = _mm256_mul_pd(ombi, trb);
    ur2 = _mm256_fmsub_pd(ombr, tra, ur2);
    ur3 = _mm256_fmsub_pd(ombr, trb, ur3);
    ui2 = _mm256_fmadd_pd(ombr, tia, ui2);
    ui3 = _mm256_fmadd_pd(ombr, tib, ui3);
    ///---
    _mm256_storeu_pd(r0 + i, ur0);
    _mm256_storeu_pd(r1 + i, ur1);
    _mm256_storeu_pd(r2 + i, ur2);
    _mm256_storeu_pd(r3 + i, ur3);
    _mm256_storeu_pd(i0 + i, ui0);
    _mm256_storeu_pd(i1 + i, ui1);
    _mm256_storeu_pd(i2 + i, ui2);
    _mm256_storeu_pd(i3 + i, ui3);
  }
}

void reim_ifft_bfs_16_avx2_fma(uint32_t m, double* re, double* im, double** omg) {
  uint32_t log2m = _mm_popcnt_u32(m - 1);  // log2(m);
  for (uint32_t off = 0; off < m; off += 16) {
    reim_ifft16_avx_fma(re + off, im + off, *omg);
    *omg += 16;
  }
  uint32_t h = 16;
  uint32_t ms2 = m >> 1;  // m/2
  while (h < ms2) {
    uint32_t mm = h << 2;
    for (uint32_t off = 0; off < m; off += mm) {
      reim_invbitwiddle_ifft_avx2_fma(h, re + off, im + off, *omg);
      *omg += 4;
    }
    h = mm;
  }
  if (log2m & 1) {
    // if (h!=ms2) abort(); // bug
    //  do the first twiddle iteration normally
    reim_invtwiddle_ifft_avx2_fma(h, re, im, *omg);
    *omg += 2;
    // h = m;
  }
}

void reim_ifft_rec_16_avx2_fma(uint32_t m, double* re, double* im, double** omg) {
  if (m <= 2048) return reim_ifft_bfs_16_avx2_fma(m, re, im, omg);
  const uint32_t h = m >> 1;  // m / 2;
  reim_ifft_rec_16_avx2_fma(h, re, im, omg);
  reim_ifft_rec_16_avx2_fma(h, re + h, im + h, omg);
  reim_invtwiddle_ifft_avx2_fma(h, re, im, *omg);
  *omg += 2;
}

void reim_ifft_avx2_fma(const REIM_IFFT_PRECOMP* precomp, double* dat) {
  const int32_t m = precomp->m;
  double* omg = precomp->powomegas;
  double* re = dat;
  double* im = dat + m;
  if (m <= 16) {
    switch (m) {
      case 1:
        return;
      case 2:
        return reim_ifft2_ref(re, im, omg);
      case 4:
        return reim_ifft4_avx_fma(re, im, omg);
      case 8:
        return reim_ifft8_avx_fma(re, im, omg);
      case 16:
        return reim_ifft16_avx_fma(re, im, omg);
      default:
        abort();  // m is not a power of 2
    }
  }
  if (m <= 2048) return reim_ifft_bfs_16_avx2_fma(m, re, im, &omg);
  return reim_ifft_rec_16_avx2_fma(m, re, im, &omg);
}
