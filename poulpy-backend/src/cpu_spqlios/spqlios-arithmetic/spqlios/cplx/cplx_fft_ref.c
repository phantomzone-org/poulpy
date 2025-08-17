#include <memory.h>
#include <stdio.h>

#include "../commons_private.h"
#include "cplx_fft_internal.h"
#include "cplx_fft_private.h"

/** @brief (a,b) <- (a+omega.b,a-omega.b) */
void ctwiddle(CPLX a, CPLX b, const CPLX om) {
  double re = om[0] * b[0] - om[1] * b[1];
  double im = om[0] * b[1] + om[1] * b[0];
  b[0] = a[0] - re;
  b[1] = a[1] - im;
  a[0] += re;
  a[1] += im;
}

/**  @brief (a,b) <- (a+i.omega.b,a-i.omega.b) */
void citwiddle(CPLX a, CPLX b, const CPLX om) {
  double re = -om[1] * b[0] - om[0] * b[1];
  double im = -om[1] * b[1] + om[0] * b[0];
  b[0] = a[0] - re;
  b[1] = a[1] - im;
  a[0] += re;
  a[1] += im;
}

/**
 * @brief FFT modulo X^16-omega^2 (in registers)
 * @param data contains 16 complexes
 * @param omega 8 complexes in this order:
 *     omega,alpha,beta,j.beta,gamma,j.gamma,k.gamma,kj.gamma
 *     alpha = sqrt(omega), beta = sqrt(alpha), gamma = sqrt(beta)
 *     j = sqrt(i), k=sqrt(j)
 */
void cplx_fft16_ref(void* data, const void* omega) {
  CPLX* d = data;
  const CPLX* om = omega;
  // first pass
  for (uint64_t i = 0; i < 8; ++i) {
    ctwiddle(d[0 + i], d[8 + i], om[0]);
  }
  //
  ctwiddle(d[0], d[4], om[1]);
  ctwiddle(d[1], d[5], om[1]);
  ctwiddle(d[2], d[6], om[1]);
  ctwiddle(d[3], d[7], om[1]);
  citwiddle(d[8], d[12], om[1]);
  citwiddle(d[9], d[13], om[1]);
  citwiddle(d[10], d[14], om[1]);
  citwiddle(d[11], d[15], om[1]);
  //
  ctwiddle(d[0], d[2], om[2]);
  ctwiddle(d[1], d[3], om[2]);
  citwiddle(d[4], d[6], om[2]);
  citwiddle(d[5], d[7], om[2]);
  ctwiddle(d[8], d[10], om[3]);
  ctwiddle(d[9], d[11], om[3]);
  citwiddle(d[12], d[14], om[3]);
  citwiddle(d[13], d[15], om[3]);
  //
  ctwiddle(d[0], d[1], om[4]);
  citwiddle(d[2], d[3], om[4]);
  ctwiddle(d[4], d[5], om[5]);
  citwiddle(d[6], d[7], om[5]);
  ctwiddle(d[8], d[9], om[6]);
  citwiddle(d[10], d[11], om[6]);
  ctwiddle(d[12], d[13], om[7]);
  citwiddle(d[14], d[15], om[7]);
}

double cos_2pix(double x) { return m_accurate_cos(2 * M_PI * x); }
double sin_2pix(double x) { return m_accurate_sin(2 * M_PI * x); }
void cplx_set_e2pix(CPLX res, double x) {
  res[0] = cos_2pix(x);
  res[1] = sin_2pix(x);
}

void cplx_fft16_precomp(const double entry_pwr, CPLX** omg) {
  static const double j_pow = 1. / 8.;
  static const double k_pow = 1. / 16.;
  const double pom = entry_pwr / 2.;
  const double pom_2 = entry_pwr / 4.;
  const double pom_4 = entry_pwr / 8.;
  const double pom_8 = entry_pwr / 16.;
  cplx_set_e2pix((*omg)[0], pom);
  cplx_set_e2pix((*omg)[1], pom_2);
  cplx_set_e2pix((*omg)[2], pom_4);
  cplx_set_e2pix((*omg)[3], pom_4 + j_pow);
  cplx_set_e2pix((*omg)[4], pom_8);
  cplx_set_e2pix((*omg)[5], pom_8 + j_pow);
  cplx_set_e2pix((*omg)[6], pom_8 + k_pow);
  cplx_set_e2pix((*omg)[7], pom_8 + j_pow + k_pow);
  *omg += 8;
}

/**
 * @brief h twiddles-fft on the same omega
 * (also called merge-fft)merges 2 times h evaluations of even/odd polynomials into 2h evaluations of a sigle polynomial
 * Input:  P_0(z),...,P_{h-1}(z),P_h(z),...,P_{2h-1}(z)
 * Output: Q_0(y),...,Q_{h-1}(y),Q_0(-y),...,Q_{h-1}(-y)
 * where Q_i(X)=P_i(X^2)+X.P_{h+i}(X^2) and y^2 = z
 * @param h number of "coefficients" h >= 1
 * @param data 2h complex coefficients interleaved and 256b aligned
 * @param powom y represented as (yre,yim)
 */
void cplx_twiddle_fft_ref(int32_t h, CPLX* data, const CPLX powom) {
  CPLX* d0 = data;
  CPLX* d1 = data + h;
  for (uint64_t i = 0; i < h; ++i) {
    ctwiddle(d0[i], d1[i], powom);
  }
}

void cplx_bitwiddle_fft_ref(int32_t h, CPLX* data, const CPLX powom[2]) {
  CPLX* d0 = data;
  CPLX* d1 = data + h;
  CPLX* d2 = data + 2 * h;
  CPLX* d3 = data + 3 * h;
  for (uint64_t i = 0; i < h; ++i) {
    ctwiddle(d0[i], d2[i], powom[0]);
    ctwiddle(d1[i], d3[i], powom[0]);
  }
  for (uint64_t i = 0; i < h; ++i) {
    ctwiddle(d0[i], d1[i], powom[1]);
    citwiddle(d2[i], d3[i], powom[1]);
  }
}

/**
 * Input:  P_0(z),P_1(z)
 * Output: Q(y),Q(-y)
 * where Q(X)=P_0(X^2)+X.P_1(X^2) and y^2 = z
 * @param data 2 complexes coefficients interleaved and 256b aligned
 * @param powom (z,-z) interleaved: (zre,zim,-zre,-zim)
 */
void merge_fft_last_ref(CPLX* data, const CPLX powom) {
  CPLX prod;
  cplx_mul(prod, data[1], powom);
  cplx_sub(data[1], data[0], prod);
  cplx_add(data[0], data[0], prod);
}

void cplx_fft_ref_bfs_2(CPLX* dat, const CPLX** omg, uint32_t m) {
  CPLX* data = (CPLX*)dat;
  CPLX* const dend = data + m;
  for (int32_t h = m / 2; h >= 2; h >>= 1) {
    for (CPLX* d = data; d < dend; d += 2 * h) {
      if (memcmp((*omg)[0], (*omg)[1], 8) != 0) abort();
      cplx_twiddle_fft_ref(h, d, **omg);
      *omg += 2;
    }
#if 0
    printf("after merge %d: ", h);
    for (uint64_t ii=0; ii<nn/2; ++ii) {
      printf("%.6lf %.6lf ",data[ii][0],data[ii][1]);
    }
    printf("\n");
#endif
  }
  for (CPLX* d = data; d < dend; d += 2) {
    // TODO see if encoding changes
    if ((*omg)[0][0] != -(*omg)[1][0]) abort();
    if ((*omg)[0][1] != -(*omg)[1][1]) abort();
    merge_fft_last_ref(d, **omg);
    *omg += 2;
  }
#if 0
  printf("after last: ");
  for (uint64_t ii=0; ii<nn/2; ++ii) {
    printf("%.6lf %.6lf ",data[ii][0],data[ii][1]);
  }
  printf("\n");
#endif
}

void cplx_fft_ref_bfs_16(CPLX* dat, const CPLX** omg, uint32_t m) {
  CPLX* data = (CPLX*)dat;
  CPLX* const dend = data + m;
  uint32_t mm = m;
  uint32_t log2m = log2(m);
  if (log2m % 2 == 1) {
    cplx_twiddle_fft_ref(mm / 2, data, **omg);
    *omg += 2;
    mm >>= 1;
  }
  while (mm > 16) {
    uint32_t h = mm / 4;
    for (CPLX* d = data; d < dend; d += mm) {
      cplx_bitwiddle_fft_ref(h, d, *omg);
      *omg += 2;
    }
    mm = h;
  }
  for (CPLX* d = data; d < dend; d += 16) {
    cplx_fft16_ref(d, *omg);
    *omg += 8;
  }
#if 0
  printf("after last: ");
  for (uint64_t ii=0; ii<nn/2; ++ii) {
    printf("%.6lf %.6lf ",data[ii][0],data[ii][1]);
  }
  printf("\n");
#endif
}

/** @brief fft modulo X^m-exp(i.2pi.entry+pwr) -- reference code */
void cplx_fft_naive(const uint32_t m, const double entry_pwr, CPLX* data) {
  if (m == 1) return;
  const double pom = entry_pwr / 2.;
  const uint32_t h = m / 2;
  // apply the twiddle factors
  CPLX cpom;
  cplx_set_e2pix(cpom, pom);
  for (uint64_t i = 0; i < h; ++i) {
    ctwiddle(data[i], data[i + h], cpom);
  }
  // do the recursive calls
  cplx_fft_naive(h, pom, data);
  cplx_fft_naive(h, pom + 0.5, data + h);
}

/** @brief fills omega for cplx_fft_bfs_16 modulo X^m-exp(i.2.pi.entry_pwr) */
void fill_cplx_fft_omegas_bfs_16(const double entry_pwr, CPLX** omg, uint32_t m) {
  uint32_t mm = m;
  uint32_t log2m = log2(m);
  double ss = entry_pwr;
  if (log2m % 2 == 1) {
    uint32_t h = mm / 2;
    double pom = ss / 2.;
    for (uint32_t i = 0; i < m / mm; i++) {
      cplx_set_e2pix(omg[0][0], pom + fracrevbits(i) / 2.);
      cplx_set(omg[0][1], omg[0][0]);
      *omg += 2;
    }
    mm = h;
    ss = pom;
  }
  while (mm > 16) {
    double pom = ss / 4.;
    uint32_t h = mm / 4;
    for (uint32_t i = 0; i < m / mm; i++) {
      double om = pom + fracrevbits(i) / 4.;
      cplx_set_e2pix(omg[0][0], 2. * om);
      cplx_set_e2pix(omg[0][1], om);
      *omg += 2;
    }
    mm = h;
    ss = pom;
  }
  {
    // mm=16
    for (uint32_t i = 0; i < m / 16; i++) {
      cplx_fft16_precomp(ss + fracrevbits(i), omg);
    }
  }
}

/** @brief fills omega for cplx_fft_bfs_2 modulo X^m-exp(i.2.pi.entry_pwr) */
void fill_cplx_fft_omegas_bfs_2(const double entry_pwr, CPLX** omg, uint32_t m) {
  double pom = entry_pwr / 2.;
  for (int32_t h = m / 2; h >= 2; h >>= 1) {
    for (uint32_t i = 0; i < m / (2 * h); i++) {
      cplx_set_e2pix(omg[0][0], pom + fracrevbits(i) / 2.);
      cplx_set(omg[0][1], omg[0][0]);
      *omg += 2;
    }
    pom /= 2;
  }
  {
    // h=1
    for (uint32_t i = 0; i < m / 2; i++) {
      cplx_set_e2pix((*omg)[0], pom + fracrevbits(i) / 2.);
      cplx_neg((*omg)[1], (*omg)[0]);
      *omg += 2;
    }
  }
}

/** @brief fills omega for cplx_fft_rec modulo X^m-exp(i.2.pi.entry_pwr) */
void fill_cplx_fft_omegas_rec_16(const double entry_pwr, CPLX** omg, uint32_t m) {
  // note that the cases below are for recursive calls only!
  // externally, this function shall only be called with m>=4096
  if (m == 1) return;
  if (m <= 8) return fill_cplx_fft_omegas_bfs_2(entry_pwr, omg, m);
  if (m <= 2048) return fill_cplx_fft_omegas_bfs_16(entry_pwr, omg, m);
  double pom = entry_pwr / 2.;
  cplx_set_e2pix((*omg)[0], pom);
  cplx_set_e2pix((*omg)[1], pom);
  *omg += 2;
  fill_cplx_fft_omegas_rec_16(pom, omg, m / 2);
  fill_cplx_fft_omegas_rec_16(pom + 0.5, omg, m / 2);
}

void cplx_fft_ref_rec_16(CPLX* dat, const CPLX** omg, uint32_t m) {
  if (m == 1) return;
  if (m <= 8) return cplx_fft_ref_bfs_2(dat, omg, m);
  if (m <= 2048) return cplx_fft_ref_bfs_16(dat, omg, m);
  const uint32_t h = m / 2;
  if (memcmp((*omg)[0], (*omg)[1], 8) != 0) abort();
  cplx_twiddle_fft_ref(h, dat, **omg);
  *omg += 2;
  cplx_fft_ref_rec_16(dat, omg, h);
  cplx_fft_ref_rec_16(dat + h, omg, h);
}

void cplx_fft_ref(const CPLX_FFT_PRECOMP* precomp, void* d) {
  CPLX* data = (CPLX*)d;
  const int32_t m = precomp->m;
  const CPLX* omg = (CPLX*)precomp->powomegas;
  if (m == 1) return;
  if (m <= 8) return cplx_fft_ref_bfs_2(data, &omg, m);
  if (m <= 2048) return cplx_fft_ref_bfs_16(data, &omg, m);
  cplx_fft_ref_rec_16(data, &omg, m);
}

EXPORT CPLX_FFT_PRECOMP* new_cplx_fft_precomp(uint32_t m, uint32_t num_buffers) {
  const uint64_t OMG_SPACE = ceilto64b((2 * m) * sizeof(CPLX));
  const uint64_t BUF_SIZE = ceilto64b(m * sizeof(CPLX));
  void* reps = malloc(sizeof(CPLX_FFT_PRECOMP) + 63  // padding
                      + OMG_SPACE                    // tables //TODO 16?
                      + num_buffers * BUF_SIZE       // buffers
  );
  uint64_t aligned_addr = ceilto64b((uint64_t)(reps) + sizeof(CPLX_FFT_PRECOMP));
  CPLX_FFT_PRECOMP* r = (CPLX_FFT_PRECOMP*)reps;
  r->m = m;
  r->buf_size = BUF_SIZE;
  r->powomegas = (double*)aligned_addr;
  r->aligned_buffers = (void*)(aligned_addr + OMG_SPACE);
  // fill in powomegas
  CPLX* omg = (CPLX*)r->powomegas;
  if (m <= 8) {
    fill_cplx_fft_omegas_bfs_2(0.25, &omg, m);
  } else if (m <= 2048) {
    fill_cplx_fft_omegas_bfs_16(0.25, &omg, m);
  } else {
    fill_cplx_fft_omegas_rec_16(0.25, &omg, m);
  }
  if (((uint64_t)omg) - aligned_addr > OMG_SPACE) abort();
  // dispatch the right implementation
  {
    if (m <= 4) {
      // currently, we do not have any acceletated
      // implementation for m<=4
      r->function = cplx_fft_ref;
    } else if (CPU_SUPPORTS("fma")) {
      r->function = cplx_fft_avx2_fma;
    } else {
      r->function = cplx_fft_ref;
    }
  }
  return reps;
}

EXPORT void* cplx_fft_precomp_get_buffer(const CPLX_FFT_PRECOMP* tables, uint32_t buffer_index) {
  return (uint8_t*)tables->aligned_buffers + buffer_index * tables->buf_size;
}

EXPORT void cplx_fft_simple(uint32_t m, void* data) {
  static CPLX_FFT_PRECOMP* p[31] = {0};
  CPLX_FFT_PRECOMP** f = p + log2m(m);
  if (!*f) *f = new_cplx_fft_precomp(m, 0);
  (*f)->function(*f, data);
}

EXPORT void cplx_fft(const CPLX_FFT_PRECOMP* tables, void* data) { tables->function(tables, data); }
