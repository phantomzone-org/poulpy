#include <memory.h>

#include "../commons_private.h"
#include "cplx_fft.h"
#include "cplx_fft_internal.h"
#include "cplx_fft_private.h"

/** @brief (a,b) <- (a+b,omegabar.(a-b)) */
void invctwiddle(CPLX a, CPLX b, const CPLX ombar) {
  double diffre = a[0] - b[0];
  double diffim = a[1] - b[1];
  a[0] = a[0] + b[0];
  a[1] = a[1] + b[1];
  b[0] = diffre * ombar[0] - diffim * ombar[1];
  b[1] = diffre * ombar[1] + diffim * ombar[0];
}

/**  @brief (a,b) <- (a+b,-i.omegabar(a-b)) */
void invcitwiddle(CPLX a, CPLX b, const CPLX ombar) {
  double diffre = a[0] - b[0];
  double diffim = a[1] - b[1];
  a[0] = a[0] + b[0];
  a[1] = a[1] + b[1];
  //-i(x+iy)=-ix+y
  b[0] = diffre * ombar[1] + diffim * ombar[0];
  b[1] = -diffre * ombar[0] + diffim * ombar[1];
}

/** @brief exp(-i.2pi.x) */
void cplx_set_e2pimx(CPLX res, double x) {
  res[0] = m_accurate_cos(2 * M_PI * x);
  res[1] = -m_accurate_sin(2 * M_PI * x);
}
/**
 * @brief this computes the sequence: 0,1/2,1/4,3/4,1/8,5/8,3/8,7/8,...
 * essentially: the bits of (i+1) in lsb order on the basis (1/2^k) mod 1*/
double fracrevbits(uint32_t i);
/** @brief fft modulo X^m-exp(i.2pi.entry+pwr) -- reference code */
void cplx_ifft_naive(const uint32_t m, const double entry_pwr, CPLX* data) {
  if (m == 1) return;
  const double pom = entry_pwr / 2.;
  const uint32_t h = m / 2;
  CPLX cpom;
  cplx_set_e2pimx(cpom, pom);
  // do the recursive calls
  cplx_ifft_naive(h, pom, data);
  cplx_ifft_naive(h, pom + 0.5, data + h);
  // apply the inverse twiddle factors
  for (uint64_t i = 0; i < h; ++i) {
    invctwiddle(data[i], data[i + h], cpom);
  }
}

void cplx_ifft16_precomp(const double entry_pwr, CPLX** omg) {
  static const double j_pow = 1. / 8.;
  static const double k_pow = 1. / 16.;
  const double pom = entry_pwr / 2.;
  const double pom_2 = entry_pwr / 4.;
  const double pom_4 = entry_pwr / 8.;
  const double pom_8 = entry_pwr / 16.;
  cplx_set_e2pimx((*omg)[0], pom_8);
  cplx_set_e2pimx((*omg)[1], pom_8 + j_pow);
  cplx_set_e2pimx((*omg)[2], pom_8 + k_pow);
  cplx_set_e2pimx((*omg)[3], pom_8 + j_pow + k_pow);
  cplx_set_e2pimx((*omg)[4], pom_4);
  cplx_set_e2pimx((*omg)[5], pom_4 + j_pow);
  cplx_set_e2pimx((*omg)[6], pom_2);
  cplx_set_e2pimx((*omg)[7], pom);
  *omg += 8;
}

/**
 * @brief iFFT modulo X^16-omega^2 (in registers)
 * @param data contains 16 complexes
 * @param omegabar 8 complexes in this order:
 *     gammabar,jb.gammabar,kb.gammabar,kbjb.gammabar,
 *     betabar,jb.betabar,alphabar,omegabar
 *     alpha = sqrt(omega), beta = sqrt(alpha), gamma = sqrt(beta)
 *     jb = sqrt(ib), kb=sqrt(jb)
 */
void cplx_ifft16_ref(void* data, const void* omegabar) {
  CPLX* d = data;
  const CPLX* om = omegabar;
  // fourth pass inverse
  invctwiddle(d[0], d[1], om[0]);
  invcitwiddle(d[2], d[3], om[0]);
  invctwiddle(d[4], d[5], om[1]);
  invcitwiddle(d[6], d[7], om[1]);
  invctwiddle(d[8], d[9], om[2]);
  invcitwiddle(d[10], d[11], om[2]);
  invctwiddle(d[12], d[13], om[3]);
  invcitwiddle(d[14], d[15], om[3]);
  // third pass inverse
  invctwiddle(d[0], d[2], om[4]);
  invctwiddle(d[1], d[3], om[4]);
  invcitwiddle(d[4], d[6], om[4]);
  invcitwiddle(d[5], d[7], om[4]);
  invctwiddle(d[8], d[10], om[5]);
  invctwiddle(d[9], d[11], om[5]);
  invcitwiddle(d[12], d[14], om[5]);
  invcitwiddle(d[13], d[15], om[5]);
  // second pass inverse
  invctwiddle(d[0], d[4], om[6]);
  invctwiddle(d[1], d[5], om[6]);
  invctwiddle(d[2], d[6], om[6]);
  invctwiddle(d[3], d[7], om[6]);
  invcitwiddle(d[8], d[12], om[6]);
  invcitwiddle(d[9], d[13], om[6]);
  invcitwiddle(d[10], d[14], om[6]);
  invcitwiddle(d[11], d[15], om[6]);
  // first pass
  for (uint64_t i = 0; i < 8; ++i) {
    invctwiddle(d[0 + i], d[8 + i], om[7]);
  }
}

void cplx_ifft_ref_bfs_2(CPLX* dat, const CPLX** omg, uint32_t m) {
  CPLX* const dend = dat + m;
  for (CPLX* d = dat; d < dend; d += 2) {
    split_fft_last_ref(d, (*omg)[0]);
    *omg += 1;
  }
#if 0
  printf("after first: ");
  for (uint64_t ii=0; ii<nn/2; ++ii) {
    printf("%.6lf %.6lf ",data[ii][0],data[ii][1]);
  }
  printf("\n");
#endif
  int32_t Ms2 = m / 2;
  for (int32_t h = 2; h <= Ms2; h <<= 1) {
    for (CPLX* d = dat; d < dend; d += 2 * h) {
      if (memcmp((*omg)[0], (*omg)[1], 8) != 0) abort();
      cplx_split_fft_ref(h, d, **omg);
      *omg += 2;
    }
#if 0
    printf("after split %d: ", h);
    for (uint64_t ii=0; ii<nn/2; ++ii) {
      printf("%.6lf %.6lf ",data[ii][0],data[ii][1]);
    }
    printf("\n");
#endif
  }
}

void cplx_ifft_ref_bfs_16(CPLX* dat, const CPLX** omg, uint32_t m) {
  const uint64_t log2m = log2(m);
  CPLX* const dend = dat + m;
  // h=1,2,4,8 use the 16-dim macroblock
  for (CPLX* d = dat; d < dend; d += 16) {
    cplx_ifft16_ref(d, *omg);
    *omg += 8;
  }
#if 0
  printf("after first: ");
  for (uint64_t ii=0; ii<nn/2; ++ii) {
    printf("%.6lf %.6lf ",data[ii][0],data[ii][1]);
  }
  printf("\n");
#endif
  int32_t h = 16;
  if (log2m % 2 != 0) {
    // if parity needs it, uses one regular twiddle
    for (CPLX* d = dat; d < dend; d += 2 * h) {
      cplx_split_fft_ref(h, d, **omg);
      *omg += 1;
    }
    h = 32;
  }
  // h=16,...,2*floor(Ms2/2) use the bitwiddle
  for (; h < m; h <<= 2) {
    for (CPLX* d = dat; d < dend; d += 4 * h) {
      cplx_bisplit_fft_ref(h, d, *omg);
      *omg += 2;
    }
#if 0
    printf("after split %d: ", h);
    for (uint64_t ii=0; ii<nn/2; ++ii) {
      printf("%.6lf %.6lf ",data[ii][0],data[ii][1]);
    }
    printf("\n");
#endif
  }
}

void fill_cplx_ifft_omegas_bfs_16(const double entry_pwr, CPLX** omg, uint32_t m) {
  const uint64_t log2m = log2(m);
  double pwr = entry_pwr * 16. / m;
  {
    // h=8
    for (uint32_t i = 0; i < m / 16; i++) {
      cplx_ifft16_precomp(pwr + fracrevbits(i), omg);
    }
  }
  int32_t h = 16;
  if (log2m % 2 != 0) {
    // if parity needs it, uses one regular twiddle
    for (uint32_t i = 0; i < m / (2 * h); i++) {
      cplx_set_e2pimx(omg[0][0], pwr + fracrevbits(i) / 2.);
      *omg += 1;
    }
    pwr *= 2.;
    h = 32;
  }
  for (; h < m; h <<= 2) {
    for (uint32_t i = 0; i < m / (2 * h); i+=2) {
      cplx_set_e2pimx(omg[0][0], pwr + fracrevbits(i) / 2.);
      cplx_set_e2pimx(omg[0][1], 2.*pwr + fracrevbits(i));
      *omg += 2;
    }
    pwr *= 4.;
  }
}

void fill_cplx_ifft_omegas_bfs_2(const double entry_pwr, CPLX** omg, uint32_t m) {
  double pom = entry_pwr / m;
  {
    // h=1
    for (uint32_t i = 0; i < m / 2; i++) {
      cplx_set_e2pimx((*omg)[0], pom + fracrevbits(i) / 2.);
      *omg += 1;  // optim function reads by 1
    }
  }
  for (int32_t h = 2; h <= m / 2; h <<= 1) {
    pom *= 2;
    for (uint32_t i = 0; i < m / (2 * h); i++) {
      cplx_set_e2pimx(omg[0][0], pom + fracrevbits(i) / 2.);
      cplx_set(omg[0][1], omg[0][0]);
      *omg += 2;
    }
  }
}

void fill_cplx_ifft_omegas_rec_16(const double entry_pwr, CPLX** omg, uint32_t m) {
  if (m == 1) return;
  if (m <= 8) return fill_cplx_ifft_omegas_bfs_2(entry_pwr, omg, m);
  if (m <= 2048) return fill_cplx_ifft_omegas_bfs_16(entry_pwr, omg, m);
  double pom = entry_pwr / 2.;
  uint32_t h = m / 2;
  fill_cplx_ifft_omegas_rec_16(pom, omg, h);
  fill_cplx_ifft_omegas_rec_16(pom + 0.5, omg, h);
  cplx_set_e2pimx((*omg)[0], pom);
  cplx_set((*omg)[1], (*omg)[0]);
  *omg += 2;
}

void cplx_ifft_ref_rec_16(CPLX* dat, const CPLX** omg, uint32_t m) {
  if (m == 1) return;
  if (m <= 8) return cplx_ifft_ref_bfs_2(dat, omg, m);
  if (m <= 2048) return cplx_ifft_ref_bfs_16(dat, omg, m);
  const uint32_t h = m / 2;
  cplx_ifft_ref_rec_16(dat, omg, h);
  cplx_ifft_ref_rec_16(dat + h, omg, h);
  if (memcmp((*omg)[0], (*omg)[1], 8) != 0) abort();
  cplx_split_fft_ref(h, dat, **omg);
  *omg += 2;
}

EXPORT void cplx_ifft_ref(const CPLX_IFFT_PRECOMP* precomp, void* d) {
  CPLX* data = (CPLX*)d;
  const int32_t m = precomp->m;
  const CPLX* omg = (CPLX*)precomp->powomegas;
  if (m == 1) return;
  if (m <= 8) return cplx_ifft_ref_bfs_2(data, &omg, m);
  if (m <= 2048) return cplx_ifft_ref_bfs_16(data, &omg, m);
  cplx_ifft_ref_rec_16(data, &omg, m);
}

EXPORT CPLX_IFFT_PRECOMP* new_cplx_ifft_precomp(uint32_t m, uint32_t num_buffers) {
  const uint64_t OMG_SPACE = ceilto64b(2 * m * sizeof(CPLX));
  const uint64_t BUF_SIZE = ceilto64b(m * sizeof(CPLX));
  void* reps = malloc(sizeof(CPLX_IFFT_PRECOMP) + 63    // padding
                      + OMG_SPACE                       // tables
                      + num_buffers * BUF_SIZE  // buffers
  );
  uint64_t aligned_addr = ceilto64b((uint64_t) reps + sizeof(CPLX_IFFT_PRECOMP));
  CPLX_IFFT_PRECOMP* r = (CPLX_IFFT_PRECOMP*)reps;
  r->m = m;
  r->buf_size = BUF_SIZE;
  r->powomegas = (double*)aligned_addr;
  r->aligned_buffers = (void*)(aligned_addr + OMG_SPACE);
  // fill in powomegas
  CPLX* omg = (CPLX*)r->powomegas;
  fill_cplx_ifft_omegas_rec_16(0.25, &omg, m);
  if (((uint64_t)omg) - aligned_addr > OMG_SPACE) abort();
  {
    if (m <= 4) {
      // currently, we do not have any acceletated
      // implementation for m<=4
      r->function = cplx_ifft_ref;
    } else if (CPU_SUPPORTS("fma")) {
      r->function = cplx_ifft_avx2_fma;
    } else {
      r->function = cplx_ifft_ref;
    }
  }
  return reps;
}

EXPORT void* cplx_ifft_precomp_get_buffer(const CPLX_IFFT_PRECOMP* itables, uint32_t buffer_index) {
  return (uint8_t*) itables->aligned_buffers + buffer_index * itables->buf_size;
}

EXPORT void cplx_ifft(const CPLX_IFFT_PRECOMP* itables, void* data) {
  itables->function(itables, data);
}

EXPORT void cplx_ifft_simple(uint32_t m, void* data) {
  static CPLX_IFFT_PRECOMP* p[31] = {0};
  CPLX_IFFT_PRECOMP** f = p + log2m(m);
  if (!*f) *f = new_cplx_ifft_precomp(m, 0);
  (*f)->function(*f, data);
}

