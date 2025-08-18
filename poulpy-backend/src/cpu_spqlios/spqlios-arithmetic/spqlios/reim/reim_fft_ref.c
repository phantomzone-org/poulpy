#include "../commons_private.h"
#include "reim_fft_internal.h"
#include "reim_fft_private.h"

EXPORT void reim_fft_simple(uint32_t m, void* data) {
  static REIM_FFT_PRECOMP* p[31] = {0};
  REIM_FFT_PRECOMP** f = p + log2m(m);
  if (!*f) *f = new_reim_fft_precomp(m, 0);
  (*f)->function(*f, data);
}

EXPORT void reim_ifft_simple(uint32_t m, void* data) {
  static REIM_IFFT_PRECOMP* p[31] = {0};
  REIM_IFFT_PRECOMP** f = p + log2m(m);
  if (!*f) *f = new_reim_ifft_precomp(m, 0);
  (*f)->function(*f, data);
}

EXPORT void reim_fftvec_add_simple(uint32_t m, void* r, const void* a, const void* b) {
  static REIM_FFTVEC_ADD_PRECOMP* p[31] = {0};
  REIM_FFTVEC_ADD_PRECOMP** f = p + log2m(m);
  if (!*f) *f = new_reim_fftvec_add_precomp(m);
  (*f)->function(*f, r, a, b);
}

EXPORT void reim_fftvec_sub_simple(uint32_t m, void* r, const void* a, const void* b) {
  static REIM_FFTVEC_SUB_PRECOMP* p[31] = {0};
  REIM_FFTVEC_SUB_PRECOMP** f = p + log2m(m);
  if (!*f) *f = new_reim_fftvec_sub_precomp(m);
  (*f)->function(*f, r, a, b);
}

EXPORT void reim_fftvec_mul_simple(uint32_t m, void* r, const void* a, const void* b) {
  static REIM_FFTVEC_MUL_PRECOMP* p[31] = {0};
  REIM_FFTVEC_MUL_PRECOMP** f = p + log2m(m);
  if (!*f) *f = new_reim_fftvec_mul_precomp(m);
  (*f)->function(*f, r, a, b);
}

EXPORT void reim_fftvec_addmul_simple(uint32_t m, void* r, const void* a, const void* b) {
  static REIM_FFTVEC_ADDMUL_PRECOMP* p[31] = {0};
  REIM_FFTVEC_ADDMUL_PRECOMP** f = p + log2m(m);
  if (!*f) *f = new_reim_fftvec_addmul_precomp(m);
  (*f)->function(*f, r, a, b);
}

void reim_ctwiddle(double* ra, double* ia, double* rb, double* ib, double omre, double omim) {
  double newrt = *rb * omre - *ib * omim;
  double newit = *rb * omim + *ib * omre;
  *rb = *ra - newrt;
  *ib = *ia - newit;
  *ra = *ra + newrt;
  *ia = *ia + newit;
}

// i (omre + i omim) = -omim + i omre
void reim_citwiddle(double* ra, double* ia, double* rb, double* ib, double omre, double omim) {
  double newrt = -*rb * omim - *ib * omre;
  double newit = *rb * omre - *ib * omim;
  *rb = *ra - newrt;
  *ib = *ia - newit;
  *ra = *ra + newrt;
  *ia = *ia + newit;
}

void reim_fft16_ref(double* dre, double* dim, const void* pom) {
  const double* om = (const double*)pom;
  {
    double omre = om[0];
    double omim = om[1];
    reim_ctwiddle(&dre[0], &dim[0], &dre[8], &dim[8], omre, omim);
    reim_ctwiddle(&dre[1], &dim[1], &dre[9], &dim[9], omre, omim);
    reim_ctwiddle(&dre[2], &dim[2], &dre[10], &dim[10], omre, omim);
    reim_ctwiddle(&dre[3], &dim[3], &dre[11], &dim[11], omre, omim);
    reim_ctwiddle(&dre[4], &dim[4], &dre[12], &dim[12], omre, omim);
    reim_ctwiddle(&dre[5], &dim[5], &dre[13], &dim[13], omre, omim);
    reim_ctwiddle(&dre[6], &dim[6], &dre[14], &dim[14], omre, omim);
    reim_ctwiddle(&dre[7], &dim[7], &dre[15], &dim[15], omre, omim);
  }
  {
    double omre = om[2];
    double omim = om[3];
    reim_ctwiddle(&dre[0], &dim[0], &dre[4], &dim[4], omre, omim);
    reim_ctwiddle(&dre[1], &dim[1], &dre[5], &dim[5], omre, omim);
    reim_ctwiddle(&dre[2], &dim[2], &dre[6], &dim[6], omre, omim);
    reim_ctwiddle(&dre[3], &dim[3], &dre[7], &dim[7], omre, omim);
    reim_citwiddle(&dre[8], &dim[8], &dre[12], &dim[12], omre, omim);
    reim_citwiddle(&dre[9], &dim[9], &dre[13], &dim[13], omre, omim);
    reim_citwiddle(&dre[10], &dim[10], &dre[14], &dim[14], omre, omim);
    reim_citwiddle(&dre[11], &dim[11], &dre[15], &dim[15], omre, omim);
  }
  {
    double omare = om[4];
    double omaim = om[5];
    double ombre = om[6];
    double ombim = om[7];
    reim_ctwiddle(&dre[0], &dim[0], &dre[2], &dim[2], omare, omaim);
    reim_ctwiddle(&dre[1], &dim[1], &dre[3], &dim[3], omare, omaim);
    reim_citwiddle(&dre[4], &dim[4], &dre[6], &dim[6], omare, omaim);
    reim_citwiddle(&dre[5], &dim[5], &dre[7], &dim[7], omare, omaim);
    reim_ctwiddle(&dre[8], &dim[8], &dre[10], &dim[10], ombre, ombim);
    reim_ctwiddle(&dre[9], &dim[9], &dre[11], &dim[11], ombre, ombim);
    reim_citwiddle(&dre[12], &dim[12], &dre[14], &dim[14], ombre, ombim);
    reim_citwiddle(&dre[13], &dim[13], &dre[15], &dim[15], ombre, ombim);
  }
  {
    double omare = om[8];
    double ombre = om[9];
    double omcre = om[10];
    double omdre = om[11];
    double omaim = om[12];
    double ombim = om[13];
    double omcim = om[14];
    double omdim = om[15];
    reim_ctwiddle(&dre[0], &dim[0], &dre[1], &dim[1], omare, omaim);
    reim_citwiddle(&dre[2], &dim[2], &dre[3], &dim[3], omare, omaim);
    reim_ctwiddle(&dre[4], &dim[4], &dre[5], &dim[5], ombre, ombim);
    reim_citwiddle(&dre[6], &dim[6], &dre[7], &dim[7], ombre, ombim);
    reim_ctwiddle(&dre[8], &dim[8], &dre[9], &dim[9], omcre, omcim);
    reim_citwiddle(&dre[10], &dim[10], &dre[11], &dim[11], omcre, omcim);
    reim_ctwiddle(&dre[12], &dim[12], &dre[13], &dim[13], omdre, omdim);
    reim_citwiddle(&dre[14], &dim[14], &dre[15], &dim[15], omdre, omdim);
  }
}

void fill_reim_fft16_omegas(const double entry_pwr, double** omg) {
  const double j_pow = 1. / 8.;
  const double k_pow = 1. / 16.;
  const double pin = entry_pwr / 2.;
  const double pin_2 = entry_pwr / 4.;
  const double pin_4 = entry_pwr / 8.;
  const double pin_8 = entry_pwr / 16.;
  // 0 and 1 are real and imag of om
  (*omg)[0] = cos(2. * M_PI * pin);
  (*omg)[1] = sin(2. * M_PI * pin);
  // 2 and 3 are real and imag of om^1/2
  (*omg)[2] = cos(2. * M_PI * (pin_2));
  (*omg)[3] = sin(2. * M_PI * (pin_2));
  // (4,5) and (6,7) are real and imag of om^1/4 and j.om^1/4
  (*omg)[4] = cos(2. * M_PI * (pin_4));
  (*omg)[5] = sin(2. * M_PI * (pin_4));
  (*omg)[6] = cos(2. * M_PI * (pin_4 + j_pow));
  (*omg)[7] = sin(2. * M_PI * (pin_4 + j_pow));
  // ((8,9,10,11),(12,13,14,15)) are 4 reals then 4 imag of om^1/8*(1,k,j,kj)
  (*omg)[8] = cos(2. * M_PI * (pin_8));
  (*omg)[9] = cos(2. * M_PI * (pin_8 + j_pow));
  (*omg)[10] = cos(2. * M_PI * (pin_8 + k_pow));
  (*omg)[11] = cos(2. * M_PI * (pin_8 + j_pow + k_pow));
  (*omg)[12] = sin(2. * M_PI * (pin_8));
  (*omg)[13] = sin(2. * M_PI * (pin_8 + j_pow));
  (*omg)[14] = sin(2. * M_PI * (pin_8 + k_pow));
  (*omg)[15] = sin(2. * M_PI * (pin_8 + j_pow + k_pow));
  *omg += 16;
}

void reim_fft8_ref(double* dre, double* dim, const void* pom) {
  const double* om = (const double*)pom;
  {
    double omre = om[0];
    double omim = om[1];
    reim_ctwiddle(&dre[0], &dim[0], &dre[4], &dim[4], omre, omim);
    reim_ctwiddle(&dre[1], &dim[1], &dre[5], &dim[5], omre, omim);
    reim_ctwiddle(&dre[2], &dim[2], &dre[6], &dim[6], omre, omim);
    reim_ctwiddle(&dre[3], &dim[3], &dre[7], &dim[7], omre, omim);
  }
  {
    double omare = om[2];
    double omaim = om[3];
    reim_ctwiddle(&dre[0], &dim[0], &dre[2], &dim[2], omare, omaim);
    reim_ctwiddle(&dre[1], &dim[1], &dre[3], &dim[3], omare, omaim);
    reim_citwiddle(&dre[4], &dim[4], &dre[6], &dim[6], omare, omaim);
    reim_citwiddle(&dre[5], &dim[5], &dre[7], &dim[7], omare, omaim);
  }
  {
    double omare = om[4];
    double ombre = om[5];
    double omaim = om[6];
    double ombim = om[7];
    reim_ctwiddle(&dre[0], &dim[0], &dre[1], &dim[1], omare, omaim);
    reim_citwiddle(&dre[2], &dim[2], &dre[3], &dim[3], omare, omaim);
    reim_ctwiddle(&dre[4], &dim[4], &dre[5], &dim[5], ombre, ombim);
    reim_citwiddle(&dre[6], &dim[6], &dre[7], &dim[7], ombre, ombim);
  }
}

void fill_reim_fft8_omegas(const double entry_pwr, double** omg) {
  const double j_pow = 1. / 8.;
  const double pin = entry_pwr / 2.;
  const double pin_2 = entry_pwr / 4.;
  const double pin_4 = entry_pwr / 8.;
  // 0 and 1 are real and imag of om
  (*omg)[0] = cos(2. * M_PI * pin);
  (*omg)[1] = sin(2. * M_PI * pin);
  // 2 and 3 are real and imag of om^1/2
  (*omg)[2] = cos(2. * M_PI * (pin_2));
  (*omg)[3] = sin(2. * M_PI * (pin_2));
  // (4,5) and (6,7) are real and imag of om^1/4 and j.om^1/4
  (*omg)[4] = cos(2. * M_PI * (pin_4));
  (*omg)[5] = cos(2. * M_PI * (pin_4 + j_pow));
  (*omg)[6] = sin(2. * M_PI * (pin_4));
  (*omg)[7] = sin(2. * M_PI * (pin_4 + j_pow));
  *omg += 8;
}

void reim_fft4_ref(double* dre, double* dim, const void* pom) {
  const double* om = (const double*)pom;
  {
    double omare = om[0];
    double omaim = om[1];
    reim_ctwiddle(&dre[0], &dim[0], &dre[2], &dim[2], omare, omaim);
    reim_ctwiddle(&dre[1], &dim[1], &dre[3], &dim[3], omare, omaim);
  }
  {
    double omare = om[2];
    double omaim = om[3];
    reim_ctwiddle(&dre[0], &dim[0], &dre[1], &dim[1], omare, omaim);
    reim_citwiddle(&dre[2], &dim[2], &dre[3], &dim[3], omare, omaim);
  }
}

void fill_reim_fft4_omegas(const double entry_pwr, double** omg) {
  const double pin = entry_pwr / 2.;
  const double pin_2 = entry_pwr / 4.;
  // 0 and 1 are real and imag of om
  (*omg)[0] = cos(2. * M_PI * pin);
  (*omg)[1] = sin(2. * M_PI * pin);
  // 2 and 3 are real and imag of om^1/2
  (*omg)[2] = cos(2. * M_PI * (pin_2));
  (*omg)[3] = sin(2. * M_PI * (pin_2));
  *omg += 4;
}

void reim_fft2_ref(double* dre, double* dim, const void* pom) {
  const double* om = (const double*)pom;
  {
    double omare = om[0];
    double omaim = om[1];
    reim_ctwiddle(&dre[0], &dim[0], &dre[1], &dim[1], omare, omaim);
  }
}

void fill_reim_fft2_omegas(const double entry_pwr, double** omg) {
  const double pin = entry_pwr / 2.;
  // 0 and 1 are real and imag of om
  (*omg)[0] = cos(2. * M_PI * pin);
  (*omg)[1] = sin(2. * M_PI * pin);
  *omg += 2;
}

void reim_twiddle_fft_ref(uint64_t h, double* re, double* im, double om[2]) {
  for (uint64_t i = 0; i < h; ++i) {
    reim_ctwiddle(&re[i], &im[i], &re[h + i], &im[h + i], om[0], om[1]);
  }
}

void reim_bitwiddle_fft_ref(uint64_t h, double* re, double* im, double om[4]) {
  double* r0 = re;
  double* r1 = re + h;
  double* r2 = re + 2 * h;
  double* r3 = re + 3 * h;
  double* i0 = im;
  double* i1 = im + h;
  double* i2 = im + 2 * h;
  double* i3 = im + 3 * h;
  for (uint64_t i = 0; i < h; ++i) {
    reim_ctwiddle(&r0[i], &i0[i], &r2[i], &i2[i], om[0], om[1]);
    reim_ctwiddle(&r1[i], &i1[i], &r3[i], &i3[i], om[0], om[1]);
  }
  for (uint64_t i = 0; i < h; ++i) {
    reim_ctwiddle(&r0[i], &i0[i], &r1[i], &i1[i], om[2], om[3]);
    reim_citwiddle(&r2[i], &i2[i], &r3[i], &i3[i], om[2], om[3]);
  }
}

void reim_fft_bfs_16_ref(uint64_t m, double* re, double* im, double** omg) {
  uint64_t log2m = log2(m);
  uint64_t mm = m;
  if (log2m % 2 != 0) {
    uint64_t h = mm >> 1;
    // do the first twiddle iteration normally
    reim_twiddle_fft_ref(h, re, im, *omg);
    *omg += 2;
    mm = h;
  }
  while (mm > 16) {
    uint64_t h = mm >> 2;
    for (uint64_t off = 0; off < m; off += mm) {
      reim_bitwiddle_fft_ref(h, re + off, im + off, *omg);
      *omg += 4;
    }
    mm = h;
  }
  if (mm != 16) abort();  // bug!
  for (uint64_t off = 0; off < m; off += 16) {
    reim_fft16_ref(re + off, im + off, *omg);
    *omg += 16;
  }
}

void fill_reim_fft_bfs_16_omegas(uint64_t m, double entry_pwr, double** omg) {
  uint64_t log2m = log2(m);
  uint64_t mm = m;
  double ss = entry_pwr;
  if (log2m % 2 != 0) {
    uint64_t h = mm >> 1;
    double s = ss / 2.;
    // do the first twiddle iteration normally
    (*omg)[0] = cos(2 * M_PI * s);
    (*omg)[1] = sin(2 * M_PI * s);
    *omg += 2;
    mm = h;
    ss = s;
  }
  while (mm > 16) {
    uint64_t h = mm >> 2;
    double s = ss / 4.;
    for (uint64_t off = 0; off < m; off += mm) {
      double rs0 = s + fracrevbits(off / mm) / 4.;
      double rs1 = 2. * rs0;
      (*omg)[0] = cos(2 * M_PI * rs1);
      (*omg)[1] = sin(2 * M_PI * rs1);
      (*omg)[2] = cos(2 * M_PI * rs0);
      (*omg)[3] = sin(2 * M_PI * rs0);
      *omg += 4;
    }
    mm = h;
    ss = s;
  }
  if (mm != 16) abort();  // bug!
  for (uint64_t off = 0; off < m; off += 16) {
    double s = ss + fracrevbits(off / 16);
    fill_reim_fft16_omegas(s, omg);
  }
}

void reim_fft_rec_16_ref(uint64_t m, double* re, double* im, double** omg) {
  if (m <= 2048) return reim_fft_bfs_16_ref(m, re, im, omg);
  const uint32_t h = m / 2;
  reim_twiddle_fft_ref(h, re, im, *omg);
  *omg += 2;
  reim_fft_rec_16_ref(h, re, im, omg);
  reim_fft_rec_16_ref(h, re + h, im + h, omg);
}

void fill_reim_fft_rec_16_omegas(uint64_t m, double entry_pwr, double** omg) {
  if (m <= 2048) return fill_reim_fft_bfs_16_omegas(m, entry_pwr, omg);
  const uint64_t h = m / 2;
  const double s = entry_pwr / 2;
  (*omg)[0] = cos(2 * M_PI * s);
  (*omg)[1] = sin(2 * M_PI * s);
  *omg += 2;
  fill_reim_fft_rec_16_omegas(h, s, omg);
  fill_reim_fft_rec_16_omegas(h, s + 0.5, omg);
}

void reim_fft_ref(const REIM_FFT_PRECOMP* precomp, double* dat) {
  const int32_t m = precomp->m;
  double* omg = precomp->powomegas;
  double* re = dat;
  double* im = dat + m;
  if (m <= 16) {
    switch (m) {
      case 1:
        return;
      case 2:
        return reim_fft2_ref(re, im, omg);
      case 4:
        return reim_fft4_ref(re, im, omg);
      case 8:
        return reim_fft8_ref(re, im, omg);
      case 16:
        return reim_fft16_ref(re, im, omg);
      default:
        abort();  // m is not a power of 2
    }
  }
  if (m <= 2048) return reim_fft_bfs_16_ref(m, re, im, &omg);
  return reim_fft_rec_16_ref(m, re, im, &omg);
}

EXPORT REIM_FFT_PRECOMP* new_reim_fft_precomp(uint32_t m, uint32_t num_buffers) {
  const uint64_t OMG_SPACE = ceilto64b(2 * m * sizeof(double));
  const uint64_t BUF_SIZE = ceilto64b(2 * m * sizeof(double));
  void* reps = malloc(sizeof(REIM_FFT_PRECOMP)  // base
                      + 63                      // padding
                      + OMG_SPACE               // tables //TODO 16?
                      + num_buffers * BUF_SIZE  // buffers
  );
  uint64_t aligned_addr = ceilto64b((uint64_t)(reps) + sizeof(REIM_FFT_PRECOMP));
  REIM_FFT_PRECOMP* r = (REIM_FFT_PRECOMP*)reps;
  r->m = m;
  r->buf_size = BUF_SIZE;
  r->powomegas = (double*)aligned_addr;
  r->aligned_buffers = (void*)(aligned_addr + OMG_SPACE);
  // fill in powomegas
  double* omg = (double*)r->powomegas;
  if (m <= 16) {
    switch (m) {
      case 1:
        break;
      case 2:
        fill_reim_fft2_omegas(0.25, &omg);
        break;
      case 4:
        fill_reim_fft4_omegas(0.25, &omg);
        break;
      case 8:
        fill_reim_fft8_omegas(0.25, &omg);
        break;
      case 16:
        fill_reim_fft16_omegas(0.25, &omg);
        break;
      default:
        abort();  // m is not a power of 2
    }
  } else if (m <= 2048) {
    fill_reim_fft_bfs_16_omegas(m, 0.25, &omg);
  } else {
    fill_reim_fft_rec_16_omegas(m, 0.25, &omg);
  }
  if (((uint64_t)omg) - aligned_addr > OMG_SPACE) abort();
  // dispatch the right implementation
  {
    if (CPU_SUPPORTS("fma")) {
      r->function = reim_fft_avx2_fma;
    } else {
      r->function = reim_fft_ref;
    }
  }
  return reps;
}

void reim_naive_fft(uint64_t m, double entry_pwr, double* re, double* im) {
  if (m == 1) return;
  // twiddle
  const uint64_t h = m / 2;
  const double s = entry_pwr / 2.;
  const double sre = cos(2 * M_PI * s);
  const double sim = sin(2 * M_PI * s);
  for (uint64_t j = 0; j < h; ++j) {
    double pre = re[h + j] * sre - im[h + j] * sim;
    double pim = im[h + j] * sre + re[h + j] * sim;
    re[h + j] = re[j] - pre;
    im[h + j] = im[j] - pim;
    re[j] += pre;
    im[j] += pim;
  }
  reim_naive_fft(h, s, re, im);
  reim_naive_fft(h, s + 0.5, re + h, im + h);
}
