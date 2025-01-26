#include "../commons_private.h"
#include "reim_fft_internal.h"
#include "reim_fft_private.h"

void reim_invctwiddle(double* ra, double* ia, double* rb, double* ib, double omre, double omim) {
  double rdiff = *ra - *rb;
  double idiff = *ia - *ib;
  *ra = *ra + *rb;
  *ia = *ia + *ib;
  *rb = rdiff * omre - idiff * omim;
  *ib = rdiff * omim + idiff * omre;
}

// i (omre + i omim) = -omim + i omre
void reim_invcitwiddle(double* ra, double* ia, double* rb, double* ib, double omre, double omim) {
  double rdiff = *ra - *rb;
  double idiff = *ia - *ib;
  *ra = *ra + *rb;
  *ia = *ia + *ib;
  *rb = rdiff * omim + idiff * omre;
  *ib = - rdiff * omre + idiff * omim;
}

void reim_ifft16_ref(double* dre, double* dim, const void* pom) {
  const double* om = (const double*)pom;
  {
    double omare = om[0];
    double ombre = om[1];
    double omcre = om[2];
    double omdre = om[3];
    double omaim = om[4];
    double ombim = om[5];
    double omcim = om[6];
    double omdim = om[7];
    reim_invctwiddle(&dre[0], &dim[0], &dre[1], &dim[1], omare, omaim);
    reim_invcitwiddle(&dre[2], &dim[2], &dre[3], &dim[3], omare, omaim);
    reim_invctwiddle(&dre[4], &dim[4], &dre[5], &dim[5], ombre, ombim);
    reim_invcitwiddle(&dre[6], &dim[6], &dre[7], &dim[7], ombre, ombim);
    reim_invctwiddle(&dre[8], &dim[8], &dre[9], &dim[9], omcre, omcim);
    reim_invcitwiddle(&dre[10], &dim[10], &dre[11], &dim[11], omcre, omcim);
    reim_invctwiddle(&dre[12], &dim[12], &dre[13], &dim[13], omdre, omdim);
    reim_invcitwiddle(&dre[14], &dim[14], &dre[15], &dim[15], omdre, omdim);
  }
  {
    double omare = om[8];
    double omaim = om[9];
    double ombre = om[10];
    double ombim = om[11];
    reim_invctwiddle(&dre[0], &dim[0], &dre[2], &dim[2], omare, omaim);
    reim_invctwiddle(&dre[1], &dim[1], &dre[3], &dim[3], omare, omaim);
    reim_invcitwiddle(&dre[4], &dim[4], &dre[6], &dim[6], omare, omaim);
    reim_invcitwiddle(&dre[5], &dim[5], &dre[7], &dim[7], omare, omaim);
    reim_invctwiddle(&dre[8], &dim[8], &dre[10], &dim[10], ombre, ombim);
    reim_invctwiddle(&dre[9], &dim[9], &dre[11], &dim[11], ombre, ombim);
    reim_invcitwiddle(&dre[12], &dim[12], &dre[14], &dim[14], ombre, ombim);
    reim_invcitwiddle(&dre[13], &dim[13], &dre[15], &dim[15], ombre, ombim);
  }
  {
    double omre = om[12];
    double omim = om[13];
    reim_invctwiddle(&dre[0], &dim[0], &dre[4], &dim[4], omre, omim);
    reim_invctwiddle(&dre[1], &dim[1], &dre[5], &dim[5], omre, omim);
    reim_invctwiddle(&dre[2], &dim[2], &dre[6], &dim[6], omre, omim);
    reim_invctwiddle(&dre[3], &dim[3], &dre[7], &dim[7], omre, omim);
    reim_invcitwiddle(&dre[8], &dim[8], &dre[12], &dim[12], omre, omim);
    reim_invcitwiddle(&dre[9], &dim[9], &dre[13], &dim[13], omre, omim);
    reim_invcitwiddle(&dre[10], &dim[10], &dre[14], &dim[14], omre, omim);
    reim_invcitwiddle(&dre[11], &dim[11], &dre[15], &dim[15], omre, omim);
  }
  {
    double omre = om[14];
    double omim = om[15];
    reim_invctwiddle(&dre[0], &dim[0], &dre[8], &dim[8], omre, omim);
    reim_invctwiddle(&dre[1], &dim[1], &dre[9], &dim[9], omre, omim);
    reim_invctwiddle(&dre[2], &dim[2], &dre[10], &dim[10], omre, omim);
    reim_invctwiddle(&dre[3], &dim[3], &dre[11], &dim[11], omre, omim);
    reim_invctwiddle(&dre[4], &dim[4], &dre[12], &dim[12], omre, omim);
    reim_invctwiddle(&dre[5], &dim[5], &dre[13], &dim[13], omre, omim);
    reim_invctwiddle(&dre[6], &dim[6], &dre[14], &dim[14], omre, omim);
    reim_invctwiddle(&dre[7], &dim[7], &dre[15], &dim[15], omre, omim);
  }
}

void fill_reim_ifft16_omegas(const double entry_pwr, double** omg) {
  const double j_pow = 1. / 8.;
  const double k_pow = 1. / 16.;
  const double pin = entry_pwr / 2.;
  const double pin_2 = entry_pwr / 4.;
  const double pin_4 = entry_pwr / 8.;
  const double pin_8 = entry_pwr / 16.;
  // ((8,9,10,11),(12,13,14,15)) are 4 reals then 4 imag of om^1/8*(1,k,j,kj)
  (*omg)[0] = cos(2. * M_PI * (pin_8));
  (*omg)[1] = cos(2. * M_PI * (pin_8 + j_pow));
  (*omg)[2] = cos(2. * M_PI * (pin_8 + k_pow));
  (*omg)[3] = cos(2. * M_PI * (pin_8 + j_pow + k_pow));
  (*omg)[4] = - sin(2. * M_PI * (pin_8));
  (*omg)[5] = - sin(2. * M_PI * (pin_8 + j_pow));
  (*omg)[6] = - sin(2. * M_PI * (pin_8 + k_pow));
  (*omg)[7] = - sin(2. * M_PI * (pin_8 + j_pow + k_pow));
  // (4,5) and (6,7) are real and imag of om^1/4 and j.om^1/4
  (*omg)[8] = cos(2. * M_PI * (pin_4));
  (*omg)[9] = - sin(2. * M_PI * (pin_4));
  (*omg)[10] = cos(2. * M_PI * (pin_4 + j_pow));
  (*omg)[11] = - sin(2. * M_PI * (pin_4 + j_pow));
  // 2 and 3 are real and imag of om^1/2
  (*omg)[12] = cos(2. * M_PI * (pin_2));
  (*omg)[13] = - sin(2. * M_PI * (pin_2));
  // 0 and 1 are real and imag of om
  (*omg)[14] = cos(2. * M_PI * pin);
  (*omg)[15] = - sin(2. * M_PI * pin);
  *omg += 16;
}

void reim_ifft8_ref(double* dre, double* dim, const void* pom) {
  const double* om = (const double*)pom;
  {
    double omare = om[0];
    double ombre = om[1];
    double omaim = om[2];
    double ombim = om[3];
    reim_invctwiddle(&dre[0], &dim[0], &dre[1], &dim[1], omare, omaim);
    reim_invcitwiddle(&dre[2], &dim[2], &dre[3], &dim[3], omare, omaim);
    reim_invctwiddle(&dre[4], &dim[4], &dre[5], &dim[5], ombre, ombim);
    reim_invcitwiddle(&dre[6], &dim[6], &dre[7], &dim[7], ombre, ombim);
  }
  {
    double omare = om[4];
    double omaim = om[5];
    reim_invctwiddle(&dre[0], &dim[0], &dre[2], &dim[2], omare, omaim);
    reim_invctwiddle(&dre[1], &dim[1], &dre[3], &dim[3], omare, omaim);
    reim_invcitwiddle(&dre[4], &dim[4], &dre[6], &dim[6], omare, omaim);
    reim_invcitwiddle(&dre[5], &dim[5], &dre[7], &dim[7], omare, omaim);
  }
  {
    double omre = om[6];
    double omim = om[7];
    reim_invctwiddle(&dre[0], &dim[0], &dre[4], &dim[4], omre, omim);
    reim_invctwiddle(&dre[1], &dim[1], &dre[5], &dim[5], omre, omim);
    reim_invctwiddle(&dre[2], &dim[2], &dre[6], &dim[6], omre, omim);
    reim_invctwiddle(&dre[3], &dim[3], &dre[7], &dim[7], omre, omim);
  }
}

void fill_reim_ifft8_omegas(const double entry_pwr, double** omg) {
  const double j_pow = 1. / 8.;
  const double pin = entry_pwr / 2.;
  const double pin_2 = entry_pwr / 4.;
  const double pin_4 = entry_pwr / 8.;
  // (4,5) and (6,7) are real and imag of om^1/4 and j.om^1/4
  (*omg)[0] = cos(2. * M_PI * (pin_4));
  (*omg)[1] = cos(2. * M_PI * (pin_4 + j_pow));
  (*omg)[2] = - sin(2. * M_PI * (pin_4));
  (*omg)[3] = - sin(2. * M_PI * (pin_4 + j_pow));
  // 2 and 3 are real and imag of om^1/2
  (*omg)[4] = cos(2. * M_PI * (pin_2));
  (*omg)[5] = - sin(2. * M_PI * (pin_2));
  // 0 and 1 are real and imag of om
  (*omg)[6] = cos(2. * M_PI * pin);
  (*omg)[7] = - sin(2. * M_PI * pin);
  *omg += 8;
}

void reim_ifft4_ref(double* dre, double* dim, const void* pom) {
  const double* om = (const double*)pom;
  {
    double omare = om[0];
    double omaim = om[1];
    reim_invctwiddle(&dre[0], &dim[0], &dre[1], &dim[1], omare, omaim);
    reim_invcitwiddle(&dre[2], &dim[2], &dre[3], &dim[3], omare, omaim);
  }
  {
    double omare = om[2];
    double omaim = om[3];
    reim_invctwiddle(&dre[0], &dim[0], &dre[2], &dim[2], omare, omaim);
    reim_invctwiddle(&dre[1], &dim[1], &dre[3], &dim[3], omare, omaim);
  }
}

void fill_reim_ifft4_omegas(const double entry_pwr, double** omg) {
  const double pin = entry_pwr / 2.;
  const double pin_2 = entry_pwr / 4.;
  // 2 and 3 are real and imag of om^1/2
  (*omg)[0] = cos(2. * M_PI * (pin_2));
  (*omg)[1] = - sin(2. * M_PI * (pin_2));
  // 0 and 1 are real and imag of om
  (*omg)[2] = cos(2. * M_PI * pin);
  (*omg)[3] = - sin(2. * M_PI * pin);
  *omg += 4;
}

void reim_ifft2_ref(double* dre, double* dim, const void* pom) {
  const double* om = (const double*)pom;
  {
    double omare = om[0];
    double omaim = om[1];
    reim_invctwiddle(&dre[0], &dim[0], &dre[1], &dim[1], omare, omaim);
  }
}

void fill_reim_ifft2_omegas(const double entry_pwr, double** omg) {
  const double pin = entry_pwr / 2.;
  // 0 and 1 are real and imag of om
  (*omg)[0] = cos(2. * M_PI * pin);
  (*omg)[1] = - sin(2. * M_PI * pin);
  *omg += 2;
}

void reim_invtwiddle_ifft_ref(uint64_t h, double* re, double* im, double om[2]) {
  for (uint64_t i=0; i<h; ++i) {
    reim_invctwiddle(&re[i],&im[i],&re[h+i],&im[h+i], om[0], om[1]);
  }
}

void reim_invbitwiddle_ifft_ref(uint64_t h, double* re, double* im, double om[4]) {
  double* r0 = re;
  double* r1 = re + h;
  double* r2 = re + 2*h;
  double* r3 = re + 3*h;
  double* i0 = im;
  double* i1 = im + h;
  double* i2 = im + 2*h;
  double* i3 = im + 3*h;
  for (uint64_t i=0; i<h; ++i) {
    reim_invctwiddle(&r0[i],&i0[i],&r1[i],&i1[i], om[0], om[1]);
    reim_invcitwiddle(&r2[i],&i2[i],&r3[i],&i3[i], om[0], om[1]);
  }
  for (uint64_t i=0; i<h; ++i) {
    reim_invctwiddle(&r0[i],&i0[i],&r2[i],&i2[i], om[2], om[3]);
    reim_invctwiddle(&r1[i],&i1[i],&r3[i],&i3[i], om[2], om[3]);
  }
}

void reim_ifft_bfs_16_ref(uint64_t m, double* re, double* im, double** omg) {
  uint64_t log2m = log2(m);
  for (uint64_t off = 0; off < m; off += 16) {
    reim_ifft16_ref(re+off, im+off, *omg);
    *omg += 16;
  }
  uint64_t h = 16;
  uint64_t ms2 = m/2;
  while (h < ms2) {
    uint64_t mm = h << 2;
    for (uint64_t off = 0; off < m; off += mm) {
      reim_invbitwiddle_ifft_ref(h, re + off, im + off, *omg);
      *omg += 4;
    }
    h = mm;
  }
  if (log2m % 2 != 0) {
    if (h!=ms2) abort(); // bug
    // do the first twiddle iteration normally
    reim_invtwiddle_ifft_ref(h, re, im, *omg);
    *omg += 2;
    h = m;
  }
}

void fill_reim_ifft_bfs_16_omegas(uint64_t m, double entry_pwr, double** omg) {
  uint64_t log2m = log2(m);
  //uint64_t mm = 16;
  double ss = entry_pwr * 16. / m;
  for (uint64_t off = 0; off < m; off += 16) {
    double s = ss + fracrevbits(off/16);
    fill_reim_ifft16_omegas(s, omg);
  }
  uint64_t h = 16;
  uint64_t ms2 = m/2;
  while (h < ms2) {
    uint64_t mm = h << 2;
    for (uint64_t off = 0; off < m; off += mm) {
      double rs0 = ss + fracrevbits(off / mm) / 4.;
      double rs1 = 2. * rs0;
      (*omg)[0] = cos(2 * M_PI * rs0);
      (*omg)[1] = -sin(2 * M_PI * rs0);
      (*omg)[2] = cos(2 * M_PI * rs1);
      (*omg)[3] = -sin(2 * M_PI * rs1);
      *omg += 4;
    }
    ss *= 4.;
    h = mm;
  }
  if (log2m % 2 != 0) {
    if (h!=ms2) abort(); // bug
    // do the first twiddle iteration normally
    (*omg)[0] = cos(2 * M_PI * ss);
    (*omg)[1] = - sin(2 * M_PI * ss);
    *omg += 2;
    h = m;
    ss *= 2.;
  }
  if (ss!=entry_pwr) abort();
}

void reim_ifft_rec_16_ref(uint64_t m, double* re, double* im, double** omg) {
  if (m <= 2048) return reim_ifft_bfs_16_ref(m, re, im, omg);
  const uint32_t h = m / 2;
  reim_ifft_rec_16_ref(h, re, im, omg);
  reim_ifft_rec_16_ref(h, re + h, im + h, omg);
  reim_invtwiddle_ifft_ref(h, re, im, *omg);
  *omg += 2;
}

void fill_reim_ifft_rec_16_omegas(uint64_t m, double entry_pwr, double** omg) {
  if (m <= 2048) return fill_reim_ifft_bfs_16_omegas(m, entry_pwr, omg);
  const uint64_t h = m / 2;
  const double s = entry_pwr / 2;
  fill_reim_ifft_rec_16_omegas(h, s, omg);
  fill_reim_ifft_rec_16_omegas(h, s + 0.5, omg);
  (*omg)[0] = cos(2 * M_PI * s);
  (*omg)[1] = - sin(2 * M_PI * s);
  *omg += 2;
}

void reim_ifft_ref(const REIM_IFFT_PRECOMP* precomp, double* dat) {
  const int32_t m = precomp->m;
  double* omg = precomp->powomegas;
  double* re = dat;
  double* im = dat+m;
  if (m <= 16) {
    switch (m) {
      case 1:
        return;
      case 2:
        return reim_ifft2_ref(re, im, omg);
      case 4:
        return reim_ifft4_ref(re, im, omg);
      case 8:
        return reim_ifft8_ref(re, im, omg);
      case 16:
        return reim_ifft16_ref(re, im, omg);
      default:
        abort(); // m is not a power of 2
    }
  }
  if (m <= 2048) return reim_ifft_bfs_16_ref(m, re, im, &omg);
  return reim_ifft_rec_16_ref(m, re, im, &omg);
}

EXPORT REIM_IFFT_PRECOMP* new_reim_ifft_precomp(uint32_t m, uint32_t num_buffers) {
  const uint64_t OMG_SPACE = ceilto64b(2 * m * sizeof(double));
  const uint64_t BUF_SIZE = ceilto64b(2 * m * sizeof(double));
  void* reps = malloc(sizeof(REIM_IFFT_PRECOMP)          // base
                      + 63                              // padding
                      + OMG_SPACE                       // tables //TODO 16?
                      + num_buffers * BUF_SIZE  // buffers
  );
  uint64_t aligned_addr = ceilto64b((uint64_t)(reps) + sizeof(REIM_IFFT_PRECOMP));
  REIM_IFFT_PRECOMP* r = (REIM_IFFT_PRECOMP*)reps;
  r->m = m;
  r->buf_size = BUF_SIZE;
  r->powomegas = (double*)aligned_addr;
  r->aligned_buffers = (void*)(aligned_addr + OMG_SPACE);
  // fill in powomegas
  double* omg = (double*) r->powomegas;
  if (m <= 16) {
    switch (m) {
      case 1:
        break;
      case 2:
        fill_reim_ifft2_omegas(0.25, &omg);
        break;
      case 4:
        fill_reim_ifft4_omegas(0.25, &omg);
        break;
      case 8:
        fill_reim_ifft8_omegas(0.25, &omg);
        break;
      case 16:
        fill_reim_ifft16_omegas(0.25, &omg);
        break;
      default:
        abort();  // m is not a power of 2
    }
  } else if (m <= 2048) {
    fill_reim_ifft_bfs_16_omegas(m, 0.25, &omg);
  } else {
    fill_reim_ifft_rec_16_omegas(m, 0.25, &omg);
  }
  if (((uint64_t)omg) - aligned_addr > OMG_SPACE) abort();
  // dispatch the right implementation
  {
    if (CPU_SUPPORTS("fma")) {
      r->function = reim_ifft_avx2_fma;
    } else {
      r->function = reim_ifft_ref;
    }
  }
  return reps;
}


void reim_naive_ifft(uint64_t m, double entry_pwr, double* re, double* im) {
  if (m == 1) return;
  // twiddle
  const uint64_t h = m / 2;
  const double s = entry_pwr / 2.;
  reim_naive_ifft(h, s, re, im);
  reim_naive_ifft(h, s + 0.5, re + h, im + h);
  const double sre = cos(2 * M_PI * s);
  const double sim = - sin(2 * M_PI * s);
  for (uint64_t j = 0; j < h; ++j) {
    double rdiff = re[j] - re[h+j];
    double idiff = im[j] - im[h+j];
    re[j] = re[j] + re[h+j];
    im[j] = im[j] + im[h+j];
    re[h+j] = rdiff * sre - idiff * sim;
    im[h + j] = idiff * sre + rdiff * sim;
  }
}
