#ifndef SPQLIOS_REIM_FFT_CORE_TEMPLATE_H
#define SPQLIOS_REIM_FFT_CORE_TEMPLATE_H

// this file contains the main template for the fft strategy.
// it is meant to be included once for each specialization (ref, avx, neon)
// all the leaf functions it uses shall be defined in the following macros,
// before including this header.

#if !defined(reim_fft16_f) || !defined(reim_fft16_pom_offset) || !defined(fill_reim_fft16_omegas_f)
#error "missing reim16 definitions"
#endif
#if !defined(reim_fft8_f) || !defined(reim_fft8_pom_offset) || !defined(fill_reim_fft8_omegas_f)
#error "missing reim8 definitions"
#endif
#if !defined(reim_fft4_f) || !defined(reim_fft4_pom_offset) || !defined(fill_reim_fft4_omegas_f)
#error "missing reim4 definitions"
#endif
#if !defined(reim_fft2_f) || !defined(reim_fft2_pom_offset) || !defined(fill_reim_fft2_omegas_f)
#error "missing reim2 definitions"
#endif
#if !defined(reim_twiddle_fft_f) || !defined(reim_twiddle_fft_pom_offset) || !defined(fill_reim_twiddle_fft_omegas_f)
#error "missing twiddle definitions"
#endif
#if !defined(reim_bitwiddle_fft_f) || !defined(reim_bitwiddle_fft_pom_offset) || \
    !defined(fill_reim_bitwiddle_fft_omegas_f)
#error "missing bitwiddle definitions"
#endif
#if !defined(reim_fft_bfs_16_f) || !defined(fill_reim_fft_bfs_16_omegas_f)
#error "missing bfs_16 definitions"
#endif
#if !defined(reim_fft_rec_16_f) || !defined(fill_reim_fft_rec_16_omegas_f)
#error "missing rec_16 definitions"
#endif
#if !defined(reim_fft_f) || !defined(fill_reim_fft_omegas_f)
#error "missing main definitions"
#endif

void reim_fft_bfs_16_f(uint64_t m, double* re, double* im, double** omg) {
  uint64_t log2m = log2(m);
  uint64_t mm = m;
  if (log2m & 1) {
    uint64_t h = mm >> 1;
    // do the first twiddle iteration normally
    reim_twiddle_fft_f(h, re, im, *omg);
    *omg += reim_twiddle_fft_pom_offset;
    mm = h;
  }
  while (mm > 16) {
    uint64_t h = mm >> 2;
    for (uint64_t off = 0; off < m; off += mm) {
      reim_bitwiddle_fft_f(h, re + off, im + off, *omg);
      *omg += reim_bitwiddle_fft_pom_offset;
    }
    mm = h;
  }
  if (mm != 16) abort();  // bug!
  for (uint64_t off = 0; off < m; off += 16) {
    reim_fft16_f(re + off, im + off, *omg);
    *omg += reim_fft16_pom_offset;
  }
}

void fill_reim_fft_bfs_16_omegas_f(uint64_t m, double entry_pwr, double** omg) {
  uint64_t log2m = log2(m);
  uint64_t mm = m;
  double ss = entry_pwr;
  if (log2m % 2 != 0) {
    uint64_t h = mm >> 1;
    double s = ss / 2.;
    // do the first twiddle iteration normally
    fill_reim_twiddle_fft_omegas_f(s, omg);
    mm = h;
    ss = s;
  }
  while (mm > 16) {
    uint64_t h = mm >> 2;
    double s = ss / 4.;
    for (uint64_t off = 0; off < m; off += mm) {
      double rs0 = s + fracrevbits(off / mm) / 4.;
      fill_reim_bitwiddle_fft_omegas_f(rs0, omg);
    }
    mm = h;
    ss = s;
  }
  if (mm != 16) abort();  // bug!
  for (uint64_t off = 0; off < m; off += 16) {
    double s = ss + fracrevbits(off / 16);
    fill_reim_fft16_omegas_f(s, omg);
  }
}

void reim_fft_rec_16_f(uint64_t m, double* re, double* im, double** omg) {
  if (m <= 2048) return reim_fft_bfs_16_f(m, re, im, omg);
  const uint32_t h = m >> 1;
  reim_twiddle_fft_f(h, re, im, *omg);
  *omg += reim_twiddle_fft_pom_offset;
  reim_fft_rec_16_f(h, re, im, omg);
  reim_fft_rec_16_f(h, re + h, im + h, omg);
}

void fill_reim_fft_rec_16_omegas_f(uint64_t m, double entry_pwr, double** omg) {
  if (m <= 2048) return fill_reim_fft_bfs_16_omegas_f(m, entry_pwr, omg);
  const uint64_t h = m / 2;
  const double s = entry_pwr / 2;
  fill_reim_twiddle_fft_omegas_f(s, omg);
  fill_reim_fft_rec_16_omegas_f(h, s, omg);
  fill_reim_fft_rec_16_omegas_f(h, s + 0.5, omg);
}

void reim_fft_f(const REIM_FFT_PRECOMP* precomp, double* dat) {
  const int32_t m = precomp->m;
  double* omg = precomp->powomegas;
  double* re = dat;
  double* im = dat + m;
  if (m <= 16) {
    switch (m) {
      case 1:
        return;
      case 2:
        return reim_fft2_f(re, im, omg);
      case 4:
        return reim_fft4_f(re, im, omg);
      case 8:
        return reim_fft8_f(re, im, omg);
      case 16:
        return reim_fft16_f(re, im, omg);
      default:
        abort();  // m is not a power of 2
    }
  }
  if (m <= 2048) return reim_fft_bfs_16_f(m, re, im, &omg);
  return reim_fft_rec_16_f(m, re, im, &omg);
}

void fill_reim_fft_omegas_f(uint64_t m, double entry_pwr, double** omg) {
  if (m <= 16) {
    switch (m) {
      case 1:
        break;
      case 2:
        fill_reim_fft2_omegas_f(entry_pwr, omg);
        break;
      case 4:
        fill_reim_fft4_omegas_f(entry_pwr, omg);
        break;
      case 8:
        fill_reim_fft8_omegas_f(entry_pwr, omg);
        break;
      case 16:
        fill_reim_fft16_omegas_f(entry_pwr, omg);
        break;
      default:
        abort();  // m is not a power of 2
    }
  } else if (m <= 2048) {
    fill_reim_fft_bfs_16_omegas_f(m, entry_pwr, omg);
  } else {
    fill_reim_fft_rec_16_omegas_f(m, entry_pwr, omg);
  }
}

#endif  // SPQLIOS_REIM_FFT_CORE_TEMPLATE_H
