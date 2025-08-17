/*
 * This file is adapted from the implementation of the FFT on Arm64/Neon
 * available in https://github.com/cothan/Falcon-Arm (neon/fft.c).
 * =============================================================================
 * Copyright (c) 2022 by Cryptographic Engineering Research Group (CERG)
 * ECE Department, George Mason University
 * Fairfax, VA, U.S.A.
 * @author: Duc Tri Nguyen dnguye69@gmu.edu, cothannguyen@gmail.com
 * Licensed under the Apache License, Version 2.0 (the "License");
 * =============================================================================
 *
 * The original source file has been modified by the authors of spqlios-arithmetic
 * to be interfaced with the dynamic twiddle factors generator of the spqlios-arithmetic
 * library, as well as the recursive dfs strategy for large complex dimensions m>=2048.
 */

#include <math.h>

#include "../commons.h"
#include "../ext/neon_accel/macrof.h"
#include "../ext/neon_accel/macrofx4.h"

void fill_reim_fft16_omegas_neon(const double entry_pwr, double** omg) {
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
  (*omg)[10] = cos(2. * M_PI * (pin_8 + j_pow));
  (*omg)[12] = cos(2. * M_PI * (pin_8 + k_pow));
  (*omg)[14] = cos(2. * M_PI * (pin_8 + j_pow + k_pow));
  (*omg)[9] = sin(2. * M_PI * (pin_8));
  (*omg)[11] = sin(2. * M_PI * (pin_8 + j_pow));
  (*omg)[13] = sin(2. * M_PI * (pin_8 + k_pow));
  (*omg)[15] = sin(2. * M_PI * (pin_8 + j_pow + k_pow));
  *omg += 16;
}

EXPORT void reim_fft16_neon(double* dre, double* dim, const void* omega) {
  const double* pom = omega;
  // Total SIMD register: 28 = 24 + 4
  float64x2x2_t s_re_im;                                         // 2
  float64x2x4_t x_re, x_im, y_re, y_im, t_re, t_im, v_re, v_im;  // 32

  {
    /*
    Level 2
    (   8,   24) * (   0,    1)
    (   9,   25) * (   0,    1)
    (  10,   26) * (   0,    1)
    (  11,   27) * (   0,    1)
    (  12,   28) * (   0,    1)
    (  13,   29) * (   0,    1)
    (  14,   30) * (   0,    1)
    (  15,   31) * (   0,    1)

    (   8,   24) = (   0,   16) - @
    (   9,   25) = (   1,   17) - @
    (  10,   26) = (   2,   18) - @
    (  11,   27) = (   3,   19) - @
    (  12,   28) = (   4,   20) - @
    (  13,   29) = (   5,   21) - @
    (  14,   30) = (   6,   22) - @
    (  15,   31) = (   7,   23) - @

    (   0,   16) = (   0,   16) + @
    (   1,   17) = (   1,   17) + @
    (   2,   18) = (   2,   18) + @
    (   3,   19) = (   3,   19) + @
    (   4,   20) = (   4,   20) + @
    (   5,   21) = (   5,   21) + @
    (   6,   22) = (   6,   22) + @
    (   7,   23) = (   7,   23) + @
    */
    vload(s_re_im.val[0], pom);

    vloadx4(y_re, dre + 8);
    vloadx4(y_im, dim + 8);

    FWD_TOP_LANEx4(v_re, v_im, y_re, y_im, s_re_im.val[0]);

    vloadx4(x_re, dre);
    vloadx4(x_im, dim);

    FWD_BOTx4(x_re, x_im, y_re, y_im, v_re, v_im);

    // vstorex4(dre, x_re);
    // vstorex4(dim, x_im);
    // vstorex4(dre + 8, y_re);
    // vstorex4(dim + 8, y_im);
    // return;
  }
  {
    /*
    Level 3

    (   4,   20) * (   0,    1)
    (   5,   21) * (   0,    1)
    (   6,   22) * (   0,    1)
    (   7,   23) * (   0,    1)

    (   4,   20) = (   0,   16) - @
    (   5,   21) = (   1,   17) - @
    (   6,   22) = (   2,   18) - @
    (   7,   23) = (   3,   19) - @

    (   0,   16) = (   0,   16) + @
    (   1,   17) = (   1,   17) + @
    (   2,   18) = (   2,   18) + @
    (   3,   19) = (   3,   19) + @

    (  12,   28) * (   0,    1)
    (  13,   29) * (   0,    1)
    (  14,   30) * (   0,    1)
    (  15,   31) * (   0,    1)

    (  12,   28) = (   8,   24) - j@
    (  13,   29) = (   9,   25) - j@
    (  14,   30) = (  10,   26) - j@
    (  15,   31) = (  11,   27) - j@

    (   8,   24) = (   8,   24) + j@
    (   9,   25) = (   9,   25) + j@
    (  10,   26) = (  10,   26) + j@
    (  11,   27) = (  11,   27) + j@
    */

    // vloadx4(y_re, dre + 8);
    // vloadx4(y_im, dim + 8);
    // vloadx4(x_re, dre);
    // vloadx4(x_im, dim);

    vload(s_re_im.val[0], pom + 2);

    FWD_TOP_LANE(t_re.val[0], t_im.val[0], x_re.val[2], x_im.val[2], s_re_im.val[0]);
    FWD_TOP_LANE(t_re.val[1], t_im.val[1], x_re.val[3], x_im.val[3], s_re_im.val[0]);
    FWD_TOP_LANE(t_re.val[2], t_im.val[2], y_re.val[2], y_im.val[2], s_re_im.val[0]);
    FWD_TOP_LANE(t_re.val[3], t_im.val[3], y_re.val[3], y_im.val[3], s_re_im.val[0]);

    FWD_BOT(x_re.val[0], x_im.val[0], x_re.val[2], x_im.val[2], t_re.val[0], t_im.val[0]);
    FWD_BOT(x_re.val[1], x_im.val[1], x_re.val[3], x_im.val[3], t_re.val[1], t_im.val[1]);
    FWD_BOTJ(y_re.val[0], y_im.val[0], y_re.val[2], y_im.val[2], t_re.val[2], t_im.val[2]);
    FWD_BOTJ(y_re.val[1], y_im.val[1], y_re.val[3], y_im.val[3], t_re.val[3], t_im.val[3]);

    // vstorex4(dre, x_re);
    // vstorex4(dim, x_im);
    // vstorex4(dre + 8, y_re);
    // vstorex4(dim + 8, y_im);
    // return;
  }
  {
    /*
    Level 4

    (   2,   18) * (   0,    1)
    (   3,   19) * (   0,    1)
    (   6,   22) * (   0,    1)
    (   7,   23) * (   0,    1)

    (   2,   18) = (   0,   16) - @
    (   3,   19) = (   1,   17) - @
    (   0,   16) = (   0,   16) + @
    (   1,   17) = (   1,   17) + @

    (   6,   22) = (   4,   20) - j@
    (   7,   23) = (   5,   21) - j@
    (   4,   20) = (   4,   20) + j@
    (   5,   21) = (   5,   21) + j@

    (  10,   26) * (   2,    3)
    (  11,   27) * (   2,    3)
    (  14,   30) * (   2,    3)
    (  15,   31) * (   2,    3)

    (  10,   26) = (   8,   24) - @
    (  11,   27) = (   9,   25) - @
    (   8,   24) = (   8,   24) + @
    (   9,   25) = (   9,   25) + @

    (  14,   30) = (  12,   28) - j@
    (  15,   31) = (  13,   29) - j@
    (  12,   28) = (  12,   28) + j@
    (  13,   29) = (  13,   29) + j@
    */
    // vloadx4(y_re, dre + 8);
    // vloadx4(y_im, dim + 8);
    // vloadx4(x_re, dre);
    // vloadx4(x_im, dim);

    vloadx2(s_re_im, pom + 4);

    FWD_TOP_LANE(t_re.val[0], t_im.val[0], x_re.val[1], x_im.val[1], s_re_im.val[0]);
    FWD_TOP_LANE(t_re.val[1], t_im.val[1], x_re.val[3], x_im.val[3], s_re_im.val[0]);
    FWD_TOP_LANE(t_re.val[2], t_im.val[2], y_re.val[1], y_im.val[1], s_re_im.val[1]);
    FWD_TOP_LANE(t_re.val[3], t_im.val[3], y_re.val[3], y_im.val[3], s_re_im.val[1]);

    FWD_BOT(x_re.val[0], x_im.val[0], x_re.val[1], x_im.val[1], t_re.val[0], t_im.val[0]);
    FWD_BOTJ(x_re.val[2], x_im.val[2], x_re.val[3], x_im.val[3], t_re.val[1], t_im.val[1]);
    FWD_BOT(y_re.val[0], y_im.val[0], y_re.val[1], y_im.val[1], t_re.val[2], t_im.val[2]);
    FWD_BOTJ(y_re.val[2], y_im.val[2], y_re.val[3], y_im.val[3], t_re.val[3], t_im.val[3]);

    // vstorex4(dre, x_re);
    // vstorex4(dim, x_im);
    // vstorex4(dre + 8, y_re);
    // vstorex4(dim + 8, y_im);
    // return;
  }
  {
    /*
    Level 5

    (   1,   17) * (   0,    1)
    (   5,   21) * (   2,    3)
    ------
    (   1,   17) = (   0,   16) - @
    (   5,   21) = (   4,   20) - @
    (   0,   16) = (   0,   16) + @
    (   4,   20) = (   4,   20) + @

    (   3,   19) * (   0,    1)
    (   7,   23) * (   2,    3)
    ------
    (   3,   19) = (   2,   18) - j@
    (   7,   23) = (   6,   22) - j@
    (   2,   18) = (   2,   18) + j@
    (   6,   22) = (   6,   22) + j@

    (   9,   25) * (   4,    5)
    (  13,   29) * (   6,    7)
    ------
    (   9,   25) = (   8,   24) - @
    (  13,   29) = (  12,   28) - @
    (   8,   24) = (   8,   24) + @
    (  12,   28) = (  12,   28) + @

    (  11,   27) * (   4,    5)
    (  15,   31) * (   6,    7)
    ------
    (  11,   27) = (  10,   26) - j@
    (  15,   31) = (  14,   30) - j@
    (  10,   26) = (  10,   26) + j@
    (  14,   30) = (  14,   30) + j@

    before transpose_f64
    x_re: 0, 1 |  2,  3 |  4,  5 |  6,  7
    y_re: 8, 9 | 10, 11 | 12, 13 | 14, 15
    after transpose_f64
    x_re: 0, 4 |  2,  6 |  1,  5 |  3,  7
    y_re: 8, 12|  9,  13| 10, 14 | 11, 15
    after swap
    x_re: 0, 4 |  1,  5 | 2,  6 |  3,  7
    y_re: 8, 12| 10, 14 | 9,  13| 11, 15
    */

    // vloadx4(y_re, dre + 8);
    // vloadx4(y_im, dim + 8);
    // vloadx4(x_re, dre);
    // vloadx4(x_im, dim);

    transpose_f64(x_re, x_re, v_re, 0, 2, 0);
    transpose_f64(x_re, x_re, v_re, 1, 3, 1);
    transpose_f64(x_im, x_im, v_im, 0, 2, 0);
    transpose_f64(x_im, x_im, v_im, 1, 3, 1);

    v_re.val[0] = x_re.val[2];
    x_re.val[2] = x_re.val[1];
    x_re.val[1] = v_re.val[0];

    v_im.val[0] = x_im.val[2];
    x_im.val[2] = x_im.val[1];
    x_im.val[1] = v_im.val[0];

    transpose_f64(y_re, y_re, v_re, 0, 2, 2);
    transpose_f64(y_re, y_re, v_re, 1, 3, 3);
    transpose_f64(y_im, y_im, v_im, 0, 2, 2);
    transpose_f64(y_im, y_im, v_im, 1, 3, 3);

    v_re.val[0] = y_re.val[2];
    y_re.val[2] = y_re.val[1];
    y_re.val[1] = v_re.val[0];

    v_im.val[0] = y_im.val[2];
    y_im.val[2] = y_im.val[1];
    y_im.val[1] = v_im.val[0];

    // double pom8[] = {pom[8], pom[12], pom[9], pom[13]};
    vload2(s_re_im, pom + 8);
    // vload2(s_re_im, pom8);

    FWD_TOP(t_re.val[0], t_im.val[0], x_re.val[1], x_im.val[1], s_re_im.val[0], s_re_im.val[1]);
    FWD_TOP(t_re.val[1], t_im.val[1], x_re.val[3], x_im.val[3], s_re_im.val[0], s_re_im.val[1]);

    // double pom12[] = {pom[10], pom[14], pom[11], pom[15]};
    vload2(s_re_im, pom + 12);
    // vload2(s_re_im, pom12);

    FWD_TOP(t_re.val[2], t_im.val[2], y_re.val[1], y_im.val[1], s_re_im.val[0], s_re_im.val[1]);
    FWD_TOP(t_re.val[3], t_im.val[3], y_re.val[3], y_im.val[3], s_re_im.val[0], s_re_im.val[1]);

    FWD_BOT(x_re.val[0], x_im.val[0], x_re.val[1], x_im.val[1], t_re.val[0], t_im.val[0]);
    FWD_BOTJ(x_re.val[2], x_im.val[2], x_re.val[3], x_im.val[3], t_re.val[1], t_im.val[1]);

    vstore4(dre, x_re);
    vstore4(dim, x_im);

    FWD_BOT(y_re.val[0], y_im.val[0], y_re.val[1], y_im.val[1], t_re.val[2], t_im.val[2]);
    FWD_BOTJ(y_re.val[2], y_im.val[2], y_re.val[3], y_im.val[3], t_re.val[3], t_im.val[3]);

    vstore4(dre + 8, y_re);
    vstore4(dim + 8, y_im);
  }
}

void reim_twiddle_fft_neon(uint64_t h, double* re, double* im, double om[2]) {
  // Total SIMD register: 28 = 24 + 4
  if (h < 8) abort();                                // bug
  float64x2_t s_re_im;                               // 2
  float64x2x4_t x_re, x_im, y_re, y_im, v_re, v_im;  // 32
  vload(s_re_im, om);
  for (uint64_t blk = 0; blk < h; blk += 8) {
    double* dre = re + blk;
    double* dim = im + blk;
    vloadx4(y_re, dre + h);
    vloadx4(y_im, dim + h);
    FWD_TOP_LANEx4(v_re, v_im, y_re, y_im, s_re_im);
    vloadx4(x_re, dre);
    vloadx4(x_im, dim);
    FWD_BOTx4(x_re, x_im, y_re, y_im, v_re, v_im);
    vstorex4(dre, x_re);
    vstorex4(dim, x_im);
    vstorex4(dre + h, y_re);
    vstorex4(dim + h, y_im);
  }
}

void reim_ctwiddle(double* ra, double* ia, double* rb, double* ib, double omre, double omim);
// i (omre + i omim) = -omim + i omre
void reim_citwiddle(double* ra, double* ia, double* rb, double* ib, double omre, double omim);

void reim_bitwiddle_fft_neon(uint64_t h, double* re, double* im, double om[4]) {
  // Total SIMD register: 28 = 24 + 4
  if (h < 4) abort();  // bug
  double* r0 = re;
  double* r1 = re + h;
  double* r2 = re + 2 * h;
  double* r3 = re + 3 * h;
  double* i0 = im;
  double* i1 = im + h;
  double* i2 = im + 2 * h;
  double* i3 = im + 3 * h;
  float64x2x2_t s_re_im;                     // 2
  float64x2x4_t v_re, v_im;                  // 2
  float64x2x2_t x0_re, x0_im, x1_re, x1_im;  // 32
  float64x2x2_t x2_re, x2_im, x3_re, x3_im;  // 32
  vloadx2(s_re_im, om);
  for (uint64_t blk = 0; blk < h; blk += 4) {
    {
      vloadx2(x2_re, r2 + blk);
      vloadx2(x3_re, r3 + blk);
      vloadx2(x2_im, i2 + blk);
      vloadx2(x3_im, i3 + blk);
      FWD_TOP_LANE(v_re.val[0], v_im.val[0], x2_re.val[0], x2_im.val[0], s_re_im.val[0]);
      FWD_TOP_LANE(v_re.val[1], v_im.val[1], x2_re.val[1], x2_im.val[1], s_re_im.val[0]);
      FWD_TOP_LANE(v_re.val[2], v_im.val[2], x3_re.val[0], x3_im.val[0], s_re_im.val[0]);
      FWD_TOP_LANE(v_re.val[3], v_im.val[3], x3_re.val[1], x3_im.val[1], s_re_im.val[0]);
      vloadx2(x0_re, r0 + blk);
      vloadx2(x1_re, r1 + blk);
      vloadx2(x0_im, i0 + blk);
      vloadx2(x1_im, i1 + blk);
      FWD_BOT(x0_re.val[0], x0_im.val[0], x2_re.val[0], x2_im.val[0], v_re.val[0], v_im.val[0]);
      FWD_BOT(x0_re.val[1], x0_im.val[1], x2_re.val[1], x2_im.val[1], v_re.val[1], v_im.val[1]);
      FWD_BOT(x1_re.val[0], x1_im.val[0], x3_re.val[0], x3_im.val[0], v_re.val[2], v_im.val[2]);
      FWD_BOT(x1_re.val[1], x1_im.val[1], x3_re.val[1], x3_im.val[1], v_re.val[3], v_im.val[3]);
    }
    {
      FWD_TOP_LANE(v_re.val[0], v_im.val[0], x1_re.val[0], x1_im.val[0], s_re_im.val[1]);
      FWD_TOP_LANE(v_re.val[1], v_im.val[1], x1_re.val[1], x1_im.val[1], s_re_im.val[1]);
      FWD_TOP_LANE(v_re.val[2], v_im.val[2], x3_re.val[0], x3_im.val[0], s_re_im.val[1]);
      FWD_TOP_LANE(v_re.val[3], v_im.val[3], x3_re.val[1], x3_im.val[1], s_re_im.val[1]);
      FWD_BOT(x0_re.val[0], x0_im.val[0], x1_re.val[0], x1_im.val[0], v_re.val[0], v_im.val[0]);
      FWD_BOT(x0_re.val[1], x0_im.val[1], x1_re.val[1], x1_im.val[1], v_re.val[1], v_im.val[1]);
      FWD_BOTJ(x2_re.val[0], x2_im.val[0], x3_re.val[0], x3_im.val[0], v_re.val[2], v_im.val[2]);
      FWD_BOTJ(x2_re.val[1], x2_im.val[1], x3_re.val[1], x3_im.val[1], v_re.val[3], v_im.val[3]);
      vstorex2(r0 + blk, x0_re);
      vstorex2(r1 + blk, x1_re);
      vstorex2(r2 + blk, x2_re);
      vstorex2(r3 + blk, x3_re);
      vstorex2(i0 + blk, x0_im);
      vstorex2(i1 + blk, x1_im);
      vstorex2(i2 + blk, x2_im);
      vstorex2(i3 + blk, x3_im);
    }
  }
}

#if 0
static void ZfN(iFFT_log2)(fpr *f)
{
 /*
 y_re: 1 = (2 - 3) * 5 + (0 - 1) * 4
 y_im: 3 = (2 - 3) * 4 - (0 - 1) * 5
 x_re: 0 = 0 + 1
 x_im: 2 = 2 + 3

 Turn out this vectorize code is too short to be executed,
 the scalar version is consistently faster

 float64x2x2_t tmp;
 float64x2_t v, s, t;

 // 0: 0, 2
 // 1: 1, 3

 vload2(tmp, &f[0]);
 vload(s, &fpr_gm_tab[4]);

 vfsub(v, tmp.val[0], tmp.val[1]);
 vfadd(tmp.val[0], tmp.val[0], tmp.val[1]);

 // y_im: 3 = (2 - 3) * 4  - (0 - 1) * 5
 // y_re: 1 = (2 - 3) * 5  + (0 - 1) * 4
 vswap(t, v);

 vfmul_lane(tmp.val[1], s, t, 0);
 vfcmla_90(tmp.val[1], t, s);

 vfmuln(tmp.val[0], tmp.val[0], 0.5);
 vfmuln(tmp.val[1], tmp.val[1], 0.5);

 vswap(tmp.val[1], tmp.val[1]);

 vstore2(&f[0], tmp);
 */

 fpr x_re, x_im, y_re, y_im, s;
 x_re = f[0];
 y_re = f[1];
 x_im = f[2];
 y_im = f[3];
 s = fpr_tab_log2[0] * 0.5;

 f[0] = (x_re + y_re) * 0.5;
 f[2] = (x_im + y_im) * 0.5;

 x_re = (x_re - y_re) * s;
 x_im = (x_im - y_im) * s;

 f[1] = x_im + x_re;
 f[3] = x_im - x_re;
}

static void ZfN(iFFT_log3)(fpr *f)
{
 /*
  * Total instructions: 27
 y_re: 1 = (4 - 5) *  9 + (0 - 1) *  8
 y_re: 3 = (6 - 7) * 11 + (2 - 3) * 10
 y_im: 5 = (4 - 5) *  8 - (0 - 1) *  9
 y_im: 7 = (6 - 7) * 10 - (2 - 3) * 11
 x_re: 0 = 0 + 1
 x_re: 2 = 2 + 3
 x_im: 4 = 4 + 5
 x_im: 6 = 6 + 7
  */
 // 0: 0, 2 - 0: 0, 4
 // 1: 1, 3 - 1: 1, 5
 // 2: 4, 6 - 2: 2, 6
 // 3: 5, 7 - 3: 3, 7
 float64x2x4_t tmp;
 float64x2x2_t x_re_im, y_re_im, v, s_re_im;

 vload2(x_re_im, &f[0]);
 vload2(y_re_im, &f[4]);

 vfsub(v.val[0], x_re_im.val[0], x_re_im.val[1]);
 vfsub(v.val[1], y_re_im.val[0], y_re_im.val[1]);
 vfadd(x_re_im.val[0], x_re_im.val[0], x_re_im.val[1]);
 vfadd(x_re_im.val[1], y_re_im.val[0], y_re_im.val[1]);

 // 0: 8, 10
 // 1: 9, 11
 vload2(s_re_im, &fpr_tab_log3[0]);

 vfmul(y_re_im.val[0], v.val[1], s_re_im.val[1]);
 vfmla(y_re_im.val[0], y_re_im.val[0], v.val[0], s_re_im.val[0]);
 vfmul(y_re_im.val[1], v.val[1], s_re_im.val[0]);
 vfmls(y_re_im.val[1], y_re_im.val[1], v.val[0], s_re_im.val[1]);

 // x: 0,2 | 4,6
 // y: 1,3 | 5,7
 tmp.val[0] = vtrn1q_f64(x_re_im.val[0], y_re_im.val[0]);
 tmp.val[1] = vtrn2q_f64(x_re_im.val[0], y_re_im.val[0]);
 tmp.val[2] = vtrn1q_f64(x_re_im.val[1], y_re_im.val[1]);
 tmp.val[3] = vtrn2q_f64(x_re_im.val[1], y_re_im.val[1]);
 // tmp: 0,1 | 2,3 | 4,5 | 6,7
 /*
 y_re: 2 = (4 - 6) * 4 + (0 - 2) * 4
 y_re: 3 = (5 - 7) * 4 + (1 - 3) * 4
 y_im: 6 = (4 - 6) * 4 - (0 - 2) * 4
 y_im: 7 = (5 - 7) * 4 - (1 - 3) * 4
 x_re: 0 = 0 + 2
 x_re: 1 = 1 + 3
 x_im: 4 = 4 + 6
 x_im: 5 = 5 + 7
 */
 s_re_im.val[0] = vld1q_dup_f64(&fpr_tab_log2[0]);

 vfadd(x_re_im.val[0], tmp.val[0], tmp.val[1]);
 vfadd(x_re_im.val[1], tmp.val[2], tmp.val[3]);
 vfsub(v.val[0], tmp.val[0], tmp.val[1]);
 vfsub(v.val[1], tmp.val[2], tmp.val[3]);

 vfmuln(tmp.val[0], x_re_im.val[0], 0.25);
 vfmuln(tmp.val[2], x_re_im.val[1], 0.25);

 vfmuln(s_re_im.val[0], s_re_im.val[0], 0.25);

 vfmul(y_re_im.val[0], v.val[0], s_re_im.val[0]);
 vfmul(y_re_im.val[1], v.val[1], s_re_im.val[0]);

 vfadd(tmp.val[1], y_re_im.val[1], y_re_im.val[0]);
 vfsub(tmp.val[3], y_re_im.val[1], y_re_im.val[0]);

 vstorex4(&f[0], tmp);
}

static void ZfN(iFFT_log4)(fpr *f)
{
 /*
   * (   0,    8) - (   1,    9)
   * (   4,   12) - (   5,   13)
   *
   * (   0,    8) + (   1,    9)
   * (   4,   12) + (   5,   13)
   *
   * (   3,   11) - (   2,   10)
   * (   7,   15) - (   6,   14)
   *
   * (   2,   10) + (   3,   11)
   * (   6,   14) + (   7,   15)
   *
   * (   1,    9) = @ * (   0,    1)
   * (   5,   13) = @ * (   2,    3)
   *
   * (   3,   11) = j@ * (   0,    1)
   * (   7,   15) = j@ * (   2,    3)
  */

 float64x2x4_t re, im, t;
 float64x2x2_t t_re, t_im, s_re_im;

 vload4(re, &f[0]);
 vload4(im, &f[8]);

 INV_TOPJ (t_re.val[0], t_im.val[0], re.val[0], im.val[0], re.val[1], im.val[1]);
 INV_TOPJm(t_re.val[1], t_im.val[1], re.val[2], im.val[2], re.val[3], im.val[3]);

 vload2(s_re_im, &fpr_tab_log4[0]);

 INV_BOTJ (re.val[1], im.val[1], t_re.val[0], t_im.val[0], s_re_im.val[0], s_re_im.val[1]);
 INV_BOTJm(re.val[3], im.val[3], t_re.val[1], t_im.val[1], s_re_im.val[0], s_re_im.val[1]);

 /*
   * (   0,    8) - (   2,   10)
   * (   1,    9) - (   3,   11)
   *
   * (   0,    8) + (   2,   10)
   * (   1,    9) + (   3,   11)
   *
   * (   2,   10) = @ * (   0,    1)
   * (   3,   11) = @ * (   0,    1)
   *
   * (   6,   14) - (   4,   12)
   * (   7,   15) - (   5,   13)
   *
   * (   4,   12) + (   6,   14)
   * (   5,   13) + (   7,   15)
   *
   * (   6,   14) = j@ * (   0,    1)
   * (   7,   15) = j@ * (   0,    1)
  */

 // re: 0, 4 | 1, 5 | 2, 6 | 3, 7
 // im: 8, 12| 9, 13|10, 14|11, 15

 transpose_f64(re, re, t, 0, 1, 0);
 transpose_f64(re, re, t, 2, 3, 1);
 transpose_f64(im, im, t, 0, 1, 2);
 transpose_f64(im, im, t, 2, 3, 3);

 // re: 0, 1 | 4,  5 | 2, 3 | 6, 7
 // im: 8, 9 | 12, 13|10, 11| 14, 15
 t.val[0] = re.val[1];
 re.val[1] = re.val[2];
 re.val[2] = t.val[0];

 t.val[1]  = im.val[1];
 im.val[1] = im.val[2];
 im.val[2] = t.val[1];

 // re: 0, 1 |  2,  3| 4,  5 | 6, 7
 // im: 8, 9 | 10, 11| 12, 13| 14, 15

 INV_TOPJ (t_re.val[0], t_im.val[0], re.val[0], im.val[0], re.val[1], im.val[1]);
 INV_TOPJm(t_re.val[1], t_im.val[1], re.val[2], im.val[2], re.val[3], im.val[3]);

 vload(s_re_im.val[0], &fpr_tab_log3[0]);

 INV_BOTJ_LANE (re.val[1], im.val[1], t_re.val[0], t_im.val[0], s_re_im.val[0]);
 INV_BOTJm_LANE(re.val[3], im.val[3], t_re.val[1], t_im.val[1], s_re_im.val[0]);

 /*
    * (   0,    8) - (   4,   12)
    * (   1,    9) - (   5,   13)
    * (   0,    8) + (   4,   12)
    * (   1,    9) + (   5,   13)
    *
    * (   2,   10) - (   6,   14)
    * (   3,   11) - (   7,   15)
    * (   2,   10) + (   6,   14)
    * (   3,   11) + (   7,   15)
    *
    * (   4,   12) = @ * (   0,    1)
    * (   5,   13) = @ * (   0,    1)
    *
    * (   6,   14) = @ * (   0,    1)
    * (   7,   15) = @ * (   0,    1)
  */

 INV_TOPJ(t_re.val[0], t_im.val[0], re.val[0], im.val[0], re.val[2], im.val[2]);
 INV_TOPJ(t_re.val[1], t_im.val[1], re.val[1], im.val[1], re.val[3], im.val[3]);

 vfmuln(re.val[0], re.val[0], 0.12500000000);
 vfmuln(re.val[1], re.val[1], 0.12500000000);
 vfmuln(im.val[0], im.val[0], 0.12500000000);
 vfmuln(im.val[1], im.val[1], 0.12500000000);

 s_re_im.val[0] = vld1q_dup_f64(&fpr_tab_log2[0]);

 vfmuln(s_re_im.val[0], s_re_im.val[0], 0.12500000000);

 vfmul(t_re.val[0], t_re.val[0], s_re_im.val[0]);
 vfmul(t_re.val[1], t_re.val[1], s_re_im.val[0]);
 vfmul(t_im.val[0], t_im.val[0], s_re_im.val[0]);
 vfmul(t_im.val[1], t_im.val[1], s_re_im.val[0]);

 vfsub(im.val[2], t_im.val[0], t_re.val[0]);
 vfsub(im.val[3], t_im.val[1], t_re.val[1]);
 vfadd(re.val[2], t_im.val[0], t_re.val[0]);
 vfadd(re.val[3], t_im.val[1], t_re.val[1]);

 vstorex4(&f[0], re);
 vstorex4(&f[8], im);
}

static
   void ZfN(iFFT_log5)(fpr *f, const unsigned logn, const unsigned last)
{
 // Total SIMD register: 26 = 24 + 2
 float64x2x4_t x_re, x_im, y_re, y_im, t_re, t_im; // 24
 float64x2x2_t s_re_im;                            // 2
 const unsigned n = 1 << logn;
 const unsigned hn = n >> 1;

 int level = logn;
 const fpr *fpr_tab5 = fpr_table[level--],
           *fpr_tab4 = fpr_table[level--],
           *fpr_tab3 = fpr_table[level--],
           *fpr_tab2 = fpr_table[level];
 int k2 = 0, k3 = 0, k4 = 0, k5 = 0;

 for (unsigned j = 0; j < hn; j += 16)
 {
   /*
        * (   0,   16) - (   1,   17)
        * (   4,   20) - (   5,   21)
        * (   0,   16) + (   1,   17)
        * (   4,   20) + (   5,   21)
        * (   1,   17) = @ * (   0,    1)
        * (   5,   21) = @ * (   2,    3)
        *
        * (   2,   18) - (   3,   19)
        * (   6,   22) - (   7,   23)
        * (   2,   18) + (   3,   19)
        * (   6,   22) + (   7,   23)
        * (   3,   19) = j@ * (   0,    1)
        * (   7,   23) = j@ * (   2,    3)
        *
        * (   8,   24) - (   9,   25)
        * (  12,   28) - (  13,   29)
        * (   8,   24) + (   9,   25)
        * (  12,   28) + (  13,   29)
        * (   9,   25) = @ * (   4,    5)
        * (  13,   29) = @ * (   6,    7)
        *
        * (  10,   26) - (  11,   27)
        * (  14,   30) - (  15,   31)
        * (  10,   26) + (  11,   27)
        * (  14,   30) + (  15,   31)
        * (  11,   27) = j@ * (   4,    5)
        * (  15,   31) = j@ * (   6,    7)
    */

   vload4(x_re, &f[j]);
   vload4(x_im, &f[j + hn]);

   INV_TOPJ(t_re.val[0], t_im.val[0], x_re.val[0], x_im.val[0], x_re.val[1], x_im.val[1]);
   INV_TOPJm(t_re.val[2], t_im.val[2], x_re.val[2], x_im.val[2], x_re.val[3], x_im.val[3]);

   vload4(y_re, &f[j + 8]);
   vload4(y_im, &f[j + 8 + hn])

       INV_TOPJ(t_re.val[1], t_im.val[1], y_re.val[0], y_im.val[0], y_re.val[1], y_im.val[1]);
   INV_TOPJm(t_re.val[3], t_im.val[3], y_re.val[2], y_im.val[2], y_re.val[3], y_im.val[3]);

   vload2(s_re_im, &fpr_tab5[k5]);
   k5 += 4;

   INV_BOTJ (x_re.val[1], x_im.val[1], t_re.val[0], t_im.val[0], s_re_im.val[0], s_re_im.val[1]);
   INV_BOTJm(x_re.val[3], x_im.val[3], t_re.val[2], t_im.val[2], s_re_im.val[0], s_re_im.val[1]);

   vload2(s_re_im, &fpr_tab5[k5]);
   k5 += 4;

   INV_BOTJ (y_re.val[1], y_im.val[1], t_re.val[1], t_im.val[1], s_re_im.val[0], s_re_im.val[1]);
   INV_BOTJm(y_re.val[3], y_im.val[3], t_re.val[3], t_im.val[3], s_re_im.val[0], s_re_im.val[1]);


   // x_re: 0, 4 | 1, 5 | 2, 6 | 3, 7
   // y_re: 8, 12| 9, 13|10, 14|11, 15

   transpose_f64(x_re, x_re, t_re, 0, 1, 0);
   transpose_f64(x_re, x_re, t_re, 2, 3, 1);
   transpose_f64(y_re, y_re, t_re, 0, 1, 2);
   transpose_f64(y_re, y_re, t_re, 2, 3, 3);

   transpose_f64(x_im, x_im, t_im, 0, 1, 0);
   transpose_f64(x_im, x_im, t_im, 2, 3, 1);
   transpose_f64(y_im, y_im, t_im, 0, 1, 2);
   transpose_f64(y_im, y_im, t_im, 2, 3, 3);

   // x_re: 0, 1 | 4, 5 | 2, 3 | 6, 7
   // y_re: 8, 9 | 12,13|10,11 |14, 15

   t_re.val[0] = x_re.val[1];
   x_re.val[1] = x_re.val[2];
   x_re.val[2] = t_re.val[0];

   t_re.val[1] = y_re.val[1];
   y_re.val[1] = y_re.val[2];
   y_re.val[2] = t_re.val[1];


   t_im.val[0] = x_im.val[1];
   x_im.val[1] = x_im.val[2];
   x_im.val[2] = t_im.val[0];

   t_im.val[1] = y_im.val[1];
   y_im.val[1] = y_im.val[2];
   y_im.val[2] = t_im.val[1];
   // x_re: 0, 1 |  2,  3| 4,  5 | 6, 7
   // y_re: 8, 9 | 10, 11| 12, 13| 14, 15

   /*
        * (   0,   16) - (   2,   18)
        * (   1,   17) - (   3,   19)
        * (   0,   16) + (   2,   18)
        * (   1,   17) + (   3,   19)
        * (   2,   18) = @ * (   0,    1)
        * (   3,   19) = @ * (   0,    1)
        *
        * (   4,   20) - (   6,   22)
        * (   5,   21) - (   7,   23)
        * (   4,   20) + (   6,   22)
        * (   5,   21) + (   7,   23)
        * (   6,   22) = j@ * (   0,    1)
        * (   7,   23) = j@ * (   0,    1)
        *
        * (   8,   24) - (  10,   26)
        * (   9,   25) - (  11,   27)
        * (   8,   24) + (  10,   26)
        * (   9,   25) + (  11,   27)
        * (  10,   26) = @ * (   2,    3)
        * (  11,   27) = @ * (   2,    3)
        *
        * (  12,   28) - (  14,   30)
        * (  13,   29) - (  15,   31)
        * (  12,   28) + (  14,   30)
        * (  13,   29) + (  15,   31)
        * (  14,   30) = j@ * (   2,    3)
        * (  15,   31) = j@ * (   2,    3)
    */

   INV_TOPJ (t_re.val[0], t_im.val[0], x_re.val[0], x_im.val[0], x_re.val[1], x_im.val[1]);
   INV_TOPJm(t_re.val[1], t_im.val[1], x_re.val[2], x_im.val[2], x_re.val[3], x_im.val[3]);

   INV_TOPJ (t_re.val[2], t_im.val[2], y_re.val[0], y_im.val[0], y_re.val[1], y_im.val[1]);
   INV_TOPJm(t_re.val[3], t_im.val[3], y_re.val[2], y_im.val[2], y_re.val[3], y_im.val[3]);

   vloadx2(s_re_im, &fpr_tab4[k4]);
   k4 += 4;

   INV_BOTJ_LANE (x_re.val[1], x_im.val[1], t_re.val[0], t_im.val[0], s_re_im.val[0]);
   INV_BOTJm_LANE(x_re.val[3], x_im.val[3], t_re.val[1], t_im.val[1], s_re_im.val[0]);

   INV_BOTJ_LANE (y_re.val[1], y_im.val[1], t_re.val[2], t_im.val[2], s_re_im.val[1]);
   INV_BOTJm_LANE(y_re.val[3], y_im.val[3], t_re.val[3], t_im.val[3], s_re_im.val[1]);

   /*
        * (   0,   16) - (   4,   20)
        * (   1,   17) - (   5,   21)
        * (   0,   16) + (   4,   20)
        * (   1,   17) + (   5,   21)
        * (   4,   20) = @ * (   0,    1)
        * (   5,   21) = @ * (   0,    1)
        *
        * (   2,   18) - (   6,   22)
        * (   3,   19) - (   7,   23)
        * (   2,   18) + (   6,   22)
        * (   3,   19) + (   7,   23)
        * (   6,   22) = @ * (   0,    1)
        * (   7,   23) = @ * (   0,    1)
        *
        * (   8,   24) - (  12,   28)
        * (   9,   25) - (  13,   29)
        * (   8,   24) + (  12,   28)
        * (   9,   25) + (  13,   29)
        * (  12,   28) = j@ * (   0,    1)
        * (  13,   29) = j@ * (   0,    1)
        *
        * (  10,   26) - (  14,   30)
        * (  11,   27) - (  15,   31)
        * (  10,   26) + (  14,   30)
        * (  11,   27) + (  15,   31)
        * (  14,   30) = j@ * (   0,    1)
        * (  15,   31) = j@ * (   0,    1)
    */

   INV_TOPJ (t_re.val[0], t_im.val[0], x_re.val[0], x_im.val[0], x_re.val[2], x_im.val[2]);
   INV_TOPJ (t_re.val[1], t_im.val[1], x_re.val[1], x_im.val[1], x_re.val[3], x_im.val[3]);

   INV_TOPJm(t_re.val[2], t_im.val[2], y_re.val[0], y_im.val[0], y_re.val[2], y_im.val[2]);
   INV_TOPJm(t_re.val[3], t_im.val[3], y_re.val[1], y_im.val[1], y_re.val[3], y_im.val[3]);

   vload(s_re_im.val[0], &fpr_tab3[k3]);
   k3 += 2;

   INV_BOTJ_LANE(x_re.val[2], x_im.val[2], t_re.val[0], t_im.val[0], s_re_im.val[0]);
   INV_BOTJ_LANE(x_re.val[3], x_im.val[3], t_re.val[1], t_im.val[1], s_re_im.val[0]);

   INV_BOTJm_LANE(y_re.val[2], y_im.val[2], t_re.val[2], t_im.val[2], s_re_im.val[0]);
   INV_BOTJm_LANE(y_re.val[3], y_im.val[3], t_re.val[3], t_im.val[3], s_re_im.val[0]);

   /*
        * (   0,   16) - (   8,   24)
        * (   1,   17) - (   9,   25)
        * (   0,   16) + (   8,   24)
        * (   1,   17) + (   9,   25)
        * (   8,   24) = @ * (   0,    1)
        * (   9,   25) = @ * (   0,    1)
        *
        * (   2,   18) - (  10,   26)
        * (   3,   19) - (  11,   27)
        * (   2,   18) + (  10,   26)
        * (   3,   19) + (  11,   27)
        * (  10,   26) = @ * (   0,    1)
        * (  11,   27) = @ * (   0,    1)
        *
        * (   4,   20) - (  12,   28)
        * (   5,   21) - (  13,   29)
        * (   4,   20) + (  12,   28)
        * (   5,   21) + (  13,   29)
        * (  12,   28) = @ * (   0,    1)
        * (  13,   29) = @ * (   0,    1)
        *
        * (   6,   22) - (  14,   30)
        * (   7,   23) - (  15,   31)
        * (   6,   22) + (  14,   30)
        * (   7,   23) + (  15,   31)
        * (  14,   30) = @ * (   0,    1)
        * (  15,   31) = @ * (   0,    1)
    */


   if ( (j >> 4) & 1)
   {
     INV_TOPJmx4(t_re, t_im, x_re, x_im, y_re, y_im);
   }
   else
   {
     INV_TOPJx4(t_re, t_im, x_re, x_im, y_re, y_im);
   }

   vload(s_re_im.val[0], &fpr_tab2[k2]);
   k2 += 2 * ((j & 31) == 16);

   if (last)
   {
     vfmuln(s_re_im.val[0], s_re_im.val[0], fpr_p2_tab[logn]);
     vfmulnx4(x_re, x_re, fpr_p2_tab[logn]);
     vfmulnx4(x_im, x_im, fpr_p2_tab[logn]);
   }
   vstorex4(&f[j], x_re);
   vstorex4(&f[j + hn], x_im);

   if (logn == 5)
   {
     // Special case in fpr_tab_log2 where re == im
     vfmulx4_i(t_re, t_re, s_re_im.val[0]);
     vfmulx4_i(t_im, t_im, s_re_im.val[0]);

     vfaddx4(y_re, t_im, t_re);
     vfsubx4(y_im, t_im, t_re);
   }
   else
   {
     if ((j >> 4) & 1)
     {
       INV_BOTJm_LANEx4(y_re, y_im, t_re, t_im, s_re_im.val[0]);
     }
     else
     {
       INV_BOTJ_LANEx4(y_re, y_im, t_re, t_im, s_re_im.val[0]);
     }
   }

   vstorex4(&f[j + 8], y_re);
   vstorex4(&f[j + 8 + hn], y_im);
 }
}

static
   void ZfN(iFFT_logn1)(fpr *f, const unsigned logn, const unsigned last)
{
 // Total SIMD register 26 = 24 + 2
 float64x2x4_t a_re, a_im, b_re, b_im, t_re, t_im; // 24
 float64x2_t s_re_im;                              // 2

 const unsigned n = 1 << logn;
 const unsigned hn = n >> 1;
 const unsigned ht = n >> 2;

 for (unsigned j = 0; j < ht; j+= 8)
 {
   vloadx4(a_re, &f[j]);
   vloadx4(a_im, &f[j + hn]);
   vloadx4(b_re, &f[j + ht]);
   vloadx4(b_im, &f[j + ht + hn]);

   INV_TOPJx4(t_re, t_im, a_re, a_im, b_re, b_im);

   s_re_im = vld1q_dup_f64(&fpr_tab_log2[0]);

   if (last)
   {
     vfmuln(s_re_im, s_re_im, fpr_p2_tab[logn]);
     vfmulnx4(a_re, a_re, fpr_p2_tab[logn]);
     vfmulnx4(a_im, a_im, fpr_p2_tab[logn]);
   }

   vstorex4(&f[j], a_re);
   vstorex4(&f[j + hn], a_im);

   vfmulx4_i(t_re, t_re, s_re_im);
   vfmulx4_i(t_im, t_im, s_re_im);

   vfaddx4(b_re, t_im, t_re);
   vfsubx4(b_im, t_im, t_re);

   vstorex4(&f[j + ht], b_re);
   vstorex4(&f[j + ht + hn], b_im);
 }
}

// static
// void ZfN(iFFT_logn2)(fpr *f, const unsigned logn, const unsigned level, unsigned last)
// {
//     const unsigned int falcon_n = 1 << logn;
//     const unsigned int hn = falcon_n >> 1;

//     // Total SIMD register: 26 = 16 + 8 + 2
//     float64x2x4_t t_re, t_im;                 // 8
//     float64x2x2_t x1_re, x2_re, x1_im, x2_im,
//                   y1_re, y2_re, y1_im, y2_im; // 16
//     float64x2_t s1_re_im, s2_re_im;           // 2

//     const fpr *fpr_inv_tab1 = NULL, *fpr_inv_tab2 = NULL;
//     unsigned l, len, start, j, k1, k2;
//     unsigned bar = logn - 4 - 2;
//     unsigned Jm;

//     for (l = 4; l < logn - level - 1; l += 2)
//     {
//         len = 1 << l;
//         last -= 1;
//         fpr_inv_tab1 = fpr_table[bar--];
//         fpr_inv_tab2 = fpr_table[bar--];
//         k1 = 0; k2 = 0;

//         for (start = 0; start < hn; start += 1 << (l + 2))
//         {
//             vload(s1_re_im, &fpr_inv_tab1[k1]);
//             vload(s2_re_im, &fpr_inv_tab2[k2]);
//             k1 += 2;
//             k2 += 2 * ((start & 127) == 64);
//             if (!last)
//             {
//                 vfmuln(s2_re_im, s2_re_im, fpr_p2_tab[logn]);
//             }
//             Jm = (start >> (l+ 2)) & 1;
//             for (j = start; j < start + len; j += 4)
//             {
//                 /*
//                 Level 6
//                  * (   0,   64) - (  16,   80)
//                  * (   1,   65) - (  17,   81)
//                  * (   0,   64) + (  16,   80)
//                  * (   1,   65) + (  17,   81)
//                  * (  16,   80) = @ * (   0,    1)
//                  * (  17,   81) = @ * (   0,    1)
//                  *
//                  * (   2,   66) - (  18,   82)
//                  * (   3,   67) - (  19,   83)
//                  * (   2,   66) + (  18,   82)
//                  * (   3,   67) + (  19,   83)
//                  * (  18,   82) = @ * (   0,    1)
//                  * (  19,   83) = @ * (   0,    1)
//                  *
//                  * (  32,   96) - (  48,  112)
//                  * (  33,   97) - (  49,  113)
//                  * (  32,   96) + (  48,  112)
//                  * (  33,   97) + (  49,  113)
//                  * (  48,  112) = j@ * (   0,    1)
//                  * (  49,  113) = j@ * (   0,    1)
//                  *
//                  * (  34,   98) - (  50,  114)
//                  * (  35,   99) - (  51,  115)
//                  * (  34,   98) + (  50,  114)
//                  * (  35,   99) + (  51,  115)
//                  * (  50,  114) = j@ * (   0,    1)
//                  * (  51,  115) = j@ * (   0,    1)
//                  */
//                 // x1: 0 -> 4 | 64 -> 67
//                 // y1: 16 -> 19 | 80 -> 81
//                 // x2: 32 -> 35 | 96 -> 99
//                 // y2: 48 -> 51 | 112 -> 115
//                 vloadx2(x1_re, &f[j]);
//                 vloadx2(x1_im, &f[j + hn]);
//                 vloadx2(y1_re, &f[j + len]);
//                 vloadx2(y1_im, &f[j + len + hn]);

//                 INV_TOPJ (t_re.val[0], t_im.val[0], x1_re.val[0], x1_im.val[0], y1_re.val[0], y1_im.val[0]);
//                 INV_TOPJ (t_re.val[1], t_im.val[1], x1_re.val[1], x1_im.val[1], y1_re.val[1], y1_im.val[1]);

//                 vloadx2(x2_re, &f[j + 2*len]);
//                 vloadx2(x2_im, &f[j + 2*len + hn]);
//                 vloadx2(y2_re, &f[j + 3*len]);
//                 vloadx2(y2_im, &f[j + 3*len + hn]);

//                 INV_TOPJm(t_re.val[2], t_im.val[2], x2_re.val[0], x2_im.val[0], y2_re.val[0], y2_im.val[0]);
//                 INV_TOPJm(t_re.val[3], t_im.val[3], x2_re.val[1], x2_im.val[1], y2_re.val[1], y2_im.val[1]);

//                 INV_BOTJ_LANE (y1_re.val[0], y1_im.val[0], t_re.val[0], t_im.val[0], s1_re_im);
//                 INV_BOTJ_LANE (y1_re.val[1], y1_im.val[1], t_re.val[1], t_im.val[1], s1_re_im);

//                 INV_BOTJm_LANE(y2_re.val[0], y2_im.val[0], t_re.val[2], t_im.val[2], s1_re_im);
//                 INV_BOTJm_LANE(y2_re.val[1], y2_im.val[1], t_re.val[3], t_im.val[3], s1_re_im);
//                 /*
//                  * Level 7
//                  * (   0,   64) - (  32,   96)
//                  * (   1,   65) - (  33,   97)
//                  * (   0,   64) + (  32,   96)
//                  * (   1,   65) + (  33,   97)
//                  * (  32,   96) = @ * (   0,    1)
//                  * (  33,   97) = @ * (   0,    1)
//                  *
//                  * (   2,   66) - (  34,   98)
//                  * (   3,   67) - (  35,   99)
//                  * (   2,   66) + (  34,   98)
//                  * (   3,   67) + (  35,   99)
//                  * (  34,   98) = @ * (   0,    1)
//                  * (  35,   99) = @ * (   0,    1)
//                  * ----
//                  * (  16,   80) - (  48,  112)
//                  * (  17,   81) - (  49,  113)
//                  * (  16,   80) + (  48,  112)
//                  * (  17,   81) + (  49,  113)
//                  * (  48,  112) = @ * (   0,    1)
//                  * (  49,  113) = @ * (   0,    1)
//                  *
//                  * (  18,   82) - (  50,  114)
//                  * (  19,   83) - (  51,  115)
//                  * (  18,   82) + (  50,  114)
//                  * (  19,   83) + (  51,  115)
//                  * (  50,  114) = @ * (   0,    1)
//                  * (  51,  115) = @ * (   0,    1)
//                  */

//                 if (Jm)
//                 {
//                     INV_TOPJm(t_re.val[0], t_im.val[0], x1_re.val[0], x1_im.val[0], x2_re.val[0], x2_im.val[0]);
//                     INV_TOPJm(t_re.val[1], t_im.val[1], x1_re.val[1], x1_im.val[1], x2_re.val[1], x2_im.val[1]);

//                     INV_TOPJm(t_re.val[2], t_im.val[2], y1_re.val[0], y1_im.val[0], y2_re.val[0], y2_im.val[0]);
//                     INV_TOPJm(t_re.val[3], t_im.val[3], y1_re.val[1], y1_im.val[1], y2_re.val[1], y2_im.val[1]);

//                     INV_BOTJm_LANE(x2_re.val[0], x2_im.val[0], t_re.val[0], t_im.val[0], s2_re_im);
//                     INV_BOTJm_LANE(x2_re.val[1], x2_im.val[1], t_re.val[1], t_im.val[1], s2_re_im);
//                     INV_BOTJm_LANE(y2_re.val[0], y2_im.val[0], t_re.val[2], t_im.val[2], s2_re_im);
//                     INV_BOTJm_LANE(y2_re.val[1], y2_im.val[1], t_re.val[3], t_im.val[3], s2_re_im);
//                 }
//                 else
//                 {
//                     INV_TOPJ(t_re.val[0], t_im.val[0], x1_re.val[0], x1_im.val[0], x2_re.val[0], x2_im.val[0]);
//                     INV_TOPJ(t_re.val[1], t_im.val[1], x1_re.val[1], x1_im.val[1], x2_re.val[1], x2_im.val[1]);

//                     INV_TOPJ(t_re.val[2], t_im.val[2], y1_re.val[0], y1_im.val[0], y2_re.val[0], y2_im.val[0]);
//                     INV_TOPJ(t_re.val[3], t_im.val[3], y1_re.val[1], y1_im.val[1], y2_re.val[1], y2_im.val[1]);

//                     INV_BOTJ_LANE(x2_re.val[0], x2_im.val[0], t_re.val[0], t_im.val[0], s2_re_im);
//                     INV_BOTJ_LANE(x2_re.val[1], x2_im.val[1], t_re.val[1], t_im.val[1], s2_re_im);
//                     INV_BOTJ_LANE(y2_re.val[0], y2_im.val[0], t_re.val[2], t_im.val[2], s2_re_im);
//                     INV_BOTJ_LANE(y2_re.val[1], y2_im.val[1], t_re.val[3], t_im.val[3], s2_re_im);
//                 }

//                 vstorex2(&f[j + 2*len], x2_re);
//                 vstorex2(&f[j + 2*len + hn], x2_im);

//                 vstorex2(&f[j + 3*len], y2_re);
//                 vstorex2(&f[j + 3*len + hn], y2_im);

//                 if (!last)
//                 {
//                     vfmuln(x1_re.val[0], x1_re.val[0], fpr_p2_tab[logn]);
//                     vfmuln(x1_re.val[1], x1_re.val[1], fpr_p2_tab[logn]);
//                     vfmuln(x1_im.val[0], x1_im.val[0], fpr_p2_tab[logn]);
//                     vfmuln(x1_im.val[1], x1_im.val[1], fpr_p2_tab[logn]);

//                     vfmuln(y1_re.val[0], y1_re.val[0], fpr_p2_tab[logn]);
//                     vfmuln(y1_re.val[1], y1_re.val[1], fpr_p2_tab[logn]);
//                     vfmuln(y1_im.val[0], y1_im.val[0], fpr_p2_tab[logn]);
//                     vfmuln(y1_im.val[1], y1_im.val[1], fpr_p2_tab[logn]);
//                 }

//                 vstorex2(&f[j], x1_re);
//                 vstorex2(&f[j + hn], x1_im);

//                 vstorex2(&f[j + len], y1_re);
//                 vstorex2(&f[j + len + hn], y1_im);

//             }
//         }
//     }
// }



static
   void ZfN(iFFT_logn2)(fpr *f, const unsigned logn, const unsigned level, unsigned last)
{
 const unsigned int falcon_n = 1 << logn;
 const unsigned int hn = falcon_n >> 1;

 // Total SIMD register: 26 = 16 + 8 + 2
 float64x2x4_t t_re, t_im;                 // 8
 float64x2x2_t x1_re, x2_re, x1_im, x2_im,
     y1_re, y2_re, y1_im, y2_im; // 16
 float64x2_t s1_re_im, s2_re_im;           // 2

 const fpr *fpr_inv_tab1 = NULL, *fpr_inv_tab2 = NULL;
 unsigned l, len, start, j, k1, k2;
 unsigned bar = logn - 4;

 for (l = 4; l < logn - level - 1; l += 2)
 {
   len = 1 << l;
   last -= 1;
   fpr_inv_tab1 = fpr_table[bar--];
   fpr_inv_tab2 = fpr_table[bar--];
   k1 = 0; k2 = 0;

   for (start = 0; start < hn; start += 1 << (l + 2))
   {
     vload(s1_re_im, &fpr_inv_tab1[k1]);
     vload(s2_re_im, &fpr_inv_tab2[k2]);
     k1 += 2;
     k2 += 2 * ((start & 127) == 64);
     if (!last)
     {
       vfmuln(s2_re_im, s2_re_im, fpr_p2_tab[logn]);
     }
     for (j = start; j < start + len; j += 4)
     {
       /*
       Level 6
        * (   0,   64) - (  16,   80)
        * (   1,   65) - (  17,   81)
        * (   0,   64) + (  16,   80)
        * (   1,   65) + (  17,   81)
        * (  16,   80) = @ * (   0,    1)
        * (  17,   81) = @ * (   0,    1)
        *
        * (   2,   66) - (  18,   82)
        * (   3,   67) - (  19,   83)
        * (   2,   66) + (  18,   82)
        * (   3,   67) + (  19,   83)
        * (  18,   82) = @ * (   0,    1)
        * (  19,   83) = @ * (   0,    1)
        *
        * (  32,   96) - (  48,  112)
        * (  33,   97) - (  49,  113)
        * (  32,   96) + (  48,  112)
        * (  33,   97) + (  49,  113)
        * (  48,  112) = j@ * (   0,    1)
        * (  49,  113) = j@ * (   0,    1)
        *
        * (  34,   98) - (  50,  114)
        * (  35,   99) - (  51,  115)
        * (  34,   98) + (  50,  114)
        * (  35,   99) + (  51,  115)
        * (  50,  114) = j@ * (   0,    1)
        * (  51,  115) = j@ * (   0,    1)
        */
       // x1: 0 -> 4 | 64 -> 67
       // y1: 16 -> 19 | 80 -> 81
       // x2: 32 -> 35 | 96 -> 99
       // y2: 48 -> 51 | 112 -> 115
       vloadx2(x1_re, &f[j]);
       vloadx2(x1_im, &f[j + hn]);
       vloadx2(y1_re, &f[j + len]);
       vloadx2(y1_im, &f[j + len + hn]);

       INV_TOPJ (t_re.val[0], t_im.val[0], x1_re.val[0], x1_im.val[0], y1_re.val[0], y1_im.val[0]);
       INV_TOPJ (t_re.val[1], t_im.val[1], x1_re.val[1], x1_im.val[1], y1_re.val[1], y1_im.val[1]);

       vloadx2(x2_re, &f[j + 2*len]);
       vloadx2(x2_im, &f[j + 2*len + hn]);
       vloadx2(y2_re, &f[j + 3*len]);
       vloadx2(y2_im, &f[j + 3*len + hn]);

       INV_TOPJm(t_re.val[2], t_im.val[2], x2_re.val[0], x2_im.val[0], y2_re.val[0], y2_im.val[0]);
       INV_TOPJm(t_re.val[3], t_im.val[3], x2_re.val[1], x2_im.val[1], y2_re.val[1], y2_im.val[1]);

       INV_BOTJ_LANE (y1_re.val[0], y1_im.val[0], t_re.val[0], t_im.val[0], s1_re_im);
       INV_BOTJ_LANE (y1_re.val[1], y1_im.val[1], t_re.val[1], t_im.val[1], s1_re_im);

       INV_BOTJm_LANE(y2_re.val[0], y2_im.val[0], t_re.val[2], t_im.val[2], s1_re_im);
       INV_BOTJm_LANE(y2_re.val[1], y2_im.val[1], t_re.val[3], t_im.val[3], s1_re_im);
       /*
                * Level 7
                * (   0,   64) - (  32,   96)
                * (   1,   65) - (  33,   97)
                * (   0,   64) + (  32,   96)
                * (   1,   65) + (  33,   97)
                * (  32,   96) = @ * (   0,    1)
                * (  33,   97) = @ * (   0,    1)
                *
                * (   2,   66) - (  34,   98)
                * (   3,   67) - (  35,   99)
                * (   2,   66) + (  34,   98)
                * (   3,   67) + (  35,   99)
                * (  34,   98) = @ * (   0,    1)
                * (  35,   99) = @ * (   0,    1)
                * ----
                * (  16,   80) - (  48,  112)
                * (  17,   81) - (  49,  113)
                * (  16,   80) + (  48,  112)
                * (  17,   81) + (  49,  113)
                * (  48,  112) = @ * (   0,    1)
                * (  49,  113) = @ * (   0,    1)
                *
                * (  18,   82) - (  50,  114)
                * (  19,   83) - (  51,  115)
                * (  18,   82) + (  50,  114)
                * (  19,   83) + (  51,  115)
                * (  50,  114) = @ * (   0,    1)
                * (  51,  115) = @ * (   0,    1)
        */


       INV_TOPJ(t_re.val[0], t_im.val[0], x1_re.val[0], x1_im.val[0], x2_re.val[0], x2_im.val[0]);
       INV_TOPJ(t_re.val[1], t_im.val[1], x1_re.val[1], x1_im.val[1], x2_re.val[1], x2_im.val[1]);

       INV_TOPJ(t_re.val[2], t_im.val[2], y1_re.val[0], y1_im.val[0], y2_re.val[0], y2_im.val[0]);
       INV_TOPJ(t_re.val[3], t_im.val[3], y1_re.val[1], y1_im.val[1], y2_re.val[1], y2_im.val[1]);

       INV_BOTJ_LANE(x2_re.val[0], x2_im.val[0], t_re.val[0], t_im.val[0], s2_re_im);
       INV_BOTJ_LANE(x2_re.val[1], x2_im.val[1], t_re.val[1], t_im.val[1], s2_re_im);
       INV_BOTJ_LANE(y2_re.val[0], y2_im.val[0], t_re.val[2], t_im.val[2], s2_re_im);
       INV_BOTJ_LANE(y2_re.val[1], y2_im.val[1], t_re.val[3], t_im.val[3], s2_re_im);

       vstorex2(&f[j + 2*len], x2_re);
       vstorex2(&f[j + 2*len + hn], x2_im);

       vstorex2(&f[j + 3*len], y2_re);
       vstorex2(&f[j + 3*len + hn], y2_im);

       if (!last)
       {
         vfmuln(x1_re.val[0], x1_re.val[0], fpr_p2_tab[logn]);
         vfmuln(x1_re.val[1], x1_re.val[1], fpr_p2_tab[logn]);
         vfmuln(x1_im.val[0], x1_im.val[0], fpr_p2_tab[logn]);
         vfmuln(x1_im.val[1], x1_im.val[1], fpr_p2_tab[logn]);

         vfmuln(y1_re.val[0], y1_re.val[0], fpr_p2_tab[logn]);
         vfmuln(y1_re.val[1], y1_re.val[1], fpr_p2_tab[logn]);
         vfmuln(y1_im.val[0], y1_im.val[0], fpr_p2_tab[logn]);
         vfmuln(y1_im.val[1], y1_im.val[1], fpr_p2_tab[logn]);
       }

       vstorex2(&f[j], x1_re);
       vstorex2(&f[j + hn], x1_im);

       vstorex2(&f[j + len], y1_re);
       vstorex2(&f[j + len + hn], y1_im);

     }
     //
     start += 1 << (l + 2);
     if (start >= hn) break;

     vload(s1_re_im, &fpr_inv_tab1[k1]);
     vload(s2_re_im, &fpr_inv_tab2[k2]);
     k1 += 2;
     k2 += 2 * ((start & 127) == 64);
     if (!last)
     {
       vfmuln(s2_re_im, s2_re_im, fpr_p2_tab[logn]);
     }

     for (j = start; j < start + len; j += 4)
     {
       /*
       Level 6
        * (   0,   64) - (  16,   80)
        * (   1,   65) - (  17,   81)
        * (   0,   64) + (  16,   80)
        * (   1,   65) + (  17,   81)
        * (  16,   80) = @ * (   0,    1)
        * (  17,   81) = @ * (   0,    1)
        *
        * (   2,   66) - (  18,   82)
        * (   3,   67) - (  19,   83)
        * (   2,   66) + (  18,   82)
        * (   3,   67) + (  19,   83)
        * (  18,   82) = @ * (   0,    1)
        * (  19,   83) = @ * (   0,    1)
        *
        * (  32,   96) - (  48,  112)
        * (  33,   97) - (  49,  113)
        * (  32,   96) + (  48,  112)
        * (  33,   97) + (  49,  113)
        * (  48,  112) = j@ * (   0,    1)
        * (  49,  113) = j@ * (   0,    1)
        *
        * (  34,   98) - (  50,  114)
        * (  35,   99) - (  51,  115)
        * (  34,   98) + (  50,  114)
        * (  35,   99) + (  51,  115)
        * (  50,  114) = j@ * (   0,    1)
        * (  51,  115) = j@ * (   0,    1)
        */
       // x1: 0 -> 4 | 64 -> 67
       // y1: 16 -> 19 | 80 -> 81
       // x2: 32 -> 35 | 96 -> 99
       // y2: 48 -> 51 | 112 -> 115
       vloadx2(x1_re, &f[j]);
       vloadx2(x1_im, &f[j + hn]);
       vloadx2(y1_re, &f[j + len]);
       vloadx2(y1_im, &f[j + len + hn]);

       INV_TOPJ (t_re.val[0], t_im.val[0], x1_re.val[0], x1_im.val[0], y1_re.val[0], y1_im.val[0]);
       INV_TOPJ (t_re.val[1], t_im.val[1], x1_re.val[1], x1_im.val[1], y1_re.val[1], y1_im.val[1]);

       vloadx2(x2_re, &f[j + 2*len]);
       vloadx2(x2_im, &f[j + 2*len + hn]);
       vloadx2(y2_re, &f[j + 3*len]);
       vloadx2(y2_im, &f[j + 3*len + hn]);

       INV_TOPJm(t_re.val[2], t_im.val[2], x2_re.val[0], x2_im.val[0], y2_re.val[0], y2_im.val[0]);
       INV_TOPJm(t_re.val[3], t_im.val[3], x2_re.val[1], x2_im.val[1], y2_re.val[1], y2_im.val[1]);

       INV_BOTJ_LANE (y1_re.val[0], y1_im.val[0], t_re.val[0], t_im.val[0], s1_re_im);
       INV_BOTJ_LANE (y1_re.val[1], y1_im.val[1], t_re.val[1], t_im.val[1], s1_re_im);

       INV_BOTJm_LANE(y2_re.val[0], y2_im.val[0], t_re.val[2], t_im.val[2], s1_re_im);
       INV_BOTJm_LANE(y2_re.val[1], y2_im.val[1], t_re.val[3], t_im.val[3], s1_re_im);
       /*
                * Level 7
                * (   0,   64) - (  32,   96)
                * (   1,   65) - (  33,   97)
                * (   0,   64) + (  32,   96)
                * (   1,   65) + (  33,   97)
                * (  32,   96) = @ * (   0,    1)
                * (  33,   97) = @ * (   0,    1)
                *
                * (   2,   66) - (  34,   98)
                * (   3,   67) - (  35,   99)
                * (   2,   66) + (  34,   98)
                * (   3,   67) + (  35,   99)
                * (  34,   98) = @ * (   0,    1)
                * (  35,   99) = @ * (   0,    1)
                * ----
                * (  16,   80) - (  48,  112)
                * (  17,   81) - (  49,  113)
                * (  16,   80) + (  48,  112)
                * (  17,   81) + (  49,  113)
                * (  48,  112) = @ * (   0,    1)
                * (  49,  113) = @ * (   0,    1)
                *
                * (  18,   82) - (  50,  114)
                * (  19,   83) - (  51,  115)
                * (  18,   82) + (  50,  114)
                * (  19,   83) + (  51,  115)
                * (  50,  114) = @ * (   0,    1)
                * (  51,  115) = @ * (   0,    1)
        */

       INV_TOPJm(t_re.val[0], t_im.val[0], x1_re.val[0], x1_im.val[0], x2_re.val[0], x2_im.val[0]);
       INV_TOPJm(t_re.val[1], t_im.val[1], x1_re.val[1], x1_im.val[1], x2_re.val[1], x2_im.val[1]);

       INV_TOPJm(t_re.val[2], t_im.val[2], y1_re.val[0], y1_im.val[0], y2_re.val[0], y2_im.val[0]);
       INV_TOPJm(t_re.val[3], t_im.val[3], y1_re.val[1], y1_im.val[1], y2_re.val[1], y2_im.val[1]);

       INV_BOTJm_LANE(x2_re.val[0], x2_im.val[0], t_re.val[0], t_im.val[0], s2_re_im);
       INV_BOTJm_LANE(x2_re.val[1], x2_im.val[1], t_re.val[1], t_im.val[1], s2_re_im);
       INV_BOTJm_LANE(y2_re.val[0], y2_im.val[0], t_re.val[2], t_im.val[2], s2_re_im);
       INV_BOTJm_LANE(y2_re.val[1], y2_im.val[1], t_re.val[3], t_im.val[3], s2_re_im);

       vstorex2(&f[j + 2*len], x2_re);
       vstorex2(&f[j + 2*len + hn], x2_im);

       vstorex2(&f[j + 3*len], y2_re);
       vstorex2(&f[j + 3*len + hn], y2_im);

       if (!last)
       {
         vfmuln(x1_re.val[0], x1_re.val[0], fpr_p2_tab[logn]);
         vfmuln(x1_re.val[1], x1_re.val[1], fpr_p2_tab[logn]);
         vfmuln(x1_im.val[0], x1_im.val[0], fpr_p2_tab[logn]);
         vfmuln(x1_im.val[1], x1_im.val[1], fpr_p2_tab[logn]);

         vfmuln(y1_re.val[0], y1_re.val[0], fpr_p2_tab[logn]);
         vfmuln(y1_re.val[1], y1_re.val[1], fpr_p2_tab[logn]);
         vfmuln(y1_im.val[0], y1_im.val[0], fpr_p2_tab[logn]);
         vfmuln(y1_im.val[1], y1_im.val[1], fpr_p2_tab[logn]);
       }

       vstorex2(&f[j], x1_re);
       vstorex2(&f[j + hn], x1_im);

       vstorex2(&f[j + len], y1_re);
       vstorex2(&f[j + len + hn], y1_im);

     }
     //
   }
 }
}


/*
* Support logn from [1, 10]
* Can be easily extended to logn > 10
*/
void ZfN(iFFT)(fpr *f, const unsigned logn)
{
 const unsigned level = (logn - 5) & 1;

 switch (logn)
 {
   case 2:
     ZfN(iFFT_log2)(f);
     break;

   case 3:
     ZfN(iFFT_log3)(f);
     break;

   case 4:
     ZfN(iFFT_log4)(f);
     break;

   case 5:
     ZfN(iFFT_log5)(f, 5, 1);
     break;

   case 6:
     ZfN(iFFT_log5)(f, logn, 0);
     ZfN(iFFT_logn1)(f, logn, 1);
     break;

   case 7:
   case 9:
     ZfN(iFFT_log5)(f, logn, 0);
     ZfN(iFFT_logn2)(f, logn, level, 1);
     break;

   case 8:
   case 10:
     ZfN(iFFT_log5)(f, logn, 0);
     ZfN(iFFT_logn2)(f, logn, level, 0);
     ZfN(iFFT_logn1)(f, logn, 1);
     break;

   default:
     break;
 }
}
#endif

// generic fft stuff

#include "../commons_private.h"
#include "reim_fft_internal.h"
#include "reim_fft_private.h"

void reim_ctwiddle(double* ra, double* ia, double* rb, double* ib, double omre, double omim);
void reim_citwiddle(double* ra, double* ia, double* rb, double* ib, double omre, double omim);

void reim_fft16_ref(double* dre, double* dim, const void* pom);
void fill_reim_fft16_omegas(const double entry_pwr, double** omg);
void reim_fft8_ref(double* dre, double* dim, const void* pom);
void reim_twiddle_fft_ref(uint64_t h, double* re, double* im, double om[2]);
void fill_reim_twiddle_fft_ref(const double s, double** omg);
void reim_bitwiddle_fft_ref(uint64_t h, double* re, double* im, double om[4]);
void fill_reim_bitwiddle_fft_ref(const double s, double** omg);
void reim_fft_rec_16_ref(uint64_t m, double* re, double* im, double** omg);
void fill_reim_fft_rec_16_omegas(uint64_t m, double entry_pwr, double** omg);

void fill_reim_twiddle_fft_omegas_ref(const double rs0, double** omg) {
  (*omg)[0] = cos(2 * M_PI * rs0);
  (*omg)[1] = sin(2 * M_PI * rs0);
  *omg += 2;
}

void fill_reim_bitwiddle_fft_omegas_ref(const double rs0, double** omg) {
  double rs1 = 2. * rs0;
  (*omg)[0] = cos(2 * M_PI * rs1);
  (*omg)[1] = sin(2 * M_PI * rs1);
  (*omg)[2] = cos(2 * M_PI * rs0);
  (*omg)[3] = sin(2 * M_PI * rs0);
  *omg += 4;
}

#define reim_fft16_f reim_fft16_neon
#define reim_fft16_pom_offset 16
#define fill_reim_fft16_omegas_f fill_reim_fft16_omegas_neon
// currently, m=4 uses the ref implem (corner case with low impact) TODO!!
#define reim_fft8_f reim_fft8_ref
#define reim_fft8_pom_offset 8
#define fill_reim_fft8_omegas_f fill_reim_fft8_omegas
// currently, m=4 uses the ref implem (corner case with low impact) TODO!!
#define reim_fft4_f reim_fft4_ref
#define reim_fft4_pom_offset 4
#define fill_reim_fft4_omegas_f fill_reim_fft4_omegas
// m = 2 will use the ref implem, since intrinsics don't provide any speed-up
#define reim_fft2_f reim_fft2_ref
#define reim_fft2_pom_offset 2
#define fill_reim_fft2_omegas_f fill_reim_fft2_omegas

// neon twiddle use the same omegas layout as the ref implem
#define reim_twiddle_fft_f reim_twiddle_fft_neon
#define reim_twiddle_fft_pom_offset 2
#define fill_reim_twiddle_fft_omegas_f fill_reim_twiddle_fft_omegas_ref

// neon bi-twiddle use the same omegas layout as the ref implem
#define reim_bitwiddle_fft_f reim_bitwiddle_fft_neon
#define reim_bitwiddle_fft_pom_offset 4
#define fill_reim_bitwiddle_fft_omegas_f fill_reim_bitwiddle_fft_omegas_ref

// template functions to produce
#define reim_fft_bfs_16_f reim_fft_bfs_16_neon
#define fill_reim_fft_bfs_16_omegas_f fill_reim_fft_bfs_16_omegas_neon
#define reim_fft_rec_16_f reim_fft_rec_16_neon
#define fill_reim_fft_rec_16_omegas_f fill_reim_fft_rec_16_omegas_neon
#define reim_fft_f reim_fft_neon
#define fill_reim_fft_omegas_f fill_reim_fft_omegas_neon

#include "reim_fft_core_template.h"

EXPORT REIM_FFT_PRECOMP* new_reim_fft_precomp_neon(uint32_t m, uint32_t num_buffers) {
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
  fill_reim_fft_omegas_f(m, 0.25, &omg);
  if (((uint64_t)omg) - aligned_addr > OMG_SPACE) abort();
  // dispatch the right implementation
  //{
  //  if (CPU_SUPPORTS("fma")) {
  //    r->function = reim_fft_avx2_fma;
  //  } else {
  //    r->function = reim_fft_ref;
  //  }
  //}
  r->function = reim_fft_f;
  return reps;
}
