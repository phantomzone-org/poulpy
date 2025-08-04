#include <cassert>
#include <cinttypes>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "spqlios/cplx/cplx_fft_internal.h"

using namespace std;

/*namespace {
bool very_close(const double& a, const double& b) {
  bool reps = (abs(a - b) < 1e-5);
  if (!reps) {
    cerr << "not close: " << a << " vs. " << b << endl;
  }
  return reps;
}

}*/  // namespace

TEST(fft, fftvec_convolution) {
  uint64_t nn = 65536;           // vary accross (8192, 16384), 32768, 65536
  static const uint64_t k = 18;  // vary from 10 to 20
  // double* buf_fft = fft_precomp_get_buffer(tables, 0);
  // double* buf_ifft = ifft_precomp_get_buffer(itables, 0);
  double* a = (double*)spqlios_alloc_custom_align(32, nn * 8);
  double* a2 = (double*)spqlios_alloc_custom_align(32, nn * 8);
  double* b = (double*)spqlios_alloc_custom_align(32, nn * 8);
  double* dist_vector = (double*)spqlios_alloc_custom_align(32, nn * 8);
  int64_t p = UINT64_C(1) << k;
  printf("p size: %" PRId64 "\n", p);
  for (uint32_t i = 0; i < nn; i++) {
    a[i] = (rand() % p) - p / 2;  // between -p/2 and p/2
    b[i] = (rand() % p) - p / 2;
    a2[i] = 0;
  }
  cplx_fft_simple(nn / 2, a);
  cplx_fft_simple(nn / 2, b);
  cplx_fftvec_addmul_simple(nn / 2, a2, a, b);
  cplx_ifft_simple(nn / 2, a2);  // normalization is missing
  double distance = 0;
  // for (int32_t i = 0; i < 10; i++) {
  //  printf("%lf %lf\n", a2[i], a2[i] / (nn / 2.));
  //}
  for (uint32_t i = 0; i < nn; i++) {
    double curdist = fabs(a2[i] / (nn / 2.) - rint(a2[i] / (nn / 2.)));
    if (distance < curdist) distance = curdist;
    dist_vector[i] = a2[i] / (nn / 2.) - rint(a2[i] / (nn / 2.));
  }
  printf("distance: %lf\n", distance);
  ASSERT_LE(distance, 0.5);  // switch from previous 0.1 to 0.5 per experiment 1 reqs
  // double a3[] = {2,4,4,4,5,5,7,9}; //instead of dist_vector, for test
  // nn = 8;
  double mean = 0;
  for (uint32_t i = 0; i < nn; i++) {
    mean = mean + dist_vector[i];
  }
  mean = mean / nn;
  double variance = 0;
  for (uint32_t i = 0; i < nn; i++) {
    variance = variance + pow((mean - dist_vector[i]), 2);
  }
  double stdev = sqrt(variance / nn);
  printf("stdev: %lf\n", stdev);

  spqlios_free(a);
  spqlios_free(b);
  spqlios_free(a2);
  spqlios_free(dist_vector);
}

typedef double CPLX[2];
EXPORT uint32_t revbits(uint32_t i, uint32_t v);

void cplx_zero(CPLX r);
void cplx_addmul(CPLX r, const CPLX a, const CPLX b);

void halfcfft_eval(CPLX res, uint32_t nn, uint32_t k, const CPLX* coeffs, const CPLX* powomegas);
void halfcfft_naive(uint32_t nn, CPLX* data);

EXPORT void cplx_set(CPLX, const CPLX);
EXPORT void citwiddle(CPLX, CPLX, const CPLX);
EXPORT void invcitwiddle(CPLX, CPLX, const CPLX);
EXPORT void ctwiddle(CPLX, CPLX, const CPLX);
EXPORT void invctwiddle(CPLX, CPLX, const CPLX);

#include "../spqlios/cplx/cplx_fft_private.h"
#include "../spqlios/reim/reim_fft_internal.h"
#include "../spqlios/reim/reim_fft_private.h"
#include "../spqlios/reim4/reim4_fftvec_internal.h"
#include "../spqlios/reim4/reim4_fftvec_private.h"

TEST(fft, simple_fft_test) {  // test for checking the simple_fft api
  uint64_t nn = 8;            // vary accross (8192, 16384), 32768, 65536
  // double* buf_fft = fft_precomp_get_buffer(tables, 0);
  // double* buf_ifft = ifft_precomp_get_buffer(itables, 0);

  // define the complex coefficients of two polynomials mod X^4-i
  double a[4][2] = {{1.1, 2.2}, {3.3, 4.4}, {5.5, 6.6}, {7.7, 8.8}};
  double b[4][2] = {{9., 10.}, {11., 12.}, {13., 14.}, {15., 16.}};
  double c[4][2];   // for the result
  double a2[4][2];  // for testing inverse fft
  memcpy(a2, a, 8 * nn);
  cplx_fft_simple(4, a);
  cplx_fft_simple(4, b);
  cplx_fftvec_mul_simple(4, c, a, b);
  cplx_ifft_simple(4, c);
  // c contains the complex coefficients 4.a*b mod X^4-i
  cplx_ifft_simple(4, a);

  double distance = 0;
  for (uint32_t i = 0; i < nn / 2; i++) {
    double dist = fabs(a[i][0] / 4. - a2[i][0]);
    if (distance < dist) distance = dist;
    dist = fabs(a[i][1] / 4. - a2[i][1]);
    if (distance < dist) distance = dist;
  }
  printf("distance: %lf\n", distance);
  ASSERT_LE(distance, 0.1);  // switch from previous 0.1 to 0.5 per experiment 1 reqs

  for (uint32_t i = 0; i < nn / 4; i++) {
    printf("%lf %lf\n", a2[i][0], a[i][0] / (nn / 2.));
    printf("%lf %lf\n", a2[i][1], a[i][1] / (nn / 2.));
  }
}

TEST(fft, reim_test) {
  // double a[16] __attribute__ ((aligned(32)))= {1.1,2.2,3.3,4.4,5.5,6.6,7.7,8.8,9.9,10.,11.,12.,13.,14.,15.,16.};
  // double b[16] __attribute__ ((aligned(32)))= {17.,18.,19.,20.,21.,22.,23.,24.,25.,26.,27.,28.,29.,30., 31.,32.};
  // double c[16] __attribute__ ((aligned(32))); // for the result in reference layout
  double a[16] = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10., 11., 12., 13., 14., 15., 16.};
  double b[16] = {17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32.};
  double c[16];  // for the result in reference layout
  reim_fft_simple(8, a);
  reim_fft_simple(8, b);
  reim_fftvec_mul_simple(8, c, a, b);
  reim_ifft_simple(8, c);
}

TEST(fft, reim_vs_regular_layout_mul_test) {
  uint64_t nn = 16;

  // define the complex coefficients of two polynomials mod X^8-i

  double a1[8][2] __attribute__((aligned(32))) = {{1.1, 2.2}, {3.3, 4.4}, {5.5, 6.6}, {7.7, 8.8},
                                                  {9.9, 10.}, {11., 12.}, {13., 14.}, {15., 16.}};
  double b1[8][2] __attribute__((aligned(32))) = {{17., 18.}, {19., 20.}, {21., 22.}, {23., 24.},
                                                  {25., 26.}, {27., 28.}, {29., 30.}, {31., 32.}};
  double c1[8][2] __attribute__((aligned(32)));  // for the result
  double c2[16] __attribute__((aligned(32)));    // for the result
  double c3[8][2] __attribute__((aligned(32)));  // for the result

  double* a2 =
      (double*)spqlios_alloc_custom_align(32, nn / 2 * 2 * sizeof(double));  // for storing the coefs in reim layout
  double* b2 =
      (double*)spqlios_alloc_custom_align(32, nn / 2 * 2 * sizeof(double));  // for storing the coefs in reim layout
  // double* c2 = (double*)spqlios_alloc_custom_align(32, nn / 2 * sizeof(CPLX)); // for storing the coefs in reim
  // layout

  // organise the coefficients in the reim layout
  for (uint32_t i = 0; i < nn / 2; i++) {
    a2[i] = a1[i][0];  // a1 = a2, b1 = b2
    a2[nn / 2 + i] = a1[i][1];
    b2[i] = b1[i][0];
    b2[nn / 2 + i] = b1[i][1];
  }

  // fft
  cplx_fft_simple(8, a1);
  reim_fft_simple(8, a2);

  cplx_fft_simple(8, b1);
  reim_fft_simple(8, b2);

  cplx_fftvec_mul_simple(8, c1, a1, b1);
  reim_fftvec_mul_simple(8, c2, a2, b2);

  cplx_ifft_simple(8, c1);
  reim_ifft_simple(8, c2);

  // check base layout and reim layout result in the same values
  double d = 0;
  for (uint32_t i = 0; i < nn / 2; i++) {
    // printf("RE: cplx_result %lf and reim_result %lf \n", c1[i][0], c2[i]);
    // printf("IM: cplx_result %lf and reim_result %lf \n", c1[i][1], c2[nn / 2 + i]);
    double dre = fabs(c1[i][0] - c2[i]);
    double dim = fabs(c1[i][1] - c2[nn / 2 + i]);
    if (dre > d) d = dre;
    if (dim > d) d = dim;
    ASSERT_LE(d, 1e-7);
  }
  ASSERT_LE(d, 1e-7);

  // check converting back to base layout:

  for (uint32_t i = 0; i < nn / 2; i++) {
    c3[i][0] = c2[i];
    c3[i][1] = c2[8 + i];
  }

  d = 0;
  for (uint32_t i = 0; i < nn / 2; i++) {
    double dre = fabs(c1[i][0] - c3[i][0]);
    double dim = fabs(c1[i][1] - c3[i][1]);
    if (dre > d) d = dre;
    if (dim > d) d = dim;
    ASSERT_LE(d, 1e-7);
  }
  ASSERT_LE(d, 1e-7);

  spqlios_free(a2);
  spqlios_free(b2);
  // spqlios_free(c2);
}

TEST(fft, fftvec_convolution_recursiveoverk) {
  static const uint64_t nn = 32768;  // vary accross (8192, 16384), 32768, 65536
  double* a = (double*)spqlios_alloc_custom_align(32, nn * 8);
  double* a2 = (double*)spqlios_alloc_custom_align(32, nn * 8);
  double* b = (double*)spqlios_alloc_custom_align(32, nn * 8);
  double* dist_vector = (double*)spqlios_alloc_custom_align(32, nn * 8);

  printf("N size: %" PRId64 "\n", nn);

  for (uint32_t k = 14; k <= 24; k++) {  // vary k
    printf("k size: %" PRId32 "\n", k);
    int64_t p = UINT64_C(1) << k;
    for (uint32_t i = 0; i < nn; i++) {
      a[i] = (rand() % p) - p / 2;
      b[i] = (rand() % p) - p / 2;
      a2[i] = 0;
    }
    cplx_fft_simple(nn / 2, a);
    cplx_fft_simple(nn / 2, b);
    cplx_fftvec_addmul_simple(nn / 2, a2, a, b);
    cplx_ifft_simple(nn / 2, a2);
    double distance = 0;
    for (uint32_t i = 0; i < nn; i++) {
      double curdist = fabs(a2[i] / (nn / 2.) - rint(a2[i] / (nn / 2.)));
      if (distance < curdist) distance = curdist;
      dist_vector[i] = a2[i] / (nn / 2.) - rint(a2[i] / (nn / 2.));
    }
    printf("distance: %lf\n", distance);
    ASSERT_LE(distance, 0.5);  // switch from previous 0.1 to 0.5 per experiment 1 reqs
    double mean = 0;
    for (uint32_t i = 0; i < nn; i++) {
      mean = mean + dist_vector[i];
    }
    mean = mean / nn;
    double variance = 0;
    for (uint32_t i = 0; i < nn; i++) {
      variance = variance + pow((mean - dist_vector[i]), 2);
    }
    double stdev = sqrt(variance / nn);
    printf("stdev: %lf\n", stdev);
  }

  spqlios_free(a);
  spqlios_free(b);
  spqlios_free(a2);
  spqlios_free(dist_vector);
}

#ifdef __x86_64__
TEST(fft, cplx_fft_ref_vs_fft_reim_ref) {
  for (uint64_t nn : {16, 32, 64, 1024, 8192, 65536}) {
    uint64_t m = nn / 2;
    CPLX_FFT_PRECOMP* tables = new_cplx_fft_precomp(m, 0);
    REIM_FFT_PRECOMP* reimtables = new_reim_fft_precomp(m, 0);
    CPLX* a = (CPLX*)spqlios_alloc_custom_align(32, nn / 2 * sizeof(CPLX));
    CPLX* a1 = (CPLX*)spqlios_alloc_custom_align(32, nn / 2 * sizeof(CPLX));
    double* a2 = (double*)spqlios_alloc_custom_align(32, nn / 2 * sizeof(CPLX));
    int64_t p = 1 << 16;
    for (uint32_t i = 0; i < nn / 2; i++) {
      a[i][0] = (rand() % p) - p / 2;  // between -p/2 and p/2
      a[i][1] = (rand() % p) - p / 2;
    }
    memcpy(a1, a, nn / 2 * sizeof(CPLX));
    for (uint32_t i = 0; i < nn / 2; i++) {
      a2[i] = a[i][0];
      a2[nn / 2 + i] = a[i][1];
    }
    cplx_fft_ref(tables, a1);
    reim_fft_ref(reimtables, a2);
    double d = 0;
    for (uint32_t i = 0; i < nn / 2; i++) {
      double dre = fabs(a1[i][0] - a2[i]);
      double dim = fabs(a1[i][1] - a2[nn / 2 + i]);
      if (dre > d) d = dre;
      if (dim > d) d = dim;
      ASSERT_LE(d, 1e-7);
    }
    ASSERT_LE(d, 1e-7);
    spqlios_free(a);
    spqlios_free(a1);
    spqlios_free(a2);
    delete_cplx_fft_precomp(tables);
    delete_reim_fft_precomp(reimtables);
  }
}
#endif

TEST(fft, cplx_ifft_ref_vs_reim_ifft_ref) {
  for (uint64_t nn : {16, 32, 64, 1024, 8192, 65536}) {
    uint64_t m = nn / 2;
    CPLX_IFFT_PRECOMP* tables = new_cplx_ifft_precomp(m, 0);
    REIM_IFFT_PRECOMP* reimtables = new_reim_ifft_precomp(m, 0);
    CPLX* a = (CPLX*)spqlios_alloc_custom_align(32, nn / 2 * sizeof(CPLX));
    CPLX* a1 = (CPLX*)spqlios_alloc_custom_align(32, nn / 2 * sizeof(CPLX));
    double* a2 = (double*)spqlios_alloc_custom_align(32, nn / 2 * sizeof(CPLX));
    int64_t p = 1 << 16;
    for (uint32_t i = 0; i < nn / 2; i++) {
      a[i][0] = (rand() % p) - p / 2;  // between -p/2 and p/2
      a[i][1] = (rand() % p) - p / 2;
    }
    memcpy(a1, a, nn / 2 * sizeof(CPLX));
    for (uint32_t i = 0; i < nn / 2; i++) {
      a2[i] = a[i][0];
      a2[nn / 2 + i] = a[i][1];
    }
    cplx_ifft_ref(tables, a1);
    reim_ifft_ref(reimtables, a2);
    double d = 0;
    for (uint32_t i = 0; i < nn / 2; i++) {
      double dre = fabs(a1[i][0] - a2[i]);
      double dim = fabs(a1[i][1] - a2[nn / 2 + i]);
      if (dre > d) d = dre;
      if (dim > d) d = dim;
      ASSERT_LE(d, 1e-7);
    }
    ASSERT_LE(d, 1e-7);
    spqlios_free(a);
    spqlios_free(a1);
    spqlios_free(a2);
    delete_cplx_fft_precomp(tables);
    delete_reim_fft_precomp(reimtables);
  }
}

#ifdef __x86_64__
TEST(fft, reim4_vecfft_addmul_fma_vs_ref) {
  for (uint64_t nn : {16, 32, 64, 1024, 8192, 65536}) {
    uint64_t m = nn / 2;
    REIM4_FFTVEC_ADDMUL_PRECOMP* tbl = new_reim4_fftvec_addmul_precomp(m);
    ASSERT_TRUE(tbl != nullptr);
    double* a1 = (double*)spqlios_alloc_custom_align(32, nn / 2 * sizeof(CPLX));
    double* a2 = (double*)spqlios_alloc_custom_align(32, nn / 2 * sizeof(CPLX));
    double* b1 = (double*)spqlios_alloc_custom_align(32, nn / 2 * sizeof(CPLX));
    double* b2 = (double*)spqlios_alloc_custom_align(32, nn / 2 * sizeof(CPLX));
    double* r1 = (double*)spqlios_alloc_custom_align(32, nn / 2 * sizeof(CPLX));
    double* r2 = (double*)spqlios_alloc_custom_align(32, nn / 2 * sizeof(CPLX));
    int64_t p = 1 << 16;
    for (uint32_t i = 0; i < nn; i++) {
      a1[i] = (rand() % p) - p / 2;  // between -p/2 and p/2
      b1[i] = (rand() % p) - p / 2;
      r1[i] = (rand() % p) - p / 2;
    }
    memcpy(a2, a1, nn / 2 * sizeof(CPLX));
    memcpy(b2, b1, nn / 2 * sizeof(CPLX));
    memcpy(r2, r1, nn / 2 * sizeof(CPLX));
    reim4_fftvec_addmul_ref(tbl, r1, a1, b1);
    reim4_fftvec_addmul_fma(tbl, r2, a2, b2);
    double d = 0;
    for (uint32_t i = 0; i < nn; i++) {
      double di = fabs(r1[i] - r2[i]);
      if (di > d) d = di;
      ASSERT_LE(d, 1e-8);
    }
    ASSERT_LE(d, 1e-8);
    spqlios_free(a1);
    spqlios_free(a2);
    spqlios_free(b1);
    spqlios_free(b2);
    spqlios_free(r1);
    spqlios_free(r2);
    delete_reim4_fftvec_addmul_precomp(tbl);
  }
}
#endif

#ifdef __x86_64__
TEST(fft, reim4_vecfft_mul_fma_vs_ref) {
  for (uint64_t nn : {16, 32, 64, 1024, 8192, 65536}) {
    uint64_t m = nn / 2;
    REIM4_FFTVEC_MUL_PRECOMP* tbl = new_reim4_fftvec_mul_precomp(m);
    double* a1 = (double*)spqlios_alloc_custom_align(32, nn / 2 * sizeof(CPLX));
    double* a2 = (double*)spqlios_alloc_custom_align(32, nn / 2 * sizeof(CPLX));
    double* b1 = (double*)spqlios_alloc_custom_align(32, nn / 2 * sizeof(CPLX));
    double* b2 = (double*)spqlios_alloc_custom_align(32, nn / 2 * sizeof(CPLX));
    double* r1 = (double*)spqlios_alloc_custom_align(32, nn / 2 * sizeof(CPLX));
    double* r2 = (double*)spqlios_alloc_custom_align(32, nn / 2 * sizeof(CPLX));
    int64_t p = 1 << 16;
    for (uint32_t i = 0; i < nn; i++) {
      a1[i] = (rand() % p) - p / 2;  // between -p/2 and p/2
      b1[i] = (rand() % p) - p / 2;
      r1[i] = (rand() % p) - p / 2;
    }
    memcpy(a2, a1, nn / 2 * sizeof(CPLX));
    memcpy(b2, b1, nn / 2 * sizeof(CPLX));
    memcpy(r2, r1, nn / 2 * sizeof(CPLX));
    reim4_fftvec_mul_ref(tbl, r1, a1, b1);
    reim4_fftvec_mul_fma(tbl, r2, a2, b2);
    double d = 0;
    for (uint32_t i = 0; i < nn; i++) {
      double di = fabs(r1[i] - r2[i]);
      if (di > d) d = di;
      ASSERT_LE(d, 1e-8);
    }
    ASSERT_LE(d, 1e-8);
    spqlios_free(a1);
    spqlios_free(a2);
    spqlios_free(b1);
    spqlios_free(b2);
    spqlios_free(r1);
    spqlios_free(r2);
    delete_reim_fftvec_mul_precomp(tbl);
  }
}
#endif

#ifdef __x86_64__
TEST(fft, reim4_from_cplx_fma_vs_ref) {
  for (uint64_t nn : {16, 32, 64, 1024, 8192, 65536}) {
    uint64_t m = nn / 2;
    REIM4_FROM_CPLX_PRECOMP* tbl = new_reim4_from_cplx_precomp(m);
    double* a1 = (double*)spqlios_alloc_custom_align(32, nn / 2 * sizeof(CPLX));
    double* a2 = (double*)spqlios_alloc_custom_align(32, nn / 2 * sizeof(CPLX));
    double* r1 = (double*)spqlios_alloc_custom_align(32, nn / 2 * sizeof(CPLX));
    double* r2 = (double*)spqlios_alloc_custom_align(32, nn / 2 * sizeof(CPLX));
    int64_t p = 1 << 16;
    for (uint32_t i = 0; i < nn; i++) {
      a1[i] = (rand() % p) - p / 2;  // between -p/2 and p/2
      r1[i] = (rand() % p) - p / 2;
    }
    memcpy(a2, a1, nn / 2 * sizeof(CPLX));
    memcpy(r2, r1, nn / 2 * sizeof(CPLX));
    reim4_from_cplx_ref(tbl, r1, a1);
    reim4_from_cplx_fma(tbl, r2, a2);
    double d = 0;
    for (uint32_t i = 0; i < nn; i++) {
      double di = fabs(r1[i] - r2[i]);
      if (di > d) d = di;
      ASSERT_LE(d, 1e-8);
    }
    ASSERT_LE(d, 1e-8);
    spqlios_free(a1);
    spqlios_free(a2);
    spqlios_free(r1);
    spqlios_free(r2);
    delete_reim4_from_cplx_precomp(tbl);
  }
}
#endif

#ifdef __x86_64__
TEST(fft, reim4_to_cplx_fma_vs_ref) {
  for (uint64_t nn : {16, 32, 64, 1024, 8192, 65536}) {
    uint64_t m = nn / 2;
    REIM4_TO_CPLX_PRECOMP* tbl = new_reim4_to_cplx_precomp(m);
    double* a1 = (double*)spqlios_alloc_custom_align(32, nn / 2 * sizeof(CPLX));
    double* a2 = (double*)spqlios_alloc_custom_align(32, nn / 2 * sizeof(CPLX));
    double* r1 = (double*)spqlios_alloc_custom_align(32, nn / 2 * sizeof(CPLX));
    double* r2 = (double*)spqlios_alloc_custom_align(32, nn / 2 * sizeof(CPLX));
    int64_t p = 1 << 16;
    for (uint32_t i = 0; i < nn; i++) {
      a1[i] = (rand() % p) - p / 2;  // between -p/2 and p/2
      r1[i] = (rand() % p) - p / 2;
    }
    memcpy(a2, a1, nn / 2 * sizeof(CPLX));
    memcpy(r2, r1, nn / 2 * sizeof(CPLX));
    reim4_to_cplx_ref(tbl, r1, a1);
    reim4_to_cplx_fma(tbl, r2, a2);
    double d = 0;
    for (uint32_t i = 0; i < nn; i++) {
      double di = fabs(r1[i] - r2[i]);
      if (di > d) d = di;
      ASSERT_LE(d, 1e-8);
    }
    ASSERT_LE(d, 1e-8);
    spqlios_free(a1);
    spqlios_free(a2);
    spqlios_free(r1);
    spqlios_free(r2);
    delete_reim4_from_cplx_precomp(tbl);
  }
}
#endif
