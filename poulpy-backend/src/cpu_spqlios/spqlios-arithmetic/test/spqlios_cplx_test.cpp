#include <cmath>

#include "gtest/gtest.h"
#include "spqlios/commons_private.h"
#include "spqlios/cplx/cplx_fft.h"
#include "spqlios/cplx/cplx_fft_internal.h"
#include "spqlios/cplx/cplx_fft_private.h"

#ifdef __x86_64__
TEST(fft, ifft16_fma_vs_ref) {
  CPLX data[16];
  CPLX omega[8];
  for (uint64_t i = 0; i < 32; ++i) ((double*)data)[i] = 2 * i + 1;  //(rand()%100)-50;
  for (uint64_t i = 0; i < 16; ++i) ((double*)omega)[i] = i + 1;     //(rand()%100)-50;
  CPLX copydata[16];
  CPLX copyomega[8];
  memcpy(copydata, data, sizeof(copydata));
  memcpy(copyomega, omega, sizeof(copyomega));
  cplx_ifft16_avx_fma(data, omega);
  cplx_ifft16_ref(copydata, copyomega);
  double distance = 0;
  for (uint64_t i = 0; i < 16; ++i) {
    double d1 = fabs(data[i][0] - copydata[i][0]);
    double d2 = fabs(data[i][0] - copydata[i][0]);
    if (d1 > distance) distance = d1;
    if (d2 > distance) distance = d2;
  }
  /*
  printf("data:\n");
  for (uint64_t i=0; i<4; ++i) {
    for (uint64_t j=0; j<8; ++j) {
      printf("%.5lf ", data[4 * i + j / 2][j % 2]);
    }
    printf("\n");
  }
  printf("copydata:\n");
  for (uint64_t i=0; i<4; ++i) {
    for (uint64_t j=0; j<8; ++j) {
      printf("%5.5lf ", copydata[4 * i + j / 2][j % 2]);
    }
    printf("\n");
  }
  */
  ASSERT_EQ(distance, 0);
}

#endif

void cplx_zero(CPLX r) { r[0] = r[1] = 0; }
void cplx_addmul(CPLX r, const CPLX a, const CPLX b) {
  double re = r[0] + a[0] * b[0] - a[1] * b[1];
  double im = r[1] + a[0] * b[1] + a[1] * b[0];
  r[0] = re;
  r[1] = im;
}

void halfcfft_eval(CPLX res, uint32_t nn, uint32_t k, const CPLX* coeffs, const CPLX* powomegas) {
  const uint32_t N = nn / 2;
  cplx_zero(res);
  for (uint64_t i = 0; i < N; ++i) {
    cplx_addmul(res, coeffs[i], powomegas[(k * i) % (2 * nn)]);
  }
}
void halfcfft_naive(uint32_t nn, CPLX* data) {
  const uint32_t N = nn / 2;
  CPLX* in = (CPLX*)malloc(N * sizeof(CPLX));
  CPLX* powomega = (CPLX*)malloc(2 * nn * sizeof(CPLX));
  for (uint64_t i = 0; i < (2 * nn); ++i) {
    powomega[i][0] = m_accurate_cos((M_PI * i) / nn);
    powomega[i][1] = m_accurate_sin((M_PI * i) / nn);
  }
  memcpy(in, data, N * sizeof(CPLX));
  for (uint64_t j = 0; j < N; ++j) {
    uint64_t p = rint(log2(N)) + 2;
    uint64_t k = revbits(p, j) + 1;
    halfcfft_eval(data[j], nn, k, in, powomega);
  }
  free(powomega);
  free(in);
}

#ifdef __x86_64__
TEST(fft, fft16_fma_vs_ref) {
  CPLX data[16];
  CPLX omega[8];
  for (uint64_t i = 0; i < 32; ++i) ((double*)data)[i] = rand() % 1000;
  for (uint64_t i = 0; i < 16; ++i) ((double*)omega)[i] = rand() % 1000;
  CPLX copydata[16];
  CPLX copyomega[8];
  memcpy(copydata, data, sizeof(copydata));
  memcpy(copyomega, omega, sizeof(copyomega));
  cplx_fft16_avx_fma(data, omega);
  cplx_fft16_ref(copydata, omega);
  double distance = 0;
  for (uint64_t i = 0; i < 16; ++i) {
    double d1 = fabs(data[i][0] - copydata[i][0]);
    double d2 = fabs(data[i][0] - copydata[i][0]);
    if (d1 > distance) distance = d1;
    if (d2 > distance) distance = d2;
  }
  ASSERT_EQ(distance, 0);
}
#endif

TEST(fft, citwiddle_then_invcitwiddle) {
  CPLX om;
  CPLX ombar;
  CPLX data[2];
  CPLX copydata[2];
  om[0] = cos(3);
  om[1] = sin(3);
  ombar[0] = om[0];
  ombar[1] = -om[1];
  data[0][0] = 47;
  data[0][1] = 23;
  data[1][0] = -12;
  data[1][1] = -9;
  memcpy(copydata, data, sizeof(copydata));
  citwiddle(data[0], data[1], om);
  invcitwiddle(data[0], data[1], ombar);
  double distance = 0;
  for (uint64_t i = 0; i < 2; ++i) {
    double d1 = fabs(data[i][0] - 2 * copydata[i][0]);
    double d2 = fabs(data[i][1] - 2 * copydata[i][1]);
    if (d1 > distance) distance = d1;
    if (d2 > distance) distance = d2;
  }
  ASSERT_LE(distance, 1e-9);
}

TEST(fft, ctwiddle_then_invctwiddle) {
  CPLX om;
  CPLX ombar;
  CPLX data[2];
  CPLX copydata[2];
  om[0] = cos(3);
  om[1] = sin(3);
  ombar[0] = om[0];
  ombar[1] = -om[1];
  data[0][0] = 47;
  data[0][1] = 23;
  data[1][0] = -12;
  data[1][1] = -9;
  memcpy(copydata, data, sizeof(copydata));
  ctwiddle(data[0], data[1], om);
  invctwiddle(data[0], data[1], ombar);
  double distance = 0;
  for (uint64_t i = 0; i < 2; ++i) {
    double d1 = fabs(data[i][0] - 2 * copydata[i][0]);
    double d2 = fabs(data[i][1] - 2 * copydata[i][1]);
    if (d1 > distance) distance = d1;
    if (d2 > distance) distance = d2;
  }
  ASSERT_LE(distance, 1e-9);
}

TEST(fft, fft16_then_ifft16_ref) {
  CPLX full_omegas[64];
  CPLX full_omegabars[64];
  for (uint64_t i = 0; i < 64; ++i) {
    full_omegas[i][0] = cos(M_PI * i / 32.);
    full_omegas[i][1] = sin(M_PI * i / 32.);
    full_omegabars[i][0] = full_omegas[i][0];
    full_omegabars[i][1] = -full_omegas[i][1];
  }
  CPLX omega[8];
  CPLX omegabar[8];
  cplx_set(omega[0], full_omegas[8]);         // j
  cplx_set(omega[1], full_omegas[4]);         // k
  cplx_set(omega[2], full_omegas[2]);         // l
  cplx_set(omega[3], full_omegas[10]);        // lj
  cplx_set(omega[4], full_omegas[1]);         // n
  cplx_set(omega[5], full_omegas[9]);         // nj
  cplx_set(omega[6], full_omegas[5]);         // nk
  cplx_set(omega[7], full_omegas[13]);        // njk
  cplx_set(omegabar[0], full_omegabars[1]);   // n
  cplx_set(omegabar[1], full_omegabars[9]);   // nj
  cplx_set(omegabar[2], full_omegabars[5]);   // nk
  cplx_set(omegabar[3], full_omegabars[13]);  // njk
  cplx_set(omegabar[4], full_omegabars[2]);   // l
  cplx_set(omegabar[5], full_omegabars[10]);  // lj
  cplx_set(omegabar[6], full_omegabars[4]);   // k
  cplx_set(omegabar[7], full_omegabars[8]);   // j
  CPLX data[16];
  CPLX copydata[16];
  for (uint64_t i = 0; i < 32; ++i) ((double*)data)[i] = rand() % 1000;
  memcpy(copydata, data, sizeof(copydata));
  cplx_fft16_ref(data, omega);
  cplx_ifft16_ref(data, omegabar);
  double distance = 0;
  for (uint64_t i = 0; i < 16; ++i) {
    double d1 = fabs(data[i][0] - 16 * copydata[i][0]);
    double d2 = fabs(data[i][0] - 16 * copydata[i][0]);
    if (d1 > distance) distance = d1;
    if (d2 > distance) distance = d2;
  }
  ASSERT_LE(distance, 1e-9);
}

TEST(fft, halfcfft_ref_vs_naive) {
  for (uint64_t nn : {4, 8, 16, 64, 256, 8192}) {
    uint64_t m = nn / 2;
    CPLX_FFT_PRECOMP* tables = new_cplx_fft_precomp(m, 0);
    CPLX* a = (CPLX*)spqlios_alloc_custom_align(32, m * sizeof(CPLX));
    CPLX* a1 = (CPLX*)spqlios_alloc_custom_align(32, m * sizeof(CPLX));
    CPLX* a2 = (CPLX*)spqlios_alloc_custom_align(32, m * sizeof(CPLX));
    int64_t p = 1 << 16;
    for (uint32_t i = 0; i < m; i++) {
      a[i][0] = (rand() % p) - p / 2;  // between -p/2 and p/2
      a[i][1] = (rand() % p) - p / 2;
    }
    memcpy(a1, a, m * sizeof(CPLX));
    memcpy(a2, a, m * sizeof(CPLX));

    halfcfft_naive(nn, a1);
    cplx_fft_naive(m, 0.25, a2);

    double d = 0;
    for (uint32_t i = 0; i < nn / 2; i++) {
      double dre = fabs(a1[i][0] - a2[i][0]);
      double dim = fabs(a1[i][1] - a2[i][1]);
      if (dre > d) d = dre;
      if (dim > d) d = dim;
    }
    ASSERT_LE(d, nn * 1e-10) << nn;
    spqlios_free(a);
    spqlios_free(a1);
    spqlios_free(a2);
    delete_cplx_fft_precomp(tables);
  }
}

#ifdef __x86_64__
TEST(fft, halfcfft_fma_vs_ref) {
  typedef void (*FFTF)(const CPLX_FFT_PRECOMP*, void* data);
  for (FFTF fft : {cplx_fft_ref, cplx_fft_avx2_fma}) {
    for (uint64_t nn : {8, 16, 32, 64, 1024, 8192, 65536}) {
      uint64_t m = nn / 2;
      CPLX_FFT_PRECOMP* tables = new_cplx_fft_precomp(m, 0);
      CPLX* a = (CPLX*)spqlios_alloc_custom_align(32, nn / 2 * sizeof(CPLX));
      CPLX* a1 = (CPLX*)spqlios_alloc_custom_align(32, nn / 2 * sizeof(CPLX));
      CPLX* a2 = (CPLX*)spqlios_alloc_custom_align(32, nn / 2 * sizeof(CPLX));
      int64_t p = 1 << 16;
      for (uint32_t i = 0; i < nn / 2; i++) {
        a[i][0] = (rand() % p) - p / 2;  // between -p/2 and p/2
        a[i][1] = (rand() % p) - p / 2;
      }
      memcpy(a1, a, nn / 2 * sizeof(CPLX));
      memcpy(a2, a, nn / 2 * sizeof(CPLX));
      cplx_fft_naive(m, 0.25, a2);
      fft(tables, a1);
      double d = 0;
      for (uint32_t i = 0; i < nn / 2; i++) {
        double dre = fabs(a1[i][0] - a2[i][0]);
        double dim = fabs(a1[i][1] - a2[i][1]);
        if (dre > d) d = dre;
        if (dim > d) d = dim;
      }
      ASSERT_LE(d, nn * 1e-10) << nn;
      spqlios_free(a);
      spqlios_free(a1);
      spqlios_free(a2);
      delete_cplx_fft_precomp(tables);
    }
  }
}
#endif

TEST(fft, halfcfft_then_ifft_ref) {
  for (uint64_t nn : {4, 8, 16, 32, 64, 1024, 8192, 65536}) {
    uint64_t m = nn / 2;
    CPLX_FFT_PRECOMP* tables = new_cplx_fft_precomp(m, 0);
    CPLX_IFFT_PRECOMP* itables = new_cplx_ifft_precomp(m, 0);
    CPLX* a = (CPLX*)spqlios_alloc_custom_align(32, nn / 2 * sizeof(CPLX));
    CPLX* a1 = (CPLX*)spqlios_alloc_custom_align(32, nn / 2 * sizeof(CPLX));
    int64_t p = 1 << 16;
    for (uint32_t i = 0; i < nn / 2; i++) {
      a[i][0] = (rand() % p) - p / 2;  // between -p/2 and p/2
      a[i][1] = (rand() % p) - p / 2;
    }
    memcpy(a1, a, nn / 2 * sizeof(CPLX));
    cplx_fft_ref(tables, a1);
    cplx_ifft_ref(itables, a1);
    double d = 0;
    for (uint32_t i = 0; i < nn / 2; i++) {
      double dre = fabs(a[i][0] - a1[i][0] / (nn / 2));
      double dim = fabs(a[i][1] - a1[i][1] / (nn / 2));
      if (dre > d) d = dre;
      if (dim > d) d = dim;
    }
    ASSERT_LE(d, 1e-8);
    spqlios_free(a);
    spqlios_free(a1);
    delete_cplx_fft_precomp(tables);
    delete_cplx_ifft_precomp(itables);
  }
}

#ifdef __x86_64__
TEST(fft, halfcfft_ifft_fma_vs_ref) {
  for (IFFT_FUNCTION ifft : {cplx_ifft_ref, cplx_ifft_avx2_fma}) {
    for (uint64_t nn : {8, 16, 32, 1024, 4096, 8192, 65536}) {
      uint64_t m = nn / 2;
      CPLX_IFFT_PRECOMP* itables = new_cplx_ifft_precomp(m, 0);
      CPLX* a = (CPLX*)spqlios_alloc_custom_align(32, nn / 2 * sizeof(CPLX));
      CPLX* a1 = (CPLX*)spqlios_alloc_custom_align(32, nn / 2 * sizeof(CPLX));
      CPLX* a2 = (CPLX*)spqlios_alloc_custom_align(32, nn / 2 * sizeof(CPLX));
      int64_t p = 1 << 16;
      for (uint32_t i = 0; i < nn / 2; i++) {
        a[i][0] = (rand() % p) - p / 2;  // between -p/2 and p/2
        a[i][1] = (rand() % p) - p / 2;
      }
      memcpy(a1, a, nn / 2 * sizeof(CPLX));
      memcpy(a2, a, nn / 2 * sizeof(CPLX));
      cplx_ifft_naive(m, 0.25, a2);
      ifft(itables, a1);
      double d = 0;
      for (uint32_t i = 0; i < nn / 2; i++) {
        double dre = fabs(a1[i][0] - a2[i][0]);
        double dim = fabs(a1[i][1] - a2[i][1]);
        if (dre > d) d = dre;
        if (dim > d) d = dim;
      }
      ASSERT_LE(d, 1e-8);
      spqlios_free(a);
      spqlios_free(a1);
      spqlios_free(a2);
      delete_cplx_ifft_precomp(itables);
    }
  }
}
#endif

// test the reference and simple implementations of mul on all dimensions
TEST(fftvec, cplx_fftvec_mul_ref) {
  for (uint64_t nn : {2, 4, 8, 16, 32, 1024, 4096, 8192, 65536}) {
    uint64_t m = nn / 2;
    CPLX_FFTVEC_MUL_PRECOMP* precomp = new_cplx_fftvec_mul_precomp(m);
    CPLX* a = new CPLX[m];
    CPLX* b = new CPLX[m];
    CPLX* r0 = new CPLX[m];
    CPLX* r1 = new CPLX[m];
    CPLX* r2 = new CPLX[m];
    int64_t p = 1 << 16;
    for (uint32_t i = 0; i < m; i++) {
      a[i][0] = (rand() % p) - p / 2;  // between -p/2 and p/2
      a[i][1] = (rand() % p) - p / 2;
      b[i][0] = (rand() % p) - p / 2;  // between -p/2 and p/2
      b[i][1] = (rand() % p) - p / 2;
      r2[i][0] = r1[i][0] = r0[i][0] = (rand() % p) - p / 2;  // between -p/2 and p/2
      r2[i][1] = r1[i][1] = r0[i][1] = (rand() % p) - p / 2;
    }
    cplx_fftvec_mul_simple(m, r0, a, b);
    cplx_fftvec_mul_ref(precomp, r1, a, b);
    for (uint32_t i = 0; i < m; i++) {
      r2[i][0] = a[i][0] * b[i][0] - a[i][1] * b[i][1];
      r2[i][1] = a[i][0] * b[i][1] + a[i][1] * b[i][0];
      ASSERT_LE(fabs(r1[i][0] - r2[i][0]) + fabs(r1[i][1] - r2[i][1]), 1e-8);
      ASSERT_LE(fabs(r0[i][0] - r2[i][0]) + fabs(r0[i][1] - r2[i][1]), 1e-8);
    }
    delete[] a;
    delete[] b;
    delete[] r0;
    delete[] r1;
    delete[] r2;
    delete_cplx_fftvec_mul_precomp(precomp);
  }
}

// test the reference and simple implementations of addmul on all dimensions
TEST(fftvec, cplx_fftvec_addmul_ref) {
  for (uint64_t nn : {2, 4, 8, 16, 32, 1024, 4096, 8192, 65536}) {
    uint64_t m = nn / 2;
    CPLX_FFTVEC_ADDMUL_PRECOMP* precomp = new_cplx_fftvec_addmul_precomp(m);
    CPLX* a = new CPLX[m];
    CPLX* b = new CPLX[m];
    CPLX* r0 = new CPLX[m];
    CPLX* r1 = new CPLX[m];
    CPLX* r2 = new CPLX[m];
    int64_t p = 1 << 16;
    for (uint32_t i = 0; i < m; i++) {
      a[i][0] = (rand() % p) - p / 2;  // between -p/2 and p/2
      a[i][1] = (rand() % p) - p / 2;
      b[i][0] = (rand() % p) - p / 2;  // between -p/2 and p/2
      b[i][1] = (rand() % p) - p / 2;
      r2[i][0] = r1[i][0] = r0[i][0] = (rand() % p) - p / 2;  // between -p/2 and p/2
      r2[i][1] = r1[i][1] = r0[i][1] = (rand() % p) - p / 2;
    }
    cplx_fftvec_addmul_simple(m, r0, a, b);
    cplx_fftvec_addmul_ref(precomp, r1, a, b);
    for (uint32_t i = 0; i < m; i++) {
      r2[i][0] += a[i][0] * b[i][0] - a[i][1] * b[i][1];
      r2[i][1] += a[i][0] * b[i][1] + a[i][1] * b[i][0];
      ASSERT_LE(fabs(r1[i][0] - r2[i][0]) + fabs(r1[i][1] - r2[i][1]), 1e-8);
      ASSERT_LE(fabs(r0[i][0] - r2[i][0]) + fabs(r0[i][1] - r2[i][1]), 1e-8);
    }
    delete[] a;
    delete[] b;
    delete[] r0;
    delete[] r1;
    delete[] r2;
    delete_cplx_fftvec_addmul_precomp(precomp);
  }
}

// comparative tests between mul ref vs. optimized (only relevant dimensions)
TEST(fftvec, cplx_fftvec_mul_ref_vs_optim) {
  struct totest {
    FFTVEC_MUL_FUNCTION f;
    uint64_t min_m;
    totest(FFTVEC_MUL_FUNCTION f, uint64_t min_m) : f(f), min_m(min_m) {}
  };
  std::vector<totest> totestset;
  totestset.emplace_back(cplx_fftvec_mul, 1);
#ifdef __x86_64__
  totestset.emplace_back(cplx_fftvec_mul_fma, 8);
#endif
  for (uint64_t m : {1, 2, 4, 8, 16, 1024, 4096, 8192, 65536}) {
    CPLX_FFTVEC_MUL_PRECOMP* precomp = new_cplx_fftvec_mul_precomp(m);
    for (const totest& t : totestset) {
      if (t.min_m > m) continue;
      CPLX* a = new CPLX[m];
      CPLX* b = new CPLX[m];
      CPLX* r1 = new CPLX[m];
      CPLX* r2 = new CPLX[m];
      int64_t p = 1 << 16;
      for (uint32_t i = 0; i < m; i++) {
        a[i][0] = (rand() % p) - p / 2;  // between -p/2 and p/2
        a[i][1] = (rand() % p) - p / 2;
        b[i][0] = (rand() % p) - p / 2;  // between -p/2 and p/2
        b[i][1] = (rand() % p) - p / 2;
        r2[i][0] = r1[i][0] = (rand() % p) - p / 2;  // between -p/2 and p/2
        r2[i][1] = r1[i][1] = (rand() % p) - p / 2;
      }
      t.f(precomp, r1, a, b);
      cplx_fftvec_mul_ref(precomp, r2, a, b);
      for (uint32_t i = 0; i < m; i++) {
        double dre = fabs(r1[i][0] - r2[i][0]);
        double dim = fabs(r1[i][1] - r2[i][1]);
        ASSERT_LE(dre, 1e-8);
        ASSERT_LE(dim, 1e-8);
      }
      delete[] a;
      delete[] b;
      delete[] r1;
      delete[] r2;
    }
    delete_cplx_fftvec_mul_precomp(precomp);
  }
}

// comparative tests between addmul ref vs. optimized (only relevant dimensions)
TEST(fftvec, cplx_fftvec_addmul_ref_vs_optim) {
  struct totest {
    FFTVEC_ADDMUL_FUNCTION f;
    uint64_t min_m;
    totest(FFTVEC_ADDMUL_FUNCTION f, uint64_t min_m) : f(f), min_m(min_m) {}
  };
  std::vector<totest> totestset;
  totestset.emplace_back(cplx_fftvec_addmul, 1);
#ifdef __x86_64__
  totestset.emplace_back(cplx_fftvec_addmul_fma, 8);
#endif
  for (uint64_t m : {1, 2, 4, 8, 16, 1024, 4096, 8192, 65536}) {
    CPLX_FFTVEC_ADDMUL_PRECOMP* precomp = new_cplx_fftvec_addmul_precomp(m);
    for (const totest& t : totestset) {
      if (t.min_m > m) continue;
      CPLX* a = new CPLX[m];
      CPLX* b = new CPLX[m];
      CPLX* r1 = new CPLX[m];
      CPLX* r2 = new CPLX[m];
      int64_t p = 1 << 16;
      for (uint32_t i = 0; i < m; i++) {
        a[i][0] = (rand() % p) - p / 2;  // between -p/2 and p/2
        a[i][1] = (rand() % p) - p / 2;
        b[i][0] = (rand() % p) - p / 2;  // between -p/2 and p/2
        b[i][1] = (rand() % p) - p / 2;
        r2[i][0] = r1[i][0] = (rand() % p) - p / 2;  // between -p/2 and p/2
        r2[i][1] = r1[i][1] = (rand() % p) - p / 2;
      }
      t.f(precomp, r1, a, b);
      cplx_fftvec_addmul_ref(precomp, r2, a, b);
      for (uint32_t i = 0; i < m; i++) {
        double dre = fabs(r1[i][0] - r2[i][0]);
        double dim = fabs(r1[i][1] - r2[i][1]);
        ASSERT_LE(dre, 1e-8);
        ASSERT_LE(dim, 1e-8);
      }
      delete[] a;
      delete[] b;
      delete[] r1;
      delete[] r2;
    }
    delete_cplx_fftvec_addmul_precomp(precomp);
  }
}
