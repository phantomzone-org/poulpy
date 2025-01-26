#include <inttypes.h>

#include <cmath>

#include "gtest/gtest.h"
#include "spqlios/commons_private.h"
#include "spqlios/cplx/cplx_fft_internal.h"
#include "spqlios/reim/reim_fft_internal.h"
#include "spqlios/reim/reim_fft_private.h"

#ifdef __x86_64__
TEST(fft, reim_fft_avx2_vs_fft_reim_ref) {
  for (uint64_t nn : {16, 32, 64, 1024, 8192, 65536}) {
    uint64_t m = nn / 2;
    // CPLX_FFT_PRECOMP* tables = new_cplx_fft_precomp(m, 0);
    REIM_FFT_PRECOMP* reimtables = new_reim_fft_precomp(m, 0);
    CPLX* a = (CPLX*)spqlios_alloc_custom_align(32, nn / 2 * sizeof(CPLX));
    double* a1 = (double*)spqlios_alloc_custom_align(32, nn / 2 * sizeof(CPLX));
    double* a2 = (double*)spqlios_alloc_custom_align(32, nn / 2 * sizeof(CPLX));
    int64_t p = 1 << 16;
    for (uint32_t i = 0; i < nn / 2; i++) {
      a[i][0] = (rand() % p) - p / 2;  // between -p/2 and p/2
      a[i][1] = (rand() % p) - p / 2;
    }
    memcpy(a1, a, nn / 2 * sizeof(CPLX));
    memcpy(a2, a, nn / 2 * sizeof(CPLX));
    reim_fft_ref(reimtables, a2);
    reim_fft_avx2_fma(reimtables, a1);
    double d = 0;
    for (uint32_t i = 0; i < nn / 2; i++) {
      double dre = fabs(a1[i] - a2[i]);
      double dim = fabs(a1[nn / 2 + i] - a2[nn / 2 + i]);
      if (dre > d) d = dre;
      if (dim > d) d = dim;
      ASSERT_LE(d, nn * 1e-10) << nn;
    }
    ASSERT_LE(d, nn * 1e-10) << nn;
    spqlios_free(a);
    spqlios_free(a1);
    spqlios_free(a2);
    // delete_cplx_fft_precomp(tables);
    delete_reim_fft_precomp(reimtables);
  }
}
#endif

#ifdef __x86_64__
TEST(fft, reim_ifft_avx2_vs_reim_ifft_ref) {
  for (uint64_t nn : {16, 32, 64, 1024, 8192, 65536}) {
    uint64_t m = nn / 2;
    // CPLX_FFT_PRECOMP* tables = new_cplx_fft_precomp(m, 0);
    REIM_IFFT_PRECOMP* reimtables = new_reim_ifft_precomp(m, 0);
    CPLX* a = (CPLX*)spqlios_alloc_custom_align(32, nn / 2 * sizeof(CPLX));
    double* a1 = (double*)spqlios_alloc_custom_align(32, nn / 2 * sizeof(CPLX));
    double* a2 = (double*)spqlios_alloc_custom_align(32, nn / 2 * sizeof(CPLX));
    int64_t p = 1 << 16;
    for (uint32_t i = 0; i < nn / 2; i++) {
      a[i][0] = (rand() % p) - p / 2;  // between -p/2 and p/2
      a[i][1] = (rand() % p) - p / 2;
    }
    memcpy(a1, a, nn / 2 * sizeof(CPLX));
    memcpy(a2, a, nn / 2 * sizeof(CPLX));
    reim_ifft_ref(reimtables, a2);
    reim_ifft_avx2_fma(reimtables, a1);
    double d = 0;
    for (uint32_t i = 0; i < nn / 2; i++) {
      double dre = fabs(a1[i] - a2[i]);
      double dim = fabs(a1[nn / 2 + i] - a2[nn / 2 + i]);
      if (dre > d) d = dre;
      if (dim > d) d = dim;
      ASSERT_LE(d, 1e-8);
    }
    ASSERT_LE(d, 1e-8);
    spqlios_free(a);
    spqlios_free(a1);
    spqlios_free(a2);
    // delete_cplx_fft_precomp(tables);
    delete_reim_fft_precomp(reimtables);
  }
}
#endif

#ifdef __x86_64__
TEST(fft, reim_vecfft_addmul_fma_vs_ref) {
  for (uint64_t nn : {16, 32, 64, 1024, 8192, 65536}) {
    uint64_t m = nn / 2;
    REIM_FFTVEC_ADDMUL_PRECOMP* tbl = new_reim_fftvec_addmul_precomp(m);
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
    reim_fftvec_addmul_ref(tbl, r1, a1, b1);
    reim_fftvec_addmul_fma(tbl, r2, a2, b2);
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
    delete_reim_fftvec_addmul_precomp(tbl);
  }
}
#endif

#ifdef __x86_64__
TEST(fft, reim_vecfft_mul_fma_vs_ref) {
  for (uint64_t nn : {16, 32, 64, 1024, 8192, 65536}) {
    uint64_t m = nn / 2;
    REIM_FFTVEC_MUL_PRECOMP* tbl = new_reim_fftvec_mul_precomp(m);
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
    reim_fftvec_mul_ref(tbl, r1, a1, b1);
    reim_fftvec_mul_fma(tbl, r2, a2, b2);
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

typedef void (*FILL_REIM_FFT_OMG_F)(const double entry_pwr, double** omg);
typedef void (*REIM_FFT_F)(double* dre, double* dim, const void* omega);

// template to test a fixed-dimension fft vs. naive
template <uint64_t N>
void test_reim_fft_ref_vs_naive(FILL_REIM_FFT_OMG_F fill_omega_f, REIM_FFT_F reim_fft_f) {
  double om[N];
  double data[2 * N];
  double datacopy[2 * N];
  double* omg = om;
  fill_omega_f(0.25, &omg);
  ASSERT_EQ(omg - om, ptrdiff_t(N));  // it may depend on N
  for (uint64_t i = 0; i < N; ++i) {
    datacopy[i] = data[i] = (rand() % 100) - 50;
    datacopy[N + i] = data[N + i] = (rand() % 100) - 50;
  }
  reim_fft_f(datacopy, datacopy + N, om);
  reim_naive_fft(N, 0.25, data, data + N);
  double d = 0;
  for (uint64_t i = 0; i < 2 * N; ++i) {
    d += fabs(datacopy[i] - data[i]);
  }
  ASSERT_LE(d, 1e-7);
}

template <uint64_t N>
void test_reim_fft_ref_vs_accel(REIM_FFT_F reim_fft_ref_f, REIM_FFT_F reim_fft_accel_f) {
  double om[N];
  double data[2 * N];
  double datacopy[2 * N];
  for (uint64_t i = 0; i < N; ++i) {
    om[i] = (rand() % 100) - 50;
    datacopy[i] = data[i] = (rand() % 100) - 50;
    datacopy[N + i] = data[N + i] = (rand() % 100) - 50;
  }
  reim_fft_ref_f(datacopy, datacopy + N, om);
  reim_fft_accel_f(data, data + N, om);
  double d = 0;
  for (uint64_t i = 0; i < 2 * N; ++i) {
    d += fabs(datacopy[i] - data[i]);
  }
  if (d > 1e-15) {
    for (uint64_t i = 0; i < N; ++i) {
      printf("%" PRId64 " %lf %lf %lf %lf\n", i, data[i], data[N + i], datacopy[i], datacopy[N + i]);
    }
    ASSERT_LE(d, 0);
  }
}

TEST(fft, reim_fft16_ref_vs_naive) { test_reim_fft_ref_vs_naive<16>(fill_reim_fft16_omegas, reim_fft16_ref); }
#ifdef __aarch64__
TEST(fft, reim_fft16_neon_vs_naive) { test_reim_fft_ref_vs_naive<16>(fill_reim_fft16_omegas_neon, reim_fft16_neon); }
#endif

#ifdef __x86_64__
TEST(fft, reim_fft16_ref_vs_fma) { test_reim_fft_ref_vs_accel<16>(reim_fft16_ref, reim_fft16_avx_fma); }
#endif

#ifdef __aarch64__
static void reim_fft16_ref_neon_pom(double* dre, double* dim, const void* omega) {
  const double* pom = (double*) omega;
  // put the omegas in neon order
  double x_pom[] = {
    pom[0], pom[1], pom[2], pom[3],
    pom[4],pom[5], pom[6], pom[7],
    pom[8], pom[10],pom[12], pom[14],
    pom[9], pom[11],pom[13], pom[15]
  };
  reim_fft16_ref(dre, dim, x_pom);
}
TEST(fft, reim_fft16_ref_vs_neon) { test_reim_fft_ref_vs_accel<16>(reim_fft16_ref_neon_pom, reim_fft16_neon); }
#endif

TEST(fft, reim_fft8_ref_vs_naive) { test_reim_fft_ref_vs_naive<8>(fill_reim_fft8_omegas, reim_fft8_ref); }

#ifdef __x86_64__
TEST(fft, reim_fft8_ref_vs_fma) { test_reim_fft_ref_vs_accel<8>(reim_fft8_ref, reim_fft8_avx_fma); }
#endif

TEST(fft, reim_fft4_ref_vs_naive) { test_reim_fft_ref_vs_naive<4>(fill_reim_fft4_omegas, reim_fft4_ref); }

#ifdef __x86_64__
TEST(fft, reim_fft4_ref_vs_fma) { test_reim_fft_ref_vs_accel<4>(reim_fft4_ref, reim_fft4_avx_fma); }
#endif

TEST(fft, reim_fft2_ref_vs_naive) { test_reim_fft_ref_vs_naive<2>(fill_reim_fft2_omegas, reim_fft2_ref); }

TEST(fft, reim_fft_bfs_16_ref_vs_naive) {
  for (const uint64_t m : {16, 32, 64, 128, 256, 512, 1024, 2048}) {
    std::vector<double> om(2 * m);
    std::vector<double> data(2 * m);
    std::vector<double> datacopy(2 * m);
    double* omg = om.data();
    fill_reim_fft_bfs_16_omegas(m, 0.25, &omg);
    ASSERT_LE(omg - om.data(), ptrdiff_t(2 * m));  // it may depend on m
    for (uint64_t i = 0; i < m; ++i) {
      datacopy[i] = data[i] = (rand() % 100) - 50;
      datacopy[m + i] = data[m + i] = (rand() % 100) - 50;
    }
    omg = om.data();
    reim_fft_bfs_16_ref(m, datacopy.data(), datacopy.data() + m, &omg);
    reim_naive_fft(m, 0.25, data.data(), data.data() + m);
    double d = 0;
    for (uint64_t i = 0; i < 2 * m; ++i) {
      d += fabs(datacopy[i] - data[i]);
    }
    ASSERT_LE(d, 1e-7);
  }
}

TEST(fft, reim_fft_rec_16_ref_vs_naive) {
  for (const uint64_t m : {2048, 4096, 8192, 32768, 65536}) {
    std::vector<double> om(2 * m);
    std::vector<double> data(2 * m);
    std::vector<double> datacopy(2 * m);
    double* omg = om.data();
    fill_reim_fft_rec_16_omegas(m, 0.25, &omg);
    ASSERT_LE(omg - om.data(), ptrdiff_t(2 * m));  // it may depend on m
    for (uint64_t i = 0; i < m; ++i) {
      datacopy[i] = data[i] = (rand() % 100) - 50;
      datacopy[m + i] = data[m + i] = (rand() % 100) - 50;
    }
    omg = om.data();
    reim_fft_rec_16_ref(m, datacopy.data(), datacopy.data() + m, &omg);
    reim_naive_fft(m, 0.25, data.data(), data.data() + m);
    double d = 0;
    for (uint64_t i = 0; i < 2 * m; ++i) {
      d += fabs(datacopy[i] - data[i]);
    }
    ASSERT_LE(d, 1e-5);
  }
}

TEST(fft, reim_fft_ref_vs_naive) {
  for (const uint64_t m : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 32768, 65536}) {
    std::vector<double> om(2 * m);
    std::vector<double> data(2 * m);
    std::vector<double> datacopy(2 * m);
    REIM_FFT_PRECOMP* precomp = new_reim_fft_precomp(m, 0);
    for (uint64_t i = 0; i < m; ++i) {
      datacopy[i] = data[i] = (rand() % 100) - 50;
      datacopy[m + i] = data[m + i] = (rand() % 100) - 50;
    }
    reim_fft_ref(precomp, datacopy.data());
    reim_naive_fft(m, 0.25, data.data(), data.data() + m);
    double d = 0;
    for (uint64_t i = 0; i < 2 * m; ++i) {
      d += fabs(datacopy[i] - data[i]);
    }
    ASSERT_LE(d, 1e-5) << m;
    delete_reim_fft_precomp(precomp);
  }
}

#ifdef __aarch64__
EXPORT REIM_FFT_PRECOMP* new_reim_fft_precomp_neon(uint32_t m, uint32_t num_buffers);
EXPORT void reim_fft_neon(const REIM_FFT_PRECOMP* precomp, double* d);
TEST(fft, reim_fft_neon_vs_naive) {
  for (const uint64_t m : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 32768, 65536}) {
    std::vector<double> om(2 * m);
    std::vector<double> data(2 * m);
    std::vector<double> datacopy(2 * m);
    REIM_FFT_PRECOMP* precomp = new_reim_fft_precomp_neon(m, 0);
    for (uint64_t i = 0; i < m; ++i) {
      datacopy[i] = data[i] = (rand() % 100) - 50;
      datacopy[m + i] = data[m + i] = (rand() % 100) - 50;
    }
    reim_fft_neon(precomp, datacopy.data());
    reim_naive_fft(m, 0.25, data.data(), data.data() + m);
    double d = 0;
    for (uint64_t i = 0; i < 2 * m; ++i) {
      d += fabs(datacopy[i] - data[i]);
    }
    ASSERT_LE(d, 1e-5) << m;
    delete_reim_fft_precomp(precomp);
  }
}
#endif

typedef void (*FILL_REIM_IFFT_OMG_F)(const double entry_pwr, double** omg);
typedef void (*REIM_IFFT_F)(double* dre, double* dim, const void* omega);

// template to test a fixed-dimension fft vs. naive
template <uint64_t N>
void test_reim_ifft_ref_vs_naive(FILL_REIM_IFFT_OMG_F fill_omega_f, REIM_IFFT_F reim_ifft_f) {
  double om[N];
  double data[2 * N];
  double datacopy[2 * N];
  double* omg = om;
  fill_omega_f(0.25, &omg);
  ASSERT_EQ(omg - om, ptrdiff_t(N));  // it may depend on N
  for (uint64_t i = 0; i < N; ++i) {
    datacopy[i] = data[i] = (rand() % 100) - 50;
    datacopy[N + i] = data[N + i] = (rand() % 100) - 50;
  }
  reim_ifft_f(datacopy, datacopy + N, om);
  reim_naive_ifft(N, 0.25, data, data + N);
  double d = 0;
  for (uint64_t i = 0; i < 2 * N; ++i) {
    d += fabs(datacopy[i] - data[i]);
  }
  ASSERT_LE(d, 1e-7);
}

template <uint64_t N>
void test_reim_ifft_ref_vs_accel(REIM_IFFT_F reim_ifft_ref_f, REIM_IFFT_F reim_ifft_accel_f) {
  double om[N];
  double data[2 * N];
  double datacopy[2 * N];
  for (uint64_t i = 0; i < N; ++i) {
    om[i] = (rand() % 100) - 50;
    datacopy[i] = data[i] = (rand() % 100) - 50;
    datacopy[N + i] = data[N + i] = (rand() % 100) - 50;
  }
  reim_ifft_ref_f(datacopy, datacopy + N, om);
  reim_ifft_accel_f(data, data + N, om);
  double d = 0;
  for (uint64_t i = 0; i < 2 * N; ++i) {
    d += fabs(datacopy[i] - data[i]);
  }
  if (d > 1e-15) {
    for (uint64_t i = 0; i < N; ++i) {
      printf("%" PRId64 " %lf %lf %lf %lf\n", i, data[i], data[N + i], datacopy[i], datacopy[N + i]);
    }
    ASSERT_LE(d, 0);
  }
}

TEST(fft, reim_ifft16_ref_vs_naive) { test_reim_ifft_ref_vs_naive<16>(fill_reim_ifft16_omegas, reim_ifft16_ref); }

#ifdef __x86_64__
TEST(fft, reim_ifft16_ref_vs_fma) { test_reim_ifft_ref_vs_accel<16>(reim_ifft16_ref, reim_ifft16_avx_fma); }
#endif

TEST(fft, reim_ifft8_ref_vs_naive) { test_reim_ifft_ref_vs_naive<8>(fill_reim_ifft8_omegas, reim_ifft8_ref); }

#ifdef __x86_64__
TEST(fft, reim_ifft8_ref_vs_fma) { test_reim_ifft_ref_vs_accel<8>(reim_ifft8_ref, reim_ifft8_avx_fma); }
#endif

TEST(fft, reim_ifft4_ref_vs_naive) { test_reim_ifft_ref_vs_naive<4>(fill_reim_ifft4_omegas, reim_ifft4_ref); }

#ifdef __x86_64__
TEST(fft, reim_ifft4_ref_vs_fma) { test_reim_ifft_ref_vs_accel<4>(reim_ifft4_ref, reim_ifft4_avx_fma); }
#endif

TEST(fft, reim_ifft2_ref_vs_naive) { test_reim_ifft_ref_vs_naive<2>(fill_reim_ifft2_omegas, reim_ifft2_ref); }

TEST(fft, reim_ifft_bfs_16_ref_vs_naive) {
  for (const uint64_t m : {16, 32, 64, 128, 256, 512, 1024, 2048}) {
    std::vector<double> om(2 * m);
    std::vector<double> data(2 * m);
    std::vector<double> datacopy(2 * m);
    double* omg = om.data();
    fill_reim_ifft_bfs_16_omegas(m, 0.25, &omg);
    ASSERT_LE(omg - om.data(), ptrdiff_t(2 * m));  // it may depend on m
    for (uint64_t i = 0; i < m; ++i) {
      datacopy[i] = data[i] = (rand() % 100) - 50;
      datacopy[m + i] = data[m + i] = (rand() % 100) - 50;
    }
    omg = om.data();
    reim_ifft_bfs_16_ref(m, datacopy.data(), datacopy.data() + m, &omg);
    reim_naive_ifft(m, 0.25, data.data(), data.data() + m);
    double d = 0;
    for (uint64_t i = 0; i < 2 * m; ++i) {
      d += fabs(datacopy[i] - data[i]);
    }
    ASSERT_LE(d, 1e-7);
  }
}

TEST(fft, reim_ifft_rec_16_ref_vs_naive) {
  for (const uint64_t m : {2048, 4096, 8192, 32768, 65536}) {
    std::vector<double> om(2 * m);
    std::vector<double> data(2 * m);
    std::vector<double> datacopy(2 * m);
    double* omg = om.data();
    fill_reim_ifft_rec_16_omegas(m, 0.25, &omg);
    ASSERT_LE(omg - om.data(), ptrdiff_t(2 * m));  // it may depend on m
    for (uint64_t i = 0; i < m; ++i) {
      datacopy[i] = data[i] = (rand() % 100) - 50;
      datacopy[m + i] = data[m + i] = (rand() % 100) - 50;
    }
    omg = om.data();
    reim_ifft_rec_16_ref(m, datacopy.data(), datacopy.data() + m, &omg);
    reim_naive_ifft(m, 0.25, data.data(), data.data() + m);
    double d = 0;
    for (uint64_t i = 0; i < 2 * m; ++i) {
      d += fabs(datacopy[i] - data[i]);
    }
    ASSERT_LE(d, 1e-5);
  }
}

TEST(fft, reim_ifft_ref_vs_naive) {
  for (const uint64_t m : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 32768, 65536}) {
    std::vector<double> om(2 * m);
    std::vector<double> data(2 * m);
    std::vector<double> datacopy(2 * m);
    REIM_IFFT_PRECOMP* precomp = new_reim_ifft_precomp(m, 0);
    for (uint64_t i = 0; i < m; ++i) {
      datacopy[i] = data[i] = (rand() % 100) - 50;
      datacopy[m + i] = data[m + i] = (rand() % 100) - 50;
    }
    reim_ifft_ref(precomp, datacopy.data());
    reim_naive_ifft(m, 0.25, data.data(), data.data() + m);
    double d = 0;
    for (uint64_t i = 0; i < 2 * m; ++i) {
      d += fabs(datacopy[i] - data[i]);
    }
    ASSERT_LE(d, 1e-5) << m;
    delete_reim_ifft_precomp(precomp);
  }
}
