#include <gtest/gtest.h>
#include <spqlios/reim/reim_fft_internal.h>

#include "testlib/test_commons.h"

TEST(reim_conversions, reim_to_tnx) {
  for (uint32_t m : {1, 2, 64, 128, 512}) {
    for (double divisor : {1, 2, int(m)}) {
      for (uint32_t log2overhead : {1, 2, 10, 18, 35, 42}) {
        double maxdiff = pow(2., log2overhead - 50);
        std::vector<double> data(2 * m);
        std::vector<double> dout(2 * m);
        for (uint64_t i = 0; i < 2 * m; ++i) {
          data[i] = (uniform_f64_01() - 0.5) * pow(2., log2overhead + 1) * divisor;
        }
        REIM_TO_TNX_PRECOMP* p = new_reim_to_tnx_precomp(m, divisor, 18);
        reim_to_tnx_ref(p, dout.data(), data.data());
        for (uint64_t i = 0; i < 2 * m; ++i) {
          ASSERT_LE(fabs(dout[i]), 0.5);
          double diff = dout[i] - data[i] / divisor;
          double fracdiff = diff - rint(diff);
          ASSERT_LE(fabs(fracdiff), maxdiff);
        }
        delete_reim_to_tnx_precomp(p);
      }
    }
  }
}

#ifdef __x86_64__
TEST(reim_conversions, reim_to_tnx_ref_vs_avx) {
  for (uint32_t m : {8, 16, 64, 128, 512}) {
    for (double divisor : {1, 2, int(m)}) {
      for (uint32_t log2overhead : {1, 2, 10, 18, 35, 42}) {
        // double maxdiff = pow(2., log2overhead - 50);
        std::vector<double> data(2 * m);
        std::vector<double> dout1(2 * m);
        std::vector<double> dout2(2 * m);
        for (uint64_t i = 0; i < 2 * m; ++i) {
          data[i] = (uniform_f64_01() - 0.5) * pow(2., log2overhead + 1) * divisor;
        }
        REIM_TO_TNX_PRECOMP* p = new_reim_to_tnx_precomp(m, divisor, 18);
        reim_to_tnx_ref(p, dout1.data(), data.data());
        reim_to_tnx_avx(p, dout2.data(), data.data());
        for (uint64_t i = 0; i < 2 * m; ++i) {
          ASSERT_LE(fabs(dout1[i] - dout2[i]), 0.);
        }
        delete_reim_to_tnx_precomp(p);
      }
    }
  }
}
#endif

typedef typeof(reim_from_znx64_ref) reim_from_znx64_f;

void test_reim_from_znx64(reim_from_znx64_f reim_from_znx64, uint64_t maxbnd) {
  for (uint32_t m : {4, 8, 16, 64, 16384}) {
    REIM_FROM_ZNX64_PRECOMP* p = new_reim_from_znx64_precomp(m, maxbnd);
    std::vector<int64_t> data(2 * m);
    std::vector<double> dout(2 * m);
    for (uint64_t i = 0; i < 2 * m; ++i) {
      int64_t magnitude = int64_t(uniform_u64() % (maxbnd + 1));
      data[i] = uniform_i64() >> (63 - magnitude);
      REQUIRE_DRAMATICALLY(abs(data[i]) <= (INT64_C(1) << magnitude), "pb");
    }
    reim_from_znx64(p, dout.data(), data.data());
    for (uint64_t i = 0; i < 2 * m; ++i) {
      ASSERT_EQ(dout[i], double(data[i])) << dout[i] << " " << data[i];
    }
    delete_reim_from_znx64_precomp(p);
  }
}

TEST(reim_conversions, reim_from_znx64) {
  for (uint64_t maxbnd : {50}) {
    test_reim_from_znx64(reim_from_znx64, maxbnd);
  }
}
TEST(reim_conversions, reim_from_znx64_ref) { test_reim_from_znx64(reim_from_znx64_ref, 50); }
#ifdef __x86_64__
TEST(reim_conversions, reim_from_znx64_avx2_bnd50_fma) { test_reim_from_znx64(reim_from_znx64_bnd50_fma, 50); }
#endif

typedef typeof(reim_to_znx64_ref) reim_to_znx64_f;

void test_reim_to_znx64(reim_to_znx64_f reim_to_znx64_fcn, int64_t maxbnd) {
  for (uint32_t m : {4, 8, 16, 64, 16384}) {
    for (double divisor : {1, 2, int(m)}) {
      REIM_TO_ZNX64_PRECOMP* p = new_reim_to_znx64_precomp(m, divisor, maxbnd);
      std::vector<double> data(2 * m);
      std::vector<int64_t> dout(2 * m);
      for (uint64_t i = 0; i < 2 * m; ++i) {
        int64_t magnitude = int64_t(uniform_u64() % (maxbnd + 11)) - 10;
        data[i] = (uniform_f64_01() - 0.5) * pow(2., magnitude + 1) * divisor;
      }
      reim_to_znx64_fcn(p, dout.data(), data.data());
      for (uint64_t i = 0; i < 2 * m; ++i) {
        ASSERT_LE(dout[i] - data[i] / divisor, 0.5) << dout[i] << " " << data[i];
      }
      delete_reim_to_znx64_precomp(p);
    }
  }
}

TEST(reim_conversions, reim_to_znx64) {
  for (uint64_t maxbnd : {63, 50}) {
    test_reim_to_znx64(reim_to_znx64, maxbnd);
  }
}
TEST(reim_conversions, reim_to_znx64_ref) { test_reim_to_znx64(reim_to_znx64_ref, 63); }
#ifdef __x86_64__
TEST(reim_conversions, reim_to_znx64_avx2_bnd63_fma) { test_reim_to_znx64(reim_to_znx64_avx2_bnd63_fma, 63); }
TEST(reim_conversions, reim_to_znx64_avx2_bnd50_fma) { test_reim_to_znx64(reim_to_znx64_avx2_bnd50_fma, 50); }
#endif
