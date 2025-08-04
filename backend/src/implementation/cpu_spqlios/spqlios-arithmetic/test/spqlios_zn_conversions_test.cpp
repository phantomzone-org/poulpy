#include <gtest/gtest.h>
#include <spqlios/arithmetic/zn_arithmetic_private.h>

#include "testlib/test_commons.h"

template <typename SRC_T, typename DST_T>
static void test_conv(void (*conv_f)(const MOD_Z*, DST_T* res, uint64_t res_size, const SRC_T* a, uint64_t a_size),
                      DST_T (*ideal_conv_f)(SRC_T x), SRC_T (*random_f)()) {
  MOD_Z* module = new_z_module_info(DEFAULT);
  for (uint64_t a_size : {0, 1, 2, 42}) {
    for (uint64_t res_size : {0, 1, 2, 42}) {
      for (uint64_t trials = 0; trials < 100; ++trials) {
        std::vector<SRC_T> a(a_size);
        std::vector<DST_T> res(res_size);
        uint64_t msize = std::min(a_size, res_size);
        for (SRC_T& x : a) x = random_f();
        conv_f(module, res.data(), res_size, a.data(), a_size);
        for (uint64_t i = 0; i < msize; ++i) {
          DST_T expect = ideal_conv_f(a[i]);
          DST_T actual = res[i];
          ASSERT_EQ(expect, actual);
        }
        for (uint64_t i = msize; i < res_size; ++i) {
          DST_T expect = 0;
          SRC_T actual = res[i];
          ASSERT_EQ(expect, actual);
        }
      }
    }
  }
  delete_z_module_info(module);
}

static int32_t ideal_dbl_to_tn32(double a) {
  double _2p32 = INT64_C(1) << 32;
  double a_mod_1 = a - rint(a);
  int64_t t = rint(a_mod_1 * _2p32);
  return int32_t(t);
}

static double random_f64_10() { return uniform_f64_bounds(-10, 10); }

static void test_dbl_to_tn32(DBL_TO_TN32_F dbl_to_tn32_f) {
  test_conv(dbl_to_tn32_f, ideal_dbl_to_tn32, random_f64_10);
}

TEST(zn_arithmetic, dbl_to_tn32) { test_dbl_to_tn32(dbl_to_tn32); }
TEST(zn_arithmetic, dbl_to_tn32_ref) { test_dbl_to_tn32(dbl_to_tn32_ref); }

static double ideal_tn32_to_dbl(int32_t a) {
  const double _2p32 = INT64_C(1) << 32;
  return double(a) / _2p32;
}

static int32_t random_t32() { return uniform_i64_bits(32); }

static void test_tn32_to_dbl(TN32_TO_DBL_F tn32_to_dbl_f) { test_conv(tn32_to_dbl_f, ideal_tn32_to_dbl, random_t32); }

TEST(zn_arithmetic, tn32_to_dbl) { test_tn32_to_dbl(tn32_to_dbl); }
TEST(zn_arithmetic, tn32_to_dbl_ref) { test_tn32_to_dbl(tn32_to_dbl_ref); }

static int32_t ideal_dbl_round_to_i32(double a) { return int32_t(rint(a)); }

static double random_dbl_explaw_18() { return uniform_f64_bounds(-1., 1.) * pow(2., uniform_u64_bits(6) % 19); }

static void test_dbl_round_to_i32(DBL_ROUND_TO_I32_F dbl_round_to_i32_f) {
  test_conv(dbl_round_to_i32_f, ideal_dbl_round_to_i32, random_dbl_explaw_18);
}

TEST(zn_arithmetic, dbl_round_to_i32) { test_dbl_round_to_i32(dbl_round_to_i32); }
TEST(zn_arithmetic, dbl_round_to_i32_ref) { test_dbl_round_to_i32(dbl_round_to_i32_ref); }

static double ideal_i32_to_dbl(int32_t a) { return double(a); }

static int32_t random_i32_explaw_18() { return uniform_i64_bits(uniform_u64_bits(6) % 19); }

static void test_i32_to_dbl(I32_TO_DBL_F i32_to_dbl_f) {
  test_conv(i32_to_dbl_f, ideal_i32_to_dbl, random_i32_explaw_18);
}

TEST(zn_arithmetic, i32_to_dbl) { test_i32_to_dbl(i32_to_dbl); }
TEST(zn_arithmetic, i32_to_dbl_ref) { test_i32_to_dbl(i32_to_dbl_ref); }

static int64_t ideal_dbl_round_to_i64(double a) { return rint(a); }

static double random_dbl_explaw_50() { return uniform_f64_bounds(-1., 1.) * pow(2., uniform_u64_bits(7) % 51); }

static void test_dbl_round_to_i64(DBL_ROUND_TO_I64_F dbl_round_to_i64_f) {
  test_conv(dbl_round_to_i64_f, ideal_dbl_round_to_i64, random_dbl_explaw_50);
}

TEST(zn_arithmetic, dbl_round_to_i64) { test_dbl_round_to_i64(dbl_round_to_i64); }
TEST(zn_arithmetic, dbl_round_to_i64_ref) { test_dbl_round_to_i64(dbl_round_to_i64_ref); }

static double ideal_i64_to_dbl(int64_t a) { return double(a); }

static int64_t random_i64_explaw_50() { return uniform_i64_bits(uniform_u64_bits(7) % 51); }

static void test_i64_to_dbl(I64_TO_DBL_F i64_to_dbl_f) {
  test_conv(i64_to_dbl_f, ideal_i64_to_dbl, random_i64_explaw_50);
}

TEST(zn_arithmetic, i64_to_dbl) { test_i64_to_dbl(i64_to_dbl); }
TEST(zn_arithmetic, i64_to_dbl_ref) { test_i64_to_dbl(i64_to_dbl_ref); }
