#include <gtest/gtest.h>
#include <spqlios/arithmetic/vec_rnx_arithmetic_private.h>

#include "testlib/test_commons.h"

template <typename SRC_T, typename DST_T>
static void test_conv(void (*conv_f)(const MOD_RNX*,                                   //
                                     DST_T* res, uint64_t res_size, uint64_t res_sl,   //
                                     const SRC_T* a, uint64_t a_size, uint64_t a_sl),  //
                      DST_T (*ideal_conv_f)(SRC_T x),                                  //
                      SRC_T (*random_f)()                                              //
) {
  for (uint64_t nn : {2, 4, 16, 64}) {
    MOD_RNX* module = new_rnx_module_info(nn, FFT64);
    for (uint64_t a_size : {0, 1, 2, 5}) {
      for (uint64_t res_size : {0, 1, 3, 5}) {
        for (uint64_t trials = 0; trials < 20; ++trials) {
          uint64_t a_sl = nn + uniform_u64_bits(2);
          uint64_t res_sl = nn + uniform_u64_bits(2);
          std::vector<SRC_T> a(a_sl * a_size);
          std::vector<DST_T> res(res_sl * res_size);
          uint64_t msize = std::min(a_size, res_size);
          for (uint64_t i = 0; i < a_size; ++i) {
            for (uint64_t j = 0; j < nn; ++j) {
              a[i * a_sl + j] = random_f();
            }
          }
          conv_f(module, res.data(), res_size, res_sl, a.data(), a_size, a_sl);
          for (uint64_t i = 0; i < msize; ++i) {
            for (uint64_t j = 0; j < nn; ++j) {
              SRC_T aij = a[i * a_sl + j];
              DST_T expect = ideal_conv_f(aij);
              DST_T actual = res[i * res_sl + j];
              ASSERT_EQ(expect, actual);
            }
          }
          for (uint64_t i = msize; i < res_size; ++i) {
            DST_T expect = 0;
            for (uint64_t j = 0; j < nn; ++j) {
              SRC_T actual = res[i * res_sl + j];
              ASSERT_EQ(expect, actual);
            }
          }
        }
      }
    }
    delete_rnx_module_info(module);
  }
}

static int32_t ideal_dbl_to_tn32(double a) {
  double _2p32 = INT64_C(1) << 32;
  double a_mod_1 = a - rint(a);
  int64_t t = rint(a_mod_1 * _2p32);
  return int32_t(t);
}

static double random_f64_10() { return uniform_f64_bounds(-10, 10); }

static void test_vec_rnx_to_tnx32(VEC_RNX_TO_TNX32_F vec_rnx_to_tnx32_f) {
  test_conv(vec_rnx_to_tnx32_f, ideal_dbl_to_tn32, random_f64_10);
}

TEST(vec_rnx_arithmetic, vec_rnx_to_tnx32) { test_vec_rnx_to_tnx32(vec_rnx_to_tnx32); }
TEST(vec_rnx_arithmetic, vec_rnx_to_tnx32_ref) { test_vec_rnx_to_tnx32(vec_rnx_to_tnx32_ref); }

static double ideal_tn32_to_dbl(int32_t a) {
  const double _2p32 = INT64_C(1) << 32;
  return double(a) / _2p32;
}

static int32_t random_t32() { return uniform_i64_bits(32); }

static void test_vec_rnx_from_tnx32(VEC_RNX_FROM_TNX32_F vec_rnx_from_tnx32_f) {
  test_conv(vec_rnx_from_tnx32_f, ideal_tn32_to_dbl, random_t32);
}

TEST(vec_rnx_arithmetic, vec_rnx_from_tnx32) { test_vec_rnx_from_tnx32(vec_rnx_from_tnx32); }
TEST(vec_rnx_arithmetic, vec_rnx_from_tnx32_ref) { test_vec_rnx_from_tnx32(vec_rnx_from_tnx32_ref); }

static int32_t ideal_dbl_round_to_i32(double a) { return int32_t(rint(a)); }

static double random_dbl_explaw_18() { return uniform_f64_bounds(-1., 1.) * pow(2., uniform_u64_bits(6) % 19); }

static void test_vec_rnx_to_znx32(VEC_RNX_TO_ZNX32_F vec_rnx_to_znx32_f) {
  test_conv(vec_rnx_to_znx32_f, ideal_dbl_round_to_i32, random_dbl_explaw_18);
}

TEST(zn_arithmetic, vec_rnx_to_znx32) { test_vec_rnx_to_znx32(vec_rnx_to_znx32); }
TEST(zn_arithmetic, vec_rnx_to_znx32_ref) { test_vec_rnx_to_znx32(vec_rnx_to_znx32_ref); }

static double ideal_i32_to_dbl(int32_t a) { return double(a); }

static int32_t random_i32_explaw_18() { return uniform_i64_bits(uniform_u64_bits(6) % 19); }

static void test_vec_rnx_from_znx32(VEC_RNX_FROM_ZNX32_F vec_rnx_from_znx32_f) {
  test_conv(vec_rnx_from_znx32_f, ideal_i32_to_dbl, random_i32_explaw_18);
}

TEST(zn_arithmetic, vec_rnx_from_znx32) { test_vec_rnx_from_znx32(vec_rnx_from_znx32); }
TEST(zn_arithmetic, vec_rnx_from_znx32_ref) { test_vec_rnx_from_znx32(vec_rnx_from_znx32_ref); }

static double ideal_dbl_to_tndbl(double a) { return a - rint(a); }

static void test_vec_rnx_to_tnxdbl(VEC_RNX_TO_TNXDBL_F vec_rnx_to_tnxdbl_f) {
  test_conv(vec_rnx_to_tnxdbl_f, ideal_dbl_to_tndbl, random_f64_10);
}

TEST(zn_arithmetic, vec_rnx_to_tnxdbl) { test_vec_rnx_to_tnxdbl(vec_rnx_to_tnxdbl); }
TEST(zn_arithmetic, vec_rnx_to_tnxdbl_ref) { test_vec_rnx_to_tnxdbl(vec_rnx_to_tnxdbl_ref); }

#if 0
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
#endif
