#include <gtest/gtest.h>

#include "spqlios/arithmetic/vec_rnx_arithmetic_private.h"
#include "spqlios/reim/reim_fft.h"
#include "testlib/vec_rnx_layout.h"

// disabling this test by default, since it depicts on purpose wrong accesses
#if 0
TEST(rnx_layout, valgrind_antipattern_test) {
  uint64_t n = 4;
  rnx_vec_f64_layout v(n, 7, 13);
  // this should be ok
  v.set(0, rnx_f64::zero(n));
  // this should abort (wrong ring dimension)
  ASSERT_DEATH(v.set(3, rnx_f64::zero(2 * n)), "");
  // this should abort (out of bounds)
  ASSERT_DEATH(v.set(8, rnx_f64::zero(n)), "");
  // this should be ok
  ASSERT_EQ(v.get_copy_zext(0), rnx_f64::zero(n));
  // should be an uninit read
  ASSERT_TRUE(!(v.get_copy_zext(2) == rnx_f64::zero(n)));  // should be uninit
  // should be an invalid read (inter-slice)
  ASSERT_NE(v.data()[4], 0);
  ASSERT_EQ(v.data()[2], 0);  // should be ok
  // should be an uninit read
  ASSERT_NE(v.data()[13], 0);  // should be uninit
}
#endif

// test of binary operations

// test for out of place calls
template <typename ACTUAL_FCN, typename EXPECT_FCN>
void test_vec_rnx_elemw_binop_outplace(ACTUAL_FCN binop, EXPECT_FCN ref_binop) {
  for (uint64_t n : {2, 4, 8, 128}) {
    RNX_MODULE_TYPE mtype = FFT64;
    MOD_RNX* mod = new_rnx_module_info(n, mtype);
    for (uint64_t sa : {7, 13, 15}) {
      for (uint64_t sb : {7, 13, 15}) {
        for (uint64_t sc : {7, 13, 15}) {
          uint64_t a_sl = uniform_u64_bits(3) * 5 + n;
          uint64_t b_sl = uniform_u64_bits(3) * 5 + n;
          uint64_t c_sl = uniform_u64_bits(3) * 5 + n;
          rnx_vec_f64_layout la(n, sa, a_sl);
          rnx_vec_f64_layout lb(n, sb, b_sl);
          rnx_vec_f64_layout lc(n, sc, c_sl);
          std::vector<rnx_f64> expect(sc);
          for (uint64_t i = 0; i < sa; ++i) {
            la.set(i, rnx_f64::random_log2bound(n, 1.));
          }
          for (uint64_t i = 0; i < sb; ++i) {
            lb.set(i, rnx_f64::random_log2bound(n, 1.));
          }
          for (uint64_t i = 0; i < sc; ++i) {
            expect[i] = ref_binop(la.get_copy_zext(i), lb.get_copy_zext(i));
          }
          binop(mod,                  // N
                lc.data(), sc, c_sl,  // res
                la.data(), sa, a_sl,  // a
                lb.data(), sb, b_sl);
          for (uint64_t i = 0; i < sc; ++i) {
            ASSERT_EQ(lc.get_copy_zext(i), expect[i]);
          }
        }
      }
    }
    delete_rnx_module_info(mod);
  }
}
// test for inplace1 calls
template <typename ACTUAL_FCN, typename EXPECT_FCN>
void test_vec_rnx_elemw_binop_inplace1(ACTUAL_FCN binop, EXPECT_FCN ref_binop) {
  for (uint64_t n : {2, 4, 64}) {
    RNX_MODULE_TYPE mtype = FFT64;
    MOD_RNX* mod = new_rnx_module_info(n, mtype);
    for (uint64_t sa : {3, 9, 12}) {
      for (uint64_t sb : {3, 9, 12}) {
        uint64_t a_sl = uniform_u64_bits(3) * 5 + n;
        uint64_t b_sl = uniform_u64_bits(3) * 5 + n;
        rnx_vec_f64_layout la(n, sa, a_sl);
        rnx_vec_f64_layout lb(n, sb, b_sl);
        std::vector<rnx_f64> expect(sa);
        for (uint64_t i = 0; i < sa; ++i) {
          la.set(i, rnx_f64::random_log2bound(n, 1.));
        }
        for (uint64_t i = 0; i < sb; ++i) {
          lb.set(i, rnx_f64::random_log2bound(n, 1.));
        }
        for (uint64_t i = 0; i < sa; ++i) {
          expect[i] = ref_binop(la.get_copy_zext(i), lb.get_copy_zext(i));
        }
        binop(mod,                  // N
              la.data(), sa, a_sl,  // res
              la.data(), sa, a_sl,  // a
              lb.data(), sb, b_sl);
        for (uint64_t i = 0; i < sa; ++i) {
          ASSERT_EQ(la.get_copy_zext(i), expect[i]);
        }
      }
    }
    delete_rnx_module_info(mod);
  }
}
// test for inplace2 calls
template <typename ACTUAL_FCN, typename EXPECT_FCN>
void test_vec_rnx_elemw_binop_inplace2(ACTUAL_FCN binop, EXPECT_FCN ref_binop) {
  for (uint64_t n : {4, 32, 64}) {
    RNX_MODULE_TYPE mtype = FFT64;
    MOD_RNX* mod = new_rnx_module_info(n, mtype);
    for (uint64_t sa : {3, 9, 12}) {
      for (uint64_t sb : {3, 9, 12}) {
        uint64_t a_sl = uniform_u64_bits(3) * 5 + n;
        uint64_t b_sl = uniform_u64_bits(3) * 5 + n;
        rnx_vec_f64_layout la(n, sa, a_sl);
        rnx_vec_f64_layout lb(n, sb, b_sl);
        std::vector<rnx_f64> expect(sb);
        for (uint64_t i = 0; i < sa; ++i) {
          la.set(i, rnx_f64::random_log2bound(n, 1.));
        }
        for (uint64_t i = 0; i < sb; ++i) {
          lb.set(i, rnx_f64::random_log2bound(n, 1.));
        }
        for (uint64_t i = 0; i < sb; ++i) {
          expect[i] = ref_binop(la.get_copy_zext(i), lb.get_copy_zext(i));
        }
        binop(mod,                  // N
              lb.data(), sb, b_sl,  // res
              la.data(), sa, a_sl,  // a
              lb.data(), sb, b_sl);
        for (uint64_t i = 0; i < sb; ++i) {
          ASSERT_EQ(lb.get_copy_zext(i), expect[i]);
        }
      }
    }
    delete_rnx_module_info(mod);
  }
}
// test for inplace3 calls
template <typename ACTUAL_FCN, typename EXPECT_FCN>
void test_vec_rnx_elemw_binop_inplace3(ACTUAL_FCN binop, EXPECT_FCN ref_binop) {
  for (uint64_t n : {2, 16, 1024}) {
    RNX_MODULE_TYPE mtype = FFT64;
    MOD_RNX* mod = new_rnx_module_info(n, mtype);
    for (uint64_t sa : {2, 6, 11}) {
      uint64_t a_sl = uniform_u64_bits(3) * 5 + n;
      rnx_vec_f64_layout la(n, sa, a_sl);
      std::vector<rnx_f64> expect(sa);
      for (uint64_t i = 0; i < sa; ++i) {
        la.set(i, rnx_f64::random_log2bound(n, 1.));
      }
      for (uint64_t i = 0; i < sa; ++i) {
        expect[i] = ref_binop(la.get_copy_zext(i), la.get_copy_zext(i));
      }
      binop(mod,                  // N
            la.data(), sa, a_sl,  // res
            la.data(), sa, a_sl,  // a
            la.data(), sa, a_sl);
      for (uint64_t i = 0; i < sa; ++i) {
        ASSERT_EQ(la.get_copy_zext(i), expect[i]);
      }
    }
    delete_rnx_module_info(mod);
  }
}
template <typename ACTUAL_FCN, typename EXPECT_FCN>
void test_vec_rnx_elemw_binop(ACTUAL_FCN binop, EXPECT_FCN ref_binop) {
  test_vec_rnx_elemw_binop_outplace(binop, ref_binop);
  test_vec_rnx_elemw_binop_inplace1(binop, ref_binop);
  test_vec_rnx_elemw_binop_inplace2(binop, ref_binop);
  test_vec_rnx_elemw_binop_inplace3(binop, ref_binop);
}

static rnx_f64 poly_add(const rnx_f64& a, const rnx_f64& b) { return a + b; }
static rnx_f64 poly_sub(const rnx_f64& a, const rnx_f64& b) { return a - b; }
TEST(vec_rnx, vec_rnx_add) { test_vec_rnx_elemw_binop(vec_rnx_add, poly_add); }
TEST(vec_rnx, vec_rnx_add_ref) { test_vec_rnx_elemw_binop(vec_rnx_add_ref, poly_add); }
#ifdef __x86_64__
TEST(vec_rnx, vec_rnx_add_avx) { test_vec_rnx_elemw_binop(vec_rnx_add_avx, poly_add); }
#endif
TEST(vec_rnx, vec_rnx_sub) { test_vec_rnx_elemw_binop(vec_rnx_sub, poly_sub); }
TEST(vec_rnx, vec_rnx_sub_ref) { test_vec_rnx_elemw_binop(vec_rnx_sub_ref, poly_sub); }
#ifdef __x86_64__
TEST(vec_rnx, vec_rnx_sub_avx) { test_vec_rnx_elemw_binop(vec_rnx_sub_avx, poly_sub); }
#endif

// test for out of place calls
template <typename ACTUAL_FCN, typename EXPECT_FCN>
void test_vec_rnx_elemw_unop_param_outplace(ACTUAL_FCN test_mul_xp_minus_one, EXPECT_FCN ref_mul_xp_minus_one,
                                            int64_t (*param_gen)()) {
  for (uint64_t n : {2, 4, 8, 128}) {
    MOD_RNX* mod = new_rnx_module_info(n, FFT64);
    for (uint64_t sa : {7, 13, 15}) {
      for (uint64_t sb : {7, 13, 15}) {
        {
          int64_t p = param_gen();
          uint64_t a_sl = uniform_u64_bits(3) * 5 + n;
          uint64_t b_sl = uniform_u64_bits(3) * 4 + n;
          rnx_vec_f64_layout la(n, sa, a_sl);
          rnx_vec_f64_layout lb(n, sb, b_sl);
          std::vector<rnx_f64> expect(sb);
          for (uint64_t i = 0; i < sa; ++i) {
            la.set(i, rnx_f64::random_log2bound(n, 0));
          }
          for (uint64_t i = 0; i < sb; ++i) {
            expect[i] = ref_mul_xp_minus_one(p, la.get_copy_zext(i));
          }
          test_mul_xp_minus_one(mod,                  //
                                p,                    //
                                lb.data(), sb, b_sl,  //
                                la.data(), sa, a_sl   //
          );
          for (uint64_t i = 0; i < sb; ++i) {
            ASSERT_EQ(lb.get_copy_zext(i), expect[i]) << n << " " << sa << " " << sb << " " << i;
          }
        }
      }
    }
    delete_rnx_module_info(mod);
  }
}

// test for inplace calls
template <typename ACTUAL_FCN, typename EXPECT_FCN>
void test_vec_rnx_elemw_unop_param_inplace(ACTUAL_FCN actual_function, EXPECT_FCN ref_function,
                                           int64_t (*param_gen)()) {
  for (uint64_t n : {2, 16, 1024}) {
    MOD_RNX* mod = new_rnx_module_info(n, FFT64);
    for (uint64_t sa : {2, 6, 11}) {
      {
        int64_t p = param_gen();
        uint64_t a_sl = uniform_u64_bits(3) * 5 + n;
        rnx_vec_f64_layout la(n, sa, a_sl);
        std::vector<rnx_f64> expect(sa);
        for (uint64_t i = 0; i < sa; ++i) {
          la.set(i, rnx_f64::random_log2bound(n, 0));
        }
        for (uint64_t i = 0; i < sa; ++i) {
          expect[i] = ref_function(p, la.get_copy_zext(i));
        }
        actual_function(mod,                  // N
                        p,                    //;
                        la.data(), sa, a_sl,  // res
                        la.data(), sa, a_sl   // a
        );
        for (uint64_t i = 0; i < sa; ++i) {
          ASSERT_EQ(la.get_copy_zext(i), expect[i]) << n << " " << sa << " " << i;
        }
      }
    }
    delete_rnx_module_info(mod);
  }
}

static int64_t random_mul_xp_minus_one_param() { return uniform_i64(); }
static int64_t random_automorphism_param() { return 2 * uniform_i64() + 1; }
static int64_t random_rotation_param() { return uniform_i64(); }

template <typename ACTUAL_FCN, typename EXPECT_FCN>
void test_vec_rnx_elemw_mul_xp_minus_one(ACTUAL_FCN binop, EXPECT_FCN ref_binop) {
  test_vec_rnx_elemw_unop_param_outplace(binop, ref_binop, random_mul_xp_minus_one_param);
  test_vec_rnx_elemw_unop_param_inplace(binop, ref_binop, random_mul_xp_minus_one_param);
}
template <typename ACTUAL_FCN, typename EXPECT_FCN>
void test_vec_rnx_elemw_rotate(ACTUAL_FCN binop, EXPECT_FCN ref_binop) {
  test_vec_rnx_elemw_unop_param_outplace(binop, ref_binop, random_rotation_param);
  test_vec_rnx_elemw_unop_param_inplace(binop, ref_binop, random_rotation_param);
}
template <typename ACTUAL_FCN, typename EXPECT_FCN>
void test_vec_rnx_elemw_automorphism(ACTUAL_FCN binop, EXPECT_FCN ref_binop) {
  test_vec_rnx_elemw_unop_param_outplace(binop, ref_binop, random_automorphism_param);
  test_vec_rnx_elemw_unop_param_inplace(binop, ref_binop, random_automorphism_param);
}

static rnx_f64 poly_mul_xp_minus_one(const int64_t p, const rnx_f64& a) {
  uint64_t n = a.nn();
  rnx_f64 res(n);
  for (uint64_t i = 0; i < n; ++i) {
    res.set_coeff(i, a.get_coeff(i - p) - a.get_coeff(i));
  }
  return res;
}
static rnx_f64 poly_rotate(const int64_t p, const rnx_f64& a) {
  uint64_t n = a.nn();
  rnx_f64 res(n);
  for (uint64_t i = 0; i < n; ++i) {
    res.set_coeff(i, a.get_coeff(i - p));
  }
  return res;
}
static rnx_f64 poly_automorphism(const int64_t p, const rnx_f64& a) {
  uint64_t n = a.nn();
  rnx_f64 res(n);
  for (uint64_t i = 0; i < n; ++i) {
    res.set_coeff(i * p, a.get_coeff(i));
  }
  return res;
}

TEST(vec_rnx, vec_rnx_mul_xp_minus_one) {
  test_vec_rnx_elemw_mul_xp_minus_one(vec_rnx_mul_xp_minus_one, poly_mul_xp_minus_one);
}
TEST(vec_rnx, vec_rnx_mul_xp_minus_one_ref) {
  test_vec_rnx_elemw_mul_xp_minus_one(vec_rnx_mul_xp_minus_one_ref, poly_mul_xp_minus_one);
}

TEST(vec_rnx, vec_rnx_rotate) { test_vec_rnx_elemw_rotate(vec_rnx_rotate, poly_rotate); }
TEST(vec_rnx, vec_rnx_rotate_ref) { test_vec_rnx_elemw_rotate(vec_rnx_rotate_ref, poly_rotate); }
TEST(vec_rnx, vec_rnx_automorphism) { test_vec_rnx_elemw_automorphism(vec_rnx_automorphism, poly_automorphism); }
TEST(vec_rnx, vec_rnx_automorphism_ref) {
  test_vec_rnx_elemw_automorphism(vec_rnx_automorphism_ref, poly_automorphism);
}

// test for out of place calls
template <typename ACTUAL_FCN, typename EXPECT_FCN>
void test_vec_rnx_elemw_unop_outplace(ACTUAL_FCN actual_function, EXPECT_FCN ref_function) {
  for (uint64_t n : {2, 4, 8, 128}) {
    MOD_RNX* mod = new_rnx_module_info(n, FFT64);
    for (uint64_t sa : {7, 13, 15}) {
      for (uint64_t sb : {7, 13, 15}) {
        {
          uint64_t a_sl = uniform_u64_bits(3) * 5 + n;
          uint64_t b_sl = uniform_u64_bits(3) * 4 + n;
          rnx_vec_f64_layout la(n, sa, a_sl);
          rnx_vec_f64_layout lb(n, sb, b_sl);
          std::vector<rnx_f64> expect(sb);
          for (uint64_t i = 0; i < sa; ++i) {
            la.set(i, rnx_f64::random_log2bound(n, 0));
          }
          for (uint64_t i = 0; i < sb; ++i) {
            expect[i] = ref_function(la.get_copy_zext(i));
          }
          actual_function(mod,                  //
                          lb.data(), sb, b_sl,  //
                          la.data(), sa, a_sl   //
          );
          for (uint64_t i = 0; i < sb; ++i) {
            ASSERT_EQ(lb.get_copy_zext(i), expect[i]) << n << " " << sa << " " << sb << " " << i;
          }
        }
      }
    }
    delete_rnx_module_info(mod);
  }
}

// test for inplace calls
template <typename ACTUAL_FCN, typename EXPECT_FCN>
void test_vec_rnx_elemw_unop_inplace(ACTUAL_FCN actual_function, EXPECT_FCN ref_function) {
  for (uint64_t n : {2, 16, 1024}) {
    MOD_RNX* mod = new_rnx_module_info(n, FFT64);
    for (uint64_t sa : {2, 6, 11}) {
      {
        uint64_t a_sl = uniform_u64_bits(3) * 5 + n;
        rnx_vec_f64_layout la(n, sa, a_sl);
        std::vector<rnx_f64> expect(sa);
        for (uint64_t i = 0; i < sa; ++i) {
          la.set(i, rnx_f64::random_log2bound(n, 0));
        }
        for (uint64_t i = 0; i < sa; ++i) {
          expect[i] = ref_function(la.get_copy_zext(i));
        }
        actual_function(mod,                  // N
                        la.data(), sa, a_sl,  // res
                        la.data(), sa, a_sl   // a
        );
        for (uint64_t i = 0; i < sa; ++i) {
          ASSERT_EQ(la.get_copy_zext(i), expect[i]) << n << " " << sa << " " << i;
        }
      }
    }
    delete_rnx_module_info(mod);
  }
}
template <typename ACTUAL_FCN, typename EXPECT_FCN>
void test_vec_rnx_elemw_unop(ACTUAL_FCN unnop, EXPECT_FCN ref_unnop) {
  test_vec_rnx_elemw_unop_outplace(unnop, ref_unnop);
  test_vec_rnx_elemw_unop_inplace(unnop, ref_unnop);
}

static rnx_f64 poly_copy(const rnx_f64& a) { return a; }
static rnx_f64 poly_negate(const rnx_f64& a) { return -a; }

TEST(vec_rnx, vec_rnx_copy) { test_vec_rnx_elemw_unop(vec_rnx_copy, poly_copy); }
TEST(vec_rnx, vec_rnx_copy_ref) { test_vec_rnx_elemw_unop(vec_rnx_copy_ref, poly_copy); }
TEST(vec_rnx, vec_rnx_negate) { test_vec_rnx_elemw_unop(vec_rnx_negate, poly_negate); }
TEST(vec_rnx, vec_rnx_negate_ref) { test_vec_rnx_elemw_unop(vec_rnx_negate_ref, poly_negate); }
#ifdef __x86_64__
TEST(vec_rnx, vec_rnx_negate_avx) { test_vec_rnx_elemw_unop(vec_rnx_negate_avx, poly_negate); }
#endif

// test for inplace calls
void test_vec_rnx_zero(VEC_RNX_ZERO_F actual_function) {
  for (uint64_t n : {2, 16, 1024}) {
    MOD_RNX* mod = new_rnx_module_info(n, FFT64);
    for (uint64_t sa : {2, 6, 11}) {
      {
        uint64_t a_sl = uniform_u64_bits(3) * 5 + n;
        rnx_vec_f64_layout la(n, sa, a_sl);
        const rnx_f64 ZERO = rnx_f64::zero(n);
        for (uint64_t i = 0; i < sa; ++i) {
          la.set(i, rnx_f64::random_log2bound(n, 0));
        }
        actual_function(mod,                 // N
                        la.data(), sa, a_sl  // res
        );
        for (uint64_t i = 0; i < sa; ++i) {
          ASSERT_EQ(la.get_copy_zext(i), ZERO) << n << " " << sa << " " << i;
        }
      }
    }
    delete_rnx_module_info(mod);
  }
}

TEST(vec_rnx, vec_rnx_zero) { test_vec_rnx_zero(vec_rnx_zero); }

TEST(vec_rnx, vec_rnx_zero_ref) { test_vec_rnx_zero(vec_rnx_zero_ref); }
