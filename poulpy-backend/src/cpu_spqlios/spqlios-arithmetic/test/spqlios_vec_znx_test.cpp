#include <cstdint>
#include <utility>

#include "../spqlios/arithmetic/vec_znx_arithmetic.h"
#include "gtest/gtest.h"
#include "spqlios/arithmetic/vec_znx_arithmetic_private.h"
#include "spqlios/coeffs/coeffs_arithmetic.h"
#include "test/testlib/mod_q120.h"
#include "test/testlib/negacyclic_polynomial.h"
#include "testlib/fft64_dft.h"
#include "testlib/polynomial_vector.h"

TEST(fft64_layouts, dft_idft_fft64) {
  uint64_t n = 128;
  // create a random polynomial
  znx_i64 p(n);
  for (uint64_t i = 0; i < n; ++i) {
    p.set_coeff(i, uniform_i64_bits(36));
  }
  // call fft
  reim_fft64vec q = simple_fft64(p);
  // call ifft and round
  znx_i64 r = simple_rint_ifft64(q);
  ASSERT_EQ(p, r);
}

TEST(znx_layout, valid_test) {
  uint64_t n = 4;
  znx_vec_i64_layout v(n, 7, 13);
  // this should be ok
  v.set(0, znx_i64::zero(n));
  // this should be ok
  ASSERT_EQ(v.get_copy_zext(0), znx_i64::zero(n));
  ASSERT_EQ(v.data()[2], 0);  // should be ok
  // this is also ok (zero extended vector)
  ASSERT_EQ(v.get_copy_zext(1000), znx_i64::zero(n));
}

// disabling this test by default, since it depicts on purpose wrong accesses
#if 0
TEST(znx_layout, valgrind_antipattern_test) {
  uint64_t n = 4;
  znx_vec_i64_layout v(n, 7, 13);
  // this should be ok
  v.set(0, znx_i64::zero(n));
  // this should abort (wrong ring dimension)
  ASSERT_DEATH(v.set(3, znx_i64::zero(2 * n)), "");
  // this should abort (out of bounds)
  ASSERT_DEATH(v.set(8, znx_i64::zero(n)), "");
  // this should be ok
  ASSERT_EQ(v.get_copy_zext(0), znx_i64::zero(n));
  // should be an uninit read
  ASSERT_TRUE(!(v.get_copy_zext(2) == znx_i64::zero(n)));  // should be uninit
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
void test_vec_znx_elemw_binop_outplace(ACTUAL_FCN binop, EXPECT_FCN ref_binop) {
  for (uint64_t n : {2, 4, 8, 128}) {
    MODULE_TYPE mtype = uniform_u64() % 2 == 0 ? FFT64 : NTT120;
    MODULE* mod = new_module_info(n, mtype);
    for (uint64_t sa : {7, 13, 15}) {
      for (uint64_t sb : {7, 13, 15}) {
        for (uint64_t sc : {7, 13, 15}) {
          uint64_t a_sl = uniform_u64_bits(3) * 5 + n;
          uint64_t b_sl = uniform_u64_bits(3) * 5 + n;
          uint64_t c_sl = uniform_u64_bits(3) * 5 + n;
          znx_vec_i64_layout la(n, sa, a_sl);
          znx_vec_i64_layout lb(n, sb, b_sl);
          znx_vec_i64_layout lc(n, sc, c_sl);
          std::vector<znx_i64> expect(sc);
          for (uint64_t i = 0; i < sa; ++i) {
            la.set(i, znx_i64::random_log2bound(n, 62));
          }
          for (uint64_t i = 0; i < sb; ++i) {
            lb.set(i, znx_i64::random_log2bound(n, 62));
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
    delete_module_info(mod);
  }
}
// test for inplace1 calls
template <typename ACTUAL_FCN, typename EXPECT_FCN>
void test_vec_znx_elemw_binop_inplace1(ACTUAL_FCN binop, EXPECT_FCN ref_binop) {
  for (uint64_t n : {2, 4, 64}) {
    MODULE_TYPE mtype = uniform_u64() % 2 == 0 ? FFT64 : NTT120;
    MODULE* mod = new_module_info(n, mtype);
    for (uint64_t sa : {3, 9, 12}) {
      for (uint64_t sb : {3, 9, 12}) {
        uint64_t a_sl = uniform_u64_bits(3) * 5 + n;
        uint64_t b_sl = uniform_u64_bits(3) * 5 + n;
        znx_vec_i64_layout la(n, sa, a_sl);
        znx_vec_i64_layout lb(n, sb, b_sl);
        std::vector<znx_i64> expect(sa);
        for (uint64_t i = 0; i < sa; ++i) {
          la.set(i, znx_i64::random_log2bound(n, 62));
        }
        for (uint64_t i = 0; i < sb; ++i) {
          lb.set(i, znx_i64::random_log2bound(n, 62));
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
    delete_module_info(mod);
  }
}
// test for inplace2 calls
template <typename ACTUAL_FCN, typename EXPECT_FCN>
void test_vec_znx_elemw_binop_inplace2(ACTUAL_FCN binop, EXPECT_FCN ref_binop) {
  for (uint64_t n : {4, 32, 64}) {
    MODULE_TYPE mtype = uniform_u64() % 2 == 0 ? FFT64 : NTT120;
    MODULE* mod = new_module_info(n, mtype);
    for (uint64_t sa : {3, 9, 12}) {
      for (uint64_t sb : {3, 9, 12}) {
        uint64_t a_sl = uniform_u64_bits(3) * 5 + n;
        uint64_t b_sl = uniform_u64_bits(3) * 5 + n;
        znx_vec_i64_layout la(n, sa, a_sl);
        znx_vec_i64_layout lb(n, sb, b_sl);
        std::vector<znx_i64> expect(sb);
        for (uint64_t i = 0; i < sa; ++i) {
          la.set(i, znx_i64::random_log2bound(n, 62));
        }
        for (uint64_t i = 0; i < sb; ++i) {
          lb.set(i, znx_i64::random_log2bound(n, 62));
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
    delete_module_info(mod);
  }
}
// test for inplace3 calls
template <typename ACTUAL_FCN, typename EXPECT_FCN>
void test_vec_znx_elemw_binop_inplace3(ACTUAL_FCN binop, EXPECT_FCN ref_binop) {
  for (uint64_t n : {2, 16, 1024}) {
    MODULE_TYPE mtype = uniform_u64() % 2 == 0 ? FFT64 : NTT120;
    MODULE* mod = new_module_info(n, mtype);
    for (uint64_t sa : {2, 6, 11}) {
      uint64_t a_sl = uniform_u64_bits(3) * 5 + n;
      znx_vec_i64_layout la(n, sa, a_sl);
      std::vector<znx_i64> expect(sa);
      for (uint64_t i = 0; i < sa; ++i) {
        la.set(i, znx_i64::random_log2bound(n, 62));
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
    delete_module_info(mod);
  }
}
template <typename ACTUAL_FCN, typename EXPECT_FCN>
void test_vec_znx_elemw_binop(ACTUAL_FCN binop, EXPECT_FCN ref_binop) {
  test_vec_znx_elemw_binop_outplace(binop, ref_binop);
  test_vec_znx_elemw_binop_inplace1(binop, ref_binop);
  test_vec_znx_elemw_binop_inplace2(binop, ref_binop);
  test_vec_znx_elemw_binop_inplace3(binop, ref_binop);
}

static znx_i64 poly_add(const znx_i64& a, const znx_i64& b) { return a + b; }
TEST(vec_znx, vec_znx_add) { test_vec_znx_elemw_binop(vec_znx_add, poly_add); }
TEST(vec_znx, vec_znx_add_ref) { test_vec_znx_elemw_binop(vec_znx_add_ref, poly_add); }
#ifdef __x86_64__
TEST(vec_znx, vec_znx_add_avx) { test_vec_znx_elemw_binop(vec_znx_add_avx, poly_add); }
#endif

static znx_i64 poly_sub(const znx_i64& a, const znx_i64& b) { return a - b; }
TEST(vec_znx, vec_znx_sub) { test_vec_znx_elemw_binop(vec_znx_sub, poly_sub); }
TEST(vec_znx, vec_znx_sub_ref) { test_vec_znx_elemw_binop(vec_znx_sub_ref, poly_sub); }
#ifdef __x86_64__
TEST(vec_znx, vec_znx_sub_avx) { test_vec_znx_elemw_binop(vec_znx_sub_avx, poly_sub); }
#endif

// test of rotation operations

// test for out of place calls
template <typename ACTUAL_FCN, typename EXPECT_FCN>
void test_vec_znx_elemw_unop_param_outplace(ACTUAL_FCN test_rotate, EXPECT_FCN ref_rotate, int64_t (*param_gen)()) {
  for (uint64_t n : {2, 4, 8, 128}) {
    MODULE_TYPE mtype = uniform_u64() % 2 == 0 ? FFT64 : NTT120;
    MODULE* mod = new_module_info(n, mtype);
    for (uint64_t sa : {7, 13, 15}) {
      for (uint64_t sb : {7, 13, 15}) {
        {
          int64_t p = param_gen();
          uint64_t a_sl = uniform_u64_bits(3) * 5 + n;
          uint64_t b_sl = uniform_u64_bits(3) * 4 + n;
          znx_vec_i64_layout la(n, sa, a_sl);
          znx_vec_i64_layout lb(n, sb, b_sl);
          std::vector<znx_i64> expect(sb);
          for (uint64_t i = 0; i < sa; ++i) {
            la.set(i, znx_i64::random_log2bound(n, 62));
          }
          for (uint64_t i = 0; i < sb; ++i) {
            expect[i] = ref_rotate(p, la.get_copy_zext(i));
          }
          test_rotate(mod,                  //
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
    delete_module_info(mod);
  }
}

// test for inplace calls
template <typename ACTUAL_FCN, typename EXPECT_FCN>
void test_vec_znx_elemw_unop_param_inplace(ACTUAL_FCN test_rotate, EXPECT_FCN ref_rotate, int64_t (*param_gen)()) {
  for (uint64_t n : {2, 16, 1024}) {
    MODULE_TYPE mtype = uniform_u64() % 2 == 0 ? FFT64 : NTT120;
    MODULE* mod = new_module_info(n, mtype);
    for (uint64_t sa : {2, 6, 11}) {
      {
        int64_t p = param_gen();
        uint64_t a_sl = uniform_u64_bits(3) * 5 + n;
        znx_vec_i64_layout la(n, sa, a_sl);
        std::vector<znx_i64> expect(sa);
        for (uint64_t i = 0; i < sa; ++i) {
          la.set(i, znx_i64::random_log2bound(n, 62));
        }
        for (uint64_t i = 0; i < sa; ++i) {
          expect[i] = ref_rotate(p, la.get_copy_zext(i));
        }
        test_rotate(mod,                  // N
                    p,                    //;
                    la.data(), sa, a_sl,  // res
                    la.data(), sa, a_sl   // a
        );
        for (uint64_t i = 0; i < sa; ++i) {
          ASSERT_EQ(la.get_copy_zext(i), expect[i]) << n << " " << sa << " " << i;
        }
      }
    }
    delete_module_info(mod);
  }
}

static int64_t random_rotate_param() { return uniform_i64(); }

template <typename ACTUAL_FCN, typename EXPECT_FCN>
void test_vec_znx_elemw_rotate(ACTUAL_FCN binop, EXPECT_FCN ref_binop) {
  test_vec_znx_elemw_unop_param_outplace(binop, ref_binop, random_rotate_param);
  test_vec_znx_elemw_unop_param_inplace(binop, ref_binop, random_rotate_param);
}

static znx_i64 poly_rotate(const int64_t p, const znx_i64& a) {
  uint64_t n = a.nn();
  znx_i64 res(n);
  for (uint64_t i = 0; i < n; ++i) {
    res.set_coeff(i, a.get_coeff(i - p));
  }
  return res;
}
TEST(vec_znx, vec_znx_rotate) { test_vec_znx_elemw_rotate(vec_znx_rotate, poly_rotate); }
TEST(vec_znx, vec_znx_rotate_ref) { test_vec_znx_elemw_rotate(vec_znx_rotate_ref, poly_rotate); }

static int64_t random_automorphism_param() { return uniform_i64() | 1; }

template <typename ACTUAL_FCN, typename EXPECT_FCN>
void test_vec_znx_elemw_automorphism(ACTUAL_FCN unop, EXPECT_FCN ref_unop) {
  test_vec_znx_elemw_unop_param_outplace(unop, ref_unop, random_automorphism_param);
  test_vec_znx_elemw_unop_param_inplace(unop, ref_unop, random_automorphism_param);
}

static znx_i64 poly_automorphism(const int64_t p, const znx_i64& a) {
  uint64_t n = a.nn();
  znx_i64 res(n);
  for (uint64_t i = 0; i < n; ++i) {
    res.set_coeff(i * p, a.get_coeff(i));
  }
  return res;
}

TEST(vec_znx, vec_znx_automorphism) { test_vec_znx_elemw_automorphism(vec_znx_automorphism, poly_automorphism); }
TEST(vec_znx, vec_znx_automorphism_ref) {
  test_vec_znx_elemw_automorphism(vec_znx_automorphism_ref, poly_automorphism);
}

// test for out of place calls
template <typename ACTUAL_FCN, typename EXPECT_FCN>
void test_vec_znx_elemw_unop_outplace(ACTUAL_FCN test_unop, EXPECT_FCN ref_unop) {
  for (uint64_t n : {2, 4, 8, 128}) {
    MODULE_TYPE mtype = uniform_u64() % 2 == 0 ? FFT64 : NTT120;
    MODULE* mod = new_module_info(n, mtype);
    for (uint64_t sa : {7, 13, 15}) {
      for (uint64_t sb : {7, 13, 15}) {
        {
          uint64_t a_sl = uniform_u64_bits(3) * 5 + n;
          uint64_t b_sl = uniform_u64_bits(3) * 4 + n;
          znx_vec_i64_layout la(n, sa, a_sl);
          znx_vec_i64_layout lb(n, sb, b_sl);
          std::vector<znx_i64> expect(sb);
          for (uint64_t i = 0; i < sa; ++i) {
            la.set(i, znx_i64::random_log2bound(n, 62));
          }
          for (uint64_t i = 0; i < sb; ++i) {
            expect[i] = ref_unop(la.get_copy_zext(i));
          }
          test_unop(mod,                  //
                    lb.data(), sb, b_sl,  //
                    la.data(), sa, a_sl   //
          );
          for (uint64_t i = 0; i < sb; ++i) {
            ASSERT_EQ(lb.get_copy_zext(i), expect[i]) << n << " " << sa << " " << sb << " " << i;
          }
        }
      }
    }
    delete_module_info(mod);
  }
}

// test for inplace calls
template <typename ACTUAL_FCN, typename EXPECT_FCN>
void test_vec_znx_elemw_unop_inplace(ACTUAL_FCN test_unop, EXPECT_FCN ref_unop) {
  for (uint64_t n : {2, 16, 1024}) {
    MODULE_TYPE mtype = uniform_u64() % 2 == 0 ? FFT64 : NTT120;
    MODULE* mod = new_module_info(n, mtype);
    for (uint64_t sa : {2, 6, 11}) {
      {
        uint64_t a_sl = uniform_u64_bits(3) * 5 + n;
        znx_vec_i64_layout la(n, sa, a_sl);
        std::vector<znx_i64> expect(sa);
        for (uint64_t i = 0; i < sa; ++i) {
          la.set(i, znx_i64::random_log2bound(n, 62));
        }
        for (uint64_t i = 0; i < sa; ++i) {
          expect[i] = ref_unop(la.get_copy_zext(i));
        }
        test_unop(mod,                  // N
                  la.data(), sa, a_sl,  // res
                  la.data(), sa, a_sl   // a
        );
        for (uint64_t i = 0; i < sa; ++i) {
          ASSERT_EQ(la.get_copy_zext(i), expect[i]) << n << " " << sa << " " << i;
        }
      }
    }
    delete_module_info(mod);
  }
}

template <typename ACTUAL_FCN, typename EXPECT_FCN>
void test_vec_znx_elemw_unop(ACTUAL_FCN unop, EXPECT_FCN ref_unop) {
  test_vec_znx_elemw_unop_outplace(unop, ref_unop);
  test_vec_znx_elemw_unop_inplace(unop, ref_unop);
}

static znx_i64 poly_copy(const znx_i64& a) { return a; }

TEST(vec_znx, vec_znx_copy) { test_vec_znx_elemw_unop(vec_znx_copy, poly_copy); }
TEST(vec_znx, vec_znx_copy_ref) { test_vec_znx_elemw_unop(vec_znx_copy_ref, poly_copy); }

static znx_i64 poly_negate(const znx_i64& a) { return -a; }

TEST(vec_znx, vec_znx_negate) { test_vec_znx_elemw_unop(vec_znx_negate, poly_negate); }
TEST(vec_znx, vec_znx_negate_ref) { test_vec_znx_elemw_unop(vec_znx_negate_ref, poly_negate); }
#ifdef __x86_64__
TEST(vec_znx, vec_znx_negate_avx) { test_vec_znx_elemw_unop(vec_znx_negate_avx, poly_negate); }
#endif

static void test_vec_znx_zero(VEC_ZNX_ZERO_F zero) {
  for (uint64_t n : {2, 16, 1024}) {
    MODULE_TYPE mtype = uniform_u64() % 2 == 0 ? FFT64 : NTT120;
    MODULE* mod = new_module_info(n, mtype);
    for (uint64_t sa : {2, 6, 11}) {
      {
        uint64_t a_sl = uniform_u64_bits(3) * 5 + n;
        znx_vec_i64_layout la(n, sa, a_sl);
        for (uint64_t i = 0; i < sa; ++i) {
          la.set(i, znx_i64::random_log2bound(n, 62));
        }
        zero(mod,                 // N
             la.data(), sa, a_sl  // res
        );
        znx_i64 ZERO = znx_i64::zero(n);
        for (uint64_t i = 0; i < sa; ++i) {
          ASSERT_EQ(la.get_copy_zext(i), ZERO) << n << " " << sa << " " << i;
        }
      }
    }
    delete_module_info(mod);
  }
}

TEST(vec_znx, vec_znx_zero) { test_vec_znx_zero(vec_znx_zero); }
TEST(vec_znx, vec_znx_zero_ref) { test_vec_znx_zero(vec_znx_zero_ref); }

static void vec_poly_normalize(const uint64_t base_k, std::vector<znx_i64>& in) {
  if (in.size() > 0) {
    uint64_t n = in.front().nn();

    znx_i64 out = znx_i64::random_log2bound(n, 62);
    znx_i64 cinout(n);
    for (int64_t i = in.size() - 1; i >= 0; --i) {
      znx_normalize(n, base_k, in[i].data(), cinout.data(), in[i].data(), cinout.data());
    }
  }
}

template <typename ACTUAL_FCN, typename TMP_BYTES_FNC>
void test_vec_znx_normalize_outplace(ACTUAL_FCN test_normalize, TMP_BYTES_FNC tmp_bytes) {
  for (uint64_t n : {2, 16, 1024}) {
    MODULE_TYPE mtype = uniform_u64() % 2 == 0 ? FFT64 : NTT120;
    MODULE* mod = new_module_info(n, mtype);
    for (uint64_t sa : {1, 2, 6, 11}) {
      for (uint64_t sb : {1, 2, 6, 11}) {
        for (uint64_t base_k : {19}) {
          uint64_t a_sl = uniform_u64_bits(3) * 5 + n;
          znx_vec_i64_layout la(n, sa, a_sl);
          for (uint64_t i = 0; i < sa; ++i) {
            la.set(i, znx_i64::random_log2bound(n, 62));
          }

          std::vector<znx_i64> la_norm;
          for (uint64_t i = 0; i < sa; ++i) {
            la_norm.push_back(la.get_copy_zext(i));
          }
          vec_poly_normalize(base_k, la_norm);

          uint64_t b_sl = uniform_u64_bits(3) * 5 + n;
          znx_vec_i64_layout lb(n, sb, b_sl);

          const uint64_t tmp_size = tmp_bytes(mod, n);
          uint8_t* tmp = new uint8_t[tmp_size];
          test_normalize(mod,                  
                         n, // N
                         base_k,               // base_k
                         lb.data(), sb, b_sl,  // res
                         la.data(), sa, a_sl,  // a
                         tmp);
          delete[] tmp;

          for (uint64_t i = 0; i < std::min(sa, sb); ++i) {
            ASSERT_EQ(lb.get_copy_zext(i), la_norm[i]) << n << " " << sa << " " << sb << " " << i;
          }
          znx_i64 zero(n);
          for (uint64_t i = std::min(sa, sb); i < sb; ++i) {
            ASSERT_EQ(lb.get_copy_zext(i), zero) << n << " " << sa << " " << sb << " " << i;
          }
        }
      }
    }
    delete_module_info(mod);
  }
}

TEST(vec_znx, vec_znx_normalize_outplace) {
  test_vec_znx_normalize_outplace(vec_znx_normalize_base2k, vec_znx_normalize_base2k_tmp_bytes);
}
TEST(vec_znx, vec_znx_normalize_outplace_ref) {
  test_vec_znx_normalize_outplace(vec_znx_normalize_base2k_ref, vec_znx_normalize_base2k_tmp_bytes_ref);
}

template <typename ACTUAL_FCN, typename TMP_BYTES_FNC>
void test_vec_znx_normalize_inplace(ACTUAL_FCN test_normalize, TMP_BYTES_FNC tmp_bytes) {
  for (uint64_t n : {2, 16, 1024}) {
    MODULE_TYPE mtype = uniform_u64() % 2 == 0 ? FFT64 : NTT120;
    MODULE* mod = new_module_info(n, mtype);
    for (uint64_t sa : {2, 6, 11}) {
      for (uint64_t base_k : {19}) {
        uint64_t a_sl = uniform_u64_bits(3) * 5 + n;
        znx_vec_i64_layout la(n, sa, a_sl);
        for (uint64_t i = 0; i < sa; ++i) {
          la.set(i, znx_i64::random_log2bound(n, 62));
        }

        std::vector<znx_i64> la_norm;
        for (uint64_t i = 0; i < sa; ++i) {
          la_norm.push_back(la.get_copy_zext(i));
        }
        vec_poly_normalize(base_k, la_norm);

        const uint64_t tmp_size = tmp_bytes(mod, n);
        uint8_t* tmp = new uint8_t[tmp_size];
        test_normalize(mod,                  
                       n, // N
                       base_k,               // base_k
                       la.data(), sa, a_sl,  // res
                       la.data(), sa, a_sl,  // a
                       tmp);
        delete[] tmp;
        for (uint64_t i = 0; i < sa; ++i) {
          ASSERT_EQ(la.get_copy_zext(i), la_norm[i]) << n << " " << sa << " " << i;
        }
      }
    }
    delete_module_info(mod);
  }
}

TEST(vec_znx, vec_znx_normalize_inplace) {
  test_vec_znx_normalize_inplace(vec_znx_normalize_base2k, vec_znx_normalize_base2k_tmp_bytes);
}
TEST(vec_znx, vec_znx_normalize_inplace_ref) {
  test_vec_znx_normalize_inplace(vec_znx_normalize_base2k_ref, vec_znx_normalize_base2k_tmp_bytes_ref);
}
