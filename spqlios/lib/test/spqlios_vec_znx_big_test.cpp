#include <gtest/gtest.h>

#include "spqlios/arithmetic/vec_znx_arithmetic_private.h"
#include "test/testlib/polynomial_vector.h"
#include "testlib/fft64_layouts.h"
#include "testlib/test_commons.h"

#define def_rand_big(varname, ringdim, varsize)       \
  fft64_vec_znx_big_layout varname(ringdim, varsize); \
  varname.fill_random()

#define def_rand_small(varname, ringdim, varsize)            \
  znx_vec_i64_layout varname(ringdim, varsize, 2 * ringdim); \
  varname.fill_random()

#define test_prelude(ringdim, moduletype, dim1, dim2, dim3) \
  uint64_t n = ringdim;                                     \
  MODULE* module = new_module_info(ringdim, moduletype);    \
  for (uint64_t sa : {dim1, dim2, dim3}) {                  \
    for (uint64_t sb : {dim1, dim2, dim3}) {                \
      for (uint64_t sr : {dim1, dim2, dim3})

#define test_end() \
  }                \
  }                \
  free(module)

void test_fft64_vec_znx_big_add(VEC_ZNX_BIG_ADD_F vec_znx_big_add_fcn) {
  test_prelude(8, FFT64, 3, 5, 7) {
    fft64_vec_znx_big_layout r(n, sr);
    def_rand_big(a, n, sa);
    def_rand_big(b, n, sb);
    vec_znx_big_add_fcn(module, r.data, sr, a.data, sa, b.data, sb);
    for (uint64_t i = 0; i < sr; ++i) {
      ASSERT_EQ(r.get_copy(i), a.get_copy_zext(i) + b.get_copy_zext(i));
    }
  }
  test_end();
}
void test_fft64_vec_znx_big_add_small(VEC_ZNX_BIG_ADD_SMALL_F vec_znx_big_add_fcn) {
  test_prelude(16, FFT64, 2, 4, 5) {
    fft64_vec_znx_big_layout r(n, sr);
    def_rand_big(a, n, sa);
    def_rand_small(b, n, sb);
    vec_znx_big_add_fcn(module, r.data, sr, a.data, sa, b.data(), sb, 2 * n);
    for (uint64_t i = 0; i < sr; ++i) {
      ASSERT_EQ(r.get_copy(i), a.get_copy_zext(i) + b.get_copy_zext(i));
    }
  }
  test_end();
}
void test_fft64_vec_znx_big_add_small2(VEC_ZNX_BIG_ADD_SMALL2_F vec_znx_big_add_fcn) {
  test_prelude(64, FFT64, 3, 6, 7) {
    fft64_vec_znx_big_layout r(n, sr);
    def_rand_small(a, n, sa);
    def_rand_small(b, n, sb);
    vec_znx_big_add_fcn(module, r.data, sr, a.data(), sa, 2 * n, b.data(), sb, 2 * n);
    for (uint64_t i = 0; i < sr; ++i) {
      ASSERT_EQ(r.get_copy(i), a.get_copy_zext(i) + b.get_copy_zext(i));
    }
  }
  test_end();
}

TEST(fft64_vec_znx_big, fft64_vec_znx_big_add) { test_fft64_vec_znx_big_add(fft64_vec_znx_big_add); }
TEST(vec_znx_big, vec_znx_big_add) { test_fft64_vec_znx_big_add(vec_znx_big_add); }

TEST(fft64_vec_znx_big, fft64_vec_znx_big_add_small) { test_fft64_vec_znx_big_add_small(fft64_vec_znx_big_add_small); }
TEST(vec_znx_big, vec_znx_big_add_small) { test_fft64_vec_znx_big_add_small(vec_znx_big_add_small); }

TEST(fft64_vec_znx_big, fft64_vec_znx_big_add_small2) {
  test_fft64_vec_znx_big_add_small2(fft64_vec_znx_big_add_small2);
}
TEST(vec_znx_big, vec_znx_big_add_small2) { test_fft64_vec_znx_big_add_small2(vec_znx_big_add_small2); }

void test_fft64_vec_znx_big_sub(VEC_ZNX_BIG_SUB_F vec_znx_big_sub_fcn) {
  test_prelude(16, FFT64, 3, 5, 7) {
    fft64_vec_znx_big_layout r(n, sr);
    def_rand_big(a, n, sa);
    def_rand_big(b, n, sb);
    vec_znx_big_sub_fcn(module, r.data, sr, a.data, sa, b.data, sb);
    for (uint64_t i = 0; i < sr; ++i) {
      ASSERT_EQ(r.get_copy(i), a.get_copy_zext(i) - b.get_copy_zext(i));
    }
  }
  test_end();
}
void test_fft64_vec_znx_big_sub_small_a(VEC_ZNX_BIG_SUB_SMALL_A_F vec_znx_big_sub_fcn) {
  test_prelude(32, FFT64, 2, 4, 5) {
    fft64_vec_znx_big_layout r(n, sr);
    def_rand_small(a, n, sa);
    def_rand_big(b, n, sb);
    vec_znx_big_sub_fcn(module, r.data, sr, a.data(), sa, 2 * n, b.data, sb);
    for (uint64_t i = 0; i < sr; ++i) {
      ASSERT_EQ(r.get_copy(i), a.get_copy_zext(i) - b.get_copy_zext(i));
    }
  }
  test_end();
}
void test_fft64_vec_znx_big_sub_small_b(VEC_ZNX_BIG_SUB_SMALL_B_F vec_znx_big_sub_fcn) {
  test_prelude(16, FFT64, 2, 4, 5) {
    fft64_vec_znx_big_layout r(n, sr);
    def_rand_big(a, n, sa);
    def_rand_small(b, n, sb);
    vec_znx_big_sub_fcn(module, r.data, sr, a.data, sa, b.data(), sb, 2 * n);
    for (uint64_t i = 0; i < sr; ++i) {
      ASSERT_EQ(r.get_copy(i), a.get_copy_zext(i) - b.get_copy_zext(i));
    }
  }
  test_end();
}
void test_fft64_vec_znx_big_sub_small2(VEC_ZNX_BIG_SUB_SMALL2_F vec_znx_big_sub_fcn) {
  test_prelude(8, FFT64, 3, 6, 7) {
    fft64_vec_znx_big_layout r(n, sr);
    def_rand_small(a, n, sa);
    def_rand_small(b, n, sb);
    vec_znx_big_sub_fcn(module, r.data, sr, a.data(), sa, 2 * n, b.data(), sb, 2 * n);
    for (uint64_t i = 0; i < sr; ++i) {
      ASSERT_EQ(r.get_copy(i), a.get_copy_zext(i) - b.get_copy_zext(i));
    }
  }
  test_end();
}

TEST(fft64_vec_znx_big, fft64_vec_znx_big_sub) { test_fft64_vec_znx_big_sub(fft64_vec_znx_big_sub); }
TEST(vec_znx_big, vec_znx_big_sub) { test_fft64_vec_znx_big_sub(vec_znx_big_sub); }

TEST(fft64_vec_znx_big, fft64_vec_znx_big_sub_small_a) {
  test_fft64_vec_znx_big_sub_small_a(fft64_vec_znx_big_sub_small_a);
}
TEST(vec_znx_big, vec_znx_big_sub_small_a) { test_fft64_vec_znx_big_sub_small_a(vec_znx_big_sub_small_a); }

TEST(fft64_vec_znx_big, fft64_vec_znx_big_sub_small_b) {
  test_fft64_vec_znx_big_sub_small_b(fft64_vec_znx_big_sub_small_b);
}
TEST(vec_znx_big, vec_znx_big_sub_small_b) { test_fft64_vec_znx_big_sub_small_b(vec_znx_big_sub_small_b); }

TEST(fft64_vec_znx_big, fft64_vec_znx_big_sub_small2) {
  test_fft64_vec_znx_big_sub_small2(fft64_vec_znx_big_sub_small2);
}
TEST(vec_znx_big, vec_znx_big_sub_small2) { test_fft64_vec_znx_big_sub_small2(vec_znx_big_sub_small2); }

static void test_vec_znx_big_normalize(VEC_ZNX_BIG_NORMALIZE_BASE2K_F normalize,
                                       VEC_ZNX_BIG_NORMALIZE_BASE2K_TMP_BYTES_F normalize_tmp_bytes) {
  // in the FFT64 case, big_normalize is just a forward.
  // we will just test that the functions are callable
  uint64_t n = 16;
  uint64_t k = 12;
  MODULE* module = new_module_info(n, FFT64);
  for (uint64_t sa : {3, 5, 7}) {
    for (uint64_t sr : {3, 5, 7}) {
      uint64_t r_sl = n + 3;
      def_rand_big(a, n, sa);
      znx_vec_i64_layout r(n, sr, r_sl);
      std::vector<uint8_t> tmp_space(normalize_tmp_bytes(module));
      normalize(module, k, r.data(), sr, r_sl, a.data, sa, tmp_space.data());
    }
  }
  delete_module_info(module);
}

TEST(vec_znx_big, fft64_vec_znx_big_normalize_base2k) {
  test_vec_znx_big_normalize(fft64_vec_znx_big_normalize_base2k, fft64_vec_znx_big_normalize_base2k_tmp_bytes);
}
TEST(vec_znx_big, vec_znx_big_normalize_base2k) {
  test_vec_znx_big_normalize(vec_znx_big_normalize_base2k, vec_znx_big_normalize_base2k_tmp_bytes);
}

static void test_vec_znx_big_range_normalize(  //
    VEC_ZNX_BIG_RANGE_NORMALIZE_BASE2K_F normalize,
    VEC_ZNX_BIG_RANGE_NORMALIZE_BASE2K_TMP_BYTES_F normalize_tmp_bytes) {
  // in the FFT64 case, big_normalize is just a forward.
  // we will just test that the functions are callable
  uint64_t n = 16;
  uint64_t k = 11;
  MODULE* module = new_module_info(n, FFT64);
  for (uint64_t sa : {6, 15, 21}) {
    for (uint64_t sr : {3, 5, 7}) {
      uint64_t r_sl = n + 3;
      def_rand_big(a, n, sa);
      uint64_t a_start = uniform_u64_bits(30) % (sa / 2);
      uint64_t a_end = sa - (uniform_u64_bits(30) % (sa / 2));
      uint64_t a_step = (uniform_u64_bits(30) % 3) + 1;
      uint64_t range_size = (a_end + a_step - 1 - a_start) / a_step;
      fft64_vec_znx_big_layout aextr(n, range_size);
      for (uint64_t i = 0, idx = a_start; idx < a_end; ++i, idx += a_step) {
        aextr.set(i, a.get_copy(idx));
      }
      znx_vec_i64_layout r(n, sr, r_sl);
      znx_vec_i64_layout r2(n, sr, r_sl);
      // tmp_space is large-enough for both
      std::vector<uint8_t> tmp_space(normalize_tmp_bytes(module));
      normalize(module, k, r.data(), sr, r_sl, a.data, a_start, a_end, a_step, tmp_space.data());
      fft64_vec_znx_big_normalize_base2k(module, k, r2.data(), sr, r_sl, aextr.data, range_size, tmp_space.data());
      for (uint64_t i = 0; i < sr; ++i) {
        ASSERT_EQ(r.get_copy(i), r2.get_copy(i));
      }
    }
  }
  delete_module_info(module);
}

TEST(vec_znx_big, fft64_vec_znx_big_range_normalize_base2k) {
  test_vec_znx_big_range_normalize(fft64_vec_znx_big_range_normalize_base2k,
                                   fft64_vec_znx_big_range_normalize_base2k_tmp_bytes);
}
TEST(vec_znx_big, vec_znx_big_range_normalize_base2k) {
  test_vec_znx_big_range_normalize(vec_znx_big_range_normalize_base2k, vec_znx_big_range_normalize_base2k_tmp_bytes);
}

static void test_vec_znx_big_rotate(VEC_ZNX_BIG_ROTATE_F rotate) {
  // in the FFT64 case, big_normalize is just a forward.
  // we will just test that the functions are callable
  uint64_t n = 16;
  int64_t p = 12;
  MODULE* module = new_module_info(n, FFT64);
  for (uint64_t sa : {3, 5, 7}) {
    for (uint64_t sr : {3, 5, 7}) {
      def_rand_big(a, n, sa);
      fft64_vec_znx_big_layout r(n, sr);
      rotate(module, p, r.data, sr, a.data, sa);
      for (uint64_t i = 0; i < sr; ++i) {
        znx_i64 aa = a.get_copy_zext(i);
        znx_i64 expect(n);
        for (uint64_t j = 0; j < n; ++j) {
          expect.set_coeff(j, aa.get_coeff(int64_t(j) - p));
        }
        znx_i64 actual = r.get_copy(i);
        ASSERT_EQ(expect, actual);
      }
    }
  }
  delete_module_info(module);
}

TEST(vec_znx_big, fft64_vec_znx_big_rotate) { test_vec_znx_big_rotate(fft64_vec_znx_big_rotate); }
TEST(vec_znx_big, vec_znx_big_rotate) { test_vec_znx_big_rotate(vec_znx_big_rotate); }

static void test_vec_znx_big_automorphism(VEC_ZNX_BIG_AUTOMORPHISM_F automorphism) {
  // in the FFT64 case, big_normalize is just a forward.
  // we will just test that the functions are callable
  uint64_t n = 16;
  int64_t p = 11;
  MODULE* module = new_module_info(n, FFT64);
  for (uint64_t sa : {3, 5, 7}) {
    for (uint64_t sr : {3, 5, 7}) {
      def_rand_big(a, n, sa);
      fft64_vec_znx_big_layout r(n, sr);
      automorphism(module, p, r.data, sr, a.data, sa);
      for (uint64_t i = 0; i < sr; ++i) {
        znx_i64 aa = a.get_copy_zext(i);
        znx_i64 expect(n);
        for (uint64_t j = 0; j < n; ++j) {
          expect.set_coeff(p * j, aa.get_coeff(j));
        }
        znx_i64 actual = r.get_copy(i);
        ASSERT_EQ(expect, actual);
      }
    }
  }
  delete_module_info(module);
}

TEST(vec_znx_big, fft64_vec_znx_big_automorphism) { test_vec_znx_big_automorphism(fft64_vec_znx_big_automorphism); }
TEST(vec_znx_big, vec_znx_big_automorphism) { test_vec_znx_big_automorphism(vec_znx_big_automorphism); }
