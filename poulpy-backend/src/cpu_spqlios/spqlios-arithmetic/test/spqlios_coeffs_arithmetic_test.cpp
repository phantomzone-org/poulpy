#include <gtest/gtest.h>
#include <sys/types.h>

#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "../spqlios/coeffs/coeffs_arithmetic.h"
#include "test/testlib/mod_q120.h"
#include "testlib/negacyclic_polynomial.h"
#include "testlib/test_commons.h"

/// tests of element-wise operations
template <typename T, typename F, typename G>
void test_elemw_op(F elemw_op, G poly_elemw_op) {
  for (uint64_t n : {1, 2, 4, 8, 16, 64, 256, 4096}) {
    polynomial<T> a = polynomial<T>::random(n);
    polynomial<T> b = polynomial<T>::random(n);
    polynomial<T> expect(n);
    polynomial<T> actual(n);
    // out of place
    expect = poly_elemw_op(a, b);
    elemw_op(n, actual.data(), a.data(), b.data());
    ASSERT_EQ(actual, expect);
    // in place 1
    actual = polynomial<T>::random(n);
    expect = poly_elemw_op(actual, b);
    elemw_op(n, actual.data(), actual.data(), b.data());
    ASSERT_EQ(actual, expect);
    // in place 2
    actual = polynomial<T>::random(n);
    expect = poly_elemw_op(a, actual);
    elemw_op(n, actual.data(), a.data(), actual.data());
    ASSERT_EQ(actual, expect);
    // in place 3
    actual = polynomial<T>::random(n);
    expect = poly_elemw_op(actual, actual);
    elemw_op(n, actual.data(), actual.data(), actual.data());
    ASSERT_EQ(actual, expect);
  }
}

static polynomial<int64_t> poly_i64_add(const polynomial<int64_t>& u, polynomial<int64_t>& v) { return u + v; }
static polynomial<int64_t> poly_i64_sub(const polynomial<int64_t>& u, polynomial<int64_t>& v) { return u - v; }
TEST(coeffs_arithmetic, znx_add_i64_ref) { test_elemw_op<int64_t>(znx_add_i64_ref, poly_i64_add); }
TEST(coeffs_arithmetic, znx_sub_i64_ref) { test_elemw_op<int64_t>(znx_sub_i64_ref, poly_i64_sub); }
#ifdef __x86_64__
TEST(coeffs_arithmetic, znx_add_i64_avx) { test_elemw_op<int64_t>(znx_add_i64_avx, poly_i64_add); }
TEST(coeffs_arithmetic, znx_sub_i64_avx) { test_elemw_op<int64_t>(znx_sub_i64_avx, poly_i64_sub); }
#endif

/// tests of element-wise operations
template <typename T, typename F, typename G>
void test_elemw_unary_op(F elemw_op, G poly_elemw_op) {
  for (uint64_t n : {1, 2, 4, 8, 16, 64, 256, 4096}) {
    polynomial<T> a = polynomial<T>::random(n);
    polynomial<T> expect(n);
    polynomial<T> actual(n);
    // out of place
    expect = poly_elemw_op(a);
    elemw_op(n, actual.data(), a.data());
    ASSERT_EQ(actual, expect);
    // in place
    actual = polynomial<T>::random(n);
    expect = poly_elemw_op(actual);
    elemw_op(n, actual.data(), actual.data());
    ASSERT_EQ(actual, expect);
  }
}

static polynomial<int64_t> poly_i64_neg(const polynomial<int64_t>& u) { return -u; }
static polynomial<int64_t> poly_i64_copy(const polynomial<int64_t>& u) { return u; }
TEST(coeffs_arithmetic, znx_neg_i64_ref) { test_elemw_unary_op<int64_t>(znx_negate_i64_ref, poly_i64_neg); }
TEST(coeffs_arithmetic, znx_copy_i64_ref) { test_elemw_unary_op<int64_t>(znx_copy_i64_ref, poly_i64_copy); }
#ifdef __x86_64__
TEST(coeffs_arithmetic, znx_neg_i64_avx) { test_elemw_unary_op<int64_t>(znx_negate_i64_avx, poly_i64_neg); }
#endif

/// tests of the rotations out of place
template <typename T, typename F>
void test_rotation_outplace(F rotate) {
  for (uint64_t n : {1, 2, 4, 8, 16, 64, 256, 4096}) {
    polynomial<T> poly = polynomial<T>::random(n);
    polynomial<T> expect(n);
    polynomial<T> actual(n);
    for (uint64_t trial = 0; trial < 10; ++trial) {
      int64_t p = uniform_i64_bits(32);
      // rotate by p
      for (uint64_t i = 0; i < n; ++i) {
        expect.set_coeff(i, poly.get_coeff(i - p));
      }
      // rotate using the function
      rotate(n, p, actual.data(), poly.data());
      ASSERT_EQ(actual, expect);
    }
  }
}

TEST(coeffs_arithmetic, rnx_rotate_f64) { test_rotation_outplace<double>(rnx_rotate_f64); }
TEST(coeffs_arithmetic, znx_rotate_i64) { test_rotation_outplace<int64_t>(znx_rotate_i64); }

/// tests of the rotations out of place
template <typename T, typename F>
void test_rotation_inplace(F rotate) {
  for (uint64_t n : {1, 2, 4, 8, 16, 64, 256, 4096}) {
    polynomial<T> poly = polynomial<T>::random(n);
    polynomial<T> expect(n);
    for (uint64_t trial = 0; trial < 10; ++trial) {
      polynomial<T> actual = poly;
      int64_t p = uniform_i64_bits(32);
      // rotate by p
      for (uint64_t i = 0; i < n; ++i) {
        expect.set_coeff(i, poly.get_coeff(i - p));
      }
      // rotate using the function
      rotate(n, p, actual.data());
      ASSERT_EQ(actual, expect);
    }
  }
}

TEST(coeffs_arithmetic, rnx_rotate_inplace_f64) { test_rotation_inplace<double>(rnx_rotate_inplace_f64); }

TEST(coeffs_arithmetic, znx_rotate_inplace_i64) { test_rotation_inplace<int64_t>(znx_rotate_inplace_i64); }

/// tests of the rotations out of place
template <typename T, typename F>
void test_mul_xp_minus_one_outplace(F rotate) {
  for (uint64_t n : {1, 2, 4, 8, 16, 64, 256, 4096}) {
    polynomial<T> poly = polynomial<T>::random(n);
    polynomial<T> expect(n);
    polynomial<T> actual(n);
    for (uint64_t trial = 0; trial < 10; ++trial) {
      int64_t p = uniform_i64_bits(32);
      // rotate by p
      for (uint64_t i = 0; i < n; ++i) {
        expect.set_coeff(i, poly.get_coeff(i - p) - poly.get_coeff(i));
      }
      // rotate using the function
      rotate(n, p, actual.data(), poly.data());
      ASSERT_EQ(actual, expect);
    }
  }
}

TEST(coeffs_arithmetic, rnx_mul_xp_minus_one_f64) { test_mul_xp_minus_one_outplace<double>(rnx_mul_xp_minus_one_f64); }
TEST(coeffs_arithmetic, znx_mul_xp_minus_one_i64) { test_mul_xp_minus_one_outplace<int64_t>(znx_mul_xp_minus_one_i64); }

/// tests of the rotations out of place
template <typename T, typename F>
void test_mul_xp_minus_one_inplace(F rotate) {
  for (uint64_t n : {1, 2, 4, 8, 16, 64, 256, 4096}) {
    polynomial<T> poly = polynomial<T>::random(n);
    polynomial<T> expect(n);
    for (uint64_t trial = 0; trial < 10; ++trial) {
      polynomial<T> actual = poly;
      int64_t p = uniform_i64_bits(32);
      // rotate by p
      for (uint64_t i = 0; i < n; ++i) {
        expect.set_coeff(i, poly.get_coeff(i - p) - poly.get_coeff(i));
      }
      // rotate using the function
      rotate(n, p, actual.data());
      ASSERT_EQ(actual, expect);
    }
  }
}

TEST(coeffs_arithmetic, rnx_mul_xp_minus_one_inplace_f64) {
  test_mul_xp_minus_one_inplace<double>(rnx_mul_xp_minus_one_inplace_f64);
}

// TEST(coeffs_arithmetic, znx_mul_xp_minus_one_inplace_i64) {
// test_mul_xp_minus_one_inplace<int64_t>(znx_rotate_inplace_i64); }
/// tests of the automorphisms out of place
template <typename T, typename F>
void test_automorphism_outplace(F automorphism) {
  for (uint64_t n : {1, 2, 4, 8, 16, 64, 256, 4096}) {
    polynomial<T> poly = polynomial<T>::random(n);
    polynomial<T> expect(n);
    polynomial<T> actual(n);
    for (uint64_t trial = 0; trial < 10; ++trial) {
      int64_t p = uniform_i64_bits(32) | int64_t(1);  // make it odd
      // automorphism p
      for (uint64_t i = 0; i < n; ++i) {
        expect.set_coeff(i * p, poly.get_coeff(i));
      }
      // rotate using the function
      automorphism(n, p, actual.data(), poly.data());
      ASSERT_EQ(actual, expect);
    }
  }
}

TEST(coeffs_arithmetic, rnx_automorphism_f64) { test_automorphism_outplace<double>(rnx_automorphism_f64); }
TEST(coeffs_arithmetic, znx_automorphism_i64) { test_automorphism_outplace<int64_t>(znx_automorphism_i64); }

/// tests of the automorphisms out of place
template <typename T, typename F>
void test_automorphism_inplace(F automorphism) {
  for (uint64_t n : {1, 2, 4, 8, 16, 64, 256, 4096, 16384}) {
    polynomial<T> poly = polynomial<T>::random(n);
    polynomial<T> expect(n);
    for (uint64_t trial = 0; trial < 20; ++trial) {
      polynomial<T> actual = poly;
      int64_t p = uniform_i64_bits(32) | int64_t(1);  // make it odd
      // automorphism p
      for (uint64_t i = 0; i < n; ++i) {
        expect.set_coeff(i * p, poly.get_coeff(i));
      }
      automorphism(n, p, actual.data());
      if (!(actual == expect)) {
        std::cerr << "automorphism p: " << p << std::endl;
        for (uint64_t i = 0; i < n; ++i) {
          std::cerr << i << " " << actual.get_coeff(i) << " vs " << expect.get_coeff(i) << " "
                    << (actual.get_coeff(i) == expect.get_coeff(i)) << std::endl;
        }
      }
      ASSERT_EQ(actual, expect);
    }
  }
}
TEST(coeffs_arithmetic, rnx_automorphism_inplace_f64) {
  test_automorphism_inplace<double>(rnx_automorphism_inplace_f64);
}
TEST(coeffs_arithmetic, znx_automorphism_inplace_i64) {
  test_automorphism_inplace<int64_t>(znx_automorphism_inplace_i64);
}

// TODO: write a test later!

/**
 * @brief res = (X^p-1).in
 * @param nn the ring dimension
 * @param p must be between -2nn <= p <= 2nn
 * @param in is a rnx/znx vector of dimension nn
 */
EXPORT void rnx_mul_xp_minus_one(uint64_t nn, int64_t p, double* res, const double* in);
EXPORT void znx_mul_xp_minus_one(uint64_t nn, int64_t p, int64_t* res, const int64_t* in);

// normalize with no carry in nor carry out
template <uint8_t inplace_flag, typename F>
void test_znx_normalize(F normalize) {
  for (const uint64_t n : {1, 2, 4, 8, 16, 64, 256, 4096}) {
    polynomial<int64_t> inp = znx_i64::random_log2bound(n, 62);
    if (n >= 2) {
      inp.set_coeff(0, -(INT64_C(1) << 62));
      inp.set_coeff(1, (INT64_C(1) << 62));
    }
    for (const uint64_t base_k : {2, 3, 19, 35, 62}) {
      polynomial<int64_t> out;
      int64_t* inp_ptr;
      if (inplace_flag == 1) {
        out = polynomial<int64_t>(inp);
        inp_ptr = out.data();
      } else {
        out = polynomial<int64_t>(n);
        inp_ptr = inp.data();
      }

      znx_normalize(n, base_k, out.data(), nullptr, inp_ptr, nullptr);
      for (uint64_t i = 0; i < n; ++i) {
        const int64_t x = inp.get_coeff(i);
        const int64_t y = out.get_coeff(i);
        const int64_t y_exp = centermod(x, INT64_C(1) << base_k);
        ASSERT_EQ(y, y_exp) << n << " " << base_k << " " << i << " " << x << " " << y;
      }
    }
  }
}

TEST(coeffs_arithmetic, znx_normalize_outplace) { test_znx_normalize<0>(znx_normalize); }
TEST(coeffs_arithmetic, znx_normalize_inplace) { test_znx_normalize<1>(znx_normalize); }

// normalize with no carry in nor carry out
template <uint8_t inplace_flag, bool has_output, typename F>
void test_znx_normalize_cout(F normalize) {
  static_assert(inplace_flag < 3, "either out or cout can be inplace with inp");
  for (const uint64_t n : {1, 2, 4, 8, 16, 64, 256, 4096}) {
    polynomial<int64_t> inp = znx_i64::random_log2bound(n, 62);
    if (n >= 2) {
      inp.set_coeff(0, -(INT64_C(1) << 62));
      inp.set_coeff(1, (INT64_C(1) << 62));
    }
    for (const uint64_t base_k : {2, 3, 19, 35, 62}) {
      polynomial<int64_t> out, cout;
      int64_t* inp_ptr;
      if (inplace_flag == 1) {
        // out and inp are the same
        out = polynomial<int64_t>(inp);
        inp_ptr = out.data();
        cout = polynomial<int64_t>(n);
      } else if (inplace_flag == 2) {
        // carry out and inp are the same
        cout = polynomial<int64_t>(inp);
        inp_ptr = cout.data();
        out = polynomial<int64_t>(n);
      } else {
        // inp, out and carry out are distinct
        out = polynomial<int64_t>(n);
        cout = polynomial<int64_t>(n);
        inp_ptr = inp.data();
      }

      znx_normalize(n, base_k, has_output ? out.data() : nullptr, cout.data(), inp_ptr, nullptr);
      for (uint64_t i = 0; i < n; ++i) {
        const int64_t x = inp.get_coeff(i);
        const int64_t co = cout.get_coeff(i);
        const int64_t y_exp = centermod((int64_t)x, INT64_C(1) << base_k);
        const int64_t co_exp = (x - y_exp) >> base_k;
        ASSERT_EQ(co, co_exp);

        if (has_output) {
          const int64_t y = out.get_coeff(i);
          ASSERT_EQ(y, y_exp);
        }
      }
    }
  }
}

TEST(coeffs_arithmetic, znx_normalize_cout_outplace) { test_znx_normalize_cout<0, false>(znx_normalize); }
TEST(coeffs_arithmetic, znx_normalize_out_cout_outplace) { test_znx_normalize_cout<0, true>(znx_normalize); }
TEST(coeffs_arithmetic, znx_normalize_cout_inplace1) { test_znx_normalize_cout<1, false>(znx_normalize); }
TEST(coeffs_arithmetic, znx_normalize_out_cout_inplace1) { test_znx_normalize_cout<1, true>(znx_normalize); }
TEST(coeffs_arithmetic, znx_normalize_cout_inplace2) { test_znx_normalize_cout<2, false>(znx_normalize); }
TEST(coeffs_arithmetic, znx_normalize_out_cout_inplace2) { test_znx_normalize_cout<2, true>(znx_normalize); }

// normalize with no carry in nor carry out
template <uint8_t inplace_flag, typename F>
void test_znx_normalize_cin(F normalize) {
  static_assert(inplace_flag < 3, "either inp or cin can be inplace with out");
  for (const uint64_t n : {1, 2, 4, 8, 16, 64, 256, 4096}) {
    polynomial<int64_t> inp = znx_i64::random_log2bound(n, 62);
    if (n >= 4) {
      inp.set_coeff(0, -(INT64_C(1) << 62));
      inp.set_coeff(1, -(INT64_C(1) << 62));
      inp.set_coeff(2, (INT64_C(1) << 62));
      inp.set_coeff(3, (INT64_C(1) << 62));
    }
    for (const uint64_t base_k : {2, 3, 19, 35, 62}) {
      polynomial<int64_t> cin = znx_i64::random_log2bound(n, 62);
      if (n >= 4) {
        inp.set_coeff(0, -(INT64_C(1) << 62));
        inp.set_coeff(1, (INT64_C(1) << 62));
        inp.set_coeff(0, -(INT64_C(1) << 62));
        inp.set_coeff(1, (INT64_C(1) << 62));
      }

      polynomial<int64_t> out;
      int64_t *inp_ptr, *cin_ptr;
      if (inplace_flag == 1) {
        // out and inp are the same
        out = polynomial<int64_t>(inp);
        inp_ptr = out.data();
        cin_ptr = cin.data();
      } else if (inplace_flag == 2) {
        // out and carry in are the same
        out = polynomial<int64_t>(cin);
        inp_ptr = inp.data();
        cin_ptr = out.data();
      } else {
        // inp, carry in and out are distinct
        out = polynomial<int64_t>(n);
        inp_ptr = inp.data();
        cin_ptr = cin.data();
      }

      znx_normalize(n, base_k, out.data(), nullptr, inp_ptr, cin_ptr);
      for (uint64_t i = 0; i < n; ++i) {
        const int64_t x = inp.get_coeff(i);
        const int64_t ci = cin.get_coeff(i);
        const int64_t y = out.get_coeff(i);

        const __int128_t xp = (__int128_t)x + ci;
        const int64_t y_exp = centermod((int64_t)xp, INT64_C(1) << base_k);

        ASSERT_EQ(y, y_exp) << n << " " << base_k << " " << i << " " << x << " " << y << " " << ci;
      }
    }
  }
}

TEST(coeffs_arithmetic, znx_normalize_cin_outplace) { test_znx_normalize_cin<0>(znx_normalize); }
TEST(coeffs_arithmetic, znx_normalize_cin_inplace1) { test_znx_normalize_cin<1>(znx_normalize); }
TEST(coeffs_arithmetic, znx_normalize_cin_inplace2) { test_znx_normalize_cin<2>(znx_normalize); }

// normalize with no carry in nor carry out
template <uint8_t inplace_flag, bool has_output, typename F>
void test_znx_normalize_cin_cout(F normalize) {
  static_assert(inplace_flag < 7, "either inp or cin can be inplace with out");
  for (const uint64_t n : {1, 2, 4, 8, 16, 64, 256, 4096}) {
    polynomial<int64_t> inp = znx_i64::random_log2bound(n, 62);
    if (n >= 4) {
      inp.set_coeff(0, -(INT64_C(1) << 62));
      inp.set_coeff(1, -(INT64_C(1) << 62));
      inp.set_coeff(2, (INT64_C(1) << 62));
      inp.set_coeff(3, (INT64_C(1) << 62));
    }
    for (const uint64_t base_k : {2, 3, 19, 35, 62}) {
      polynomial<int64_t> cin = znx_i64::random_log2bound(n, 62);
      if (n >= 4) {
        inp.set_coeff(0, -(INT64_C(1) << 62));
        inp.set_coeff(1, (INT64_C(1) << 62));
        inp.set_coeff(0, -(INT64_C(1) << 62));
        inp.set_coeff(1, (INT64_C(1) << 62));
      }

      polynomial<int64_t> out, cout;
      int64_t *inp_ptr, *cin_ptr;
      if (inplace_flag == 1) {
        // out == inp
        out = polynomial<int64_t>(inp);
        cout = polynomial<int64_t>(n);
        inp_ptr = out.data();
        cin_ptr = cin.data();
      } else if (inplace_flag == 2) {
        // cout == inp
        out = polynomial<int64_t>(n);
        cout = polynomial<int64_t>(inp);
        inp_ptr = cout.data();
        cin_ptr = cin.data();
      } else if (inplace_flag == 3) {
        // out == cin
        out = polynomial<int64_t>(cin);
        cout = polynomial<int64_t>(n);
        inp_ptr = inp.data();
        cin_ptr = out.data();
      } else if (inplace_flag == 4) {
        // cout == cin
        out = polynomial<int64_t>(n);
        cout = polynomial<int64_t>(cin);
        inp_ptr = inp.data();
        cin_ptr = cout.data();
      } else if (inplace_flag == 5) {
        // out == inp, cout == cin
        out = polynomial<int64_t>(inp);
        cout = polynomial<int64_t>(cin);
        inp_ptr = out.data();
        cin_ptr = cout.data();
      } else if (inplace_flag == 6) {
        // out == cin, cout == inp
        out = polynomial<int64_t>(cin);
        cout = polynomial<int64_t>(inp);
        inp_ptr = cout.data();
        cin_ptr = out.data();
      } else {
        out = polynomial<int64_t>(n);
        cout = polynomial<int64_t>(n);
        inp_ptr = inp.data();
        cin_ptr = cin.data();
      }

      znx_normalize(n, base_k, has_output ? out.data() : nullptr, cout.data(), inp_ptr, cin_ptr);
      for (uint64_t i = 0; i < n; ++i) {
        const int64_t x = inp.get_coeff(i);
        const int64_t ci = cin.get_coeff(i);
        const int64_t co = cout.get_coeff(i);

        const __int128_t xp = (__int128_t)x + ci;
        const int64_t y_exp = centermod((int64_t)xp, INT64_C(1) << base_k);
        const int64_t co_exp = (xp - y_exp) >> base_k;
        ASSERT_EQ(co, co_exp);

        if (has_output) {
          const int64_t y = out.get_coeff(i);
          ASSERT_EQ(y, y_exp);
        }
      }
    }
  }
}

TEST(coeffs_arithmetic, znx_normalize_cin_cout_outplace) { test_znx_normalize_cin_cout<0, false>(znx_normalize); }
TEST(coeffs_arithmetic, znx_normalize_out_cin_cout_outplace) { test_znx_normalize_cin_cout<0, true>(znx_normalize); }
TEST(coeffs_arithmetic, znx_normalize_cin_cout_inplace1) { test_znx_normalize_cin_cout<1, false>(znx_normalize); }
TEST(coeffs_arithmetic, znx_normalize_out_cin_cout_inplace1) { test_znx_normalize_cin_cout<1, true>(znx_normalize); }
TEST(coeffs_arithmetic, znx_normalize_cin_cout_inplace2) { test_znx_normalize_cin_cout<2, false>(znx_normalize); }
TEST(coeffs_arithmetic, znx_normalize_out_cin_cout_inplace2) { test_znx_normalize_cin_cout<2, true>(znx_normalize); }
TEST(coeffs_arithmetic, znx_normalize_cin_cout_inplace3) { test_znx_normalize_cin_cout<3, false>(znx_normalize); }
TEST(coeffs_arithmetic, znx_normalize_out_cin_cout_inplace3) { test_znx_normalize_cin_cout<3, true>(znx_normalize); }
TEST(coeffs_arithmetic, znx_normalize_cin_cout_inplace4) { test_znx_normalize_cin_cout<4, false>(znx_normalize); }
TEST(coeffs_arithmetic, znx_normalize_out_cin_cout_inplace4) { test_znx_normalize_cin_cout<4, true>(znx_normalize); }
TEST(coeffs_arithmetic, znx_normalize_cin_cout_inplace5) { test_znx_normalize_cin_cout<5, false>(znx_normalize); }
TEST(coeffs_arithmetic, znx_normalize_out_cin_cout_inplace5) { test_znx_normalize_cin_cout<5, true>(znx_normalize); }
TEST(coeffs_arithmetic, znx_normalize_cin_cout_inplace6) { test_znx_normalize_cin_cout<6, false>(znx_normalize); }
TEST(coeffs_arithmetic, znx_normalize_out_cin_cout_inplace6) { test_znx_normalize_cin_cout<6, true>(znx_normalize); }
