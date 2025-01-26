#include <gtest/gtest.h>

#include <iostream>
#include <random>

#include "../spqlios/reim4/reim4_arithmetic.h"
#include "test/testlib/reim4_elem.h"

/// Actual tests

typedef typeof(reim4_extract_1blk_from_reim_ref) reim4_extract_1blk_from_reim_f;
void test_reim4_extract_1blk_from_reim(reim4_extract_1blk_from_reim_f reim4_extract_1blk_from_reim) {
  static const uint64_t numtrials = 100;
  for (uint64_t m : {4, 8, 16, 1024, 4096, 32768}) {
    double* v = (double*)malloc(2 * m * sizeof(double));
    double* w = (double*)malloc(8 * sizeof(double));
    reim_view vv(m, v);
    for (uint64_t i = 0; i < numtrials; ++i) {
      reim4_elem el = gaussian_reim4();
      uint64_t blk = rand() % (m / 4);
      vv.set_blk(blk, el);
      reim4_extract_1blk_from_reim(m, blk, w, v);
      reim4_elem actual(w);
      ASSERT_EQ(el, actual);
    }
    free(v);
    free(w);
  }
}

TEST(reim4_arithmetic, reim4_extract_1blk_from_reim_ref) {
  test_reim4_extract_1blk_from_reim(reim4_extract_1blk_from_reim_ref);
}
#ifdef __x86_64__
TEST(reim4_arithmetic, reim4_extract_1blk_from_reim_avx) {
  test_reim4_extract_1blk_from_reim(reim4_extract_1blk_from_reim_avx);
}
#endif

typedef typeof(reim4_save_1blk_to_reim_ref) reim4_save_1blk_to_reim_f;
void test_reim4_save_1blk_to_reim(reim4_save_1blk_to_reim_f reim4_save_1blk_to_reim) {
  static const uint64_t numtrials = 100;
  for (uint64_t m : {4, 8, 16, 1024, 4096, 32768}) {
    double* v = (double*)malloc(2 * m * sizeof(double));
    double* w = (double*)malloc(8 * sizeof(double));
    reim_view vv(m, v);
    for (uint64_t i = 0; i < numtrials; ++i) {
      reim4_elem el = gaussian_reim4();
      el.save_as(w);
      uint64_t blk = rand() % (m / 4);
      reim4_save_1blk_to_reim_ref(m, blk, v, w);
      reim4_elem actual = vv.get_blk(blk);
      ASSERT_EQ(el, actual);
    }
    free(v);
    free(w);
  }
}

TEST(reim4_arithmetic, reim4_save_1blk_to_reim_ref) { test_reim4_save_1blk_to_reim(reim4_save_1blk_to_reim_ref); }
#ifdef __x86_64__
TEST(reim4_arithmetic, reim4_save_1blk_to_reim_avx) { test_reim4_save_1blk_to_reim(reim4_save_1blk_to_reim_avx); }
#endif

typedef typeof(reim4_extract_1blk_from_contiguous_reim_ref) reim4_extract_1blk_from_contiguous_reim_f;
void test_reim4_extract_1blk_from_contiguous_reim(
    reim4_extract_1blk_from_contiguous_reim_f reim4_extract_1blk_from_contiguous_reim) {
  static const uint64_t numtrials = 20;
  for (uint64_t m : {4, 8, 16, 1024, 4096, 32768}) {
    for (uint64_t nrows : {1, 2, 5, 128}) {
      double* v = (double*)malloc(2 * m * nrows * sizeof(double));
      double* w = (double*)malloc(8 * nrows * sizeof(double));
      reim_vector_view vv(m, nrows, v);
      reim4_array_view ww(nrows, w);
      for (uint64_t i = 0; i < numtrials; ++i) {
        uint64_t blk = rand() % (m / 4);
        for (uint64_t j = 0; j < nrows; ++j) {
          reim4_elem el = gaussian_reim4();
          vv.row(j).set_blk(blk, el);
        }
        reim4_extract_1blk_from_contiguous_reim_ref(m, nrows, blk, w, v);
        for (uint64_t j = 0; j < nrows; ++j) {
          reim4_elem el = vv.row(j).get_blk(blk);
          reim4_elem actual = ww.get(j);
          ASSERT_EQ(el, actual);
        }
      }
      free(v);
      free(w);
    }
  }
}

TEST(reim4_arithmetic, reim4_extract_1blk_from_contiguous_reim_ref) {
  test_reim4_extract_1blk_from_contiguous_reim(reim4_extract_1blk_from_contiguous_reim_ref);
}
#ifdef __x86_64__
TEST(reim4_arithmetic, reim4_extract_1blk_from_contiguous_reim_avx) {
  test_reim4_extract_1blk_from_contiguous_reim(reim4_extract_1blk_from_contiguous_reim_avx);
}
#endif

// test of basic arithmetic functions

TEST(reim4_arithmetic, add) {
  reim4_elem x = gaussian_reim4();
  reim4_elem y = gaussian_reim4();
  reim4_elem expect = x + y;
  reim4_elem actual;
  reim4_add(actual.value, x.value, y.value);
  ASSERT_EQ(actual, expect);
}

TEST(reim4_arithmetic, mul) {
  reim4_elem x = gaussian_reim4();
  reim4_elem y = gaussian_reim4();
  reim4_elem expect = x * y;
  reim4_elem actual;
  reim4_mul(actual.value, x.value, y.value);
  ASSERT_EQ(actual, expect);
}

TEST(reim4_arithmetic, add_mul) {
  reim4_elem x = gaussian_reim4();
  reim4_elem y = gaussian_reim4();
  reim4_elem z = gaussian_reim4();
  reim4_elem expect = z;
  reim4_elem actual = z;
  expect += x * y;
  reim4_add_mul(actual.value, x.value, y.value);
  ASSERT_EQ(actual, expect) << infty_dist(expect, actual);
}

// test of dot products

typedef typeof(reim4_vec_mat1col_product_ref) reim4_vec_mat1col_product_f;
void test_reim4_vec_mat1col_product(reim4_vec_mat1col_product_f product) {
  for (uint64_t ell : {1, 2, 5, 13, 69, 129}) {
    std::vector<double> actual(8);
    std::vector<double> a(ell * 8);
    std::vector<double> b(ell * 8);
    reim4_array_view va(ell, a.data());
    reim4_array_view vb(ell, b.data());
    reim4_array_view vactual(1, actual.data());
    // initialize random values
    for (uint64_t i = 0; i < ell; ++i) {
      va.set(i, gaussian_reim4());
      vb.set(i, gaussian_reim4());
    }
    // compute the mat1col product
    reim4_elem expect;
    for (uint64_t i = 0; i < ell; ++i) {
      expect += va.get(i) * vb.get(i);
    }
    // compute the actual product
    product(ell, actual.data(), a.data(), b.data());
    // compare
    ASSERT_LE(infty_dist(vactual.get(0), expect), 1e-10);
  }
}

TEST(reim4_arithmetic, reim4_vec_mat1col_product_ref) { test_reim4_vec_mat1col_product(reim4_vec_mat1col_product_ref); }
#ifdef __x86_64__
TEST(reim4_arena, reim4_vec_mat1col_product_avx2) { test_reim4_vec_mat1col_product(reim4_vec_mat1col_product_avx2); }
#endif

typedef typeof(reim4_vec_mat2cols_product_ref) reim4_vec_mat2col_product_f;
void test_reim4_vec_mat2cols_product(reim4_vec_mat2col_product_f product) {
  for (uint64_t ell : {1, 2, 5, 13, 69, 129}) {
    std::vector<double> actual(16);
    std::vector<double> a(ell * 8);
    std::vector<double> b(ell * 16);
    reim4_array_view va(ell, a.data());
    reim4_matrix_view vb(ell, 2, b.data());
    reim4_array_view vactual(2, actual.data());
    // initialize random values
    for (uint64_t i = 0; i < ell; ++i) {
      va.set(i, gaussian_reim4());
      vb.set(i, 0, gaussian_reim4());
      vb.set(i, 1, gaussian_reim4());
    }
    // compute the mat1col product
    reim4_elem expect[2];
    for (uint64_t i = 0; i < ell; ++i) {
      expect[0] += va.get(i) * vb.get(i, 0);
      expect[1] += va.get(i) * vb.get(i, 1);
    }
    // compute the actual product
    product(ell, actual.data(), a.data(), b.data());
    // compare
    ASSERT_LE(infty_dist(vactual.get(0), expect[0]), 1e-10);
    ASSERT_LE(infty_dist(vactual.get(1), expect[1]), 1e-10);
  }
}

TEST(reim4_arithmetic, reim4_vec_mat2cols_product_ref) {
  test_reim4_vec_mat2cols_product(reim4_vec_mat2cols_product_ref);
}
#ifdef __x86_64__
TEST(reim4_arithmetic, reim4_vec_mat2cols_product_avx2) {
  test_reim4_vec_mat2cols_product(reim4_vec_mat2cols_product_avx2);
}
#endif

// for now, we do not need avx implementations,
// so we will keep a single test function
TEST(reim4_arithmetic, reim4_vec_convolution_ref) {
  for (uint64_t sizea : {1, 2, 3, 5, 8}) {
    for (uint64_t sizeb : {1, 3, 6, 9, 13}) {
      std::vector<double> a(8 * sizea);
      std::vector<double> b(8 * sizeb);
      std::vector<double> expect(8 * (sizea + sizeb - 1));
      std::vector<double> actual(8 * (sizea + sizeb - 1));
      reim4_array_view va(sizea, a.data());
      reim4_array_view vb(sizeb, b.data());
      std::vector<reim4_elem> vexpect(sizea + sizeb + 3);
      reim4_array_view vactual(sizea + sizeb - 1, actual.data());
      for (uint64_t i = 0; i < sizea; ++i) {
        va.set(i, gaussian_reim4());
      }
      for (uint64_t j = 0; j < sizeb; ++j) {
        vb.set(j, gaussian_reim4());
      }
      // manual convolution
      for (uint64_t i = 0; i < sizea; ++i) {
        for (uint64_t j = 0; j < sizeb; ++j) {
          vexpect[i + j] += va.get(i) * vb.get(j);
        }
      }
      // partial convolution single coeff
      for (uint64_t k = 0; k < sizea + sizeb + 3; ++k) {
        double dest[8] = {0};
        reim4_convolution_1coeff_ref(k, dest, a.data(), sizea, b.data(), sizeb);
        ASSERT_LE(infty_dist(reim4_elem(dest), vexpect[k]), 1e-10);
      }
      // partial convolution dual coeff
      for (uint64_t k = 0; k < sizea + sizeb + 2; ++k) {
        double dest[16] = {0};
        reim4_convolution_2coeff_ref(k, dest, a.data(), sizea, b.data(), sizeb);
        ASSERT_LE(infty_dist(reim4_elem(dest), vexpect[k]), 1e-10);
        ASSERT_LE(infty_dist(reim4_elem(dest + 8), vexpect[k + 1]), 1e-10);
      }
      // actual convolution
      reim4_convolution_ref(actual.data(), sizea + sizeb - 1, 0, a.data(), sizea, b.data(), sizeb);
      for (uint64_t k = 0; k < sizea + sizeb - 1; ++k) {
        ASSERT_LE(infty_dist(vactual.get(k), vexpect[k]), 1e-10) << k;
      }
    }
  }
}

EXPORT void reim4_convolution_ref(double* dest, uint64_t dest_size, uint64_t dest_offset, const double* a,
                                  uint64_t sizea, const double* b, uint64_t sizeb);
