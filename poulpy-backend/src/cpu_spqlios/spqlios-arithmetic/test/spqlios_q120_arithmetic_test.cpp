#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "spqlios/q120/q120_arithmetic.h"
#include "test/testlib/negacyclic_polynomial.h"
#include "test/testlib/ntt120_layouts.h"
#include "testlib/mod_q120.h"

typedef typeof(q120_vec_mat1col_product_baa_ref) vec_mat1col_product_baa_f;

void test_vec_mat1col_product_baa(vec_mat1col_product_baa_f vec_mat1col_product_baa) {
  q120_mat1col_product_baa_precomp* precomp = q120_new_vec_mat1col_product_baa_precomp();
  for (uint64_t ell : {1, 2, 100, 10000}) {
    std::vector<uint64_t> a(ell * 4);
    std::vector<uint64_t> b(ell * 4);
    std::vector<uint64_t> res(4);
    uint64_t* pa = a.data();
    uint64_t* pb = b.data();
    uint64_t* pr = res.data();
    // generate some random data
    uniform_q120b(pr);
    for (uint64_t i = 0; i < ell; ++i) {
      uniform_q120a(pa + 4 * i);
      uniform_q120a(pb + 4 * i);
    }
    // compute the expected result
    mod_q120 expect_r;
    for (uint64_t i = 0; i < ell; ++i) {
      expect_r += mod_q120::from_q120a(pa + 4 * i) * mod_q120::from_q120a(pb + 4 * i);
    }
    // compute the function
    vec_mat1col_product_baa(precomp, ell, (q120b*)pr, (q120a*)pa, (q120a*)pb);
    mod_q120 comp_r = mod_q120::from_q120b(pr);
    // check for equality
    ASSERT_EQ(comp_r, expect_r) << ell;
  }
  q120_delete_vec_mat1col_product_baa_precomp(precomp);
}

TEST(q120_arithmetic, q120_vec_mat1col_product_baa_ref) {
  test_vec_mat1col_product_baa(q120_vec_mat1col_product_baa_ref);
}
#ifdef __x86_64__
TEST(q120_arithmetic, q120_vec_mat1col_product_baa_avx2) {
  test_vec_mat1col_product_baa(q120_vec_mat1col_product_baa_avx2);
}
#endif

typedef typeof(q120_vec_mat1col_product_bbb_ref) vec_mat1col_product_bbb_f;

void test_vec_mat1col_product_bbb(vec_mat1col_product_bbb_f vec_mat1col_product_bbb) {
  q120_mat1col_product_bbb_precomp* precomp = q120_new_vec_mat1col_product_bbb_precomp();
  for (uint64_t ell : {1, 2, 100, 10000}) {
    std::vector<uint64_t> a(ell * 4);
    std::vector<uint64_t> b(ell * 4);
    std::vector<uint64_t> res(4);
    uint64_t* pa = a.data();
    uint64_t* pb = b.data();
    uint64_t* pr = res.data();
    // generate some random data
    uniform_q120b(pr);
    for (uint64_t i = 0; i < ell; ++i) {
      uniform_q120b(pa + 4 * i);
      uniform_q120b(pb + 4 * i);
    }
    // compute the expected result
    mod_q120 expect_r;
    for (uint64_t i = 0; i < ell; ++i) {
      expect_r += mod_q120::from_q120b(pa + 4 * i) * mod_q120::from_q120b(pb + 4 * i);
    }
    // compute the function
    vec_mat1col_product_bbb(precomp, ell, (q120b*)pr, (q120b*)pa, (q120b*)pb);
    mod_q120 comp_r = mod_q120::from_q120b(pr);
    // check for equality
    ASSERT_EQ(comp_r, expect_r);
  }
  q120_delete_vec_mat1col_product_bbb_precomp(precomp);
}

TEST(q120_arithmetic, q120_vec_mat1col_product_bbb_ref) {
  test_vec_mat1col_product_bbb(q120_vec_mat1col_product_bbb_ref);
}
#ifdef __x86_64__
TEST(q120_arithmetic, q120_vec_mat1col_product_bbb_avx2) {
  test_vec_mat1col_product_bbb(q120_vec_mat1col_product_bbb_avx2);
}
#endif

typedef typeof(q120_vec_mat1col_product_bbc_ref) vec_mat1col_product_bbc_f;

void test_vec_mat1col_product_bbc(vec_mat1col_product_bbc_f vec_mat1col_product_bbc) {
  q120_mat1col_product_bbc_precomp* precomp = q120_new_vec_mat1col_product_bbc_precomp();
  for (uint64_t ell : {1, 2, 100, 10000}) {
    std::vector<uint64_t> a(ell * 4);
    std::vector<uint64_t> b(ell * 4);
    std::vector<uint64_t> res(4);
    uint64_t* pa = a.data();
    uint64_t* pb = b.data();
    uint64_t* pr = res.data();
    // generate some random data
    uniform_q120b(pr);
    for (uint64_t i = 0; i < ell; ++i) {
      uniform_q120b(pa + 4 * i);
      uniform_q120c(pb + 4 * i);
    }
    // compute the expected result
    mod_q120 expect_r;
    for (uint64_t i = 0; i < ell; ++i) {
      expect_r += mod_q120::from_q120b(pa + 4 * i) * mod_q120::from_q120c(pb + 4 * i);
    }
    // compute the function
    vec_mat1col_product_bbc(precomp, ell, (q120b*)pr, (q120b*)pa, (q120c*)pb);
    mod_q120 comp_r = mod_q120::from_q120b(pr);
    // check for equality
    ASSERT_EQ(comp_r, expect_r);
  }
  q120_delete_vec_mat1col_product_bbc_precomp(precomp);
}

TEST(q120_arithmetic, q120_vec_mat1col_product_bbc_ref) {
  test_vec_mat1col_product_bbc(q120_vec_mat1col_product_bbc_ref);
}
#ifdef __x86_64__
TEST(q120_arithmetic, q120_vec_mat1col_product_bbc_avx2) {
  test_vec_mat1col_product_bbc(q120_vec_mat1col_product_bbc_avx2);
}
#endif

EXPORT void q120x2_vec_mat2cols_product_bbc_avx2(q120_mat1col_product_bbc_precomp* precomp, const uint64_t ell,
                                                 q120b* const res, const q120b* const x, const q120c* const y);
EXPORT void q120x2_vec_mat1col_product_bbc_avx2(q120_mat1col_product_bbc_precomp* precomp, const uint64_t ell,
                                                q120b* const res, const q120b* const x, const q120c* const y);

typedef typeof(q120x2_vec_mat2cols_product_bbc_avx2) q120x2_prod_bbc_f;

void test_q120x2_vec_mat2cols_product_bbc(q120x2_prod_bbc_f q120x2_prod_bbc) {
  q120_mat1col_product_bbc_precomp* precomp = q120_new_vec_mat1col_product_bbc_precomp();
  for (uint64_t ell : {1, 2, 100, 10000}) {
    std::vector<uint64_t> a(ell * 8);
    std::vector<uint64_t> b(ell * 16);
    std::vector<uint64_t> res(16);
    uint64_t* pa = a.data();
    uint64_t* pb = b.data();
    uint64_t* pr = res.data();
    // generate some random data
    uniform_q120b(pr);
    for (uint64_t i = 0; i < 2 * ell; ++i) {
      uniform_q120b(pa + 4 * i);
    }
    for (uint64_t i = 0; i < 4 * ell; ++i) {
      uniform_q120c(pb + 4 * i);
    }
    // compute the expected result
    mod_q120 expect_r[4];
    for (uint64_t i = 0; i < ell; ++i) {
      mod_q120 va = mod_q120::from_q120b(pa + 8 * i);
      mod_q120 vb = mod_q120::from_q120b(pa + 8 * i + 4);
      mod_q120 m1a = mod_q120::from_q120c(pb + 16 * i);
      mod_q120 m1b = mod_q120::from_q120c(pb + 16 * i + 4);
      mod_q120 m2a = mod_q120::from_q120c(pb + 16 * i + 8);
      mod_q120 m2b = mod_q120::from_q120c(pb + 16 * i + 12);
      expect_r[0] += va * m1a;
      expect_r[1] += vb * m1b;
      expect_r[2] += va * m2a;
      expect_r[3] += vb * m2b;
    }
    // compute the function
    q120x2_prod_bbc(precomp, ell, (q120b*)pr, (q120b*)pa, (q120c*)pb);
    // check for equality
    ASSERT_EQ(mod_q120::from_q120b(pr), expect_r[0]);
    ASSERT_EQ(mod_q120::from_q120b(pr + 4), expect_r[1]);
    ASSERT_EQ(mod_q120::from_q120b(pr + 8), expect_r[2]);
    ASSERT_EQ(mod_q120::from_q120b(pr + 12), expect_r[3]);
  }
  q120_delete_vec_mat1col_product_bbc_precomp(precomp);
}

TEST(q120_arithmetic, q120x2_vec_mat2cols_product_bbc_ref) {
  test_q120x2_vec_mat2cols_product_bbc(q120x2_vec_mat2cols_product_bbc_ref);
}
#ifdef __x86_64__
TEST(q120_arithmetic, q120x2_vec_mat2cols_product_bbc_avx2) {
  test_q120x2_vec_mat2cols_product_bbc(q120x2_vec_mat2cols_product_bbc_avx2);
}
#endif

typedef typeof(q120x2_vec_mat1col_product_bbc_avx2) q120x2_c1_prod_bbc_f;

void test_q120x2_vec_mat1col_product_bbc(q120x2_c1_prod_bbc_f q120x2_c1_prod_bbc) {
  q120_mat1col_product_bbc_precomp* precomp = q120_new_vec_mat1col_product_bbc_precomp();
  for (uint64_t ell : {1, 2, 100, 10000}) {
    std::vector<uint64_t> a(ell * 8);
    std::vector<uint64_t> b(ell * 8);
    std::vector<uint64_t> res(8);
    uint64_t* pa = a.data();
    uint64_t* pb = b.data();
    uint64_t* pr = res.data();
    // generate some random data
    uniform_q120b(pr);
    for (uint64_t i = 0; i < 2 * ell; ++i) {
      uniform_q120b(pa + 4 * i);
    }
    for (uint64_t i = 0; i < 2 * ell; ++i) {
      uniform_q120c(pb + 4 * i);
    }
    // compute the expected result
    mod_q120 expect_r[2];
    for (uint64_t i = 0; i < ell; ++i) {
      mod_q120 va = mod_q120::from_q120b(pa + 8 * i);
      mod_q120 vb = mod_q120::from_q120b(pa + 8 * i + 4);
      mod_q120 m1a = mod_q120::from_q120c(pb + 8 * i);
      mod_q120 m1b = mod_q120::from_q120c(pb + 8 * i + 4);
      expect_r[0] += va * m1a;
      expect_r[1] += vb * m1b;
    }
    // compute the function
    q120x2_c1_prod_bbc(precomp, ell, (q120b*)pr, (q120b*)pa, (q120c*)pb);
    // check for equality
    ASSERT_EQ(mod_q120::from_q120b(pr), expect_r[0]);
    ASSERT_EQ(mod_q120::from_q120b(pr + 4), expect_r[1]);
  }
  q120_delete_vec_mat1col_product_bbc_precomp(precomp);
}

TEST(q120_arithmetic, q120x2_vec_mat1col_product_bbc_ref) {
  test_q120x2_vec_mat1col_product_bbc(q120x2_vec_mat1col_product_bbc_ref);
}
#ifdef __x86_64__
TEST(q120_arithmetic, q120x2_vec_mat1col_product_bbc_avx2) {
  test_q120x2_vec_mat1col_product_bbc(q120x2_vec_mat1col_product_bbc_avx2);
}
#endif

typedef typeof(q120x2_extract_1blk_from_q120b_ref) q120x2_extract_f;
void test_q120x2_extract_1blk(q120x2_extract_f q120x2_extract) {
  for (uint64_t n : {2, 4, 64}) {
    ntt120_vec_znx_dft_layout v(n, 1);
    std::vector<uint64_t> r(8);
    std::vector<uint64_t> expect(8);
    for (uint64_t blk = 0; blk < n / 2; ++blk) {
      for (uint64_t i = 0; i < 8; ++i) {
        expect[i] = uniform_u64();
      }
      memcpy(v.get_blk(0, blk), expect.data(), 8 * sizeof(uint64_t));
      q120x2_extract_1blk_from_q120b_ref(n, blk, (q120x2b*)r.data(), (q120b*)v.data);
      ASSERT_EQ(r, expect);
    }
  }
}

TEST(q120_arithmetic, q120x2_extract_1blk_from_q120b_ref) {
  test_q120x2_extract_1blk(q120x2_extract_1blk_from_q120b_ref);
}

typedef typeof(q120x2_extract_1blk_from_contiguous_q120b_ref) q120x2_extract_vec_f;
void test_q120x2_extract_1blk_vec(q120x2_extract_vec_f q120x2_extract) {
  for (uint64_t n : {2, 4, 32}) {
    for (uint64_t size : {1, 2, 7}) {
      ntt120_vec_znx_dft_layout v(n, size);
      std::vector<uint64_t> r(8 * size);
      std::vector<uint64_t> expect(8 * size);
      for (uint64_t blk = 0; blk < n / 2; ++blk) {
        for (uint64_t i = 0; i < 8 * size; ++i) {
          expect[i] = uniform_u64();
        }
        for (uint64_t i = 0; i < size; ++i) {
          memcpy(v.get_blk(i, blk), expect.data() + 8 * i, 8 * sizeof(uint64_t));
        }
        q120x2_extract(n, size, blk, (q120x2b*)r.data(), (q120b*)v.data);
        ASSERT_EQ(r, expect);
      }
    }
  }
}

TEST(q120_arithmetic, q120x2_extract_1blk_from_contiguous_q120b_ref) {
  test_q120x2_extract_1blk_vec(q120x2_extract_1blk_from_contiguous_q120b_ref);
}

typedef typeof(q120x2b_save_1blk_to_q120b_ref) q120x2_save_f;
void test_q120x2_save_1blk(q120x2_save_f q120x2_save) {
  for (uint64_t n : {2, 4, 64}) {
    ntt120_vec_znx_dft_layout v(n, 1);
    std::vector<uint64_t> r(8);
    std::vector<uint64_t> expect(8);
    for (uint64_t blk = 0; blk < n / 2; ++blk) {
      for (uint64_t i = 0; i < 8; ++i) {
        expect[i] = uniform_u64();
      }
      q120x2_save(n, blk, (q120b*)v.data, (q120x2b*)expect.data());
      memcpy(r.data(), v.get_blk(0, blk), 8 * sizeof(uint64_t));
      ASSERT_EQ(r, expect);
    }
  }
}

TEST(q120_arithmetic, q120x2b_save_1blk_to_q120b_ref) { test_q120x2_save_1blk(q120x2b_save_1blk_to_q120b_ref); }

TEST(q120_arithmetic, q120_add_bbb_simple) {
  for (const uint64_t n : {2, 4, 1024}) {
    std::vector<uint64_t> a(n * 4);
    std::vector<uint64_t> b(n * 4);
    std::vector<uint64_t> r(n * 4);
    uint64_t* pa = a.data();
    uint64_t* pb = b.data();
    uint64_t* pr = r.data();

    // generate some random data
    for (uint64_t i = 0; i < n; ++i) {
      uniform_q120b(pa + 4 * i);
      uniform_q120b(pb + 4 * i);
    }

    // compute the function
    q120_add_bbb_simple(n, (q120b*)pr, (q120b*)pa, (q120b*)pb);

    for (uint64_t i = 0; i < n; ++i) {
      mod_q120 ae = mod_q120::from_q120b(pa + 4 * i);
      mod_q120 be = mod_q120::from_q120b(pb + 4 * i);
      mod_q120 re = mod_q120::from_q120b(pr + 4 * i);

      ASSERT_EQ(ae + be, re);
    }
  }
}

TEST(q120_arithmetic, q120_add_ccc_simple) {
  for (const uint64_t n : {2, 4, 1024}) {
    std::vector<uint64_t> a(n * 4);
    std::vector<uint64_t> b(n * 4);
    std::vector<uint64_t> r(n * 4);
    uint64_t* pa = a.data();
    uint64_t* pb = b.data();
    uint64_t* pr = r.data();

    // generate some random data
    for (uint64_t i = 0; i < n; ++i) {
      uniform_q120c(pa + 4 * i);
      uniform_q120c(pb + 4 * i);
    }

    // compute the function
    q120_add_ccc_simple(n, (q120c*)pr, (q120c*)pa, (q120c*)pb);

    for (uint64_t i = 0; i < n; ++i) {
      mod_q120 ae = mod_q120::from_q120c(pa + 4 * i);
      mod_q120 be = mod_q120::from_q120c(pb + 4 * i);
      mod_q120 re = mod_q120::from_q120c(pr + 4 * i);

      ASSERT_EQ(ae + be, re);
    }
  }
}

TEST(q120_arithmetic, q120_c_from_b_simple) {
  for (const uint64_t n : {2, 4, 1024}) {
    std::vector<uint64_t> a(n * 4);
    std::vector<uint64_t> r(n * 4);
    uint64_t* pa = a.data();
    uint64_t* pr = r.data();

    // generate some random data
    for (uint64_t i = 0; i < n; ++i) {
      uniform_q120b(pa + 4 * i);
    }

    // compute the function
    q120_c_from_b_simple(n, (q120c*)pr, (q120b*)pa);

    for (uint64_t i = 0; i < n; ++i) {
      mod_q120 ae = mod_q120::from_q120b(pa + 4 * i);
      mod_q120 re = mod_q120::from_q120c(pr + 4 * i);

      ASSERT_EQ(ae, re);
    }
  }
}

TEST(q120_arithmetic, q120_b_from_znx64_simple) {
  for (const uint64_t n : {2, 4, 1024}) {
    znx_i64 x = znx_i64::random_log2bound(n, 62);
    std::vector<uint64_t> r(n * 4);
    uint64_t* pr = r.data();

    q120_b_from_znx64_simple(n, (q120b*)pr, x.data());

    for (uint64_t i = 0; i < n; ++i) {
      mod_q120 re = mod_q120::from_q120b(pr + 4 * i);

      for (uint64_t k = 0; k < 4; ++k) {
        ASSERT_EQ(centermod(x.get_coeff(i), mod_q120::Qi[k]), re.a[k]);
      }
    }
  }
}

TEST(q120_arithmetic, q120_c_from_znx64_simple) {
  for (const uint64_t n : {2, 4, 1024}) {
    znx_i64 x = znx_i64::random(n);
    std::vector<uint64_t> r(n * 4);
    uint64_t* pr = r.data();

    q120_c_from_znx64_simple(n, (q120c*)pr, x.data());

    for (uint64_t i = 0; i < n; ++i) {
      mod_q120 re = mod_q120::from_q120c(pr + 4 * i);

      for (uint64_t k = 0; k < 4; ++k) {
        ASSERT_EQ(centermod(x.get_coeff(i), mod_q120::Qi[k]), re.a[k]);
      }
    }
  }
}

TEST(q120_arithmetic, q120_b_to_znx128_simple) {
  for (const uint64_t n : {2, 4, 1024}) {
    std::vector<uint64_t> x(n * 4);
    uint64_t* px = x.data();

    // generate some random data
    for (uint64_t i = 0; i < n; ++i) {
      uniform_q120b(px + 4 * i);
    }

    znx_i128 r(n);
    q120_b_to_znx128_simple(n, r.data(), (q120b*)px);

    for (uint64_t i = 0; i < n; ++i) {
      mod_q120 xe = mod_q120::from_q120b(px + 4 * i);
      for (uint64_t k = 0; k < 4; ++k) {
        ASSERT_EQ(centermod((int64_t)(r.get_coeff(i) % mod_q120::Qi[k]), mod_q120::Qi[k]), xe.a[k]);
      }
    }
  }
}
