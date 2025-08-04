#include <gtest/gtest.h>

#include <cstdint>

#include "../spqlios/arithmetic/vec_znx_arithmetic_private.h"
#include "spqlios/arithmetic/vec_znx_arithmetic.h"
#include "test/testlib/ntt120_dft.h"
#include "test/testlib/ntt120_layouts.h"
#include "testlib/fft64_dft.h"
#include "testlib/fft64_layouts.h"
#include "testlib/polynomial_vector.h"

static void test_fft64_vec_znx_dft(VEC_ZNX_DFT_F dft) {
  for (uint64_t n : {2, 4, 128}) {
    MODULE* module = new_module_info(n, FFT64);
    for (uint64_t sa : {3, 5, 8}) {
      for (uint64_t sr : {3, 5, 8}) {
        uint64_t a_sl = n + uniform_u64_bits(2);
        znx_vec_i64_layout a(n, sa, a_sl);
        fft64_vec_znx_dft_layout res(n, sr);
        a.fill_random(42);
        std::vector<reim_fft64vec> expect(sr);
        for (uint64_t i = 0; i < sr; ++i) {
          expect[i] = simple_fft64(a.get_copy_zext(i));
        }
        // test the function
        thash hash_before = a.content_hash();
        dft(module, res.data, sr, a.data(), sa, a_sl);
        ASSERT_EQ(a.content_hash(), hash_before);
        for (uint64_t i = 0; i < sr; ++i) {
          reim_fft64vec actual = res.get_copy_zext(i);
          ASSERT_LE(infty_dist(actual, expect[i]), 1e-10);
        }
      }
    }
    delete_module_info(module);
  }
}

static void test_fft64_vec_dft_add(VEC_DFT_ADD_F dft_add) {
  for (uint64_t n : {2, 4, 128}) {
    MODULE* module = new_module_info(n, FFT64);
    for (uint64_t sa : {3, 5, 8}) {
      for (uint64_t sr : {3, 5, 8}) {
        fft64_vec_znx_dft_layout a(n, sa);
        fft64_vec_znx_dft_layout b(n, sa);
        fft64_vec_znx_dft_layout res(n, sr);
        a.fill_dft_random_log2bound(42);
        b.fill_dft_random_log2bound(42);
        std::vector<reim_fft64vec> expect(sr);
        for (uint64_t i = 0; i < sr; ++i) {
          std::vector<double> v(n);
          if (i < sa) {
            reim_fftvec_add(module->mod.fft64.add_fft, v.data(), ((double*)a.data) + i * n, ((double*)b.data) + i * n);
          } else {
            std::fill(v.begin(), v.end(), 0.0);
          }
          expect[i] = reim_fft64vec(n, v.data());
        }
        // test the function
        thash a_hash_before = a.content_hash();
        thash b_hash_before = b.content_hash();
        dft_add(module, res.data, sr, a.data, sa, b.data, sa);
        ASSERT_EQ(a.content_hash(), a_hash_before);
        ASSERT_EQ(b.content_hash(), b_hash_before);
        for (uint64_t i = 0; i < sr; ++i) {
          reim_fft64vec actual = res.get_copy_zext(i);
          ASSERT_LE(infty_dist(actual, expect[i]), 1e-10);
        }
      }
    }
    delete_module_info(module);
  }
}

#ifdef __x86_64__
// FIXME: currently, it only works on avx
static void test_ntt120_vec_znx_dft(VEC_ZNX_DFT_F dft) {
  for (uint64_t n : {2, 4, 128}) {
    MODULE* module = new_module_info(n, NTT120);
    for (uint64_t sa : {3, 5, 8}) {
      for (uint64_t sr : {3, 5, 8}) {
        uint64_t a_sl = n + uniform_u64_bits(2);
        znx_vec_i64_layout a(n, sa, a_sl);
        ntt120_vec_znx_dft_layout res(n, sr);
        a.fill_random(42);
        std::vector<q120_nttvec> expect(sr);
        for (uint64_t i = 0; i < sr; ++i) {
          expect[i] = simple_ntt120(a.get_copy_zext(i));
        }
        // test the function
        thash hash_before = a.content_hash();
        dft(module, res.data, sr, a.data(), sa, a_sl);
        ASSERT_EQ(a.content_hash(), hash_before);
        for (uint64_t i = 0; i < sr; ++i) {
          q120_nttvec actual = res.get_copy_zext(i);
          if (!(actual == expect[i])) {
            for (uint64_t j = 0; j < n; ++j) {
              std::cerr << actual.v[j] << " vs " << expect[i].v[j] << std::endl;
            }
          }
          ASSERT_EQ(actual, expect[i]);
        }
      }
    }
    delete_module_info(module);
  }
}
#endif

TEST(vec_znx_dft, fft64_vec_znx_dft) { test_fft64_vec_znx_dft(fft64_vec_znx_dft); }
#ifdef __x86_64__
// FIXME: currently, it only works on avx
TEST(vec_znx_dft, ntt120_vec_znx_dft) { test_ntt120_vec_znx_dft(ntt120_vec_znx_dft_avx); }
#endif
TEST(vec_znx_dft, vec_znx_dft) {
  test_fft64_vec_znx_dft(vec_znx_dft);
#ifdef __x86_64__
  // FIXME: currently, it only works on avx
  test_ntt120_vec_znx_dft(ntt120_vec_znx_dft_avx);
#endif
}

TEST(vec_dft_add, fft64_vec_dft_add) { test_fft64_vec_dft_add(fft64_vec_dft_add); }

static void test_fft64_vec_znx_idft(VEC_ZNX_IDFT_F idft, VEC_ZNX_IDFT_TMP_A_F idft_tmp_a,
                                    VEC_ZNX_IDFT_TMP_BYTES_F idft_tmp_bytes) {
  for (uint64_t n : {2, 4, 64, 128}) {
    MODULE* module = new_module_info(n, FFT64);
    uint64_t tmp_size = idft_tmp_bytes ? idft_tmp_bytes(module) : 0;
    std::vector<uint8_t> tmp(tmp_size);
    for (uint64_t sa : {3, 5, 8}) {
      for (uint64_t sr : {3, 5, 8}) {
        fft64_vec_znx_dft_layout a(n, sa);
        fft64_vec_znx_big_layout res(n, sr);
        a.fill_dft_random_log2bound(22);
        std::vector<znx_i64> expect(sr);
        for (uint64_t i = 0; i < sr; ++i) {
          expect[i] = simple_rint_ifft64(a.get_copy_zext(i));
        }
        // test the function
        if (idft_tmp_bytes) {
          thash hash_before = a.content_hash();
          idft(module, res.data, sr, a.data, sa, tmp.data());
          ASSERT_EQ(a.content_hash(), hash_before);
        } else {
          idft_tmp_a(module, res.data, sr, a.data, sa);
        }
        for (uint64_t i = 0; i < sr; ++i) {
          znx_i64 actual = res.get_copy_zext(i);
          // ASSERT_EQ(res.get_copy_zext(i), expect[i]);
          if (!(actual == expect[i])) {
            for (uint64_t j = 0; j < n; ++j) {
              std::cerr << actual.get_coeff(j) << " dft vs. " << expect[i].get_coeff(j) << std::endl;
            }
            FAIL();
          }
        }
      }
    }
    delete_module_info(module);
  }
}

TEST(vec_znx_dft, fft64_vec_znx_idft) {
  test_fft64_vec_znx_idft(fft64_vec_znx_idft, nullptr, fft64_vec_znx_idft_tmp_bytes);
}
TEST(vec_znx_dft, fft64_vec_znx_idft_tmp_a) { test_fft64_vec_znx_idft(nullptr, fft64_vec_znx_idft_tmp_a, nullptr); }

#ifdef __x86_64__
// FIXME: currently, it only works on avx
static void test_ntt120_vec_znx_idft(VEC_ZNX_IDFT_F idft, VEC_ZNX_IDFT_TMP_A_F idft_tmp_a,
                                     VEC_ZNX_IDFT_TMP_BYTES_F idft_tmp_bytes) {
  for (uint64_t n : {2, 4, 64, 128}) {
    MODULE* module = new_module_info(n, NTT120);
    uint64_t tmp_size = idft_tmp_bytes ? idft_tmp_bytes(module) : 0;
    std::vector<uint8_t> tmp(tmp_size);
    for (uint64_t sa : {3, 5, 8}) {
      for (uint64_t sr : {3, 5, 8}) {
        ntt120_vec_znx_dft_layout a(n, sa);
        ntt120_vec_znx_big_layout res(n, sr);
        a.fill_random();
        std::vector<znx_i128> expect(sr);
        for (uint64_t i = 0; i < sr; ++i) {
          expect[i] = simple_intt120(a.get_copy_zext(i));
        }
        // test the function
        if (idft_tmp_bytes) {
          thash hash_before = a.content_hash();
          idft(module, res.data, sr, a.data, sa, tmp.data());
          ASSERT_EQ(a.content_hash(), hash_before);
        } else {
          idft_tmp_a(module, res.data, sr, a.data, sa);
        }
        for (uint64_t i = 0; i < sr; ++i) {
          znx_i128 actual = res.get_copy_zext(i);
          ASSERT_EQ(res.get_copy_zext(i), expect[i]);
          // if (!(actual == expect[i])) {
          //   for (uint64_t j = 0; j < n; ++j) {
          //     std::cerr << actual.get_coeff(j) << " dft vs. " << expect[i].get_coeff(j) << std::endl;
          //   }
          //   FAIL();
          // }
        }
      }
    }
    delete_module_info(module);
  }
}

TEST(vec_znx_dft, ntt120_vec_znx_idft) {
  test_ntt120_vec_znx_idft(ntt120_vec_znx_idft_avx, nullptr, ntt120_vec_znx_idft_tmp_bytes_avx);
}
TEST(vec_znx_dft, ntt120_vec_znx_idft_tmp_a) {
  test_ntt120_vec_znx_idft(nullptr, ntt120_vec_znx_idft_tmp_a_avx, nullptr);
}
#endif
TEST(vec_znx_dft, vec_znx_idft) {
  test_fft64_vec_znx_idft(vec_znx_idft, nullptr, vec_znx_idft_tmp_bytes);
#ifdef __x86_64__
  // FIXME: currently, only supported on avx
  test_ntt120_vec_znx_idft(vec_znx_idft, nullptr, vec_znx_idft_tmp_bytes);
#endif
}
TEST(vec_znx_dft, vec_znx_idft_tmp_a) {
  test_fft64_vec_znx_idft(nullptr, vec_znx_idft_tmp_a, nullptr);
#ifdef __x86_64__
  // FIXME: currently, only supported on avx
  test_ntt120_vec_znx_idft(nullptr, vec_znx_idft_tmp_a, nullptr);
#endif
}
