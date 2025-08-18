#include <gtest/gtest.h>

#include "../spqlios/arithmetic/vec_znx_arithmetic_private.h"
#include "testlib/negacyclic_polynomial.h"

static void test_znx_small_single_product(ZNX_SMALL_SINGLE_PRODUCT_F product,
                                          ZNX_SMALL_SINGLE_PRODUCT_TMP_BYTES_F product_tmp_bytes) {
  for (const uint64_t nn : {2, 4, 8, 64}) {
    MODULE* module = new_module_info(nn, FFT64);
    znx_i64 a = znx_i64::random_log2bound(nn, 20);
    znx_i64 b = znx_i64::random_log2bound(nn, 20);
    znx_i64 expect = naive_product(a, b);
    znx_i64 actual(nn);
    std::vector<uint8_t> tmp(znx_small_single_product_tmp_bytes(module, nn));
    fft64_znx_small_single_product(module, actual.data(), a.data(), b.data(), tmp.data());
    ASSERT_EQ(actual, expect) << actual.get_coeff(0) << " vs. " << expect.get_coeff(0);
    delete_module_info(module);
  }
}

TEST(znx_small, fft64_znx_small_single_product) {
  test_znx_small_single_product(fft64_znx_small_single_product, fft64_znx_small_single_product_tmp_bytes);
}
TEST(znx_small, znx_small_single_product) {
  test_znx_small_single_product(znx_small_single_product, znx_small_single_product_tmp_bytes);
}
