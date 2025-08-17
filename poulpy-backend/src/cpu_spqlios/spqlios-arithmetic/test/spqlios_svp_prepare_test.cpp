#include <gtest/gtest.h>

#include "../spqlios/arithmetic/vec_znx_arithmetic_private.h"
#include "testlib/fft64_dft.h"
#include "testlib/fft64_layouts.h"
#include "testlib/polynomial_vector.h"

// todo: remove when registered
typedef typeof(fft64_svp_prepare_ref) SVP_PREPARE_F;

void test_fft64_svp_prepare(SVP_PREPARE_F svp_prepare) {
  for (uint64_t n : {2, 4, 8, 64, 128}) {
    MODULE* module = new_module_info(n, FFT64);
    znx_i64 in = znx_i64::random_log2bound(n, 40);
    fft64_svp_ppol_layout out(n);
    reim_fft64vec expect = simple_fft64(in);
    svp_prepare(module, out.data, in.data());
    const double* ed = (double*)expect.data();
    const double* ac = (double*)out.data;
    for (uint64_t i = 0; i < n; ++i) {
      ASSERT_LE(abs(ed[i] - ac[i]), 1e-10) << i << n;
    }
    delete_module_info(module);
  }
}

TEST(svp_prepare, fft64_svp_prepare_ref) { test_fft64_svp_prepare(fft64_svp_prepare_ref); }
TEST(svp_prepare, svp_prepare) { test_fft64_svp_prepare(svp_prepare); }
