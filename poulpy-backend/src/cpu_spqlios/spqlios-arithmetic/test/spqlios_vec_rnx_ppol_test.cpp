#include <gtest/gtest.h>

#include "spqlios/arithmetic/vec_rnx_arithmetic_private.h"
#include "spqlios/reim/reim_fft.h"
#include "test/testlib/vec_rnx_layout.h"

static void test_vec_rnx_svp_prepare(RNX_SVP_PREPARE_F* rnx_svp_prepare, BYTES_OF_RNX_SVP_PPOL_F* tmp_bytes) {
  for (uint64_t n : {2, 4, 8, 64}) {
    MOD_RNX* mod = new_rnx_module_info(n, FFT64);
    const double invm = 1. / mod->m;

    rnx_f64 in = rnx_f64::random_log2bound(n, 40);
    rnx_f64 in_divide_by_m = rnx_f64::zero(n);
    for (uint64_t i = 0; i < n; ++i) {
      in_divide_by_m.set_coeff(i, in.get_coeff(i) * invm);
    }
    fft64_rnx_svp_ppol_layout out(n);
    reim_fft64vec expect = simple_fft64(in_divide_by_m);
    rnx_svp_prepare(mod, out.data, in.data());
    const double* ed = (double*)expect.data();
    const double* ac = (double*)out.data;
    for (uint64_t i = 0; i < n; ++i) {
      ASSERT_LE(abs(ed[i] - ac[i]), 1e-10) << i << n;
    }
    delete_rnx_module_info(mod);
  }
}
TEST(vec_rnx, vec_rnx_svp_prepare) { test_vec_rnx_svp_prepare(rnx_svp_prepare, bytes_of_rnx_svp_ppol); }
TEST(vec_rnx, vec_rnx_svp_prepare_ref) {
  test_vec_rnx_svp_prepare(fft64_rnx_svp_prepare_ref, fft64_bytes_of_rnx_svp_ppol);
}

static void test_vec_rnx_svp_apply(RNX_SVP_APPLY_F* apply) {
  for (uint64_t n : {2, 4, 8, 64, 128}) {
    MOD_RNX* mod = new_rnx_module_info(n, FFT64);

    // poly 1 to multiply - create and prepare
    fft64_rnx_svp_ppol_layout ppol(n);
    ppol.fill_random(1.);
    for (uint64_t sa : {3, 5, 8}) {
      for (uint64_t sr : {3, 5, 8}) {
        uint64_t a_sl = n + uniform_u64_bits(2);
        uint64_t r_sl = n + uniform_u64_bits(2);
        // poly 2 to multiply
        rnx_vec_f64_layout a(n, sa, a_sl);
        a.fill_random(19);

        // original operation result
        rnx_vec_f64_layout res(n, sr, r_sl);
        thash hash_a_before = a.content_hash();
        thash hash_ppol_before = ppol.content_hash();
        apply(mod, res.data(), sr, r_sl, ppol.data, a.data(), sa, a_sl);
        ASSERT_EQ(a.content_hash(), hash_a_before);
        ASSERT_EQ(ppol.content_hash(), hash_ppol_before);
        // create expected value
        reim_fft64vec ppo = ppol.get_copy();
        std::vector<rnx_f64> expect(sr);
        for (uint64_t i = 0; i < sr; ++i) {
          expect[i] = simple_ifft64(ppo * simple_fft64(a.get_copy_zext(i)));
        }
        // this is the largest precision we can safely expect
        double prec_expect = n * pow(2., 19 - 50);
        for (uint64_t i = 0; i < sr; ++i) {
          rnx_f64 actual = res.get_copy_zext(i);
          ASSERT_LE(infty_dist(actual, expect[i]), prec_expect);
        }
      }
    }
    delete_rnx_module_info(mod);
  }
}
TEST(vec_rnx, vec_rnx_svp_apply) { test_vec_rnx_svp_apply(rnx_svp_apply); }
TEST(vec_rnx, vec_rnx_svp_apply_ref) { test_vec_rnx_svp_apply(fft64_rnx_svp_apply_ref); }
