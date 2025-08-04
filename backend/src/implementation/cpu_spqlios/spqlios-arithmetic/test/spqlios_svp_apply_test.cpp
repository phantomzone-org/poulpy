#include <gtest/gtest.h>

#include "../spqlios/arithmetic/vec_znx_arithmetic_private.h"
#include "testlib/fft64_dft.h"
#include "testlib/fft64_layouts.h"
#include "testlib/polynomial_vector.h"

void test_fft64_svp_apply_dft(SVP_APPLY_DFT_F svp) {
  for (uint64_t n : {2, 4, 8, 64, 128}) {
    MODULE* module = new_module_info(n, FFT64);
    // poly 1 to multiply - create and prepare
    fft64_svp_ppol_layout ppol(n);
    ppol.fill_random(1.);
    for (uint64_t sa : {3, 5, 8}) {
      for (uint64_t sr : {3, 5, 8}) {
        uint64_t a_sl = n + uniform_u64_bits(2);
        // poly 2 to multiply
        znx_vec_i64_layout a(n, sa, a_sl);
        a.fill_random(19);
        // original operation result
        fft64_vec_znx_dft_layout res(n, sr);
        thash hash_a_before = a.content_hash();
        thash hash_ppol_before = ppol.content_hash();
        svp(module, res.data, sr, ppol.data, a.data(), sa, a_sl);
        ASSERT_EQ(a.content_hash(), hash_a_before);
        ASSERT_EQ(ppol.content_hash(), hash_ppol_before);
        // create expected value
        reim_fft64vec ppo = ppol.get_copy();
        std::vector<reim_fft64vec> expect(sr);
        for (uint64_t i = 0; i < sr; ++i) {
          expect[i] = ppo * simple_fft64(a.get_copy_zext(i));
        }
        // this is the largest precision we can safely expect
        double prec_expect = n * pow(2., 19 - 52);
        for (uint64_t i = 0; i < sr; ++i) {
          reim_fft64vec actual = res.get_copy_zext(i);
          ASSERT_LE(infty_dist(actual, expect[i]), prec_expect);
        }
      }
    }

    delete_module_info(module);
  }
}

TEST(fft64_svp_apply_dft, svp_apply_dft) { test_fft64_svp_apply_dft(svp_apply_dft); }
TEST(fft64_svp_apply_dft, fft64_svp_apply_dft_ref) { test_fft64_svp_apply_dft(fft64_svp_apply_dft_ref); }