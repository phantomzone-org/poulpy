#include "gtest/gtest.h"
#include "spqlios/arithmetic/vec_rnx_arithmetic_private.h"
#include "testlib/vec_rnx_layout.h"

static void test_rnx_approxdecomp(RNX_APPROXDECOMP_FROM_TNXDBL_F approxdec) {
  for (const uint64_t nn : {2, 4, 8, 32}) {
    MOD_RNX* module = new_rnx_module_info(nn, FFT64);
    for (const uint64_t ell : {1, 2, 7}) {
      for (const uint64_t k : {2, 5}) {
        TNXDBL_APPROXDECOMP_GADGET* gadget = new_tnxdbl_approxdecomp_gadget(module, k, ell);
        for (const uint64_t res_size : {ell, ell - 1, ell + 1}) {
          const uint64_t res_sl = nn + uniform_u64_bits(2);
          rnx_vec_f64_layout in(nn, 1, nn);
          in.fill_random(3);
          rnx_vec_f64_layout out(nn, res_size, res_sl);
          approxdec(module, gadget, out.data(), res_size, res_sl, in.data());
          // reconstruct the output
          uint64_t msize = std::min(res_size, ell);
          double err_bnd = msize == ell ? pow(2., -double(msize * k) - 1) : pow(2., -double(msize * k));
          for (uint64_t j = 0; j < nn; ++j) {
            double in_j = in.data()[j];
            double out_j = 0;
            for (uint64_t i = 0; i < res_size; ++i) {
              out_j += out.get_copy(i).get_coeff(j) * pow(2., -double((i + 1) * k));
            }
            double err = out_j - in_j;
            double err_abs = fabs(err - rint(err));
            ASSERT_LE(err_abs, err_bnd);
          }
        }
        delete_tnxdbl_approxdecomp_gadget(gadget);
      }
    }
    delete_rnx_module_info(module);
  }
}

TEST(vec_rnx, rnx_approxdecomp) { test_rnx_approxdecomp(rnx_approxdecomp_from_tnxdbl); }
TEST(vec_rnx, rnx_approxdecomp_ref) { test_rnx_approxdecomp(rnx_approxdecomp_from_tnxdbl_ref); }
#ifdef __x86_64__
TEST(vec_rnx, rnx_approxdecomp_avx) { test_rnx_approxdecomp(rnx_approxdecomp_from_tnxdbl_avx); }
#endif
