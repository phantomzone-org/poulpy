#include "gtest/gtest.h"
#include "spqlios/arithmetic/zn_arithmetic_private.h"
#include "testlib/test_commons.h"

template <typename INTTYPE>
static void test_tndbl_approxdecomp(                                                                                //
    void (*approxdec)(const MOD_Z*, const TNDBL_APPROXDECOMP_GADGET*, INTTYPE*, uint64_t, const double*, uint64_t)  //
) {
  for (const uint64_t nn : {1, 3, 8, 51}) {
    MOD_Z* module = new_z_module_info(DEFAULT);
    for (const uint64_t ell : {1, 2, 7}) {
      for (const uint64_t k : {2, 5}) {
        TNDBL_APPROXDECOMP_GADGET* gadget = new_tndbl_approxdecomp_gadget(module, k, ell);
        for (const uint64_t res_size : {ell * nn}) {
          std::vector<double> in(nn);
          std::vector<INTTYPE> out(res_size);
          for (double& x : in) x = uniform_f64_bounds(-10, 10);
          approxdec(module, gadget, out.data(), res_size, in.data(), nn);
          // reconstruct the output
          double err_bnd = pow(2., -double(ell * k) - 1);
          for (uint64_t j = 0; j < nn; ++j) {
            double in_j = in[j];
            double out_j = 0;
            for (uint64_t i = 0; i < ell; ++i) {
              out_j += out[ell * j + i] * pow(2., -double((i + 1) * k));
            }
            double err = out_j - in_j;
            double err_abs = fabs(err - rint(err));
            ASSERT_LE(err_abs, err_bnd);
          }
        }
        delete_tndbl_approxdecomp_gadget(gadget);
      }
    }
    delete_z_module_info(module);
  }
}

TEST(vec_rnx, i8_tndbl_rnx_approxdecomp) { test_tndbl_approxdecomp(i8_approxdecomp_from_tndbl); }
TEST(vec_rnx, default_i8_tndbl_rnx_approxdecomp) { test_tndbl_approxdecomp(default_i8_approxdecomp_from_tndbl_ref); }

TEST(vec_rnx, i16_tndbl_rnx_approxdecomp) { test_tndbl_approxdecomp(i16_approxdecomp_from_tndbl); }
TEST(vec_rnx, default_i16_tndbl_rnx_approxdecomp) { test_tndbl_approxdecomp(default_i16_approxdecomp_from_tndbl_ref); }

TEST(vec_rnx, i32_tndbl_rnx_approxdecomp) { test_tndbl_approxdecomp(i32_approxdecomp_from_tndbl); }
TEST(vec_rnx, default_i32_tndbl_rnx_approxdecomp) { test_tndbl_approxdecomp(default_i32_approxdecomp_from_tndbl_ref); }
