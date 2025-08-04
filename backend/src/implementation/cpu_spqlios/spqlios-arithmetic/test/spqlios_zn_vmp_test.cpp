#include "gtest/gtest.h"
#include "spqlios/arithmetic/zn_arithmetic_private.h"
#include "testlib/zn_layouts.h"

static void test_zn_vmp_prepare(ZN32_VMP_PREPARE_CONTIGUOUS_F prepare_contiguous) {
  MOD_Z* module = new_z_module_info(DEFAULT);
  for (uint64_t nrows : {1, 2, 5, 15}) {
    for (uint64_t ncols : {1, 2, 32, 42, 67}) {
      std::vector<int32_t> src(nrows * ncols);
      zn32_pmat_layout out(nrows, ncols);
      for (int32_t& x : src) x = uniform_i64_bits(32);
      prepare_contiguous(module, out.data, src.data(), nrows, ncols);
      for (uint64_t i = 0; i < nrows; ++i) {
        for (uint64_t j = 0; j < ncols; ++j) {
          int32_t in = src[i * ncols + j];
          int32_t actual = out.get(i, j);
          ASSERT_EQ(actual, in);
        }
      }
    }
  }
  delete_z_module_info(module);
}

TEST(zn, zn32_vmp_prepare_contiguous) { test_zn_vmp_prepare(zn32_vmp_prepare_contiguous); }
TEST(zn, default_zn32_vmp_prepare_contiguous_ref) { test_zn_vmp_prepare(default_zn32_vmp_prepare_contiguous_ref); }

static void test_zn_vmp_prepare(ZN32_VMP_PREPARE_DBLPTR_F prepare_dblptr) {
  MOD_Z* module = new_z_module_info(DEFAULT);
  for (uint64_t nrows : {1, 2, 5, 15}) {
    for (uint64_t ncols : {1, 2, 32, 42, 67}) {
      std::vector<int32_t> src(nrows * ncols);
      zn32_pmat_layout out(nrows, ncols);
      for (int32_t& x : src) x = uniform_i64_bits(32);
      const int32_t** mat_dblptr = (const int32_t**)malloc(nrows * sizeof(int32_t*));
      for (size_t row_i = 0; row_i < nrows; row_i++) {
        mat_dblptr[row_i] = &src.data()[row_i * ncols];
      };
      prepare_dblptr(module, out.data, mat_dblptr, nrows, ncols);
      for (uint64_t i = 0; i < nrows; ++i) {
        for (uint64_t j = 0; j < ncols; ++j) {
          int32_t in = src[i * ncols + j];
          int32_t actual = out.get(i, j);
          ASSERT_EQ(actual, in);
        }
      }
    }
  }
  delete_z_module_info(module);
}

TEST(zn, zn32_vmp_prepare_dblptr) { test_zn_vmp_prepare(zn32_vmp_prepare_dblptr); }
TEST(zn, default_zn32_vmp_prepare_dblptr_ref) { test_zn_vmp_prepare(default_zn32_vmp_prepare_dblptr_ref); }

template <typename INTTYPE>
static void test_zn_vmp_apply(void (*apply)(const MOD_Z*, int32_t*, uint64_t, const INTTYPE*, uint64_t,
                                            const ZN32_VMP_PMAT*, uint64_t, uint64_t)) {
  MOD_Z* module = new_z_module_info(DEFAULT);
  for (uint64_t nrows : {1, 2, 5, 15}) {
    for (uint64_t ncols : {1, 2, 32, 42, 67}) {
      for (uint64_t a_size : {1, 2, 5, 15}) {
        for (uint64_t res_size : {1, 2, 32, 42, 67}) {
          std::vector<INTTYPE> a(a_size);
          zn32_pmat_layout out(nrows, ncols);
          std::vector<int32_t> res(res_size);
          for (INTTYPE& x : a) x = uniform_i64_bits(32);
          out.fill_random();
          std::vector<int32_t> expect = vmp_product(a.data(), a_size, res_size, out);
          apply(module, res.data(), res_size, a.data(), a_size, out.data, nrows, ncols);
          for (uint64_t i = 0; i < res_size; ++i) {
            int32_t exp = expect[i];
            int32_t actual = res[i];
            ASSERT_EQ(actual, exp);
          }
        }
      }
    }
  }
  delete_z_module_info(module);
}

TEST(zn, zn32_vmp_apply_i32) { test_zn_vmp_apply(zn32_vmp_apply_i32); }
TEST(zn, zn32_vmp_apply_i16) { test_zn_vmp_apply(zn32_vmp_apply_i16); }
TEST(zn, zn32_vmp_apply_i8) { test_zn_vmp_apply(zn32_vmp_apply_i8); }

TEST(zn, default_zn32_vmp_apply_i32_ref) { test_zn_vmp_apply(default_zn32_vmp_apply_i32_ref); }
TEST(zn, default_zn32_vmp_apply_i16_ref) { test_zn_vmp_apply(default_zn32_vmp_apply_i16_ref); }
TEST(zn, default_zn32_vmp_apply_i8_ref) { test_zn_vmp_apply(default_zn32_vmp_apply_i8_ref); }

#ifdef __x86_64__
TEST(zn, default_zn32_vmp_apply_i32_avx) { test_zn_vmp_apply(default_zn32_vmp_apply_i32_avx); }
TEST(zn, default_zn32_vmp_apply_i16_avx) { test_zn_vmp_apply(default_zn32_vmp_apply_i16_avx); }
TEST(zn, default_zn32_vmp_apply_i8_avx) { test_zn_vmp_apply(default_zn32_vmp_apply_i8_avx); }
#endif
