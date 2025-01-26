#include <gtest/gtest.h>

#include "../spqlios/arithmetic/vec_znx_arithmetic_private.h"
#include "testlib/fft64_layouts.h"
#include "testlib/polynomial_vector.h"

static void test_vmp_prepare_contiguous(VMP_PREPARE_CONTIGUOUS_F* prepare_contiguous,
                                        VMP_PREPARE_CONTIGUOUS_TMP_BYTES_F* tmp_bytes) {
  // tests when n < 8
  for (uint64_t nn : {2, 4}) {
    MODULE* module = new_module_info(nn, FFT64);
    for (uint64_t nrows : {1, 2, 5}) {
      for (uint64_t ncols : {2, 6, 7}) {
        znx_vec_i64_layout mat(nn, nrows * ncols, nn);
        fft64_vmp_pmat_layout pmat(nn, nrows, ncols);
        mat.fill_random(30);
        std::vector<uint8_t> tmp_space(fft64_vmp_prepare_contiguous_tmp_bytes(module, nrows, ncols));
        thash hash_before = mat.content_hash();
        prepare_contiguous(module, pmat.data, mat.data(), nrows, ncols, tmp_space.data());
        ASSERT_EQ(mat.content_hash(), hash_before);
        for (uint64_t row = 0; row < nrows; ++row) {
          for (uint64_t col = 0; col < ncols; ++col) {
            const double* pmatv = (double*)pmat.data + (col * nrows + row) * nn;
            reim_fft64vec tmp = simple_fft64(mat.get_copy(row * ncols + col));
            const double* tmpv = tmp.data();
            for (uint64_t i = 0; i < nn; ++i) {
              ASSERT_LE(abs(pmatv[i] - tmpv[i]), 1e-10);
            }
          }
        }
      }
    }
    delete_module_info(module);
  }
  // tests when n >= 8
  for (uint64_t nn : {8, 32}) {
    MODULE* module = new_module_info(nn, FFT64);
    uint64_t nblk = nn / 8;
    for (uint64_t nrows : {1, 2, 5}) {
      for (uint64_t ncols : {2, 6, 7}) {
        znx_vec_i64_layout mat(nn, nrows * ncols, nn);
        fft64_vmp_pmat_layout pmat(nn, nrows, ncols);
        mat.fill_random(30);
        std::vector<uint8_t> tmp_space(tmp_bytes(module, nrows, ncols));
        thash hash_before = mat.content_hash();
        prepare_contiguous(module, pmat.data, mat.data(), nrows, ncols, tmp_space.data());
        ASSERT_EQ(mat.content_hash(), hash_before);
        for (uint64_t row = 0; row < nrows; ++row) {
          for (uint64_t col = 0; col < ncols; ++col) {
            reim_fft64vec tmp = simple_fft64(mat.get_copy(row * ncols + col));
            for (uint64_t blk = 0; blk < nblk; ++blk) {
              reim4_elem expect = tmp.get_blk(blk);
              reim4_elem actual = pmat.get(row, col, blk);
              ASSERT_LE(infty_dist(actual, expect), 1e-10);
            }
          }
        }
      }
    }
    delete_module_info(module);
  }
}

TEST(vec_znx, vmp_prepare_contiguous) {
  test_vmp_prepare_contiguous(vmp_prepare_contiguous, vmp_prepare_contiguous_tmp_bytes);
}
TEST(vec_znx, fft64_vmp_prepare_contiguous_ref) {
  test_vmp_prepare_contiguous(fft64_vmp_prepare_contiguous_ref, fft64_vmp_prepare_contiguous_tmp_bytes);
}
#ifdef __x86_64__
TEST(vec_znx, fft64_vmp_prepare_contiguous_avx) {
  test_vmp_prepare_contiguous(fft64_vmp_prepare_contiguous_avx, fft64_vmp_prepare_contiguous_tmp_bytes);
}
#endif

static void test_vmp_apply(VMP_APPLY_DFT_TO_DFT_F* apply, VMP_APPLY_DFT_TO_DFT_TMP_BYTES_F* tmp_bytes) {
  for (uint64_t nn : {2, 4, 8, 64}) {
    MODULE* module = new_module_info(nn, FFT64);
    for (uint64_t mat_nrows : {1, 4, 7}) {
      for (uint64_t mat_ncols : {1, 2, 5}) {
        for (uint64_t in_size : {1, 4, 7}) {
          for (uint64_t out_size : {1, 2, 5}) {
            fft64_vec_znx_dft_layout in(nn, in_size);
            fft64_vmp_pmat_layout pmat(nn, mat_nrows, mat_ncols);
            fft64_vec_znx_dft_layout out(nn, out_size);
            in.fill_random(0);
            pmat.fill_random(0);
            // naive computation of the product
            std::vector<reim_fft64vec> expect(out_size, reim_fft64vec(nn));
            for (uint64_t col = 0; col < out_size; ++col) {
              reim_fft64vec ex = reim_fft64vec::zero(nn);
              for (uint64_t row = 0; row < std::min(mat_nrows, in_size); ++row) {
                ex += pmat.get_zext(row, col) * in.get_copy_zext(row);
              }
              expect[col] = ex;
            }
            // apply the product
            std::vector<uint8_t> tmp(tmp_bytes(module, out_size, in_size, mat_nrows, mat_ncols));
            apply(module, out.data, out_size, in.data, in_size, pmat.data, mat_nrows, mat_ncols, tmp.data());
            // check that the output is close from the expectation
            for (uint64_t col = 0; col < out_size; ++col) {
              reim_fft64vec actual = out.get_copy_zext(col);
              ASSERT_LE(infty_dist(actual, expect[col]), 1e-10);
            }
          }
        }
      }
    }
    delete_module_info(module);
  }
}

TEST(vec_znx, vmp_apply_to_dft) { test_vmp_apply(vmp_apply_dft_to_dft, vmp_apply_dft_to_dft_tmp_bytes); }
TEST(vec_znx, fft64_vmp_apply_dft_to_dft_ref) {
  test_vmp_apply(fft64_vmp_apply_dft_to_dft_ref, fft64_vmp_apply_dft_to_dft_tmp_bytes);
}
#ifdef __x86_64__
TEST(vec_znx, fft64_vmp_apply_dft_to_dft_avx) {
  test_vmp_apply(fft64_vmp_apply_dft_to_dft_avx, fft64_vmp_apply_dft_to_dft_tmp_bytes);
}
#endif
