#include "gtest/gtest.h"
#include "../spqlios/arithmetic/vec_rnx_arithmetic_private.h"
#include "../spqlios/reim/reim_fft.h"
#include "testlib/vec_rnx_layout.h"

static void test_vmp_apply_dft_to_dft_outplace(  //
    RNX_VMP_APPLY_DFT_TO_DFT_F* apply,           //
    RNX_VMP_APPLY_DFT_TO_DFT_TMP_BYTES_F* tmp_bytes) {
  for (uint64_t nn : {2, 4, 8, 64}) {
    MOD_RNX* module = new_rnx_module_info(nn, FFT64);
    for (uint64_t mat_nrows : {1, 4, 7}) {
      for (uint64_t mat_ncols : {1, 2, 5}) {
        for (uint64_t in_size : {1, 4, 7}) {
          for (uint64_t out_size : {1, 2, 5}) {
            const uint64_t in_sl = nn + uniform_u64_bits(2);
            const uint64_t out_sl = nn + uniform_u64_bits(2);
            rnx_vec_f64_layout in(nn, in_size, in_sl);
            fft64_rnx_vmp_pmat_layout pmat(nn, mat_nrows, mat_ncols);
            rnx_vec_f64_layout out(nn, out_size, out_sl);
            in.fill_random(0);
            pmat.fill_random(0);
            // naive computation of the product
            std::vector<reim_fft64vec> expect(out_size, reim_fft64vec(nn));
            for (uint64_t col = 0; col < out_size; ++col) {
              reim_fft64vec ex = reim_fft64vec::zero(nn);
              for (uint64_t row = 0; row < std::min(mat_nrows, in_size); ++row) {
                ex += pmat.get_zext(row, col) * in.get_dft_copy(row);
              }
              expect[col] = ex;
            }
            // apply the product
            std::vector<uint8_t> tmp(tmp_bytes(module, out_size, in_size, mat_nrows, mat_ncols));
            apply(module,                           //
                  out.data(), out_size, out_sl,     //
                  in.data(), in_size, in_sl,        //
                  pmat.data, mat_nrows, mat_ncols,  //
                  tmp.data());
            // check that the output is close from the expectation
            for (uint64_t col = 0; col < out_size; ++col) {
              reim_fft64vec actual = out.get_dft_copy_zext(col);
              ASSERT_LE(infty_dist(actual, expect[col]), 1e-10);
            }
          }
        }
      }
    }
    delete_rnx_module_info(module);
  }
}

static void test_vmp_apply_dft_to_dft_inplace(  //
    RNX_VMP_APPLY_DFT_TO_DFT_F* apply,          //
    RNX_VMP_APPLY_DFT_TO_DFT_TMP_BYTES_F* tmp_bytes) {
  for (uint64_t nn : {2, 4, 8, 64}) {
    MOD_RNX* module = new_rnx_module_info(nn, FFT64);
    for (uint64_t mat_nrows : {1, 2, 6}) {
      for (uint64_t mat_ncols : {1, 2, 7, 8}) {
        for (uint64_t in_size : {1, 3, 6}) {
          for (uint64_t out_size : {1, 3, 6}) {
            const uint64_t in_out_sl = nn + uniform_u64_bits(2);
            rnx_vec_f64_layout in_out(nn, std::max(in_size, out_size), in_out_sl);
            fft64_rnx_vmp_pmat_layout pmat(nn, mat_nrows, mat_ncols);
            in_out.fill_random(0);
            pmat.fill_random(0);
            // naive computation of the product
            std::vector<reim_fft64vec> expect(out_size, reim_fft64vec(nn));
            for (uint64_t col = 0; col < out_size; ++col) {
              reim_fft64vec ex = reim_fft64vec::zero(nn);
              for (uint64_t row = 0; row < std::min(mat_nrows, in_size); ++row) {
                ex += pmat.get_zext(row, col) * in_out.get_dft_copy(row);
              }
              expect[col] = ex;
            }
            // apply the product
            std::vector<uint8_t> tmp(tmp_bytes(module, out_size, in_size, mat_nrows, mat_ncols));
            apply(module,                              //
                  in_out.data(), out_size, in_out_sl,  //
                  in_out.data(), in_size, in_out_sl,   //
                  pmat.data, mat_nrows, mat_ncols,     //
                  tmp.data());
            // check that the output is close from the expectation
            for (uint64_t col = 0; col < out_size; ++col) {
              reim_fft64vec actual = in_out.get_dft_copy_zext(col);
              ASSERT_LE(infty_dist(actual, expect[col]), 1e-10);
            }
          }
        }
      }
    }
    delete_rnx_module_info(module);
  }
}

static void test_vmp_apply_dft_to_dft(  //
    RNX_VMP_APPLY_DFT_TO_DFT_F* apply,  //
    RNX_VMP_APPLY_DFT_TO_DFT_TMP_BYTES_F* tmp_bytes) {
  test_vmp_apply_dft_to_dft_outplace(apply, tmp_bytes);
  test_vmp_apply_dft_to_dft_inplace(apply, tmp_bytes);
}

TEST(vec_rnx, vmp_apply_to_dft) {
  test_vmp_apply_dft_to_dft(rnx_vmp_apply_dft_to_dft, rnx_vmp_apply_dft_to_dft_tmp_bytes);
}
TEST(vec_rnx, fft64_vmp_apply_dft_to_dft_ref) {
  test_vmp_apply_dft_to_dft(fft64_rnx_vmp_apply_dft_to_dft_ref, fft64_rnx_vmp_apply_dft_to_dft_tmp_bytes_ref);
}
#ifdef __x86_64__
TEST(vec_rnx, fft64_vmp_apply_dft_to_dft_avx) {
  test_vmp_apply_dft_to_dft(fft64_rnx_vmp_apply_dft_to_dft_avx, fft64_rnx_vmp_apply_dft_to_dft_tmp_bytes_avx);
}
#endif

/// rnx_vmp_prepare

static void test_vmp_prepare_contiguous(RNX_VMP_PREPARE_CONTIGUOUS_F* prepare_contiguous,
                                        RNX_VMP_PREPARE_CONTIGUOUS_TMP_BYTES_F* tmp_bytes) {
  // tests when n < 8
  for (uint64_t nn : {2, 4}) {
    const double one_over_m = 2. / nn;
    MOD_RNX* module = new_rnx_module_info(nn, FFT64);
    for (uint64_t nrows : {1, 2, 5}) {
      for (uint64_t ncols : {2, 6, 7}) {
        rnx_vec_f64_layout mat(nn, nrows * ncols, nn);
        fft64_rnx_vmp_pmat_layout pmat(nn, nrows, ncols);
        mat.fill_random(0);
        std::vector<uint8_t> tmp_space(tmp_bytes(module));
        thash hash_before = mat.content_hash();
        prepare_contiguous(module, pmat.data, mat.data(), nrows, ncols, tmp_space.data());
        ASSERT_EQ(mat.content_hash(), hash_before);
        for (uint64_t row = 0; row < nrows; ++row) {
          for (uint64_t col = 0; col < ncols; ++col) {
            const double* pmatv = (double*)pmat.data + (col * nrows + row) * nn;
            reim_fft64vec tmp = one_over_m * simple_fft64(mat.get_copy(row * ncols + col));
            const double* tmpv = tmp.data();
            for (uint64_t i = 0; i < nn; ++i) {
              ASSERT_LE(abs(pmatv[i] - tmpv[i]), 1e-10);
            }
          }
        }
      }
    }
    delete_rnx_module_info(module);
  }
  // tests when n >= 8
  for (uint64_t nn : {8, 32}) {
    const double one_over_m = 2. / nn;
    MOD_RNX* module = new_rnx_module_info(nn, FFT64);
    uint64_t nblk = nn / 8;
    for (uint64_t nrows : {1, 2, 5}) {
      for (uint64_t ncols : {2, 6, 7}) {
        rnx_vec_f64_layout mat(nn, nrows * ncols, nn);
        fft64_rnx_vmp_pmat_layout pmat(nn, nrows, ncols);
        mat.fill_random(0);
        std::vector<uint8_t> tmp_space(tmp_bytes(module));
        thash hash_before = mat.content_hash();
        prepare_contiguous(module, pmat.data, mat.data(), nrows, ncols, tmp_space.data());
        ASSERT_EQ(mat.content_hash(), hash_before);
        for (uint64_t row = 0; row < nrows; ++row) {
          for (uint64_t col = 0; col < ncols; ++col) {
            reim_fft64vec tmp = one_over_m * simple_fft64(mat.get_copy(row * ncols + col));
            for (uint64_t blk = 0; blk < nblk; ++blk) {
              reim4_elem expect = tmp.get_blk(blk);
              reim4_elem actual = pmat.get(row, col, blk);
              ASSERT_LE(infty_dist(actual, expect), 1e-10);
            }
          }
        }
      }
    }
    delete_rnx_module_info(module);
  }
}

TEST(vec_rnx, vmp_prepare_contiguous) {
  test_vmp_prepare_contiguous(rnx_vmp_prepare_contiguous, rnx_vmp_prepare_contiguous_tmp_bytes);
}
TEST(vec_rnx, fft64_vmp_prepare_contiguous_ref) {
  test_vmp_prepare_contiguous(fft64_rnx_vmp_prepare_contiguous_ref, fft64_rnx_vmp_prepare_contiguous_tmp_bytes_ref);
}
#ifdef __x86_64__
TEST(vec_rnx, fft64_vmp_prepare_contiguous_avx) {
  test_vmp_prepare_contiguous(fft64_rnx_vmp_prepare_contiguous_avx, fft64_rnx_vmp_prepare_contiguous_tmp_bytes_avx);
}
#endif

/// rnx_vmp_apply_dft_to_dft

static void test_vmp_apply_tmp_a_outplace(  //
    RNX_VMP_APPLY_TMP_A_F* apply,           //
    RNX_VMP_APPLY_TMP_A_TMP_BYTES_F* tmp_bytes) {
  for (uint64_t nn : {2, 4, 8, 64}) {
    MOD_RNX* module = new_rnx_module_info(nn, FFT64);
    for (uint64_t mat_nrows : {1, 4, 7}) {
      for (uint64_t mat_ncols : {1, 2, 5}) {
        for (uint64_t in_size : {1, 4, 7}) {
          for (uint64_t out_size : {1, 2, 5}) {
            const uint64_t in_sl = nn + uniform_u64_bits(2);
            const uint64_t out_sl = nn + uniform_u64_bits(2);
            rnx_vec_f64_layout in(nn, in_size, in_sl);
            fft64_rnx_vmp_pmat_layout pmat(nn, mat_nrows, mat_ncols);
            rnx_vec_f64_layout out(nn, out_size, out_sl);
            in.fill_random(0);
            pmat.fill_random(0);
            // naive computation of the product
            std::vector<rnx_f64> expect(out_size);
            for (uint64_t col = 0; col < out_size; ++col) {
              reim_fft64vec ex = reim_fft64vec::zero(nn);
              for (uint64_t row = 0; row < std::min(mat_nrows, in_size); ++row) {
                ex += pmat.get_zext(row, col) * simple_fft64(in.get_copy(row));
              }
              expect[col] = simple_ifft64(ex);
            }
            // apply the product
            std::vector<uint8_t> tmp(tmp_bytes(module, out_size, in_size, mat_nrows, mat_ncols));
            apply(module,                           //
                  out.data(), out_size, out_sl,     //
                  in.data(), in_size, in_sl,        //
                  pmat.data, mat_nrows, mat_ncols,  //
                  tmp.data());
            // check that the output is close from the expectation
            for (uint64_t col = 0; col < out_size; ++col) {
              rnx_f64 actual = out.get_copy_zext(col);
              ASSERT_LE(infty_dist(actual, expect[col]), 1e-10);
            }
          }
        }
      }
    }
    delete_rnx_module_info(module);
  }
}

static void test_vmp_apply_tmp_a_inplace(  //
    RNX_VMP_APPLY_TMP_A_F* apply,          //
    RNX_VMP_APPLY_TMP_A_TMP_BYTES_F* tmp_bytes) {
  for (uint64_t nn : {2, 4, 8, 64}) {
    MOD_RNX* module = new_rnx_module_info(nn, FFT64);
    for (uint64_t mat_nrows : {1, 4, 7}) {
      for (uint64_t mat_ncols : {1, 2, 5}) {
        for (uint64_t in_size : {1, 4, 7}) {
          for (uint64_t out_size : {1, 2, 5}) {
            const uint64_t in_out_sl = nn + uniform_u64_bits(2);
            rnx_vec_f64_layout in_out(nn, std::max(in_size, out_size), in_out_sl);
            fft64_rnx_vmp_pmat_layout pmat(nn, mat_nrows, mat_ncols);
            in_out.fill_random(0);
            pmat.fill_random(0);
            // naive computation of the product
            std::vector<rnx_f64> expect(out_size);
            for (uint64_t col = 0; col < out_size; ++col) {
              reim_fft64vec ex = reim_fft64vec::zero(nn);
              for (uint64_t row = 0; row < std::min(mat_nrows, in_size); ++row) {
                ex += pmat.get_zext(row, col) * simple_fft64(in_out.get_copy(row));
              }
              expect[col] = simple_ifft64(ex);
            }
            // apply the product
            std::vector<uint8_t> tmp(tmp_bytes(module, out_size, in_size, mat_nrows, mat_ncols));
            apply(module,                              //
                  in_out.data(), out_size, in_out_sl,  //
                  in_out.data(), in_size, in_out_sl,   //
                  pmat.data, mat_nrows, mat_ncols,     //
                  tmp.data());
            // check that the output is close from the expectation
            for (uint64_t col = 0; col < out_size; ++col) {
              rnx_f64 actual = in_out.get_copy_zext(col);
              ASSERT_LE(infty_dist(actual, expect[col]), 1e-10);
            }
          }
        }
      }
    }
    delete_rnx_module_info(module);
  }
}

static void test_vmp_apply_tmp_a(  //
    RNX_VMP_APPLY_TMP_A_F* apply,  //
    RNX_VMP_APPLY_TMP_A_TMP_BYTES_F* tmp_bytes) {
  test_vmp_apply_tmp_a_outplace(apply, tmp_bytes);
  test_vmp_apply_tmp_a_inplace(apply, tmp_bytes);
}

TEST(vec_znx, fft64_vmp_apply_tmp_a) { test_vmp_apply_tmp_a(rnx_vmp_apply_tmp_a, rnx_vmp_apply_tmp_a_tmp_bytes); }
TEST(vec_znx, fft64_vmp_apply_tmp_a_ref) {
  test_vmp_apply_tmp_a(fft64_rnx_vmp_apply_tmp_a_ref, fft64_rnx_vmp_apply_tmp_a_tmp_bytes_ref);
}
#ifdef __x86_64__
TEST(vec_znx, fft64_vmp_apply_tmp_a_avx) {
  test_vmp_apply_tmp_a(fft64_rnx_vmp_apply_tmp_a_avx, fft64_rnx_vmp_apply_tmp_a_tmp_bytes_avx);
}
#endif
