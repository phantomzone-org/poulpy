#include <assert.h>
#include <immintrin.h>
#include <string.h>

#include "../coeffs/coeffs_arithmetic.h"
#include "../reim/reim_fft.h"
#include "../reim4/reim4_arithmetic.h"
#include "vec_rnx_arithmetic_private.h"

/** @brief prepares a vmp matrix (contiguous row-major version) */
EXPORT void fft64_rnx_vmp_prepare_contiguous_avx(       //
    const MOD_RNX* module,                              // N
    RNX_VMP_PMAT* pmat,                                 // output
    const double* mat, uint64_t nrows, uint64_t ncols,  // a
    uint8_t* tmp_space                                  // scratch space
) {
  // there is an edge case if nn < 8
  const uint64_t nn = module->n;
  const uint64_t m = module->m;

  double* const dtmp = (double*)tmp_space;
  double* const output_mat = (double*)pmat;
  double* start_addr = (double*)pmat;
  uint64_t offset = nrows * ncols * 8;

  if (nn >= 8) {
    for (uint64_t row_i = 0; row_i < nrows; row_i++) {
      for (uint64_t col_i = 0; col_i < ncols; col_i++) {
        rnx_divide_by_m_avx(nn, m, dtmp, mat + (row_i * ncols + col_i) * nn);
        reim_fft(module->precomp.fft64.p_fft, dtmp);

        if (col_i == (ncols - 1) && (ncols % 2 == 1)) {
          // special case: last column out of an odd column number
          start_addr = output_mat + col_i * nrows * 8  // col == ncols-1
                       + row_i * 8;
        } else {
          // general case: columns go by pair
          start_addr = output_mat + (col_i / 2) * (2 * nrows) * 8  // second: col pair index
                       + row_i * 2 * 8                             // third: row index
                       + (col_i % 2) * 8;
        }

        for (uint64_t blk_i = 0; blk_i < m / 4; blk_i++) {
          // extract blk from tmp and save it
          reim4_extract_1blk_from_reim_avx(m, blk_i, start_addr + blk_i * offset, dtmp);
        }
      }
    }
  } else {
    for (uint64_t row_i = 0; row_i < nrows; row_i++) {
      for (uint64_t col_i = 0; col_i < ncols; col_i++) {
        double* res = output_mat + (col_i * nrows + row_i) * nn;
        rnx_divide_by_m_avx(nn, m, res, mat + (row_i * ncols + col_i) * nn);
        reim_fft(module->precomp.fft64.p_fft, res);
      }
    }
  }
}

/** @brief prepares a vmp matrix (mat[row]+col*N points to the item) */
EXPORT void fft64_rnx_vmp_prepare_dblptr_avx(            //
    const MOD_RNX* module,                               // N
    RNX_VMP_PMAT* pmat,                                  // output
    const double** mat, uint64_t nrows, uint64_t ncols,  // a
    uint8_t* tmp_space                                   // scratch space
) {
  for (uint64_t row_i = 0; row_i < nrows; row_i++) {
    fft64_rnx_vmp_prepare_row_avx(module, pmat, mat[row_i], row_i, nrows, ncols, tmp_space);
  }
}

/** @brief prepares the ith-row of a vmp matrix with nrows and ncols */
EXPORT void fft64_rnx_vmp_prepare_row_avx(                              //
    const MOD_RNX* module,                                              // N
    RNX_VMP_PMAT* pmat,                                                 // output
    const double* row, uint64_t row_i, uint64_t nrows, uint64_t ncols,  // a
    uint8_t* tmp_space                                                  // scratch space
) {
  // there is an edge case if nn < 8
  const uint64_t nn = module->n;
  const uint64_t m = module->m;

  double* const dtmp = (double*)tmp_space;
  double* const output_mat = (double*)pmat;
  double* start_addr = (double*)pmat;
  uint64_t offset = nrows * ncols * 8;

  if (nn >= 8) {
    for (uint64_t col_i = 0; col_i < ncols; col_i++) {
      rnx_divide_by_m_avx(nn, m, dtmp, row + col_i * nn);
      reim_fft(module->precomp.fft64.p_fft, dtmp);

      if (col_i == (ncols - 1) && (ncols % 2 == 1)) {
        // special case: last column out of an odd column number
        start_addr = output_mat + col_i * nrows * 8  // col == ncols-1
                     + row_i * 8;
      } else {
        // general case: columns go by pair
        start_addr = output_mat + (col_i / 2) * (2 * nrows) * 8  // second: col pair index
                     + row_i * 2 * 8                             // third: row index
                     + (col_i % 2) * 8;
      }

      for (uint64_t blk_i = 0; blk_i < m / 4; blk_i++) {
        // extract blk from tmp and save it
        reim4_extract_1blk_from_reim_avx(m, blk_i, start_addr + blk_i * offset, dtmp);
      }
    }
  } else {
    for (uint64_t col_i = 0; col_i < ncols; col_i++) {
      double* res = output_mat + (col_i * nrows + row_i) * nn;
      rnx_divide_by_m_avx(nn, m, res, row + col_i * nn);
      reim_fft(module->precomp.fft64.p_fft, res);
    }
  }
}

/** @brief minimal size of the tmp_space */
EXPORT void fft64_rnx_vmp_apply_dft_to_dft_avx(                //
    const MOD_RNX* module,                                     // N
    double* res, uint64_t res_size, uint64_t res_sl,           // res
    const double* a_dft, uint64_t a_size, uint64_t a_sl,       // a
    const RNX_VMP_PMAT* pmat, uint64_t nrows, uint64_t ncols,  // prep matrix
    uint8_t* tmp_space                                         // scratch space (a_size*sizeof(reim4) bytes)
) {
  const uint64_t m = module->m;
  const uint64_t nn = module->n;

  double* mat2cols_output = (double*)tmp_space;     // 128 bytes
  double* extracted_blk = (double*)tmp_space + 16;  // 64*min(nrows,a_size) bytes

  double* mat_input = (double*)pmat;

  const uint64_t row_max = nrows < a_size ? nrows : a_size;
  const uint64_t col_max = ncols < res_size ? ncols : res_size;

  if (row_max > 0 && col_max > 0) {
    if (nn >= 8) {
      // let's do some prefetching of the GSW key, since on some cpus,
      // it helps
      const uint64_t ms4 = m >> 2;  // m/4
      const uint64_t gsw_iter_doubles = 8 * nrows * ncols;
      const uint64_t pref_doubles = 1200;
      const double* gsw_pref_ptr = mat_input;
      const double* const gsw_ptr_end = mat_input + ms4 * gsw_iter_doubles;
      const double* gsw_pref_ptr_target = mat_input + pref_doubles;
      for (; gsw_pref_ptr < gsw_pref_ptr_target; gsw_pref_ptr += 8) {
        __builtin_prefetch(gsw_pref_ptr, 0, _MM_HINT_T0);
      }
      const double* mat_blk_start;
      uint64_t blk_i;
      for (blk_i = 0, mat_blk_start = mat_input; blk_i < ms4; blk_i++, mat_blk_start += gsw_iter_doubles) {
        // prefetch the next iteration
        if (gsw_pref_ptr_target < gsw_ptr_end) {
          gsw_pref_ptr_target += gsw_iter_doubles;
          if (gsw_pref_ptr_target > gsw_ptr_end) gsw_pref_ptr_target = gsw_ptr_end;
          for (; gsw_pref_ptr < gsw_pref_ptr_target; gsw_pref_ptr += 8) {
            __builtin_prefetch(gsw_pref_ptr, 0, _MM_HINT_T0);
          }
        }
        reim4_extract_1blk_from_contiguous_reim_sl_avx(m, a_sl, row_max, blk_i, extracted_blk, a_dft);
        // apply mat2cols
        for (uint64_t col_i = 0; col_i < col_max - 1; col_i += 2) {
          uint64_t col_offset = col_i * (8 * nrows);
          reim4_vec_mat2cols_product_avx2(row_max, mat2cols_output, extracted_blk, mat_blk_start + col_offset);

          reim4_save_1blk_to_reim_avx(m, blk_i, res + col_i * res_sl, mat2cols_output);
          reim4_save_1blk_to_reim_avx(m, blk_i, res + (col_i + 1) * res_sl, mat2cols_output + 8);
        }

        // check if col_max is odd, then special case
        if (col_max % 2 == 1) {
          uint64_t last_col = col_max - 1;
          uint64_t col_offset = last_col * (8 * nrows);

          // the last column is alone in the pmat: vec_mat1col
          if (ncols == col_max) {
            reim4_vec_mat1col_product_avx2(row_max, mat2cols_output, extracted_blk, mat_blk_start + col_offset);
          } else {
            // the last column is part of a colpair in the pmat: vec_mat2cols and ignore the second position
            reim4_vec_mat2cols_product_avx2(row_max, mat2cols_output, extracted_blk, mat_blk_start + col_offset);
          }
          reim4_save_1blk_to_reim_avx(m, blk_i, res + last_col * res_sl, mat2cols_output);
        }
      }
    } else {
      const double* in;
      uint64_t in_sl;
      if (res == a_dft) {
        // it is in place: copy the input vector
        in = (double*)tmp_space;
        in_sl = nn;
        // vec_rnx_copy(module, (double*)tmp_space, row_max, nn, a_dft, row_max, a_sl);
        for (uint64_t row_i = 0; row_i < row_max; row_i++) {
          memcpy((double*)tmp_space + row_i * nn, a_dft + row_i * a_sl, nn * sizeof(double));
        }
      } else {
        // it is out of place: do the product directly
        in = a_dft;
        in_sl = a_sl;
      }
      for (uint64_t col_i = 0; col_i < col_max; col_i++) {
        double* pmat_col = mat_input + col_i * nrows * nn;
        {
          reim_fftvec_mul(module->precomp.fft64.p_fftvec_mul,  //
                          res + col_i * res_sl,                //
                          in,                                  //
                          pmat_col);
        }
        for (uint64_t row_i = 1; row_i < row_max; row_i++) {
          reim_fftvec_addmul(module->precomp.fft64.p_fftvec_addmul,  //
                             res + col_i * res_sl,                   //
                             in + row_i * in_sl,                     //
                             pmat_col + row_i * nn);
        }
      }
    }
  }
  // zero out remaining bytes (if any)
  for (uint64_t i = col_max; i < res_size; ++i) {
    memset(res + i * res_sl, 0, nn * sizeof(double));
  }
}

/** @brief applies a vmp product res = a x pmat */
EXPORT void fft64_rnx_vmp_apply_tmp_a_avx(                     //
    const MOD_RNX* module,                                     // N
    double* res, uint64_t res_size, uint64_t res_sl,           // res (addr must be != a)
    double* tmpa, uint64_t a_size, uint64_t a_sl,              // a (will be overwritten)
    const RNX_VMP_PMAT* pmat, uint64_t nrows, uint64_t ncols,  // prep matrix
    uint8_t* tmp_space                                         // scratch space
) {
  const uint64_t nn = module->n;
  const uint64_t rows = nrows < a_size ? nrows : a_size;
  const uint64_t cols = ncols < res_size ? ncols : res_size;

  // fft is done in place on the input (tmpa is destroyed)
  for (uint64_t i = 0; i < rows; ++i) {
    reim_fft(module->precomp.fft64.p_fft, tmpa + i * a_sl);
  }
  fft64_rnx_vmp_apply_dft_to_dft_avx(module,              //
                                     res, cols, res_sl,   //
                                     tmpa, rows, a_sl,    //
                                     pmat, nrows, ncols,  //
                                     tmp_space);
  // ifft is done in place on the output
  for (uint64_t i = 0; i < cols; ++i) {
    reim_ifft(module->precomp.fft64.p_ifft, res + i * res_sl);
  }
  // zero out the remaining positions
  for (uint64_t i = cols; i < res_size; ++i) {
    memset(res + i * res_sl, 0, nn * sizeof(double));
  }
}
