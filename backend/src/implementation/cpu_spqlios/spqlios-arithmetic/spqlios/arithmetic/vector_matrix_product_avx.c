#include <string.h>

#include "../reim4/reim4_arithmetic.h"
#include "vec_znx_arithmetic_private.h"

/** @brief prepares a vmp matrix (contiguous row-major version) */
EXPORT void fft64_vmp_prepare_contiguous_avx(const MODULE* module,                                // N
                                             VMP_PMAT* pmat,                                      // output
                                             const int64_t* mat, uint64_t nrows, uint64_t ncols,  // a
                                             uint8_t* tmp_space                                   // scratch space
) {
  // there is an edge case if nn < 8
  const uint64_t nn = module->nn;
  const uint64_t m = module->m;

  double* output_mat = (double*)pmat;
  double* start_addr = (double*)pmat;
  uint64_t offset = nrows * ncols * 8;

  if (nn >= 8) {
    for (uint64_t row_i = 0; row_i < nrows; row_i++) {
      for (uint64_t col_i = 0; col_i < ncols; col_i++) {
        reim_from_znx64(module->mod.fft64.p_conv, (SVP_PPOL*)tmp_space, mat + (row_i * ncols + col_i) * nn);
        reim_fft(module->mod.fft64.p_fft, (double*)tmp_space);

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
          reim4_extract_1blk_from_reim_avx(m, blk_i, start_addr + blk_i * offset, (double*)tmp_space);
        }
      }
    }
  } else {
    for (uint64_t row_i = 0; row_i < nrows; row_i++) {
      for (uint64_t col_i = 0; col_i < ncols; col_i++) {
        double* res = (double*)pmat + (col_i * nrows + row_i) * nn;
        reim_from_znx64(module->mod.fft64.p_conv, (SVP_PPOL*)res, mat + (row_i * ncols + col_i) * nn);
        reim_fft(module->mod.fft64.p_fft, res);
      }
    }
  }
}

double* get_blk_addr(int row, int col, int nrows, int ncols, VMP_PMAT* pmat);

void fft64_store_svp_ppol_into_vmp_pmat_row_blk_avx(uint64_t nn, uint64_t m, const SVP_PPOL* svp_ppol, uint64_t row_i,
                                                    uint64_t col_i, uint64_t nrows, uint64_t ncols, VMP_PMAT* pmat) {
  double* start_addr = get_blk_addr(row_i, col_i, nrows, ncols, pmat);
  uint64_t offset = nrows * ncols * 8;
  for (uint64_t blk_i = 0; blk_i < m / 4; blk_i++) {
    reim4_extract_1blk_from_reim_avx(m, blk_i, start_addr + blk_i * offset, (double*)svp_ppol);
  }
}

/** @brief applies a vmp product (result in DFT space) abd adds to res inplace */
EXPORT void fft64_vmp_apply_dft_add_avx(const MODULE* module,                              // N
                                        VEC_ZNX_DFT* res, uint64_t res_size,               // res
                                        const int64_t* a, uint64_t a_size, uint64_t a_sl,  // a
                                        const VMP_PMAT* pmat, uint64_t nrows, uint64_t ncols,
                                        uint64_t pmat_scale,  // prep matrix
                                        uint8_t* tmp_space    // scratch space
) {
  const uint64_t nn = module->nn;
  const uint64_t rows = nrows < a_size ? nrows : a_size;

  VEC_ZNX_DFT* a_dft = (VEC_ZNX_DFT*)tmp_space;
  uint8_t* new_tmp_space = (uint8_t*)tmp_space + rows * nn * sizeof(double);

  fft64_vec_znx_dft(module, a_dft, rows, a, a_size, a_sl);
  fft64_vmp_apply_dft_to_dft_add_avx(module, res, res_size, a_dft, a_size, pmat, nrows, ncols, pmat_scale,
                                     new_tmp_space);
}

/** @brief applies a vmp product (result in DFT space) */
EXPORT void fft64_vmp_apply_dft_avx(const MODULE* module,                                  // N
                                    VEC_ZNX_DFT* res, uint64_t res_size,                   // res
                                    const int64_t* a, uint64_t a_size, uint64_t a_sl,      // a
                                    const VMP_PMAT* pmat, uint64_t nrows, uint64_t ncols,  // prep matrix
                                    uint8_t* tmp_space                                     // scratch space
) {
  const uint64_t nn = module->nn;
  const uint64_t rows = nrows < a_size ? nrows : a_size;

  VEC_ZNX_DFT* a_dft = (VEC_ZNX_DFT*)tmp_space;
  uint8_t* new_tmp_space = (uint8_t*)tmp_space + rows * nn * sizeof(double);

  fft64_vec_znx_dft(module, a_dft, rows, a, a_size, a_sl);
  fft64_vmp_apply_dft_to_dft_avx(module, res, res_size, a_dft, a_size, pmat, nrows, ncols, new_tmp_space);
}

EXPORT void fft64_vmp_apply_dft_to_dft_add_avx(const MODULE* module,                       // N
                                               VEC_ZNX_DFT* res, const uint64_t res_size,  // res
                                               const VEC_ZNX_DFT* a_dft, uint64_t a_size,  // a
                                               const VMP_PMAT* pmat, const uint64_t nrows, const uint64_t ncols,
                                               uint64_t pmat_scale,  // prep matrix
                                               uint8_t* tmp_space    // scratch space (a_size*sizeof(reim4) bytes)
) {
  const uint64_t m = module->m;
  const uint64_t nn = module->nn;

  double* mat2cols_output = (double*)tmp_space;     // 128 bytes
  double* extracted_blk = (double*)tmp_space + 16;  // 64*min(nrows,a_size) bytes

  double* mat_input = (double*)pmat;
  double* vec_input = (double*)a_dft;
  double* vec_output = (double*)res;

  const uint64_t row_max = nrows < a_size ? nrows : a_size;
  const uint64_t col_max = ncols < res_size ? ncols : res_size;

  if (nn >= 8) {
    for (uint64_t blk_i = 0; blk_i < m / 4; blk_i++) {
      double* mat_blk_start = mat_input + blk_i * (8 * nrows * ncols);

      reim4_extract_1blk_from_contiguous_reim_avx(m, row_max, blk_i, (double*)extracted_blk, (double*)a_dft);

      if (pmat_scale % 2 == 0) {
        for (uint64_t col_res = 0, col_pmat = pmat_scale; col_pmat < col_max - 1; col_res += 2, col_pmat += 2) {
          uint64_t col_offset = col_pmat * (8 * nrows);
          reim4_vec_mat2cols_product_avx2(row_max, mat2cols_output, extracted_blk, mat_blk_start + col_offset);
          reim4_add_1blk_to_reim_avx(m, blk_i, vec_output + col_res * nn, mat2cols_output);
          reim4_add_1blk_to_reim_avx(m, blk_i, vec_output + (col_res + 1) * nn, mat2cols_output + 8);
        }
      } else {
        uint64_t col_offset = (pmat_scale - 1) * (8 * nrows);
        reim4_vec_mat2cols_product_avx2(row_max, mat2cols_output, extracted_blk, mat_blk_start + col_offset);
        reim4_add_1blk_to_reim_avx(m, blk_i, vec_output, mat2cols_output + 8);

        for (uint64_t col_res = 1, col_pmat = pmat_scale + 1; col_pmat < col_max - 1; col_res += 2, col_pmat += 2) {
          uint64_t col_offset = col_pmat * (8 * nrows);
          reim4_vec_mat2cols_product_avx2(row_max, mat2cols_output, extracted_blk, mat_blk_start + col_offset);
          reim4_add_1blk_to_reim_avx(m, blk_i, vec_output + col_res * nn, mat2cols_output);
          reim4_add_1blk_to_reim_avx(m, blk_i, vec_output + (col_res + 1) * nn, mat2cols_output + 8);
        }
      }

      // check if col_max is odd, then special case
      if (col_max % 2 == 1) {
        uint64_t last_col = col_max - 1;
        uint64_t col_offset = last_col * (8 * nrows);

        if (last_col >= pmat_scale) {
          // the last column is alone in the pmat: vec_mat1col
          if (ncols == col_max)
            reim4_vec_mat1col_product_avx2(row_max, mat2cols_output, extracted_blk, mat_blk_start + col_offset);
          else {
            // the last column is part of a colpair in the pmat: vec_mat2cols and ignore the second position
            reim4_vec_mat2cols_product_avx2(row_max, mat2cols_output, extracted_blk, mat_blk_start + col_offset);
          }
          reim4_add_1blk_to_reim_avx(m, blk_i, vec_output + (last_col - pmat_scale) * nn, mat2cols_output);
        }
      }
    }
  } else {
    for (uint64_t col_res = 0, col_pmat = pmat_scale; col_pmat < col_max; col_res += 1, col_pmat += 1) {
      double* pmat_col = mat_input + col_pmat * nrows * nn;
      for (uint64_t row_i = 0; row_i < row_max; row_i++) {
        reim_fftvec_addmul(module->mod.fft64.p_addmul, vec_output + col_res * nn, vec_input + row_i * nn,
                           pmat_col + row_i * nn);
      }
    }
  }

  // zero out remaining bytes
  memset(vec_output + col_max * nn, 0, (res_size - col_max) * nn * sizeof(double));
}

/** @brief this inner function could be very handy */
EXPORT void fft64_vmp_apply_dft_to_dft_avx(const MODULE* module,                       // N
                                           VEC_ZNX_DFT* res, const uint64_t res_size,  // res
                                           const VEC_ZNX_DFT* a_dft, uint64_t a_size,  // a
                                           const VMP_PMAT* pmat, const uint64_t nrows,
                                           const uint64_t ncols,  // prep matrix
                                           uint8_t* tmp_space     // scratch space (a_size*sizeof(reim4) bytes)
) {
  const uint64_t m = module->m;
  const uint64_t nn = module->nn;

  double* mat2cols_output = (double*)tmp_space;     // 128 bytes
  double* extracted_blk = (double*)tmp_space + 16;  // 64*min(nrows,a_size) bytes

  double* mat_input = (double*)pmat;
  double* vec_input = (double*)a_dft;
  double* vec_output = (double*)res;

  const uint64_t row_max = nrows < a_size ? nrows : a_size;
  const uint64_t col_max = ncols < res_size ? ncols : res_size;

  if (nn >= 8) {
    for (uint64_t blk_i = 0; blk_i < m / 4; blk_i++) {
      double* mat_blk_start = mat_input + blk_i * (8 * nrows * ncols);

      reim4_extract_1blk_from_contiguous_reim_avx(m, row_max, blk_i, (double*)extracted_blk, (double*)a_dft);
      // apply mat2cols
      for (uint64_t col_i = 0; col_i < col_max - 1; col_i += 2) {
        uint64_t col_offset = col_i * (8 * nrows);
        reim4_vec_mat2cols_product_avx2(row_max, mat2cols_output, extracted_blk, mat_blk_start + col_offset);

        reim4_save_1blk_to_reim_avx(m, blk_i, vec_output + col_i * nn, mat2cols_output);
        reim4_save_1blk_to_reim_avx(m, blk_i, vec_output + (col_i + 1) * nn, mat2cols_output + 8);
      }

      // check if col_max is odd, then special case
      if (col_max % 2 == 1) {
        uint64_t last_col = col_max - 1;
        uint64_t col_offset = last_col * (8 * nrows);

        // the last column is alone in the pmat: vec_mat1col
        if (ncols == col_max)
          reim4_vec_mat1col_product_avx2(row_max, mat2cols_output, extracted_blk, mat_blk_start + col_offset);
        else {
          // the last column is part of a colpair in the pmat: vec_mat2cols and ignore the second position
          reim4_vec_mat2cols_product_avx2(row_max, mat2cols_output, extracted_blk, mat_blk_start + col_offset);
        }
        reim4_save_1blk_to_reim_avx(m, blk_i, vec_output + last_col * nn, mat2cols_output);
      }
    }
  } else {
    for (uint64_t col_i = 0; col_i < col_max; col_i++) {
      double* pmat_col = mat_input + col_i * nrows * nn;
      for (uint64_t row_i = 0; row_i < 1; row_i++) {
        reim_fftvec_mul(module->mod.fft64.mul_fft, vec_output + col_i * nn, vec_input + row_i * nn,
                        pmat_col + row_i * nn);
      }
      for (uint64_t row_i = 1; row_i < row_max; row_i++) {
        reim_fftvec_addmul(module->mod.fft64.p_addmul, vec_output + col_i * nn, vec_input + row_i * nn,
                           pmat_col + row_i * nn);
      }
    }
  }

  // zero out remaining bytes
  memset(vec_output + col_max * nn, 0, (res_size - col_max) * nn * sizeof(double));
}