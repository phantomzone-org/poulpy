#include <string.h>

#include "../coeffs/coeffs_arithmetic.h"
#include "../reim4/reim4_arithmetic.h"
#include "vec_znx_arithmetic_private.h"

// specialized function (ref)

// Note: these functions do not have an avx variant.
#define znx_copy_i64_avx znx_copy_i64_ref
#define znx_zero_i64_avx znx_zero_i64_ref

EXPORT void vec_znx_add_avx(const MODULE* module,                              // N
                            int64_t* res, uint64_t res_size, uint64_t res_sl,  // res
                            const int64_t* a, uint64_t a_size, uint64_t a_sl,  // a
                            const int64_t* b, uint64_t b_size, uint64_t b_sl   // b
) {
  const uint64_t nn = module->nn;
  if (a_size <= b_size) {
    const uint64_t sum_idx = res_size < a_size ? res_size : a_size;
    const uint64_t copy_idx = res_size < b_size ? res_size : b_size;
    // add up to the smallest dimension
    for (uint64_t i = 0; i < sum_idx; ++i) {
      znx_add_i64_avx(nn, res + i * res_sl, a + i * a_sl, b + i * b_sl);
    }
    // then copy to the largest dimension
    for (uint64_t i = sum_idx; i < copy_idx; ++i) {
      znx_copy_i64_avx(nn, res + i * res_sl, b + i * b_sl);
    }
    // then extend with zeros
    for (uint64_t i = copy_idx; i < res_size; ++i) {
      znx_zero_i64_avx(nn, res + i * res_sl);
    }
  } else {
    const uint64_t sum_idx = res_size < b_size ? res_size : b_size;
    const uint64_t copy_idx = res_size < a_size ? res_size : a_size;
    // add up to the smallest dimension
    for (uint64_t i = 0; i < sum_idx; ++i) {
      znx_add_i64_avx(nn, res + i * res_sl, a + i * a_sl, b + i * b_sl);
    }
    // then copy to the largest dimension
    for (uint64_t i = sum_idx; i < copy_idx; ++i) {
      znx_copy_i64_avx(nn, res + i * res_sl, a + i * a_sl);
    }
    // then extend with zeros
    for (uint64_t i = copy_idx; i < res_size; ++i) {
      znx_zero_i64_avx(nn, res + i * res_sl);
    }
  }
}

EXPORT void vec_znx_sub_avx(const MODULE* module,                              // N
                            int64_t* res, uint64_t res_size, uint64_t res_sl,  // res
                            const int64_t* a, uint64_t a_size, uint64_t a_sl,  // a
                            const int64_t* b, uint64_t b_size, uint64_t b_sl   // b
) {
  const uint64_t nn = module->nn;
  if (a_size <= b_size) {
    const uint64_t sub_idx = res_size < a_size ? res_size : a_size;
    const uint64_t copy_idx = res_size < b_size ? res_size : b_size;
    // subtract up to the smallest dimension
    for (uint64_t i = 0; i < sub_idx; ++i) {
      znx_sub_i64_avx(nn, res + i * res_sl, a + i * a_sl, b + i * b_sl);
    }
    // then negate to the largest dimension
    for (uint64_t i = sub_idx; i < copy_idx; ++i) {
      znx_negate_i64_avx(nn, res + i * res_sl, b + i * b_sl);
    }
    // then extend with zeros
    for (uint64_t i = copy_idx; i < res_size; ++i) {
      znx_zero_i64_avx(nn, res + i * res_sl);
    }
  } else {
    const uint64_t sub_idx = res_size < b_size ? res_size : b_size;
    const uint64_t copy_idx = res_size < a_size ? res_size : a_size;
    // subtract up to the smallest dimension
    for (uint64_t i = 0; i < sub_idx; ++i) {
      znx_sub_i64_avx(nn, res + i * res_sl, a + i * a_sl, b + i * b_sl);
    }
    // then copy to the largest dimension
    for (uint64_t i = sub_idx; i < copy_idx; ++i) {
      znx_copy_i64_avx(nn, res + i * res_sl, a + i * a_sl);
    }
    // then extend with zeros
    for (uint64_t i = copy_idx; i < res_size; ++i) {
      znx_zero_i64_avx(nn, res + i * res_sl);
    }
  }
}

EXPORT void vec_znx_negate_avx(const MODULE* module,                              // N
                               int64_t* res, uint64_t res_size, uint64_t res_sl,  // res
                               const int64_t* a, uint64_t a_size, uint64_t a_sl   // a
) {
  uint64_t nn = module->nn;
  uint64_t smin = res_size < a_size ? res_size : a_size;
  for (uint64_t i = 0; i < smin; ++i) {
    znx_negate_i64_avx(nn, res + i * res_sl, a + i * a_sl);
  }
  for (uint64_t i = smin; i < res_size; ++i) {
    znx_zero_i64_ref(nn, res + i * res_sl);
  }
}
