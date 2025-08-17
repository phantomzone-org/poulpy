#include <string.h>

#include "../coeffs/coeffs_arithmetic.h"
#include "vec_rnx_arithmetic_private.h"

void rnx_add_ref(uint64_t nn, double* res, const double* a, const double* b) {
  for (uint64_t i = 0; i < nn; ++i) {
    res[i] = a[i] + b[i];
  }
}

void rnx_sub_ref(uint64_t nn, double* res, const double* a, const double* b) {
  for (uint64_t i = 0; i < nn; ++i) {
    res[i] = a[i] - b[i];
  }
}

void rnx_negate_ref(uint64_t nn, double* res, const double* a) {
  for (uint64_t i = 0; i < nn; ++i) {
    res[i] = -a[i];
  }
}

/** @brief sets res = a + b */
EXPORT void vec_rnx_add_ref(                          //
    const MOD_RNX* module,                            // N
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl,  // a
    const double* b, uint64_t b_size, uint64_t b_sl   // b
) {
  const uint64_t nn = module->n;
  if (a_size < b_size) {
    const uint64_t msize = res_size < a_size ? res_size : a_size;
    const uint64_t nsize = res_size < b_size ? res_size : b_size;
    for (uint64_t i = 0; i < msize; ++i) {
      rnx_add_ref(nn, res + i * res_sl, a + i * a_sl, b + i * b_sl);
    }
    for (uint64_t i = msize; i < nsize; ++i) {
      memcpy(res + i * res_sl, b + i * b_sl, nn * sizeof(double));
    }
    for (uint64_t i = nsize; i < res_size; ++i) {
      memset(res + i * res_sl, 0, nn * sizeof(double));
    }
  } else {
    const uint64_t msize = res_size < b_size ? res_size : b_size;
    const uint64_t nsize = res_size < a_size ? res_size : a_size;
    for (uint64_t i = 0; i < msize; ++i) {
      rnx_add_ref(nn, res + i * res_sl, a + i * a_sl, b + i * b_sl);
    }
    for (uint64_t i = msize; i < nsize; ++i) {
      memcpy(res + i * res_sl, a + i * a_sl, nn * sizeof(double));
    }
    for (uint64_t i = nsize; i < res_size; ++i) {
      memset(res + i * res_sl, 0, nn * sizeof(double));
    }
  }
}

/** @brief sets res = 0 */
EXPORT void vec_rnx_zero_ref(                        //
    const MOD_RNX* module,                           // N
    double* res, uint64_t res_size, uint64_t res_sl  // res
) {
  const uint64_t nn = module->n;
  for (uint64_t i = 0; i < res_size; ++i) {
    memset(res + i * res_sl, 0, nn * sizeof(double));
  }
}

/** @brief sets res = a */
EXPORT void vec_rnx_copy_ref(                         //
    const MOD_RNX* module,                            // N
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl   // a
) {
  const uint64_t nn = module->n;

  const uint64_t rot_end_idx = res_size < a_size ? res_size : a_size;
  // rotate up to the smallest dimension
  for (uint64_t i = 0; i < rot_end_idx; ++i) {
    double* res_ptr = res + i * res_sl;
    const double* a_ptr = a + i * a_sl;
    memcpy(res_ptr, a_ptr, nn * sizeof(double));
  }
  // then extend with zeros
  for (uint64_t i = rot_end_idx; i < res_size; ++i) {
    memset(res + i * res_sl, 0, nn * sizeof(double));
  }
}

/** @brief sets res = -a */
EXPORT void vec_rnx_negate_ref(                       //
    const MOD_RNX* module,                            // N
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl   // a
) {
  const uint64_t nn = module->n;

  const uint64_t rot_end_idx = res_size < a_size ? res_size : a_size;
  // rotate up to the smallest dimension
  for (uint64_t i = 0; i < rot_end_idx; ++i) {
    double* res_ptr = res + i * res_sl;
    const double* a_ptr = a + i * a_sl;
    rnx_negate_ref(nn, res_ptr, a_ptr);
  }
  // then extend with zeros
  for (uint64_t i = rot_end_idx; i < res_size; ++i) {
    memset(res + i * res_sl, 0, nn * sizeof(double));
  }
}

/** @brief sets res = a - b */
EXPORT void vec_rnx_sub_ref(                          //
    const MOD_RNX* module,                            // N
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl,  // a
    const double* b, uint64_t b_size, uint64_t b_sl   // b
) {
  const uint64_t nn = module->n;
  if (a_size < b_size) {
    const uint64_t msize = res_size < a_size ? res_size : a_size;
    const uint64_t nsize = res_size < b_size ? res_size : b_size;
    for (uint64_t i = 0; i < msize; ++i) {
      rnx_sub_ref(nn, res + i * res_sl, a + i * a_sl, b + i * b_sl);
    }
    for (uint64_t i = msize; i < nsize; ++i) {
      rnx_negate_ref(nn, res + i * res_sl, b + i * b_sl);
    }
    for (uint64_t i = nsize; i < res_size; ++i) {
      memset(res + i * res_sl, 0, nn * sizeof(double));
    }
  } else {
    const uint64_t msize = res_size < b_size ? res_size : b_size;
    const uint64_t nsize = res_size < a_size ? res_size : a_size;
    for (uint64_t i = 0; i < msize; ++i) {
      rnx_sub_ref(nn, res + i * res_sl, a + i * a_sl, b + i * b_sl);
    }
    for (uint64_t i = msize; i < nsize; ++i) {
      memcpy(res + i * res_sl, a + i * a_sl, nn * sizeof(double));
    }
    for (uint64_t i = nsize; i < res_size; ++i) {
      memset(res + i * res_sl, 0, nn * sizeof(double));
    }
  }
}

/** @brief sets res = a . X^p */
EXPORT void vec_rnx_rotate_ref(                       //
    const MOD_RNX* module,                            // N
    const int64_t p,                                  // rotation value
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl   // a
) {
  const uint64_t nn = module->n;

  const uint64_t rot_end_idx = res_size < a_size ? res_size : a_size;
  // rotate up to the smallest dimension
  for (uint64_t i = 0; i < rot_end_idx; ++i) {
    double* res_ptr = res + i * res_sl;
    const double* a_ptr = a + i * a_sl;
    if (res_ptr == a_ptr) {
      rnx_rotate_inplace_f64(nn, p, res_ptr);
    } else {
      rnx_rotate_f64(nn, p, res_ptr, a_ptr);
    }
  }
  // then extend with zeros
  for (uint64_t i = rot_end_idx; i < res_size; ++i) {
    memset(res + i * res_sl, 0, nn * sizeof(double));
  }
}

/** @brief sets res = a(X^p) */
EXPORT void vec_rnx_automorphism_ref(                 //
    const MOD_RNX* module,                            // N
    int64_t p,                                        // X -> X^p
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl   // a
) {
  const uint64_t nn = module->n;

  const uint64_t rot_end_idx = res_size < a_size ? res_size : a_size;
  // rotate up to the smallest dimension
  for (uint64_t i = 0; i < rot_end_idx; ++i) {
    double* res_ptr = res + i * res_sl;
    const double* a_ptr = a + i * a_sl;
    if (res_ptr == a_ptr) {
      rnx_automorphism_inplace_f64(nn, p, res_ptr);
    } else {
      rnx_automorphism_f64(nn, p, res_ptr, a_ptr);
    }
  }
  // then extend with zeros
  for (uint64_t i = rot_end_idx; i < res_size; ++i) {
    memset(res + i * res_sl, 0, nn * sizeof(double));
  }
}

/** @brief sets res = a . (X^p - 1) */
EXPORT void vec_rnx_mul_xp_minus_one_ref(             //
    const MOD_RNX* module,                            // N
    const int64_t p,                                  // rotation value
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl   // a
) {
  const uint64_t nn = module->n;

  const uint64_t rot_end_idx = res_size < a_size ? res_size : a_size;
  // rotate up to the smallest dimension
  for (uint64_t i = 0; i < rot_end_idx; ++i) {
    double* res_ptr = res + i * res_sl;
    const double* a_ptr = a + i * a_sl;
    if (res_ptr == a_ptr) {
      rnx_mul_xp_minus_one_inplace_f64(nn, p, res_ptr);
    } else {
      rnx_mul_xp_minus_one_f64(nn, p, res_ptr, a_ptr);
    }
  }
  // then extend with zeros
  for (uint64_t i = rot_end_idx; i < res_size; ++i) {
    memset(res + i * res_sl, 0, nn * sizeof(double));
  }
}
