#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../coeffs/coeffs_arithmetic.h"
#include "../q120/q120_arithmetic.h"
#include "../q120/q120_ntt.h"
#include "../reim/reim_fft_internal.h"
#include "../reim4/reim4_arithmetic.h"
#include "vec_znx_arithmetic.h"
#include "vec_znx_arithmetic_private.h"

// general function (virtual dispatch)

EXPORT void vec_znx_add(const MODULE* module,                              // N
                        int64_t* res, uint64_t res_size, uint64_t res_sl,  // res
                        const int64_t* a, uint64_t a_size, uint64_t a_sl,  // a
                        const int64_t* b, uint64_t b_size, uint64_t b_sl   // b
) {
  module->func.vec_znx_add(module,                 // N
                           res, res_size, res_sl,  // res
                           a, a_size, a_sl,        // a
                           b, b_size, b_sl         // b
  );
}

EXPORT void vec_znx_sub(const MODULE* module,                              // N
                        int64_t* res, uint64_t res_size, uint64_t res_sl,  // res
                        const int64_t* a, uint64_t a_size, uint64_t a_sl,  // a
                        const int64_t* b, uint64_t b_size, uint64_t b_sl   // b
) {
  module->func.vec_znx_sub(module,                 // N
                           res, res_size, res_sl,  // res
                           a, a_size, a_sl,        // a
                           b, b_size, b_sl         // b
  );
}

EXPORT void vec_znx_rotate(const MODULE* module,                              // N
                           const int64_t p,                                   // rotation value
                           int64_t* res, uint64_t res_size, uint64_t res_sl,  // res
                           const int64_t* a, uint64_t a_size, uint64_t a_sl   // a
) {
  module->func.vec_znx_rotate(module,                 // N
                              p,                      // p
                              res, res_size, res_sl,  // res
                              a, a_size, a_sl         // a
  );
}

EXPORT void vec_znx_automorphism(const MODULE* module,                              // N
                                 const int64_t p,                                   // X->X^p
                                 int64_t* res, uint64_t res_size, uint64_t res_sl,  // res
                                 const int64_t* a, uint64_t a_size, uint64_t a_sl   // a
) {
  module->func.vec_znx_automorphism(module,                 // N
                                    p,                      // p
                                    res, res_size, res_sl,  // res
                                    a, a_size, a_sl         // a
  );
}

EXPORT void vec_znx_normalize_base2k(const MODULE* module, uint64_t nn,                              // N
                                     uint64_t log2_base2k,                              // output base 2^K
                                     int64_t* res, uint64_t res_size, uint64_t res_sl,  // res
                                     const int64_t* a, uint64_t a_size, uint64_t a_sl,  // a
                                     uint8_t* tmp_space                                 // scratch space of size >= N
) {
  module->func.vec_znx_normalize_base2k(module, nn,                 // N
                                        log2_base2k,            // log2_base2k
                                        res, res_size, res_sl,  // res
                                        a, a_size, a_sl,        // a
                                        tmp_space);
}

EXPORT uint64_t vec_znx_normalize_base2k_tmp_bytes(const MODULE* module, uint64_t nn  // N
) {
  return module->func.vec_znx_normalize_base2k_tmp_bytes(module, nn  // N
  );
}

// specialized function (ref)

EXPORT void vec_znx_add_ref(const MODULE* module,                              // N
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
      znx_add_i64_ref(nn, res + i * res_sl, a + i * a_sl, b + i * b_sl);
    }
    // then copy to the largest dimension
    for (uint64_t i = sum_idx; i < copy_idx; ++i) {
      znx_copy_i64_ref(nn, res + i * res_sl, b + i * b_sl);
    }
    // then extend with zeros
    for (uint64_t i = copy_idx; i < res_size; ++i) {
      znx_zero_i64_ref(nn, res + i * res_sl);
    }
  } else {
    const uint64_t sum_idx = res_size < b_size ? res_size : b_size;
    const uint64_t copy_idx = res_size < a_size ? res_size : a_size;
    // add up to the smallest dimension
    for (uint64_t i = 0; i < sum_idx; ++i) {
      znx_add_i64_ref(nn, res + i * res_sl, a + i * a_sl, b + i * b_sl);
    }
    // then copy to the largest dimension
    for (uint64_t i = sum_idx; i < copy_idx; ++i) {
      znx_copy_i64_ref(nn, res + i * res_sl, a + i * a_sl);
    }
    // then extend with zeros
    for (uint64_t i = copy_idx; i < res_size; ++i) {
      znx_zero_i64_ref(nn, res + i * res_sl);
    }
  }
}

EXPORT void vec_znx_sub_ref(const MODULE* module,                              // N
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
      znx_sub_i64_ref(nn, res + i * res_sl, a + i * a_sl, b + i * b_sl);
    }
    // then negate to the largest dimension
    for (uint64_t i = sub_idx; i < copy_idx; ++i) {
      znx_negate_i64_ref(nn, res + i * res_sl, b + i * b_sl);
    }
    // then extend with zeros
    for (uint64_t i = copy_idx; i < res_size; ++i) {
      znx_zero_i64_ref(nn, res + i * res_sl);
    }
  } else {
    const uint64_t sub_idx = res_size < b_size ? res_size : b_size;
    const uint64_t copy_idx = res_size < a_size ? res_size : a_size;
    // subtract up to the smallest dimension
    for (uint64_t i = 0; i < sub_idx; ++i) {
      znx_sub_i64_ref(nn, res + i * res_sl, a + i * a_sl, b + i * b_sl);
    }
    // then copy to the largest dimension
    for (uint64_t i = sub_idx; i < copy_idx; ++i) {
      znx_copy_i64_ref(nn, res + i * res_sl, a + i * a_sl);
    }
    // then extend with zeros
    for (uint64_t i = copy_idx; i < res_size; ++i) {
      znx_zero_i64_ref(nn, res + i * res_sl);
    }
  }
}

EXPORT void vec_znx_rotate_ref(const MODULE* module,                              // N
                               const int64_t p,                                   // rotation value
                               int64_t* res, uint64_t res_size, uint64_t res_sl,  // res
                               const int64_t* a, uint64_t a_size, uint64_t a_sl   // a
) {
  const uint64_t nn = module->nn;

  const uint64_t rot_end_idx = res_size < a_size ? res_size : a_size;
  // rotate up to the smallest dimension
  for (uint64_t i = 0; i < rot_end_idx; ++i) {
    int64_t* res_ptr = res + i * res_sl;
    const int64_t* a_ptr = a + i * a_sl;
    if (res_ptr == a_ptr) {
      znx_rotate_inplace_i64(nn, p, res_ptr);
    } else {
      znx_rotate_i64(nn, p, res_ptr, a_ptr);
    }
  }
  // then extend with zeros
  for (uint64_t i = rot_end_idx; i < res_size; ++i) {
    znx_zero_i64_ref(nn, res + i * res_sl);
  }
}

EXPORT void vec_znx_automorphism_ref(const MODULE* module,                              // N
                                     const int64_t p,                                   // X->X^p
                                     int64_t* res, uint64_t res_size, uint64_t res_sl,  // res
                                     const int64_t* a, uint64_t a_size, uint64_t a_sl   // a
) {
  const uint64_t nn = module->nn;

  const uint64_t auto_end_idx = res_size < a_size ? res_size : a_size;

  for (uint64_t i = 0; i < auto_end_idx; ++i) {
    int64_t* res_ptr = res + i * res_sl;
    const int64_t* a_ptr = a + i * a_sl;
    if (res_ptr == a_ptr) {
      znx_automorphism_inplace_i64(nn, p, res_ptr);
    } else {
      znx_automorphism_i64(nn, p, res_ptr, a_ptr);
    }
  }
  // then extend with zeros
  for (uint64_t i = auto_end_idx; i < res_size; ++i) {
    znx_zero_i64_ref(nn, res + i * res_sl);
  }
}

EXPORT void vec_znx_normalize_base2k_ref(const MODULE* module, uint64_t nn,                              // N
                                         uint64_t log2_base2k,                              // output base 2^K
                                         int64_t* res, uint64_t res_size, uint64_t res_sl,  // res
                                         const int64_t* a, uint64_t a_size, uint64_t a_sl,  // a
                                         uint8_t* tmp_space  // scratch space of size >= N
) {

  // use MSB limb of res for carry propagation
  int64_t* cout = (int64_t*)tmp_space;
  int64_t* cin = 0x0;

  // propagate carry until first limb of res
  int64_t i = a_size - 1;
  for (; i >= res_size; --i) {
    znx_normalize(nn, log2_base2k, 0x0, cout, a + i * a_sl, cin);
    cin = cout;
  }

  // propagate carry and normalize
  for (; i >= 1; --i) {
    znx_normalize(nn, log2_base2k, res + i * res_sl, cout, a + i * a_sl, cin);
    cin = cout;
  }

  // normalize last limb
  znx_normalize(nn, log2_base2k, res, 0x0, a, cin);

  // extend result with zeros
  for (uint64_t i = a_size; i < res_size; ++i) {
    znx_zero_i64_ref(nn, res + i * res_sl);
  }
}

EXPORT uint64_t vec_znx_normalize_base2k_tmp_bytes_ref(const MODULE* module, uint64_t nn  // N
) {
  return nn * sizeof(int64_t);
}

// alias have to be defined in this unit: do not move
#ifdef __APPLE__
EXPORT uint64_t fft64_vec_znx_big_range_normalize_base2k_tmp_bytes(  //
    const MODULE* module,                                             // N
    uint64_t nn
) {
  return vec_znx_normalize_base2k_tmp_bytes_ref(module, nn);
}
EXPORT uint64_t fft64_vec_znx_big_normalize_base2k_tmp_bytes(  //
    const MODULE* module,                                       // N
    uint64_t nn
) {
  return vec_znx_normalize_base2k_tmp_bytes_ref(module, nn);
}
#else
EXPORT uint64_t fft64_vec_znx_big_normalize_base2k_tmp_bytes(  //
    const MODULE* module,                                       // N
    uint64_t nn
    ) __attribute((alias("vec_znx_normalize_base2k_tmp_bytes_ref")));

EXPORT uint64_t fft64_vec_znx_big_range_normalize_base2k_tmp_bytes(  //
    const MODULE* module,                                             // N
    uint64_t nn
    ) __attribute((alias("vec_znx_normalize_base2k_tmp_bytes_ref")));
#endif

/** @brief sets res = 0 */
EXPORT void vec_znx_zero(const MODULE* module,                             // N
                         int64_t* res, uint64_t res_size, uint64_t res_sl  // res
) {
  module->func.vec_znx_zero(module, res, res_size, res_sl);
}

/** @brief sets res = a */
EXPORT void vec_znx_copy(const MODULE* module,                              // N
                         int64_t* res, uint64_t res_size, uint64_t res_sl,  // res
                         const int64_t* a, uint64_t a_size, uint64_t a_sl   // a
) {
  module->func.vec_znx_copy(module, res, res_size, res_sl, a, a_size, a_sl);
}

/** @brief sets res = a */
EXPORT void vec_znx_negate(const MODULE* module,                              // N
                           int64_t* res, uint64_t res_size, uint64_t res_sl,  // res
                           const int64_t* a, uint64_t a_size, uint64_t a_sl   // a
) {
  module->func.vec_znx_negate(module, res, res_size, res_sl, a, a_size, a_sl);
}

EXPORT void vec_znx_zero_ref(const MODULE* module,                             // N
                             int64_t* res, uint64_t res_size, uint64_t res_sl  // res
) {
  uint64_t nn = module->nn;
  for (uint64_t i = 0; i < res_size; ++i) {
    znx_zero_i64_ref(nn, res + i * res_sl);
  }
}

EXPORT void vec_znx_copy_ref(const MODULE* module,                              // N
                             int64_t* res, uint64_t res_size, uint64_t res_sl,  // res
                             const int64_t* a, uint64_t a_size, uint64_t a_sl   // a
) {
  uint64_t nn = module->nn;
  uint64_t smin = res_size < a_size ? res_size : a_size;
  for (uint64_t i = 0; i < smin; ++i) {
    znx_copy_i64_ref(nn, res + i * res_sl, a + i * a_sl);
  }
  for (uint64_t i = smin; i < res_size; ++i) {
    znx_zero_i64_ref(nn, res + i * res_sl);
  }
}

EXPORT void vec_znx_negate_ref(const MODULE* module,                              // N
                               int64_t* res, uint64_t res_size, uint64_t res_sl,  // res
                               const int64_t* a, uint64_t a_size, uint64_t a_sl   // a
) {
  uint64_t nn = module->nn;
  uint64_t smin = res_size < a_size ? res_size : a_size;
  for (uint64_t i = 0; i < smin; ++i) {
    znx_negate_i64_ref(nn, res + i * res_sl, a + i * a_sl);
  }
  for (uint64_t i = smin; i < res_size; ++i) {
    znx_zero_i64_ref(nn, res + i * res_sl);
  }
}
