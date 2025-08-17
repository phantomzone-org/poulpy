#include <memory.h>

#include "vec_rnx_arithmetic_private.h"
#include "zn_arithmetic_private.h"

EXPORT void vec_rnx_to_znx32_ref(                      //
    const MOD_RNX* module,                             // N
    int32_t* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl    // a
) {
  const uint64_t nn = module->n;
  const uint64_t msize = res_size < a_size ? res_size : a_size;
  for (uint64_t i = 0; i < msize; ++i) {
    dbl_round_to_i32_ref(NULL, res + i * res_sl, nn, a + i * a_sl, nn);
  }
  for (uint64_t i = msize; i < res_size; ++i) {
    memset(res + i * res_sl, 0, nn * sizeof(int32_t));
  }
}

EXPORT void vec_rnx_from_znx32_ref(                   //
    const MOD_RNX* module,                            // N
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const int32_t* a, uint64_t a_size, uint64_t a_sl  // a
) {
  const uint64_t nn = module->n;
  const uint64_t msize = res_size < a_size ? res_size : a_size;
  for (uint64_t i = 0; i < msize; ++i) {
    i32_to_dbl_ref(NULL, res + i * res_sl, nn, a + i * a_sl, nn);
  }
  for (uint64_t i = msize; i < res_size; ++i) {
    memset(res + i * res_sl, 0, nn * sizeof(int32_t));
  }
}
EXPORT void vec_rnx_to_tnx32_ref(                      //
    const MOD_RNX* module,                             // N
    int32_t* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl    // a
) {
  const uint64_t nn = module->n;
  const uint64_t msize = res_size < a_size ? res_size : a_size;
  for (uint64_t i = 0; i < msize; ++i) {
    dbl_to_tn32_ref(NULL, res + i * res_sl, nn, a + i * a_sl, nn);
  }
  for (uint64_t i = msize; i < res_size; ++i) {
    memset(res + i * res_sl, 0, nn * sizeof(int32_t));
  }
}
EXPORT void vec_rnx_from_tnx32_ref(                   //
    const MOD_RNX* module,                            // N
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const int32_t* a, uint64_t a_size, uint64_t a_sl  // a
) {
  const uint64_t nn = module->n;
  const uint64_t msize = res_size < a_size ? res_size : a_size;
  for (uint64_t i = 0; i < msize; ++i) {
    tn32_to_dbl_ref(NULL, res + i * res_sl, nn, a + i * a_sl, nn);
  }
  for (uint64_t i = msize; i < res_size; ++i) {
    memset(res + i * res_sl, 0, nn * sizeof(int32_t));
  }
}

static void dbl_to_tndbl_ref(         //
    const void* UNUSED,               // N
    double* res, uint64_t res_size,   // res
    const double* a, uint64_t a_size  // a
) {
  static const double OFF_CST = INT64_C(3) << 51;
  const uint64_t msize = res_size < a_size ? res_size : a_size;
  for (uint64_t i = 0; i < msize; ++i) {
    double ai = a[i] + OFF_CST;
    res[i] = a[i] - (ai - OFF_CST);
  }
  memset(res + msize, 0, (res_size - msize) * sizeof(double));
}

EXPORT void vec_rnx_to_tnxdbl_ref(                    //
    const MOD_RNX* module,                            // N
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl   // a
) {
  const uint64_t nn = module->n;
  const uint64_t msize = res_size < a_size ? res_size : a_size;
  for (uint64_t i = 0; i < msize; ++i) {
    dbl_to_tndbl_ref(NULL, res + i * res_sl, nn, a + i * a_sl, nn);
  }
  for (uint64_t i = msize; i < res_size; ++i) {
    memset(res + i * res_sl, 0, nn * sizeof(int32_t));
  }
}
