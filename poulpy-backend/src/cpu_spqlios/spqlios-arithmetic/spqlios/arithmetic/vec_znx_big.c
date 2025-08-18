#include "vec_znx_arithmetic_private.h"

EXPORT uint64_t bytes_of_vec_znx_big(const MODULE* module,  // N
                                     uint64_t size) {
  return module->func.bytes_of_vec_znx_big(module, size);
}

// public wrappers

/** @brief sets res = a+b */
EXPORT void vec_znx_big_add(const MODULE* module,                   // N
                            VEC_ZNX_BIG* res, uint64_t res_size,    // res
                            const VEC_ZNX_BIG* a, uint64_t a_size,  // a
                            const VEC_ZNX_BIG* b, uint64_t b_size   // b
) {
  module->func.vec_znx_big_add(module, res, res_size, a, a_size, b, b_size);
}

/** @brief sets res = a+b */
EXPORT void vec_znx_big_add_small(const MODULE* module,                             // N
                                  VEC_ZNX_BIG* res, uint64_t res_size,              // res
                                  const VEC_ZNX_BIG* a, uint64_t a_size,            // a
                                  const int64_t* b, uint64_t b_size, uint64_t b_sl  // b
) {
  module->func.vec_znx_big_add_small(module, res, res_size, a, a_size, b, b_size, b_sl);
}

EXPORT void vec_znx_big_add_small2(const MODULE* module,                              // N
                                   VEC_ZNX_BIG* res, uint64_t res_size,               // res
                                   const int64_t* a, uint64_t a_size, uint64_t a_sl,  // a
                                   const int64_t* b, uint64_t b_size, uint64_t b_sl   // b
) {
  module->func.vec_znx_big_add_small2(module, res, res_size, a, a_size, a_sl, b, b_size, b_sl);
}

/** @brief sets res = a-b */
EXPORT void vec_znx_big_sub(const MODULE* module,                   // N
                            VEC_ZNX_BIG* res, uint64_t res_size,    // res
                            const VEC_ZNX_BIG* a, uint64_t a_size,  // a
                            const VEC_ZNX_BIG* b, uint64_t b_size   // b
) {
  module->func.vec_znx_big_sub(module, res, res_size, a, a_size, b, b_size);
}

EXPORT void vec_znx_big_sub_small_b(const MODULE* module,                             // N
                                    VEC_ZNX_BIG* res, uint64_t res_size,              // res
                                    const VEC_ZNX_BIG* a, uint64_t a_size,            // a
                                    const int64_t* b, uint64_t b_size, uint64_t b_sl  // b
) {
  module->func.vec_znx_big_sub_small_b(module, res, res_size, a, a_size, b, b_size, b_sl);
}

EXPORT void vec_znx_big_sub_small_a(const MODULE* module,                              // N
                                    VEC_ZNX_BIG* res, uint64_t res_size,               // res
                                    const int64_t* a, uint64_t a_size, uint64_t a_sl,  // a
                                    const VEC_ZNX_BIG* b, uint64_t b_size              // b
) {
  module->func.vec_znx_big_sub_small_a(module, res, res_size, a, a_size, a_sl, b, b_size);
}
EXPORT void vec_znx_big_sub_small2(const MODULE* module,                              // N
                                   VEC_ZNX_BIG* res, uint64_t res_size,               // res
                                   const int64_t* a, uint64_t a_size, uint64_t a_sl,  // a
                                   const int64_t* b, uint64_t b_size, uint64_t b_sl   // b
) {
  module->func.vec_znx_big_sub_small2(module, res, res_size, a, a_size, a_sl, b, b_size, b_sl);
}

/** @brief sets res = a . X^p */
EXPORT void vec_znx_big_rotate(const MODULE* module,                  // N
                               int64_t p,                             // rotation value
                               VEC_ZNX_BIG* res, uint64_t res_size,   // res
                               const VEC_ZNX_BIG* a, uint64_t a_size  // a
) {
  module->func.vec_znx_big_rotate(module, p, res, res_size, a, a_size);
}

/** @brief sets res = a(X^p) */
EXPORT void vec_znx_big_automorphism(const MODULE* module,                  // N
                                     int64_t p,                             // X-X^p
                                     VEC_ZNX_BIG* res, uint64_t res_size,   // res
                                     const VEC_ZNX_BIG* a, uint64_t a_size  // a
) {
  module->func.vec_znx_big_automorphism(module, p, res, res_size, a, a_size);
}

// private wrappers

EXPORT uint64_t fft64_bytes_of_vec_znx_big(const MODULE* module,  // N
                                           uint64_t size) {
  return module->nn * size * sizeof(double);
}

EXPORT VEC_ZNX_BIG* new_vec_znx_big(const MODULE* module,  // N
                                    uint64_t size) {
  return spqlios_alloc(bytes_of_vec_znx_big(module, size));
}

EXPORT void delete_vec_znx_big(VEC_ZNX_BIG* res) { spqlios_free(res); }

/** @brief sets res = a+b */
EXPORT void fft64_vec_znx_big_add(const MODULE* module,                   // N
                                  VEC_ZNX_BIG* res, uint64_t res_size,    // res
                                  const VEC_ZNX_BIG* a, uint64_t a_size,  // a
                                  const VEC_ZNX_BIG* b, uint64_t b_size   // b
) {
  const uint64_t n = module->nn;
  vec_znx_add(module,                      //
              (int64_t*)res, res_size, n,  //
              (int64_t*)a, a_size, n,      //
              (int64_t*)b, b_size, n);
}
/** @brief sets res = a+b */
EXPORT void fft64_vec_znx_big_add_small(const MODULE* module,                             // N
                                        VEC_ZNX_BIG* res, uint64_t res_size,              // res
                                        const VEC_ZNX_BIG* a, uint64_t a_size,            // a
                                        const int64_t* b, uint64_t b_size, uint64_t b_sl  // b
) {
  const uint64_t n = module->nn;
  vec_znx_add(module,                      //
              (int64_t*)res, res_size, n,  //
              (int64_t*)a, a_size, n,      //
              b, b_size, b_sl);
}
EXPORT void fft64_vec_znx_big_add_small2(const MODULE* module,                              // N
                                         VEC_ZNX_BIG* res, uint64_t res_size,               // res
                                         const int64_t* a, uint64_t a_size, uint64_t a_sl,  // a
                                         const int64_t* b, uint64_t b_size, uint64_t b_sl   // b
) {
  const uint64_t n = module->nn;
  vec_znx_add(module,                      //
              (int64_t*)res, res_size, n,  //
              a, a_size, a_sl,             //
              b, b_size, b_sl);
}

/** @brief sets res = a-b */
EXPORT void fft64_vec_znx_big_sub(const MODULE* module,                   // N
                                  VEC_ZNX_BIG* res, uint64_t res_size,    // res
                                  const VEC_ZNX_BIG* a, uint64_t a_size,  // a
                                  const VEC_ZNX_BIG* b, uint64_t b_size   // b
) {
  const uint64_t n = module->nn;
  vec_znx_sub(module,                      //
              (int64_t*)res, res_size, n,  //
              (int64_t*)a, a_size, n,      //
              (int64_t*)b, b_size, n);
}

EXPORT void fft64_vec_znx_big_sub_small_b(const MODULE* module,                             // N
                                          VEC_ZNX_BIG* res, uint64_t res_size,              // res
                                          const VEC_ZNX_BIG* a, uint64_t a_size,            // a
                                          const int64_t* b, uint64_t b_size, uint64_t b_sl  // b
) {
  const uint64_t n = module->nn;
  vec_znx_sub(module,                      //
              (int64_t*)res, res_size, n,  //
              (int64_t*)a, a_size,         //
              n, b, b_size, b_sl);
}
EXPORT void fft64_vec_znx_big_sub_small_a(const MODULE* module,                              // N
                                          VEC_ZNX_BIG* res, uint64_t res_size,               // res
                                          const int64_t* a, uint64_t a_size, uint64_t a_sl,  // a
                                          const VEC_ZNX_BIG* b, uint64_t b_size              // b
) {
  const uint64_t n = module->nn;
  vec_znx_sub(module,                      //
              (int64_t*)res, res_size, n,  //
              a, a_size, a_sl,             //
              (int64_t*)b, b_size, n);
}
EXPORT void fft64_vec_znx_big_sub_small2(const MODULE* module,                              // N
                                         VEC_ZNX_BIG* res, uint64_t res_size,               // res
                                         const int64_t* a, uint64_t a_size, uint64_t a_sl,  // a
                                         const int64_t* b, uint64_t b_size, uint64_t b_sl   // b
) {
  const uint64_t n = module->nn;
  vec_znx_sub(module,                   //
              (int64_t*)res, res_size,  //
              n, a, a_size,             //
              a_sl, b, b_size, b_sl);
}

/** @brief sets res = a . X^p */
EXPORT void fft64_vec_znx_big_rotate(const MODULE* module,                  // N
                                     int64_t p,                             // rotation value
                                     VEC_ZNX_BIG* res, uint64_t res_size,   // res
                                     const VEC_ZNX_BIG* a, uint64_t a_size  // a
) {
  uint64_t nn = module->nn;
  vec_znx_rotate(module, p, (int64_t*)res, res_size, nn, (int64_t*)a, a_size, nn);
}

/** @brief sets res = a(X^p) */
EXPORT void fft64_vec_znx_big_automorphism(const MODULE* module,                  // N
                                           int64_t p,                             // X-X^p
                                           VEC_ZNX_BIG* res, uint64_t res_size,   // res
                                           const VEC_ZNX_BIG* a, uint64_t a_size  // a
) {
  uint64_t nn = module->nn;
  vec_znx_automorphism(module, p, (int64_t*)res, res_size, nn, (int64_t*)a, a_size, nn);
}

EXPORT void vec_znx_big_normalize_base2k(const MODULE* module,                              // MODULE
                                         uint64_t nn,                                       // N
                                         uint64_t k,                                        // base-2^k
                                         int64_t* res, uint64_t res_size, uint64_t res_sl,  // res
                                         const VEC_ZNX_BIG* a, uint64_t a_size,             // a
                                         uint8_t* tmp_space                                 // temp space
) {
  module->func.vec_znx_big_normalize_base2k(module,                 // MODULE
                                            nn,                     // N
                                            k,                      // base-2^k
                                            res, res_size, res_sl,  // res
                                            a, a_size,              // a
                                            tmp_space);
}

EXPORT uint64_t vec_znx_big_normalize_base2k_tmp_bytes(const MODULE* module, uint64_t nn  // N
) {
  return module->func.vec_znx_big_normalize_base2k_tmp_bytes(module, nn  // N
  );
}

/** @brief sets res = k-normalize(a.subrange) -- output in int64 coeffs space */
EXPORT void vec_znx_big_range_normalize_base2k(                                                  //
    const MODULE* module,                                                                        // MODULE
    uint64_t nn,                                                                                 // N
    uint64_t log2_base2k,                                                                        // base-2^k
    int64_t* res, uint64_t res_size, uint64_t res_sl,                                            // res
    const VEC_ZNX_BIG* a, uint64_t a_range_begin, uint64_t a_range_xend, uint64_t a_range_step,  // range
    uint8_t* tmp_space                                                                           // temp space
) {
  module->func.vec_znx_big_range_normalize_base2k(module, nn, log2_base2k, res, res_size, res_sl, a, a_range_begin,
                                                  a_range_xend, a_range_step, tmp_space);
}

/** @brief returns the minimal byte length of scratch space for vec_znx_big_range_normalize_base2k */
EXPORT uint64_t vec_znx_big_range_normalize_base2k_tmp_bytes(  //
    const MODULE* module,                                       // MODULE
    uint64_t nn                                               // N
) {
  return module->func.vec_znx_big_range_normalize_base2k_tmp_bytes(module, nn);
}

EXPORT void fft64_vec_znx_big_normalize_base2k(const MODULE* module,                              // MODULE
                                               uint64_t nn,                                       // N
                                               uint64_t k,                                        // base-2^k
                                               int64_t* res, uint64_t res_size, uint64_t res_sl,  // res
                                               const VEC_ZNX_BIG* a, uint64_t a_size,             // a
                                               uint8_t* tmp_space) {
  uint64_t a_sl = nn;
  module->func.vec_znx_normalize_base2k(module,                     // N
                                        nn,
                                        k,                          // log2_base2k
                                        res, res_size, res_sl,      // res
                                        (int64_t*)a, a_size, a_sl,  // a
                                        tmp_space);
}

EXPORT void fft64_vec_znx_big_range_normalize_base2k(                         //
    const MODULE* module,                                                     // MODULE
    uint64_t nn,                                                              // N
    uint64_t k,                                                               // base-2^k
    int64_t* res, uint64_t res_size, uint64_t res_sl,                         // res
    const VEC_ZNX_BIG* a, uint64_t a_begin, uint64_t a_end, uint64_t a_step,  // a
    uint8_t* tmp_space) {
  // convert the range indexes to int64[] slices
  const int64_t* a_st = ((int64_t*)a) + nn * a_begin;
  const uint64_t a_size = (a_end + a_step - 1 - a_begin) / a_step;
  const uint64_t a_sl = nn * a_step;
  // forward the call
  module->func.vec_znx_normalize_base2k(module,                 // MODULE
                                        nn,                     // N
                                        k,                      // log2_base2k
                                        res, res_size, res_sl,  // res
                                        a_st, a_size, a_sl,     // a
                                        tmp_space);
}
