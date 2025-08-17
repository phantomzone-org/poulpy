#include "vec_znx_arithmetic_private.h"

/** @brief res = a * b : small integer polynomial product  */
EXPORT void fft64_znx_small_single_product(const MODULE* module,  // N
                                           int64_t* res,          // output
                                           const int64_t* a,      // a
                                           const int64_t* b,      // b
                                           uint8_t* tmp) {
  const uint64_t nn = module->nn;
  double* const ffta = (double*)tmp;
  double* const fftb = ((double*)tmp) + nn;
  reim_from_znx64(module->mod.fft64.p_conv, ffta, a);
  reim_from_znx64(module->mod.fft64.p_conv, fftb, b);
  reim_fft(module->mod.fft64.p_fft, ffta);
  reim_fft(module->mod.fft64.p_fft, fftb);
  reim_fftvec_mul_simple(module->m, ffta, ffta, fftb);
  reim_ifft(module->mod.fft64.p_ifft, ffta);
  reim_to_znx64(module->mod.fft64.p_reim_to_znx, res, ffta);
}

/** @brief tmp bytes required for znx_small_single_product  */
EXPORT uint64_t fft64_znx_small_single_product_tmp_bytes(const MODULE* module, uint64_t nn) {
  return 2 * nn * sizeof(double);
}

/** @brief res = a * b : small integer polynomial product  */
EXPORT void znx_small_single_product(const MODULE* module,  // N
                                     int64_t* res,          // output
                                     const int64_t* a,      // a
                                     const int64_t* b,      // b
                                     uint8_t* tmp) {
  module->func.znx_small_single_product(module, res, a, b, tmp);
}

/** @brief tmp bytes required for znx_small_single_product  */
EXPORT uint64_t znx_small_single_product_tmp_bytes(const MODULE* module, uint64_t nn) {
  return module->func.znx_small_single_product_tmp_bytes(module, nn);
}
