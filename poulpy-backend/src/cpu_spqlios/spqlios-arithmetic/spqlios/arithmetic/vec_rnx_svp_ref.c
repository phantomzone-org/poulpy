#include <string.h>

#include "../coeffs/coeffs_arithmetic.h"
#include "vec_rnx_arithmetic_private.h"

EXPORT uint64_t fft64_bytes_of_rnx_svp_ppol(const MOD_RNX* module) { return module->n * sizeof(double); }

EXPORT RNX_SVP_PPOL* new_rnx_svp_ppol(const MOD_RNX* module) { return spqlios_alloc(bytes_of_rnx_svp_ppol(module)); }

EXPORT void delete_rnx_svp_ppol(RNX_SVP_PPOL* ppol) { spqlios_free(ppol); }

/** @brief prepares a svp polynomial  */
EXPORT void fft64_rnx_svp_prepare_ref(const MOD_RNX* module,  // N
                                      RNX_SVP_PPOL* ppol,     // output
                                      const double* pol       // a
) {
  double* const dppol = (double*)ppol;
  rnx_divide_by_m_ref(module->n, module->m, dppol, pol);
  reim_fft(module->precomp.fft64.p_fft, dppol);
}

EXPORT void fft64_rnx_svp_apply_ref(                  //
    const MOD_RNX* module,                            // N
    double* res, uint64_t res_size, uint64_t res_sl,  // output
    const RNX_SVP_PPOL* ppol,                         // prepared pol
    const double* a, uint64_t a_size, uint64_t a_sl   // a
) {
  const uint64_t nn = module->n;
  double* const dppol = (double*)ppol;

  const uint64_t auto_end_idx = res_size < a_size ? res_size : a_size;
  for (uint64_t i = 0; i < auto_end_idx; ++i) {
    const double* a_ptr = a + i * a_sl;
    double* const res_ptr = res + i * res_sl;
    // copy the polynomial to res, apply fft in place, call fftvec
    // _mul, apply ifft in place.
    memcpy(res_ptr, a_ptr, nn * sizeof(double));
    reim_fft(module->precomp.fft64.p_fft, (double*)res_ptr);
    reim_fftvec_mul(module->precomp.fft64.p_fftvec_mul, res_ptr, res_ptr, dppol);
    reim_ifft(module->precomp.fft64.p_ifft, res_ptr);
  }

  // then extend with zeros
  for (uint64_t i = auto_end_idx; i < res_size; ++i) {
    memset(res + i * res_sl, 0, nn * sizeof(double));
  }
}
