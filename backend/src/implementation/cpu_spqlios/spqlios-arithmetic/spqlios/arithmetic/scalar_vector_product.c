#include <string.h>

#include "vec_znx_arithmetic_private.h"

EXPORT uint64_t bytes_of_svp_ppol(const MODULE* module) { return module->func.bytes_of_svp_ppol(module); }

EXPORT uint64_t fft64_bytes_of_svp_ppol(const MODULE* module) { return module->nn * sizeof(double); }

EXPORT SVP_PPOL* new_svp_ppol(const MODULE* module) { return spqlios_alloc(bytes_of_svp_ppol(module)); }

EXPORT void delete_svp_ppol(SVP_PPOL* ppol) { spqlios_free(ppol); }

// public wrappers
EXPORT void svp_prepare(const MODULE* module,  // N
                        SVP_PPOL* ppol,        // output
                        const int64_t* pol     // a
) {
  module->func.svp_prepare(module, ppol, pol);
}

/** @brief prepares a svp polynomial  */
EXPORT void fft64_svp_prepare_ref(const MODULE* module,  // N
                                  SVP_PPOL* ppol,        // output
                                  const int64_t* pol     // a
) {
  reim_from_znx64(module->mod.fft64.p_conv, ppol, pol);
  reim_fft(module->mod.fft64.p_fft, (double*)ppol);
}

EXPORT void svp_apply_dft(const MODULE* module,                       // N
                          const VEC_ZNX_DFT* res, uint64_t res_size,  // output
                          const SVP_PPOL* ppol,                       // prepared pol
                          const int64_t* a, uint64_t a_size, uint64_t a_sl) {
  module->func.svp_apply_dft(module,  // N
                             res,
                             res_size,  // output
                             ppol,      // prepared pol
                             a, a_size, a_sl);
}

EXPORT void svp_apply_dft_to_dft(const MODULE* module,  // N
                                 const VEC_ZNX_DFT* res, uint64_t res_size,
                                 uint64_t res_cols,     // output
                                 const SVP_PPOL* ppol,  // prepared pol
                                 const VEC_ZNX_DFT* a, uint64_t a_size, uint64_t a_cols) {
  module->func.svp_apply_dft_to_dft(module,                   // N
                                    res, res_size, res_cols,  // output
                                    ppol, a, a_size, a_cols   // prepared pol
  );
}

// result = ppol * a
EXPORT void fft64_svp_apply_dft_ref(const MODULE* module,                             // N
                                    const VEC_ZNX_DFT* res, uint64_t res_size,        // output
                                    const SVP_PPOL* ppol,                             // prepared pol
                                    const int64_t* a, uint64_t a_size, uint64_t a_sl  // a
) {
  const uint64_t nn = module->nn;
  double* const dres = (double*)res;
  double* const dppol = (double*)ppol;

  const uint64_t auto_end_idx = res_size < a_size ? res_size : a_size;
  for (uint64_t i = 0; i < auto_end_idx; ++i) {
    const int64_t* a_ptr = a + i * a_sl;
    double* const res_ptr = dres + i * nn;
    // copy the polynomial to res, apply fft in place, call fftvec_mul in place.
    reim_from_znx64(module->mod.fft64.p_conv, res_ptr, a_ptr);
    reim_fft(module->mod.fft64.p_fft, res_ptr);
    reim_fftvec_mul(module->mod.fft64.mul_fft, res_ptr, res_ptr, dppol);
  }

  // then extend with zeros
  memset(dres + auto_end_idx * nn, 0, (res_size - auto_end_idx) * nn * sizeof(double));
}

// result = ppol * a
EXPORT void fft64_svp_apply_dft_to_dft_ref(const MODULE* module,  // N
                                           const VEC_ZNX_DFT* res, uint64_t res_size,
                                           uint64_t res_cols,     // output
                                           const SVP_PPOL* ppol,  // prepared pol
                                           const VEC_ZNX_DFT* a, uint64_t a_size,
                                           uint64_t a_cols  // a
) {
  const uint64_t nn = module->nn;
  const uint64_t res_sl = nn * res_cols;
  const uint64_t a_sl = nn * a_cols;
  double* const dres = (double*)res;
  double* const da = (double*)a;
  double* const dppol = (double*)ppol;

  const uint64_t auto_end_idx = res_size < a_size ? res_size : a_size;
  for (uint64_t i = 0; i < auto_end_idx; ++i) {
    const double* a_ptr = da + i * a_sl;
    double* const res_ptr = dres + i * res_sl;
    reim_fftvec_mul(module->mod.fft64.mul_fft, res_ptr, a_ptr, dppol);
  }

  // then extend with zeros
  for (uint64_t i = auto_end_idx; i < res_size; i++) {
    memset(dres + i * res_sl, 0, nn * sizeof(double));
  }
}
