#include <string.h>

#include "../q120/q120_arithmetic.h"
#include "vec_znx_arithmetic_private.h"

EXPORT uint64_t vec_znx_dft_automorphism_tmp_bytes(const MODULE* module) {
  return module->func.vec_znx_dft_automorphism_tmp_bytes(module);
}

EXPORT void vec_znx_dft_automorphism(const MODULE* module, int64_t p, VEC_ZNX_DFT* res_dft, uint64_t res_size,
                                     const VEC_ZNX_DFT* a_dft, uint64_t a_size, uint8_t* tmp_bytes) {
  return module->func.vec_znx_dft_automorphism(module, p, res_dft, res_size, a_dft, a_size, tmp_bytes);
}

EXPORT void vec_znx_dft(const MODULE* module,                             // N
                        VEC_ZNX_DFT* res, uint64_t res_size,              // res
                        const int64_t* a, uint64_t a_size, uint64_t a_sl  // a
) {
  return module->func.vec_znx_dft(module, res, res_size, a, a_size, a_sl);
}

EXPORT void vec_dft_add(const MODULE* module,                   // N
                        VEC_ZNX_DFT* res, uint64_t res_size,    // res
                        const VEC_ZNX_DFT* a, uint64_t a_size,  // a
                        const VEC_ZNX_DFT* b, uint64_t b_size   // b
) {
  return module->func.vec_dft_add(module, res, res_size, a, a_size, b, b_size);
}

EXPORT void vec_dft_sub(const MODULE* module,                   // N
                        VEC_ZNX_DFT* res, uint64_t res_size,    // res
                        const VEC_ZNX_DFT* a, uint64_t a_size,  // a
                        const VEC_ZNX_DFT* b, uint64_t b_size   // b
) {
  return module->func.vec_dft_sub(module, res, res_size, a, a_size, b, b_size);
}

EXPORT void vec_znx_idft(const MODULE* module,                       // N
                         VEC_ZNX_BIG* res, uint64_t res_size,        // res
                         const VEC_ZNX_DFT* a_dft, uint64_t a_size,  // a
                         uint8_t* tmp                                // scratch space
) {
  return module->func.vec_znx_idft(module, res, res_size, a_dft, a_size, tmp);
}

EXPORT uint64_t vec_znx_idft_tmp_bytes(const MODULE* module) { return module->func.vec_znx_idft_tmp_bytes(module); }

EXPORT void vec_znx_idft_tmp_a(const MODULE* module,                 // N
                               VEC_ZNX_BIG* res, uint64_t res_size,  // res
                               VEC_ZNX_DFT* a_dft, uint64_t a_size   // a is overwritten
) {
  return module->func.vec_znx_idft_tmp_a(module, res, res_size, a_dft, a_size);
}

EXPORT uint64_t bytes_of_vec_znx_dft(const MODULE* module,  // N
                                     uint64_t size) {
  return module->func.bytes_of_vec_znx_dft(module, size);
}

// fft64 backend
EXPORT uint64_t fft64_bytes_of_vec_znx_dft(const MODULE* module,  // N
                                           uint64_t size) {
  return module->nn * size * sizeof(double);
}

EXPORT VEC_ZNX_DFT* new_vec_znx_dft(const MODULE* module,  // N
                                    uint64_t size) {
  return spqlios_alloc(bytes_of_vec_znx_dft(module, size));
}

EXPORT void delete_vec_znx_dft(VEC_ZNX_DFT* res) { spqlios_free(res); }

EXPORT uint64_t fft64_vec_znx_dft_automorphism_tmp_bytes(const MODULE* module) {
  return reim_fftvec_automorphism_inplace_tmp_bytes(module->mod.fft64.p_automorphism);
}

EXPORT void fft64_vec_znx_dft(const MODULE* module,                             // N
                              VEC_ZNX_DFT* res, uint64_t res_size,              // res
                              const int64_t* a, uint64_t a_size, uint64_t a_sl  // a
) {
  const uint64_t smin = res_size < a_size ? res_size : a_size;
  const uint64_t nn = module->nn;

  for (uint64_t i = 0; i < smin; i++) {
    reim_from_znx64(module->mod.fft64.p_conv, ((double*)res) + i * nn, a + i * a_sl);
    reim_fft(module->mod.fft64.p_fft, ((double*)res) + i * nn);
  }

  // fill up remaining part with 0's
  double* const dres = (double*)res;
  memset(dres + smin * nn, 0, (res_size - smin) * nn * sizeof(double));
}

EXPORT void fft64_vec_dft_add(const MODULE* module,                   // N
                              VEC_ZNX_DFT* res, uint64_t res_size,    // res
                              const VEC_ZNX_DFT* a, uint64_t a_size,  // a
                              const VEC_ZNX_DFT* b, uint64_t b_size   // b
) {
  const uint64_t smin0 = a_size < b_size ? a_size : b_size;
  const uint64_t smin = res_size < smin0 ? res_size : smin0;
  const uint64_t nn = module->nn;

  for (uint64_t i = 0; i < smin; i++) {
    reim_fftvec_add(module->mod.fft64.add_fft, ((double*)res) + i * nn, ((double*)a) + i * nn, ((double*)b) + i * nn);
  }

  // fill remain `res` part with 0's
  double* const dres = (double*)res;
  memset(dres + smin * nn, 0, (res_size - smin) * nn * sizeof(double));
}

EXPORT void fft64_vec_dft_sub(const MODULE* module,                   // N
                              VEC_ZNX_DFT* res, uint64_t res_size,    // res
                              const VEC_ZNX_DFT* a, uint64_t a_size,  // a
                              const VEC_ZNX_DFT* b, uint64_t b_size   // b
) {
  const uint64_t smin0 = a_size < b_size ? a_size : b_size;
  const uint64_t smin = res_size < smin0 ? res_size : smin0;
  const uint64_t nn = module->nn;

  for (uint64_t i = 0; i < smin; i++) {
    reim_fftvec_sub(module->mod.fft64.sub_fft, ((double*)res) + i * nn, ((double*)a) + i * nn, ((double*)b) + i * nn);
  }

  // fill remain `res` part with 0's
  double* const dres = (double*)res;
  memset(dres + smin * nn, 0, (res_size - smin) * nn * sizeof(double));
}

EXPORT void fft64_vec_znx_dft_automorphism_ref(const MODULE* module, int64_t p, VEC_ZNX_DFT* res_dft, uint64_t res_size,
                                               const VEC_ZNX_DFT* a_dft, uint64_t a_size, uint8_t* tmp_bytes) {
  const uint64_t smin = res_size < a_size ? res_size : a_size;
  if (res_dft == a_dft) {
    reim_fftvec_automorphism_inplace(module->mod.fft64.p_automorphism, p, (double*)res_dft, smin, tmp_bytes);
  } else {
    reim_fftvec_automorphism(module->mod.fft64.p_automorphism, p, (double*)res_dft, (double*)a_dft, smin);
  }

  int64_t* const fres_dft = (int64_t*)res_dft;
  uint64_t nn = module->nn;
  memset(fres_dft + smin * nn, 0, (res_size - smin) * nn * sizeof(double));
}

EXPORT void fft64_vec_znx_idft(const MODULE* module,                       // N
                               VEC_ZNX_BIG* res, uint64_t res_size,        // res
                               const VEC_ZNX_DFT* a_dft, uint64_t a_size,  // a
                               uint8_t* tmp                                // unused
) {
  const uint64_t nn = module->nn;
  const uint64_t smin = res_size < a_size ? res_size : a_size;
  if ((double*)res != (double*)a_dft) {
    memcpy(res, a_dft, smin * nn * sizeof(double));
  }

  for (uint64_t i = 0; i < smin; i++) {
    reim_ifft(module->mod.fft64.p_ifft, ((double*)res) + i * nn);
    reim_to_znx64(module->mod.fft64.p_reim_to_znx, ((int64_t*)res) + i * nn, ((int64_t*)res) + i * nn);
  }

  // fill up remaining part with 0's
  int64_t* const dres = (int64_t*)res;
  memset(dres + smin * nn, 0, (res_size - smin) * nn * sizeof(double));
}

EXPORT uint64_t fft64_vec_znx_idft_tmp_bytes(const MODULE* module) { return 0; }

EXPORT void fft64_vec_znx_idft_tmp_a(const MODULE* module,                 // N
                                     VEC_ZNX_BIG* res, uint64_t res_size,  // res
                                     VEC_ZNX_DFT* a_dft, uint64_t a_size   // a is overwritten
) {
  const uint64_t nn = module->nn;
  const uint64_t smin = res_size < a_size ? res_size : a_size;

  int64_t* const tres = (int64_t*)res;
  double* const ta = (double*)a_dft;
  for (uint64_t i = 0; i < smin; i++) {
    reim_ifft(module->mod.fft64.p_ifft, ta + i * nn);
    reim_to_znx64(module->mod.fft64.p_reim_to_znx, tres + i * nn, ta + i * nn);
  }

  // fill up remaining part with 0's
  memset(tres + smin * nn, 0, (res_size - smin) * nn * sizeof(double));
}

// ntt120 backend

EXPORT void ntt120_vec_znx_dft_avx(const MODULE* module,                             // N
                                   VEC_ZNX_DFT* res, uint64_t res_size,              // res
                                   const int64_t* a, uint64_t a_size, uint64_t a_sl  // a
) {
  const uint64_t nn = module->nn;
  const uint64_t smin = res_size < a_size ? res_size : a_size;

  int64_t* tres = (int64_t*)res;
  for (uint64_t i = 0; i < smin; i++) {
    q120_b_from_znx64_simple(nn, (q120b*)(tres + i * nn * 4), a + i * a_sl);
    q120_ntt_bb_avx2(module->mod.q120.p_ntt, (q120b*)(tres + i * nn * 4));
  }

  // fill up remaining part with 0's
  memset(tres + smin * nn * 4, 0, (res_size - smin) * nn * 4 * sizeof(int64_t));
}

EXPORT void ntt120_vec_znx_idft_avx(const MODULE* module,                       // N
                                    VEC_ZNX_BIG* res, uint64_t res_size,        // res
                                    const VEC_ZNX_DFT* a_dft, uint64_t a_size,  // a
                                    uint8_t* tmp) {
  const uint64_t nn = module->nn;
  const uint64_t smin = res_size < a_size ? res_size : a_size;

  __int128_t* const tres = (__int128_t*)res;
  const int64_t* const ta = (int64_t*)a_dft;
  for (uint64_t i = 0; i < smin; i++) {
    memcpy(tmp, ta + i * nn * 4, nn * 4 * sizeof(uint64_t));
    q120_intt_bb_avx2(module->mod.q120.p_intt, (q120b*)tmp);
    q120_b_to_znx128_simple(nn, tres + i * nn, (q120b*)tmp);
  }

  // fill up remaining part with 0's
  memset(tres + smin * nn, 0, (res_size - smin) * nn * sizeof(*tres));
}

EXPORT uint64_t ntt120_vec_znx_idft_tmp_bytes_avx(const MODULE* module) { return module->nn * 4 * sizeof(uint64_t); }

EXPORT void ntt120_vec_znx_idft_tmp_a_avx(const MODULE* module,                 // N
                                          VEC_ZNX_BIG* res, uint64_t res_size,  // res
                                          VEC_ZNX_DFT* a_dft, uint64_t a_size   // a is overwritten
) {
  const uint64_t nn = module->nn;
  const uint64_t smin = res_size < a_size ? res_size : a_size;

  __int128_t* const tres = (__int128_t*)res;
  int64_t* const ta = (int64_t*)a_dft;
  for (uint64_t i = 0; i < smin; i++) {
    q120_intt_bb_avx2(module->mod.q120.p_intt, (q120b*)(ta + i * nn * 4));
    q120_b_to_znx128_simple(nn, tres + i * nn, (q120b*)(ta + i * nn * 4));
  }

  // fill up remaining part with 0's
  memset(tres + smin * nn, 0, (res_size - smin) * nn * sizeof(*tres));
}
