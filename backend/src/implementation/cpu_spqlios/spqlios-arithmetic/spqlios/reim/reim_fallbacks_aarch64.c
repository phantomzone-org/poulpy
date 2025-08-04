#include "reim_fft_private.h"

EXPORT void reim_fftvec_addmul_fma(const REIM_FFTVEC_ADDMUL_PRECOMP* precomp, double* r, const double* a,
                                   const double* b) {
  UNDEFINED();
}
EXPORT void reim_fftvec_mul_fma(const REIM_FFTVEC_MUL_PRECOMP* precomp, double* r, const double* a, const double* b) {
  UNDEFINED();
}
EXPORT void reim_fftvec_add_fma(const REIM_FFTVEC_ADD_PRECOMP* precomp, double* r, const double* a, const double* b) {
  UNDEFINED();
}
EXPORT void reim_fftvec_sub_fma(const REIM_FFTVEC_SUB_PRECOMP* precomp, double* r, const double* a, const double* b) {
  UNDEFINED();
}

EXPORT void reim_fft_avx2_fma(const REIM_FFT_PRECOMP* tables, double* data) { UNDEFINED(); }
EXPORT void reim_ifft_avx2_fma(const REIM_IFFT_PRECOMP* tables, double* data) { UNDEFINED(); }

// EXPORT void reim_fft(const REIM_FFT_PRECOMP* tables, double* data) { tables->function(tables, data); }
// EXPORT void reim_ifft(const REIM_IFFT_PRECOMP* tables, double* data) { tables->function(tables, data); }
