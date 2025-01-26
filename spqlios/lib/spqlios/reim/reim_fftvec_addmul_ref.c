#include "reim_fft_internal.h"
#include "reim_fft_private.h"

EXPORT void reim_fftvec_addmul_ref(const REIM_FFTVEC_ADDMUL_PRECOMP* precomp, double* r, const double* a,
                                   const double* b) {
  const uint64_t m = precomp->m;
  for (uint64_t i = 0; i < m; ++i) {
    double re = a[i] * b[i] - a[i + m] * b[i + m];
    double im = a[i] * b[i + m] + a[i + m] * b[i];
    r[i] += re;
    r[i + m] += im;
  }
}

EXPORT void reim_fftvec_mul_ref(const REIM_FFTVEC_MUL_PRECOMP* precomp, double* r, const double* a, const double* b) {
  const uint64_t m = precomp->m;
  for (uint64_t i = 0; i < m; ++i) {
    double re = a[i] * b[i] - a[i + m] * b[i + m];
    double im = a[i] * b[i + m] + a[i + m] * b[i];
    r[i] = re;
    r[i + m] = im;
  }
}

EXPORT REIM_FFTVEC_ADDMUL_PRECOMP* new_reim_fftvec_addmul_precomp(uint32_t m) {
  REIM_FFTVEC_ADDMUL_PRECOMP* reps = malloc(sizeof(REIM_FFTVEC_ADDMUL_PRECOMP));
  reps->m = m;
  if (CPU_SUPPORTS("fma")) {
    if (m >= 4) {
      reps->function = reim_fftvec_addmul_fma;
    } else {
      reps->function = reim_fftvec_addmul_ref;
    }
  } else {
    reps->function = reim_fftvec_addmul_ref;
  }
  return reps;
}

EXPORT REIM_FFTVEC_MUL_PRECOMP* new_reim_fftvec_mul_precomp(uint32_t m) {
  REIM_FFTVEC_MUL_PRECOMP* reps = malloc(sizeof(REIM_FFTVEC_MUL_PRECOMP));
  reps->m = m;
  if (CPU_SUPPORTS("fma")) {
    if (m >= 4) {
      reps->function = reim_fftvec_mul_fma;
    } else {
      reps->function = reim_fftvec_mul_ref;
    }
  } else {
    reps->function = reim_fftvec_mul_ref;
  }
  return reps;
}

