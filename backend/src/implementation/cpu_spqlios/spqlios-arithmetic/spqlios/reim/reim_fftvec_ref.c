#include <string.h>

#include "../commons_private.h"
#include "reim_fft_internal.h"
#include "reim_fft_private.h"

// Computes X^i -> X^(p*i) in the fourier domain for a reim vector is size 2 * m * size
// This function cannot be evaluated in place.
EXPORT void reim_fftvec_automorphism_ref(const REIM_FFTVEC_AUTOMORPHISM_PRECOMP* tables, int64_t p, double* r,
                                         const double* a, uint64_t a_size) {
  const uint64_t m = tables->m;
  const uint64_t nn = 2 * m;
  const uint64_t* irev = tables->irev;
  const uint64_t mask = (4 * m - 1);
  const uint64_t conj = !((p & 3) == 1);
  p = p & mask;
  p *= 1 - (conj << 1);
  if (a_size == 1) {
    for (uint64_t i = 0; i < m; ++i) {
      uint64_t i_rev = 2 * irev[i] + 1;
      i_rev = (((p * i_rev) & mask) - 1) >> 1;
      uint64_t j = irev[i_rev];
      r[i] = a[j];
      double x = a[j + m];
      r[i + m] = conj ? -x : x;
    }
  } else {
    for (uint64_t i = 0; i < m; ++i) {
      uint64_t i_rev = 2 * irev[i] + 1;
      i_rev = (((p * i_rev) & mask) - 1) >> 1;
      uint64_t j = irev[i_rev];
      for (uint64_t k = 0; k < a_size; ++k) {
        uint64_t offset_re = k * nn;
        uint64_t offset_im = offset_re + m;
        r[i + offset_re] = a[j + offset_re];
        double x = a[j + offset_im];
        r[i + offset_im] = conj ? -x : x;
      }
    }
  }
}

// Computes the permutation index for an automorphism X^{i} -> X^{i*p} in the fourier domain.
EXPORT void reim_fftvec_automorphism_lut_precomp_ref(const REIM_FFTVEC_AUTOMORPHISM_PRECOMP* tables, int64_t p,
                                                     uint64_t* precomp  // size m
) {
  const uint64_t m = tables->m;
  const uint64_t* irev = tables->irev;
  const uint64_t mask = (4 * m - 1);
  const uint64_t conj = !((p & 3) == 1);
  p = p & mask;
  p *= 1 - (conj << 1);
  for (uint64_t i = 0; i < m; ++i) {
    uint64_t i_rev = 2 * irev[i] + 1;
    i_rev = (((p * i_rev) & mask) - 1) >> 1;
    uint64_t j = irev[i_rev];
    precomp[i] = j;
  }
}

// Computes X^{i} -> X^{i*p} in the fourier domain for a reim vector of size m using a precomputed lut permutation.
void reim_fftvec_automorphism_with_lut_ref(const REIM_FFTVEC_AUTOMORPHISM_PRECOMP* tables, uint64_t* precomp, double* r,
                                           const double* a) {
  const uint64_t m = tables->m;
  for (uint64_t i = 0; i < m; ++i) {
    uint64_t j = precomp[i];
    r[i] = a[j];
    r[i + m] = a[j + m];
  }
}

// Computes X^{i} -> X^{i*-p} in the fourier domain for a reim vector of size m using a precomputed lut permutation.
void reim_fftvec_automorphism_conj_with_lut_ref(const REIM_FFTVEC_AUTOMORPHISM_PRECOMP* tables, uint64_t* precomp,
                                                double* r, const double* a) {
  const uint64_t m = tables->m;
  for (uint64_t i = 0; i < m; ++i) {
    uint64_t j = precomp[i];
    r[i] = a[j];
    r[i + m] = -a[j + m];
  }
}

// Returns the minimum number of temporary bytes used by reim_fftvec_automorphism_inplace_ref.
EXPORT uint64_t reim_fftvec_automorphism_inplace_tmp_bytes_ref(const REIM_FFTVEC_AUTOMORPHISM_PRECOMP* tables) {
  const uint64_t m = tables->m;
  return m * (2 * sizeof(double) + sizeof(uint64_t));
}

// Computes X^i -> X^(p*i) in the fourier domain for a reim vector is size 2 * m * a_size
// This function cannot be evaluated in place.
EXPORT void reim_fftvec_automorphism_inplace_ref(const REIM_FFTVEC_AUTOMORPHISM_PRECOMP* tables, int64_t p, double* a,
                                                 uint64_t a_size,
                                                 uint8_t* tmp_bytes  // m * (2*sizeof(double) + sizeof(uint64_t))
) {
  const uint64_t m = tables->m;
  const uint64_t nn = 2 * m;
  const uint64_t* irev = tables->irev;
  const uint64_t mask = (4 * m - 1);
  const uint64_t conj = !((p & 3) == 1);

  double* tmp = (double*)tmp_bytes;
  p = p & mask;
  if (a_size == 1) {
    p *= 1 - (conj << 1);
    for (uint64_t i = 0; i < m; ++i) {
      uint64_t i_rev = 2 * irev[i] + 1;
      i_rev = (((p * i_rev) & mask) - 1) >> 1;
      uint64_t j = irev[i_rev];
      tmp[i] = a[j];
      double x = a[j + m];
      tmp[i + m] = conj ? -x : x;
    }
    memcpy(a, tmp, nn * sizeof(double));
  } else {
    uint64_t* lut = (uint64_t*)(tmp_bytes + nn * sizeof(double));
    reim_fftvec_automorphism_lut_precomp_ref(tables, p, lut);
    for (uint64_t i = 0; i < a_size; ++i) {
      if (conj == 1) {
        reim_fftvec_automorphism_conj_with_lut_ref(tables, lut, tmp, a + i * nn);
      } else {
        reim_fftvec_automorphism_with_lut_ref(tables, lut, tmp, a + i * nn);
      }
      memcpy(a + i * nn, tmp, nn * sizeof(double));
    }
  }
}

EXPORT void reim_fftvec_addmul_ref(const REIM_FFTVEC_ADDMUL_PRECOMP* tables, double* r, const double* a,
                                   const double* b) {
  const uint64_t m = tables->m;
  for (uint64_t i = 0; i < m; ++i) {
    double re = a[i] * b[i] - a[i + m] * b[i + m];
    double im = a[i] * b[i + m] + a[i + m] * b[i];
    r[i] += re;
    r[i + m] += im;
  }
}

EXPORT void reim_fftvec_add_ref(const REIM_FFTVEC_ADD_PRECOMP* precomp, double* r, const double* a, const double* b) {
  const uint64_t m = precomp->m;
  for (uint64_t i = 0; i < m; ++i) {
    double re = a[i] + b[i];
    double im = a[i + m] + b[i + m];
    r[i] = re;
    r[i + m] = im;
  }
}

EXPORT void reim_fftvec_sub_ref(const REIM_FFTVEC_SUB_PRECOMP* precomp, double* r, const double* a, const double* b) {
  const uint64_t m = precomp->m;
  for (uint64_t i = 0; i < m; ++i) {
    double re = a[i] - b[i];
    double im = a[i + m] - b[i + m];
    r[i] = re;
    r[i + m] = im;
  }
}

EXPORT void reim_fftvec_mul_ref(const REIM_FFTVEC_MUL_PRECOMP* tables, double* r, const double* a, const double* b) {
  const uint64_t m = tables->m;
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

EXPORT REIM_FFTVEC_ADD_PRECOMP* new_reim_fftvec_add_precomp(uint32_t m) {
  REIM_FFTVEC_ADD_PRECOMP* reps = malloc(sizeof(REIM_FFTVEC_ADD_PRECOMP));
  reps->m = m;
  if (CPU_SUPPORTS("fma")) {
    if (m >= 4) {
      reps->function = reim_fftvec_add_fma;
    } else {
      reps->function = reim_fftvec_add_ref;
    }
  } else {
    reps->function = reim_fftvec_add_ref;
  }
  return reps;
}

EXPORT REIM_FFTVEC_SUB_PRECOMP* new_reim_fftvec_sub_precomp(uint32_t m) {
  REIM_FFTVEC_SUB_PRECOMP* reps = malloc(sizeof(REIM_FFTVEC_SUB_PRECOMP));
  reps->m = m;
  if (CPU_SUPPORTS("fma")) {
    if (m >= 4) {
      reps->function = reim_fftvec_sub_fma;
    } else {
      reps->function = reim_fftvec_sub_ref;
    }
  } else {
    reps->function = reim_fftvec_sub_ref;
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

EXPORT REIM_FFTVEC_AUTOMORPHISM_PRECOMP* new_reim_fftvec_automorphism_precomp(uint32_t m) {
  REIM_FFTVEC_AUTOMORPHISM_PRECOMP* reps = malloc(sizeof(REIM_FFTVEC_AUTOMORPHISM_PRECOMP));
  reps->m = m;
  reps->function.apply = reim_fftvec_automorphism_ref;
  reps->function.apply_inplace = reim_fftvec_automorphism_inplace_ref;
  reps->function.apply_inplace_tmp_bytes = reim_fftvec_automorphism_inplace_tmp_bytes_ref;
  const uint64_t nn = 2 * m;
  reps->irev = malloc(sizeof(uint64_t) * nn);
  uint32_t lognn = log2m(nn);
  for (uint32_t i = 0; i < nn; i++) {
    reps->irev[i] = (uint64_t)revbits(lognn, i);
  }
  return reps;
}
