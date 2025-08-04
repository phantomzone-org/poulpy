#include <errno.h>
#include <string.h>

#include "../commons_private.h"
#include "reim_fft_internal.h"
#include "reim_fft_private.h"

void reim_from_znx64_ref(const REIM_FROM_ZNX64_PRECOMP* precomp, void* r, const int64_t* x) {
  // naive version of the function (just cast)
  const uint64_t nn = precomp->m << 1;
  double* res = (double*)r;
  for (uint64_t i = 0; i < nn; ++i) {
    res[i] = (double)x[i];
  }
}

void* init_reim_from_znx64_precomp(REIM_FROM_ZNX64_PRECOMP* const res, uint32_t m, uint32_t log2bound) {
  if (m & (m - 1)) return spqlios_error("m must be a power of 2");
  // currently, we are going to use the trick add 3.2^51, mask the exponent, reinterpret bits as double.
  // therefore we need the input values to be < 2^50.
  if (log2bound > 50) return spqlios_error("Invalid log2bound error: must be in [0,50]");
  res->m = m;
  FROM_ZNX64_FUNC resf = reim_from_znx64_ref;
  if (m >= 8) {
    if (CPU_SUPPORTS("avx2")) {
      resf = reim_from_znx64_bnd50_fma;
    }
  }
  res->function = resf;
  return res;
}

EXPORT REIM_FROM_ZNX64_PRECOMP* new_reim_from_znx64_precomp(uint32_t m, uint32_t log2bound) {
  REIM_FROM_ZNX64_PRECOMP* res = malloc(sizeof(*res));
  if (!res) return spqlios_error(strerror(errno));
  return spqlios_keep_or_free(res, init_reim_from_znx64_precomp(res, m, log2bound));
}

EXPORT void reim_from_znx64(const REIM_FROM_ZNX64_PRECOMP* tables, void* r, const int64_t* a) {
  tables->function(tables, r, a);
}

/**
 * @brief Simpler API for the znx64 to reim conversion.
 * For each dimension, the precomputed tables for this dimension are generated automatically the first time.
 * It is advised to do one dry-run call per desired dimension before using in a multithread environment
 */
EXPORT void reim_from_znx64_simple(uint32_t m, uint32_t log2bound, void* r, const int64_t* a) {
  // not checking for log2bound which is not relevant here
  static REIM_FROM_ZNX64_PRECOMP precomp[32];
  REIM_FROM_ZNX64_PRECOMP* p = precomp + log2m(m);
  if (!p->function) {
    if (!init_reim_from_znx64_precomp(p, m, log2bound)) abort();
  }
  p->function(p, r, a);
}

void reim_from_znx32_ref(const REIM_FROM_ZNX32_PRECOMP* precomp, void* r, const int32_t* x) { NOT_IMPLEMENTED(); }
void reim_from_znx32_avx2_fma(const REIM_FROM_ZNX32_PRECOMP* precomp, void* r, const int32_t* x) { NOT_IMPLEMENTED(); }

void* init_reim_from_znx32_precomp(REIM_FROM_ZNX32_PRECOMP* const res, uint32_t m, uint32_t log2bound) {
  if (m & (m - 1)) return spqlios_error("m must be a power of 2");
  if (log2bound > 32) return spqlios_error("Invalid log2bound error: must be in [0,32]");
  res->m = m;
  // TODO: check selection logic
  if (CPU_SUPPORTS("avx2")) {
    if (m >= 8) {
      res->function = reim_from_znx32_avx2_fma;
    } else {
      res->function = reim_from_znx32_ref;
    }
  } else {
    res->function = reim_from_znx32_ref;
  }
  return res;
}

EXPORT REIM_FROM_ZNX32_PRECOMP* new_reim_from_znx32_precomp(uint32_t m, uint32_t log2bound) {
  REIM_FROM_ZNX32_PRECOMP* res = malloc(sizeof(*res));
  if (!res) return spqlios_error(strerror(errno));
  return spqlios_keep_or_free(res, init_reim_from_znx32_precomp(res, m, log2bound));
}

void reim_from_tnx32_ref(const REIM_FROM_TNX32_PRECOMP* precomp, void* r, const int32_t* x) { NOT_IMPLEMENTED(); }
void reim_from_tnx32_avx2_fma(const REIM_FROM_TNX32_PRECOMP* precomp, void* r, const int32_t* x) { NOT_IMPLEMENTED(); }

void* init_reim_from_tnx32_precomp(REIM_FROM_TNX32_PRECOMP* const res, uint32_t m) {
  if (m & (m - 1)) return spqlios_error("m must be a power of 2");
  res->m = m;
  // TODO: check selection logic
  if (CPU_SUPPORTS("avx2")) {
    if (m >= 8) {
      res->function = reim_from_tnx32_avx2_fma;
    } else {
      res->function = reim_from_tnx32_ref;
    }
  } else {
    res->function = reim_from_tnx32_ref;
  }
  return res;
}

EXPORT REIM_FROM_TNX32_PRECOMP* new_reim_from_tnx32_precomp(uint32_t m) {
  REIM_FROM_TNX32_PRECOMP* res = malloc(sizeof(*res));
  if (!res) return spqlios_error(strerror(errno));
  return spqlios_keep_or_free(res, init_reim_from_tnx32_precomp(res, m));
}

void reim_to_tnx32_ref(const REIM_TO_TNX32_PRECOMP* precomp, int32_t* r, const void* x) { NOT_IMPLEMENTED(); }
void reim_to_tnx32_avx2_fma(const REIM_TO_TNX32_PRECOMP* precomp, int32_t* r, const void* x) { NOT_IMPLEMENTED(); }

void* init_reim_to_tnx32_precomp(REIM_TO_TNX32_PRECOMP* const res, uint32_t m, double divisor, uint32_t log2overhead) {
  if (m & (m - 1)) return spqlios_error("m must be a power of 2");
  if (is_not_pow2_double(&divisor)) return spqlios_error("divisor must be a power of 2");
  if (log2overhead > 52) return spqlios_error("log2overhead is too large");
  res->m = m;
  res->divisor = divisor;
  // TODO: check selection logic
  if (CPU_SUPPORTS("avx2")) {
    if (log2overhead <= 18) {
      if (m >= 8) {
        res->function = reim_to_tnx32_avx2_fma;
      } else {
        res->function = reim_to_tnx32_ref;
      }
    } else {
      res->function = reim_to_tnx32_ref;
    }
  } else {
    res->function = reim_to_tnx32_ref;
  }
  return res;
}

EXPORT REIM_TO_TNX32_PRECOMP* new_reim_to_tnx32_precomp(uint32_t m, double divisor, uint32_t log2overhead) {
  REIM_TO_TNX32_PRECOMP* res = malloc(sizeof(*res));
  if (!res) return spqlios_error(strerror(errno));
  return spqlios_keep_or_free(res, init_reim_to_tnx32_precomp(res, m, divisor, log2overhead));
}

EXPORT void reim_from_znx32_simple(uint32_t m, uint32_t log2bound, void* r, const int32_t* x) {
  static REIM_FROM_ZNX32_PRECOMP* p[31] = {0};
  REIM_FROM_ZNX32_PRECOMP** f = p + log2m(m);
  if (!*f) *f = new_reim_from_znx32_precomp(m, log2bound);
  (*f)->function(*f, r, x);
}

EXPORT void reim_from_tnx32_simple(uint32_t m, void* r, const int32_t* x) {
  static REIM_FROM_TNX32_PRECOMP* p[31] = {0};
  REIM_FROM_TNX32_PRECOMP** f = p + log2m(m);
  if (!*f) *f = new_reim_from_tnx32_precomp(m);
  (*f)->function(*f, r, x);
}

EXPORT void reim_to_tnx32_simple(uint32_t m, double divisor, uint32_t log2overhead, int32_t* r, const void* x) {
  static REIM_TO_TNX32_PRECOMP* p[31] = {0};
  REIM_TO_TNX32_PRECOMP** f = p + log2m(m);
  if (!*f) *f = new_reim_to_tnx32_precomp(m, divisor, log2overhead);
  (*f)->function(*f, r, x);
}

void reim_to_znx64_ref(const REIM_TO_ZNX64_PRECOMP* precomp, int64_t* r, const void* x) {
  // for now, we stick to a slow implem
  uint64_t nn = precomp->m << 1;
  const double* v = (double*)x;
  double invdiv = 1. / precomp->divisor;
  for (uint64_t i = 0; i < nn; ++i) {
    r[i] = (int64_t)rint(v[i] * invdiv);
  }
}

void* init_reim_to_znx64_precomp(REIM_TO_ZNX64_PRECOMP* const res, uint32_t m, double divisor, uint32_t log2bound) {
  if (m & (m - 1)) return spqlios_error("m must be a power of 2");
  if (is_not_pow2_double(&divisor)) return spqlios_error("divisor must be a power of 2");
  if (log2bound > 64) return spqlios_error("log2bound is too large");
  res->m = m;
  res->divisor = divisor;
  TO_ZNX64_FUNC resf = reim_to_znx64_ref;
  if (CPU_SUPPORTS("avx2") && m >= 8) {
    if (log2bound <= 50) {
      resf = reim_to_znx64_avx2_bnd50_fma;
    } else {
      resf = reim_to_znx64_avx2_bnd63_fma;
    }
  }
  res->function = resf;  // must be the last one set
  return res;
}

EXPORT REIM_TO_ZNX64_PRECOMP* new_reim_to_znx64_precomp(uint32_t m, double divisor, uint32_t log2bound) {
  REIM_TO_ZNX64_PRECOMP* res = malloc(sizeof(*res));
  if (!res) return spqlios_error(strerror(errno));
  return spqlios_keep_or_free(res, init_reim_to_znx64_precomp(res, m, divisor, log2bound));
}

EXPORT void reim_to_znx64(const REIM_TO_ZNX64_PRECOMP* precomp, int64_t* r, const void* a) {
  precomp->function(precomp, r, a);
}

/**
 * @brief Simpler API for the znx64 to reim conversion.
 */
EXPORT void reim_to_znx64_simple(uint32_t m, double divisor, uint32_t log2bound, int64_t* r, const void* a) {
  // not checking distinguishing <=50 or not
  static __thread REIM_TO_ZNX64_PRECOMP p;
  static __thread uint32_t prev_log2bound;
  if (!p.function || p.m != m || p.divisor != divisor || prev_log2bound != log2bound) {
    if (!init_reim_to_znx64_precomp(&p, m, divisor, log2bound)) abort();
    prev_log2bound = log2bound;
  }
  p.function(&p, r, a);
}
