#include <errno.h>
#include <string.h>

#include "../commons_private.h"
#include "cplx_fft_internal.h"
#include "cplx_fft_private.h"

EXPORT void cplx_from_znx32_ref(const CPLX_FROM_ZNX32_PRECOMP* precomp, void* r, const int32_t* x) {
  const uint32_t m = precomp->m;
  const int32_t* inre = x;
  const int32_t* inim = x + m;
  CPLX* out = r;
  for (uint32_t i = 0; i < m; ++i) {
    out[i][0] = (double)inre[i];
    out[i][1] = (double)inim[i];
  }
}

EXPORT void cplx_from_tnx32_ref(const CPLX_FROM_TNX32_PRECOMP* precomp, void* r, const int32_t* x) {
  static const double _2p32 = 1. / (INT64_C(1) << 32);
  const uint32_t m = precomp->m;
  const int32_t* inre = x;
  const int32_t* inim = x + m;
  CPLX* out = r;
  for (uint32_t i = 0; i < m; ++i) {
    out[i][0] = ((double)inre[i]) * _2p32;
    out[i][1] = ((double)inim[i]) * _2p32;
  }
}

EXPORT void cplx_to_tnx32_ref(const CPLX_TO_TNX32_PRECOMP* precomp, int32_t* r, const void* x) {
  static const double _2p32 = (INT64_C(1) << 32);
  const uint32_t m = precomp->m;
  double factor = _2p32 / precomp->divisor;
  int32_t* outre = r;
  int32_t* outim = r + m;
  const CPLX* in = x;
  // Note: this formula will only work if abs(in) < 2^32
  for (uint32_t i = 0; i < m; ++i) {
    outre[i] = (int32_t)(int64_t)(rint(in[i][0] * factor));
    outim[i] = (int32_t)(int64_t)(rint(in[i][1] * factor));
  }
}

void* init_cplx_from_znx32_precomp(CPLX_FROM_ZNX32_PRECOMP* res, uint32_t m) {
  res->m = m;
  if (CPU_SUPPORTS("avx2")) {
    if (m >= 8) {
      res->function = cplx_from_znx32_avx2_fma;
    } else {
      res->function = cplx_from_znx32_ref;
    }
  } else {
    res->function = cplx_from_znx32_ref;
  }
  return res;
}

CPLX_FROM_ZNX32_PRECOMP* new_cplx_from_znx32_precomp(uint32_t m) {
  CPLX_FROM_ZNX32_PRECOMP* res = malloc(sizeof(CPLX_FROM_ZNX32_PRECOMP));
  if (!res) return spqlios_error(strerror(errno));
  return spqlios_keep_or_free(res, init_cplx_from_znx32_precomp(res, m));
}

void* init_cplx_from_tnx32_precomp(CPLX_FROM_TNX32_PRECOMP* res, uint32_t m) {
  res->m = m;
  if (CPU_SUPPORTS("avx2")) {
    if (m >= 8) {
      res->function = cplx_from_tnx32_avx2_fma;
    } else {
      res->function = cplx_from_tnx32_ref;
    }
  } else {
    res->function = cplx_from_tnx32_ref;
  }
  return res;
}

CPLX_FROM_TNX32_PRECOMP* new_cplx_from_tnx32_precomp(uint32_t m) {
  CPLX_FROM_TNX32_PRECOMP* res = malloc(sizeof(CPLX_FROM_TNX32_PRECOMP));
  if (!res) return spqlios_error(strerror(errno));
  return spqlios_keep_or_free(res, init_cplx_from_tnx32_precomp(res, m));
}

void* init_cplx_to_tnx32_precomp(CPLX_TO_TNX32_PRECOMP* res, uint32_t m, double divisor, uint32_t log2overhead) {
  if (is_not_pow2_double(&divisor)) return spqlios_error("divisor must be a power of 2");
  if (m & (m - 1)) return spqlios_error("m must be a power of 2");
  if (log2overhead > 52) return spqlios_error("log2overhead is too large");
  res->m = m;
  res->divisor = divisor;
  if (CPU_SUPPORTS("avx2")) {
    if (log2overhead <= 18) {
      if (m >= 8) {
        res->function = cplx_to_tnx32_avx2_fma;
      } else {
        res->function = cplx_to_tnx32_ref;
      }
    } else {
      res->function = cplx_to_tnx32_ref;
    }
  } else {
    res->function = cplx_to_tnx32_ref;
  }
  return res;
}

EXPORT CPLX_TO_TNX32_PRECOMP* new_cplx_to_tnx32_precomp(uint32_t m, double divisor, uint32_t log2overhead) {
  CPLX_TO_TNX32_PRECOMP* res = malloc(sizeof(CPLX_TO_TNX32_PRECOMP));
  if (!res) return spqlios_error(strerror(errno));
  return spqlios_keep_or_free(res, init_cplx_to_tnx32_precomp(res, m, divisor, log2overhead));
}

/**
 * @brief Simpler API for the znx32 to cplx conversion.
 * For each dimension, the precomputed tables for this dimension are generated automatically the first time.
 * It is advised to do one dry-run call per desired dimension before using in a multithread environment */
EXPORT void cplx_from_znx32_simple(uint32_t m, void* r, const int32_t* x) {
  // not checking for log2bound which is not relevant here
  static CPLX_FROM_ZNX32_PRECOMP precomp[32];
  CPLX_FROM_ZNX32_PRECOMP* p = precomp + log2m(m);
  if (!p->function) {
    if (!init_cplx_from_znx32_precomp(p, m)) abort();
  }
  p->function(p, r, x);
}

/**
 * @brief Simpler API for the tnx32 to cplx conversion.
 * For each dimension, the precomputed tables for this dimension are generated automatically the first time.
 * It is advised to do one dry-run call per desired dimension before using in a multithread environment */
EXPORT void cplx_from_tnx32_simple(uint32_t m, void* r, const int32_t* x) {
  static CPLX_FROM_TNX32_PRECOMP precomp[32];
  CPLX_FROM_TNX32_PRECOMP* p = precomp + log2m(m);
  if (!p->function) {
    if (!init_cplx_from_tnx32_precomp(p, m)) abort();
  }
  p->function(p, r, x);
}
/**
 * @brief Simpler API for the cplx to tnx32 conversion.
 * For each dimension, the precomputed tables for this dimension are generated automatically the first time.
 * It is advised to do one dry-run call per desired dimension before using in a multithread environment */
EXPORT void cplx_to_tnx32_simple(uint32_t m, double divisor, uint32_t log2overhead, int32_t* r, const void* x) {
  struct LAST_CPLX_TO_TNX32_PRECOMP {
    CPLX_TO_TNX32_PRECOMP p;
    double last_divisor;
    double last_log2over;
  };
  static __thread struct LAST_CPLX_TO_TNX32_PRECOMP precomp[32];
  struct LAST_CPLX_TO_TNX32_PRECOMP* p = precomp + log2m(m);
  if (!p->p.function || divisor != p->last_divisor || log2overhead != p->last_log2over) {
    memset(p, 0, sizeof(*p));
    if (!init_cplx_to_tnx32_precomp(&p->p, m, divisor, log2overhead)) abort();
    p->last_divisor = divisor;
    p->last_log2over = log2overhead;
  }
  p->p.function(&p->p, r, x);
}
