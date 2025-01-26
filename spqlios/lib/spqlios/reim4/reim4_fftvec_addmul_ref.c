#include <errno.h>
#include <memory.h>
#include <stdlib.h>
#include <string.h>

#include "../commons_private.h"
#include "reim4_fftvec_internal.h"
#include "reim4_fftvec_private.h"

void* init_reim4_fftvec_addmul_precomp(REIM4_FFTVEC_ADDMUL_PRECOMP* res, uint32_t m) {
  res->m = m;
  if (CPU_SUPPORTS("fma")) {
    if (m >= 2) {
      res->function = reim4_fftvec_addmul_fma;
    } else {
      res->function = reim4_fftvec_addmul_ref;
    }
  } else {
    res->function = reim4_fftvec_addmul_ref;
  }
  return res;
}

EXPORT REIM4_FFTVEC_ADDMUL_PRECOMP* new_reim4_fftvec_addmul_precomp(uint32_t m) {
  REIM4_FFTVEC_ADDMUL_PRECOMP* res = malloc(sizeof(*res));
  if (!res) return spqlios_error(strerror(errno));
  return spqlios_keep_or_free(res, init_reim4_fftvec_addmul_precomp(res, m));
}

EXPORT void reim4_fftvec_addmul_ref(const REIM4_FFTVEC_ADDMUL_PRECOMP* precomp, double* r, const double* a,
                                    const double* b) {
  const uint64_t m = precomp->m;
  for (uint64_t j = 0; j < m / 4; ++j) {
    for (uint64_t i = 0; i < 4; ++i) {
      double re = a[i] * b[i] - a[i + 4] * b[i + 4];
      double im = a[i] * b[i + 4] + a[i + 4] * b[i];
      r[i] += re;
      r[i + 4] += im;
    }
    a += 8;
    b += 8;
    r += 8;
  }
}

EXPORT void reim4_fftvec_addmul_simple(uint32_t m, double* r, const double* a, const double* b) {
  static REIM4_FFTVEC_ADDMUL_PRECOMP precomp[32];
  REIM4_FFTVEC_ADDMUL_PRECOMP* p = precomp + log2m(m);
  if (!p->function) {
    if (!init_reim4_fftvec_addmul_precomp(p, m)) abort();
  }
  p->function(p, r, a, b);
}

void* init_reim4_fftvec_mul_precomp(REIM4_FFTVEC_MUL_PRECOMP* res, uint32_t m) {
  res->m = m;
  if (CPU_SUPPORTS("fma")) {
    if (m >= 4) {
      res->function = reim4_fftvec_mul_fma;
    } else {
      res->function = reim4_fftvec_mul_ref;
    }
  } else {
    res->function = reim4_fftvec_mul_ref;
  }
  return res;
}

EXPORT REIM4_FFTVEC_MUL_PRECOMP* new_reim4_fftvec_mul_precomp(uint32_t m) {
  REIM4_FFTVEC_MUL_PRECOMP* res = malloc(sizeof(*res));
  if (!res) return spqlios_error(strerror(errno));
  return spqlios_keep_or_free(res, init_reim4_fftvec_mul_precomp(res, m));
}

EXPORT void reim4_fftvec_mul_ref(const REIM4_FFTVEC_MUL_PRECOMP* precomp, double* r, const double* a, const double* b) {
  const uint64_t m = precomp->m;
  for (uint64_t j = 0; j < m / 4; ++j) {
    for (uint64_t i = 0; i < 4; ++i) {
      double re = a[i] * b[i] - a[i + 4] * b[i + 4];
      double im = a[i] * b[i + 4] + a[i + 4] * b[i];
      r[i] = re;
      r[i + 4] = im;
    }
    a += 8;
    b += 8;
    r += 8;
  }
}

EXPORT void reim4_fftvec_mul_simple(uint32_t m, double* r, const double* a, const double* b) {
  static REIM4_FFTVEC_MUL_PRECOMP precomp[32];
  REIM4_FFTVEC_MUL_PRECOMP* p = precomp + log2m(m);
  if (!p->function) {
    if (!init_reim4_fftvec_mul_precomp(p, m)) abort();
  }
  p->function(p, r, a, b);
}
