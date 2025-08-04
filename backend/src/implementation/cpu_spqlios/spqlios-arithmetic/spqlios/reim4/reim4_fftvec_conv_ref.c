#include <errno.h>
#include <memory.h>
#include <stdlib.h>
#include <string.h>

#include "../commons_private.h"
#include "reim4_fftvec_internal.h"
#include "reim4_fftvec_private.h"

EXPORT void reim4_from_cplx_ref(const REIM4_FROM_CPLX_PRECOMP* tables, double* r, const void* a) {
  const double* x = (double*)a;
  const uint64_t m = tables->m;
  for (uint64_t i = 0; i < m / 4; ++i) {
    double r0 = x[0];
    double i0 = x[1];
    double r1 = x[2];
    double i1 = x[3];
    double r2 = x[4];
    double i2 = x[5];
    double r3 = x[6];
    double i3 = x[7];
    r[0] = r0;
    r[1] = r2;
    r[2] = r1;
    r[3] = r3;
    r[4] = i0;
    r[5] = i2;
    r[6] = i1;
    r[7] = i3;
    x += 8;
    r += 8;
  }
}

void* init_reim4_from_cplx_precomp(REIM4_FROM_CPLX_PRECOMP* res, uint32_t nn) {
  res->m = nn / 2;
  if (CPU_SUPPORTS("fma")) {
    if (nn >= 4) {
      res->function = reim4_from_cplx_fma;
    } else {
      res->function = reim4_from_cplx_ref;
    }
  } else {
    res->function = reim4_from_cplx_ref;
  }
  return res;
}

EXPORT REIM4_FROM_CPLX_PRECOMP* new_reim4_from_cplx_precomp(uint32_t m) {
  REIM4_FROM_CPLX_PRECOMP* res = malloc(sizeof(*res));
  if (!res) return spqlios_error(strerror(errno));
  return spqlios_keep_or_free(res, init_reim4_from_cplx_precomp(res, m));
}

EXPORT void reim4_from_cplx_simple(uint32_t m, double* r, const void* a) {
  static REIM4_FROM_CPLX_PRECOMP precomp[32];
  REIM4_FROM_CPLX_PRECOMP* p = precomp + log2m(m);
  if (!p->function) {
    if (!init_reim4_from_cplx_precomp(p, m)) abort();
  }
  p->function(p, r, a);
}

EXPORT void reim4_to_cplx_ref(const REIM4_TO_CPLX_PRECOMP* tables, void* r, const double* a) {
  double* y = (double*)r;
  const uint64_t m = tables->m;
  for (uint64_t i = 0; i < m / 4; ++i) {
    double r0 = a[0];
    double r2 = a[1];
    double r1 = a[2];
    double r3 = a[3];
    double i0 = a[4];
    double i2 = a[5];
    double i1 = a[6];
    double i3 = a[7];
    y[0] = r0;
    y[1] = i0;
    y[2] = r1;
    y[3] = i1;
    y[4] = r2;
    y[5] = i2;
    y[6] = r3;
    y[7] = i3;
    a += 8;
    y += 8;
  }
}

void* init_reim4_to_cplx_precomp(REIM4_TO_CPLX_PRECOMP* res, uint32_t m) {
  res->m = m;
  if (CPU_SUPPORTS("fma")) {
    if (m >= 2) {
      res->function = reim4_to_cplx_fma;
    } else {
      res->function = reim4_to_cplx_ref;
    }
  } else {
    res->function = reim4_to_cplx_ref;
  }
  return res;
}

EXPORT REIM4_TO_CPLX_PRECOMP* new_reim4_to_cplx_precomp(uint32_t m) {
  REIM4_TO_CPLX_PRECOMP* res = malloc(sizeof(*res));
  if (!res) return spqlios_error(strerror(errno));
  return spqlios_keep_or_free(res, init_reim4_to_cplx_precomp(res, m));
}

EXPORT void reim4_to_cplx_simple(uint32_t m, void* r, const double* a) {
  static REIM4_TO_CPLX_PRECOMP precomp[32];
  REIM4_TO_CPLX_PRECOMP* p = precomp + log2m(m);
  if (!p->function) {
    if (!init_reim4_to_cplx_precomp(p, m)) abort();
  }
  p->function(p, r, a);
}
