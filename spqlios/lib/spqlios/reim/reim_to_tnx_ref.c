#include <errno.h>
#include <string.h>

#include "../commons_private.h"
#include "reim_fft_internal.h"
#include "reim_fft_private.h"

EXPORT void reim_to_tnx_basic_ref(const REIM_TO_TNX_PRECOMP* tables, double* r, const double* x) {
  const uint64_t n = tables->m << 1;
  double divisor = tables->divisor;
  for (uint64_t i=0; i<n; ++i) {
    double ri = x[i]/divisor;
    r[i] = ri - rint(ri);
  }
}

typedef union {double d; uint64_t u;} dblui64_t;

EXPORT void reim_to_tnx_ref(const REIM_TO_TNX_PRECOMP* tables, double* r, const double* x) {
  const uint64_t n = tables->m << 1;
  double add_cst = tables->add_cst;
  uint64_t mask_and = tables->mask_and;
  uint64_t mask_or = tables->mask_or;
  double sub_cst = tables->sub_cst;
  dblui64_t cur;
  for (uint64_t i=0; i<n; ++i) {
    cur.d = x[i] + add_cst;
    cur.u &= mask_and;
    cur.u |= mask_or;
    r[i] = cur.d - sub_cst;
  }
}

void* init_reim_to_tnx_precomp(REIM_TO_TNX_PRECOMP* const res, uint32_t m, double divisor, uint32_t log2overhead) {
  if (m & (m - 1)) return spqlios_error("m must be a power of 2");
  if (is_not_pow2_double(&divisor)) return spqlios_error("divisor must be a power of 2");
  if (log2overhead > 52) return spqlios_error("log2overhead is too large");
  res->m = m;
  res->divisor = divisor;
  res->log2overhead = log2overhead;
  //                 52 + 11 + 1
  // ......1.......01(1)|expo|sign
  // .......=========(1)|expo|sign  msbbits = log2ovh + 2 + 11 + 1
  uint64_t nbits = 50 - log2overhead;
  dblui64_t ovh_cst;
  ovh_cst.d = 0.5 + (6<<log2overhead);
  res->add_cst = ovh_cst.d * divisor;
  res->mask_and = ((UINT64_C(1) << nbits) - 1);
  res->mask_or = ovh_cst.u & ((UINT64_C(-1)) << nbits);
  res->sub_cst = ovh_cst.d;
  // TODO: check selection logic
  if (CPU_SUPPORTS("avx2")) {
      if (m >= 8) {
        res->function = reim_to_tnx_avx;
      } else {
        res->function = reim_to_tnx_ref;
      }
  } else {
    res->function = reim_to_tnx_ref;
  }
  return res;
}

EXPORT REIM_TO_TNX_PRECOMP* new_reim_to_tnx_precomp(uint32_t m, double divisor, uint32_t log2overhead) {
  REIM_TO_TNX_PRECOMP* res = malloc(sizeof(*res));
  if (!res) return spqlios_error(strerror(errno));
  return spqlios_keep_or_free(res, init_reim_to_tnx_precomp(res, m, divisor, log2overhead));
}

EXPORT void reim_to_tnx(const REIM_TO_TNX_PRECOMP* tables, double* r, const double* x) {
  tables->function(tables, r, x);
}
