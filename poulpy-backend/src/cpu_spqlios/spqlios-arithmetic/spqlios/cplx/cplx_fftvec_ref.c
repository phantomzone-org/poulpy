#include <string.h>

#include "../commons_private.h"
#include "cplx_fft_internal.h"
#include "cplx_fft_private.h"

EXPORT void cplx_fftvec_addmul_ref(const CPLX_FFTVEC_ADDMUL_PRECOMP* precomp, void* r, const void* a, const void* b) {
  const uint32_t m = precomp->m;
  const CPLX* aa = (CPLX*)a;
  const CPLX* bb = (CPLX*)b;
  CPLX* rr = (CPLX*)r;
  for (uint32_t i = 0; i < m; ++i) {
    const double re = aa[i][0] * bb[i][0] - aa[i][1] * bb[i][1];
    const double im = aa[i][0] * bb[i][1] + aa[i][1] * bb[i][0];
    rr[i][0] += re;
    rr[i][1] += im;
  }
}

EXPORT void cplx_fftvec_mul_ref(const CPLX_FFTVEC_MUL_PRECOMP* precomp, void* r, const void* a, const void* b) {
  const uint32_t m = precomp->m;
  const CPLX* aa = (CPLX*)a;
  const CPLX* bb = (CPLX*)b;
  CPLX* rr = (CPLX*)r;
  for (uint32_t i = 0; i < m; ++i) {
    const double re = aa[i][0] * bb[i][0] - aa[i][1] * bb[i][1];
    const double im = aa[i][0] * bb[i][1] + aa[i][1] * bb[i][0];
    rr[i][0] = re;
    rr[i][1] = im;
  }
}

EXPORT void* init_cplx_fftvec_addmul_precomp(CPLX_FFTVEC_ADDMUL_PRECOMP* r, uint32_t m) {
  if (m & (m - 1)) return spqlios_error("m must be a power of two");
  r->m = m;
  if (m <= 4) {
    r->function = cplx_fftvec_addmul_ref;
  } else if (CPU_SUPPORTS("fma")) {
    r->function = cplx_fftvec_addmul_fma;
  } else {
    r->function = cplx_fftvec_addmul_ref;
  }
  return r;
}

EXPORT void* init_cplx_fftvec_mul_precomp(CPLX_FFTVEC_MUL_PRECOMP* r, uint32_t m) {
  if (m & (m - 1)) return spqlios_error("m must be a power of two");
  r->m = m;
  if (m <= 4) {
    r->function = cplx_fftvec_mul_ref;
  } else if (CPU_SUPPORTS("fma")) {
    r->function = cplx_fftvec_mul_fma;
  } else {
    r->function = cplx_fftvec_mul_ref;
  }
  return r;
}

EXPORT CPLX_FFTVEC_ADDMUL_PRECOMP* new_cplx_fftvec_addmul_precomp(uint32_t m) {
  CPLX_FFTVEC_ADDMUL_PRECOMP* r = malloc(sizeof(CPLX_FFTVEC_MUL_PRECOMP));
  return spqlios_keep_or_free(r, init_cplx_fftvec_addmul_precomp(r, m));
}

EXPORT CPLX_FFTVEC_MUL_PRECOMP* new_cplx_fftvec_mul_precomp(uint32_t m) {
  CPLX_FFTVEC_MUL_PRECOMP* r = malloc(sizeof(CPLX_FFTVEC_MUL_PRECOMP));
  return spqlios_keep_or_free(r, init_cplx_fftvec_mul_precomp(r, m));
}

EXPORT void cplx_fftvec_mul_simple(uint32_t m, void* r, const void* a, const void* b) {
  static CPLX_FFTVEC_MUL_PRECOMP p[31] = {0};
  CPLX_FFTVEC_MUL_PRECOMP* f = p + log2m(m);
  if (!f->function) {
    if (!init_cplx_fftvec_mul_precomp(f, m)) abort();
  }
  f->function(f, r, a, b);
}

EXPORT void cplx_fftvec_addmul_simple(uint32_t m, void* r, const void* a, const void* b) {
  static CPLX_FFTVEC_ADDMUL_PRECOMP p[31] = {0};
  CPLX_FFTVEC_ADDMUL_PRECOMP* f = p + log2m(m);
  if (!f->function) {
    if (!init_cplx_fftvec_addmul_precomp(f, m)) abort();
  }
  f->function(f, r, a, b);
}
