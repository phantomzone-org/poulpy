#include "reim4_fftvec_internal.h"
#include "reim4_fftvec_private.h"

EXPORT void reim4_fftvec_addmul(const REIM4_FFTVEC_ADDMUL_PRECOMP* tables, double* r, const double* a,
                                const double* b) {
  tables->function(tables, r, a, b);
}

EXPORT void reim4_fftvec_mul(const REIM4_FFTVEC_MUL_PRECOMP* tables, double* r, const double* a, const double* b) {
  tables->function(tables, r, a, b);
}

EXPORT void reim4_from_cplx(const REIM4_FROM_CPLX_PRECOMP* tables, double* r, const void* a) {
  tables->function(tables, r, a);
}

EXPORT void reim4_to_cplx(const REIM4_TO_CPLX_PRECOMP* tables, void* r, const double* a) {
  tables->function(tables, r, a);
}
