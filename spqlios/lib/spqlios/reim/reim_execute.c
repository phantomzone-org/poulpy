#include "reim_fft_internal.h"
#include "reim_fft_private.h"

EXPORT void reim_from_znx32(const REIM_FROM_ZNX32_PRECOMP* tables, void* r, const int32_t* a) {
  tables->function(tables, r, a);
}

EXPORT void reim_from_tnx32(const REIM_FROM_TNX32_PRECOMP* tables, void* r, const int32_t* a) {
  tables->function(tables, r, a);
}

EXPORT void reim_to_tnx32(const REIM_TO_TNX32_PRECOMP* tables, int32_t* r, const void* a) {
  tables->function(tables, r, a);
}

EXPORT void reim_fftvec_mul(const REIM_FFTVEC_MUL_PRECOMP* tables, double* r, const double* a, const double* b) {
  tables->function(tables, r, a, b);
}

EXPORT void reim_fftvec_addmul(const REIM_FFTVEC_ADDMUL_PRECOMP* tables, double* r, const double* a, const double* b) {
  tables->function(tables, r, a, b);
}
