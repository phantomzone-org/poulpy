#include "cplx_fft_internal.h"
#include "cplx_fft_private.h"

EXPORT void cplx_from_znx32(const CPLX_FROM_ZNX32_PRECOMP* tables, void* r, const int32_t* a) {
  tables->function(tables, r, a);
}
EXPORT void cplx_from_tnx32(const CPLX_FROM_TNX32_PRECOMP* tables, void* r, const int32_t* a) {
  tables->function(tables, r, a);
}
EXPORT void cplx_to_tnx32(const CPLX_TO_TNX32_PRECOMP* tables, int32_t* r, const void* a) {
  tables->function(tables, r, a);
}
EXPORT void cplx_fftvec_mul(const CPLX_FFTVEC_MUL_PRECOMP* tables, void* r, const void* a, const void* b) {
  tables->function(tables, r, a, b);
}
EXPORT void cplx_fftvec_addmul(const CPLX_FFTVEC_ADDMUL_PRECOMP* tables, void* r, const void* a, const void* b) {
  tables->function(tables, r, a, b);
}
