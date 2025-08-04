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

EXPORT void reim_fftvec_add(const REIM_FFTVEC_ADD_PRECOMP* tables, double* r, const double* a, const double* b) {
  tables->function(tables, r, a, b);
}

EXPORT void reim_fftvec_sub(const REIM_FFTVEC_SUB_PRECOMP* tables, double* r, const double* a, const double* b) {
  tables->function(tables, r, a, b);
}

EXPORT void reim_fftvec_mul(const REIM_FFTVEC_MUL_PRECOMP* tables, double* r, const double* a, const double* b) {
  tables->function(tables, r, a, b);
}

EXPORT void reim_fftvec_addmul(const REIM_FFTVEC_ADDMUL_PRECOMP* tables, double* r, const double* a, const double* b) {
  tables->function(tables, r, a, b);
}

EXPORT void reim_fftvec_automorphism(const REIM_FFTVEC_AUTOMORPHISM_PRECOMP* tables, int64_t p, double* r,
                                     const double* a, uint64_t a_size) {
  tables->function.apply(tables, p, r, a, a_size);
}

EXPORT void reim_fftvec_automorphism_inplace(const REIM_FFTVEC_AUTOMORPHISM_PRECOMP* tables, int64_t p, double* a,
                                             uint64_t a_size, uint8_t* tmp_bytes) {
  tables->function.apply_inplace(tables, p, a, a_size, tmp_bytes);
}

EXPORT uint64_t reim_fftvec_automorphism_inplace_tmp_bytes(const REIM_FFTVEC_AUTOMORPHISM_PRECOMP* tables) {
  return tables->function.apply_inplace_tmp_bytes(tables);
}
