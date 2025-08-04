#ifndef SPQLIOS_REIM4_FFTVEC_INTERNAL_H
#define SPQLIOS_REIM4_FFTVEC_INTERNAL_H

#include "reim4_fftvec_public.h"

EXPORT void reim4_fftvec_mul_ref(const REIM4_FFTVEC_MUL_PRECOMP* tables, double* r, const double* a, const double* b);
EXPORT void reim4_fftvec_mul_fma(const REIM4_FFTVEC_MUL_PRECOMP* tables, double* r, const double* a, const double* b);

EXPORT void reim4_fftvec_addmul_ref(const REIM4_FFTVEC_ADDMUL_PRECOMP* tables, double* r, const double* a,
                                    const double* b);
EXPORT void reim4_fftvec_addmul_fma(const REIM4_FFTVEC_ADDMUL_PRECOMP* tables, double* r, const double* a,
                                    const double* b);

EXPORT void reim4_from_cplx_ref(const REIM4_FROM_CPLX_PRECOMP* tables, double* r, const void* a);
EXPORT void reim4_from_cplx_fma(const REIM4_FROM_CPLX_PRECOMP* tables, double* r, const void* a);

EXPORT void reim4_to_cplx_ref(const REIM4_TO_CPLX_PRECOMP* tables, void* r, const double* a);
EXPORT void reim4_to_cplx_fma(const REIM4_TO_CPLX_PRECOMP* tables, void* r, const double* a);

#endif  // SPQLIOS_REIM4_FFTVEC_INTERNAL_H
