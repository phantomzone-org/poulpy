#ifndef SPQLIOS_REIM4_FFTVEC_PUBLIC_H
#define SPQLIOS_REIM4_FFTVEC_PUBLIC_H

#include "../commons.h"

typedef struct reim4_addmul_precomp REIM4_FFTVEC_ADDMUL_PRECOMP;
typedef struct reim4_mul_precomp REIM4_FFTVEC_MUL_PRECOMP;
typedef struct reim4_from_cplx_precomp REIM4_FROM_CPLX_PRECOMP;
typedef struct reim4_to_cplx_precomp REIM4_TO_CPLX_PRECOMP;

EXPORT REIM4_FFTVEC_MUL_PRECOMP* new_reim4_fftvec_mul_precomp(uint32_t m);
EXPORT void reim4_fftvec_mul(const REIM4_FFTVEC_MUL_PRECOMP* tables, double* r, const double* a, const double* b);
#define delete_reim4_fftvec_mul_precomp free

EXPORT REIM4_FFTVEC_ADDMUL_PRECOMP* new_reim4_fftvec_addmul_precomp(uint32_t nn);
EXPORT void reim4_fftvec_addmul(const REIM4_FFTVEC_ADDMUL_PRECOMP* tables, double* r, const double* a, const double* b);
#define delete_reim4_fftvec_addmul_precomp free

/**
 * @brief prepares a conversion from the cplx fftvec layout to the reim4 layout.
 * @param m complex dimension m from C[X] mod X^m-i.
 */
EXPORT REIM4_FROM_CPLX_PRECOMP* new_reim4_from_cplx_precomp(uint32_t m);
EXPORT void reim4_from_cplx(const REIM4_FROM_CPLX_PRECOMP* tables, double* r, const void* a);
#define delete_reim4_from_cplx_precomp free

/**
 * @brief prepares a conversion from the reim4 fftvec layout to the cplx layout
 * @param m the complex dimension m from C[X] mod X^m-i.
 */
EXPORT REIM4_TO_CPLX_PRECOMP* new_reim4_to_cplx_precomp(uint32_t m);
EXPORT void reim4_to_cplx(const REIM4_TO_CPLX_PRECOMP* tables, void* r, const double* a);
#define delete_reim4_to_cplx_precomp free

/**
 * @brief Simpler API for the fftvec multiplication function.
 * For each dimension, the precomputed tables for this dimension are generated automatically the first time.
 * It is advised to do one dry-run call per desired dimension before using in a multithread environment */
EXPORT void reim4_fftvec_mul_simple(uint32_t m, double* r, const double* a, const double* b);

/**
 * @brief Simpler API for the fftvec addmul function.
 * For each dimension, the precomputed tables for this dimension are generated automatically the first time.
 * It is advised to do one dry-run call per desired dimension before using in a multithread environment */
EXPORT void reim4_fftvec_addmul_simple(uint32_t m, double* r, const double* a, const double* b);

/**
 * @brief Simpler API for cplx conversion.
 * For each dimension, the precomputed tables for this dimension are generated automatically the first time.
 * It is advised to do one dry-run call per desired dimension before using in a multithread environment */
EXPORT void reim4_from_cplx_simple(uint32_t m, double* r, const void* a);

/**
 * @brief Simpler API for to cplx conversion.
 * For each dimension, the precomputed tables for this dimension are generated automatically the first time.
 * It is advised to do one dry-run call per desired dimension before using in a multithread environment */
EXPORT void reim4_to_cplx_simple(uint32_t m, void* r, const double* a);

#endif  // SPQLIOS_REIM4_FFTVEC_PUBLIC_H