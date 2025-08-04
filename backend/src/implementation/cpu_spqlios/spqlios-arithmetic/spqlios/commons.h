#ifndef SPQLIOS_COMMONS_H
#define SPQLIOS_COMMONS_H

#ifdef __cplusplus
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#define EXPORT extern "C"
#define EXPORT_DECL extern "C"
#else
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define EXPORT
#define EXPORT_DECL extern
#define nullptr 0x0;
#endif

#define UNDEFINED()                    \
  {                                    \
    fprintf(stderr, "UNDEFINED!!!\n"); \
    abort();                           \
  }
#define NOT_IMPLEMENTED()                    \
  {                                          \
    fprintf(stderr, "NOT IMPLEMENTED!!!\n"); \
    abort();                                 \
  }
#define FATAL_ERROR(MESSAGE)                   \
  {                                            \
    fprintf(stderr, "ERROR: %s\n", (MESSAGE)); \
    abort();                                   \
  }

EXPORT void* UNDEFINED_p_ii(int32_t n, int32_t m);
EXPORT void* UNDEFINED_p_uu(uint32_t n, uint32_t m);
EXPORT double* UNDEFINED_dp_pi(const void* p, int32_t n);
EXPORT void* UNDEFINED_vp_pi(const void* p, int32_t n);
EXPORT void* UNDEFINED_vp_pu(const void* p, uint32_t n);
EXPORT void UNDEFINED_v_vpdp(const void* p, double* a);
EXPORT void UNDEFINED_v_vpvp(const void* p, void* a);
EXPORT double* NOT_IMPLEMENTED_dp_i(int32_t n);
EXPORT void* NOT_IMPLEMENTED_vp_i(int32_t n);
EXPORT void* NOT_IMPLEMENTED_vp_u(uint32_t n);
EXPORT void NOT_IMPLEMENTED_v_dp(double* a);
EXPORT void NOT_IMPLEMENTED_v_vp(void* p);
EXPORT void NOT_IMPLEMENTED_v_idpdpdp(int32_t n, double* a, const double* b, const double* c);
EXPORT void NOT_IMPLEMENTED_v_uvpcvpcvp(uint32_t n, void* r, const void* a, const void* b);
EXPORT void NOT_IMPLEMENTED_v_uvpvpcvp(uint32_t n, void* a, void* b, const void* o);

// windows

#if defined(_WIN32) || defined(__APPLE__)
#define __always_inline inline __attribute((always_inline))
#endif

EXPORT void spqlios_free(void* address);

EXPORT void* spqlios_alloc(uint64_t size);
EXPORT void* spqlios_alloc_custom_align(uint64_t align, uint64_t size);

#define USE_LIBM_SIN_COS
#ifndef USE_LIBM_SIN_COS
// if at some point, we want to remove the libm dependency, we can
// consider this:
EXPORT double internal_accurate_cos(double x);
EXPORT double internal_accurate_sin(double x);
EXPORT void internal_accurate_sincos(double* rcos, double* rsin, double x);
#define m_accurate_cos internal_accurate_cos
#define m_accurate_sin internal_accurate_sin
#else
// let's use libm sin and cos
#define m_accurate_cos cos
#define m_accurate_sin sin
#endif

#endif  // SPQLIOS_COMMONS_H
