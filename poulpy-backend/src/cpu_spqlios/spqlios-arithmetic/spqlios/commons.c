#include "commons.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

EXPORT void* UNDEFINED_p_ii(int32_t n, int32_t m) { UNDEFINED(); }
EXPORT void* UNDEFINED_p_uu(uint32_t n, uint32_t m) { UNDEFINED(); }
EXPORT double* UNDEFINED_dp_pi(const void* p, int32_t n) { UNDEFINED(); }
EXPORT void* UNDEFINED_vp_pi(const void* p, int32_t n) { UNDEFINED(); }
EXPORT void* UNDEFINED_vp_pu(const void* p, uint32_t n) { UNDEFINED(); }
EXPORT void UNDEFINED_v_vpdp(const void* p, double* a) { UNDEFINED(); }
EXPORT void UNDEFINED_v_vpvp(const void* p, void* a) { UNDEFINED(); }
EXPORT double* NOT_IMPLEMENTED_dp_i(int32_t n) { NOT_IMPLEMENTED(); }
EXPORT void* NOT_IMPLEMENTED_vp_i(int32_t n) { NOT_IMPLEMENTED(); }
EXPORT void* NOT_IMPLEMENTED_vp_u(uint32_t n) { NOT_IMPLEMENTED(); }
EXPORT void NOT_IMPLEMENTED_v_dp(double* a) { NOT_IMPLEMENTED(); }
EXPORT void NOT_IMPLEMENTED_v_vp(void* p) { NOT_IMPLEMENTED(); }
EXPORT void NOT_IMPLEMENTED_v_idpdpdp(int32_t n, double* a, const double* b, const double* c) { NOT_IMPLEMENTED(); }
EXPORT void NOT_IMPLEMENTED_v_uvpcvpcvp(uint32_t n, void* r, const void* a, const void* b) { NOT_IMPLEMENTED(); }
EXPORT void NOT_IMPLEMENTED_v_uvpvpcvp(uint32_t n, void* a, void* b, const void* o) { NOT_IMPLEMENTED(); }

#ifdef _WIN32
#define __always_inline inline __attribute((always_inline))
#endif

void internal_accurate_sincos(double* rcos, double* rsin, double x) {
  double _4_x_over_pi = 4 * x / M_PI;
  int64_t int_part = ((int64_t)rint(_4_x_over_pi)) & 7;
  double frac_part = _4_x_over_pi - (double)(int_part);
  double frac_x = M_PI * frac_part / 4.;
  // compute the taylor series
  double cosp = 1.;
  double sinp = 0.;
  double powx = 1.;
  int64_t nn = 0;
  while (fabs(powx) > 1e-20) {
    ++nn;
    powx = powx * frac_x / (double)(nn);  // x^n/n!
    switch (nn & 3) {
      case 0:
        cosp += powx;
        break;
      case 1:
        sinp += powx;
        break;
      case 2:
        cosp -= powx;
        break;
      case 3:
        sinp -= powx;
        break;
      default:
        abort();  // impossible
    }
  }
  // final multiplication
  switch (int_part) {
    case 0:
      *rcos = cosp;
      *rsin = sinp;
      break;
    case 1:
      *rcos = M_SQRT1_2 * (cosp - sinp);
      *rsin = M_SQRT1_2 * (cosp + sinp);
      break;
    case 2:
      *rcos = -sinp;
      *rsin = cosp;
      break;
    case 3:
      *rcos = -M_SQRT1_2 * (cosp + sinp);
      *rsin = M_SQRT1_2 * (cosp - sinp);
      break;
    case 4:
      *rcos = -cosp;
      *rsin = -sinp;
      break;
    case 5:
      *rcos = -M_SQRT1_2 * (cosp - sinp);
      *rsin = -M_SQRT1_2 * (cosp + sinp);
      break;
    case 6:
      *rcos = sinp;
      *rsin = -cosp;
      break;
    case 7:
      *rcos = M_SQRT1_2 * (cosp + sinp);
      *rsin = -M_SQRT1_2 * (cosp - sinp);
      break;
    default:
      abort();  // impossible
  }
  if (fabs(cos(x) - *rcos) > 1e-10 || fabs(sin(x) - *rsin) > 1e-10) {
    printf("cos(%.17lf) =? %.17lf instead of %.17lf\n", x, *rcos, cos(x));
    printf("sin(%.17lf) =? %.17lf instead of %.17lf\n", x, *rsin, sin(x));
    printf("fracx = %.17lf\n", frac_x);
    printf("cosp = %.17lf\n", cosp);
    printf("sinp = %.17lf\n", sinp);
    printf("nn = %d\n", (int)(nn));
  }
}

double internal_accurate_cos(double x) {
  double rcos, rsin;
  internal_accurate_sincos(&rcos, &rsin, x);
  return rcos;
}
double internal_accurate_sin(double x) {
  double rcos, rsin;
  internal_accurate_sincos(&rcos, &rsin, x);
  return rsin;
}

EXPORT void spqlios_debug_free(void* addr) { free((uint8_t*)addr - 64); }

EXPORT void* spqlios_debug_alloc(uint64_t size) { return (uint8_t*)malloc(size + 64) + 64; }

EXPORT void spqlios_free(void* addr) {
#ifndef NDEBUG
  // in debug mode, we deallocated with spqlios_debug_free()
  spqlios_debug_free(addr);
#else
  // in release mode, the function will free aligned memory
#ifdef _WIN32
  _aligned_free(addr);
#else
  free(addr);
#endif
#endif
}

EXPORT void* spqlios_alloc(uint64_t size) {
#ifndef NDEBUG
  // in debug mode, the function will not necessarily have any particular alignment
  // it will also ensure that memory can only be deallocated with spqlios_free()
  return spqlios_debug_alloc(size);
#else
  // in release mode, the function will return 64-bytes aligned memory
#ifdef _WIN32
  void* reps = _aligned_malloc((size + 63) & (UINT64_C(-64)), 64);
#else
  void* reps = aligned_alloc(64, (size + 63) & (UINT64_C(-64)));
#endif
  if (reps == 0) FATAL_ERROR("Out of memory");
  return reps;
#endif
}

EXPORT void* spqlios_alloc_custom_align(uint64_t align, uint64_t size) {
#ifndef NDEBUG
  // in debug mode, the function will not necessarily have any particular alignment
  // it will also ensure that memory can only be deallocated with spqlios_free()
  return spqlios_debug_alloc(size);
#else
  // in release mode, the function will return aligned memory
#ifdef _WIN32
  void* reps = _aligned_malloc(size, align);
#else
  void* reps = aligned_alloc(align, size);
#endif
  if (reps == 0) FATAL_ERROR("Out of memory");
  return reps;
#endif
}