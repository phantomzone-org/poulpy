#ifndef SPQLIOS_CPLX_FFT_PRIVATE_H
#define SPQLIOS_CPLX_FFT_PRIVATE_H

#include "cplx_fft.h"

typedef struct cplx_twiddle_precomp CPLX_FFTVEC_TWIDDLE_PRECOMP;
typedef struct cplx_bitwiddle_precomp CPLX_FFTVEC_BITWIDDLE_PRECOMP;

typedef void (*IFFT_FUNCTION)(const CPLX_IFFT_PRECOMP*, void*);
typedef void (*FFT_FUNCTION)(const CPLX_FFT_PRECOMP*, void*);
// conversions
typedef void (*FROM_ZNX32_FUNCTION)(const CPLX_FROM_ZNX32_PRECOMP*, void*, const int32_t*);
typedef void (*TO_ZNX32_FUNCTION)(const CPLX_FROM_ZNX32_PRECOMP*, int32_t*, const void*);
typedef void (*FROM_TNX32_FUNCTION)(const CPLX_FROM_TNX32_PRECOMP*, void*, const int32_t*);
typedef void (*TO_TNX32_FUNCTION)(const CPLX_TO_TNX32_PRECOMP*, int32_t*, const void*);
typedef void (*FROM_RNX64_FUNCTION)(const CPLX_FROM_RNX64_PRECOMP* precomp, void* r, const double* x);
typedef void (*TO_RNX64_FUNCTION)(const CPLX_TO_RNX64_PRECOMP* precomp, double* r, const void* x);
typedef void (*ROUND_TO_RNX64_FUNCTION)(const CPLX_ROUND_TO_RNX64_PRECOMP* precomp, double* r, const void* x);
// fftvec operations
typedef void (*FFTVEC_MUL_FUNCTION)(const CPLX_FFTVEC_MUL_PRECOMP*, void*, const void*, const void*);
typedef void (*FFTVEC_ADDMUL_FUNCTION)(const CPLX_FFTVEC_ADDMUL_PRECOMP*, void*, const void*, const void*);

typedef void (*FFTVEC_TWIDDLE_FUNCTION)(const CPLX_FFTVEC_TWIDDLE_PRECOMP*, void*, const void*, const void*);
typedef void (*FFTVEC_BITWIDDLE_FUNCTION)(const CPLX_FFTVEC_BITWIDDLE_PRECOMP*, void*, uint64_t, const void*);

struct cplx_ifft_precomp {
  IFFT_FUNCTION function;
  int64_t m;
  uint64_t buf_size;
  double* powomegas;
  void* aligned_buffers;
};

struct cplx_fft_precomp {
  FFT_FUNCTION function;
  int64_t m;
  uint64_t buf_size;
  double* powomegas;
  void* aligned_buffers;
};

struct cplx_from_znx32_precomp {
  FROM_ZNX32_FUNCTION function;
  int64_t m;
};

struct cplx_to_znx32_precomp {
  TO_ZNX32_FUNCTION function;
  int64_t m;
  double divisor;
};

struct cplx_from_tnx32_precomp {
  FROM_TNX32_FUNCTION function;
  int64_t m;
};

struct cplx_to_tnx32_precomp {
  TO_TNX32_FUNCTION function;
  int64_t m;
  double divisor;
};

struct cplx_from_rnx64_precomp {
  FROM_RNX64_FUNCTION function;
  int64_t m;
};

struct cplx_to_rnx64_precomp {
  TO_RNX64_FUNCTION function;
  int64_t m;
  double divisor;
};

struct cplx_round_to_rnx64_precomp {
  ROUND_TO_RNX64_FUNCTION function;
  int64_t m;
  double divisor;
  uint32_t log2bound;
};

typedef struct cplx_mul_precomp {
  FFTVEC_MUL_FUNCTION function;
  int64_t m;
} CPLX_FFTVEC_MUL_PRECOMP;

typedef struct cplx_addmul_precomp {
  FFTVEC_ADDMUL_FUNCTION function;
  int64_t m;
} CPLX_FFTVEC_ADDMUL_PRECOMP;

struct cplx_twiddle_precomp {
  FFTVEC_TWIDDLE_FUNCTION function;
  int64_t m;
};

struct cplx_bitwiddle_precomp {
  FFTVEC_BITWIDDLE_FUNCTION function;
  int64_t m;
};

EXPORT void cplx_fftvec_twiddle_fma(const CPLX_FFTVEC_TWIDDLE_PRECOMP* tables, void* a, void* b, const void* om);
EXPORT void cplx_fftvec_twiddle_avx512(const CPLX_FFTVEC_TWIDDLE_PRECOMP* tables, void* a, void* b, const void* om);
EXPORT void cplx_fftvec_bitwiddle_fma(const CPLX_FFTVEC_BITWIDDLE_PRECOMP* tables, void* a, uint64_t slice,
                                      const void* om);
EXPORT void cplx_fftvec_bitwiddle_avx512(const CPLX_FFTVEC_BITWIDDLE_PRECOMP* tables, void* a, uint64_t slice,
                                         const void* om);

#endif  // SPQLIOS_CPLX_FFT_PRIVATE_H
