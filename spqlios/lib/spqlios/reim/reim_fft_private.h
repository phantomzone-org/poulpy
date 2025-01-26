#ifndef SPQLIOS_REIM_FFT_PRIVATE_H
#define SPQLIOS_REIM_FFT_PRIVATE_H

#include "../commons_private.h"
#include "reim_fft.h"

#define STATIC_ASSERT(condition) (void)sizeof(char[-1 + 2 * !!(condition)])

typedef struct reim_twiddle_precomp REIM_FFTVEC_TWIDDLE_PRECOMP;
typedef struct reim_bitwiddle_precomp REIM_FFTVEC_BITWIDDLE_PRECOMP;

typedef void (*FFT_FUNC)(const REIM_FFT_PRECOMP*, double*);
typedef void (*IFFT_FUNC)(const REIM_IFFT_PRECOMP*, double*);
typedef void (*FFTVEC_MUL_FUNC)(const REIM_FFTVEC_MUL_PRECOMP*, double*, const double*, const double*);
typedef void (*FFTVEC_ADDMUL_FUNC)(const REIM_FFTVEC_ADDMUL_PRECOMP*, double*, const double*, const double*);

typedef void (*FROM_ZNX32_FUNC)(const REIM_FROM_ZNX32_PRECOMP*, void*, const int32_t*);
typedef void (*FROM_ZNX64_FUNC)(const REIM_FROM_ZNX64_PRECOMP*, void*, const int64_t*);
typedef void (*FROM_TNX32_FUNC)(const REIM_FROM_TNX32_PRECOMP*, void*, const int32_t*);
typedef void (*TO_TNX32_FUNC)(const REIM_TO_TNX32_PRECOMP*, int32_t*, const void*);
typedef void (*TO_TNX_FUNC)(const REIM_TO_TNX_PRECOMP*, double*, const double*);
typedef void (*TO_ZNX64_FUNC)(const REIM_TO_ZNX64_PRECOMP*, int64_t*, const void*);
typedef void (*FFTVEC_TWIDDLE_FUNC)(const REIM_FFTVEC_TWIDDLE_PRECOMP*, void*, const void*, const void*);
typedef void (*FFTVEC_BITWIDDLE_FUNC)(const REIM_FFTVEC_BITWIDDLE_PRECOMP*, void*, uint64_t, const void*);

typedef struct reim_fft_precomp {
  FFT_FUNC function;
  int64_t m; ///< complex dimension warning: reim uses n=2N=4m
  uint64_t buf_size; ///< size of aligned_buffers (multiple of 64B)
  double* powomegas;  ///< 64B aligned
  void* aligned_buffers; ///< 64B aligned
} REIM_FFT_PRECOMP;

typedef struct reim_ifft_precomp {
  IFFT_FUNC function;
  int64_t m;  // warning: reim uses n=2N=4m
  uint64_t buf_size; ///< size of aligned_buffers (multiple of 64B)
  double* powomegas;
  void* aligned_buffers;
} REIM_IFFT_PRECOMP;

typedef struct reim_mul_precomp {
  FFTVEC_MUL_FUNC function;
  int64_t m;
} REIM_FFTVEC_MUL_PRECOMP;

typedef struct reim_addmul_precomp {
  FFTVEC_ADDMUL_FUNC function;
  int64_t m;
} REIM_FFTVEC_ADDMUL_PRECOMP;


struct reim_from_znx32_precomp {
  FROM_ZNX32_FUNC function;
  int64_t m;
};

struct reim_from_znx64_precomp {
  FROM_ZNX64_FUNC function;
  int64_t m;
};

struct reim_from_tnx32_precomp {
  FROM_TNX32_FUNC function;
  int64_t m;
};

struct reim_to_tnx32_precomp {
  TO_TNX32_FUNC function;
  int64_t m;
  double divisor;
};

struct reim_to_tnx_precomp {
  TO_TNX_FUNC function;
  int64_t m;
  double divisor;
  uint32_t log2overhead;
  double add_cst;
  uint64_t mask_and;
  uint64_t mask_or;
  double sub_cst;
};

struct reim_to_znx64_precomp {
  TO_ZNX64_FUNC function;
  int64_t m;
  double divisor;
};

struct reim_twiddle_precomp {
  FFTVEC_TWIDDLE_FUNC function;
  int64_t m;
};

struct reim_bitwiddle_precomp {
  FFTVEC_BITWIDDLE_FUNC function;
  int64_t m;
};

#endif  // SPQLIOS_REIM_FFT_PRIVATE_H
