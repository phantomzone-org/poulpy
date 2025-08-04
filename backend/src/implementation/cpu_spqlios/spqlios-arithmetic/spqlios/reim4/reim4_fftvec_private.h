#ifndef SPQLIOS_REIM4_FFTVEC_PRIVATE_H
#define SPQLIOS_REIM4_FFTVEC_PRIVATE_H

#include "reim4_fftvec_public.h"

#define STATIC_ASSERT(condition) (void)sizeof(char[-1 + 2 * !!(condition)])

typedef void (*R4_FFTVEC_MUL_FUNC)(const REIM4_FFTVEC_MUL_PRECOMP*, double*, const double*, const double*);
typedef void (*R4_FFTVEC_ADDMUL_FUNC)(const REIM4_FFTVEC_ADDMUL_PRECOMP*, double*, const double*, const double*);
typedef void (*R4_FROM_CPLX_FUNC)(const REIM4_FROM_CPLX_PRECOMP*, double*, const void*);
typedef void (*R4_TO_CPLX_FUNC)(const REIM4_TO_CPLX_PRECOMP*, void*, const double*);

struct reim4_mul_precomp {
  R4_FFTVEC_MUL_FUNC function;
  int64_t m;
};

struct reim4_addmul_precomp {
  R4_FFTVEC_ADDMUL_FUNC function;
  int64_t m;
};

struct reim4_from_cplx_precomp {
  R4_FROM_CPLX_FUNC function;
  int64_t m;
};

struct reim4_to_cplx_precomp {
  R4_TO_CPLX_FUNC function;
  int64_t m;
};

#endif  // SPQLIOS_REIM4_FFTVEC_PRIVATE_H
