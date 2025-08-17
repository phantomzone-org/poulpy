#include <memory.h>

#include "vec_rnx_arithmetic_private.h"

typedef union di {
  double dv;
  uint64_t uv;
} di_t;

/** @brief new gadget: delete with delete_tnxdbl_approxdecomp_gadget */
EXPORT TNXDBL_APPROXDECOMP_GADGET* new_tnxdbl_approxdecomp_gadget(  //
    const MOD_RNX* module,                                          // N
    uint64_t k, uint64_t ell                                        // base 2^K and size
) {
  if (k * ell > 50) return spqlios_error("gadget requires a too large fp precision");
  TNXDBL_APPROXDECOMP_GADGET* res = spqlios_alloc(sizeof(TNXDBL_APPROXDECOMP_GADGET));
  res->k = k;
  res->ell = ell;
  // double add_cst;       // double(3.2^(51-ell.K) + 1/2.(sum 2^(-iK)) for i=[0,ell[)
  union di add_cst;
  add_cst.dv = UINT64_C(3) << (51 - ell * k);
  for (uint64_t i = 0; i < ell; ++i) {
    add_cst.uv |= UINT64_C(1) << ((i + 1) * k - 1);
  }
  res->add_cst = add_cst.dv;
  // uint64_t and_mask;     // uint64(2^(K)-1)
  res->and_mask = (UINT64_C(1) << k) - 1;
  // uint64_t or_mask;      // double(2^52)
  union di or_mask;
  or_mask.dv = (UINT64_C(1) << 52);
  res->or_mask = or_mask.uv;
  // double sub_cst;       // double(2^52 + 2^(K-1))
  res->sub_cst = ((UINT64_C(1) << 52) + (UINT64_C(1) << (k - 1)));
  return res;
}

EXPORT void delete_tnxdbl_approxdecomp_gadget(TNXDBL_APPROXDECOMP_GADGET* gadget) { spqlios_free(gadget); }

/** @brief sets res = gadget_decompose(a) */
EXPORT void rnx_approxdecomp_from_tnxdbl_ref(         //
    const MOD_RNX* module,                            // N
    const TNXDBL_APPROXDECOMP_GADGET* gadget,         // output base 2^K
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a                                   // a
) {
  const uint64_t nn = module->n;
  const uint64_t k = gadget->k;
  const uint64_t ell = gadget->ell;
  const double add_cst = gadget->add_cst;
  const uint64_t and_mask = gadget->and_mask;
  const uint64_t or_mask = gadget->or_mask;
  const double sub_cst = gadget->sub_cst;
  const uint64_t msize = res_size <= ell ? res_size : ell;
  const uint64_t first_rsh = (ell - msize) * k;
  // gadget decompose column by column
  if (msize > 0) {
    double* const last_r = res + (msize - 1) * res_sl;
    for (uint64_t j = 0; j < nn; ++j) {
      double* rr = last_r + j;
      di_t t = {.dv = a[j] + add_cst};
      if (msize < ell) t.uv >>= first_rsh;
      do {
        di_t u;
        u.uv = (t.uv & and_mask) | or_mask;
        *rr = u.dv - sub_cst;
        t.uv >>= k;
        rr -= res_sl;
      } while (rr >= res);
    }
  }
  // zero-out the last slices (if any)
  for (uint64_t i = msize; i < res_size; ++i) {
    memset(res + i * res_sl, 0, nn * sizeof(double));
  }
}
