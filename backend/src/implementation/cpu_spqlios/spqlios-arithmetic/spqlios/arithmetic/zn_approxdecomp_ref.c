#include <memory.h>

#include "zn_arithmetic_private.h"

EXPORT TNDBL_APPROXDECOMP_GADGET* new_tndbl_approxdecomp_gadget(const MOD_Z* module,  //
                                                                uint64_t k, uint64_t ell) {
  if (k * ell > 50) {
    return spqlios_error("approx decomposition requested is too precise for doubles");
  }
  if (k < 1) {
    return spqlios_error("approx decomposition supports k>=1");
  }
  TNDBL_APPROXDECOMP_GADGET* res = malloc(sizeof(TNDBL_APPROXDECOMP_GADGET));
  memset(res, 0, sizeof(TNDBL_APPROXDECOMP_GADGET));
  res->k = k;
  res->ell = ell;
  double add_cst = INT64_C(3) << (51 - k * ell);
  for (uint64_t i = 0; i < ell; ++i) {
    add_cst += pow(2., -(double)(i * k + 1));
  }
  res->add_cst = add_cst;
  res->and_mask = (UINT64_C(1) << k) - 1;
  res->sub_cst = UINT64_C(1) << (k - 1);
  for (uint64_t i = 0; i < ell; ++i) res->rshifts[i] = (ell - 1 - i) * k;
  return res;
}
EXPORT void delete_tndbl_approxdecomp_gadget(TNDBL_APPROXDECOMP_GADGET* ptr) { free(ptr); }

EXPORT int default_init_tndbl_approxdecomp_gadget(const MOD_Z* module,             //
                                                  TNDBL_APPROXDECOMP_GADGET* res,  //
                                                  uint64_t k, uint64_t ell) {
  return 0;
}

typedef union {
  double dv;
  uint64_t uv;
} du_t;

#define IMPL_ixx_approxdecomp_from_tndbl_ref(ITYPE)        \
  if (res_size != a_size * gadget->ell) NOT_IMPLEMENTED(); \
  const uint64_t ell = gadget->ell;                        \
  const double add_cst = gadget->add_cst;                  \
  const uint8_t* const rshifts = gadget->rshifts;          \
  const ITYPE and_mask = gadget->and_mask;                 \
  const ITYPE sub_cst = gadget->sub_cst;                   \
  ITYPE* rr = res;                                         \
  const double* aa = a;                                    \
  const double* aaend = a + a_size;                        \
  while (aa < aaend) {                                     \
    du_t t = {.dv = *aa + add_cst};                        \
    for (uint64_t i = 0; i < ell; ++i) {                   \
      ITYPE v = (ITYPE)(t.uv >> rshifts[i]);               \
      *rr = (v & and_mask) - sub_cst;                      \
      ++rr;                                                \
    }                                                      \
    ++aa;                                                  \
  }

/** @brief sets res = gadget_decompose(a) (int8_t* output) */
EXPORT void default_i8_approxdecomp_from_tndbl_ref(const MOD_Z* module,                      // N
                                                   const TNDBL_APPROXDECOMP_GADGET* gadget,  // gadget
                                                   int8_t* res, uint64_t res_size,  // res (in general, size ell.a_size)
                                                   const double* a, uint64_t a_size  //
){IMPL_ixx_approxdecomp_from_tndbl_ref(int8_t)}

/** @brief sets res = gadget_decompose(a) (int16_t* output) */
EXPORT void default_i16_approxdecomp_from_tndbl_ref(const MOD_Z* module,                      // N
                                                    const TNDBL_APPROXDECOMP_GADGET* gadget,  // gadget
                                                    int16_t* res, uint64_t res_size,          // res
                                                    const double* a, uint64_t a_size          // a
){IMPL_ixx_approxdecomp_from_tndbl_ref(int16_t)}

/** @brief sets res = gadget_decompose(a) (int32_t* output) */
EXPORT void default_i32_approxdecomp_from_tndbl_ref(const MOD_Z* module,                      // N
                                                    const TNDBL_APPROXDECOMP_GADGET* gadget,  // gadget
                                                    int32_t* res, uint64_t res_size,          // res
                                                    const double* a, uint64_t a_size          // a
) {
  IMPL_ixx_approxdecomp_from_tndbl_ref(int32_t)
}
