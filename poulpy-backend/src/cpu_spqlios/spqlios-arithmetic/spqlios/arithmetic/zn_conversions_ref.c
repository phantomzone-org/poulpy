#include <memory.h>

#include "zn_arithmetic_private.h"

typedef union {
  double dv;
  int64_t s64v;
  int32_t s32v;
  uint64_t u64v;
  uint32_t u32v;
} di_t;

/** reduction mod 1, output in torus32 space */
EXPORT void dbl_to_tn32_ref(const MOD_Z* module,              //
                            int32_t* res, uint64_t res_size,  // res
                            const double* a, uint64_t a_size  // a
) {
  static const double ADD_CST = 0.5 + (double)(INT64_C(3) << (51 - 32));
  static const int32_t XOR_CST = (INT32_C(1) << 31);
  const uint64_t msize = res_size < a_size ? res_size : a_size;
  for (uint64_t i = 0; i < msize; ++i) {
    di_t t = {.dv = a[i] + ADD_CST};
    res[i] = t.s32v ^ XOR_CST;
  }
  memset(res + msize, 0, (res_size - msize) * sizeof(int32_t));
}

/** real centerlift mod 1, output in double space */
EXPORT void tn32_to_dbl_ref(const MOD_Z* module,               //
                            double* res, uint64_t res_size,    // res
                            const int32_t* a, uint64_t a_size  // a
) {
  static const uint32_t XOR_CST = (UINT32_C(1) << 31);
  static const di_t OR_CST = {.dv = (double)(INT64_C(1) << (52 - 32))};
  static const double SUB_CST = 0.5 + (double)(INT64_C(1) << (52 - 32));
  const uint64_t msize = res_size < a_size ? res_size : a_size;
  for (uint64_t i = 0; i < msize; ++i) {
    uint32_t ai = a[i] ^ XOR_CST;
    di_t t = {.u64v = OR_CST.u64v | (uint64_t)ai};
    res[i] = t.dv - SUB_CST;
  }
  memset(res + msize, 0, (res_size - msize) * sizeof(double));
}

/** round to the nearest int, output in i32 space */
EXPORT void dbl_round_to_i32_ref(const MOD_Z* module,              //
                                 int32_t* res, uint64_t res_size,  // res
                                 const double* a, uint64_t a_size  // a
) {
  static const double ADD_CST = (double)((INT64_C(3) << (51)) + (INT64_C(1) << (31)));
  static const int32_t XOR_CST = INT32_C(1) << 31;
  const uint64_t msize = res_size < a_size ? res_size : a_size;
  for (uint64_t i = 0; i < msize; ++i) {
    di_t t = {.dv = a[i] + ADD_CST};
    res[i] = t.s32v ^ XOR_CST;
  }
  memset(res + msize, 0, (res_size - msize) * sizeof(int32_t));
}

/** small int (int32 space) to double */
EXPORT void i32_to_dbl_ref(const MOD_Z* module,               //
                           double* res, uint64_t res_size,    // res
                           const int32_t* a, uint64_t a_size  // a
) {
  static const uint32_t XOR_CST = (UINT32_C(1) << 31);
  static const di_t OR_CST = {.dv = (double)(INT64_C(1) << 52)};
  static const double SUB_CST = (double)((INT64_C(1) << 52) + (INT64_C(1) << 31));
  const uint64_t msize = res_size < a_size ? res_size : a_size;
  for (uint64_t i = 0; i < msize; ++i) {
    uint32_t ai = a[i] ^ XOR_CST;
    di_t t = {.u64v = OR_CST.u64v | (uint64_t)ai};
    res[i] = t.dv - SUB_CST;
  }
  memset(res + msize, 0, (res_size - msize) * sizeof(double));
}

/** round to the nearest int, output in int64 space */
EXPORT void dbl_round_to_i64_ref(const MOD_Z* module,              //
                                 int64_t* res, uint64_t res_size,  // res
                                 const double* a, uint64_t a_size  // a
) {
  static const double ADD_CST = (double)(INT64_C(3) << (51));
  static const int64_t AND_CST = (INT64_C(1) << 52) - 1;
  static const int64_t SUB_CST = INT64_C(1) << 51;
  const uint64_t msize = res_size < a_size ? res_size : a_size;
  for (uint64_t i = 0; i < msize; ++i) {
    di_t t = {.dv = a[i] + ADD_CST};
    res[i] = (t.s64v & AND_CST) - SUB_CST;
  }
  memset(res + msize, 0, (res_size - msize) * sizeof(int64_t));
}

/** small int (int64 space) to double */
EXPORT void i64_to_dbl_ref(const MOD_Z* module,               //
                           double* res, uint64_t res_size,    // res
                           const int64_t* a, uint64_t a_size  // a
) {
  static const uint64_t ADD_CST = UINT64_C(1) << 51;
  static const uint64_t AND_CST = (UINT64_C(1) << 52) - 1;
  static const di_t OR_CST = {.dv = (INT64_C(1) << 52)};
  static const double SUB_CST = INT64_C(3) << 51;
  const uint64_t msize = res_size < a_size ? res_size : a_size;
  for (uint64_t i = 0; i < msize; ++i) {
    di_t t = {.u64v = ((a[i] + ADD_CST) & AND_CST) | OR_CST.u64v};
    res[i] = t.dv - SUB_CST;
  }
  memset(res + msize, 0, (res_size - msize) * sizeof(double));
}
