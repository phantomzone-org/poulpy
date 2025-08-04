#include <assert.h>
#include <math.h>
#include <stdint.h>

#include "q120_arithmetic.h"
#include "q120_common.h"

EXPORT void q120_add_bbb_simple(uint64_t nn, q120b* const res, const q120b* const x, const q120b* const y) {
  const uint64_t* x_u64 = (uint64_t*)x;
  const uint64_t* y_u64 = (uint64_t*)y;
  uint64_t* res_u64 = (uint64_t*)res;
  for (uint64_t i = 0; i < 4 * nn; i += 4) {
    res_u64[i + 0] = x_u64[i + 0] % ((uint64_t)Q1 << 33) + y_u64[i + 0] % ((uint64_t)Q1 << 33);
    res_u64[i + 1] = x_u64[i + 1] % ((uint64_t)Q2 << 33) + y_u64[i + 1] % ((uint64_t)Q2 << 33);
    res_u64[i + 2] = x_u64[i + 2] % ((uint64_t)Q3 << 33) + y_u64[i + 2] % ((uint64_t)Q3 << 33);
    res_u64[i + 3] = x_u64[i + 3] % ((uint64_t)Q4 << 33) + y_u64[i + 3] % ((uint64_t)Q4 << 33);
  }
}

EXPORT void q120_add_ccc_simple(uint64_t nn, q120c* const res, const q120c* const x, const q120c* const y) {
  const uint32_t* x_u32 = (uint32_t*)x;
  const uint32_t* y_u32 = (uint32_t*)y;
  uint32_t* res_u32 = (uint32_t*)res;
  for (uint64_t i = 0; i < 8 * nn; i += 8) {
    res_u32[i + 0] = (uint32_t)(((uint64_t)x_u32[i + 0] + (uint64_t)y_u32[i + 0]) % Q1);
    res_u32[i + 1] = (uint32_t)(((uint64_t)x_u32[i + 1] + (uint64_t)y_u32[i + 1]) % Q1);
    res_u32[i + 2] = (uint32_t)(((uint64_t)x_u32[i + 2] + (uint64_t)y_u32[i + 2]) % Q2);
    res_u32[i + 3] = (uint32_t)(((uint64_t)x_u32[i + 3] + (uint64_t)y_u32[i + 3]) % Q2);
    res_u32[i + 4] = (uint32_t)(((uint64_t)x_u32[i + 4] + (uint64_t)y_u32[i + 4]) % Q3);
    res_u32[i + 5] = (uint32_t)(((uint64_t)x_u32[i + 5] + (uint64_t)y_u32[i + 5]) % Q3);
    res_u32[i + 6] = (uint32_t)(((uint64_t)x_u32[i + 6] + (uint64_t)y_u32[i + 6]) % Q4);
    res_u32[i + 7] = (uint32_t)(((uint64_t)x_u32[i + 7] + (uint64_t)y_u32[i + 7]) % Q4);
  }
}

EXPORT void q120_c_from_b_simple(uint64_t nn, q120c* const res, const q120b* const x) {
  const uint64_t* x_u64 = (uint64_t*)x;
  uint32_t* res_u32 = (uint32_t*)res;
  for (uint64_t i = 0, j = 0; i < 4 * nn; i += 4, j += 8) {
    res_u32[j + 0] = x_u64[i + 0] % Q1;
    res_u32[j + 1] = ((uint64_t)res_u32[j + 0] << 32) % Q1;
    res_u32[j + 2] = x_u64[i + 1] % Q2;
    res_u32[j + 3] = ((uint64_t)res_u32[j + 2] << 32) % Q2;
    res_u32[j + 4] = x_u64[i + 2] % Q3;
    res_u32[j + 5] = ((uint64_t)res_u32[j + 4] << 32) % Q3;
    res_u32[j + 6] = x_u64[i + 3] % Q4;
    res_u32[j + 7] = ((uint64_t)res_u32[j + 6] << 32) % Q4;
  }
}

EXPORT void q120_b_from_znx64_simple(uint64_t nn, q120b* const res, const int64_t* const x) {
  static const int64_t MASK_HI = INT64_C(0x8000000000000000);
  static const int64_t MASK_LO = ~MASK_HI;
  static const uint64_t OQ[4] = {
      (Q1 - (UINT64_C(0x8000000000000000) % Q1)),
      (Q2 - (UINT64_C(0x8000000000000000) % Q2)),
      (Q3 - (UINT64_C(0x8000000000000000) % Q3)),
      (Q4 - (UINT64_C(0x8000000000000000) % Q4)),
  };
  uint64_t* res_u64 = (uint64_t*)res;
  for (uint64_t i = 0, j = 0; j < nn; i += 4, ++j) {
    uint64_t xj_lo = x[j] & MASK_LO;
    uint64_t xj_hi = x[j] & MASK_HI;
    res_u64[i + 0] = xj_lo + (xj_hi ? OQ[0] : 0);
    res_u64[i + 1] = xj_lo + (xj_hi ? OQ[1] : 0);
    res_u64[i + 2] = xj_lo + (xj_hi ? OQ[2] : 0);
    res_u64[i + 3] = xj_lo + (xj_hi ? OQ[3] : 0);
  }
}

static int64_t posmod(int64_t x, int64_t q) {
  int64_t t = x % q;
  if (t < 0)
    return t + q;
  else
    return t;
}

EXPORT void q120_c_from_znx64_simple(uint64_t nn, q120c* const res, const int64_t* const x) {
  uint32_t* res_u32 = (uint32_t*)res;
  for (uint64_t i = 0, j = 0; j < nn; i += 8, ++j) {
    res_u32[i + 0] = posmod(x[j], Q1);
    res_u32[i + 1] = ((uint64_t)res_u32[i + 0] << 32) % Q1;
    res_u32[i + 2] = posmod(x[j], Q2);
    res_u32[i + 3] = ((uint64_t)res_u32[i + 2] << 32) % Q2;
    res_u32[i + 4] = posmod(x[j], Q3);
    res_u32[i + 5] = ((uint64_t)res_u32[i + 4] << 32) % Q3;
    res_u32[i + 6] = posmod(x[j], Q4);
    res_u32[i + 7] = ((uint64_t)res_u32[i + 6] << 32) % Q4;
    ;
  }
}

EXPORT void q120_b_to_znx128_simple(uint64_t nn, __int128_t* const res, const q120b* const x) {
  static const __int128_t Q = (__int128_t)Q1 * Q2 * Q3 * Q4;
  static const __int128_t Qm1 = (__int128_t)Q2 * Q3 * Q4;
  static const __int128_t Qm2 = (__int128_t)Q1 * Q3 * Q4;
  static const __int128_t Qm3 = (__int128_t)Q1 * Q2 * Q4;
  static const __int128_t Qm4 = (__int128_t)Q1 * Q2 * Q3;

  const uint64_t* x_u64 = (uint64_t*)x;
  for (uint64_t i = 0, j = 0; j < nn; i += 4, ++j) {
    __int128_t tmp = 0;
    tmp += (((x_u64[i + 0] % Q1) * Q1_CRT_CST) % Q1) * Qm1;
    tmp += (((x_u64[i + 1] % Q2) * Q2_CRT_CST) % Q2) * Qm2;
    tmp += (((x_u64[i + 2] % Q3) * Q3_CRT_CST) % Q3) * Qm3;
    tmp += (((x_u64[i + 3] % Q4) * Q4_CRT_CST) % Q4) * Qm4;
    tmp %= Q;
    res[j] = (tmp >= (Q + 1) / 2) ? tmp - Q : tmp;
  }
}
