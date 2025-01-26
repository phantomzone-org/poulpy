#ifndef SPQLIOS_Q120_ARITHMETIC_DEF_H
#define SPQLIOS_Q120_ARITHMETIC_DEF_H

#include <stdint.h>

typedef struct _q120_mat1col_product_baa_precomp {
  uint64_t h;
  uint64_t h_pow_red[4];
#ifndef NDEBUG
  double res_bit_size;
#endif
} q120_mat1col_product_baa_precomp;

typedef struct _q120_mat1col_product_bbb_precomp {
  uint64_t h;
  uint64_t s1h_pow_red[4];
  uint64_t s2l_pow_red[4];
  uint64_t s2h_pow_red[4];
  uint64_t s3l_pow_red[4];
  uint64_t s3h_pow_red[4];
  uint64_t s4l_pow_red[4];
  uint64_t s4h_pow_red[4];
#ifndef NDEBUG
  double res_bit_size;
#endif
} q120_mat1col_product_bbb_precomp;

typedef struct _q120_mat1col_product_bbc_precomp {
  uint64_t h;
  uint64_t s2l_pow_red[4];
  uint64_t s2h_pow_red[4];
#ifndef NDEBUG
  double res_bit_size;
#endif
} q120_mat1col_product_bbc_precomp;

#endif  // SPQLIOS_Q120_ARITHMETIC_DEF_H
