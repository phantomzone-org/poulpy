#ifndef SPQLIOS_ZN_ARITHMETIC_PLUGIN_H
#define SPQLIOS_ZN_ARITHMETIC_PLUGIN_H

#include "zn_arithmetic.h"

typedef typeof(i8_approxdecomp_from_tndbl) I8_APPROXDECOMP_FROM_TNDBL_F;
typedef typeof(i16_approxdecomp_from_tndbl) I16_APPROXDECOMP_FROM_TNDBL_F;
typedef typeof(i32_approxdecomp_from_tndbl) I32_APPROXDECOMP_FROM_TNDBL_F;
typedef typeof(bytes_of_zn32_vmp_pmat) BYTES_OF_ZN32_VMP_PMAT_F;
typedef typeof(zn32_vmp_prepare_contiguous) ZN32_VMP_PREPARE_CONTIGUOUS_F;
typedef typeof(zn32_vmp_prepare_dblptr) ZN32_VMP_PREPARE_DBLPTR_F;
typedef typeof(zn32_vmp_prepare_row) ZN32_VMP_PREPARE_ROW_F;
typedef typeof(zn32_vmp_apply_i32) ZN32_VMP_APPLY_I32_F;
typedef typeof(zn32_vmp_apply_i16) ZN32_VMP_APPLY_I16_F;
typedef typeof(zn32_vmp_apply_i8) ZN32_VMP_APPLY_I8_F;
typedef typeof(dbl_to_tn32) DBL_TO_TN32_F;
typedef typeof(tn32_to_dbl) TN32_TO_DBL_F;
typedef typeof(dbl_round_to_i32) DBL_ROUND_TO_I32_F;
typedef typeof(i32_to_dbl) I32_TO_DBL_F;
typedef typeof(dbl_round_to_i64) DBL_ROUND_TO_I64_F;
typedef typeof(i64_to_dbl) I64_TO_DBL_F;

typedef struct z_module_vtable_t Z_MODULE_VTABLE;
struct z_module_vtable_t {
  I8_APPROXDECOMP_FROM_TNDBL_F* i8_approxdecomp_from_tndbl;
  I16_APPROXDECOMP_FROM_TNDBL_F* i16_approxdecomp_from_tndbl;
  I32_APPROXDECOMP_FROM_TNDBL_F* i32_approxdecomp_from_tndbl;
  BYTES_OF_ZN32_VMP_PMAT_F* bytes_of_zn32_vmp_pmat;
  ZN32_VMP_PREPARE_CONTIGUOUS_F* zn32_vmp_prepare_contiguous;
  ZN32_VMP_PREPARE_DBLPTR_F* zn32_vmp_prepare_dblptr;
  ZN32_VMP_PREPARE_ROW_F* zn32_vmp_prepare_row;
  ZN32_VMP_APPLY_I32_F* zn32_vmp_apply_i32;
  ZN32_VMP_APPLY_I16_F* zn32_vmp_apply_i16;
  ZN32_VMP_APPLY_I8_F* zn32_vmp_apply_i8;
  DBL_TO_TN32_F* dbl_to_tn32;
  TN32_TO_DBL_F* tn32_to_dbl;
  DBL_ROUND_TO_I32_F* dbl_round_to_i32;
  I32_TO_DBL_F* i32_to_dbl;
  DBL_ROUND_TO_I64_F* dbl_round_to_i64;
  I64_TO_DBL_F* i64_to_dbl;
};

#endif  // SPQLIOS_ZN_ARITHMETIC_PLUGIN_H
