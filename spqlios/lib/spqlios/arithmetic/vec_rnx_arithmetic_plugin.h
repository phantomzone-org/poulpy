#ifndef SPQLIOS_VEC_RNX_ARITHMETIC_PLUGIN_H
#define SPQLIOS_VEC_RNX_ARITHMETIC_PLUGIN_H

#include "vec_rnx_arithmetic.h"

typedef typeof(vec_rnx_zero) VEC_RNX_ZERO_F;
typedef typeof(vec_rnx_copy) VEC_RNX_COPY_F;
typedef typeof(vec_rnx_negate) VEC_RNX_NEGATE_F;
typedef typeof(vec_rnx_add) VEC_RNX_ADD_F;
typedef typeof(vec_rnx_sub) VEC_RNX_SUB_F;
typedef typeof(vec_rnx_rotate) VEC_RNX_ROTATE_F;
typedef typeof(vec_rnx_mul_xp_minus_one) VEC_RNX_MUL_XP_MINUS_ONE_F;
typedef typeof(vec_rnx_automorphism) VEC_RNX_AUTOMORPHISM_F;
typedef typeof(vec_rnx_to_znx32) VEC_RNX_TO_ZNX32_F;
typedef typeof(vec_rnx_from_znx32) VEC_RNX_FROM_ZNX32_F;
typedef typeof(vec_rnx_to_tnx32) VEC_RNX_TO_TNX32_F;
typedef typeof(vec_rnx_from_tnx32) VEC_RNX_FROM_TNX32_F;
typedef typeof(vec_rnx_to_tnx32x2) VEC_RNX_TO_TNX32X2_F;
typedef typeof(vec_rnx_from_tnx32x2) VEC_RNX_FROM_TNX32X2_F;
typedef typeof(vec_rnx_to_tnxdbl) VEC_RNX_TO_TNXDBL_F;
// typedef typeof(vec_rnx_from_tnxdbl) VEC_RNX_FROM_TNXDBL_F;
typedef typeof(rnx_small_single_product) RNX_SMALL_SINGLE_PRODUCT_F;
typedef typeof(rnx_small_single_product_tmp_bytes) RNX_SMALL_SINGLE_PRODUCT_TMP_BYTES_F;
typedef typeof(tnxdbl_small_single_product) TNXDBL_SMALL_SINGLE_PRODUCT_F;
typedef typeof(tnxdbl_small_single_product_tmp_bytes) TNXDBL_SMALL_SINGLE_PRODUCT_TMP_BYTES_F;
typedef typeof(znx32_small_single_product) ZNX32_SMALL_SINGLE_PRODUCT_F;
typedef typeof(znx32_small_single_product_tmp_bytes) ZNX32_SMALL_SINGLE_PRODUCT_TMP_BYTES_F;
typedef typeof(tnx32_small_single_product) TNX32_SMALL_SINGLE_PRODUCT_F;
typedef typeof(tnx32_small_single_product_tmp_bytes) TNX32_SMALL_SINGLE_PRODUCT_TMP_BYTES_F;
typedef typeof(rnx_approxdecomp_from_tnx32) RNX_APPROXDECOMP_FROM_TNX32_F;
typedef typeof(rnx_approxdecomp_from_tnx32x2) RNX_APPROXDECOMP_FROM_TNX32X2_F;
typedef typeof(rnx_approxdecomp_from_tnxdbl) RNX_APPROXDECOMP_FROM_TNXDBL_F;
typedef typeof(bytes_of_rnx_svp_ppol) BYTES_OF_RNX_SVP_PPOL_F;
typedef typeof(rnx_svp_prepare) RNX_SVP_PREPARE_F;
typedef typeof(rnx_svp_apply) RNX_SVP_APPLY_F;
typedef typeof(bytes_of_rnx_vmp_pmat) BYTES_OF_RNX_VMP_PMAT_F;
typedef typeof(rnx_vmp_prepare_contiguous) RNX_VMP_PREPARE_CONTIGUOUS_F;
typedef typeof(rnx_vmp_prepare_contiguous_tmp_bytes) RNX_VMP_PREPARE_CONTIGUOUS_TMP_BYTES_F;
typedef typeof(rnx_vmp_apply_tmp_a) RNX_VMP_APPLY_TMP_A_F;
typedef typeof(rnx_vmp_apply_tmp_a_tmp_bytes) RNX_VMP_APPLY_TMP_A_TMP_BYTES_F;
typedef typeof(rnx_vmp_apply_dft_to_dft) RNX_VMP_APPLY_DFT_TO_DFT_F;
typedef typeof(rnx_vmp_apply_dft_to_dft_tmp_bytes) RNX_VMP_APPLY_DFT_TO_DFT_TMP_BYTES_F;
typedef typeof(vec_rnx_dft) VEC_RNX_DFT_F;
typedef typeof(vec_rnx_idft) VEC_RNX_IDFT_F;

typedef struct rnx_module_vtable_t RNX_MODULE_VTABLE;
struct rnx_module_vtable_t {
  VEC_RNX_ZERO_F* vec_rnx_zero;
  VEC_RNX_COPY_F* vec_rnx_copy;
  VEC_RNX_NEGATE_F* vec_rnx_negate;
  VEC_RNX_ADD_F* vec_rnx_add;
  VEC_RNX_SUB_F* vec_rnx_sub;
  VEC_RNX_ROTATE_F* vec_rnx_rotate;
  VEC_RNX_MUL_XP_MINUS_ONE_F* vec_rnx_mul_xp_minus_one;
  VEC_RNX_AUTOMORPHISM_F* vec_rnx_automorphism;
  VEC_RNX_TO_ZNX32_F* vec_rnx_to_znx32;
  VEC_RNX_FROM_ZNX32_F* vec_rnx_from_znx32;
  VEC_RNX_TO_TNX32_F* vec_rnx_to_tnx32;
  VEC_RNX_FROM_TNX32_F* vec_rnx_from_tnx32;
  VEC_RNX_TO_TNX32X2_F* vec_rnx_to_tnx32x2;
  VEC_RNX_FROM_TNX32X2_F* vec_rnx_from_tnx32x2;
  VEC_RNX_TO_TNXDBL_F* vec_rnx_to_tnxdbl;
  RNX_SMALL_SINGLE_PRODUCT_F* rnx_small_single_product;
  RNX_SMALL_SINGLE_PRODUCT_TMP_BYTES_F* rnx_small_single_product_tmp_bytes;
  TNXDBL_SMALL_SINGLE_PRODUCT_F* tnxdbl_small_single_product;
  TNXDBL_SMALL_SINGLE_PRODUCT_TMP_BYTES_F* tnxdbl_small_single_product_tmp_bytes;
  ZNX32_SMALL_SINGLE_PRODUCT_F* znx32_small_single_product;
  ZNX32_SMALL_SINGLE_PRODUCT_TMP_BYTES_F* znx32_small_single_product_tmp_bytes;
  TNX32_SMALL_SINGLE_PRODUCT_F* tnx32_small_single_product;
  TNX32_SMALL_SINGLE_PRODUCT_TMP_BYTES_F* tnx32_small_single_product_tmp_bytes;
  RNX_APPROXDECOMP_FROM_TNX32_F* rnx_approxdecomp_from_tnx32;
  RNX_APPROXDECOMP_FROM_TNX32X2_F* rnx_approxdecomp_from_tnx32x2;
  RNX_APPROXDECOMP_FROM_TNXDBL_F* rnx_approxdecomp_from_tnxdbl;
  BYTES_OF_RNX_SVP_PPOL_F* bytes_of_rnx_svp_ppol;
  RNX_SVP_PREPARE_F* rnx_svp_prepare;
  RNX_SVP_APPLY_F* rnx_svp_apply;
  BYTES_OF_RNX_VMP_PMAT_F* bytes_of_rnx_vmp_pmat;
  RNX_VMP_PREPARE_CONTIGUOUS_F* rnx_vmp_prepare_contiguous;
  RNX_VMP_PREPARE_CONTIGUOUS_TMP_BYTES_F* rnx_vmp_prepare_contiguous_tmp_bytes;
  RNX_VMP_APPLY_TMP_A_F* rnx_vmp_apply_tmp_a;
  RNX_VMP_APPLY_TMP_A_TMP_BYTES_F* rnx_vmp_apply_tmp_a_tmp_bytes;
  RNX_VMP_APPLY_DFT_TO_DFT_F* rnx_vmp_apply_dft_to_dft;
  RNX_VMP_APPLY_DFT_TO_DFT_TMP_BYTES_F* rnx_vmp_apply_dft_to_dft_tmp_bytes;
  VEC_RNX_DFT_F* vec_rnx_dft;
  VEC_RNX_IDFT_F* vec_rnx_idft;
};

#endif  // SPQLIOS_VEC_RNX_ARITHMETIC_PLUGIN_H
