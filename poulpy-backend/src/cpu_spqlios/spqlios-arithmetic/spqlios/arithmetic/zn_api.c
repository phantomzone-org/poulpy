#include <string.h>

#include "zn_arithmetic_private.h"

void default_init_z_module_precomp(MOD_Z* module) {
  // Add here initialization of items that are in the precomp
}

void default_finalize_z_module_precomp(MOD_Z* module) {
  // Add here deleters for items that are in the precomp
}

void default_init_z_module_vtable(MOD_Z* module) {
  // Add function pointers here
  module->vtable.i8_approxdecomp_from_tndbl = default_i8_approxdecomp_from_tndbl_ref;
  module->vtable.i16_approxdecomp_from_tndbl = default_i16_approxdecomp_from_tndbl_ref;
  module->vtable.i32_approxdecomp_from_tndbl = default_i32_approxdecomp_from_tndbl_ref;
  module->vtable.zn32_vmp_prepare_contiguous = default_zn32_vmp_prepare_contiguous_ref;
  module->vtable.zn32_vmp_prepare_dblptr = default_zn32_vmp_prepare_dblptr_ref;
  module->vtable.zn32_vmp_prepare_row = default_zn32_vmp_prepare_row_ref;
  module->vtable.zn32_vmp_apply_i8 = default_zn32_vmp_apply_i8_ref;
  module->vtable.zn32_vmp_apply_i16 = default_zn32_vmp_apply_i16_ref;
  module->vtable.zn32_vmp_apply_i32 = default_zn32_vmp_apply_i32_ref;
  module->vtable.dbl_to_tn32 = dbl_to_tn32_ref;
  module->vtable.tn32_to_dbl = tn32_to_dbl_ref;
  module->vtable.dbl_round_to_i32 = dbl_round_to_i32_ref;
  module->vtable.i32_to_dbl = i32_to_dbl_ref;
  module->vtable.dbl_round_to_i64 = dbl_round_to_i64_ref;
  module->vtable.i64_to_dbl = i64_to_dbl_ref;

  // Add optimized function pointers here
  if (CPU_SUPPORTS("avx")) {
    module->vtable.zn32_vmp_apply_i8 = default_zn32_vmp_apply_i8_avx;
    module->vtable.zn32_vmp_apply_i16 = default_zn32_vmp_apply_i16_avx;
    module->vtable.zn32_vmp_apply_i32 = default_zn32_vmp_apply_i32_avx;
  }
}

void init_z_module_info(MOD_Z* module,  //
                        Z_MODULE_TYPE mtype) {
  memset(module, 0, sizeof(MOD_Z));
  module->mtype = mtype;
  switch (mtype) {
    case DEFAULT:
      default_init_z_module_precomp(module);
      default_init_z_module_vtable(module);
      break;
    default:
      NOT_SUPPORTED();  // unknown mtype
  }
}

void finalize_z_module_info(MOD_Z* module) {
  if (module->custom) module->custom_deleter(module->custom);
  switch (module->mtype) {
    case DEFAULT:
      default_finalize_z_module_precomp(module);
      // fft64_finalize_rnx_module_vtable(module); // nothing to finalize
      break;
    default:
      NOT_SUPPORTED();  // unknown mtype
  }
}

EXPORT MOD_Z* new_z_module_info(Z_MODULE_TYPE mtype) {
  MOD_Z* res = (MOD_Z*)malloc(sizeof(MOD_Z));
  init_z_module_info(res, mtype);
  return res;
}

EXPORT void delete_z_module_info(MOD_Z* module_info) {
  finalize_z_module_info(module_info);
  free(module_info);
}

//////////////// wrappers //////////////////

/** @brief sets res = gadget_decompose(a) (int8_t* output) */
EXPORT void i8_approxdecomp_from_tndbl(const MOD_Z* module,                      // N
                                       const TNDBL_APPROXDECOMP_GADGET* gadget,  // gadget
                                       int8_t* res, uint64_t res_size,           // res (in general, size ell.a_size)
                                       const double* a, uint64_t a_size) {       // a
  module->vtable.i8_approxdecomp_from_tndbl(module, gadget, res, res_size, a, a_size);
}

/** @brief sets res = gadget_decompose(a) (int16_t* output) */
EXPORT void i16_approxdecomp_from_tndbl(const MOD_Z* module,                      // N
                                        const TNDBL_APPROXDECOMP_GADGET* gadget,  // gadget
                                        int16_t* res, uint64_t res_size,          // res (in general, size ell.a_size)
                                        const double* a, uint64_t a_size) {       // a
  module->vtable.i16_approxdecomp_from_tndbl(module, gadget, res, res_size, a, a_size);
}
/** @brief sets res = gadget_decompose(a) (int32_t* output) */
EXPORT void i32_approxdecomp_from_tndbl(const MOD_Z* module,                      // N
                                        const TNDBL_APPROXDECOMP_GADGET* gadget,  // gadget
                                        int32_t* res, uint64_t res_size,          // res (in general, size ell.a_size)
                                        const double* a, uint64_t a_size) {       // a
  module->vtable.i32_approxdecomp_from_tndbl(module, gadget, res, res_size, a, a_size);
}

/** @brief prepares a vmp matrix (contiguous row-major version) */
EXPORT void zn32_vmp_prepare_contiguous(const MOD_Z* module,
                                        ZN32_VMP_PMAT* pmat,                                   // output
                                        const int32_t* mat, uint64_t nrows, uint64_t ncols) {  // a
  module->vtable.zn32_vmp_prepare_contiguous(module, pmat, mat, nrows, ncols);
}

/** @brief prepares a vmp matrix (mat[row]+col*N points to the item) */
EXPORT void zn32_vmp_prepare_dblptr(const MOD_Z* module,
                                    ZN32_VMP_PMAT* pmat,                                    // output
                                    const int32_t** mat, uint64_t nrows, uint64_t ncols) {  // a
  module->vtable.zn32_vmp_prepare_dblptr(module, pmat, mat, nrows, ncols);
}

/** @brief prepares the ith-row of a vmp matrix with nrows and ncols */
EXPORT void zn32_vmp_prepare_row(const MOD_Z* module,
                                 ZN32_VMP_PMAT* pmat,                                                   // output
                                 const int32_t* row, uint64_t row_i, uint64_t nrows, uint64_t ncols) {  // a
  module->vtable.zn32_vmp_prepare_row(module, pmat, row, row_i, nrows, ncols);
}

/** @brief applies a vmp product (int32_t* input) */
EXPORT void zn32_vmp_apply_i32(const MOD_Z* module, int32_t* res, uint64_t res_size, const int32_t* a, uint64_t a_size,
                               const ZN32_VMP_PMAT* pmat, uint64_t nrows, uint64_t ncols) {
  module->vtable.zn32_vmp_apply_i32(module, res, res_size, a, a_size, pmat, nrows, ncols);
}
/** @brief applies a vmp product (int16_t* input) */
EXPORT void zn32_vmp_apply_i16(const MOD_Z* module, int32_t* res, uint64_t res_size, const int16_t* a, uint64_t a_size,
                               const ZN32_VMP_PMAT* pmat, uint64_t nrows, uint64_t ncols) {
  module->vtable.zn32_vmp_apply_i16(module, res, res_size, a, a_size, pmat, nrows, ncols);
}

/** @brief applies a vmp product (int8_t* input) */
EXPORT void zn32_vmp_apply_i8(const MOD_Z* module, int32_t* res, uint64_t res_size, const int8_t* a, uint64_t a_size,
                              const ZN32_VMP_PMAT* pmat, uint64_t nrows, uint64_t ncols) {
  module->vtable.zn32_vmp_apply_i8(module, res, res_size, a, a_size, pmat, nrows, ncols);
}

/** reduction mod 1, output in torus32 space */
EXPORT void dbl_to_tn32(const MOD_Z* module,              //
                        int32_t* res, uint64_t res_size,  // res
                        const double* a, uint64_t a_size  // a
) {
  module->vtable.dbl_to_tn32(module, res, res_size, a, a_size);
}

/** real centerlift mod 1, output in double space */
EXPORT void tn32_to_dbl(const MOD_Z* module,               //
                        double* res, uint64_t res_size,    // res
                        const int32_t* a, uint64_t a_size  // a
) {
  module->vtable.tn32_to_dbl(module, res, res_size, a, a_size);
}

/** round to the nearest int, output in i32 space */
EXPORT void dbl_round_to_i32(const MOD_Z* module,              //
                             int32_t* res, uint64_t res_size,  // res
                             const double* a, uint64_t a_size  // a
) {
  module->vtable.dbl_round_to_i32(module, res, res_size, a, a_size);
}

/** small int (int32 space) to double */
EXPORT void i32_to_dbl(const MOD_Z* module,               //
                       double* res, uint64_t res_size,    // res
                       const int32_t* a, uint64_t a_size  // a
) {
  module->vtable.i32_to_dbl(module, res, res_size, a, a_size);
}

/** round to the nearest int, output in int64 space */
EXPORT void dbl_round_to_i64(const MOD_Z* module,              //
                             int64_t* res, uint64_t res_size,  // res
                             const double* a, uint64_t a_size  // a
) {
  module->vtable.dbl_round_to_i64(module, res, res_size, a, a_size);
}

/** small int (int64 space, <= 2^50) to double */
EXPORT void i64_to_dbl(const MOD_Z* module,               //
                       double* res, uint64_t res_size,    // res
                       const int64_t* a, uint64_t a_size  // a
) {
  module->vtable.i64_to_dbl(module, res, res_size, a, a_size);
}
