#include <string.h>

#include "vec_rnx_arithmetic_private.h"

void fft64_init_rnx_module_precomp(MOD_RNX* module) {
  // Add here initialization of items that are in the precomp
  const uint64_t m = module->m;
  module->precomp.fft64.p_fft = new_reim_fft_precomp(m, 0);
  module->precomp.fft64.p_ifft = new_reim_ifft_precomp(m, 0);
  module->precomp.fft64.p_fftvec_add = new_reim_fftvec_add_precomp(m);
  module->precomp.fft64.p_fftvec_mul = new_reim_fftvec_mul_precomp(m);
  module->precomp.fft64.p_fftvec_addmul = new_reim_fftvec_addmul_precomp(m);
}

void fft64_finalize_rnx_module_precomp(MOD_RNX* module) {
  // Add here deleters for items that are in the precomp
  delete_reim_fft_precomp(module->precomp.fft64.p_fft);
  delete_reim_ifft_precomp(module->precomp.fft64.p_ifft);
  delete_reim_fftvec_add_precomp(module->precomp.fft64.p_fftvec_add);
  delete_reim_fftvec_mul_precomp(module->precomp.fft64.p_fftvec_mul);
  delete_reim_fftvec_addmul_precomp(module->precomp.fft64.p_fftvec_addmul);
}

void fft64_init_rnx_module_vtable(MOD_RNX* module) {
  // Add function pointers here
  module->vtable.vec_rnx_add = vec_rnx_add_ref;
  module->vtable.vec_rnx_zero = vec_rnx_zero_ref;
  module->vtable.vec_rnx_copy = vec_rnx_copy_ref;
  module->vtable.vec_rnx_negate = vec_rnx_negate_ref;
  module->vtable.vec_rnx_sub = vec_rnx_sub_ref;
  module->vtable.vec_rnx_rotate = vec_rnx_rotate_ref;
  module->vtable.vec_rnx_automorphism = vec_rnx_automorphism_ref;
  module->vtable.vec_rnx_mul_xp_minus_one = vec_rnx_mul_xp_minus_one_ref;
  module->vtable.rnx_vmp_apply_dft_to_dft_tmp_bytes = fft64_rnx_vmp_apply_dft_to_dft_tmp_bytes_ref;
  module->vtable.rnx_vmp_apply_dft_to_dft = fft64_rnx_vmp_apply_dft_to_dft_ref;
  module->vtable.rnx_vmp_apply_tmp_a_tmp_bytes = fft64_rnx_vmp_apply_tmp_a_tmp_bytes_ref;
  module->vtable.rnx_vmp_apply_tmp_a = fft64_rnx_vmp_apply_tmp_a_ref;
  module->vtable.rnx_vmp_prepare_tmp_bytes = fft64_rnx_vmp_prepare_tmp_bytes_ref;
  module->vtable.rnx_vmp_prepare_contiguous = fft64_rnx_vmp_prepare_contiguous_ref;
  module->vtable.rnx_vmp_prepare_dblptr = fft64_rnx_vmp_prepare_dblptr_ref;
  module->vtable.rnx_vmp_prepare_row = fft64_rnx_vmp_prepare_row_ref;
  module->vtable.bytes_of_rnx_vmp_pmat = fft64_bytes_of_rnx_vmp_pmat;
  module->vtable.rnx_approxdecomp_from_tnxdbl = rnx_approxdecomp_from_tnxdbl_ref;
  module->vtable.vec_rnx_to_znx32 = vec_rnx_to_znx32_ref;
  module->vtable.vec_rnx_from_znx32 = vec_rnx_from_znx32_ref;
  module->vtable.vec_rnx_to_tnx32 = vec_rnx_to_tnx32_ref;
  module->vtable.vec_rnx_from_tnx32 = vec_rnx_from_tnx32_ref;
  module->vtable.vec_rnx_to_tnxdbl = vec_rnx_to_tnxdbl_ref;
  module->vtable.bytes_of_rnx_svp_ppol = fft64_bytes_of_rnx_svp_ppol;
  module->vtable.rnx_svp_prepare = fft64_rnx_svp_prepare_ref;
  module->vtable.rnx_svp_apply = fft64_rnx_svp_apply_ref;

  // Add optimized function pointers here
  if (CPU_SUPPORTS("avx")) {
    module->vtable.vec_rnx_add = vec_rnx_add_avx;
    module->vtable.vec_rnx_sub = vec_rnx_sub_avx;
    module->vtable.vec_rnx_negate = vec_rnx_negate_avx;
    module->vtable.rnx_vmp_apply_dft_to_dft_tmp_bytes = fft64_rnx_vmp_apply_dft_to_dft_tmp_bytes_avx;
    module->vtable.rnx_vmp_apply_dft_to_dft = fft64_rnx_vmp_apply_dft_to_dft_avx;
    module->vtable.rnx_vmp_apply_tmp_a_tmp_bytes = fft64_rnx_vmp_apply_tmp_a_tmp_bytes_avx;
    module->vtable.rnx_vmp_apply_tmp_a = fft64_rnx_vmp_apply_tmp_a_avx;
    module->vtable.rnx_vmp_prepare_tmp_bytes = fft64_rnx_vmp_prepare_tmp_bytes_avx;
    module->vtable.rnx_vmp_prepare_contiguous = fft64_rnx_vmp_prepare_contiguous_avx;
    module->vtable.rnx_vmp_prepare_dblptr = fft64_rnx_vmp_prepare_dblptr_avx;
    module->vtable.rnx_vmp_prepare_row = fft64_rnx_vmp_prepare_row_avx;
    module->vtable.rnx_approxdecomp_from_tnxdbl = rnx_approxdecomp_from_tnxdbl_avx;
  }
}

void init_rnx_module_info(MOD_RNX* module,  //
                          uint64_t n, RNX_MODULE_TYPE mtype) {
  memset(module, 0, sizeof(MOD_RNX));
  module->n = n;
  module->m = n >> 1;
  module->mtype = mtype;
  switch (mtype) {
    case FFT64:
      fft64_init_rnx_module_precomp(module);
      fft64_init_rnx_module_vtable(module);
      break;
    default:
      NOT_SUPPORTED();  // unknown mtype
  }
}

void finalize_rnx_module_info(MOD_RNX* module) {
  if (module->custom) module->custom_deleter(module->custom);
  switch (module->mtype) {
    case FFT64:
      fft64_finalize_rnx_module_precomp(module);
      // fft64_finalize_rnx_module_vtable(module); // nothing to finalize
      break;
    default:
      NOT_SUPPORTED();  // unknown mtype
  }
}

EXPORT MOD_RNX* new_rnx_module_info(uint64_t nn, RNX_MODULE_TYPE mtype) {
  MOD_RNX* res = (MOD_RNX*)malloc(sizeof(MOD_RNX));
  init_rnx_module_info(res, nn, mtype);
  return res;
}

EXPORT void delete_rnx_module_info(MOD_RNX* module_info) {
  finalize_rnx_module_info(module_info);
  free(module_info);
}

EXPORT uint64_t rnx_module_get_n(const MOD_RNX* module) { return module->n; }

/** @brief allocates a prepared matrix (release with delete_rnx_vmp_pmat) */
EXPORT RNX_VMP_PMAT* new_rnx_vmp_pmat(const MOD_RNX* module,             // N
                                      uint64_t nrows, uint64_t ncols) {  // dimensions
  return (RNX_VMP_PMAT*)spqlios_alloc(bytes_of_rnx_vmp_pmat(module, nrows, ncols));
}
EXPORT void delete_rnx_vmp_pmat(RNX_VMP_PMAT* ptr) { spqlios_free(ptr); }

//////////////// wrappers //////////////////

/** @brief sets res = a + b */
EXPORT void vec_rnx_add(                              //
    const MOD_RNX* module,                            // N
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl,  // a
    const double* b, uint64_t b_size, uint64_t b_sl   // b
) {
  module->vtable.vec_rnx_add(module, res, res_size, res_sl, a, a_size, a_sl, b, b_size, b_sl);
}

/** @brief sets res = 0 */
EXPORT void vec_rnx_zero(                            //
    const MOD_RNX* module,                           // N
    double* res, uint64_t res_size, uint64_t res_sl  // res
) {
  module->vtable.vec_rnx_zero(module, res, res_size, res_sl);
}

/** @brief sets res = a */
EXPORT void vec_rnx_copy(                             //
    const MOD_RNX* module,                            // N
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl   // a
) {
  module->vtable.vec_rnx_copy(module, res, res_size, res_sl, a, a_size, a_sl);
}

/** @brief sets res = -a */
EXPORT void vec_rnx_negate(                           //
    const MOD_RNX* module,                            // N
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl   // a
) {
  module->vtable.vec_rnx_negate(module, res, res_size, res_sl, a, a_size, a_sl);
}

/** @brief sets res = a - b */
EXPORT void vec_rnx_sub(                              //
    const MOD_RNX* module,                            // N
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl,  // a
    const double* b, uint64_t b_size, uint64_t b_sl   // b
) {
  module->vtable.vec_rnx_sub(module, res, res_size, res_sl, a, a_size, a_sl, b, b_size, b_sl);
}

/** @brief sets res = a . X^p */
EXPORT void vec_rnx_rotate(                           //
    const MOD_RNX* module,                            // N
    const int64_t p,                                  // rotation value
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl   // a
) {
  module->vtable.vec_rnx_rotate(module, p, res, res_size, res_sl, a, a_size, a_sl);
}

/** @brief sets res = a(X^p) */
EXPORT void vec_rnx_automorphism(                     //
    const MOD_RNX* module,                            // N
    int64_t p,                                        // X -> X^p
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl   // a
) {
  module->vtable.vec_rnx_automorphism(module, p, res, res_size, res_sl, a, a_size, a_sl);
}

EXPORT void vec_rnx_mul_xp_minus_one(                 //
    const MOD_RNX* module,                            // N
    const int64_t p,                                  // rotation value
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl   // a
) {
  module->vtable.vec_rnx_mul_xp_minus_one(module, p, res, res_size, res_sl, a, a_size, a_sl);
}
/** @brief number of bytes in a RNX_VMP_PMAT (for manual allocation) */
EXPORT uint64_t bytes_of_rnx_vmp_pmat(const MOD_RNX* module,             // N
                                      uint64_t nrows, uint64_t ncols) {  // dimensions
  return module->vtable.bytes_of_rnx_vmp_pmat(module, nrows, ncols);
}

/** @brief prepares a vmp matrix (contiguous row-major version) */
EXPORT void rnx_vmp_prepare_contiguous(               //
    const MOD_RNX* module,                            // N
    RNX_VMP_PMAT* pmat,                               // output
    const double* a, uint64_t nrows, uint64_t ncols,  // a
    uint8_t* tmp_space                                // scratch space
) {
  module->vtable.rnx_vmp_prepare_contiguous(module, pmat, a, nrows, ncols, tmp_space);
}

/** @brief prepares a vmp matrix (mat[row]+col*N points to the item) */
EXPORT void rnx_vmp_prepare_dblptr(                    //
    const MOD_RNX* module,                             // N
    RNX_VMP_PMAT* pmat,                                // output
    const double** a, uint64_t nrows, uint64_t ncols,  // a
    uint8_t* tmp_space                                 // scratch space
) {
  module->vtable.rnx_vmp_prepare_dblptr(module, pmat, a, nrows, ncols, tmp_space);
}

/** @brief prepares the ith-row of a vmp matrix with nrows and ncols */
EXPORT void rnx_vmp_prepare_row(                                      //
    const MOD_RNX* module,                                            // N
    RNX_VMP_PMAT* pmat,                                               // output
    const double* a, uint64_t row_i, uint64_t nrows, uint64_t ncols,  // a
    uint8_t* tmp_space                                                // scratch space
) {
  module->vtable.rnx_vmp_prepare_row(module, pmat, a, row_i, nrows, ncols, tmp_space);
}

/** @brief number of scratch bytes necessary to prepare a matrix */
EXPORT uint64_t rnx_vmp_prepare_tmp_bytes(const MOD_RNX* module) {
  return module->vtable.rnx_vmp_prepare_tmp_bytes(module);
}

/** @brief applies a vmp product res = a x pmat */
EXPORT void rnx_vmp_apply_tmp_a(                               //
    const MOD_RNX* module,                                     // N
    double* res, uint64_t res_size, uint64_t res_sl,           // res
    double* tmpa, uint64_t a_size, uint64_t a_sl,              // a (will be overwritten)
    const RNX_VMP_PMAT* pmat, uint64_t nrows, uint64_t ncols,  // prep matrix
    uint8_t* tmp_space                                         // scratch space
) {
  module->vtable.rnx_vmp_apply_tmp_a(module, res, res_size, res_sl, tmpa, a_size, a_sl, pmat, nrows, ncols, tmp_space);
}

EXPORT uint64_t rnx_vmp_apply_tmp_a_tmp_bytes(  //
    const MOD_RNX* module,                      // N
    uint64_t res_size,                          // res size
    uint64_t a_size,                            // a size
    uint64_t nrows, uint64_t ncols              // prep matrix dims
) {
  return module->vtable.rnx_vmp_apply_tmp_a_tmp_bytes(module, res_size, a_size, nrows, ncols);
}

/** @brief minimal size of the tmp_space */
EXPORT void rnx_vmp_apply_dft_to_dft(                          //
    const MOD_RNX* module,                                     // N
    double* res, uint64_t res_size, uint64_t res_sl,           // res
    const double* a_dft, uint64_t a_size, uint64_t a_sl,       // a
    const RNX_VMP_PMAT* pmat, uint64_t nrows, uint64_t ncols,  // prep matrix
    uint8_t* tmp_space                                         // scratch space (a_size*sizeof(reim4) bytes)
) {
  module->vtable.rnx_vmp_apply_dft_to_dft(module, res, res_size, res_sl, a_dft, a_size, a_sl, pmat, nrows, ncols,
                                          tmp_space);
}

/** @brief minimal size of the tmp_space */
EXPORT uint64_t rnx_vmp_apply_dft_to_dft_tmp_bytes(  //
    const MOD_RNX* module,                           // N
    uint64_t res_size,                               // res
    uint64_t a_size,                                 // a
    uint64_t nrows, uint64_t ncols                   // prep matrix
) {
  return module->vtable.rnx_vmp_apply_dft_to_dft_tmp_bytes(module, res_size, a_size, nrows, ncols);
}

EXPORT uint64_t bytes_of_rnx_svp_ppol(const MOD_RNX* module) { return module->vtable.bytes_of_rnx_svp_ppol(module); }

EXPORT void rnx_svp_prepare(const MOD_RNX* module,  // N
                            RNX_SVP_PPOL* ppol,     // output
                            const double* pol       // a
) {
  module->vtable.rnx_svp_prepare(module, ppol, pol);
}

EXPORT void rnx_svp_apply(                            //
    const MOD_RNX* module,                            // N
    double* res, uint64_t res_size, uint64_t res_sl,  // output
    const RNX_SVP_PPOL* ppol,                         // prepared pol
    const double* a, uint64_t a_size, uint64_t a_sl   // a
) {
  module->vtable.rnx_svp_apply(module,                 // N
                               res, res_size, res_sl,  // output
                               ppol,                   // prepared pol
                               a, a_size, a_sl);
}

EXPORT void rnx_approxdecomp_from_tnxdbl(             //
    const MOD_RNX* module,                            // N
    const TNXDBL_APPROXDECOMP_GADGET* gadget,         // output base 2^K
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a) {                                // a
  module->vtable.rnx_approxdecomp_from_tnxdbl(module, gadget, res, res_size, res_sl, a);
}

EXPORT void vec_rnx_to_znx32(                          //
    const MOD_RNX* module,                             // N
    int32_t* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl    // a
) {
  module->vtable.vec_rnx_to_znx32(module, res, res_size, res_sl, a, a_size, a_sl);
}

EXPORT void vec_rnx_from_znx32(                       //
    const MOD_RNX* module,                            // N
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const int32_t* a, uint64_t a_size, uint64_t a_sl  // a
) {
  module->vtable.vec_rnx_from_znx32(module, res, res_size, res_sl, a, a_size, a_sl);
}

EXPORT void vec_rnx_to_tnx32(                          //
    const MOD_RNX* module,                             // N
    int32_t* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl    // a
) {
  module->vtable.vec_rnx_to_tnx32(module, res, res_size, res_sl, a, a_size, a_sl);
}

EXPORT void vec_rnx_from_tnx32(                       //
    const MOD_RNX* module,                            // N
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const int32_t* a, uint64_t a_size, uint64_t a_sl  // a
) {
  module->vtable.vec_rnx_from_tnx32(module, res, res_size, res_sl, a, a_size, a_sl);
}

EXPORT void vec_rnx_to_tnxdbl(                        //
    const MOD_RNX* module,                            // N
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl   // a
) {
  module->vtable.vec_rnx_to_tnxdbl(module, res, res_size, res_sl, a, a_size, a_sl);
}
