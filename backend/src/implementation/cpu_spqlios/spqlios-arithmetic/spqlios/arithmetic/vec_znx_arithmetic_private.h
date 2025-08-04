#ifndef SPQLIOS_VEC_ZNX_ARITHMETIC_PRIVATE_H
#define SPQLIOS_VEC_ZNX_ARITHMETIC_PRIVATE_H

#include "../commons_private.h"
#include "../q120/q120_ntt.h"
#include "vec_znx_arithmetic.h"

/**
 * Layouts families:
 *
 * fft64:
 *   K: <= 20, N: <= 65536, ell: <= 200
 *   vec<ZnX> normalized: represented by int64
 *   vec<ZnX> large: represented by int64 (expect <=52 bits)
 *   vec<ZnX> DFT: represented by double (reim_fft space)
 *   On AVX2 inftastructure, PMAT, LCNV, RCNV use a special reim4_fft space
 *
 * ntt120:
 *   K: <= 50, N: <= 65536, ell: <= 80
 *   vec<ZnX> normalized: represented by int64
 *   vec<ZnX> large: represented by int128 (expect <=120 bits)
 *   vec<ZnX> DFT: represented by int64x4 (ntt120 space)
 *   On AVX2 inftastructure, PMAT, LCNV, RCNV use a special ntt120 space
 *
 * ntt104:
 *   K: <= 40, N: <= 65536, ell: <= 80
 *   vec<ZnX> normalized: represented by int64
 *   vec<ZnX> large: represented by int128 (expect <=120 bits)
 *   vec<ZnX> DFT: represented by int64x4 (ntt120 space)
 *   On AVX512 inftastructure, PMAT, LCNV, RCNV use a special ntt104 space
 */

struct fft64_module_info_t {
  // pre-computation for reim_fft
  REIM_FFT_PRECOMP* p_fft;
  // pre-computation for add_fft
  REIM_FFTVEC_ADD_PRECOMP* add_fft;
  // pre-computation for add_fft
  REIM_FFTVEC_SUB_PRECOMP* sub_fft;
  // pre-computation for mul_fft
  REIM_FFTVEC_MUL_PRECOMP* mul_fft;
  // pre-computation for reim_from_znx6
  REIM_FROM_ZNX64_PRECOMP* p_conv;
  // pre-computation for reim_tp_znx6
  REIM_TO_ZNX64_PRECOMP* p_reim_to_znx;
  // pre-computation for reim_fft
  REIM_IFFT_PRECOMP* p_ifft;
  // pre-computation for reim_fftvec_addmul
  REIM_FFTVEC_ADDMUL_PRECOMP* p_addmul;
  // pre-computation for reim_fftvec_automorphism
  REIM_FFTVEC_AUTOMORPHISM_PRECOMP* p_automorphism;
};

struct q120_module_info_t {
  // pre-computation for q120b to q120b ntt
  q120_ntt_precomp* p_ntt;
  // pre-computation for q120b to q120b intt
  q120_ntt_precomp* p_intt;
};

// TODO add function types here
typedef typeof(vec_znx_zero) VEC_ZNX_ZERO_F;
typedef typeof(vec_znx_copy) VEC_ZNX_COPY_F;
typedef typeof(vec_znx_negate) VEC_ZNX_NEGATE_F;
typedef typeof(vec_znx_add) VEC_ZNX_ADD_F;
typedef typeof(vec_znx_dft_automorphism_tmp_bytes) VEC_ZNX_DFT_AUTOMORPHISM_TMP_BYTES_F;
typedef typeof(vec_znx_dft_automorphism) VEC_ZNX_DFT_AUTOMORPHISM_F;
typedef typeof(vec_znx_dft) VEC_ZNX_DFT_F;
typedef typeof(vec_dft_add) VEC_DFT_ADD_F;
typedef typeof(vec_dft_sub) VEC_DFT_SUB_F;
typedef typeof(vec_znx_idft) VEC_ZNX_IDFT_F;
typedef typeof(vec_znx_idft_tmp_bytes) VEC_ZNX_IDFT_TMP_BYTES_F;
typedef typeof(vec_znx_idft_tmp_a) VEC_ZNX_IDFT_TMP_A_F;
typedef typeof(vec_znx_sub) VEC_ZNX_SUB_F;
typedef typeof(vec_znx_rotate) VEC_ZNX_ROTATE_F;
typedef typeof(vec_znx_automorphism) VEC_ZNX_AUTOMORPHISM_F;
typedef typeof(vec_znx_normalize_base2k) VEC_ZNX_NORMALIZE_BASE2K_F;
typedef typeof(vec_znx_normalize_base2k_tmp_bytes) VEC_ZNX_NORMALIZE_BASE2K_TMP_BYTES_F;
typedef typeof(vec_znx_big_normalize_base2k) VEC_ZNX_BIG_NORMALIZE_BASE2K_F;
typedef typeof(vec_znx_big_normalize_base2k_tmp_bytes) VEC_ZNX_BIG_NORMALIZE_BASE2K_TMP_BYTES_F;
typedef typeof(vec_znx_big_range_normalize_base2k) VEC_ZNX_BIG_RANGE_NORMALIZE_BASE2K_F;
typedef typeof(vec_znx_big_range_normalize_base2k_tmp_bytes) VEC_ZNX_BIG_RANGE_NORMALIZE_BASE2K_TMP_BYTES_F;
typedef typeof(vec_znx_big_add) VEC_ZNX_BIG_ADD_F;
typedef typeof(vec_znx_big_add_small) VEC_ZNX_BIG_ADD_SMALL_F;
typedef typeof(vec_znx_big_add_small2) VEC_ZNX_BIG_ADD_SMALL2_F;
typedef typeof(vec_znx_big_sub) VEC_ZNX_BIG_SUB_F;
typedef typeof(vec_znx_big_sub_small_a) VEC_ZNX_BIG_SUB_SMALL_A_F;
typedef typeof(vec_znx_big_sub_small_b) VEC_ZNX_BIG_SUB_SMALL_B_F;
typedef typeof(vec_znx_big_sub_small2) VEC_ZNX_BIG_SUB_SMALL2_F;
typedef typeof(vec_znx_big_rotate) VEC_ZNX_BIG_ROTATE_F;
typedef typeof(vec_znx_big_automorphism) VEC_ZNX_BIG_AUTOMORPHISM_F;
typedef typeof(svp_prepare) SVP_PREPARE;
typedef typeof(svp_apply_dft) SVP_APPLY_DFT_F;
typedef typeof(svp_apply_dft_to_dft) SVP_APPLY_DFT_TO_DFT_F;
typedef typeof(znx_small_single_product) ZNX_SMALL_SINGLE_PRODUCT_F;
typedef typeof(znx_small_single_product_tmp_bytes) ZNX_SMALL_SINGLE_PRODUCT_TMP_BYTES_F;
typedef typeof(vmp_prepare_contiguous) VMP_PREPARE_CONTIGUOUS_F;
typedef typeof(vmp_prepare_tmp_bytes) VMP_PREPARE_TMP_BYTES_F;
typedef typeof(vmp_apply_dft) VMP_APPLY_DFT_F;
typedef typeof(vmp_apply_dft_add) VMP_APPLY_DFT_ADD_F;
typedef typeof(vmp_apply_dft_tmp_bytes) VMP_APPLY_DFT_TMP_BYTES_F;
typedef typeof(vmp_apply_dft_to_dft) VMP_APPLY_DFT_TO_DFT_F;
typedef typeof(vmp_apply_dft_to_dft_add) VMP_APPLY_DFT_TO_DFT_ADD_F;
typedef typeof(vmp_apply_dft_to_dft_tmp_bytes) VMP_APPLY_DFT_TO_DFT_TMP_BYTES_F;
typedef typeof(bytes_of_vec_znx_dft) BYTES_OF_VEC_ZNX_DFT_F;
typedef typeof(bytes_of_vec_znx_big) BYTES_OF_VEC_ZNX_BIG_F;
typedef typeof(bytes_of_svp_ppol) BYTES_OF_SVP_PPOL_F;
typedef typeof(bytes_of_vmp_pmat) BYTES_OF_VMP_PMAT_F;

struct module_virtual_functions_t {
  // TODO add functions here
  VEC_ZNX_ZERO_F* vec_znx_zero;
  VEC_ZNX_COPY_F* vec_znx_copy;
  VEC_ZNX_NEGATE_F* vec_znx_negate;
  VEC_ZNX_ADD_F* vec_znx_add;
  VEC_ZNX_DFT_F* vec_znx_dft;
  VEC_DFT_ADD_F* vec_dft_add;
  VEC_DFT_SUB_F* vec_dft_sub;
  VEC_ZNX_IDFT_F* vec_znx_idft;
  VEC_ZNX_IDFT_TMP_BYTES_F* vec_znx_idft_tmp_bytes;
  VEC_ZNX_IDFT_TMP_A_F* vec_znx_idft_tmp_a;
  VEC_ZNX_SUB_F* vec_znx_sub;
  VEC_ZNX_ROTATE_F* vec_znx_rotate;
  VEC_ZNX_AUTOMORPHISM_F* vec_znx_automorphism;
  VEC_ZNX_DFT_AUTOMORPHISM_F* vec_znx_dft_automorphism;
  VEC_ZNX_DFT_AUTOMORPHISM_TMP_BYTES_F* vec_znx_dft_automorphism_tmp_bytes;
  VEC_ZNX_NORMALIZE_BASE2K_F* vec_znx_normalize_base2k;
  VEC_ZNX_NORMALIZE_BASE2K_TMP_BYTES_F* vec_znx_normalize_base2k_tmp_bytes;
  VEC_ZNX_BIG_NORMALIZE_BASE2K_F* vec_znx_big_normalize_base2k;
  VEC_ZNX_BIG_NORMALIZE_BASE2K_TMP_BYTES_F* vec_znx_big_normalize_base2k_tmp_bytes;
  VEC_ZNX_BIG_RANGE_NORMALIZE_BASE2K_F* vec_znx_big_range_normalize_base2k;
  VEC_ZNX_BIG_RANGE_NORMALIZE_BASE2K_TMP_BYTES_F* vec_znx_big_range_normalize_base2k_tmp_bytes;
  VEC_ZNX_BIG_ADD_F* vec_znx_big_add;
  VEC_ZNX_BIG_ADD_SMALL_F* vec_znx_big_add_small;
  VEC_ZNX_BIG_ADD_SMALL2_F* vec_znx_big_add_small2;
  VEC_ZNX_BIG_SUB_F* vec_znx_big_sub;
  VEC_ZNX_BIG_SUB_SMALL_A_F* vec_znx_big_sub_small_a;
  VEC_ZNX_BIG_SUB_SMALL_B_F* vec_znx_big_sub_small_b;
  VEC_ZNX_BIG_SUB_SMALL2_F* vec_znx_big_sub_small2;
  VEC_ZNX_BIG_ROTATE_F* vec_znx_big_rotate;
  VEC_ZNX_BIG_AUTOMORPHISM_F* vec_znx_big_automorphism;
  SVP_PREPARE* svp_prepare;
  SVP_APPLY_DFT_F* svp_apply_dft;
  SVP_APPLY_DFT_TO_DFT_F* svp_apply_dft_to_dft;
  ZNX_SMALL_SINGLE_PRODUCT_F* znx_small_single_product;
  ZNX_SMALL_SINGLE_PRODUCT_TMP_BYTES_F* znx_small_single_product_tmp_bytes;
  VMP_PREPARE_CONTIGUOUS_F* vmp_prepare_contiguous;
  VMP_PREPARE_TMP_BYTES_F* vmp_prepare_tmp_bytes;
  VMP_APPLY_DFT_F* vmp_apply_dft;
  VMP_APPLY_DFT_ADD_F* vmp_apply_dft_add;
  VMP_APPLY_DFT_TMP_BYTES_F* vmp_apply_dft_tmp_bytes;
  VMP_APPLY_DFT_TO_DFT_F* vmp_apply_dft_to_dft;
  VMP_APPLY_DFT_TO_DFT_ADD_F* vmp_apply_dft_to_dft_add;
  VMP_APPLY_DFT_TO_DFT_TMP_BYTES_F* vmp_apply_dft_to_dft_tmp_bytes;
  BYTES_OF_VEC_ZNX_DFT_F* bytes_of_vec_znx_dft;
  BYTES_OF_VEC_ZNX_BIG_F* bytes_of_vec_znx_big;
  BYTES_OF_SVP_PPOL_F* bytes_of_svp_ppol;
  BYTES_OF_VMP_PMAT_F* bytes_of_vmp_pmat;
};

union backend_module_info_t {
  struct fft64_module_info_t fft64;
  struct q120_module_info_t q120;
};

struct module_info_t {
  // generic parameters
  MODULE_TYPE module_type;
  uint64_t nn;
  uint64_t m;
  // backend_dependent functions
  union backend_module_info_t mod;
  // virtual functions
  struct module_virtual_functions_t func;
};

EXPORT uint64_t fft64_bytes_of_vec_znx_dft(const MODULE* module,  // N
                                           uint64_t size);

EXPORT uint64_t fft64_bytes_of_vec_znx_big(const MODULE* module,  // N
                                           uint64_t size);

EXPORT uint64_t fft64_bytes_of_svp_ppol(const MODULE* module);  // N

EXPORT uint64_t fft64_bytes_of_vmp_pmat(const MODULE* module,  // N
                                        uint64_t nrows, uint64_t ncols);

EXPORT void vec_znx_zero_ref(const MODULE* module,                             // N
                             int64_t* res, uint64_t res_size, uint64_t res_sl  // res
);

EXPORT void vec_znx_copy_ref(const MODULE* module,                             // N
                             int64_t* res, uint64_t res_size, uint64_t res_sl,  // res
                             const int64_t* a, uint64_t a_size, uint64_t a_sl   // a
);

EXPORT void vec_znx_negate_ref(const MODULE* module,                              // N
                               int64_t* res, uint64_t res_size, uint64_t res_sl,  // res
                               const int64_t* a, uint64_t a_size, uint64_t a_sl   // a
);

EXPORT void vec_znx_negate_avx(const MODULE* module,                              // N
                               int64_t* res, uint64_t res_size, uint64_t res_sl,  // res
                               const int64_t* a, uint64_t a_size, uint64_t a_sl   // a
);

EXPORT void vec_znx_add_ref(const MODULE* module,                              // N
                            int64_t* res, uint64_t res_size, uint64_t res_sl,  // res
                            const int64_t* a, uint64_t a_size, uint64_t a_sl,  // a
                            const int64_t* b, uint64_t b_size, uint64_t b_sl   // b
);
EXPORT void vec_znx_add_avx(const MODULE* module,                              // N
                            int64_t* res, uint64_t res_size, uint64_t res_sl,  // res
                            const int64_t* a, uint64_t a_size, uint64_t a_sl,  // a
                            const int64_t* b, uint64_t b_size, uint64_t b_sl   // b
);

EXPORT void vec_znx_sub_ref(const MODULE* module,                             // N
                            int64_t* res, uint64_t res_size, uint64_t res_sl,  // res
                            const int64_t* a, uint64_t a_size, uint64_t a_sl,  // a
                            const int64_t* b, uint64_t b_size, uint64_t b_sl   // b
);

EXPORT void vec_znx_sub_avx(const MODULE* module,                              // N
                            int64_t* res, uint64_t res_size, uint64_t res_sl,  // res
                            const int64_t* a, uint64_t a_size, uint64_t a_sl,  // a
                            const int64_t* b, uint64_t b_size, uint64_t b_sl   // b
);

EXPORT void vec_znx_normalize_base2k_ref(const MODULE* module, uint64_t nn,                              // N
                                         uint64_t log2_base2k,                              // output base 2^K
                                         int64_t* res, uint64_t res_size, uint64_t res_sl,  // res
                                         const int64_t* a, uint64_t a_size, uint64_t a_sl,  // inp
                                         uint8_t* tmp_space                                 // scratch space
);

EXPORT uint64_t vec_znx_normalize_base2k_tmp_bytes_ref(const MODULE* module, uint64_t nn  // N
);

EXPORT void vec_znx_rotate_ref(const MODULE* module,                              // N
                               const int64_t p,                                   // rotation value
                               int64_t* res, uint64_t res_size, uint64_t res_sl,  // res
                               const int64_t* a, uint64_t a_size, uint64_t a_sl   // a
);

EXPORT void vec_znx_automorphism_ref(const MODULE* module,                              // N
                                     const int64_t p,                                   // X->X^p
                                     int64_t* res, uint64_t res_size, uint64_t res_sl,  // res
                                     const int64_t* a, uint64_t a_size, uint64_t a_sl   // a
);

EXPORT void vec_znx_automorphism_dft_ref(const MODULE* module,                 // N
                                         const int64_t p,                      // X->X^p
                                         VEC_ZNX_DFT* res, uint64_t res_size,  // res
                                         const VEC_ZNX_DFT* a, uint64_t a_size, uint8_t* tmp_bytes);

EXPORT void vmp_prepare_ref(const MODULE* module,                              // N
                            VMP_PMAT* pmat,                                     // output
                            const int64_t* mat, uint64_t nrows, uint64_t ncols  // a
);

EXPORT void vmp_apply_dft_ref(const MODULE* module,                                // N
                              VEC_ZNX_DFT* res, uint64_t res_size,                  // res
                              const int64_t* a, uint64_t a_size, uint64_t a_sl,     // a
                              const VMP_PMAT* pmat, uint64_t nrows, uint64_t ncols  // prep matrix
);

EXPORT void vec_dft_zero_ref(const MODULE* module,               // N
                             VEC_ZNX_DFT* res, uint64_t res_size  // res
);

EXPORT void vec_dft_add_ref(const MODULE* module,                  // N
                            VEC_ZNX_DFT* res, uint64_t res_size,    // res
                            const VEC_ZNX_DFT* a, uint64_t a_size,  // a
                            const VEC_ZNX_DFT* b, uint64_t b_size   // b
);

EXPORT void vec_dft_sub_ref(const MODULE* module,                  // N
                            VEC_ZNX_DFT* res, uint64_t res_size,    // res
                            const VEC_ZNX_DFT* a, uint64_t a_size,  // a
                            const VEC_ZNX_DFT* b, uint64_t b_size   // b
);

EXPORT void vec_dft_ref(const MODULE* module,                            // N
                        VEC_ZNX_DFT* res, uint64_t res_size,              // res
                        const int64_t* a, uint64_t a_size, uint64_t a_sl  // a
);

EXPORT void vec_idft_ref(const MODULE* module,                // N
                         VEC_ZNX_BIG* res, uint64_t res_size,  // res
                         const VEC_ZNX_DFT* a_dft, uint64_t a_size);

EXPORT void vec_znx_big_normalize_ref(const MODULE* module,                              // MODULE  
                                      uint64_t nn,                                       // N
                                      uint64_t k,                                        // base-2^k
                                      int64_t* res, uint64_t res_size, uint64_t res_sl,  // res
                                      const VEC_ZNX_BIG* a, uint64_t a_size              // a
);

/** @brief apply a svp product, result = ppol * a, presented in DFT space  */
EXPORT void fft64_svp_apply_dft_ref(const MODULE* module,                             // N
                                    const VEC_ZNX_DFT* res, uint64_t res_size,        // output
                                    const SVP_PPOL* ppol,                             // prepared pol
                                    const int64_t* a, uint64_t a_size, uint64_t a_sl  // a
);

/** @brief apply a svp product, result = ppol * a, presented in DFT space  */
EXPORT void fft64_svp_apply_dft_to_dft_ref(const MODULE* module,  // N
                                           const VEC_ZNX_DFT* res, uint64_t res_size,
                                           uint64_t res_cols,     // output
                                           const SVP_PPOL* ppol,  // prepared pol
                                           const VEC_ZNX_DFT* a, uint64_t a_size,
                                           uint64_t a_cols  // a
);

/** @brief sets res = k-normalize(a) -- output in int64 coeffs space */
EXPORT void fft64_vec_znx_big_normalize_base2k(const MODULE* module,                              // MODULE
                                               uint64_t nn,                                       // N
                                               uint64_t k,                                        // base-2^k
                                               int64_t* res, uint64_t res_size, uint64_t res_sl,  // res
                                               const VEC_ZNX_BIG* a, uint64_t a_size,             // a
                                               uint8_t* tmp_space                                 // temp space
);

/** @brief returns the minimal byte length of scratch space for vec_znx_big_normalize_base2k */
EXPORT uint64_t fft64_vec_znx_big_normalize_base2k_tmp_bytes(const MODULE* module, uint64_t nn  // N

);

/** @brief sets res = k-normalize(a.subrange) -- output in int64 coeffs space */
EXPORT void fft64_vec_znx_big_range_normalize_base2k(const MODULE* module,                              // MODULE
                                                     uint64_t nn,
                                                     uint64_t log2_base2k,                              // base-2^k
                                                     int64_t* res, uint64_t res_size, uint64_t res_sl,  // res
                                                     const VEC_ZNX_BIG* a, uint64_t a_range_begin,      // a
                                                     uint64_t a_range_xend, uint64_t a_range_step,      // range
                                                     uint8_t* tmp_space                                 // temp space
);

/** @brief returns the minimal byte length of scratch space for vec_znx_big_range_normalize_base2k */
EXPORT uint64_t fft64_vec_znx_big_range_normalize_base2k_tmp_bytes(const MODULE* module, uint64_t nn  // N
);

/** @brief returns the minimum size of scratch space requires to apply automorphism in the fourier domain */
EXPORT uint64_t fft64_vec_znx_dft_automorphism_tmp_bytes(const MODULE* module);

/** @brief sets DFT(res) = DFT(a(X^p)) */
EXPORT void fft64_vec_znx_dft_automorphism_ref(const MODULE* module,                   // N
                                               int64_t p,                              // X-X^p
                                               VEC_ZNX_DFT* res, uint64_t res_size,    // res
                                               const VEC_ZNX_DFT* a, uint64_t a_size,  // a
                                               uint8_t* tmp_bytes);

EXPORT void fft64_vec_znx_dft(const MODULE* module,                             // N
                              VEC_ZNX_DFT* res, uint64_t res_size,              // res
                              const int64_t* a, uint64_t a_size, uint64_t a_sl  // a
);

EXPORT void fft64_vec_dft_add(const MODULE* module,                   // N
                              VEC_ZNX_DFT* res, uint64_t res_size,    // res
                              const VEC_ZNX_DFT* a, uint64_t a_size,  // a
                              const VEC_ZNX_DFT* b, uint64_t b_size   // b
);

EXPORT void fft64_vec_dft_sub(const MODULE* module,                   // N
                              VEC_ZNX_DFT* res, uint64_t res_size,    // res
                              const VEC_ZNX_DFT* a, uint64_t a_size,  // a
                              const VEC_ZNX_DFT* b, uint64_t b_size   // b
);

EXPORT void fft64_vec_znx_idft(const MODULE* module,                       // N
                               VEC_ZNX_BIG* res, uint64_t res_size,        // res
                               const VEC_ZNX_DFT* a_dft, uint64_t a_size,  // a
                               uint8_t* tmp                                // scratch space
);

EXPORT uint64_t fft64_vec_znx_idft_tmp_bytes(const MODULE* module);

EXPORT void fft64_vec_znx_idft_tmp_a(const MODULE* module,                 // N
                                     VEC_ZNX_BIG* res, uint64_t res_size,  // res
                                     VEC_ZNX_DFT* a_dft, uint64_t a_size   // a is overwritten
);

EXPORT void ntt120_vec_znx_dft_avx(const MODULE* module,                             // N
                                   VEC_ZNX_DFT* res, uint64_t res_size,              // res
                                   const int64_t* a, uint64_t a_size, uint64_t a_sl  // a
);

/**   */
EXPORT void ntt120_vec_znx_idft_avx(const MODULE* module,                       // N
                                    VEC_ZNX_BIG* res, uint64_t res_size,        // res
                                    const VEC_ZNX_DFT* a_dft, uint64_t a_size,  // a
                                    uint8_t* tmp                                // scratch space
);

EXPORT uint64_t ntt120_vec_znx_idft_tmp_bytes_avx(const MODULE* module);

EXPORT void ntt120_vec_znx_idft_tmp_a_avx(const MODULE* module,                 // N
                                          VEC_ZNX_BIG* res, uint64_t res_size,  // res
                                          VEC_ZNX_DFT* a_dft, uint64_t a_size   // a is overwritten
);

// big additions/subtractions

/** @brief sets res = a+b */
EXPORT void fft64_vec_znx_big_add(const MODULE* module,                   // N
                                  VEC_ZNX_BIG* res, uint64_t res_size,    // res
                                  const VEC_ZNX_BIG* a, uint64_t a_size,  // a
                                  const VEC_ZNX_BIG* b, uint64_t b_size   // b
);
/** @brief sets res = a+b */
EXPORT void fft64_vec_znx_big_add_small(const MODULE* module,                             // N
                                        VEC_ZNX_BIG* res, uint64_t res_size,              // res
                                        const VEC_ZNX_BIG* a, uint64_t a_size,            // a
                                        const int64_t* b, uint64_t b_size, uint64_t b_sl  // b
);
EXPORT void fft64_vec_znx_big_add_small2(const MODULE* module,                              // N
                                         VEC_ZNX_BIG* res, uint64_t res_size,               // res
                                         const int64_t* a, uint64_t a_size, uint64_t a_sl,  // a
                                         const int64_t* b, uint64_t b_size, uint64_t b_sl   // b
);

/** @brief sets res = a-b */
EXPORT void fft64_vec_znx_big_sub(const MODULE* module,                   // N
                                  VEC_ZNX_BIG* res, uint64_t res_size,    // res
                                  const VEC_ZNX_BIG* a, uint64_t a_size,  // a
                                  const VEC_ZNX_BIG* b, uint64_t b_size   // b
);
EXPORT void fft64_vec_znx_big_sub_small_b(const MODULE* module,                             // N
                                          VEC_ZNX_BIG* res, uint64_t res_size,              // res
                                          const VEC_ZNX_BIG* a, uint64_t a_size,            // a
                                          const int64_t* b, uint64_t b_size, uint64_t b_sl  // b
);
EXPORT void fft64_vec_znx_big_sub_small_a(const MODULE* module,                              // N
                                          VEC_ZNX_BIG* res, uint64_t res_size,               // res
                                          const int64_t* a, uint64_t a_size, uint64_t a_sl,  // a
                                          const VEC_ZNX_BIG* b, uint64_t b_size              // b
);
EXPORT void fft64_vec_znx_big_sub_small2(const MODULE* module,                              // N
                                         VEC_ZNX_BIG* res, uint64_t res_size,               // res
                                         const int64_t* a, uint64_t a_size, uint64_t a_sl,  // a
                                         const int64_t* b, uint64_t b_size, uint64_t b_sl   // b
);

/** @brief sets res = a . X^p */
EXPORT void fft64_vec_znx_big_rotate(const MODULE* module,                  // N
                                     int64_t p,                             // rotation value
                                     VEC_ZNX_BIG* res, uint64_t res_size,   // res
                                     const VEC_ZNX_BIG* a, uint64_t a_size  // a
);

/** @brief sets res = a(X^p) */
EXPORT void fft64_vec_znx_big_automorphism(const MODULE* module,                  // N
                                           int64_t p,                             // X-X^p
                                           VEC_ZNX_BIG* res, uint64_t res_size,   // res
                                           const VEC_ZNX_BIG* a, uint64_t a_size  // a
);

/** @brief prepares a svp polynomial  */
EXPORT void fft64_svp_prepare_ref(const MODULE* module,  // N
                                  SVP_PPOL* ppol,        // output
                                  const int64_t* pol     // a
);

/** @brief res = a * b : small integer polynomial product  */
EXPORT void fft64_znx_small_single_product(const MODULE* module,  // N
                                           int64_t* res,          // output
                                           const int64_t* a,      // a
                                           const int64_t* b,      // b
                                           uint8_t* tmp);

/** @brief tmp bytes required for znx_small_single_product  */
EXPORT uint64_t fft64_znx_small_single_product_tmp_bytes(const MODULE* module);

/** @brief prepares a vmp matrix (contiguous row-major version) */
EXPORT void fft64_vmp_prepare_contiguous_ref(const MODULE* module,                                // N
                                             VMP_PMAT* pmat,                                      // output
                                             const int64_t* mat, uint64_t nrows, uint64_t ncols,  // a
                                             uint8_t* tmp_space                                   // scratch space
);

/** @brief extracts the ith-row of a vmp matrix with nrows and ncols */
EXPORT void fft64_vmp_extract_row_dft_ref(const MODULE* module, VEC_ZNX_DFT* res, const VMP_PMAT* pmat, uint64_t row_i,
                                          uint64_t nrows, uint64_t ncols);

/** @brief extracts the ith-row of a vmp matrix with nrows and ncols */
EXPORT void fft64_vmp_extract_row_ref(const MODULE* module, VEC_ZNX_BIG* res, const VMP_PMAT* pmat, uint64_t row_i,
                                      uint64_t nrows, uint64_t ncols);

/** @brief prepares a vmp matrix (contiguous row-major version) */
EXPORT void fft64_vmp_prepare_contiguous_avx(const MODULE* module,                                // N
                                             VMP_PMAT* pmat,                                      // output
                                             const int64_t* mat, uint64_t nrows, uint64_t ncols,  // a
                                             uint8_t* tmp_space                                   // scratch space
);

/** @brief minimal scratch space byte-size required for the vmp_prepare function */
EXPORT uint64_t fft64_vmp_prepare_tmp_bytes(const MODULE* module,  // N
                                            uint64_t nrows, uint64_t ncols);
/** @brief minimal scratch space byte-size required for the vmp_extract function */
EXPORT uint64_t fft64_vmp_extract_tmp_bytes(const MODULE* module,  // N
                                            uint64_t nrows, uint64_t ncols);

/** @brief applies a vmp product (result in DFT space) and adds to res inplace */
EXPORT void fft64_vmp_apply_dft_add_ref(const MODULE* module,                              // N
                                        VEC_ZNX_DFT* res, uint64_t res_size,               // res
                                        const int64_t* a, uint64_t a_size, uint64_t a_sl,  // a
                                        const VMP_PMAT* pmat, uint64_t nrows, uint64_t ncols,
                                        uint64_t pmat_scale,  // prep matrix
                                        uint8_t* tmp_space    // scratch space
);

/** @brief applies a vmp product (result in DFT space) */
EXPORT void fft64_vmp_apply_dft_ref(const MODULE* module,                                  // N
                                    VEC_ZNX_DFT* res, uint64_t res_size,                   // res
                                    const int64_t* a, uint64_t a_size, uint64_t a_sl,      // a
                                    const VMP_PMAT* pmat, uint64_t nrows, uint64_t ncols,  // prep matrix
                                    uint8_t* tmp_space                                     // scratch space
);

/** @brief applies a vmp product (result in DFT space) */
EXPORT void fft64_vmp_apply_dft_avx(const MODULE* module,                                  // N
                                    VEC_ZNX_DFT* res, uint64_t res_size,                   // res
                                    const int64_t* a, uint64_t a_size, uint64_t a_sl,      // a
                                    const VMP_PMAT* pmat, uint64_t nrows, uint64_t ncols,  // prep matrix
                                    uint8_t* tmp_space                                     // scratch space
);

/** @brief applies a vmp product (result in DFT space) and adds to res inplace*/
EXPORT void fft64_vmp_apply_dft_add_avx(const MODULE* module,                              // N
                                        VEC_ZNX_DFT* res, uint64_t res_size,               // res
                                        const int64_t* a, uint64_t a_size, uint64_t a_sl,  // a
                                        const VMP_PMAT* pmat, uint64_t nrows, uint64_t ncols,
                                        uint64_t pmat_scale,  // prep matrix
                                        uint8_t* tmp_space    // scratch space
);

/** @brief this inner function could be very handy */
EXPORT void fft64_vmp_apply_dft_to_dft_ref(const MODULE* module,                       // N
                                           VEC_ZNX_DFT* res, const uint64_t res_size,  // res
                                           const VEC_ZNX_DFT* a_dft, uint64_t a_size,  // a
                                           const VMP_PMAT* pmat, const uint64_t nrows,
                                           const uint64_t ncols,  // prep matrix
                                           uint8_t* tmp_space     // scratch space (a_size*sizeof(reim4) bytes)
);

/** @brief applies rmp product and adds to res inplace */
EXPORT void fft64_vmp_apply_dft_to_dft_add_ref(const MODULE* module,                       // N
                                               VEC_ZNX_DFT* res, const uint64_t res_size,  // res
                                               const VEC_ZNX_DFT* a_dft, uint64_t a_size,  // a
                                               const VMP_PMAT* pmat, const uint64_t nrows, const uint64_t ncols,
                                               uint64_t pmat_scale,  // prep matrix
                                               uint8_t* tmp_space    // scratch space (a_size*sizeof(reim4) bytes)
);

/** @brief this inner function could be very handy */
EXPORT void fft64_vmp_apply_dft_to_dft_avx(const MODULE* module,                       // N
                                           VEC_ZNX_DFT* res, const uint64_t res_size,  // res
                                           const VEC_ZNX_DFT* a_dft, uint64_t a_size,  // a
                                           const VMP_PMAT* pmat, const uint64_t nrows,
                                           const uint64_t ncols,  // prep matrix
                                           uint8_t* tmp_space     // scratch space (a_size*sizeof(reim4) bytes)
);

/** @brief applies rmp product and adds to res inplace */
EXPORT void fft64_vmp_apply_dft_to_dft_add_avx(const MODULE* module,                       // N
                                               VEC_ZNX_DFT* res, const uint64_t res_size,  // res
                                               const VEC_ZNX_DFT* a_dft, uint64_t a_size,  // a
                                               const VMP_PMAT* pmat, const uint64_t nrows, const uint64_t ncols,
                                               uint64_t pmat_scale,  // prep matrix
                                               uint8_t* tmp_space    // scratch space (a_size*sizeof(reim4) bytes)
);

/** @brief minimal size of the tmp_space */
EXPORT uint64_t fft64_vmp_apply_dft_tmp_bytes(const MODULE* module,           // N
                                              uint64_t res_size,              // res
                                              uint64_t a_size,                // a
                                              uint64_t nrows, uint64_t ncols  // prep matrix
);

/** @brief minimal size of the tmp_space */
EXPORT uint64_t fft64_vmp_apply_dft_to_dft_tmp_bytes(const MODULE* module,           // N
                                                     uint64_t res_size,              // res
                                                     uint64_t a_size,                // a
                                                     uint64_t nrows, uint64_t ncols  // prep matrix
);
#endif  // SPQLIOS_VEC_ZNX_ARITHMETIC_PRIVATE_H
