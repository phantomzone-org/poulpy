#ifndef SPQLIOS_VEC_ZNX_ARITHMETIC_H
#define SPQLIOS_VEC_ZNX_ARITHMETIC_H

#include <stdint.h>

#include "../commons.h"
#include "../reim/reim_fft.h"

/**
 * We support the following module families:
 * - FFT64:
 *    all the polynomials should fit at all times over 52 bits.
 *    for FHE implementations, the recommended limb-sizes are
 *    between K=10 and 20, which is good for low multiplicative depths.
 * - NTT120:
 *    all the polynomials should fit at all times over 119 bits.
 *    for FHE implementations, the recommended limb-sizes are
 *    between K=20 and 40, which is good for large multiplicative depths.
 */
typedef enum module_type_t { FFT64, NTT120 } MODULE_TYPE;

/** @brief opaque structure that describr the modules (ZnX,TnX) and the hardware */
typedef struct module_info_t MODULE;
/** @brief opaque type that represents a prepared matrix */
typedef struct vmp_pmat_t VMP_PMAT;
/** @brief opaque type that represents a vector of znx in DFT space */
typedef struct vec_znx_dft_t VEC_ZNX_DFT;
/** @brief opaque type that represents a vector of znx in large coeffs space */
typedef struct vec_znx_bigcoeff_t VEC_ZNX_BIG;
/** @brief opaque type that represents a prepared scalar vector product */
typedef struct svp_ppol_t SVP_PPOL;
/** @brief opaque type that represents a prepared left convolution vector product */
typedef struct cnv_pvec_l_t CNV_PVEC_L;
/** @brief opaque type that represents a prepared right convolution vector product */
typedef struct cnv_pvec_r_t CNV_PVEC_R;

/** @brief bytes needed for a vec_znx in DFT space */
EXPORT uint64_t bytes_of_vec_znx_dft(const MODULE* module,  // N
                                     uint64_t size);

/** @brief allocates a vec_znx in DFT space */
EXPORT VEC_ZNX_DFT* new_vec_znx_dft(const MODULE* module,  // N
                                    uint64_t size);

/** @brief frees memory from a vec_znx in DFT space */
EXPORT void delete_vec_znx_dft(VEC_ZNX_DFT* res);

/** @brief bytes needed for a vec_znx_big */
EXPORT uint64_t bytes_of_vec_znx_big(const MODULE* module,  // N
                                     uint64_t size);

/** @brief allocates a vec_znx_big */
EXPORT VEC_ZNX_BIG* new_vec_znx_big(const MODULE* module,  // N
                                    uint64_t size);
/** @brief frees memory from a vec_znx_big */
EXPORT void delete_vec_znx_big(VEC_ZNX_BIG* res);

/** @brief bytes needed for a prepared vector */
EXPORT uint64_t bytes_of_svp_ppol(const MODULE* module);  // N

/** @brief allocates a prepared vector */
EXPORT SVP_PPOL* new_svp_ppol(const MODULE* module);  // N

/** @brief frees memory for a prepared vector */
EXPORT void delete_svp_ppol(SVP_PPOL* res);

/** @brief bytes needed for a prepared matrix */
EXPORT uint64_t bytes_of_vmp_pmat(const MODULE* module,  // N
                                  uint64_t nrows, uint64_t ncols);

/** @brief allocates a prepared matrix */
EXPORT VMP_PMAT* new_vmp_pmat(const MODULE* module,  // N
                              uint64_t nrows, uint64_t ncols);

/** @brief frees memory for a prepared matrix */
EXPORT void delete_vmp_pmat(VMP_PMAT* res);

/**
 * @brief obtain a module info for ring dimension N
 * the module-info knows about:
 *  - the dimension N (or the complex dimension m=N/2)
 *  - any moduleuted fft or ntt items
 *  - the hardware (avx, arm64, x86, ...)
 */
EXPORT MODULE* new_module_info(uint64_t N, MODULE_TYPE mode);
EXPORT void delete_module_info(MODULE* module_info);
EXPORT uint64_t module_get_n(const MODULE* module);

/** @brief sets res = 0 */
EXPORT void vec_znx_zero(const MODULE* module,                             // N
                         int64_t* res, uint64_t res_size, uint64_t res_sl  // res
);

/** @brief sets res = a */
EXPORT void vec_znx_copy(const MODULE* module,                              // N
                         int64_t* res, uint64_t res_size, uint64_t res_sl,  // res
                         const int64_t* a, uint64_t a_size, uint64_t a_sl   // a
);

/** @brief sets res = a */
EXPORT void vec_znx_negate(const MODULE* module,                              // N
                           int64_t* res, uint64_t res_size, uint64_t res_sl,  // res
                           const int64_t* a, uint64_t a_size, uint64_t a_sl   // a
);

/** @brief sets res = a + b */
EXPORT void vec_znx_add(const MODULE* module,                              // N
                        int64_t* res, uint64_t res_size, uint64_t res_sl,  // res
                        const int64_t* a, uint64_t a_size, uint64_t a_sl,  // a
                        const int64_t* b, uint64_t b_size, uint64_t b_sl   // b
);

/** @brief sets res = a - b */
EXPORT void vec_znx_sub(const MODULE* module,                              // N
                        int64_t* res, uint64_t res_size, uint64_t res_sl,  // res
                        const int64_t* a, uint64_t a_size, uint64_t a_sl,  // a
                        const int64_t* b, uint64_t b_size, uint64_t b_sl   // b
);

/** @brief sets res = k-normalize-reduce(a) */
EXPORT void vec_znx_normalize_base2k(const MODULE* module,                              // MODULE
                                     uint64_t nn,                                       // N
                                     uint64_t log2_base2k,                              // output base 2^K
                                     int64_t* res, uint64_t res_size, uint64_t res_sl,  // res
                                     const int64_t* a, uint64_t a_size, uint64_t a_sl,  // a
                                     uint8_t* tmp_space                                 // scratch space (size >= N)
);

/** @brief returns the minimal byte length of scratch space for vec_znx_normalize_base2k */
EXPORT uint64_t vec_znx_normalize_base2k_tmp_bytes(const MODULE* module, uint64_t nn  // N
);

/** @brief sets res = a . X^p */
EXPORT void vec_znx_rotate(const MODULE* module,                              // N
                           const int64_t p,                                   // rotation value
                           int64_t* res, uint64_t res_size, uint64_t res_sl,  // res
                           const int64_t* a, uint64_t a_size, uint64_t a_sl   // a
);

/** @brief sets res = a(X^p) */
EXPORT void vec_znx_automorphism(const MODULE* module,                              // N
                                 const int64_t p,                                   // X-X^p
                                 int64_t* res, uint64_t res_size, uint64_t res_sl,  // res
                                 const int64_t* a, uint64_t a_size, uint64_t a_sl   // a
);

/** @brief returns the minimum scratch space to apply an automorphism in DFT */
EXPORT uint64_t vec_znx_dft_automorphism_tmp_bytes(const MODULE* module);

/** @brief sets DFT(res) = DFT(a(X^p)) */
EXPORT void vec_znx_dft_automorphism(const MODULE* module,                   // N
                                     const int64_t p,                        // X-X^p
                                     VEC_ZNX_DFT* res, uint64_t res_size,    // res
                                     const VEC_ZNX_DFT* a, uint64_t a_size,  // a
                                     uint8_t* tmp_bytes);

/** @brief sets res = 0 */
EXPORT void vec_dft_zero(const MODULE* module,                // N
                         VEC_ZNX_DFT* res, uint64_t res_size  // res
);

/** @brief sets res = a+b */
EXPORT void vec_dft_add(const MODULE* module,                   // N
                        VEC_ZNX_DFT* res, uint64_t res_size,    // res
                        const VEC_ZNX_DFT* a, uint64_t a_size,  // a
                        const VEC_ZNX_DFT* b, uint64_t b_size   // b
);

/** @brief sets res = a-b */
EXPORT void vec_dft_sub(const MODULE* module,                   // N
                        VEC_ZNX_DFT* res, uint64_t res_size,    // res
                        const VEC_ZNX_DFT* a, uint64_t a_size,  // a
                        const VEC_ZNX_DFT* b, uint64_t b_size   // b
);

/** @brief sets res = DFT(a) */
EXPORT void vec_znx_dft(const MODULE* module,                             // N
                        VEC_ZNX_DFT* res, uint64_t res_size,              // res
                        const int64_t* a, uint64_t a_size, uint64_t a_sl  // a
);

/** @brief sets res = iDFT(a_dft) -- output in big coeffs space */
EXPORT void vec_znx_idft(const MODULE* module,                       // N
                         VEC_ZNX_BIG* res, uint64_t res_size,        // res
                         const VEC_ZNX_DFT* a_dft, uint64_t a_size,  // a
                         uint8_t* tmp                                // scratch space
);

/** @brief tmp bytes required for vec_znx_idft  */
EXPORT uint64_t vec_znx_idft_tmp_bytes(const MODULE* module);

/**
 * @brief      sets res = iDFT(a_dft) -- output in big coeffs space
 *
 * @note       a_dft is overwritten
 */
EXPORT void vec_znx_idft_tmp_a(const MODULE* module,                 // N
                               VEC_ZNX_BIG* res, uint64_t res_size,  // res
                               VEC_ZNX_DFT* a_dft, uint64_t a_size   // a is overwritten
);

/** @brief sets res = a+b */
EXPORT void vec_znx_big_add(const MODULE* module,                   // N
                            VEC_ZNX_BIG* res, uint64_t res_size,    // res
                            const VEC_ZNX_BIG* a, uint64_t a_size,  // a
                            const VEC_ZNX_BIG* b, uint64_t b_size   // b
);
/** @brief sets res = a+b */
EXPORT void vec_znx_big_add_small(const MODULE* module,                             // N
                                  VEC_ZNX_BIG* res, uint64_t res_size,              // res
                                  const VEC_ZNX_BIG* a, uint64_t a_size,            // a
                                  const int64_t* b, uint64_t b_size, uint64_t b_sl  // b
);
EXPORT void vec_znx_big_add_small2(const MODULE* module,                              // N
                                   VEC_ZNX_BIG* res, uint64_t res_size,               // res
                                   const int64_t* a, uint64_t a_size, uint64_t a_sl,  // a
                                   const int64_t* b, uint64_t b_size, uint64_t b_sl   // b
);

/** @brief sets res = a-b */
EXPORT void vec_znx_big_sub(const MODULE* module,                   // N
                            VEC_ZNX_BIG* res, uint64_t res_size,    // res
                            const VEC_ZNX_BIG* a, uint64_t a_size,  // a
                            const VEC_ZNX_BIG* b, uint64_t b_size   // b
);
EXPORT void vec_znx_big_sub_small_b(const MODULE* module,                             // N
                                    VEC_ZNX_BIG* res, uint64_t res_size,              // res
                                    const VEC_ZNX_BIG* a, uint64_t a_size,            // a
                                    const int64_t* b, uint64_t b_size, uint64_t b_sl  // b
);
EXPORT void vec_znx_big_sub_small_a(const MODULE* module,                              // N
                                    VEC_ZNX_BIG* res, uint64_t res_size,               // res
                                    const int64_t* a, uint64_t a_size, uint64_t a_sl,  // a
                                    const VEC_ZNX_BIG* b, uint64_t b_size              // b
);
EXPORT void vec_znx_big_sub_small2(const MODULE* module,                              // N
                                   VEC_ZNX_BIG* res, uint64_t res_size,               // res
                                   const int64_t* a, uint64_t a_size, uint64_t a_sl,  // a
                                   const int64_t* b, uint64_t b_size, uint64_t b_sl   // b
);

/** @brief sets res = k-normalize(a) -- output in int64 coeffs space */
EXPORT void vec_znx_big_normalize_base2k(const MODULE* module,                              // MODULE
                                         uint64_t nn,                                       // N
                                         uint64_t log2_base2k,                              // base-2^k
                                         int64_t* res, uint64_t res_size, uint64_t res_sl,  // res
                                         const VEC_ZNX_BIG* a, uint64_t a_size,             // a
                                         uint8_t* tmp_space                                 // temp space
);

/** @brief returns the minimal byte length of scratch space for vec_znx_big_normalize_base2k */
EXPORT uint64_t vec_znx_big_normalize_base2k_tmp_bytes(const MODULE* module, uint64_t nn  // N
);

/** @brief sets res = k-normalize(a.subrange) -- output in int64 coeffs space */
EXPORT void vec_znx_big_range_normalize_base2k(                                                  //
    const MODULE* module,                                                                        // MODULE
    uint64_t nn,
    uint64_t log2_base2k,                                                                        // base-2^k
    int64_t* res, uint64_t res_size, uint64_t res_sl,                                            // res
    const VEC_ZNX_BIG* a, uint64_t a_range_begin, uint64_t a_range_xend, uint64_t a_range_step,  // range
    uint8_t* tmp_space                                                                           // temp space
);

/** @brief returns the minimal byte length of scratch space for vec_znx_big_range_normalize_base2k */
EXPORT uint64_t vec_znx_big_range_normalize_base2k_tmp_bytes(  //
    const MODULE* module, uint64_t nn                                       // N
);

/** @brief sets res = a . X^p */
EXPORT void vec_znx_big_rotate(const MODULE* module,                  // N
                               int64_t p,                             // rotation value
                               VEC_ZNX_BIG* res, uint64_t res_size,   // res
                               const VEC_ZNX_BIG* a, uint64_t a_size  // a
);

/** @brief sets res = a(X^p) */
EXPORT void vec_znx_big_automorphism(const MODULE* module,                  // N
                                     int64_t p,                             // X-X^p
                                     VEC_ZNX_BIG* res, uint64_t res_size,   // res
                                     const VEC_ZNX_BIG* a, uint64_t a_size  // a
);

/** @brief apply a svp product, result = ppol * a, presented in DFT space */
EXPORT void svp_apply_dft(const MODULE* module,                             // N
                          const VEC_ZNX_DFT* res, uint64_t res_size,        // output
                          const SVP_PPOL* ppol,                             // prepared pol
                          const int64_t* a, uint64_t a_size, uint64_t a_sl  // a
);

/** @brief apply a svp product, result = ppol * a, presented in DFT space */
EXPORT void svp_apply_dft_to_dft(const MODULE* module,  // N
                                 const VEC_ZNX_DFT* res, uint64_t res_size,
                                 uint64_t res_cols,                                      // output
                                 const SVP_PPOL* ppol,                                   // prepared pol
                                 const VEC_ZNX_DFT* a, uint64_t a_size, uint64_t a_cols  // a
);

/** @brief prepares a svp polynomial  */
EXPORT void svp_prepare(const MODULE* module,  // N
                        SVP_PPOL* ppol,        // output
                        const int64_t* pol     // a
);

/** @brief res = a * b : small integer polynomial product  */
EXPORT void znx_small_single_product(const MODULE* module,  // N
                                     int64_t* res,          // output
                                     const int64_t* a,      // a
                                     const int64_t* b,      // b
                                     uint8_t* tmp);

/** @brief tmp bytes required for znx_small_single_product  */
EXPORT uint64_t znx_small_single_product_tmp_bytes(const MODULE* module);

/** @brief minimal scratch space byte-size required for the vmp_prepare function */
EXPORT uint64_t vmp_prepare_tmp_bytes(const MODULE* module,  // N
                                      uint64_t nrows, uint64_t ncols);

/** @brief minimal scratch space byte-size required for the vmp_extract function */
EXPORT uint64_t vmp_extract_tmp_bytes(const MODULE* module,  // N
                                      uint64_t nrows, uint64_t ncols);

/** @brief prepares a vmp matrix (contiguous row-major version) */
EXPORT void vmp_prepare_contiguous(const MODULE* module,                                // N
                                   VMP_PMAT* pmat,                                      // output
                                   const int64_t* mat, uint64_t nrows, uint64_t ncols,  // a
                                   uint8_t* tmp_space                                   // scratch space
);

/** @brief extracts the ith-row of a vmp matrix with nrows and ncols */
EXPORT void vmp_extract_row(const MODULE* module, VEC_ZNX_BIG* res, const VMP_PMAT* pmat, uint64_t row_i,
                            uint64_t nrows, uint64_t ncols);

/** @brief extracts the ith-row of a vmp matrix with nrows and ncols */
EXPORT void vmp_extract_row_dft(const MODULE* module, VEC_ZNX_DFT* res, const VMP_PMAT* pmat, uint64_t row_i,
                                uint64_t nrows, uint64_t ncols);

/** @brief applies a vmp product (result in DFT space) adds to res inplace */
EXPORT void vmp_apply_dft_add(const MODULE* module,                                                       // N
                              VEC_ZNX_DFT* res, uint64_t res_size,                                        // res
                              const int64_t* a, uint64_t a_size, uint64_t a_sl,                           // a
                              const VMP_PMAT* pmat, uint64_t nrows, uint64_t ncols, uint64_t pmat_scale,  // prep matrix
                              uint8_t* tmp_space  // scratch space
);

/** @brief applies a vmp product (result in DFT space) */
EXPORT void vmp_apply_dft(const MODULE* module,                                  // N
                          VEC_ZNX_DFT* res, uint64_t res_size,                   // res
                          const int64_t* a, uint64_t a_size, uint64_t a_sl,      // a
                          const VMP_PMAT* pmat, uint64_t nrows, uint64_t ncols,  // prep matrix
                          uint8_t* tmp_space                                     // scratch space
);

/** @brief minimal size of the tmp_space */
EXPORT uint64_t vmp_apply_dft_tmp_bytes(const MODULE* module,           // N
                                        uint64_t res_size,              // res
                                        uint64_t a_size,                // a
                                        uint64_t nrows, uint64_t ncols  // prep matrix
);

/** @brief applies vmp product */
EXPORT void vmp_apply_dft_to_dft(const MODULE* module,                       // N
                                 VEC_ZNX_DFT* res, const uint64_t res_size,  // res
                                 const VEC_ZNX_DFT* a_dft, uint64_t a_size,  // a
                                 const VMP_PMAT* pmat, const uint64_t nrows,
                                 const uint64_t ncols,  // prep matrix
                                 uint8_t* tmp_space     // scratch space (a_size*sizeof(reim4) bytes)
);

/** @brief applies vmp product and adds to res inplace */
EXPORT void vmp_apply_dft_to_dft_add(const MODULE* module,                       // N
                                     VEC_ZNX_DFT* res, const uint64_t res_size,  // res
                                     const VEC_ZNX_DFT* a_dft, uint64_t a_size,  // a
                                     const VMP_PMAT* pmat, const uint64_t nrows, const uint64_t ncols,
                                     const uint64_t pmat_scale,  // prep matrix
                                     uint8_t* tmp_space          // scratch space (a_size*sizeof(reim4) bytes)
);

/** @brief minimal size of the tmp_space */
EXPORT uint64_t vmp_apply_dft_to_dft_tmp_bytes(const MODULE* module,           // N
                                               uint64_t res_size,              // res
                                               uint64_t a_size,                // a
                                               uint64_t nrows, uint64_t ncols  // prep matrix
);
#endif  // SPQLIOS_VEC_ZNX_ARITHMETIC_H
