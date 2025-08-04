#ifndef SPQLIOS_VEC_RNX_ARITHMETIC_H
#define SPQLIOS_VEC_RNX_ARITHMETIC_H

#include <stdint.h>

#include "../commons.h"

/**
 * We support the following module families:
 * - FFT64:
 *    the overall precision should fit at all times over 52 bits.
 */
typedef enum rnx_module_type_t { FFT64 } RNX_MODULE_TYPE;

/** @brief opaque structure that describes the modules (RnX,ZnX,TnX) and the hardware */
typedef struct rnx_module_info_t MOD_RNX;

/**
 * @brief obtain a module info for ring dimension N
 * the module-info knows about:
 *  - the dimension N (or the complex dimension m=N/2)
 *  - any moduleuted fft or ntt items
 *  - the hardware (avx, arm64, x86, ...)
 */
EXPORT MOD_RNX* new_rnx_module_info(uint64_t N, RNX_MODULE_TYPE mode);
EXPORT void delete_rnx_module_info(MOD_RNX* module_info);
EXPORT uint64_t rnx_module_get_n(const MOD_RNX* module);

// basic arithmetic

/** @brief sets res = 0 */
EXPORT void vec_rnx_zero(                            //
    const MOD_RNX* module,                           // N
    double* res, uint64_t res_size, uint64_t res_sl  // res
);

/** @brief sets res = a */
EXPORT void vec_rnx_copy(                             //
    const MOD_RNX* module,                            // N
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl   // a
);

/** @brief sets res = -a */
EXPORT void vec_rnx_negate(                           //
    const MOD_RNX* module,                            // N
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl   // a
);

/** @brief sets res = a + b */
EXPORT void vec_rnx_add(                              //
    const MOD_RNX* module,                            // N
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl,  // a
    const double* b, uint64_t b_size, uint64_t b_sl   // b
);

/** @brief sets res = a - b */
EXPORT void vec_rnx_sub(                              //
    const MOD_RNX* module,                            // N
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl,  // a
    const double* b, uint64_t b_size, uint64_t b_sl   // b
);

/** @brief sets res = a . X^p */
EXPORT void vec_rnx_rotate(                           //
    const MOD_RNX* module,                            // N
    const int64_t p,                                  // rotation value
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl   // a
);

/** @brief sets res = a . (X^p - 1) */
EXPORT void vec_rnx_mul_xp_minus_one(                 //
    const MOD_RNX* module,                            // N
    const int64_t p,                                  // rotation value
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl   // a
);

/** @brief sets res = a(X^p) */
EXPORT void vec_rnx_automorphism(                     //
    const MOD_RNX* module,                            // N
    int64_t p,                                        // X -> X^p
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl   // a
);

///////////////////////////////////////////////////////////////////
//   conversions                                                 //
///////////////////////////////////////////////////////////////////

EXPORT void vec_rnx_to_znx32(                          //
    const MOD_RNX* module,                             // N
    int32_t* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl    // a
);

EXPORT void vec_rnx_from_znx32(                       //
    const MOD_RNX* module,                            // N
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const int32_t* a, uint64_t a_size, uint64_t a_sl  // a
);

EXPORT void vec_rnx_to_tnx32(                          //
    const MOD_RNX* module,                             // N
    int32_t* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl    // a
);

EXPORT void vec_rnx_from_tnx32(                       //
    const MOD_RNX* module,                            // N
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const int32_t* a, uint64_t a_size, uint64_t a_sl  // a
);

EXPORT void vec_rnx_to_tnx32x2(                        //
    const MOD_RNX* module,                             // N
    int32_t* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl    // a
);

EXPORT void vec_rnx_from_tnx32x2(                     //
    const MOD_RNX* module,                            // N
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const int32_t* a, uint64_t a_size, uint64_t a_sl  // a
);

EXPORT void vec_rnx_to_tnxdbl(                        //
    const MOD_RNX* module,                            // N
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl   // a
);

///////////////////////////////////////////////////////////////////
//   isolated products (n.log(n), but not particularly optimized //
///////////////////////////////////////////////////////////////////

/** @brief res = a * b : small polynomial product  */
EXPORT void rnx_small_single_product(  //
    const MOD_RNX* module,             // N
    double* res,                       // output
    const double* a,                   // a
    const double* b,                   // b
    uint8_t* tmp);                     // scratch space

EXPORT uint64_t rnx_small_single_product_tmp_bytes(const MOD_RNX* module);

/** @brief res = a * b centermod 1: small polynomial product  */
EXPORT void tnxdbl_small_single_product(  //
    const MOD_RNX* module,                // N
    double* torus_res,                    // output
    const double* int_a,                  // a
    const double* torus_b,                // b
    uint8_t* tmp);                        // scratch space

EXPORT uint64_t tnxdbl_small_single_product_tmp_bytes(const MOD_RNX* module);

/** @brief res = a * b: small polynomial product  */
EXPORT void znx32_small_single_product(  //
    const MOD_RNX* module,               // N
    int32_t* int_res,                    // output
    const int32_t* int_a,                // a
    const int32_t* int_b,                // b
    uint8_t* tmp);                       // scratch space

EXPORT uint64_t znx32_small_single_product_tmp_bytes(const MOD_RNX* module);

/** @brief res = a * b centermod 1: small polynomial product  */
EXPORT void tnx32_small_single_product(  //
    const MOD_RNX* module,               // N
    int32_t* torus_res,                  // output
    const int32_t* int_a,                // a
    const int32_t* torus_b,              // b
    uint8_t* tmp);                       // scratch space

EXPORT uint64_t tnx32_small_single_product_tmp_bytes(const MOD_RNX* module);

///////////////////////////////////////////////////////////////////
//   prepared gadget decompositions (optimized)                  //
///////////////////////////////////////////////////////////////////

// decompose from tnx32

typedef struct tnx32_approxdecomp_gadget_t TNX32_APPROXDECOMP_GADGET;

/** @brief new gadget: delete with delete_tnx32_approxdecomp_gadget */
EXPORT TNX32_APPROXDECOMP_GADGET* new_tnx32_approxdecomp_gadget(  //
    const MOD_RNX* module,                                        // N
    uint64_t k, uint64_t ell                                      // base 2^K and size
);
EXPORT void delete_tnx32_approxdecomp_gadget(const MOD_RNX* module);

/** @brief sets res = gadget_decompose(a) */
EXPORT void rnx_approxdecomp_from_tnx32(              //
    const MOD_RNX* module,                            // N
    const TNX32_APPROXDECOMP_GADGET* gadget,          // output base 2^K
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const int32_t* a                                  // a
);

// decompose from tnx32x2

typedef struct tnx32x2_approxdecomp_gadget_t TNX32X2_APPROXDECOMP_GADGET;

/** @brief new gadget: delete with delete_tnx32x2_approxdecomp_gadget */
EXPORT TNX32X2_APPROXDECOMP_GADGET* new_tnx32x2_approxdecomp_gadget(const MOD_RNX* module, uint64_t ka, uint64_t ella,
                                                                    uint64_t kb, uint64_t ellb);
EXPORT void delete_tnx32x2_approxdecomp_gadget(const MOD_RNX* module);

/** @brief sets res = gadget_decompose(a) */
EXPORT void rnx_approxdecomp_from_tnx32x2(            //
    const MOD_RNX* module,                            // N
    const TNX32X2_APPROXDECOMP_GADGET* gadget,        // output base 2^K
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const int32_t* a                                  // a
);

// decompose from tnxdbl

typedef struct tnxdbl_approxdecomp_gadget_t TNXDBL_APPROXDECOMP_GADGET;

/** @brief new gadget: delete with delete_tnxdbl_approxdecomp_gadget */
EXPORT TNXDBL_APPROXDECOMP_GADGET* new_tnxdbl_approxdecomp_gadget(  //
    const MOD_RNX* module,                                          // N
    uint64_t k, uint64_t ell                                        // base 2^K and size
);
EXPORT void delete_tnxdbl_approxdecomp_gadget(TNXDBL_APPROXDECOMP_GADGET* gadget);

/** @brief sets res = gadget_decompose(a) */
EXPORT void rnx_approxdecomp_from_tnxdbl(             //
    const MOD_RNX* module,                            // N
    const TNXDBL_APPROXDECOMP_GADGET* gadget,         // output base 2^K
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a);                                 // a

///////////////////////////////////////////////////////////////////
//   prepared scalar-vector product (optimized)                  //
///////////////////////////////////////////////////////////////////

/** @brief opaque type that represents a polynomial of RnX prepared for a scalar-vector product */
typedef struct rnx_svp_ppol_t RNX_SVP_PPOL;

/** @brief number of bytes in a RNX_VMP_PMAT (for manual allocation) */
EXPORT uint64_t bytes_of_rnx_svp_ppol(const MOD_RNX* module);  // N

/** @brief allocates a prepared vector (release with delete_rnx_svp_ppol) */
EXPORT RNX_SVP_PPOL* new_rnx_svp_ppol(const MOD_RNX* module);  // N

/** @brief frees memory for a prepared vector */
EXPORT void delete_rnx_svp_ppol(RNX_SVP_PPOL* res);

/** @brief prepares a svp polynomial  */
EXPORT void rnx_svp_prepare(const MOD_RNX* module,  // N
                            RNX_SVP_PPOL* ppol,     // output
                            const double* pol       // a
);

/** @brief apply a svp product, result = ppol * a, presented in DFT space */
EXPORT void rnx_svp_apply(                            //
    const MOD_RNX* module,                            // N
    double* res, uint64_t res_size, uint64_t res_sl,  // output
    const RNX_SVP_PPOL* ppol,                         // prepared pol
    const double* a, uint64_t a_size, uint64_t a_sl   // a
);

///////////////////////////////////////////////////////////////////
//   prepared vector-matrix product (optimized)                               //
///////////////////////////////////////////////////////////////////

typedef struct rnx_vmp_pmat_t RNX_VMP_PMAT;

/** @brief number of bytes in a RNX_VMP_PMAT (for manual allocation) */
EXPORT uint64_t bytes_of_rnx_vmp_pmat(const MOD_RNX* module,            // N
                                      uint64_t nrows, uint64_t ncols);  // dimensions

/** @brief allocates a prepared matrix (release with delete_rnx_vmp_pmat) */
EXPORT RNX_VMP_PMAT* new_rnx_vmp_pmat(const MOD_RNX* module,            // N
                                      uint64_t nrows, uint64_t ncols);  // dimensions
EXPORT void delete_rnx_vmp_pmat(RNX_VMP_PMAT* ptr);

/** @brief prepares a vmp matrix (contiguous row-major version) */
EXPORT void rnx_vmp_prepare_contiguous(               //
    const MOD_RNX* module,                            // N
    RNX_VMP_PMAT* pmat,                               // output
    const double* a, uint64_t nrows, uint64_t ncols,  // a
    uint8_t* tmp_space                                // scratch space
);

/** @brief prepares a vmp matrix (mat[row]+col*N points to the item) */
EXPORT void rnx_vmp_prepare_dblptr(                    //
    const MOD_RNX* module,                             // N
    RNX_VMP_PMAT* pmat,                                // output
    const double** a, uint64_t nrows, uint64_t ncols,  // a
    uint8_t* tmp_space                                 // scratch space
);

/** @brief prepares the ith-row of a vmp matrix with nrows and ncols */
EXPORT void rnx_vmp_prepare_row(                                      //
    const MOD_RNX* module,                                            // N
    RNX_VMP_PMAT* pmat,                                               // output
    const double* a, uint64_t row_i, uint64_t nrows, uint64_t ncols,  // a
    uint8_t* tmp_space                                                // scratch space
);

/** @brief number of scratch bytes necessary to prepare a matrix */
EXPORT uint64_t rnx_vmp_prepare_tmp_bytes(const MOD_RNX* module);

/** @brief applies a vmp product res = a x pmat */
EXPORT void rnx_vmp_apply_tmp_a(                               //
    const MOD_RNX* module,                                     // N
    double* res, uint64_t res_size, uint64_t res_sl,           // res
    double* tmpa, uint64_t a_size, uint64_t a_sl,              // a (will be overwritten)
    const RNX_VMP_PMAT* pmat, uint64_t nrows, uint64_t ncols,  // prep matrix
    uint8_t* tmp_space                                         // scratch space
);

EXPORT uint64_t rnx_vmp_apply_tmp_a_tmp_bytes(  //
    const MOD_RNX* module,                      // N
    uint64_t res_size,                          // res size
    uint64_t a_size,                            // a size
    uint64_t nrows, uint64_t ncols              // prep matrix dims
);

/** @brief minimal size of the tmp_space */
EXPORT void rnx_vmp_apply_dft_to_dft(                          //
    const MOD_RNX* module,                                     // N
    double* res, uint64_t res_size, uint64_t res_sl,           // res
    const double* a_dft, uint64_t a_size, uint64_t a_sl,       // a
    const RNX_VMP_PMAT* pmat, uint64_t nrows, uint64_t ncols,  // prep matrix
    uint8_t* tmp_space                                         // scratch space (a_size*sizeof(reim4) bytes)
);

/** @brief minimal size of the tmp_space */
EXPORT uint64_t rnx_vmp_apply_dft_to_dft_tmp_bytes(  //
    const MOD_RNX* module,                           // N
    uint64_t res_size,                               // res
    uint64_t a_size,                                 // a
    uint64_t nrows, uint64_t ncols                   // prep matrix
);

/** @brief sets res = DFT(a) */
EXPORT void vec_rnx_dft(const MOD_RNX* module,                            // N
                        double* res, uint64_t res_size, uint64_t res_sl,  // res
                        const double* a, uint64_t a_size, uint64_t a_sl   // a
);

/** @brief sets res = iDFT(a_dft) -- idft is not normalized */
EXPORT void vec_rnx_idft(const MOD_RNX* module,                               // N
                         double* res, uint64_t res_size, uint64_t res_sl,     // res
                         const double* a_dft, uint64_t a_size, uint64_t a_sl  // a
);

#endif  // SPQLIOS_VEC_RNX_ARITHMETIC_H
