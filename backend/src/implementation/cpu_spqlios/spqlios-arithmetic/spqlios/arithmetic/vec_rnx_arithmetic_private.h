#ifndef SPQLIOS_VEC_RNX_ARITHMETIC_PRIVATE_H
#define SPQLIOS_VEC_RNX_ARITHMETIC_PRIVATE_H

#include "../commons_private.h"
#include "../reim/reim_fft.h"
#include "vec_rnx_arithmetic.h"
#include "vec_rnx_arithmetic_plugin.h"

typedef struct fft64_rnx_module_precomp_t FFT64_RNX_MODULE_PRECOMP;
struct fft64_rnx_module_precomp_t {
  REIM_FFT_PRECOMP* p_fft;
  REIM_IFFT_PRECOMP* p_ifft;
  REIM_FFTVEC_ADD_PRECOMP* p_fftvec_add;
  REIM_FFTVEC_MUL_PRECOMP* p_fftvec_mul;
  REIM_FFTVEC_ADDMUL_PRECOMP* p_fftvec_addmul;
};

typedef union rnx_module_precomp_t RNX_MODULE_PRECOMP;
union rnx_module_precomp_t {
  FFT64_RNX_MODULE_PRECOMP fft64;
};

void fft64_init_rnx_module_precomp(MOD_RNX* module);

void fft64_finalize_rnx_module_precomp(MOD_RNX* module);

/** @brief opaque structure that describes the modules (RnX,ZnX,TnX) and the hardware */
struct rnx_module_info_t {
  uint64_t n;
  uint64_t m;
  RNX_MODULE_TYPE mtype;
  RNX_MODULE_VTABLE vtable;
  RNX_MODULE_PRECOMP precomp;
  void* custom;
  void (*custom_deleter)(void*);
};

void init_rnx_module_info(MOD_RNX* module,  //
                          uint64_t, RNX_MODULE_TYPE mtype);

void finalize_rnx_module_info(MOD_RNX* module);

void fft64_init_rnx_module_vtable(MOD_RNX* module);

///////////////////////////////////////////////////////////////////
//   prepared gadget decompositions (optimized)                  //
///////////////////////////////////////////////////////////////////

struct tnx32_approxdec_gadget_t {
  uint64_t k;
  uint64_t ell;
  int32_t add_cst;      // 1/2.(sum 2^-(i+1)K)
  int32_t rshift_base;  // 32 - K
  int64_t and_mask;     // 2^K-1
  int64_t or_mask;      // double(2^52)
  double sub_cst;       // double(2^52 + 2^(K-1))
  uint8_t rshifts[8];   // 32 - (i+1).K
};

struct tnx32x2_approxdec_gadget_t {
  // TODO
};

struct tnxdbl_approxdecomp_gadget_t {
  uint64_t k;
  uint64_t ell;
  double add_cst;     // double(3.2^(51-ell.K) + 1/2.(sum 2^(-iK)) for i=[0,ell[)
  uint64_t and_mask;  // uint64(2^(K)-1)
  uint64_t or_mask;   // double(2^52)
  double sub_cst;     // double(2^52 + 2^(K-1))
};

EXPORT void vec_rnx_add_ref(                          //
    const MOD_RNX* module,                            // N
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl,  // a
    const double* b, uint64_t b_size, uint64_t b_sl   // b
);
EXPORT void vec_rnx_add_avx(                          //
    const MOD_RNX* module,                            // N
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl,  // a
    const double* b, uint64_t b_size, uint64_t b_sl   // b
);

/** @brief sets res = 0 */
EXPORT void vec_rnx_zero_ref(                        //
    const MOD_RNX* module,                           // N
    double* res, uint64_t res_size, uint64_t res_sl  // res
);

/** @brief sets res = a */
EXPORT void vec_rnx_copy_ref(                         //
    const MOD_RNX* module,                            // N
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl   // a
);

/** @brief sets res = -a */
EXPORT void vec_rnx_negate_ref(                       //
    const MOD_RNX* module,                            // N
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl   // a
);

/** @brief sets res = -a */
EXPORT void vec_rnx_negate_avx(                       //
    const MOD_RNX* module,                            // N
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl   // a
);

/** @brief sets res = a - b */
EXPORT void vec_rnx_sub_ref(                          //
    const MOD_RNX* module,                            // N
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl,  // a
    const double* b, uint64_t b_size, uint64_t b_sl   // b
);

/** @brief sets res = a - b */
EXPORT void vec_rnx_sub_avx(                          //
    const MOD_RNX* module,                            // N
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl,  // a
    const double* b, uint64_t b_size, uint64_t b_sl   // b
);

/** @brief sets res = a . X^p */
EXPORT void vec_rnx_rotate_ref(                       //
    const MOD_RNX* module,                            // N
    const int64_t p,                                  // rotation value
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl   // a
);

/** @brief sets res = a(X^p) */
EXPORT void vec_rnx_automorphism_ref(                 //
    const MOD_RNX* module,                            // N
    int64_t p,                                        // X -> X^p
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl   // a
);

/** @brief number of bytes in a RNX_VMP_PMAT (for manual allocation) */
EXPORT uint64_t fft64_bytes_of_rnx_vmp_pmat(const MOD_RNX* module,  // N
                                            uint64_t nrows, uint64_t ncols);

EXPORT void fft64_rnx_vmp_apply_dft_to_dft_ref(                //
    const MOD_RNX* module,                                     // N
    double* res, uint64_t res_size, uint64_t res_sl,           // res
    const double* a_dft, uint64_t a_size, uint64_t a_sl,       // a
    const RNX_VMP_PMAT* pmat, uint64_t nrows, uint64_t ncols,  // prep matrix
    uint8_t* tmp_space                                         // scratch space (a_size*sizeof(reim4) bytes)
);
EXPORT void fft64_rnx_vmp_apply_dft_to_dft_avx(                //
    const MOD_RNX* module,                                     // N
    double* res, uint64_t res_size, uint64_t res_sl,           // res
    const double* a_dft, uint64_t a_size, uint64_t a_sl,       // a
    const RNX_VMP_PMAT* pmat, uint64_t nrows, uint64_t ncols,  // prep matrix
    uint8_t* tmp_space                                         // scratch space (a_size*sizeof(reim4) bytes)
);
EXPORT uint64_t fft64_rnx_vmp_apply_dft_to_dft_tmp_bytes_ref(  //
    const MOD_RNX* module,                                     // N
    uint64_t res_size,                                         // res
    uint64_t a_size,                                           // a
    uint64_t nrows, uint64_t ncols                             // prep matrix
);
EXPORT uint64_t fft64_rnx_vmp_apply_dft_to_dft_tmp_bytes_avx(  //
    const MOD_RNX* module,                                     // N
    uint64_t res_size,                                         // res
    uint64_t a_size,                                           // a
    uint64_t nrows, uint64_t ncols                             // prep matrix
);
EXPORT void fft64_rnx_vmp_prepare_contiguous_ref(       //
    const MOD_RNX* module,                              // N
    RNX_VMP_PMAT* pmat,                                 // output
    const double* mat, uint64_t nrows, uint64_t ncols,  // a
    uint8_t* tmp_space                                  // scratch space
);
EXPORT void fft64_rnx_vmp_prepare_contiguous_avx(       //
    const MOD_RNX* module,                              // N
    RNX_VMP_PMAT* pmat,                                 // output
    const double* mat, uint64_t nrows, uint64_t ncols,  // a
    uint8_t* tmp_space                                  // scratch space
);
EXPORT void fft64_rnx_vmp_prepare_dblptr_ref(            //
    const MOD_RNX* module,                               // N
    RNX_VMP_PMAT* pmat,                                  // output
    const double** mat, uint64_t nrows, uint64_t ncols,  // a
    uint8_t* tmp_space                                   // scratch space
);
EXPORT void fft64_rnx_vmp_prepare_dblptr_avx(            //
    const MOD_RNX* module,                               // N
    RNX_VMP_PMAT* pmat,                                  // output
    const double** mat, uint64_t nrows, uint64_t ncols,  // a
    uint8_t* tmp_space                                   // scratch space
);
EXPORT void fft64_rnx_vmp_prepare_row_ref(                              //
    const MOD_RNX* module,                                              // N
    RNX_VMP_PMAT* pmat,                                                 // output
    const double* mat, uint64_t row_i, uint64_t nrows, uint64_t ncols,  // a
    uint8_t* tmp_space                                                  // scratch space
);
EXPORT void fft64_rnx_vmp_prepare_row_avx(                              //
    const MOD_RNX* module,                                              // N
    RNX_VMP_PMAT* pmat,                                                 // output
    const double* mat, uint64_t row_i, uint64_t nrows, uint64_t ncols,  // a
    uint8_t* tmp_space                                                  // scratch space
);
EXPORT uint64_t fft64_rnx_vmp_prepare_tmp_bytes_ref(const MOD_RNX* module);
EXPORT uint64_t fft64_rnx_vmp_prepare_tmp_bytes_avx(const MOD_RNX* module);

EXPORT void fft64_rnx_vmp_apply_tmp_a_ref(                     //
    const MOD_RNX* module,                                     // N
    double* res, uint64_t res_size, uint64_t res_sl,           // res (addr must be != a)
    double* tmpa, uint64_t a_size, uint64_t a_sl,              // a (will be overwritten)
    const RNX_VMP_PMAT* pmat, uint64_t nrows, uint64_t ncols,  // prep matrix
    uint8_t* tmp_space                                         // scratch space
);
EXPORT void fft64_rnx_vmp_apply_tmp_a_avx(                     //
    const MOD_RNX* module,                                     // N
    double* res, uint64_t res_size, uint64_t res_sl,           // res (addr must be != a)
    double* tmpa, uint64_t a_size, uint64_t a_sl,              // a (will be overwritten)
    const RNX_VMP_PMAT* pmat, uint64_t nrows, uint64_t ncols,  // prep matrix
    uint8_t* tmp_space                                         // scratch space
);

EXPORT uint64_t fft64_rnx_vmp_apply_tmp_a_tmp_bytes_ref(  //
    const MOD_RNX* module,                                // N
    uint64_t res_size,                                    // res
    uint64_t a_size,                                      // a
    uint64_t nrows, uint64_t ncols                        // prep matrix
);
EXPORT uint64_t fft64_rnx_vmp_apply_tmp_a_tmp_bytes_avx(  //
    const MOD_RNX* module,                                // N
    uint64_t res_size,                                    // res
    uint64_t a_size,                                      // a
    uint64_t nrows, uint64_t ncols                        // prep matrix
);

/// gadget decompositions

/** @brief sets res = gadget_decompose(a) */
EXPORT void rnx_approxdecomp_from_tnxdbl_ref(         //
    const MOD_RNX* module,                            // N
    const TNXDBL_APPROXDECOMP_GADGET* gadget,         // output base 2^K
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a);                                 // a
EXPORT void rnx_approxdecomp_from_tnxdbl_avx(         //
    const MOD_RNX* module,                            // N
    const TNXDBL_APPROXDECOMP_GADGET* gadget,         // output base 2^K
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a);                                 // a

EXPORT void vec_rnx_mul_xp_minus_one_ref(             //
    const MOD_RNX* module,                            // N
    const int64_t p,                                  // rotation value
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl   // a
);

EXPORT void vec_rnx_to_znx32_ref(                      //
    const MOD_RNX* module,                             // N
    int32_t* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl    // a
);

EXPORT void vec_rnx_from_znx32_ref(                   //
    const MOD_RNX* module,                            // N
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const int32_t* a, uint64_t a_size, uint64_t a_sl  // a
);

EXPORT void vec_rnx_to_tnx32_ref(                      //
    const MOD_RNX* module,                             // N
    int32_t* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl    // a
);

EXPORT void vec_rnx_from_tnx32_ref(                   //
    const MOD_RNX* module,                            // N
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const int32_t* a, uint64_t a_size, uint64_t a_sl  // a
);

EXPORT void vec_rnx_to_tnxdbl_ref(                    //
    const MOD_RNX* module,                            // N
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl   // a
);

EXPORT uint64_t fft64_bytes_of_rnx_svp_ppol(const MOD_RNX* module);  // N

/** @brief prepares a svp polynomial  */
EXPORT void fft64_rnx_svp_prepare_ref(const MOD_RNX* module,  // N
                                      RNX_SVP_PPOL* ppol,     // output
                                      const double* pol       // a
);

/** @brief apply a svp product, result = ppol * a, presented in DFT space */
EXPORT void fft64_rnx_svp_apply_ref(                  //
    const MOD_RNX* module,                            // N
    double* res, uint64_t res_size, uint64_t res_sl,  // output
    const RNX_SVP_PPOL* ppol,                         // prepared pol
    const double* a, uint64_t a_size, uint64_t a_sl   // a
);

#endif  // SPQLIOS_VEC_RNX_ARITHMETIC_PRIVATE_H
