#ifndef SPQLIOS_ZN_ARITHMETIC_PRIVATE_H
#define SPQLIOS_ZN_ARITHMETIC_PRIVATE_H

#include "../commons_private.h"
#include "zn_arithmetic.h"
#include "zn_arithmetic_plugin.h"

typedef struct main_z_module_precomp_t MAIN_Z_MODULE_PRECOMP;
struct main_z_module_precomp_t {
  // TODO
};

typedef union z_module_precomp_t Z_MODULE_PRECOMP;
union z_module_precomp_t {
  MAIN_Z_MODULE_PRECOMP main;
};

void main_init_z_module_precomp(MOD_Z* module);

void main_finalize_z_module_precomp(MOD_Z* module);

/** @brief opaque structure that describes the modules (RnX,ZnX,TnX) and the hardware */
struct z_module_info_t {
  Z_MODULE_TYPE mtype;
  Z_MODULE_VTABLE vtable;
  Z_MODULE_PRECOMP precomp;
  void* custom;
  void (*custom_deleter)(void*);
};

void init_z_module_info(MOD_Z* module, Z_MODULE_TYPE mtype);

void main_init_z_module_vtable(MOD_Z* module);

struct tndbl_approxdecomp_gadget_t {
  uint64_t k;
  uint64_t ell;
  double add_cst;       // 3.2^51-(K.ell) + 1/2.(sum 2^-(i+1)K)
  int64_t and_mask;     // (2^K)-1
  int64_t sub_cst;      // 2^(K-1)
  uint8_t rshifts[64];  // 2^(ell-1-i).K for i in [0:ell-1]
};

/** @brief sets res = gadget_decompose(a) (int8_t* output) */
EXPORT void default_i8_approxdecomp_from_tndbl_ref(const MOD_Z* module,                      // N
                                                   const TNDBL_APPROXDECOMP_GADGET* gadget,  // gadget
                                                   int8_t* res, uint64_t res_size,  // res (in general, size ell.a_size)
                                                   const double* a, uint64_t a_size);  // a

/** @brief sets res = gadget_decompose(a) (int16_t* output) */
EXPORT void default_i16_approxdecomp_from_tndbl_ref(const MOD_Z* module,                      // N
                                                    const TNDBL_APPROXDECOMP_GADGET* gadget,  // gadget
                                                    int16_t* res,
                                                    uint64_t res_size,  // res (in general, size ell.a_size)
                                                    const double* a, uint64_t a_size);  // a
/** @brief sets res = gadget_decompose(a) (int32_t* output) */
EXPORT void default_i32_approxdecomp_from_tndbl_ref(const MOD_Z* module,                      // N
                                                    const TNDBL_APPROXDECOMP_GADGET* gadget,  // gadget
                                                    int32_t* res,
                                                    uint64_t res_size,  // res (in general, size ell.a_size)
                                                    const double* a, uint64_t a_size);  // a

/** @brief prepares a vmp matrix (contiguous row-major version) */
EXPORT void default_zn32_vmp_prepare_contiguous_ref(  //
    const MOD_Z* module,
    ZN32_VMP_PMAT* pmat,                                // output
    const int32_t* mat, uint64_t nrows, uint64_t ncols  // a
);

/** @brief applies a vmp product (int32_t* input) */
EXPORT void default_zn32_vmp_apply_i32_ref(                      //
    const MOD_Z* module,                                         //
    int32_t* res, uint64_t res_size,                             // res
    const int32_t* a, uint64_t a_size,                           // a
    const ZN32_VMP_PMAT* pmat, uint64_t nrows, uint64_t ncols);  // prep matrix

/** @brief applies a vmp product (int16_t* input) */
EXPORT void default_zn32_vmp_apply_i16_ref(                      //
    const MOD_Z* module,                                         // N
    int32_t* res, uint64_t res_size,                             // res
    const int16_t* a, uint64_t a_size,                           // a
    const ZN32_VMP_PMAT* pmat, uint64_t nrows, uint64_t ncols);  // prep matrix

/** @brief applies a vmp product (int8_t* input) */
EXPORT void default_zn32_vmp_apply_i8_ref(                       //
    const MOD_Z* module,                                         // N
    int32_t* res, uint64_t res_size,                             // res
    const int8_t* a, uint64_t a_size,                            // a
    const ZN32_VMP_PMAT* pmat, uint64_t nrows, uint64_t ncols);  // prep matrix

/** @brief applies a vmp product (int32_t* input) */
EXPORT void default_zn32_vmp_apply_i32_avx(                      //
    const MOD_Z* module,                                         //
    int32_t* res, uint64_t res_size,                             // res
    const int32_t* a, uint64_t a_size,                           // a
    const ZN32_VMP_PMAT* pmat, uint64_t nrows, uint64_t ncols);  // prep matrix

/** @brief applies a vmp product (int16_t* input) */
EXPORT void default_zn32_vmp_apply_i16_avx(                      //
    const MOD_Z* module,                                         // N
    int32_t* res, uint64_t res_size,                             // res
    const int16_t* a, uint64_t a_size,                           // a
    const ZN32_VMP_PMAT* pmat, uint64_t nrows, uint64_t ncols);  // prep matrix

/** @brief applies a vmp product (int8_t* input) */
EXPORT void default_zn32_vmp_apply_i8_avx(                       //
    const MOD_Z* module,                                         // N
    int32_t* res, uint64_t res_size,                             // res
    const int8_t* a, uint64_t a_size,                            // a
    const ZN32_VMP_PMAT* pmat, uint64_t nrows, uint64_t ncols);  // prep matrix

// explicit conversions

/** reduction mod 1, output in torus32 space */
EXPORT void dbl_to_tn32_ref(const MOD_Z* module,              //
                            int32_t* res, uint64_t res_size,  // res
                            const double* a, uint64_t a_size  // a
);

/** real centerlift mod 1, output in double space */
EXPORT void tn32_to_dbl_ref(const MOD_Z* module,               //
                            double* res, uint64_t res_size,    // res
                            const int32_t* a, uint64_t a_size  // a
);

/** round to the nearest int, output in i32 space */
EXPORT void dbl_round_to_i32_ref(const MOD_Z* module,              //
                                 int32_t* res, uint64_t res_size,  // res
                                 const double* a, uint64_t a_size  // a
);

/** small int (int32 space) to double */
EXPORT void i32_to_dbl_ref(const MOD_Z* module,               //
                           double* res, uint64_t res_size,    // res
                           const int32_t* a, uint64_t a_size  // a
);

/** round to the nearest int, output in int64 space */
EXPORT void dbl_round_to_i64_ref(const MOD_Z* module,              //
                                 int64_t* res, uint64_t res_size,  // res
                                 const double* a, uint64_t a_size  // a
);

/** small int (int64 space) to double */
EXPORT void i64_to_dbl_ref(const MOD_Z* module,               //
                           double* res, uint64_t res_size,    // res
                           const int64_t* a, uint64_t a_size  // a
);

#endif  // SPQLIOS_ZN_ARITHMETIC_PRIVATE_H
