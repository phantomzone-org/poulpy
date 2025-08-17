#ifndef SPQLIOS_ZN_ARITHMETIC_H
#define SPQLIOS_ZN_ARITHMETIC_H

#include <stdint.h>

#include "../commons.h"

typedef enum z_module_type_t { DEFAULT } Z_MODULE_TYPE;

/** @brief opaque structure that describes the module and the hardware */
typedef struct z_module_info_t MOD_Z;

/**
 * @brief obtain a module info for ring dimension N
 * the module-info knows about:
 *  - the dimension N (or the complex dimension m=N/2)
 *  - any moduleuted fft or ntt items
 *  - the hardware (avx, arm64, x86, ...)
 */
EXPORT MOD_Z* new_z_module_info(Z_MODULE_TYPE mode);
EXPORT void delete_z_module_info(MOD_Z* module_info);

typedef struct tndbl_approxdecomp_gadget_t TNDBL_APPROXDECOMP_GADGET;

EXPORT TNDBL_APPROXDECOMP_GADGET* new_tndbl_approxdecomp_gadget(const MOD_Z* module,  //
                                                                uint64_t k,
                                                                uint64_t ell);  // base 2^k, and size

EXPORT void delete_tndbl_approxdecomp_gadget(TNDBL_APPROXDECOMP_GADGET* ptr);

/** @brief sets res = gadget_decompose(a) (int8_t* output) */
EXPORT void i8_approxdecomp_from_tndbl(const MOD_Z* module,                      // N
                                       const TNDBL_APPROXDECOMP_GADGET* gadget,  // gadget
                                       int8_t* res, uint64_t res_size,           // res (in general, size ell.a_size)
                                       const double* a, uint64_t a_size);        // a

/** @brief sets res = gadget_decompose(a) (int16_t* output) */
EXPORT void i16_approxdecomp_from_tndbl(const MOD_Z* module,                      // N
                                        const TNDBL_APPROXDECOMP_GADGET* gadget,  // gadget
                                        int16_t* res, uint64_t res_size,          // res (in general, size ell.a_size)
                                        const double* a, uint64_t a_size);        // a
/** @brief sets res = gadget_decompose(a) (int32_t* output) */
EXPORT void i32_approxdecomp_from_tndbl(const MOD_Z* module,                      // N
                                        const TNDBL_APPROXDECOMP_GADGET* gadget,  // gadget
                                        int32_t* res, uint64_t res_size,          // res (in general, size ell.a_size)
                                        const double* a, uint64_t a_size);        // a

/** @brief opaque type that represents a prepared matrix */
typedef struct zn32_vmp_pmat_t ZN32_VMP_PMAT;

/** @brief size in bytes of a prepared matrix (for custom allocation) */
EXPORT uint64_t bytes_of_zn32_vmp_pmat(const MOD_Z* module,              // N
                                       uint64_t nrows, uint64_t ncols);  // dimensions

/** @brief allocates a prepared matrix (release with delete_zn32_vmp_pmat) */
EXPORT ZN32_VMP_PMAT* new_zn32_vmp_pmat(const MOD_Z* module,              // N
                                        uint64_t nrows, uint64_t ncols);  // dimensions

/** @brief deletes a prepared matrix (release with free) */
EXPORT void delete_zn32_vmp_pmat(ZN32_VMP_PMAT* ptr);  // dimensions

/** @brief prepares a vmp matrix (contiguous row-major version) */
EXPORT void zn32_vmp_prepare_contiguous(  //
    const MOD_Z* module,
    ZN32_VMP_PMAT* pmat,                                  // output
    const int32_t* mat, uint64_t nrows, uint64_t ncols);  // a

/** @brief prepares a vmp matrix (mat[row]+col*N points to the item) */
EXPORT void zn32_vmp_prepare_dblptr(  //
    const MOD_Z* module,
    ZN32_VMP_PMAT* pmat,                                   // output
    const int32_t** mat, uint64_t nrows, uint64_t ncols);  // a

/** @brief prepares a vmp matrix (mat[row]+col*N points to the item) */
EXPORT void zn32_vmp_prepare_row(  //
    const MOD_Z* module,
    ZN32_VMP_PMAT* pmat,                                                  // output
    const int32_t* row, uint64_t row_i, uint64_t nrows, uint64_t ncols);  // a

/** @brief applies a vmp product (int32_t* input) */
EXPORT void zn32_vmp_apply_i32(                                  //
    const MOD_Z* module,                                         //
    int32_t* res, uint64_t res_size,                             // res
    const int32_t* a, uint64_t a_size,                           // a
    const ZN32_VMP_PMAT* pmat, uint64_t nrows, uint64_t ncols);  // prep matrix

/** @brief applies a vmp product (int16_t* input) */
EXPORT void zn32_vmp_apply_i16(                                  //
    const MOD_Z* module,                                         //
    int32_t* res, uint64_t res_size,                             // res
    const int16_t* a, uint64_t a_size,                           // a
    const ZN32_VMP_PMAT* pmat, uint64_t nrows, uint64_t ncols);  // prep matrix

/** @brief applies a vmp product (int8_t* input) */
EXPORT void zn32_vmp_apply_i8(                                   //
    const MOD_Z* module,                                         //
    int32_t* res, uint64_t res_size,                             // res
    const int8_t* a, uint64_t a_size,                            // a
    const ZN32_VMP_PMAT* pmat, uint64_t nrows, uint64_t ncols);  // prep matrix

// explicit conversions

/** reduction mod 1, output in torus32 space */
EXPORT void dbl_to_tn32(const MOD_Z* module,              //
                        int32_t* res, uint64_t res_size,  // res
                        const double* a, uint64_t a_size  // a
);

/** real centerlift mod 1, output in double space */
EXPORT void tn32_to_dbl(const MOD_Z* module,               //
                        double* res, uint64_t res_size,    // res
                        const int32_t* a, uint64_t a_size  // a
);

/** round to the nearest int, output in i32 space.
 * WARNING: ||a||_inf must be <= 2^18 in this function
 */
EXPORT void dbl_round_to_i32(const MOD_Z* module,              //
                             int32_t* res, uint64_t res_size,  // res
                             const double* a, uint64_t a_size  // a
);

/** small int (int32 space) to double
 * WARNING: ||a||_inf must be <= 2^18 in this function
 */
EXPORT void i32_to_dbl(const MOD_Z* module,               //
                       double* res, uint64_t res_size,    // res
                       const int32_t* a, uint64_t a_size  // a
);

/** round to the nearest int, output in int64 space
 * WARNING: ||a||_inf must be <= 2^50 in this function
 */
EXPORT void dbl_round_to_i64(const MOD_Z* module,              //
                             int64_t* res, uint64_t res_size,  // res
                             const double* a, uint64_t a_size  // a
);

/** small int (int64 space, <= 2^50) to double
 * WARNING: ||a||_inf must be <= 2^50 in this function
 */
EXPORT void i64_to_dbl(const MOD_Z* module,               //
                       double* res, uint64_t res_size,    // res
                       const int64_t* a, uint64_t a_size  // a
);

#endif  // SPQLIOS_ZN_ARITHMETIC_H
