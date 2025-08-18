#ifndef SPQLIOS_COEFFS_ARITHMETIC_H
#define SPQLIOS_COEFFS_ARITHMETIC_H

#include "../commons.h"

/** res = a + b */
EXPORT void znx_add_i64_ref(uint64_t nn, int64_t* res, const int64_t* a, const int64_t* b);
EXPORT void znx_add_i64_avx(uint64_t nn, int64_t* res, const int64_t* a, const int64_t* b);
/** res = a - b */
EXPORT void znx_sub_i64_ref(uint64_t nn, int64_t* res, const int64_t* a, const int64_t* b);
EXPORT void znx_sub_i64_avx(uint64_t nn, int64_t* res, const int64_t* a, const int64_t* b);
/** res = -a */
EXPORT void znx_negate_i64_ref(uint64_t nn, int64_t* res, const int64_t* a);
EXPORT void znx_negate_i64_avx(uint64_t nn, int64_t* res, const int64_t* a);
/** res = a */
EXPORT void znx_copy_i64_ref(uint64_t nn, int64_t* res, const int64_t* a);
/** res = 0 */
EXPORT void znx_zero_i64_ref(uint64_t nn, int64_t* res);

/** res = a / m where m is a power of 2 */
EXPORT void rnx_divide_by_m_ref(uint64_t nn, double m, double* res, const double* a);
EXPORT void rnx_divide_by_m_avx(uint64_t nn, double m, double* res, const double* a);

/**
 * @param res = X^p *in mod X^nn +1
 * @param nn the ring dimension
 * @param p a power for the rotation -2nn <= p <= 2nn
 * @param in is a rnx/znx vector of dimension nn
 */
EXPORT void rnx_rotate_f64(uint64_t nn, int64_t p, double* res, const double* in);
EXPORT void znx_rotate_i64(uint64_t nn, int64_t p, int64_t* res, const int64_t* in);
EXPORT void rnx_rotate_inplace_f64(uint64_t nn, int64_t p, double* res);
EXPORT void znx_rotate_inplace_i64(uint64_t nn, int64_t p, int64_t* res);

/**
 * @brief res(X) = in(X^p)
 * @param nn the ring dimension
 * @param p is odd integer and must be between 0 < p < 2nn
 * @param in is a rnx/znx vector of dimension nn
 */
EXPORT void rnx_automorphism_f64(uint64_t nn, int64_t p, double* res, const double* in);
EXPORT void znx_automorphism_i64(uint64_t nn, int64_t p, int64_t* res, const int64_t* in);
EXPORT void rnx_automorphism_inplace_f64(uint64_t nn, int64_t p, double* res);
EXPORT void znx_automorphism_inplace_i64(uint64_t nn, int64_t p, int64_t* res);

/**
 * @brief res = (X^p-1).in
 * @param nn the ring dimension
 * @param p must be between -2nn <= p <= 2nn
 * @param in is a rnx/znx vector of dimension nn
 */
EXPORT void rnx_mul_xp_minus_one_f64(uint64_t nn, int64_t p, double* res, const double* in);
EXPORT void znx_mul_xp_minus_one_i64(uint64_t nn, int64_t p, int64_t* res, const int64_t* in);
EXPORT void rnx_mul_xp_minus_one_inplace_f64(uint64_t nn, int64_t p, double* res);
EXPORT void znx_mul_xp_minus_one_inplace_i64(uint64_t nn, int64_t p, int64_t* res);

/**
 * @brief      Normalize input plus carry mod-2^k. The following
 *             equality holds @c {in + carry_in == out + carry_out . 2^k}.
 *
 *             @c in must be in [-2^62 .. 2^62]
 *
 *             @c out is in [ -2^(base_k-1), 2^(base_k-1) [.
 *
 *             @c carry_in and @carry_out have at most 64+1-k bits.
 *
 *             Null @c carry_in or @c carry_out are ignored.
 *
 * @param[in]  nn         the ring dimension
 * @param[in]  base_k     the base k
 * @param      out        output normalized znx
 * @param      carry_out  output carry znx
 * @param[in]  in         input znx
 * @param[in]  carry_in   input carry znx
 */
EXPORT void znx_normalize(uint64_t nn, uint64_t base_k, int64_t* out, int64_t* carry_out, const int64_t* in,
                          const int64_t* carry_in);

#endif  // SPQLIOS_COEFFS_ARITHMETIC_H
