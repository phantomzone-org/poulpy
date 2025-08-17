#ifndef SPQLIOS_CPLX_FFT_INTERNAL_H
#define SPQLIOS_CPLX_FFT_INTERNAL_H

#include "cplx_fft.h"

/** @brief a complex number contains two doubles real,imag */
typedef double CPLX[2];

EXPORT void cplx_set(CPLX r, const CPLX a);
EXPORT void cplx_neg(CPLX r, const CPLX a);
EXPORT void cplx_add(CPLX r, const CPLX a, const CPLX b);
EXPORT void cplx_sub(CPLX r, const CPLX a, const CPLX b);
EXPORT void cplx_mul(CPLX r, const CPLX a, const CPLX b);

/**
 * @brief splits 2h evaluations of one polynomials into 2 times h evaluations of even/odd polynomial
 * Input:  Q_0(y),...,Q_{h-1}(y),Q_0(-y),...,Q_{h-1}(-y)
 * Output: P_0(z),...,P_{h-1}(z),P_h(z),...,P_{2h-1}(z)
 * where Q_i(X)=P_i(X^2)+X.P_{h+i}(X^2) and y^2 = z
 * @param h number of "coefficients" h >= 1
 * @param data 2h complex coefficients interleaved and 256b aligned
 * @param powom y represented as (yre,yim)
 */
EXPORT void cplx_split_fft_ref(int32_t h, CPLX* data, const CPLX powom);
EXPORT void cplx_bisplit_fft_ref(int32_t h, CPLX* data, const CPLX powom[2]);

/**
 * Input: Q(y),Q(-y)
 * Output: P_0(z),P_1(z)
 * where Q(X)=P_0(X^2)+X.P_1(X^2) and y^2 = z
 * @param data 2 complexes coefficients interleaved and 256b aligned
 * @param powom (z,-z) interleaved: (zre,zim,-zre,-zim)
 */
EXPORT void split_fft_last_ref(CPLX* data, const CPLX powom);

EXPORT void cplx_ifft_naive(const uint32_t m, const double entry_pwr, CPLX* data);
EXPORT void cplx_ifft16_avx_fma(void* data, const void* omega);
EXPORT void cplx_ifft16_ref(void* data, const void* omega);

/**
 * @brief compute the ifft evaluations of P in place
 * ifft(data) = ifft_rec(data, i);
 * function ifft_rec(data, omega) {
 *   if #data = 1: return data
 *   let s = sqrt(omega) w. re(s)>0
 *   let (u,v) = data
 *   return split_fft([ifft_rec(u, s), ifft_rec(v, -s)],s)
 * }
 * @param itables precomputed tables (contains all the powers of omega in the order they are used)
 * @param data vector of m complexes (coeffs as input, evals as output)
 */
EXPORT void cplx_ifft_ref(const CPLX_IFFT_PRECOMP* itables, void* data);
EXPORT void cplx_ifft_avx2_fma(const CPLX_IFFT_PRECOMP* itables, void* data);
EXPORT void cplx_fft_naive(const uint32_t m, const double entry_pwr, CPLX* data);
EXPORT void cplx_fft16_avx_fma(void* data, const void* omega);
EXPORT void cplx_fft16_ref(void* data, const void* omega);

/**
 * @brief compute the fft evaluations of P in place
 * fft(data) = fft_rec(data, i);
 * function fft_rec(data, omega) {
 *   if #data = 1: return data
 *   let s = sqrt(omega) w. re(s)>0
 *   let (u,v) = merge_fft(data, s)
 *   return [fft_rec(u, s), fft_rec(v, -s)]
 * }
 * @param tables precomputed tables (contains all the powers of omega in the order they are used)
 * @param data vector of m complexes (coeffs as input, evals as output)
 */
EXPORT void cplx_fft_ref(const CPLX_FFT_PRECOMP* tables, void* data);
EXPORT void cplx_fft_avx2_fma(const CPLX_FFT_PRECOMP* tables, void* data);

/**
 * @brief merges 2 times h evaluations of even/odd polynomials into 2h evaluations of a sigle polynomial
 * Input:  P_0(z),...,P_{h-1}(z),P_h(z),...,P_{2h-1}(z)
 * Output: Q_0(y),...,Q_{h-1}(y),Q_0(-y),...,Q_{h-1}(-y)
 * where Q_i(X)=P_i(X^2)+X.P_{h+i}(X^2) and y^2 = z
 * @param h number of "coefficients" h >= 1
 * @param data 2h complex coefficients interleaved and 256b aligned
 * @param powom y represented as (yre,yim)
 */
EXPORT void cplx_twiddle_fft_ref(int32_t h, CPLX* data, const CPLX powom);

EXPORT void citwiddle(CPLX a, CPLX b, const CPLX om);
EXPORT void ctwiddle(CPLX a, CPLX b, const CPLX om);
EXPORT void invctwiddle(CPLX a, CPLX b, const CPLX ombar);
EXPORT void invcitwiddle(CPLX a, CPLX b, const CPLX ombar);

// CONVERSIONS

/** @brief r = x from ZnX (coeffs as signed int32_t's ) to double */
EXPORT void cplx_from_znx32_ref(const CPLX_FROM_ZNX32_PRECOMP* precomp, void* r, const int32_t* x);
EXPORT void cplx_from_znx32_avx2_fma(const CPLX_FROM_ZNX32_PRECOMP* precomp, void* r, const int32_t* x);
/** @brief r = x to ZnX (coeffs as signed int32_t's ) to double */
EXPORT void cplx_to_znx32_ref(const CPLX_TO_ZNX32_PRECOMP* precomp, int32_t* r, const void* x);
EXPORT void cplx_to_znx32_avx2_fma(const CPLX_TO_ZNX32_PRECOMP* precomp, int32_t* r, const void* x);
/** @brief r = x mod 1 from TnX (coeffs as signed int32_t's) to double */
EXPORT void cplx_from_tnx32_ref(const CPLX_FROM_TNX32_PRECOMP* precomp, void* r, const int32_t* x);
EXPORT void cplx_from_tnx32_avx2_fma(const CPLX_FROM_TNX32_PRECOMP* precomp, void* r, const int32_t* x);
/** @brief r = x mod 1 from TnX (coeffs as signed int32_t's) */
EXPORT void cplx_to_tnx32_ref(const CPLX_TO_TNX32_PRECOMP* precomp, int32_t* x, const void* c);
EXPORT void cplx_to_tnx32_avx2_fma(const CPLX_TO_TNX32_PRECOMP* precomp, int32_t* x, const void* c);
/** @brief r = x from RnX (coeffs as doubles ) to double */
EXPORT void cplx_from_rnx64_ref(const CPLX_FROM_RNX64_PRECOMP* precomp, void* r, const double* x);
EXPORT void cplx_from_rnx64_avx2_fma(const CPLX_FROM_RNX64_PRECOMP* precomp, void* r, const double* x);
/** @brief r = x to RnX (coeffs as doubles ) to double */
EXPORT void cplx_to_rnx64_ref(const CPLX_TO_RNX64_PRECOMP* precomp, double* r, const void* x);
EXPORT void cplx_to_rnx64_avx2_fma(const CPLX_TO_RNX64_PRECOMP* precomp, double* r, const void* x);
/** @brief r = x to integers in RnX (coeffs as doubles ) to double */
EXPORT void cplx_round_to_rnx64_ref(const CPLX_ROUND_TO_RNX64_PRECOMP* precomp, double* r, const void* x);
EXPORT void cplx_round_to_rnx64_avx2_fma(const CPLX_ROUND_TO_RNX64_PRECOMP* precomp, double* r, const void* x);

// fftvec operations
/** @brief element-wise addmul r += ab */
EXPORT void cplx_fftvec_addmul_ref(const CPLX_FFTVEC_ADDMUL_PRECOMP* precomp, void* r, const void* a, const void* b);
EXPORT void cplx_fftvec_addmul_fma(const CPLX_FFTVEC_ADDMUL_PRECOMP* tables, void* r, const void* a, const void* b);
EXPORT void cplx_fftvec_addmul_sse(const CPLX_FFTVEC_ADDMUL_PRECOMP* precomp, void* r, const void* a, const void* b);
EXPORT void cplx_fftvec_addmul_avx512(const CPLX_FFTVEC_ADDMUL_PRECOMP* precomp, void* r, const void* a, const void* b);
/** @brief element-wise mul r = ab */
EXPORT void cplx_fftvec_mul_ref(const CPLX_FFTVEC_MUL_PRECOMP* tables, void* r, const void* a, const void* b);
EXPORT void cplx_fftvec_mul_fma(const CPLX_FFTVEC_MUL_PRECOMP* tables, void* r, const void* a, const void* b);

#endif  // SPQLIOS_CPLX_FFT_INTERNAL_H
