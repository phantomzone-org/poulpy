#ifndef SPQLIOS_CPLX_FFT_H
#define SPQLIOS_CPLX_FFT_H

#include "../commons.h"

typedef struct cplx_fft_precomp CPLX_FFT_PRECOMP;
typedef struct cplx_ifft_precomp CPLX_IFFT_PRECOMP;
typedef struct cplx_mul_precomp CPLX_FFTVEC_MUL_PRECOMP;
typedef struct cplx_addmul_precomp CPLX_FFTVEC_ADDMUL_PRECOMP;
typedef struct cplx_from_znx32_precomp CPLX_FROM_ZNX32_PRECOMP;
typedef struct cplx_from_tnx32_precomp CPLX_FROM_TNX32_PRECOMP;
typedef struct cplx_to_tnx32_precomp CPLX_TO_TNX32_PRECOMP;
typedef struct cplx_to_znx32_precomp CPLX_TO_ZNX32_PRECOMP;
typedef struct cplx_from_rnx64_precomp CPLX_FROM_RNX64_PRECOMP;
typedef struct cplx_to_rnx64_precomp CPLX_TO_RNX64_PRECOMP;
typedef struct cplx_round_to_rnx64_precomp CPLX_ROUND_TO_RNX64_PRECOMP;

/**
 * @brief precomputes fft tables.
 * The FFT tables contains a constant section that is required for efficient FFT operations in dimension nn.
 * The resulting pointer is to be passed as "tables" argument to any call to the fft function.
 * The user can optionnally allocate zero or more computation buffers, which are scratch spaces that are contiguous to
 * the constant tables in memory, and allow for more efficient operations. It is the user's responsibility to ensure
 * that each of those buffers are never used simultaneously by two ffts on different threads at the same time. The fft
 * table must be deleted by delete_fft_precomp after its last usage.
 */
EXPORT CPLX_FFT_PRECOMP* new_cplx_fft_precomp(uint32_t m, uint32_t num_buffers);

/**
 * @brief gets the address of a fft buffer allocated during new_fft_precomp.
 * This buffer can be used as data pointer in subsequent calls to fft,
 * and does not need to be released afterwards.
 */
EXPORT void* cplx_fft_precomp_get_buffer(const CPLX_FFT_PRECOMP* tables, uint32_t buffer_index);

/**
 * @brief allocates a new fft buffer.
 * This buffer can be used as data pointer in subsequent calls to fft,
 * and must be deleted afterwards by calling delete_fft_buffer.
 */
EXPORT void* new_cplx_fft_buffer(uint32_t m);

/**
 * @brief allocates a new fft buffer.
 * This buffer can be used as data pointer in subsequent calls to fft,
 * and must be deleted afterwards by calling delete_fft_buffer.
 */
EXPORT void delete_cplx_fft_buffer(void* buffer);

/**
 * @brief deallocates a fft table and all its built-in buffers.
 */
#define delete_cplx_fft_precomp free

/**
 * @brief computes a direct fft in-place over data.
 */
EXPORT void cplx_fft(const CPLX_FFT_PRECOMP* tables, void* data);

EXPORT CPLX_IFFT_PRECOMP* new_cplx_ifft_precomp(uint32_t m, uint32_t num_buffers);
EXPORT void* cplx_ifft_precomp_get_buffer(const CPLX_IFFT_PRECOMP* tables, uint32_t buffer_index);
EXPORT void cplx_ifft(const CPLX_IFFT_PRECOMP* tables, void* data);
#define delete_cplx_ifft_precomp free

EXPORT CPLX_FFTVEC_MUL_PRECOMP* new_cplx_fftvec_mul_precomp(uint32_t m);
EXPORT void cplx_fftvec_mul(const CPLX_FFTVEC_MUL_PRECOMP* tables, void* r, const void* a, const void* b);
#define delete_cplx_fftvec_mul_precomp free

EXPORT CPLX_FFTVEC_ADDMUL_PRECOMP* new_cplx_fftvec_addmul_precomp(uint32_t m);
EXPORT void cplx_fftvec_addmul(const CPLX_FFTVEC_ADDMUL_PRECOMP* tables, void* r, const void* a, const void* b);
#define delete_cplx_fftvec_addmul_precomp free

/**
 * @brief prepares a conversion from ZnX to the cplx layout.
 * All the coefficients must be strictly lower than 2^log2bound in absolute value. Any attempt to use
 * this function on a larger coefficient is undefined behaviour. The resulting precomputed data must
 * be freed with `new_cplx_from_znx32_precomp`
 * @param m the target complex dimension m from C[X] mod X^m-i. Note that the inputs have n=2m
 * int32 coefficients in natural order modulo X^n+1
 * @param log2bound bound on the input coefficients. Must be between 0 and 32
 */
EXPORT CPLX_FROM_ZNX32_PRECOMP* new_cplx_from_znx32_precomp(uint32_t m);
/**
 * @brief converts from ZnX to the cplx layout.
 * @param tables precomputed data obtained by new_cplx_from_znx32_precomp.
 * @param r resulting array of m complexes coefficients mod X^m-i
 * @param x input array of n bounded integer coefficients mod X^n+1
 */
EXPORT void cplx_from_znx32(const CPLX_FROM_ZNX32_PRECOMP* tables, void* r, const int32_t* a);
/** @brief frees a precomputed conversion data initialized with new_cplx_from_znx32_precomp. */
#define delete_cplx_from_znx32_precomp free

/**
 * @brief prepares a conversion from TnX to the cplx layout.
 * @param m the target complex dimension m from C[X] mod X^m-i. Note that the inputs have n=2m
 * torus32 coefficients. The resulting precomputed data must
 * be freed with `delete_cplx_from_tnx32_precomp`
 */
EXPORT CPLX_FROM_TNX32_PRECOMP* new_cplx_from_tnx32_precomp(uint32_t m);
/**
 * @brief converts from TnX to the cplx layout.
 * @param tables precomputed data obtained by new_cplx_from_tnx32_precomp.
 * @param r resulting array of m complexes coefficients mod X^m-i
 * @param x input array of n torus32 coefficients mod X^n+1
 */
EXPORT void cplx_from_tnx32(const CPLX_FROM_TNX32_PRECOMP* tables, void* r, const int32_t* a);
/** @brief frees a precomputed conversion data initialized with new_cplx_from_tnx32_precomp. */
#define delete_cplx_from_tnx32_precomp free

/**
 * @brief prepares a rescale and conversion from the cplx layout to TnX.
 * @param m the target complex dimension m from C[X] mod X^m-i. Note that the outputs have n=2m
 * torus32 coefficients.
 * @param divisor must be a power of two. The inputs are rescaled by divisor before being reduced modulo 1.
 * Remember that the output of an iFFT must be divided by m.
 * @param log2overhead all inputs absolute values must be within divisor.2^log2overhead.
 * For any inputs outside of these bounds, the conversion is undefined behaviour.
 * The maximum supported log2overhead is 52, and the algorithm is faster for log2overhead=18.
 */
EXPORT CPLX_TO_TNX32_PRECOMP* new_cplx_to_tnx32_precomp(uint32_t m, double divisor, uint32_t log2overhead);
/**
 * @brief rescale, converts and reduce mod 1 from cplx layout to torus32.
 * @param tables precomputed data obtained by new_cplx_from_tnx32_precomp.
 * @param r resulting array of n torus32 coefficients mod X^n+1
 * @param x input array of m cplx coefficients mod X^m-i
 */
EXPORT void cplx_to_tnx32(const CPLX_TO_TNX32_PRECOMP* tables, int32_t* r, const void* a);
#define delete_cplx_to_tnx32_precomp free

EXPORT CPLX_TO_ZNX32_PRECOMP* new_cplx_to_znx32_precomp(uint32_t m, double divisor);
EXPORT void cplx_to_znx32(const CPLX_TO_ZNX32_PRECOMP* precomp, int32_t* r, const void* x);
#define delete_cplx_to_znx32_simple free

EXPORT CPLX_FROM_RNX64_PRECOMP* new_cplx_from_rnx64_simple(uint32_t m);
EXPORT void cplx_from_rnx64(const CPLX_FROM_RNX64_PRECOMP* precomp, void* r, const double* x);
#define delete_cplx_from_rnx64_simple free

EXPORT CPLX_TO_RNX64_PRECOMP* new_cplx_to_rnx64(uint32_t m, double divisor);
EXPORT void cplx_to_rnx64(const CPLX_TO_RNX64_PRECOMP* precomp, double* r, const void* x);
#define delete_cplx_round_to_rnx64_simple free

EXPORT CPLX_ROUND_TO_RNX64_PRECOMP* new_cplx_round_to_rnx64(uint32_t m, double divisor, uint32_t log2bound);
EXPORT void cplx_round_to_rnx64(const CPLX_ROUND_TO_RNX64_PRECOMP* precomp, double* r, const void* x);
#define delete_cplx_round_to_rnx64_simple free

/**
 * @brief Simpler API for the fft function.
 * For each dimension, the precomputed tables for this dimension are generated automatically.
 * It is advised to do one dry-run per desired dimension before using in a multithread environment */
EXPORT void cplx_fft_simple(uint32_t m, void* data);
/**
 * @brief Simpler API for the ifft function.
 * For each dimension, the precomputed tables for this dimension are generated automatically the first time.
 * It is advised to do one dry-run call per desired dimension in the main thread before using in a multithread
 * environment */
EXPORT void cplx_ifft_simple(uint32_t m, void* data);
/**
 * @brief Simpler API for the fftvec multiplication function.
 * For each dimension, the precomputed tables for this dimension are generated automatically the first time.
 * It is advised to do one dry-run call per desired dimension before using in a multithread environment */
EXPORT void cplx_fftvec_mul_simple(uint32_t m, void* r, const void* a, const void* b);
/**
 * @brief Simpler API for the fftvec addmul function.
 * For each dimension, the precomputed tables for this dimension are generated automatically the first time.
 * It is advised to do one dry-run call per desired dimension before using in a multithread environment */
EXPORT void cplx_fftvec_addmul_simple(uint32_t m, void* r, const void* a, const void* b);
/**
 * @brief Simpler API for the znx32 to cplx conversion.
 * For each dimension, the precomputed tables for this dimension are generated automatically the first time.
 * It is advised to do one dry-run call per desired dimension before using in a multithread environment */
EXPORT void cplx_from_znx32_simple(uint32_t m, void* r, const int32_t* x);
/**
 * @brief Simpler API for the tnx32 to cplx conversion.
 * For each dimension, the precomputed tables for this dimension are generated automatically the first time.
 * It is advised to do one dry-run call per desired dimension before using in a multithread environment */
EXPORT void cplx_from_tnx32_simple(uint32_t m, void* r, const int32_t* x);
/**
 * @brief Simpler API for the cplx to tnx32 conversion.
 * For each dimension, the precomputed tables for this dimension are generated automatically the first time.
 * It is advised to do one dry-run call per desired dimension before using in a multithread environment */
EXPORT void cplx_to_tnx32_simple(uint32_t m, double divisor, uint32_t log2overhead, int32_t* r, const void* x);

/**
 * @brief converts, divides and round from cplx to znx32 (simple API)
 * @param m the complex dimension
 * @param divisor the divisor: a power of two, often m after an ifft
 * @param r the result: must be a double array of size 2m. r must be distinct from x
 * @param x the input: must hold m complex numbers.
 */
EXPORT void cplx_to_znx32_simple(uint32_t m, double divisor, int32_t* r, const void* x);

/**
 * @brief converts from rnx64 to cplx (simple API)
 * The bound on the output is assumed to be within ]2^-31,2^31[.
 * Any coefficient that would fall outside this range is undefined behaviour.
 * @param m the complex dimension
 * @param r the result: must be an array of m complex numbers. r must be distinct from x
 * @param x the input: must be an array of 2m doubles.
 */
EXPORT void cplx_from_rnx64_simple(uint32_t m, void* r, const double* x);

/**
 * @brief converts, divides from cplx to rnx64 (simple API)
 * @param m the complex dimension
 * @param divisor the divisor: a power of two, often m after an ifft
 * @param r the result: must be a double array of size 2m. r must be distinct from x
 * @param x the input: must hold m complex numbers.
 */
EXPORT void cplx_to_rnx64_simple(uint32_t m, double divisor, double* r, const void* x);

/**
 * @brief converts, divides and round to integer from cplx to rnx32 (simple API)
 * @param m the complex dimension
 * @param divisor the divisor: a power of two, often m after an ifft
 * @param log2bound a guarantee on the log2bound of the output. log2bound<=48 will use a more efficient algorithm.
 * @param r the result: must be a double array of size 2m. r must be distinct from x
 * @param x the input: must hold m complex numbers.
 */
EXPORT void cplx_round_to_rnx64_simple(uint32_t m, double divisor, uint32_t log2bound, double* r, const void* x);

#endif  // SPQLIOS_CPLX_FFT_H
