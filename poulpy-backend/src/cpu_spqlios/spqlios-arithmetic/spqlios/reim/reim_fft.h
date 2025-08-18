#ifndef SPQLIOS_REIM_FFT_H
#define SPQLIOS_REIM_FFT_H

#include "../commons.h"

typedef struct reim_fft_precomp REIM_FFT_PRECOMP;
typedef struct reim_ifft_precomp REIM_IFFT_PRECOMP;
typedef struct reim_add_precomp REIM_FFTVEC_ADD_PRECOMP;
typedef struct reim_sub_precomp REIM_FFTVEC_SUB_PRECOMP;
typedef struct reim_mul_precomp REIM_FFTVEC_MUL_PRECOMP;
typedef struct reim_addmul_precomp REIM_FFTVEC_ADDMUL_PRECOMP;
typedef struct reim_fftvec_automorphism_precomp REIM_FFTVEC_AUTOMORPHISM_PRECOMP;
typedef struct reim_from_znx32_precomp REIM_FROM_ZNX32_PRECOMP;
typedef struct reim_from_znx64_precomp REIM_FROM_ZNX64_PRECOMP;
typedef struct reim_from_tnx32_precomp REIM_FROM_TNX32_PRECOMP;
typedef struct reim_to_tnx32_precomp REIM_TO_TNX32_PRECOMP;
typedef struct reim_to_tnx_precomp REIM_TO_TNX_PRECOMP;
typedef struct reim_to_znx64_precomp REIM_TO_ZNX64_PRECOMP;

/**
 * @brief precomputes fft tables.
 * The FFT tables contains a constant section that is required for efficient FFT operations in dimension nn.
 * The resulting pointer is to be passed as "tables" argument to any call to the fft function.
 * The user can optionnally allocate zero or more computation buffers, which are scratch spaces that are contiguous to
 * the constant tables in memory, and allow for more efficient operations. It is the user's responsibility to ensure
 * that each of those buffers are never used simultaneously by two ffts on different threads at the same time. The fft
 * table must be deleted by delete_fft_precomp after its last usage.
 */
EXPORT REIM_FFT_PRECOMP* new_reim_fft_precomp(uint32_t m, uint32_t num_buffers);

/**
 * @brief gets the address of a fft buffer allocated during new_fft_precomp.
 * This buffer can be used as data pointer in subsequent calls to fft,
 * and does not need to be released afterwards.
 */
EXPORT double* reim_fft_precomp_get_buffer(const REIM_FFT_PRECOMP* tables, uint32_t buffer_index);

/**
 * @brief allocates a new fft buffer.
 * This buffer can be used as data pointer in subsequent calls to fft,
 * and must be deleted afterwards by calling delete_fft_buffer.
 */
EXPORT double* new_reim_fft_buffer(uint32_t m);

/**
 * @brief allocates a new fft buffer.
 * This buffer can be used as data pointer in subsequent calls to fft,
 * and must be deleted afterwards by calling delete_fft_buffer.
 */
EXPORT void delete_reim_fft_buffer(double* buffer);

/**
 * @brief deallocates a fft table and all its built-in buffers.
 */
#define delete_reim_fft_precomp free

/**
 * @brief computes a direct fft in-place over data.
 */
EXPORT void reim_fft(const REIM_FFT_PRECOMP* tables, double* data);

EXPORT REIM_IFFT_PRECOMP* new_reim_ifft_precomp(uint32_t m, uint32_t num_buffers);
EXPORT double* reim_ifft_precomp_get_buffer(const REIM_IFFT_PRECOMP* tables, uint32_t buffer_index);
EXPORT void reim_ifft(const REIM_IFFT_PRECOMP* tables, double* data);
#define delete_reim_ifft_precomp free

EXPORT REIM_FFTVEC_ADD_PRECOMP* new_reim_fftvec_add_precomp(uint32_t m);
EXPORT void reim_fftvec_add(const REIM_FFTVEC_ADD_PRECOMP* tables, double* r, const double* a, const double* b);
#define delete_reim_fftvec_add_precomp free

EXPORT REIM_FFTVEC_SUB_PRECOMP* new_reim_fftvec_sub_precomp(uint32_t m);
EXPORT void reim_fftvec_sub(const REIM_FFTVEC_SUB_PRECOMP* tables, double* r, const double* a, const double* b);
#define delete_reim_fftvec_sub_precomp free

EXPORT REIM_FFTVEC_MUL_PRECOMP* new_reim_fftvec_mul_precomp(uint32_t m);
EXPORT void reim_fftvec_mul(const REIM_FFTVEC_MUL_PRECOMP* tables, double* r, const double* a, const double* b);
#define delete_reim_fftvec_mul_precomp free

EXPORT REIM_FFTVEC_ADDMUL_PRECOMP* new_reim_fftvec_addmul_precomp(uint32_t m);
EXPORT void reim_fftvec_addmul(const REIM_FFTVEC_ADDMUL_PRECOMP* tables, double* r, const double* a, const double* b);
#define delete_reim_fftvec_addmul_precomp free

EXPORT REIM_FFTVEC_AUTOMORPHISM_PRECOMP* new_reim_fftvec_automorphism_precomp(uint32_t m);
EXPORT void reim_fftvec_automorphism(const REIM_FFTVEC_AUTOMORPHISM_PRECOMP* tables, int64_t p, double* r,
                                     const double* a, uint64_t a_size);

EXPORT void reim_fftvec_automorphism_inplace(const REIM_FFTVEC_AUTOMORPHISM_PRECOMP* tables, int64_t p, double* a,
                                             uint64_t a_size, uint8_t* tmp_bytes);

EXPORT uint64_t reim_fftvec_automorphism_inplace_tmp_bytes(const REIM_FFTVEC_AUTOMORPHISM_PRECOMP* tables);

#define delete_reim_fftvec_automorphism_precomp free

/**
 * @brief prepares a conversion from ZnX to the cplx layout.
 * All the coefficients must be strictly lower than 2^log2bound in absolute value. Any attempt to use
 * this function on a larger coefficient is undefined behaviour. The resulting precomputed data must
 * be freed with `new_reim_from_znx32_precomp`
 * @param m the target complex dimension m from C[X] mod X^m-i. Note that the inputs have n=2m
 * int32 coefficients in natural order modulo X^n+1
 * @param log2bound bound on the input coefficients. Must be between 0 and 32
 */
EXPORT REIM_FROM_ZNX32_PRECOMP* new_reim_from_znx32_precomp(uint32_t m, uint32_t log2bound);

/**
 * @brief converts from ZnX to the cplx layout.
 * @param tables precomputed data obtained by new_reim_from_znx32_precomp.
 * @param r resulting array of m complexes coefficients mod X^m-i
 * @param x input array of n bounded integer coefficients mod X^n+1
 */
EXPORT void reim_from_znx32(const REIM_FROM_ZNX32_PRECOMP* tables, void* r, const int32_t* a);
/** @brief frees a precomputed conversion data initialized with new_reim_from_znx32_precomp. */
#define delete_reim_from_znx32_precomp free

/**
 * @brief converts from ZnX to the cplx layout.
 * @param tables precomputed data obtained by new_reim_from_znx64_precomp.
 * @param r resulting array of m complexes coefficients mod X^m-i
 * @param x input array of n bounded integer coefficients mod X^n+1
 */
EXPORT void reim_from_znx64(const REIM_FROM_ZNX64_PRECOMP* tables, void* r, const int64_t* a);
/** @brief frees a precomputed conversion data initialized with new_reim_from_znx32_precomp. */
EXPORT REIM_FROM_ZNX64_PRECOMP* new_reim_from_znx64_precomp(uint32_t m, uint32_t maxbnd);
#define delete_reim_from_znx64_precomp free
EXPORT void reim_from_znx64_simple(uint32_t m, uint32_t log2bound, void* r, const int64_t* a);

/**
 * @brief prepares a conversion from TnX to the cplx layout.
 * @param m the target complex dimension m from C[X] mod X^m-i. Note that the inputs have n=2m
 * torus32 coefficients. The resulting precomputed data must
 * be freed with `delete_reim_from_tnx32_precomp`
 */

EXPORT REIM_FROM_TNX32_PRECOMP* new_reim_from_tnx32_precomp(uint32_t m);
/**
 * @brief converts from TnX to the cplx layout.
 * @param tables precomputed data obtained by new_reim_from_tnx32_precomp.
 * @param r resulting array of m complexes coefficients mod X^m-i
 * @param x input array of n torus32 coefficients mod X^n+1
 */
EXPORT void reim_from_tnx32(const REIM_FROM_TNX32_PRECOMP* tables, void* r, const int32_t* a);
/** @brief frees a precomputed conversion data initialized with new_reim_from_tnx32_precomp. */
#define delete_reim_from_tnx32_precomp free

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
EXPORT REIM_TO_TNX32_PRECOMP* new_reim_to_tnx32_precomp(uint32_t m, double divisor, uint32_t log2overhead);

/**
 * @brief rescale, converts and reduce mod 1 from cplx layout to torus32.
 * @param tables precomputed data obtained by new_reim_from_tnx32_precomp.
 * @param r resulting array of n torus32 coefficients mod X^n+1
 * @param x input array of m cplx coefficients mod X^m-i
 */
EXPORT void reim_to_tnx32(const REIM_TO_TNX32_PRECOMP* tables, int32_t* r, const void* a);
#define delete_reim_to_tnx32_precomp free

/**
 * @brief prepares a rescale and conversion from the cplx layout to TnX (doubles).
 * @param m the target complex dimension m from C[X] mod X^m-i. Note that the outputs have n=2m
 * torus32 coefficients.
 * @param divisor must be a power of two. The inputs are rescaled by divisor before being reduced modulo 1.
 * Remember that the output of an iFFT must be divided by m.
 * @param log2overhead all inputs absolute values must be within divisor.2^log2overhead.
 * For any inputs outside of these bounds, the conversion is undefined behaviour.
 * The maximum supported log2overhead is 52, and the algorithm is faster for log2overhead=18.
 */
EXPORT REIM_TO_TNX_PRECOMP* new_reim_to_tnx_precomp(uint32_t m, double divisor, uint32_t log2overhead);
/**
 * @brief rescale, converts and reduce mod 1 from cplx layout to torus32.
 * @param tables precomputed data obtained by new_reim_from_tnx32_precomp.
 * @param r resulting array of n torus32 coefficients mod X^n+1
 * @param x input array of m cplx coefficients mod X^m-i
 */
EXPORT void reim_to_tnx(const REIM_TO_TNX_PRECOMP* tables, double* r, const double* a);
#define delete_reim_to_tnx_precomp free
EXPORT void reim_to_tnx_simple(uint32_t m, double divisor, uint32_t log2overhead, double* r, const double* a);

EXPORT REIM_TO_ZNX64_PRECOMP* new_reim_to_znx64_precomp(uint32_t m, double divisor, uint32_t log2bound);
#define delete_reim_to_znx64_precomp free
EXPORT void reim_to_znx64(const REIM_TO_ZNX64_PRECOMP* precomp, int64_t* r, const void* a);
EXPORT void reim_to_znx64_simple(uint32_t m, double divisor, uint32_t log2bound, int64_t* r, const void* a);

/**
 * @brief Simpler API for the fft function.
 * For each dimension, the precomputed tables for this dimension are generated automatically.
 * It is advised to do one dry-run per desired dimension before using in a multithread environment */
EXPORT void reim_fft_simple(uint32_t m, void* data);
/**
 * @brief Simpler API for the ifft function.
 * For each dimension, the precomputed tables for this dimension are generated automatically the first time.
 * It is advised to do one dry-run call per desired dimension in the main thread before using in a multithread
 * environment */
EXPORT void reim_ifft_simple(uint32_t m, void* data);
/**
 * @brief Simpler API for the fftvec addition function.
 * For each dimension, the precomputed tables for this dimension are generated automatically the first time.
 * It is advised to do one dry-run call per desired dimension before using in a multithread environment */
EXPORT void reim_fftvec_add_simple(uint32_t m, void* r, const void* a, const void* b);
/**
 * @brief Simpler API for the fftvec multiplication function.
 * For each dimension, the precomputed tables for this dimension are generated automatically the first time.
 * It is advised to do one dry-run call per desired dimension before using in a multithread environment */
EXPORT void reim_fftvec_mul_simple(uint32_t m, void* r, const void* a, const void* b);
/**
 * @brief Simpler API for the fftvec addmul function.
 * For each dimension, the precomputed tables for this dimension are generated automatically the first time.
 * It is advised to do one dry-run call per desired dimension before using in a multithread environment */
EXPORT void reim_fftvec_addmul_simple(uint32_t m, void* r, const void* a, const void* b);
/**
 * @brief Simpler API for the znx32 to cplx conversion.
 * For each dimension, the precomputed tables for this dimension are generated automatically the first time.
 * It is advised to do one dry-run call per desired dimension before using in a multithread environment */
EXPORT void reim_from_znx32_simple(uint32_t m, uint32_t log2bound, void* r, const int32_t* x);
/**
 * @brief Simpler API for the tnx32 to cplx conversion.
 * For each dimension, the precomputed tables for this dimension are generated automatically the first time.
 * It is advised to do one dry-run call per desired dimension before using in a multithread environment */
EXPORT void reim_from_tnx32_simple(uint32_t m, void* r, const int32_t* x);
/**
 * @brief Simpler API for the cplx to tnx32 conversion.
 * For each dimension, the precomputed tables for this dimension are generated automatically the first time.
 * It is advised to do one dry-run call per desired dimension before using in a multithread environment */
EXPORT void reim_to_tnx32_simple(uint32_t m, double divisor, uint32_t log2overhead, int32_t* r, const void* x);

#endif  // SPQLIOS_REIM_FFT_H
