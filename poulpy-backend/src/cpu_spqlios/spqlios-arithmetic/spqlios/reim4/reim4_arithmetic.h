#ifndef SPQLIOS_REIM4_ARITHMETIC_H
#define SPQLIOS_REIM4_ARITHMETIC_H

#include "../commons.h"

// the reim4 structure represent 4 complex numbers,
// represented as re0,re1,re2,re3,im0,im1,im2,im3

// TODO implement these basic arithmetic functions. (only ref is needed)

/** @brief dest = 0 */
EXPORT void reim4_zero(double* dest);
/** @brief dest = a + b */
EXPORT void reim4_add(double* dest, const double* u, const double* v);
/** @brief dest = a * b */
EXPORT void reim4_mul(double* dest, const double* u, const double* v);
/** @brief dest += a * b */
EXPORT void reim4_add_mul(double* dest, const double* a, const double* b);

// TODO add the dot products: vec x mat1col, vec x mat2cols functions here (ref and avx)

// TODO implement the convolution functions here (ref and avx)
/**
 * @brief k-th coefficient of the convolution product a * b.
 *
 * The k-th coefficient is defined as sum_{i+j=k} a[i].b[j].
 * (in this sum, i and j must remain within the bounds [0,sizea[ and [0,sizeb[)
 *
 * In practice, accounting for these bounds, the convolution function boils down to
 * ```
 * res := 0
 * if (k<sizeb+sizea-1) then
 *   for j from max(0,k+1-sizea) incl to min(sizeb,k+1) excl do
 *     res += a[k-j] * b[j]
 * return res
 * ```
 */
EXPORT void reim4_convolution_1coeff_ref(uint64_t k, double* dest, const double* a, uint64_t sizea, const double* b,
                                         uint64_t sizeb);
EXPORT void reim4_convolution_1coeff_avx(uint64_t k, double* dest, const double* a, uint64_t sizea, const double* b,
                                         uint64_t sizeb);
/** @brief returns two consecutive convolution coefficients: k and k+1 */
EXPORT void reim4_convolution_2coeff_ref(uint64_t k, double* dest, const double* a, uint64_t sizea, const double* b,
                                         uint64_t sizeb);
EXPORT void reim4_convolution_2coeff_avx(uint64_t k, double* dest, const double* a, uint64_t sizea, const double* b,
                                         uint64_t sizeb);

/**
 * @brief From the convolution a * b, return the coefficients between offset and offset + size
 * For the full convolution, use offset=0 and size=sizea+sizeb-1.
 */
EXPORT void reim4_convolution_ref(double* dest, uint64_t dest_size, uint64_t dest_offset, const double* a,
                                  uint64_t sizea, const double* b, uint64_t sizeb);
EXPORT void reim4_convolution_avx(double* dest, uint64_t dest_size, uint64_t dest_offset, const double* a,
                                  uint64_t sizea, const double* b, uint64_t sizeb);

// The reim fft layout that encodes m = nn/2 complex evaluations
// can be decomposed into m/4 reim4 blocks
// the i-th block of v is noted v(i);
// it contains the 4 evaluations (e[4i],e[4i+1],e[4i+2],e[4i+3])

/**
 * @brief extract 1 reim4 block from one reim vectors of m complexes
 * @param m the size of each reim
 * @param blk the block id to extract (<m/4)
 * @param dst the output: nrows reim4's  dst[i] = src[i](blk)
 * @param src the input: nrows reim's
 */
EXPORT void reim4_extract_1blk_from_reim_ref(uint64_t m, uint64_t blk,
                                             double* const dst,       // 8 doubles
                                             const double* const src  // a reim vector
);
EXPORT void reim4_extract_1blk_from_reim_avx(uint64_t m, uint64_t blk,
                                             double* const dst,       // 8 doubles
                                             const double* const src  // a reim vectors
);

/**
 * @brief extract 1 reim vectors from one reim4 block of m complexes
 * @param m the size of each reim
 * @param blk the block id to extract (<m/4)
 * @param dst the output: nrows reim4's  dst[i](blk) = src[i]
 * @param src the input: nrows reim's
 */
EXPORT void reim4_extract_reim_from_1blk_ref(uint64_t m, uint64_t blk,
                                             double* const dst,       // 8 doubles
                                             const double* const src  // a reim vector
);

/**
 * @brief extract 1 reim4 block from nrows reim vectors of m complexes
 * @param nrows the number of reim (fft) vectors
 * @param m the size of each reim
 * @param blk the block id to extract (<m/4)
 * @param dst the output: nrows reim4's  dst[i] = src[i](blk)
 * @param src the input: nrows reim's
 */
EXPORT void reim4_extract_1blk_from_contiguous_reim_ref(uint64_t m, uint64_t nrows, uint64_t blk, double* const dst,
                                                        const double* const src);
EXPORT void reim4_extract_1blk_from_contiguous_reim_avx(uint64_t m, uint64_t nrows, uint64_t blk, double* const dst,
                                                        const double* const src);

/**
 * @brief extract 1 reim4 block from nrows reim vectors of m complexes with slice sl
 * @param m the size of each reim
 * @param sl the slice size
 * @param nrows the number of reim (fft) vectors to extract
 * @param blk the block id to extract (<m/4)
 * @param dst the output: nrows reim4's  dst[i] = src[i](blk)
 * @param src the input: nrows reim's
 */
EXPORT void reim4_extract_1blk_from_contiguous_reim_sl_ref(uint64_t m, uint64_t sl, uint64_t nrows, uint64_t blk,
                                                           double* const dst, const double* const src);
EXPORT void reim4_extract_1blk_from_contiguous_reim_sl_avx(uint64_t m, uint64_t sl, uint64_t nrows, uint64_t blk,
                                                           double* const dst, const double* const src);

/**
 * @brief saves 1 single reim4 block in a reim vectors of m complexes
 * @param m the size of each reim
 * @param blk the block id to save (<m/4)
 * @param dest the output reim: dst(blk) = src
 * @param src the input reim4
 */
EXPORT void reim4_save_1blk_to_reim_ref(uint64_t m, uint64_t blk,
                                        double* dest,      // 1 reim vector of length m
                                        const double* src  // 8 doubles
);
EXPORT void reim4_save_1blk_to_reim_avx(uint64_t m, uint64_t blk,
                                        double* dest,      // 1 reim vector of length m
                                        const double* src  // 8 doubles
);

/**
 * @brief Adds 1 single reim4 block in a reim vectors of m complexes
 * @param m the size of each reim
 * @param blk the block id to save (<m/4)
 * @param dest the output reim: dst(blk) += src
 * @param src the input reim4
 */
EXPORT void reim4_add_1blk_to_reim_ref(uint64_t m, uint64_t blk,
                                       double* dest,      // 1 reim vector of length m
                                       const double* src  // 8 doubles
);
EXPORT void reim4_add_1blk_to_reim_avx(uint64_t m, uint64_t blk,
                                       double* dest,      // 1 reim vector of length m
                                       const double* src  // 8 doubles
);

// dest = sum u[i].v[i]
// make ref and avx2 implem, needs to be fast
// can accumulate the real and imaginary parts on 2 separate registers, and combine at the end
EXPORT void reim4_vec_mat1col_product_ref(const uint64_t nrows,
                                          double* const dst,      // 8 doubles
                                          const double* const u,  // nrows * 8 doubles
                                          const double* const v   // nrows * 8 doubles
);

EXPORT void reim4_vec_mat1col_product_avx2(const uint64_t nrows,
                                           double* const dst,      // 8 doubles
                                           const double* const u,  // nrows * 8 doubles
                                           const double* const v   // nrows * 8 doubles
);

// dest = sum u[i].v[i]
// make ref and avx2 implem, needs to be fast
// can accumulate the real and imaginary parts on 2 separate registers, and recombine at the end
EXPORT void reim4_vec_mat2cols_product_ref(uint64_t nrows, double* const dst,  // 16 doubles
                                           const double* const u,              // nrows * 8 doubles
                                           const double* const v               // nrows * 16 doubles
);

EXPORT void reim4_vec_mat2cols_product_avx2(uint64_t nrows, double* const dst,  // 16 doubles
                                            const double* const u,              // nrows * 8 doubles
                                            const double* const v               // nrows * 16 doubles
);

#endif  // SPQLIOS_REIM4_ARITHMETIC_H
