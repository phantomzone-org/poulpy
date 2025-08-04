#ifndef SPQLIOS_Q120_ARITHMETIC_H
#define SPQLIOS_Q120_ARITHMETIC_H

#include <stdint.h>

#include "../commons.h"
#include "q120_common.h"

typedef struct _q120_mat1col_product_baa_precomp q120_mat1col_product_baa_precomp;
typedef struct _q120_mat1col_product_bbb_precomp q120_mat1col_product_bbb_precomp;
typedef struct _q120_mat1col_product_bbc_precomp q120_mat1col_product_bbc_precomp;

EXPORT q120_mat1col_product_baa_precomp* q120_new_vec_mat1col_product_baa_precomp();
EXPORT void q120_delete_vec_mat1col_product_baa_precomp(q120_mat1col_product_baa_precomp*);
EXPORT q120_mat1col_product_bbb_precomp* q120_new_vec_mat1col_product_bbb_precomp();
EXPORT void q120_delete_vec_mat1col_product_bbb_precomp(q120_mat1col_product_bbb_precomp*);
EXPORT q120_mat1col_product_bbc_precomp* q120_new_vec_mat1col_product_bbc_precomp();
EXPORT void q120_delete_vec_mat1col_product_bbc_precomp(q120_mat1col_product_bbc_precomp*);

// ell < 10000
EXPORT void q120_vec_mat1col_product_baa_ref(q120_mat1col_product_baa_precomp*, const uint64_t ell, q120b* const res,
                                             const q120a* const x, const q120a* const y);
EXPORT void q120_vec_mat1col_product_bbb_ref(q120_mat1col_product_bbb_precomp*, const uint64_t ell, q120b* const res,
                                             const q120b* const x, const q120b* const y);
EXPORT void q120_vec_mat1col_product_bbc_ref(q120_mat1col_product_bbc_precomp*, const uint64_t ell, q120b* const res,
                                             const q120b* const x, const q120c* const y);

EXPORT void q120_vec_mat1col_product_baa_avx2(q120_mat1col_product_baa_precomp*, const uint64_t ell, q120b* const res,
                                              const q120a* const x, const q120a* const y);
EXPORT void q120_vec_mat1col_product_bbb_avx2(q120_mat1col_product_bbb_precomp*, const uint64_t ell, q120b* const res,
                                              const q120b* const x, const q120b* const y);
EXPORT void q120_vec_mat1col_product_bbc_avx2(q120_mat1col_product_bbc_precomp*, const uint64_t ell, q120b* const res,
                                              const q120b* const x, const q120c* const y);

EXPORT void q120x2_vec_mat1col_product_bbc_ref(q120_mat1col_product_bbc_precomp* precomp, const uint64_t ell,
                                               q120b* const res, const q120b* const x, const q120c* const y);
EXPORT void q120x2_vec_mat1col_product_bbc_avx2(q120_mat1col_product_bbc_precomp* precomp, const uint64_t ell,
                                                q120b* const res, const q120b* const x, const q120c* const y);
EXPORT void q120x2_vec_mat2cols_product_bbc_ref(q120_mat1col_product_bbc_precomp* precomp, const uint64_t ell,
                                                q120b* const res, const q120b* const x, const q120c* const y);
EXPORT void q120x2_vec_mat2cols_product_bbc_avx2(q120_mat1col_product_bbc_precomp* precomp, const uint64_t ell,
                                                 q120b* const res, const q120b* const x, const q120c* const y);

/**
 * @brief extract 1 q120x2 block from one q120 ntt vectors
 * @param nn the size of each vector
 * @param blk the block id to extract (<nn/2)
 * @param dst the output: nrows q120x2's dst[i] = src[i](blk)
 * @param src the input: nrows q120 ntt vecs's
 */
EXPORT void q120x2_extract_1blk_from_q120b_ref(uint64_t nn, uint64_t blk,
                                               q120x2b* const dst,     // 8 doubles
                                               const q120b* const src  // a reim vector
);
EXPORT void q120x2_extract_1blk_from_q120c_ref(uint64_t nn, uint64_t blk,
                                               q120x2c* const dst,     // 8 doubles
                                               const q120c* const src  // a reim vector
);
EXPORT void q120x2_extract_1blk_from_q120b_avx(uint64_t nn, uint64_t blk,
                                               q120x2b* const dst,     // 8 doubles
                                               const q120b* const src  // a reim vector
);
EXPORT void q120x2_extract_1blk_from_q120c_avx(uint64_t nn, uint64_t blk,
                                               q120x2c* const dst,     // 8 doubles
                                               const q120c* const src  // a reim vector
);

/**
 * @brief extract 1 reim4 block from nrows reim vectors of m complexes
 * @param nn the size of each q120
 * @param nrows the number of q120 (ntt) vectors
 * @param blk the block id to extract (<m/4)
 * @param dst the output: nrows q120x2's  dst[i] = src[i](blk)
 * @param src the input: nrows q120 ntt vectors
 */
EXPORT void q120x2_extract_1blk_from_contiguous_q120b_ref(
    uint64_t nn, uint64_t nrows, uint64_t blk,
    q120x2b* const dst,     // nrows * 2 q120
    const q120b* const src  // a contiguous array of nrows q120b vectors
);
EXPORT void q120x2_extract_1blk_from_contiguous_q120b_avx(
    uint64_t nn, uint64_t nrows, uint64_t blk,
    q120x2b* const dst,     // nrows * 2 q120
    const q120b* const src  // a contiguous array of nrows q120b vectors
);

/**
 * @brief saves 1 single q120x2 block in a q120 vectors of size nn
 * @param nn the size of the output q120
 * @param blk the block id to save (<nn/2)
 * @param dest the output q120b vector: dst(blk) = src
 * @param src the input q120x2b
 */
EXPORT void q120x2b_save_1blk_to_q120b_ref(uint64_t nn, uint64_t blk,
                                           q120b* dest,        // 1 reim vector of length m
                                           const q120x2b* src  // 8 doubles
);
EXPORT void q120x2b_save_1blk_to_q120b_avx(uint64_t nn, uint64_t blk,
                                           q120b* dest,        // 1 reim vector of length m
                                           const q120x2b* src  // 8 doubles
);

EXPORT void q120_add_bbb_simple(uint64_t nn, q120b* const res, const q120b* const x, const q120b* const y);

EXPORT void q120_add_ccc_simple(uint64_t nn, q120c* const res, const q120c* const x, const q120c* const y);

EXPORT void q120_c_from_b_simple(uint64_t nn, q120c* const res, const q120b* const x);

EXPORT void q120_b_from_znx64_simple(uint64_t nn, q120b* const res, const int64_t* const x);

EXPORT void q120_c_from_znx64_simple(uint64_t nn, q120c* const res, const int64_t* const x);

EXPORT void q120_b_to_znx128_simple(uint64_t nn, __int128_t* const res, const q120b* const x);

#endif  // SPQLIOS_Q120_ARITHMETIC_H
