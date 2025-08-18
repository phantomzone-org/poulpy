#ifndef SPQLIOS_Q120_NTT_H
#define SPQLIOS_Q120_NTT_H

#include "../commons.h"
#include "q120_common.h"

typedef struct _q120_ntt_precomp q120_ntt_precomp;

EXPORT q120_ntt_precomp* q120_new_ntt_bb_precomp(const uint64_t n);
EXPORT void q120_del_ntt_bb_precomp(q120_ntt_precomp* precomp);

EXPORT q120_ntt_precomp* q120_new_intt_bb_precomp(const uint64_t n);
EXPORT void q120_del_intt_bb_precomp(q120_ntt_precomp* precomp);

/**
 * @brief computes a direct ntt in-place over data.
 */
EXPORT void q120_ntt_bb_avx2(const q120_ntt_precomp* const precomp, q120b* const data);

/**
 * @brief computes an inverse ntt in-place over data.
 */
EXPORT void q120_intt_bb_avx2(const q120_ntt_precomp* const precomp, q120b* const data);

#endif  // SPQLIOS_Q120_NTT_H
