#include <memory.h>

#include "immintrin.h"
#include "vec_rnx_arithmetic_private.h"

/** @brief sets res = gadget_decompose(a) */
EXPORT void rnx_approxdecomp_from_tnxdbl_avx(         //
    const MOD_RNX* module,                            // N
    const TNXDBL_APPROXDECOMP_GADGET* gadget,         // output base 2^K
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a                                   // a
) {
  const uint64_t nn = module->n;
  if (nn < 4) return rnx_approxdecomp_from_tnxdbl_ref(module, gadget, res, res_size, res_sl, a);
  const uint64_t ell = gadget->ell;
  const __m256i k = _mm256_set1_epi64x(gadget->k);
  const __m256d add_cst = _mm256_set1_pd(gadget->add_cst);
  const __m256i and_mask = _mm256_set1_epi64x(gadget->and_mask);
  const __m256i or_mask = _mm256_set1_epi64x(gadget->or_mask);
  const __m256d sub_cst = _mm256_set1_pd(gadget->sub_cst);
  const uint64_t msize = res_size <= ell ? res_size : ell;
  // gadget decompose column by column
  if (msize == ell) {
    // this is the main scenario when msize == ell
    double* const last_r = res + (msize - 1) * res_sl;
    for (uint64_t j = 0; j < nn; j += 4) {
      double* rr = last_r + j;
      const double* aa = a + j;
      __m256d t_dbl = _mm256_add_pd(_mm256_loadu_pd(aa), add_cst);
      __m256i t_int = _mm256_castpd_si256(t_dbl);
      do {
        __m256i u_int = _mm256_or_si256(_mm256_and_si256(t_int, and_mask), or_mask);
        _mm256_storeu_pd(rr, _mm256_sub_pd(_mm256_castsi256_pd(u_int), sub_cst));
        t_int = _mm256_srlv_epi64(t_int, k);
        rr -= res_sl;
      } while (rr >= res);
    }
  } else if (msize > 0) {
    // otherwise, if msize < ell: there is one additional rshift
    const __m256i first_rsh = _mm256_set1_epi64x((ell - msize) * gadget->k);
    double* const last_r = res + (msize - 1) * res_sl;
    for (uint64_t j = 0; j < nn; j += 4) {
      double* rr = last_r + j;
      const double* aa = a + j;
      __m256d t_dbl = _mm256_add_pd(_mm256_loadu_pd(aa), add_cst);
      __m256i t_int = _mm256_srlv_epi64(_mm256_castpd_si256(t_dbl), first_rsh);
      do {
        __m256i u_int = _mm256_or_si256(_mm256_and_si256(t_int, and_mask), or_mask);
        _mm256_storeu_pd(rr, _mm256_sub_pd(_mm256_castsi256_pd(u_int), sub_cst));
        t_int = _mm256_srlv_epi64(t_int, k);
        rr -= res_sl;
      } while (rr >= res);
    }
  }
  // zero-out the last slices (if any)
  for (uint64_t i = msize; i < res_size; ++i) {
    memset(res + i * res_sl, 0, nn * sizeof(double));
  }
}
