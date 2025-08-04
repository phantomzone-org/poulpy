#include <assert.h>
#include <immintrin.h>

#include "reim4_arithmetic.h"

void reim4_extract_1blk_from_reim_avx(uint64_t m, uint64_t blk,
                                      double* const dst,       // nrows * 8 doubles
                                      const double* const src  // a contiguous array of nrows reim vectors
) {
  assert(blk < (m >> 2));
  const double* src_ptr = src + (blk << 2);
  double* dst_ptr = dst;
  _mm256_storeu_pd(dst_ptr, _mm256_loadu_pd(src_ptr));
  _mm256_storeu_pd(dst_ptr + 4, _mm256_loadu_pd(src_ptr + m));
}

void reim4_extract_1blk_from_contiguous_reim_avx(uint64_t m, uint64_t nrows, uint64_t blk, double* const dst,
                                                 const double* const src) {
  assert(blk < (m >> 2));
  const double* src_ptr = src + (blk << 2);
  double* dst_ptr = dst;
  for (uint64_t i = 0; i < nrows * 2; ++i) {
    _mm256_storeu_pd(dst_ptr, _mm256_loadu_pd(src_ptr));
    dst_ptr += 4;
    src_ptr += m;
  }
}

EXPORT void reim4_extract_1blk_from_contiguous_reim_sl_avx(uint64_t m, uint64_t sl, uint64_t nrows, uint64_t blk,
                                                           double* const dst, const double* const src) {
  assert(blk < (m >> 2));
  const double* src_ptr = src + (blk << 2);
  double* dst_ptr = dst;
  for (uint64_t i = 0; i < nrows; ++i) {
    _mm256_storeu_pd(dst_ptr, _mm256_loadu_pd(src_ptr));
    _mm256_storeu_pd(dst_ptr + 4, _mm256_loadu_pd(src_ptr + m));
    dst_ptr += 8;
    src_ptr += sl;
  }
}

void reim4_save_1blk_to_reim_avx(uint64_t m, uint64_t blk,
                                 double* dst,       // 1 reim vector of length m
                                 const double* src  // 8 doubles
) {
  assert(blk < (m >> 2));
  const double* src_ptr = src;
  double* dst_ptr = dst + (blk << 2);
  _mm256_storeu_pd(dst_ptr, _mm256_loadu_pd(src_ptr));
  _mm256_storeu_pd(dst_ptr + m, _mm256_loadu_pd(src_ptr + 4));
}

void reim4_add_1blk_to_reim_avx(uint64_t m, uint64_t blk,
                                double* dst,       // 1 reim vector of length m
                                const double* src  // 8 doubles
) {
  assert(blk < (m >> 2));
  const double* src_ptr = src;
  double* dst_ptr = dst + (blk << 2);

  __m256d s0 = _mm256_loadu_pd(src_ptr);
  __m256d d0 = _mm256_loadu_pd(dst_ptr);
  _mm256_storeu_pd(dst_ptr, _mm256_add_pd(s0, d0));

  __m256d s1 = _mm256_loadu_pd(src_ptr + 4);
  __m256d d1 = _mm256_loadu_pd(dst_ptr + m);
  _mm256_storeu_pd(dst_ptr + m, _mm256_add_pd(s1, d1));
}

__always_inline void cplx_prod(__m256d* re1, __m256d* re2, __m256d* im, const double* const u_ptr,
                               const double* const v_ptr) {
  const __m256d a = _mm256_loadu_pd(u_ptr);
  const __m256d c = _mm256_loadu_pd(v_ptr);
  *re1 = _mm256_fmadd_pd(a, c, *re1);

  const __m256d b = _mm256_loadu_pd(u_ptr + 4);
  *im = _mm256_fmadd_pd(b, c, *im);

  const __m256d d = _mm256_loadu_pd(v_ptr + 4);
  *re2 = _mm256_fmadd_pd(b, d, *re2);
  *im = _mm256_fmadd_pd(a, d, *im);
}

void reim4_vec_mat1col_product_avx2(const uint64_t nrows,
                                    double* const dst,      // 8 doubles
                                    const double* const u,  // nrows * 8 doubles
                                    const double* const v   // nrows * 8 doubles
) {
  __m256d re1 = _mm256_setzero_pd();
  __m256d re2 = _mm256_setzero_pd();
  __m256d im1 = _mm256_setzero_pd();
  __m256d im2 = _mm256_setzero_pd();

  const double* u_ptr = u;
  const double* v_ptr = v;
  for (uint64_t i = 0; i < nrows; ++i) {
    const __m256d a = _mm256_loadu_pd(u_ptr);
    const __m256d c = _mm256_loadu_pd(v_ptr);
    re1 = _mm256_fmadd_pd(a, c, re1);

    const __m256d b = _mm256_loadu_pd(u_ptr + 4);
    im2 = _mm256_fmadd_pd(b, c, im2);

    const __m256d d = _mm256_loadu_pd(v_ptr + 4);
    re2 = _mm256_fmadd_pd(b, d, re2);
    im1 = _mm256_fmadd_pd(a, d, im1);

    u_ptr += 8;
    v_ptr += 8;
  }

  _mm256_storeu_pd(dst, _mm256_sub_pd(re1, re2));
  _mm256_storeu_pd(dst + 4, _mm256_add_pd(im1, im2));
}

EXPORT void reim4_vec_mat2cols_product_avx2(const uint64_t nrows,
                                            double* const dst,      // 16 doubles
                                            const double* const u,  // nrows * 16 doubles
                                            const double* const v   // nrows * 16 doubles
) {
  __m256d re1 = _mm256_setzero_pd();
  __m256d im1 = _mm256_setzero_pd();
  __m256d re2 = _mm256_setzero_pd();
  __m256d im2 = _mm256_setzero_pd();

  __m256d ur, ui, ar, ai, br, bi;
  for (uint64_t i = 0; i < nrows; ++i) {
    ur = _mm256_loadu_pd(u + 8 * i);
    ui = _mm256_loadu_pd(u + 8 * i + 4);
    ar = _mm256_loadu_pd(v + 16 * i);
    ai = _mm256_loadu_pd(v + 16 * i + 4);
    br = _mm256_loadu_pd(v + 16 * i + 8);
    bi = _mm256_loadu_pd(v + 16 * i + 12);
    re1 = _mm256_fmsub_pd(ui, ai, re1);
    re2 = _mm256_fmsub_pd(ui, bi, re2);
    im1 = _mm256_fmadd_pd(ur, ai, im1);
    im2 = _mm256_fmadd_pd(ur, bi, im2);
    re1 = _mm256_fmsub_pd(ur, ar, re1);
    re2 = _mm256_fmsub_pd(ur, br, re2);
    im1 = _mm256_fmadd_pd(ui, ar, im1);
    im2 = _mm256_fmadd_pd(ui, br, im2);
  }
  _mm256_storeu_pd(dst, re1);
  _mm256_storeu_pd(dst + 4, im1);
  _mm256_storeu_pd(dst + 8, re2);
  _mm256_storeu_pd(dst + 12, im2);
}
