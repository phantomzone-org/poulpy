#include <assert.h>
#include <string.h>

#include "reim4_arithmetic.h"

// Stores the first 4 values (RE) of src + blk*4 and he first 4 values (IM) of src + blk*4 + m
// contiguously into dst
void reim4_extract_1blk_from_reim_ref(uint64_t m, uint64_t blk,
                                      double* const dst,       // 8 doubles
                                      const double* const src  // one reim4 vector
) {
  assert(blk < (m >> 2));
  const double* src_ptr = src + (blk << 2);
  // copy the real parts
  dst[0] = src_ptr[0];
  dst[1] = src_ptr[1];
  dst[2] = src_ptr[2];
  dst[3] = src_ptr[3];
  src_ptr += m;
  // copy the imaginary parts
  dst[4] = src_ptr[0];
  dst[5] = src_ptr[1];
  dst[6] = src_ptr[2];
  dst[7] = src_ptr[3];
}

void reim4_extract_reim_from_1blk_ref(uint64_t m, uint64_t blk,
                                      double* const dst,       // 8 doubles
                                      const double* const src  // one reim4 vector
) {
  assert(blk < (m >> 2));
  double* dst_ptr = dst + (blk << 2);
  // copy the real parts
  dst_ptr[0] = src[0];
  dst_ptr[1] = src[1];
  dst_ptr[2] = src[2];
  dst_ptr[3] = src[3];
  dst_ptr += m;
  // copy the imaginary parts
  dst_ptr[0] = src[4];
  dst_ptr[1] = src[5];
  dst_ptr[2] = src[6];
  dst_ptr[3] = src[7];
}

EXPORT void reim4_extract_1blk_from_contiguous_reim_ref(uint64_t m, uint64_t nrows, uint64_t blk, double* const dst,
                                                        const double* const src) {
  assert(blk < (m >> 2));

  const double* src_ptr = src + (blk << 2);
  double* dst_ptr = dst;
  for (uint64_t i = 0; i < nrows * 2; ++i) {
    dst_ptr[0] = src_ptr[0];
    dst_ptr[1] = src_ptr[1];
    dst_ptr[2] = src_ptr[2];
    dst_ptr[3] = src_ptr[3];
    dst_ptr += 4;
    src_ptr += m;
  }
}

EXPORT void reim4_extract_1blk_from_contiguous_reim_sl_ref(uint64_t m, uint64_t sl, uint64_t nrows, uint64_t blk,
                                                           double* const dst, const double* const src) {
  assert(blk < (m >> 2));

  const double* src_ptr = src + (blk << 2);
  double* dst_ptr = dst;
  const uint64_t sl_minus_m = sl - m;
  for (uint64_t i = 0; i < nrows; ++i) {
    dst_ptr[0] = src_ptr[0];
    dst_ptr[1] = src_ptr[1];
    dst_ptr[2] = src_ptr[2];
    dst_ptr[3] = src_ptr[3];
    dst_ptr += 4;
    src_ptr += m;
    dst_ptr[0] = src_ptr[0];
    dst_ptr[1] = src_ptr[1];
    dst_ptr[2] = src_ptr[2];
    dst_ptr[3] = src_ptr[3];
    dst_ptr += 4;
    src_ptr += sl_minus_m;
  }
}

// dest(i)=src
// use scalar or sse or avx? Code
// should be the inverse of reim4_extract_1col_from_reim
void reim4_save_1blk_to_reim_ref(uint64_t m, uint64_t blk,
                                 double* dst,       // 1 reim vector of length m
                                 const double* src  // 8 doubles
) {
  assert(blk < (m >> 2));
  double* dst_ptr = dst + (blk << 2);
  // save the real part
  dst_ptr[0] = src[0];
  dst_ptr[1] = src[1];
  dst_ptr[2] = src[2];
  dst_ptr[3] = src[3];
  dst_ptr += m;
  // save the imag part
  dst_ptr[0] = src[4];
  dst_ptr[1] = src[5];
  dst_ptr[2] = src[6];
  dst_ptr[3] = src[7];
}

void reim4_add_1blk_to_reim_ref(uint64_t m, uint64_t blk,
                                double* dst,       // 1 reim vector of length m
                                const double* src  // 8 doubles
) {
  assert(blk < (m >> 2));
  double* dst_ptr = dst + (blk << 2);
  // add the real part
  dst_ptr[0] += src[0];
  dst_ptr[1] += src[1];
  dst_ptr[2] += src[2];
  dst_ptr[3] += src[3];
  dst_ptr += m;
  // add the imag part
  dst_ptr[0] += src[4];
  dst_ptr[1] += src[5];
  dst_ptr[2] += src[6];
  dst_ptr[3] += src[7];
}

// dest = 0
void reim4_zero(double* const dst  // 8 doubles
) {
  for (uint64_t i = 0; i < 8; ++i) dst[i] = 0;
}

/** @brief dest = a + b */
void reim4_add(double* const dst,      // 8 doubles
               const double* const u,  // nrows * 8 doubles
               const double* const v   // nrows * 8 doubles
) {
  for (uint64_t k = 0; k < 4; ++k) {
    const double a = u[k];
    const double c = v[k];
    const double b = u[k + 4];
    const double d = v[k + 4];

    dst[k] = a + c;
    dst[k + 4] = b + d;
  }
}

/** @brief dest = a * b */
void reim4_mul(double* const dst,      // 8 doubles
               const double* const u,  // 8 doubles
               const double* const v   // 8 doubles
) {
  for (uint64_t k = 0; k < 4; ++k) {
    const double a = u[k];
    const double c = v[k];
    const double b = u[k + 4];
    const double d = v[k + 4];

    dst[k] = a * c - b * d;
    dst[k + 4] = a * d + b * c;
  }
}

/** @brief dest += a * b */
void reim4_add_mul(double* const dst,      // 8 doubles
                   const double* const u,  // 8 doubles
                   const double* const v   // 8 doubles
) {
  for (uint64_t k = 0; k < 4; ++k) {
    const double a = u[k];
    const double c = v[k];
    const double b = u[k + 4];
    const double d = v[k + 4];

    dst[k] += a * c - b * d;
    dst[k + 4] += a * d + b * c;
  }
}

/** dest = uT * v  where u is a vector of size nrows, and v is a nrows x 1 matrix */
void reim4_vec_mat1col_product_ref(const uint64_t nrows,
                                   double* const dst,      // 8 doubles
                                   const double* const u,  // nrows * 8 doubles
                                   const double* const v   // nrows * 8 doubles
) {
  reim4_zero(dst);

  for (uint64_t i = 0, j = 0; i < nrows; ++i, j += 8) {
    reim4_add_mul(dst, u + j, v + j);
  }
}

/** dest = uT * v  where u is a vector of size nrows, and v is a nrows x 2 matrix */
void reim4_vec_mat2cols_product_ref(const uint64_t nrows,
                                    double* const dst,      // 16 doubles
                                    const double* const u,  // nrows * 8 doubles
                                    const double* const v   // nrows * 16 doubles
) {
  reim4_zero(dst);
  reim4_zero(dst + 8);

  double* dst1 = dst;
  double* dst2 = dst + 8;

  for (uint64_t i = 0, j = 0; i < nrows; ++i, j += 8) {
    uint64_t double_j = j << 1;
    reim4_add_mul(dst1, u + j, v + double_j);
    reim4_add_mul(dst2, u + j, v + double_j + 8);
  }
}

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
                                         uint64_t sizeb) {
  reim4_zero(dest);
  if (k >= sizea + sizeb) return;
  uint64_t jmin = k >= sizea ? k + 1 - sizea : 0;
  uint64_t jmax = k < sizeb ? k + 1 : sizeb;
  for (uint64_t j = jmin; j < jmax; ++j) {
    reim4_add_mul(dest, a + 8 * (k - j), b + 8 * j);
  }
}

/** @brief returns two consecutive convolution coefficients: k and k+1 */
EXPORT void reim4_convolution_2coeff_ref(uint64_t k, double* dest, const double* a, uint64_t sizea, const double* b,
                                         uint64_t sizeb) {
  reim4_convolution_1coeff_ref(k, dest, a, sizea, b, sizeb);
  reim4_convolution_1coeff_ref(k + 1, dest + 8, a, sizea, b, sizeb);
}

/**
 * @brief From the convolution a * b, return the coefficients between offset and offset + size
 * For the full convolution, use offset=0 and size=sizea+sizeb-1.
 */
EXPORT void reim4_convolution_ref(double* dest, uint64_t dest_size, uint64_t dest_offset, const double* a,
                                  uint64_t sizea, const double* b, uint64_t sizeb) {
  for (uint64_t k = 0; k < dest_size; ++k) {
    reim4_convolution_1coeff_ref(k + dest_offset, dest + 8 * k, a, sizea, b, sizeb);
  }
}
