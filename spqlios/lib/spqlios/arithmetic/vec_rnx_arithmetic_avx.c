#include <immintrin.h>
#include <string.h>

#include "vec_rnx_arithmetic_private.h"

void rnx_add_avx(uint64_t nn, double* res, const double* a, const double* b) {
  if (nn < 8) {
    if (nn == 4) {
      _mm256_storeu_pd(res, _mm256_add_pd(_mm256_loadu_pd(a), _mm256_loadu_pd(b)));
    } else if (nn == 2) {
      _mm_storeu_pd(res, _mm_add_pd(_mm_loadu_pd(a), _mm_loadu_pd(b)));
    } else if (nn == 1) {
      *res = *a + *b;
    } else {
      NOT_SUPPORTED();  // not a power of 2
    }
    return;
  }
  // general case: nn >= 8
  __m256d x0, x1, x2, x3, x4, x5;
  const double* aa = a;
  const double* bb = b;
  double* rr = res;
  double* const rrend = res + nn;
  do {
    x0 = _mm256_loadu_pd(aa);
    x1 = _mm256_loadu_pd(aa + 4);
    x2 = _mm256_loadu_pd(bb);
    x3 = _mm256_loadu_pd(bb + 4);
    x4 = _mm256_add_pd(x0, x2);
    x5 = _mm256_add_pd(x1, x3);
    _mm256_storeu_pd(rr, x4);
    _mm256_storeu_pd(rr + 4, x5);
    aa += 8;
    bb += 8;
    rr += 8;
  } while (rr < rrend);
}

void rnx_sub_avx(uint64_t nn, double* res, const double* a, const double* b) {
  if (nn < 8) {
    if (nn == 4) {
      _mm256_storeu_pd(res, _mm256_sub_pd(_mm256_loadu_pd(a), _mm256_loadu_pd(b)));
    } else if (nn == 2) {
      _mm_storeu_pd(res, _mm_sub_pd(_mm_loadu_pd(a), _mm_loadu_pd(b)));
    } else if (nn == 1) {
      *res = *a - *b;
    } else {
      NOT_SUPPORTED();  // not a power of 2
    }
    return;
  }
  // general case: nn >= 8
  __m256d x0, x1, x2, x3, x4, x5;
  const double* aa = a;
  const double* bb = b;
  double* rr = res;
  double* const rrend = res + nn;
  do {
    x0 = _mm256_loadu_pd(aa);
    x1 = _mm256_loadu_pd(aa + 4);
    x2 = _mm256_loadu_pd(bb);
    x3 = _mm256_loadu_pd(bb + 4);
    x4 = _mm256_sub_pd(x0, x2);
    x5 = _mm256_sub_pd(x1, x3);
    _mm256_storeu_pd(rr, x4);
    _mm256_storeu_pd(rr + 4, x5);
    aa += 8;
    bb += 8;
    rr += 8;
  } while (rr < rrend);
}

void rnx_negate_avx(uint64_t nn, double* res, const double* b) {
  if (nn < 8) {
    if (nn == 4) {
      _mm256_storeu_pd(res, _mm256_sub_pd(_mm256_set1_pd(0), _mm256_loadu_pd(b)));
    } else if (nn == 2) {
      _mm_storeu_pd(res, _mm_sub_pd(_mm_set1_pd(0), _mm_loadu_pd(b)));
    } else if (nn == 1) {
      *res = -*b;
    } else {
      NOT_SUPPORTED();  // not a power of 2
    }
    return;
  }
  // general case: nn >= 8
  __m256d x2, x3, x4, x5;
  const __m256d ZERO = _mm256_set1_pd(0);
  const double* bb = b;
  double* rr = res;
  double* const rrend = res + nn;
  do {
    x2 = _mm256_loadu_pd(bb);
    x3 = _mm256_loadu_pd(bb + 4);
    x4 = _mm256_sub_pd(ZERO, x2);
    x5 = _mm256_sub_pd(ZERO, x3);
    _mm256_storeu_pd(rr, x4);
    _mm256_storeu_pd(rr + 4, x5);
    bb += 8;
    rr += 8;
  } while (rr < rrend);
}

/** @brief sets res = a + b */
EXPORT void vec_rnx_add_avx(                          //
    const MOD_RNX* module,                            // N
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl,  // a
    const double* b, uint64_t b_size, uint64_t b_sl   // b
) {
  const uint64_t nn = module->n;
  if (a_size < b_size) {
    const uint64_t msize = res_size < a_size ? res_size : a_size;
    const uint64_t nsize = res_size < b_size ? res_size : b_size;
    for (uint64_t i = 0; i < msize; ++i) {
      rnx_add_avx(nn, res + i * res_sl, a + i * a_sl, b + i * b_sl);
    }
    for (uint64_t i = msize; i < nsize; ++i) {
      memcpy(res + i * res_sl, b + i * b_sl, nn * sizeof(double));
    }
    for (uint64_t i = nsize; i < res_size; ++i) {
      memset(res + i * res_sl, 0, nn * sizeof(double));
    }
  } else {
    const uint64_t msize = res_size < b_size ? res_size : b_size;
    const uint64_t nsize = res_size < a_size ? res_size : a_size;
    for (uint64_t i = 0; i < msize; ++i) {
      rnx_add_avx(nn, res + i * res_sl, a + i * a_sl, b + i * b_sl);
    }
    for (uint64_t i = msize; i < nsize; ++i) {
      memcpy(res + i * res_sl, a + i * a_sl, nn * sizeof(double));
    }
    for (uint64_t i = nsize; i < res_size; ++i) {
      memset(res + i * res_sl, 0, nn * sizeof(double));
    }
  }
}

/** @brief sets res = -a */
EXPORT void vec_rnx_negate_avx(                       //
    const MOD_RNX* module,                            // N
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl   // a
) {
  const uint64_t nn = module->n;
  const uint64_t msize = res_size < a_size ? res_size : a_size;
  for (uint64_t i = 0; i < msize; ++i) {
    rnx_negate_avx(nn, res + i * res_sl, a + i * a_sl);
  }
  for (uint64_t i = msize; i < res_size; ++i) {
    memset(res + i * res_sl, 0, nn * sizeof(double));
  }
}

/** @brief sets res = a - b */
EXPORT void vec_rnx_sub_avx(                          //
    const MOD_RNX* module,                            // N
    double* res, uint64_t res_size, uint64_t res_sl,  // res
    const double* a, uint64_t a_size, uint64_t a_sl,  // a
    const double* b, uint64_t b_size, uint64_t b_sl   // b
) {
  const uint64_t nn = module->n;
  if (a_size < b_size) {
    const uint64_t msize = res_size < a_size ? res_size : a_size;
    const uint64_t nsize = res_size < b_size ? res_size : b_size;
    for (uint64_t i = 0; i < msize; ++i) {
      rnx_sub_avx(nn, res + i * res_sl, a + i * a_sl, b + i * b_sl);
    }
    for (uint64_t i = msize; i < nsize; ++i) {
      rnx_negate_avx(nn, res + i * res_sl, b + i * b_sl);
    }
    for (uint64_t i = nsize; i < res_size; ++i) {
      memset(res + i * res_sl, 0, nn * sizeof(double));
    }
  } else {
    const uint64_t msize = res_size < b_size ? res_size : b_size;
    const uint64_t nsize = res_size < a_size ? res_size : a_size;
    for (uint64_t i = 0; i < msize; ++i) {
      rnx_sub_avx(nn, res + i * res_sl, a + i * a_sl, b + i * b_sl);
    }
    for (uint64_t i = msize; i < nsize; ++i) {
      memcpy(res + i * res_sl, a + i * a_sl, nn * sizeof(double));
    }
    for (uint64_t i = nsize; i < res_size; ++i) {
      memset(res + i * res_sl, 0, nn * sizeof(double));
    }
  }
}
