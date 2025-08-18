#include <gtest/gtest.h>

#include <cmath>

#include "spqlios/cplx/cplx_fft_internal.h"
#include "spqlios/cplx/cplx_fft_private.h"

#ifdef __x86_64__
TEST(fft, cplx_from_znx32_ref_vs_fma) {
  const uint32_t m = 128;
  int32_t* src = (int32_t*)spqlios_alloc_custom_align(32, 10 * m * sizeof(int32_t));
  CPLX* dst1 = (CPLX*)(src + 2 * m);
  CPLX* dst2 = (CPLX*)(src + 6 * m);
  for (uint64_t i = 0; i < 2 * m; ++i) {
    src[i] = rand() - RAND_MAX / 2;
  }
  CPLX_FROM_ZNX32_PRECOMP precomp;
  precomp.m = m;
  cplx_from_znx32_ref(&precomp, dst1, src);
  // cplx_from_znx32_simple(m, 32, dst1, src);
  cplx_from_znx32_avx2_fma(&precomp, dst2, src);
  for (uint64_t i = 0; i < m; ++i) {
    ASSERT_EQ(dst1[i][0], dst2[i][0]);
    ASSERT_EQ(dst1[i][1], dst2[i][1]);
  }
  spqlios_free(src);
}
#endif

#ifdef __x86_64__
TEST(fft, cplx_from_tnx32_ref_vs_fma) {
  const uint32_t m = 128;
  int32_t* src = (int32_t*)spqlios_alloc_custom_align(32, 10 * m * sizeof(int32_t));
  CPLX* dst1 = (CPLX*)(src + 2 * m);
  CPLX* dst2 = (CPLX*)(src + 6 * m);
  for (uint64_t i = 0; i < 2 * m; ++i) {
    src[i] = rand() + (rand() << 20);
  }
  CPLX_FROM_TNX32_PRECOMP precomp;
  precomp.m = m;
  cplx_from_tnx32_ref(&precomp, dst1, src);
  // cplx_from_tnx32_simple(m, dst1, src);
  cplx_from_tnx32_avx2_fma(&precomp, dst2, src);
  for (uint64_t i = 0; i < m; ++i) {
    ASSERT_EQ(dst1[i][0], dst2[i][0]);
    ASSERT_EQ(dst1[i][1], dst2[i][1]);
  }
  spqlios_free(src);
}
#endif

#ifdef __x86_64__
TEST(fft, cplx_to_tnx32_ref_vs_fma) {
  for (const uint32_t m : {8, 128, 1024, 65536}) {
    for (const double divisor : {double(1), double(m), double(0.5)}) {
      CPLX* src = (CPLX*)spqlios_alloc_custom_align(32, 10 * m * sizeof(int32_t));
      int32_t* dst1 = (int32_t*)(src + m);
      int32_t* dst2 = (int32_t*)(src + 2 * m);
      for (uint64_t i = 0; i < 2 * m; ++i) {
        src[i][0] = (rand() / double(RAND_MAX) - 0.5) * pow(2., 19 - (rand() % 60)) * divisor;
        src[i][1] = (rand() / double(RAND_MAX) - 0.5) * pow(2., 19 - (rand() % 60)) * divisor;
      }
      CPLX_TO_TNX32_PRECOMP precomp;
      precomp.m = m;
      precomp.divisor = divisor;
      cplx_to_tnx32_ref(&precomp, dst1, src);
      cplx_to_tnx32_avx2_fma(&precomp, dst2, src);
      // cplx_to_tnx32_simple(m, divisor, 18, dst2, src);
      for (uint64_t i = 0; i < 2 * m; ++i) {
        double truevalue =
            (src[i % m][i / m] / divisor - floor(src[i % m][i / m] / divisor + 0.5)) * (INT64_C(1) << 32);
        if (fabs(truevalue - floor(truevalue)) == 0.5) {
          // ties can differ by 0, 1 or -1
          ASSERT_LE(abs(dst1[i] - dst2[i]), 0)
              << i << " " << dst1[i] << " " << dst2[i] << " " << truevalue << std::endl;
        } else {
          // otherwise, we should have equality
          ASSERT_LE(abs(dst1[i] - dst2[i]), 0)
              << i << " " << dst1[i] << " " << dst2[i] << " " << truevalue << std::endl;
        }
      }
      spqlios_free(src);
    }
  }
}
#endif
