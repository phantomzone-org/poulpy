#include <immintrin.h>
#include <stdio.h>

#include "cplx_fft_internal.h"
#include "cplx_fft_private.h"

typedef int32_t I8MEM[8];
typedef double D4MEM[4];

__always_inline void cplx_from_any_fma(uint64_t m, void* r, const int32_t* x, const __m256i C, const __m256d R) {
  const __m256i S = _mm256_set1_epi32(0x80000000);
  const I8MEM* inre = (I8MEM*)(x);
  const I8MEM* inim = (I8MEM*)(x+m);
  D4MEM* out = (D4MEM*) r;
  const uint64_t ms8 = m/8;
  for (uint32_t i=0; i<ms8; ++i) {
    __m256i rea = _mm256_loadu_si256((__m256i*) inre[0]);
    __m256i ima = _mm256_loadu_si256((__m256i*) inim[0]);
    rea = _mm256_add_epi32(rea, S);
    ima = _mm256_add_epi32(ima, S);
    __m256i tmpa = _mm256_unpacklo_epi32(rea, ima);
    __m256i tmpc = _mm256_unpackhi_epi32(rea, ima);
    __m256i cpla = _mm256_permute2x128_si256(tmpa,tmpc,0x20);
    __m256i cplc = _mm256_permute2x128_si256(tmpa,tmpc,0x31);
    tmpa = _mm256_unpacklo_epi32(cpla, C);
    __m256i tmpb = _mm256_unpackhi_epi32(cpla, C);
    tmpc = _mm256_unpacklo_epi32(cplc, C);
    __m256i tmpd = _mm256_unpackhi_epi32(cplc, C);
    cpla = _mm256_permute2x128_si256(tmpa,tmpb,0x20);
    __m256i cplb = _mm256_permute2x128_si256(tmpa,tmpb,0x31);
    cplc = _mm256_permute2x128_si256(tmpc,tmpd,0x20);
    __m256i cpld = _mm256_permute2x128_si256(tmpc,tmpd,0x31);
    __m256d dcpla = _mm256_sub_pd(_mm256_castsi256_pd(cpla), R);
    __m256d dcplb = _mm256_sub_pd(_mm256_castsi256_pd(cplb), R);
    __m256d dcplc = _mm256_sub_pd(_mm256_castsi256_pd(cplc), R);
    __m256d dcpld = _mm256_sub_pd(_mm256_castsi256_pd(cpld), R);
    _mm256_storeu_pd(out[0], dcpla);
    _mm256_storeu_pd(out[1], dcplb);
    _mm256_storeu_pd(out[2], dcplc);
    _mm256_storeu_pd(out[3], dcpld);
    inre += 1;
    inim += 1;
    out += 4;
  }
}

EXPORT void cplx_from_znx32_avx2_fma(const CPLX_FROM_ZNX32_PRECOMP* precomp, void* r, const int32_t* x) {
  //note: the hex code of 2^31 + 2^52 is 0x4330000080000000
  const __m256i C = _mm256_set1_epi32(0x43300000);
  const __m256d R = _mm256_set1_pd((INT64_C(1) << 31) + (INT64_C(1) << 52));
  // double XX =  INT64_C(1) + (INT64_C(1)<<31) + (INT64_C(1)<<52);
  //printf("\n\n%016lx\n", *(uint64_t*)&XX);
  //abort();
  const uint64_t m = precomp->m;
  cplx_from_any_fma(m, r, x, C, R);
}

EXPORT void cplx_from_tnx32_avx2_fma(const CPLX_FROM_TNX32_PRECOMP* precomp, void* r, const int32_t* x) {
  //note: the hex code of 2^-1 + 2^30 is 0x4130000080000000
  const __m256i C = _mm256_set1_epi32(0x41300000);
  const __m256d R = _mm256_set1_pd(0.5 + (INT64_C(1) << 20));
  // double XX =  (double)(INT64_C(1) + (INT64_C(1)<<31) + (INT64_C(1)<<52))/(INT64_C(1)<<32);
  //printf("\n\n%016lx\n", *(uint64_t*)&XX);
  //abort();
  const uint64_t m = precomp->m;
  cplx_from_any_fma(m, r, x, C, R);
}

EXPORT void cplx_to_tnx32_avx2_fma(const CPLX_TO_TNX32_PRECOMP* precomp, int32_t* r, const void* x) {
  const __m256d R = _mm256_set1_pd((0.5 + (INT64_C(3) << 19)) * precomp->divisor);
  const __m256i MASK = _mm256_set1_epi64x(0xFFFFFFFFUL);
  const __m256i S = _mm256_set1_epi32(0x80000000);
  //const __m256i IDX = _mm256_set_epi32(0,4,1,5,2,6,3,7);
  const __m256i IDX = _mm256_set_epi32(7,3,6,2,5,1,4,0);
  const uint64_t m = precomp->m;
  const uint64_t ms8 = m/8;
  I8MEM* outre = (I8MEM*) r;
  I8MEM* outim = (I8MEM*) (r+m);
  const D4MEM* in = x;
  // Note: this formula will only work if abs(in) < 2^32
  for (uint32_t i=0; i<ms8; ++i) {
    __m256d cpla = _mm256_loadu_pd(in[0]);
    __m256d cplb = _mm256_loadu_pd(in[1]);
    __m256d cplc = _mm256_loadu_pd(in[2]);
    __m256d cpld = _mm256_loadu_pd(in[3]);
    __m256i icpla = _mm256_castpd_si256(_mm256_add_pd(cpla, R));
    __m256i icplb = _mm256_castpd_si256(_mm256_add_pd(cplb, R));
    __m256i icplc = _mm256_castpd_si256(_mm256_add_pd(cplc, R));
    __m256i icpld = _mm256_castpd_si256(_mm256_add_pd(cpld, R));
    icpla = _mm256_or_si256(_mm256_and_si256(icpla, MASK), _mm256_slli_epi64(icplb, 32));
    icplc = _mm256_or_si256(_mm256_and_si256(icplc, MASK), _mm256_slli_epi64(icpld, 32));
    icpla = _mm256_xor_si256(icpla, S);
    icplc = _mm256_xor_si256(icplc, S);
    __m256i re = _mm256_unpacklo_epi64(icpla, icplc);
    __m256i im = _mm256_unpackhi_epi64(icpla, icplc);
    re = _mm256_permutevar8x32_epi32(re, IDX);
    im = _mm256_permutevar8x32_epi32(im, IDX);
    _mm256_storeu_si256((__m256i*)outre[0], re);
    _mm256_storeu_si256((__m256i*)outim[0], im);
    outre += 1;
    outim += 1;
    in += 4;
  }
}
