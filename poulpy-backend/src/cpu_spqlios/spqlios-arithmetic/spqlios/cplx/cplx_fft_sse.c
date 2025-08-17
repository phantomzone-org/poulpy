#include <immintrin.h>

#include "cplx_fft_internal.h"
#include "cplx_fft_private.h"

typedef double D2MEM[2];

EXPORT void cplx_fftvec_addmul_sse(const CPLX_FFTVEC_ADDMUL_PRECOMP* precomp, void* r, const void* a, const void* b) {
  const uint32_t m = precomp->m;
  const D2MEM* aa = (D2MEM*)a;
  const D2MEM* bb = (D2MEM*)b;
  D2MEM* rr = (D2MEM*)r;
  const D2MEM* const aend = aa + m;
  do {
    /*
BEGIN_TEMPLATE
const __m128d ari% = _mm_loadu_pd(aa[%]);
const __m128d bri% = _mm_loadu_pd(bb[%]);
const __m128d rri% = _mm_loadu_pd(rr[%]);
const __m128d bir% = _mm_shuffle_pd(bri%,bri%, 5);
const __m128d aii% = _mm_shuffle_pd(ari%,ari%, 15);
const __m128d pro% = _mm_fmaddsub_pd(aii%,bir%,rri%);
const __m128d arr% = _mm_shuffle_pd(ari%,ari%, 0);
const __m128d res% = _mm_fmaddsub_pd(arr%,bri%,pro%);
_mm_storeu_pd(rr[%],res%);
rr += @; // ONCE
aa += @; // ONCE
bb += @; // ONCE
END_TEMPLATE
     */
    // BEGIN_INTERLEAVE 2
    const __m128d ari0 = _mm_loadu_pd(aa[0]);
    const __m128d ari1 = _mm_loadu_pd(aa[1]);
    const __m128d bri0 = _mm_loadu_pd(bb[0]);
    const __m128d bri1 = _mm_loadu_pd(bb[1]);
    const __m128d rri0 = _mm_loadu_pd(rr[0]);
    const __m128d rri1 = _mm_loadu_pd(rr[1]);
    const __m128d bir0 = _mm_shuffle_pd(bri0, bri0, 0b01);
    const __m128d bir1 = _mm_shuffle_pd(bri1, bri1, 0b01);
    const __m128d aii0 = _mm_shuffle_pd(ari0, ari0, 0b11);
    const __m128d aii1 = _mm_shuffle_pd(ari1, ari1, 0b11);
    const __m128d pro0 = _mm_fmaddsub_pd(aii0, bir0, rri0);
    const __m128d pro1 = _mm_fmaddsub_pd(aii1, bir1, rri1);
    const __m128d arr0 = _mm_shuffle_pd(ari0, ari0, 0b00);
    const __m128d arr1 = _mm_shuffle_pd(ari1, ari1, 0b00);
    const __m128d res0 = _mm_fmaddsub_pd(arr0, bri0, pro0);
    const __m128d res1 = _mm_fmaddsub_pd(arr1, bri1, pro1);
    _mm_storeu_pd(rr[0], res0);
    _mm_storeu_pd(rr[1], res1);
    rr += 2;  // ONCE
    aa += 2;  // ONCE
    bb += 2;  // ONCE
              // END_INTERLEAVE
  } while (aa < aend);
}

#if 0
EXPORT void cplx_fftvec_mul_fma(const CPLX_FFTVEC_MUL_PRECOMP* precomp, void* r, const void* a, const void* b) {
  const uint32_t m = precomp->m;
  const double(*aa)[4] = (double(*)[4])a;
  const double(*bb)[4] = (double(*)[4])b;
  double(*rr)[4] = (double(*)[4])r;
  const double(*const aend)[4] = aa + (m >> 1);
  do {
    /*
BEGIN_TEMPLATE
const __m256d ari% = _mm256_loadu_pd(aa[%]);
const __m256d bri% = _mm256_loadu_pd(bb[%]);
const __m256d bir% = _mm256_shuffle_pd(bri%,bri%, 5); // conj of b
const __m256d aii% = _mm256_shuffle_pd(ari%,ari%, 15); // im of a
const __m256d pro% = _mm256_mul_pd(aii%,bir%);
const __m256d arr% = _mm256_shuffle_pd(ari%,ari%, 0); // rr of a
const __m256d res% = _mm256_fmaddsub_pd(arr%,bri%,pro%);
_mm256_storeu_pd(rr[%],res%);
END_TEMPLATE
     */
    // BEGIN_INTERLEAVE 4
const __m256d ari0 = _mm256_loadu_pd(aa[0]);
const __m256d ari1 = _mm256_loadu_pd(aa[1]);
const __m256d ari2 = _mm256_loadu_pd(aa[2]);
const __m256d ari3 = _mm256_loadu_pd(aa[3]);
const __m256d bri0 = _mm256_loadu_pd(bb[0]);
const __m256d bri1 = _mm256_loadu_pd(bb[1]);
const __m256d bri2 = _mm256_loadu_pd(bb[2]);
const __m256d bri3 = _mm256_loadu_pd(bb[3]);
const __m256d bir0 = _mm256_shuffle_pd(bri0,bri0, 5); // conj of b
const __m256d bir1 = _mm256_shuffle_pd(bri1,bri1, 5); // conj of b
const __m256d bir2 = _mm256_shuffle_pd(bri2,bri2, 5); // conj of b
const __m256d bir3 = _mm256_shuffle_pd(bri3,bri3, 5); // conj of b
const __m256d aii0 = _mm256_shuffle_pd(ari0,ari0, 15); // im of a
const __m256d aii1 = _mm256_shuffle_pd(ari1,ari1, 15); // im of a
const __m256d aii2 = _mm256_shuffle_pd(ari2,ari2, 15); // im of a
const __m256d aii3 = _mm256_shuffle_pd(ari3,ari3, 15); // im of a
const __m256d pro0 = _mm256_mul_pd(aii0,bir0);
const __m256d pro1 = _mm256_mul_pd(aii1,bir1);
const __m256d pro2 = _mm256_mul_pd(aii2,bir2);
const __m256d pro3 = _mm256_mul_pd(aii3,bir3);
const __m256d arr0 = _mm256_shuffle_pd(ari0,ari0, 0); // rr of a
const __m256d arr1 = _mm256_shuffle_pd(ari1,ari1, 0); // rr of a
const __m256d arr2 = _mm256_shuffle_pd(ari2,ari2, 0); // rr of a
const __m256d arr3 = _mm256_shuffle_pd(ari3,ari3, 0); // rr of a
const __m256d res0 = _mm256_fmaddsub_pd(arr0,bri0,pro0);
const __m256d res1 = _mm256_fmaddsub_pd(arr1,bri1,pro1);
const __m256d res2 = _mm256_fmaddsub_pd(arr2,bri2,pro2);
const __m256d res3 = _mm256_fmaddsub_pd(arr3,bri3,pro3);
_mm256_storeu_pd(rr[0],res0);
_mm256_storeu_pd(rr[1],res1);
_mm256_storeu_pd(rr[2],res2);
_mm256_storeu_pd(rr[3],res3);
    // END_INTERLEAVE
    rr += 4;
    aa += 4;
    bb += 4;
  } while (aa < aend);
}

EXPORT void cplx_fftvec_add_fma(uint32_t m, void* r, const void* a, const void* b) {
  const double(*aa)[4] = (double(*)[4])a;
  const double(*bb)[4] = (double(*)[4])b;
  double(*rr)[4] = (double(*)[4])r;
  const double(*const aend)[4] = aa + (m >> 1);
  do {
    /*
BEGIN_TEMPLATE
const __m256d ari% = _mm256_loadu_pd(aa[%]);
const __m256d bri% = _mm256_loadu_pd(bb[%]);
const __m256d res% = _mm256_add_pd(ari%,bri%);
_mm256_storeu_pd(rr[%],res%);
END_TEMPLATE
     */
    // BEGIN_INTERLEAVE 4
const __m256d ari0 = _mm256_loadu_pd(aa[0]);
const __m256d ari1 = _mm256_loadu_pd(aa[1]);
const __m256d ari2 = _mm256_loadu_pd(aa[2]);
const __m256d ari3 = _mm256_loadu_pd(aa[3]);
const __m256d bri0 = _mm256_loadu_pd(bb[0]);
const __m256d bri1 = _mm256_loadu_pd(bb[1]);
const __m256d bri2 = _mm256_loadu_pd(bb[2]);
const __m256d bri3 = _mm256_loadu_pd(bb[3]);
const __m256d res0 = _mm256_add_pd(ari0,bri0);
const __m256d res1 = _mm256_add_pd(ari1,bri1);
const __m256d res2 = _mm256_add_pd(ari2,bri2);
const __m256d res3 = _mm256_add_pd(ari3,bri3);
_mm256_storeu_pd(rr[0],res0);
_mm256_storeu_pd(rr[1],res1);
_mm256_storeu_pd(rr[2],res2);
_mm256_storeu_pd(rr[3],res3);
    // END_INTERLEAVE
    rr += 4;
    aa += 4;
    bb += 4;
  } while (aa < aend);
}

EXPORT void cplx_fftvec_sub2_to_fma(uint32_t m, void* r, const void* a, const void* b) {
  const double(*aa)[4] = (double(*)[4])a;
  const double(*bb)[4] = (double(*)[4])b;
  double(*rr)[4] = (double(*)[4])r;
  const double(*const aend)[4] = aa + (m >> 1);
  do {
    /*
BEGIN_TEMPLATE
const __m256d ari% = _mm256_loadu_pd(aa[%]);
const __m256d bri% = _mm256_loadu_pd(bb[%]);
const __m256d sum% = _mm256_add_pd(ari%,bri%);
const __m256d rri% = _mm256_loadu_pd(rr[%]);
const __m256d res% = _mm256_sub_pd(rri%,sum%);
_mm256_storeu_pd(rr[%],res%);
END_TEMPLATE
     */
    // BEGIN_INTERLEAVE 4
const __m256d ari0 = _mm256_loadu_pd(aa[0]);
const __m256d ari1 = _mm256_loadu_pd(aa[1]);
const __m256d ari2 = _mm256_loadu_pd(aa[2]);
const __m256d ari3 = _mm256_loadu_pd(aa[3]);
const __m256d bri0 = _mm256_loadu_pd(bb[0]);
const __m256d bri1 = _mm256_loadu_pd(bb[1]);
const __m256d bri2 = _mm256_loadu_pd(bb[2]);
const __m256d bri3 = _mm256_loadu_pd(bb[3]);
const __m256d sum0 = _mm256_add_pd(ari0,bri0);
const __m256d sum1 = _mm256_add_pd(ari1,bri1);
const __m256d sum2 = _mm256_add_pd(ari2,bri2);
const __m256d sum3 = _mm256_add_pd(ari3,bri3);
const __m256d rri0 = _mm256_loadu_pd(rr[0]);
const __m256d rri1 = _mm256_loadu_pd(rr[1]);
const __m256d rri2 = _mm256_loadu_pd(rr[2]);
const __m256d rri3 = _mm256_loadu_pd(rr[3]);
const __m256d res0 = _mm256_sub_pd(rri0,sum0);
const __m256d res1 = _mm256_sub_pd(rri1,sum1);
const __m256d res2 = _mm256_sub_pd(rri2,sum2);
const __m256d res3 = _mm256_sub_pd(rri3,sum3);
_mm256_storeu_pd(rr[0],res0);
_mm256_storeu_pd(rr[1],res1);
_mm256_storeu_pd(rr[2],res2);
_mm256_storeu_pd(rr[3],res3);
    // END_INTERLEAVE
    rr += 4;
    aa += 4;
    bb += 4;
  } while (aa < aend);
}

EXPORT void cplx_fftvec_copy_fma(uint32_t m, void* r, const void* a) {
  const double(*aa)[4] = (double(*)[4])a;
  double(*rr)[4] = (double(*)[4])r;
  const double(*const aend)[4] = aa + (m >> 1);
  do {
    /*
BEGIN_TEMPLATE
const __m256d ari% = _mm256_loadu_pd(aa[%]);
_mm256_storeu_pd(rr[%],ari%);
END_TEMPLATE
     */
    // BEGIN_INTERLEAVE 4
const __m256d ari0 = _mm256_loadu_pd(aa[0]);
const __m256d ari1 = _mm256_loadu_pd(aa[1]);
const __m256d ari2 = _mm256_loadu_pd(aa[2]);
const __m256d ari3 = _mm256_loadu_pd(aa[3]);
_mm256_storeu_pd(rr[0],ari0);
_mm256_storeu_pd(rr[1],ari1);
_mm256_storeu_pd(rr[2],ari2);
_mm256_storeu_pd(rr[3],ari3);
    // END_INTERLEAVE
    rr += 4;
    aa += 4;
  } while (aa < aend);
}

EXPORT void cplx_fftvec_twiddle_fma(uint32_t m, void* a, void* b, const void* omg) {
  double(*aa)[4] = (double(*)[4])a;
  double(*bb)[4] = (double(*)[4])b;
  const double(*const aend)[4] = aa + (m >> 1);
  const __m256d om = _mm256_loadu_pd(omg);
  const __m256d omrr = _mm256_shuffle_pd(om, om, 0);
  const __m256d omii = _mm256_shuffle_pd(om, om, 15);
  do {
    /*
BEGIN_TEMPLATE
const __m256d bri% = _mm256_loadu_pd(bb[%]);
const __m256d bir% = _mm256_shuffle_pd(bri%,bri%,5);
__m256d p% = _mm256_mul_pd(bir%,omii);
p% = _mm256_fmaddsub_pd(bri%,omrr,p%);
const __m256d ari% = _mm256_loadu_pd(aa[%]);
_mm256_storeu_pd(aa[%],_mm256_add_pd(ari%,p%));
_mm256_storeu_pd(bb[%],_mm256_sub_pd(ari%,p%));
END_TEMPLATE
     */
    // BEGIN_INTERLEAVE 4
const __m256d bri0 = _mm256_loadu_pd(bb[0]);
const __m256d bri1 = _mm256_loadu_pd(bb[1]);
const __m256d bri2 = _mm256_loadu_pd(bb[2]);
const __m256d bri3 = _mm256_loadu_pd(bb[3]);
const __m256d bir0 = _mm256_shuffle_pd(bri0,bri0,5);
const __m256d bir1 = _mm256_shuffle_pd(bri1,bri1,5);
const __m256d bir2 = _mm256_shuffle_pd(bri2,bri2,5);
const __m256d bir3 = _mm256_shuffle_pd(bri3,bri3,5);
__m256d p0 = _mm256_mul_pd(bir0,omii);
__m256d p1 = _mm256_mul_pd(bir1,omii);
__m256d p2 = _mm256_mul_pd(bir2,omii);
__m256d p3 = _mm256_mul_pd(bir3,omii);
p0 = _mm256_fmaddsub_pd(bri0,omrr,p0);
p1 = _mm256_fmaddsub_pd(bri1,omrr,p1);
p2 = _mm256_fmaddsub_pd(bri2,omrr,p2);
p3 = _mm256_fmaddsub_pd(bri3,omrr,p3);
const __m256d ari0 = _mm256_loadu_pd(aa[0]);
const __m256d ari1 = _mm256_loadu_pd(aa[1]);
const __m256d ari2 = _mm256_loadu_pd(aa[2]);
const __m256d ari3 = _mm256_loadu_pd(aa[3]);
_mm256_storeu_pd(aa[0],_mm256_add_pd(ari0,p0));
_mm256_storeu_pd(aa[1],_mm256_add_pd(ari1,p1));
_mm256_storeu_pd(aa[2],_mm256_add_pd(ari2,p2));
_mm256_storeu_pd(aa[3],_mm256_add_pd(ari3,p3));
_mm256_storeu_pd(bb[0],_mm256_sub_pd(ari0,p0));
_mm256_storeu_pd(bb[1],_mm256_sub_pd(ari1,p1));
_mm256_storeu_pd(bb[2],_mm256_sub_pd(ari2,p2));
_mm256_storeu_pd(bb[3],_mm256_sub_pd(ari3,p3));
    // END_INTERLEAVE
    bb += 4;
    aa += 4;
  } while (aa < aend);
}

EXPORT void cplx_fftvec_innerprod_avx2_fma(const CPLX_FFTVEC_INNERPROD_PRECOMP* precomp, const int32_t ellbar,
                                   const uint64_t lda, const uint64_t ldb,
                                   void* r, const void* a, const void* b) {
  const uint32_t m = precomp->m;
  const uint32_t blk = precomp->blk;
  const uint32_t nblocks = precomp->nblocks;
  const CPLX* aa = (CPLX*)a;
  const CPLX* bb = (CPLX*)b;
  CPLX* rr = (CPLX*)r;
  const uint64_t ldda = lda >> 4;  // in CPLX
  const uint64_t lddb = ldb >> 4;
  if (m==0) {
    memset(r, 0, m*sizeof(CPLX));
    return;
  }
  for (uint32_t k=0; k<nblocks; ++k) {
    const uint64_t offset = k*blk;
    const CPLX* aaa = aa+offset;
    const CPLX* bbb = bb+offset;
    CPLX *rrr = rr+offset;
    cplx_fftvec_mul_fma(&precomp->mul_func, rrr, aaa, bbb);
    for (int32_t i=1; i<ellbar; ++i) {
      cplx_fftvec_addmul_fma(&precomp->addmul_func, rrr, aaa + i * ldda, bbb + i * lddb);
    }
  }
}
#endif
