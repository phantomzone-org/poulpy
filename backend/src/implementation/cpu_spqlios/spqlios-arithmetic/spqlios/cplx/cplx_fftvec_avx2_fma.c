#include <immintrin.h>
#include <string.h>

#include "cplx_fft_internal.h"
#include "cplx_fft_private.h"

typedef double D4MEM[4];

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
rr += @;  // ONCE
aa += @;  // ONCE
bb += @;  // ONCE
END_TEMPLATE
     */
    // BEGIN_INTERLEAVE 4
    // This block is automatically generated from the template above
    // by the interleave.pl script. Please do not edit by hand
    const __m256d ari0 = _mm256_loadu_pd(aa[0]);
    const __m256d ari1 = _mm256_loadu_pd(aa[1]);
    const __m256d ari2 = _mm256_loadu_pd(aa[2]);
    const __m256d ari3 = _mm256_loadu_pd(aa[3]);
    const __m256d bri0 = _mm256_loadu_pd(bb[0]);
    const __m256d bri1 = _mm256_loadu_pd(bb[1]);
    const __m256d bri2 = _mm256_loadu_pd(bb[2]);
    const __m256d bri3 = _mm256_loadu_pd(bb[3]);
    const __m256d bir0 = _mm256_shuffle_pd(bri0, bri0, 5);   // conj of b
    const __m256d bir1 = _mm256_shuffle_pd(bri1, bri1, 5);   // conj of b
    const __m256d bir2 = _mm256_shuffle_pd(bri2, bri2, 5);   // conj of b
    const __m256d bir3 = _mm256_shuffle_pd(bri3, bri3, 5);   // conj of b
    const __m256d aii0 = _mm256_shuffle_pd(ari0, ari0, 15);  // im of a
    const __m256d aii1 = _mm256_shuffle_pd(ari1, ari1, 15);  // im of a
    const __m256d aii2 = _mm256_shuffle_pd(ari2, ari2, 15);  // im of a
    const __m256d aii3 = _mm256_shuffle_pd(ari3, ari3, 15);  // im of a
    const __m256d pro0 = _mm256_mul_pd(aii0, bir0);
    const __m256d pro1 = _mm256_mul_pd(aii1, bir1);
    const __m256d pro2 = _mm256_mul_pd(aii2, bir2);
    const __m256d pro3 = _mm256_mul_pd(aii3, bir3);
    const __m256d arr0 = _mm256_shuffle_pd(ari0, ari0, 0);  // rr of a
    const __m256d arr1 = _mm256_shuffle_pd(ari1, ari1, 0);  // rr of a
    const __m256d arr2 = _mm256_shuffle_pd(ari2, ari2, 0);  // rr of a
    const __m256d arr3 = _mm256_shuffle_pd(ari3, ari3, 0);  // rr of a
    const __m256d res0 = _mm256_fmaddsub_pd(arr0, bri0, pro0);
    const __m256d res1 = _mm256_fmaddsub_pd(arr1, bri1, pro1);
    const __m256d res2 = _mm256_fmaddsub_pd(arr2, bri2, pro2);
    const __m256d res3 = _mm256_fmaddsub_pd(arr3, bri3, pro3);
    _mm256_storeu_pd(rr[0], res0);
    _mm256_storeu_pd(rr[1], res1);
    _mm256_storeu_pd(rr[2], res2);
    _mm256_storeu_pd(rr[3], res3);
    rr += 4;  // ONCE
    aa += 4;  // ONCE
    bb += 4;  // ONCE
              // END_INTERLEAVE
  } while (aa < aend);
}

EXPORT void cplx_fftvec_addmul_fma(const CPLX_FFTVEC_ADDMUL_PRECOMP* precomp, void* r, const void* a, const void* b) {
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
const __m256d rri% = _mm256_loadu_pd(rr[%]);
const __m256d bir% = _mm256_shuffle_pd(bri%,bri%, 5);
const __m256d aii% = _mm256_shuffle_pd(ari%,ari%, 15);
const __m256d pro% = _mm256_fmaddsub_pd(aii%,bir%,rri%);
const __m256d arr% = _mm256_shuffle_pd(ari%,ari%, 0);
const __m256d res% = _mm256_fmaddsub_pd(arr%,bri%,pro%);
_mm256_storeu_pd(rr[%],res%);
rr += @; // ONCE
aa += @; // ONCE
bb += @; // ONCE
END_TEMPLATE
     */
    // BEGIN_INTERLEAVE 2
    // This block is automatically generated from the template above
    // by the interleave.pl script. Please do not edit by hand
    const __m256d ari0 = _mm256_loadu_pd(aa[0]);
    const __m256d ari1 = _mm256_loadu_pd(aa[1]);
    const __m256d bri0 = _mm256_loadu_pd(bb[0]);
    const __m256d bri1 = _mm256_loadu_pd(bb[1]);
    const __m256d rri0 = _mm256_loadu_pd(rr[0]);
    const __m256d rri1 = _mm256_loadu_pd(rr[1]);
    const __m256d bir0 = _mm256_shuffle_pd(bri0, bri0, 5);
    const __m256d bir1 = _mm256_shuffle_pd(bri1, bri1, 5);
    const __m256d aii0 = _mm256_shuffle_pd(ari0, ari0, 15);
    const __m256d aii1 = _mm256_shuffle_pd(ari1, ari1, 15);
    const __m256d pro0 = _mm256_fmaddsub_pd(aii0, bir0, rri0);
    const __m256d pro1 = _mm256_fmaddsub_pd(aii1, bir1, rri1);
    const __m256d arr0 = _mm256_shuffle_pd(ari0, ari0, 0);
    const __m256d arr1 = _mm256_shuffle_pd(ari1, ari1, 0);
    const __m256d res0 = _mm256_fmaddsub_pd(arr0, bri0, pro0);
    const __m256d res1 = _mm256_fmaddsub_pd(arr1, bri1, pro1);
    _mm256_storeu_pd(rr[0], res0);
    _mm256_storeu_pd(rr[1], res1);
    rr += 2;  // ONCE
    aa += 2;  // ONCE
    bb += 2;  // ONCE
              // END_INTERLEAVE
  } while (aa < aend);
}

EXPORT void cplx_fftvec_bitwiddle_fma(const CPLX_FFTVEC_BITWIDDLE_PRECOMP* precomp, void* a, uint64_t slicea,
                                      const void* omg) {
  const uint32_t m = precomp->m;
  const uint64_t OFFSET = slicea / sizeof(D4MEM);
  D4MEM* aa = (D4MEM*)a;
  const double(*const aend)[4] = aa + (m >> 1);
  const __m256d om = _mm256_loadu_pd(omg);
  const __m256d om1rr = _mm256_shuffle_pd(om, om, 0);
  const __m256d om1ii = _mm256_shuffle_pd(om, om, 15);
  const __m256d om2rr = _mm256_shuffle_pd(om, om, 0);
  const __m256d om2ii = _mm256_shuffle_pd(om, om, 0);
  const __m256d om3rr = _mm256_shuffle_pd(om, om, 15);
  const __m256d om3ii = _mm256_shuffle_pd(om, om, 15);
  do {
    /*
BEGIN_TEMPLATE
__m256d ari% = _mm256_loadu_pd(aa[%]);
__m256d bri% = _mm256_loadu_pd((aa+OFFSET)[%]);
__m256d cri% = _mm256_loadu_pd((aa+2*OFFSET)[%]);
__m256d dri% = _mm256_loadu_pd((aa+3*OFFSET)[%]);
__m256d pa% = _mm256_shuffle_pd(cri%,cri%,5);
__m256d pb% = _mm256_shuffle_pd(dri%,dri%,5);
pa% = _mm256_mul_pd(pa%,om1ii);
pb% = _mm256_mul_pd(pb%,om1ii);
pa% = _mm256_fmaddsub_pd(cri%,om1rr,pa%);
pb% = _mm256_fmaddsub_pd(dri%,om1rr,pb%);
cri% = _mm256_sub_pd(ari%,pa%);
dri% = _mm256_sub_pd(bri%,pb%);
ari% = _mm256_add_pd(ari%,pa%);
bri% = _mm256_add_pd(bri%,pb%);
pa% = _mm256_shuffle_pd(bri%,bri%,5);
pb% = _mm256_shuffle_pd(dri%,dri%,5);
pa% = _mm256_mul_pd(pa%,om2ii);
pb% = _mm256_mul_pd(pb%,om3ii);
pa% = _mm256_fmaddsub_pd(bri%,om2rr,pa%);
pb% = _mm256_fmaddsub_pd(dri%,om3rr,pb%);
bri% = _mm256_sub_pd(ari%,pa%);
dri% = _mm256_sub_pd(cri%,pb%);
ari% = _mm256_add_pd(ari%,pa%);
cri% = _mm256_add_pd(cri%,pb%);
_mm256_storeu_pd(aa[%], ari%);
_mm256_storeu_pd((aa+OFFSET)[%],bri%);
_mm256_storeu_pd((aa+2*OFFSET)[%],cri%);
_mm256_storeu_pd((aa+3*OFFSET)[%],dri%);
aa += @; // ONCE
END_TEMPLATE
     */
    // BEGIN_INTERLEAVE 1
    // This block is automatically generated from the template above
    // by the interleave.pl script. Please do not edit by hand
    __m256d ari0 = _mm256_loadu_pd(aa[0]);
    __m256d bri0 = _mm256_loadu_pd((aa + OFFSET)[0]);
    __m256d cri0 = _mm256_loadu_pd((aa + 2 * OFFSET)[0]);
    __m256d dri0 = _mm256_loadu_pd((aa + 3 * OFFSET)[0]);
    __m256d pa0 = _mm256_shuffle_pd(cri0, cri0, 5);
    __m256d pb0 = _mm256_shuffle_pd(dri0, dri0, 5);
    pa0 = _mm256_mul_pd(pa0, om1ii);
    pb0 = _mm256_mul_pd(pb0, om1ii);
    pa0 = _mm256_fmaddsub_pd(cri0, om1rr, pa0);
    pb0 = _mm256_fmaddsub_pd(dri0, om1rr, pb0);
    cri0 = _mm256_sub_pd(ari0, pa0);
    dri0 = _mm256_sub_pd(bri0, pb0);
    ari0 = _mm256_add_pd(ari0, pa0);
    bri0 = _mm256_add_pd(bri0, pb0);
    pa0 = _mm256_shuffle_pd(bri0, bri0, 5);
    pb0 = _mm256_shuffle_pd(dri0, dri0, 5);
    pa0 = _mm256_mul_pd(pa0, om2ii);
    pb0 = _mm256_mul_pd(pb0, om3ii);
    pa0 = _mm256_fmaddsub_pd(bri0, om2rr, pa0);
    pb0 = _mm256_fmaddsub_pd(dri0, om3rr, pb0);
    bri0 = _mm256_sub_pd(ari0, pa0);
    dri0 = _mm256_sub_pd(cri0, pb0);
    ari0 = _mm256_add_pd(ari0, pa0);
    cri0 = _mm256_add_pd(cri0, pb0);
    _mm256_storeu_pd(aa[0], ari0);
    _mm256_storeu_pd((aa + OFFSET)[0], bri0);
    _mm256_storeu_pd((aa + 2 * OFFSET)[0], cri0);
    _mm256_storeu_pd((aa + 3 * OFFSET)[0], dri0);
    aa += 1;  // ONCE
              // END_INTERLEAVE
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
    // This block is automatically generated from the template above
    // by the interleave.pl script. Please do not edit by hand
    const __m256d ari0 = _mm256_loadu_pd(aa[0]);
    const __m256d ari1 = _mm256_loadu_pd(aa[1]);
    const __m256d ari2 = _mm256_loadu_pd(aa[2]);
    const __m256d ari3 = _mm256_loadu_pd(aa[3]);
    const __m256d bri0 = _mm256_loadu_pd(bb[0]);
    const __m256d bri1 = _mm256_loadu_pd(bb[1]);
    const __m256d bri2 = _mm256_loadu_pd(bb[2]);
    const __m256d bri3 = _mm256_loadu_pd(bb[3]);
    const __m256d sum0 = _mm256_add_pd(ari0, bri0);
    const __m256d sum1 = _mm256_add_pd(ari1, bri1);
    const __m256d sum2 = _mm256_add_pd(ari2, bri2);
    const __m256d sum3 = _mm256_add_pd(ari3, bri3);
    const __m256d rri0 = _mm256_loadu_pd(rr[0]);
    const __m256d rri1 = _mm256_loadu_pd(rr[1]);
    const __m256d rri2 = _mm256_loadu_pd(rr[2]);
    const __m256d rri3 = _mm256_loadu_pd(rr[3]);
    const __m256d res0 = _mm256_sub_pd(rri0, sum0);
    const __m256d res1 = _mm256_sub_pd(rri1, sum1);
    const __m256d res2 = _mm256_sub_pd(rri2, sum2);
    const __m256d res3 = _mm256_sub_pd(rri3, sum3);
    _mm256_storeu_pd(rr[0], res0);
    _mm256_storeu_pd(rr[1], res1);
    _mm256_storeu_pd(rr[2], res2);
    _mm256_storeu_pd(rr[3], res3);
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
    rr += @;  // ONCE
    aa += @;  // ONCE
    bb += @;  // ONCE
    END_TEMPLATE
     */
    // BEGIN_INTERLEAVE 4
    // This block is automatically generated from the template above
    // by the interleave.pl script. Please do not edit by hand
    const __m256d ari0 = _mm256_loadu_pd(aa[0]);
    const __m256d ari1 = _mm256_loadu_pd(aa[1]);
    const __m256d ari2 = _mm256_loadu_pd(aa[2]);
    const __m256d ari3 = _mm256_loadu_pd(aa[3]);
    const __m256d bri0 = _mm256_loadu_pd(bb[0]);
    const __m256d bri1 = _mm256_loadu_pd(bb[1]);
    const __m256d bri2 = _mm256_loadu_pd(bb[2]);
    const __m256d bri3 = _mm256_loadu_pd(bb[3]);
    const __m256d res0 = _mm256_add_pd(ari0, bri0);
    const __m256d res1 = _mm256_add_pd(ari1, bri1);
    const __m256d res2 = _mm256_add_pd(ari2, bri2);
    const __m256d res3 = _mm256_add_pd(ari3, bri3);
    _mm256_storeu_pd(rr[0], res0);
    _mm256_storeu_pd(rr[1], res1);
    _mm256_storeu_pd(rr[2], res2);
    _mm256_storeu_pd(rr[3], res3);
    rr += 4;  // ONCE
    aa += 4;  // ONCE
    bb += 4;  // ONCE
    // END_INTERLEAVE
  } while (aa < aend);
}

EXPORT void cplx_fftvec_twiddle_fma(const CPLX_FFTVEC_TWIDDLE_PRECOMP* precomp, void* a, void* b, const void* omg) {
  const uint32_t m = precomp->m;
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
bb += @; // ONCE
aa += @; // ONCE
END_TEMPLATE
     */
    // BEGIN_INTERLEAVE 4
    // This block is automatically generated from the template above
    // by the interleave.pl script. Please do not edit by hand
    const __m256d bri0 = _mm256_loadu_pd(bb[0]);
    const __m256d bri1 = _mm256_loadu_pd(bb[1]);
    const __m256d bri2 = _mm256_loadu_pd(bb[2]);
    const __m256d bri3 = _mm256_loadu_pd(bb[3]);
    const __m256d bir0 = _mm256_shuffle_pd(bri0, bri0, 5);
    const __m256d bir1 = _mm256_shuffle_pd(bri1, bri1, 5);
    const __m256d bir2 = _mm256_shuffle_pd(bri2, bri2, 5);
    const __m256d bir3 = _mm256_shuffle_pd(bri3, bri3, 5);
    __m256d p0 = _mm256_mul_pd(bir0, omii);
    __m256d p1 = _mm256_mul_pd(bir1, omii);
    __m256d p2 = _mm256_mul_pd(bir2, omii);
    __m256d p3 = _mm256_mul_pd(bir3, omii);
    p0 = _mm256_fmaddsub_pd(bri0, omrr, p0);
    p1 = _mm256_fmaddsub_pd(bri1, omrr, p1);
    p2 = _mm256_fmaddsub_pd(bri2, omrr, p2);
    p3 = _mm256_fmaddsub_pd(bri3, omrr, p3);
    const __m256d ari0 = _mm256_loadu_pd(aa[0]);
    const __m256d ari1 = _mm256_loadu_pd(aa[1]);
    const __m256d ari2 = _mm256_loadu_pd(aa[2]);
    const __m256d ari3 = _mm256_loadu_pd(aa[3]);
    _mm256_storeu_pd(aa[0], _mm256_add_pd(ari0, p0));
    _mm256_storeu_pd(aa[1], _mm256_add_pd(ari1, p1));
    _mm256_storeu_pd(aa[2], _mm256_add_pd(ari2, p2));
    _mm256_storeu_pd(aa[3], _mm256_add_pd(ari3, p3));
    _mm256_storeu_pd(bb[0], _mm256_sub_pd(ari0, p0));
    _mm256_storeu_pd(bb[1], _mm256_sub_pd(ari1, p1));
    _mm256_storeu_pd(bb[2], _mm256_sub_pd(ari2, p2));
    _mm256_storeu_pd(bb[3], _mm256_sub_pd(ari3, p3));
    bb += 4;  // ONCE
    aa += 4;  // ONCE
              // END_INTERLEAVE
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
rr += @; // ONCE
aa += @; // ONCE
END_TEMPLATE
     */
    // BEGIN_INTERLEAVE 4
    // This block is automatically generated from the template above
    // by the interleave.pl script. Please do not edit by hand
    const __m256d ari0 = _mm256_loadu_pd(aa[0]);
    const __m256d ari1 = _mm256_loadu_pd(aa[1]);
    const __m256d ari2 = _mm256_loadu_pd(aa[2]);
    const __m256d ari3 = _mm256_loadu_pd(aa[3]);
    _mm256_storeu_pd(rr[0], ari0);
    _mm256_storeu_pd(rr[1], ari1);
    _mm256_storeu_pd(rr[2], ari2);
    _mm256_storeu_pd(rr[3], ari3);
    rr += 4;  // ONCE
    aa += 4;  // ONCE
              // END_INTERLEAVE
  } while (aa < aend);
}
