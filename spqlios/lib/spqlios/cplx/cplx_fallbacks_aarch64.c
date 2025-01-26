#include "cplx_fft_internal.h"
#include "cplx_fft_private.h"

EXPORT void cplx_fftvec_addmul_fma(const CPLX_FFTVEC_ADDMUL_PRECOMP* tables, void* r, const void* a, const void* b) {
  UNDEFINED();  // not defined for non x86 targets
}
EXPORT void cplx_fftvec_mul_fma(const CPLX_FFTVEC_MUL_PRECOMP* tables, void* r, const void* a, const void* b) {
  UNDEFINED();
}
EXPORT void cplx_fftvec_addmul_sse(const CPLX_FFTVEC_ADDMUL_PRECOMP* precomp, void* r, const void* a, const void* b) {
  UNDEFINED();
}
EXPORT void cplx_fftvec_addmul_avx512(const CPLX_FFTVEC_ADDMUL_PRECOMP* precomp, void* r, const void* a,
                                      const void* b) {
  UNDEFINED();
}
EXPORT void cplx_fft16_avx_fma(void* data, const void* omega) { UNDEFINED(); }
EXPORT void cplx_ifft16_avx_fma(void* data, const void* omega) { UNDEFINED(); }
EXPORT void cplx_from_znx32_avx2_fma(const CPLX_FROM_ZNX32_PRECOMP* precomp, void* r, const int32_t* x) { UNDEFINED(); }
EXPORT void cplx_from_tnx32_avx2_fma(const CPLX_FROM_TNX32_PRECOMP* precomp, void* r, const int32_t* x) { UNDEFINED(); }
EXPORT void cplx_to_tnx32_avx2_fma(const CPLX_TO_TNX32_PRECOMP* precomp, int32_t* x, const void* c) { UNDEFINED(); }
EXPORT void cplx_fft_avx2_fma(const CPLX_FFT_PRECOMP* tables, void* data){UNDEFINED()} EXPORT
    void cplx_ifft_avx2_fma(const CPLX_IFFT_PRECOMP* itables, void* data){UNDEFINED()} EXPORT
    void cplx_fftvec_twiddle_fma(const CPLX_FFTVEC_TWIDDLE_PRECOMP* tables, void* a, void* b, const void* om){
        UNDEFINED()} EXPORT void cplx_fftvec_twiddle_avx512(const CPLX_FFTVEC_TWIDDLE_PRECOMP* tables, void* a, void* b,
                                                            const void* om){UNDEFINED()} EXPORT
    void cplx_fftvec_bitwiddle_fma(const CPLX_FFTVEC_BITWIDDLE_PRECOMP* tables, void* a, uint64_t slice,
                                   const void* om){UNDEFINED()} EXPORT
    void cplx_fftvec_bitwiddle_avx512(const CPLX_FFTVEC_BITWIDDLE_PRECOMP* tables, void* a, uint64_t slice,
                                      const void* om){UNDEFINED()}

// DEPRECATED?
EXPORT void cplx_fftvec_add_fma(uint32_t m, void* r, const void* a, const void* b){UNDEFINED()} EXPORT
    void cplx_fftvec_sub2_to_fma(uint32_t m, void* r, const void* a, const void* b){UNDEFINED()} EXPORT
    void cplx_fftvec_copy_fma(uint32_t m, void* r, const void* a){UNDEFINED()}

// executors
//EXPORT void cplx_ifft(const CPLX_IFFT_PRECOMP* itables, void* data) {
//  itables->function(itables, data);
//}
//EXPORT void cplx_fft(const CPLX_FFT_PRECOMP* tables, void* data) { tables->function(tables, data); }
