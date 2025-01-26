#include "cplx_fft_private.h"
#include "../commons_private.h"

__always_inline void my_asserts() {
  STATIC_ASSERT(sizeof(FFT_FUNCTION)==8);
  STATIC_ASSERT(sizeof(CPLX_FFT_PRECOMP)==40);
  STATIC_ASSERT(sizeof(CPLX_IFFT_PRECOMP)==40);
}
