#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../commons_private.h"
#include "reim_fft_internal.h"
#include "reim_fft_private.h"

double accurate_cos(int32_t i, int32_t n) {  // cos(2pi*i/n)
  i = ((i % n) + n) % n;
  if (i >= 3 * n / 4) return cos(2. * M_PI * (n - i) / (double)(n));
  if (i >= 2 * n / 4) return -cos(2. * M_PI * (i - n / 2) / (double)(n));
  if (i >= 1 * n / 4) return -cos(2. * M_PI * (n / 2 - i) / (double)(n));
  return cos(2. * M_PI * (i) / (double)(n));
}

double accurate_sin(int32_t i, int32_t n) {  // sin(2pi*i/n)
  i = ((i % n) + n) % n;
  if (i >= 3 * n / 4) return -sin(2. * M_PI * (n - i) / (double)(n));
  if (i >= 2 * n / 4) return -sin(2. * M_PI * (i - n / 2) / (double)(n));
  if (i >= 1 * n / 4) return sin(2. * M_PI * (n / 2 - i) / (double)(n));
  return sin(2. * M_PI * (i) / (double)(n));
}

EXPORT double* reim_ifft_precomp_get_buffer(const REIM_IFFT_PRECOMP* tables, uint32_t buffer_index) {
  return (double*)((uint8_t*)tables->aligned_buffers + buffer_index * tables->buf_size);
}

EXPORT double* reim_fft_precomp_get_buffer(const REIM_FFT_PRECOMP* tables, uint32_t buffer_index) {
  return (double*)((uint8_t*)tables->aligned_buffers + buffer_index * tables->buf_size);
}

EXPORT void reim_fft(const REIM_FFT_PRECOMP* tables, double* data) { tables->function(tables, data); }
EXPORT void reim_ifft(const REIM_IFFT_PRECOMP* tables, double* data) { tables->function(tables, data); }
