#include "commons_private.h"

#include <stdio.h>
#include <stdlib.h>

#include "commons.h"

EXPORT void* spqlios_error(const char* error) {
  fputs(error, stderr);
  abort();
  return nullptr;
}
EXPORT void* spqlios_keep_or_free(void* ptr, void* ptr2) {
  if (!ptr2) {
    free(ptr);
  }
  return ptr2;
}

EXPORT uint32_t log2m(uint32_t m) {
  uint32_t a = m - 1;
  if (m & a) FATAL_ERROR("m must be a power of two");
  a = (a & 0x55555555u) + ((a >> 1) & 0x55555555u);
  a = (a & 0x33333333u) + ((a >> 2) & 0x33333333u);
  a = (a & 0x0F0F0F0Fu) + ((a >> 4) & 0x0F0F0F0Fu);
  a = (a & 0x00FF00FFu) + ((a >> 8) & 0x00FF00FFu);
  return (a & 0x0000FFFFu) + ((a >> 16) & 0x0000FFFFu);
}

EXPORT uint64_t is_not_pow2_double(void* doublevalue) { return (*(uint64_t*)doublevalue) & 0x7FFFFFFFFFFFFUL; }

uint32_t revbits(uint32_t nbits, uint32_t value) {
  uint32_t res = 0;
  for (uint32_t i = 0; i < nbits; ++i) {
    res = (res << 1) + (value & 1);
    value >>= 1;
  }
  return res;
}

/**
 * @brief this computes the sequence: 0,1/2,1/4,3/4,1/8,5/8,3/8,7/8,...
 * essentially: the bits of (i+1) in lsb order on the basis (1/2^k) mod 1*/
double fracrevbits(uint32_t i) {
  if (i == 0) return 0;
  if (i == 1) return 0.5;
  if (i % 2 == 0)
    return fracrevbits(i / 2) / 2.;
  else
    return fracrevbits((i - 1) / 2) / 2. + 0.5;
}

uint64_t ceilto64b(uint64_t size) { return (size + UINT64_C(63)) & (UINT64_C(-64)); }

uint64_t ceilto32b(uint64_t size) { return (size + UINT64_C(31)) & (UINT64_C(-32)); }
