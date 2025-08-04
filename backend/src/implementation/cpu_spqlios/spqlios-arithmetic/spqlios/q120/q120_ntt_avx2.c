#include <assert.h>
#include <immintrin.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "q120_common.h"
#include "q120_ntt_private.h"

// at which level to switch from computations by level to computations by block
#define CHANGE_MODE_N 1024

__always_inline __m256i split_precompmul_si256(__m256i inp, __m256i powomega, const uint64_t h, const __m256i mask) {
  const __m256i inp_low = _mm256_and_si256(inp, mask);
  const __m256i t1 = _mm256_mul_epu32(inp_low, powomega);

  const __m256i inp_high = _mm256_srli_epi64(inp, h);
  const __m256i powomega_high = _mm256_srli_epi64(powomega, 32);
  const __m256i t2 = _mm256_mul_epu32(inp_high, powomega_high);

  return _mm256_add_epi64(t1, t2);
}

__always_inline __m256i modq_red(const __m256i x, const uint64_t h, const __m256i mask, const __m256i _2_pow_h_modq) {
  __m256i xh = _mm256_srli_epi64(x, h);
  __m256i xl = _mm256_and_si256(x, mask);
  __m256i xh_1 = _mm256_mul_epu32(xh, _2_pow_h_modq);
  return _mm256_add_epi64(xl, xh_1);
}

void print_data(const uint64_t n, const uint64_t* const data, const uint64_t q) {
  for (uint64_t i = 0; i < n; i++) {
    printf("%" PRIu64 " ", *(data + i) % q);
  }
  printf("\n");
}

double max_bit_size(const void* const begin, const void* const end) {
  double bs = 0;
  const uint64_t* data = (uint64_t*)begin;
  while (data != end) {
    double t = log2(*(data++));
    if (bs < t) {
      bs = t;
    }
  }
  return bs;
}

void ntt_iter_first(__m256i* const begin, const __m256i* const end, const q120_ntt_step_precomp* const itData,
                    const __m256i* powomega) {
  const uint64_t h = itData->half_bs;
  const __m256i vmask = _mm256_set1_epi64x(itData->mask);

  __m256i* data = begin;
  while (data < end) {
    __m256i x = _mm256_loadu_si256(data);
    __m256i po = _mm256_loadu_si256(powomega);
    __m256i r = split_precompmul_si256(x, po, h, vmask);
    _mm256_storeu_si256(data, r);

    data++;
    powomega++;
  }
}

void ntt_iter(const uint64_t nn, __m256i* const begin, const __m256i* const end,
              const q120_ntt_step_precomp* const itData, const __m256i* const powomega) {
  assert(nn % 2 == 0);
  const uint64_t halfnn = nn / 2;

  const __m256i vq2bs = _mm256_loadu_si256((__m256i*)itData->q2bs);
  const __m256i vmask = _mm256_set1_epi64x(itData->mask);

  __m256i* data = begin;
  while (data < end) {
    __m256i* ptr1 = data;
    __m256i* ptr2 = data + halfnn;

    const __m256i a = _mm256_loadu_si256(ptr1);
    const __m256i b = _mm256_loadu_si256(ptr2);

    const __m256i ap = _mm256_add_epi64(a, b);
    _mm256_storeu_si256(ptr1, ap);

    const __m256i bp = _mm256_sub_epi64(_mm256_add_epi64(a, vq2bs), b);
    _mm256_storeu_si256(ptr2, bp);

    ptr1++;
    ptr2++;

    const __m256i* po_ptr = powomega;
    for (uint64_t i = 1; i < halfnn; ++i) {
      __m256i a = _mm256_loadu_si256(ptr1);
      __m256i b = _mm256_loadu_si256(ptr2);

      __m256i ap = _mm256_add_epi64(a, b);

      _mm256_storeu_si256(ptr1, ap);

      __m256i b1 = _mm256_sub_epi64(_mm256_add_epi64(a, vq2bs), b);
      __m256i po = _mm256_loadu_si256(po_ptr);

      __m256i bp = split_precompmul_si256(b1, po, itData->half_bs, vmask);

      _mm256_storeu_si256(ptr2, bp);

      ptr1++;
      ptr2++;
      po_ptr++;
    }
    data += nn;
  }
}

void ntt_iter_red(const uint64_t nn, __m256i* const begin, const __m256i* const end,
                  const q120_ntt_step_precomp* const itData, const __m256i* const powomega,
                  const q120_ntt_reduc_step_precomp* const reduc_precomp) {
  assert(nn % 2 == 0);
  const uint64_t halfnn = nn / 2;

  const __m256i vq2bs = _mm256_loadu_si256((__m256i*)itData->q2bs);
  const __m256i vmask = _mm256_set1_epi64x(itData->mask);

  const __m256i reduc_mask = _mm256_set1_epi64x(reduc_precomp->mask);
  const __m256i reduc_cst = _mm256_loadu_si256((__m256i*)reduc_precomp->modulo_red_cst);

  __m256i* data = begin;
  while (data < end) {
    __m256i* ptr1 = data;
    __m256i* ptr2 = data + halfnn;

    __m256i a = _mm256_loadu_si256(ptr1);
    __m256i b = _mm256_loadu_si256(ptr2);

    a = modq_red(a, reduc_precomp->h, reduc_mask, reduc_cst);
    b = modq_red(b, reduc_precomp->h, reduc_mask, reduc_cst);

    const __m256i ap = _mm256_add_epi64(a, b);
    _mm256_storeu_si256(ptr1, ap);

    const __m256i bp = _mm256_sub_epi64(_mm256_add_epi64(a, vq2bs), b);
    _mm256_storeu_si256(ptr2, bp);

    ptr1++;
    ptr2++;

    const __m256i* po_ptr = powomega;
    for (uint64_t i = 1; i < halfnn; ++i) {
      __m256i a = _mm256_loadu_si256(ptr1);
      __m256i b = _mm256_loadu_si256(ptr2);

      a = modq_red(a, reduc_precomp->h, reduc_mask, reduc_cst);
      b = modq_red(b, reduc_precomp->h, reduc_mask, reduc_cst);

      __m256i ap = _mm256_add_epi64(a, b);

      _mm256_storeu_si256(ptr1, ap);

      __m256i bp = _mm256_sub_epi64(_mm256_add_epi64(a, vq2bs), b);
      __m256i po = _mm256_loadu_si256(po_ptr);
      bp = split_precompmul_si256(bp, po, itData->half_bs, vmask);

      _mm256_storeu_si256(ptr2, bp);

      ptr1++;
      ptr2++;
      po_ptr++;
    }
    data += nn;
  }
}

EXPORT void q120_ntt_bb_avx2(const q120_ntt_precomp* const precomp, q120b* const data_ptr) {
  // assert((size_t)data_ptr % 32 == 0);  // alignment check

  const uint64_t n = precomp->n;
  if (n == 1) return;

  const q120_ntt_step_precomp* itData = precomp->level_metadata;
  const __m256i* powomega = (__m256i*)precomp->powomega;

  __m256i* const begin = (__m256i*)data_ptr;
  const __m256i* const end = ((__m256i*)data_ptr) + n;

  if (CHECK_BOUNDS) {
    double bs __attribute__((unused)) = max_bit_size((void*)begin, (void*)end);
    LOG("Input %lf %" PRIu64 "\n", bs, precomp->input_bit_size);
    assert(bs <= precomp->input_bit_size);
  }

  // first iteration a_k.omega^k
  ntt_iter_first(begin, end, itData, powomega);

  if (CHECK_BOUNDS) {
    double bs __attribute__((unused)) = max_bit_size((void*)begin, (void*)end);
    LOG("Iter %3" PRIu64 " - %lf %" PRIu64 "\n", n, bs, itData->bs);
    assert(bs < itData->bs);
  }

  powomega += n;
  itData++;

  const uint64_t split_nn = (CHANGE_MODE_N > n) ? n : CHANGE_MODE_N;
  // const uint64_t split_nn = 2;

  // computations by level
  for (uint64_t nn = n; nn > split_nn; nn /= 2) {
    const uint64_t halfnn = nn / 2;

    if (itData->reduce) {
      ntt_iter_red(nn, begin, end, itData, powomega, &precomp->reduc_metadata);
    } else {
      ntt_iter(nn, begin, end, itData, powomega);
    }

    if (CHECK_BOUNDS) {
      double bs __attribute__((unused)) = max_bit_size((void*)begin, (void*)end);
      LOG("Iter %3" PRIu64 " - %lf %" PRIu64 " %c\n", nn / 2, bs, itData->bs, itData->reduce ? '*' : ' ');
      assert(bs < itData->bs);
    }

    powomega += halfnn - 1;
    itData++;
  }

  // computations by memory block
  if (split_nn >= 2) {
    const q120_ntt_step_precomp* itData1 = itData;
    const __m256i* powomega1 = powomega;
    for (__m256i* it = begin; it < end; it += split_nn) {
      __m256i* const begin1 = it;
      const __m256i* const end1 = it + split_nn;

      itData = itData1;
      powomega = powomega1;
      for (uint64_t nn = split_nn; nn >= 2; nn /= 2) {
        const uint64_t halfnn = nn / 2;

        if (itData->reduce) {
          ntt_iter_red(nn, begin1, end1, itData, powomega, &precomp->reduc_metadata);
        } else {
          ntt_iter(nn, begin1, end1, itData, powomega);
        }

        if (CHECK_BOUNDS) {
          double bs __attribute__((unused)) = max_bit_size((uint64_t*)begin1, (uint64_t*)end1);
          // LOG("Iter %3lu - %lf %lu\n", nn / 2, bs, itData->bs);
          assert(bs < itData->bs);
        }

        powomega += halfnn - 1;
        itData++;
      }
    }
  }

  if (CHECK_BOUNDS) {
    double bs __attribute__((unused)) = max_bit_size((void*)begin, (void*)end);
    LOG("Iter %3" PRIu64 " - %lf %" PRIu64 "\n", UINT64_C(1), bs, precomp->output_bit_size);
    assert(bs < precomp->output_bit_size);
  }
}

void intt_iter(const uint64_t nn, __m256i* const begin, const __m256i* const end,
               const q120_ntt_step_precomp* const itData, const __m256i* const powomega) {
  assert(nn % 2 == 0);
  const uint64_t halfnn = nn / 2;

  const __m256i vq2bs = _mm256_loadu_si256((__m256i*)itData->q2bs);
  const __m256i vmask = _mm256_set1_epi64x(itData->mask);

  __m256i* data = begin;
  while (data < end) {
    __m256i* ptr1 = data;
    __m256i* ptr2 = data + halfnn;

    const __m256i a = _mm256_loadu_si256(ptr1);
    const __m256i b = _mm256_loadu_si256(ptr2);

    const __m256i ap = _mm256_add_epi64(a, b);
    _mm256_storeu_si256(ptr1, ap);

    const __m256i bp = _mm256_sub_epi64(_mm256_add_epi64(a, vq2bs), b);
    _mm256_storeu_si256(ptr2, bp);

    ptr1++;
    ptr2++;

    const __m256i* po_ptr = powomega;
    for (uint64_t i = 1; i < halfnn; ++i) {
      __m256i a = _mm256_loadu_si256(ptr1);
      __m256i b = _mm256_loadu_si256(ptr2);

      __m256i po = _mm256_loadu_si256(po_ptr);
      __m256i bo = split_precompmul_si256(b, po, itData->half_bs, vmask);

      __m256i ap = _mm256_add_epi64(a, bo);

      _mm256_storeu_si256(ptr1, ap);

      __m256i bp = _mm256_sub_epi64(_mm256_add_epi64(a, vq2bs), bo);

      _mm256_storeu_si256(ptr2, bp);

      ptr1++;
      ptr2++;
      po_ptr++;
    }
    data += nn;
  }
}

void intt_iter_red(const uint64_t nn, __m256i* const begin, const __m256i* const end,
                   const q120_ntt_step_precomp* const itData, const __m256i* const powomega,
                   const q120_ntt_reduc_step_precomp* const reduc_precomp) {
  assert(nn % 2 == 0);
  const uint64_t halfnn = nn / 2;

  const __m256i vq2bs = _mm256_loadu_si256((__m256i*)itData->q2bs);
  const __m256i vmask = _mm256_set1_epi64x(itData->mask);

  const __m256i reduc_mask = _mm256_set1_epi64x(reduc_precomp->mask);
  const __m256i reduc_cst = _mm256_loadu_si256((__m256i*)reduc_precomp->modulo_red_cst);

  __m256i* data = begin;
  while (data < end) {
    __m256i* ptr1 = data;
    __m256i* ptr2 = data + halfnn;

    __m256i a = _mm256_loadu_si256(ptr1);
    __m256i b = _mm256_loadu_si256(ptr2);

    a = modq_red(a, reduc_precomp->h, reduc_mask, reduc_cst);
    b = modq_red(b, reduc_precomp->h, reduc_mask, reduc_cst);

    const __m256i ap = _mm256_add_epi64(a, b);
    _mm256_storeu_si256(ptr1, ap);

    const __m256i bp = _mm256_sub_epi64(_mm256_add_epi64(a, vq2bs), b);
    _mm256_storeu_si256(ptr2, bp);

    ptr1++;
    ptr2++;

    const __m256i* po_ptr = powomega;
    for (uint64_t i = 1; i < halfnn; ++i) {
      __m256i a = _mm256_loadu_si256(ptr1);
      __m256i b = _mm256_loadu_si256(ptr2);

      a = modq_red(a, reduc_precomp->h, reduc_mask, reduc_cst);
      b = modq_red(b, reduc_precomp->h, reduc_mask, reduc_cst);

      __m256i po = _mm256_loadu_si256(po_ptr);
      __m256i bo = split_precompmul_si256(b, po, itData->half_bs, vmask);

      __m256i ap = _mm256_add_epi64(a, bo);

      _mm256_storeu_si256(ptr1, ap);

      __m256i bp = _mm256_sub_epi64(_mm256_add_epi64(a, vq2bs), bo);

      _mm256_storeu_si256(ptr2, bp);

      ptr1++;
      ptr2++;
      po_ptr++;
    }
    data += nn;
  }
}

void ntt_iter_first_red(__m256i* const begin, const __m256i* const end, const q120_ntt_step_precomp* const itData,
                        const __m256i* powomega, const q120_ntt_reduc_step_precomp* const reduc_precomp) {
  const uint64_t h = itData->half_bs;
  const __m256i vmask = _mm256_set1_epi64x(itData->mask);

  const __m256i reduc_mask = _mm256_set1_epi64x(reduc_precomp->mask);
  const __m256i reduc_cst = _mm256_loadu_si256((__m256i*)reduc_precomp->modulo_red_cst);

  __m256i* data = begin;
  while (data < end) {
    __m256i x = _mm256_loadu_si256(data);
    x = modq_red(x, reduc_precomp->h, reduc_mask, reduc_cst);
    __m256i po = _mm256_loadu_si256(powomega);
    __m256i r = split_precompmul_si256(x, po, h, vmask);
    _mm256_storeu_si256(data, r);

    data++;
    powomega++;
  }
}

EXPORT void q120_intt_bb_avx2(const q120_ntt_precomp* const precomp, q120b* const data_ptr) {
  // assert((size_t)data_ptr % 32 == 0);  // alignment check

  const uint64_t n = precomp->n;
  if (n == 1) return;

  const q120_ntt_step_precomp* itData = precomp->level_metadata;
  const __m256i* powomega = (__m256i*)precomp->powomega;

  __m256i* const begin = (__m256i*)data_ptr;
  const __m256i* const end = ((__m256i*)data_ptr) + n;

  if (CHECK_BOUNDS) {
    double bs __attribute__((unused)) = max_bit_size((void*)begin, (void*)end);
    LOG("Input %lf %" PRIu64 "\n", bs, precomp->input_bit_size);
    assert(bs <= precomp->input_bit_size);
  }

  const uint64_t split_nn = (CHANGE_MODE_N > n) ? n : CHANGE_MODE_N;

  // computations by memory block
  if (split_nn >= 2) {
    const q120_ntt_step_precomp* itData1 = itData;
    const __m256i* powomega1 = powomega;
    for (__m256i* it = begin; it < end; it += split_nn) {
      __m256i* const begin1 = it;
      const __m256i* const end1 = it + split_nn;

      itData = itData1;
      powomega = powomega1;
      for (uint64_t nn = 2; nn <= split_nn; nn *= 2) {
        const uint64_t halfnn = nn / 2;

        if (itData->reduce) {
          intt_iter_red(nn, begin1, end1, itData, powomega, &precomp->reduc_metadata);
        } else {
          intt_iter(nn, begin1, end1, itData, powomega);
        }

        if (CHECK_BOUNDS) {
          double bs __attribute__((unused)) = max_bit_size((uint64_t*)begin1, (uint64_t*)end1);
          // LOG("Iter %3lu - %lf %lu\n", nn / 2, bs, itData->bs);
          assert(bs < itData->bs);
        }

        powomega += halfnn - 1;
        itData++;
      }
    }
  }

  // computations by level
  // for (uint64_t nn = 2; nn <= n; nn *= 2) {
  for (uint64_t nn = 2 * split_nn; nn <= n; nn *= 2) {
    const uint64_t halfnn = nn / 2;

    if (itData->reduce) {
      intt_iter_red(nn, begin, end, itData, powomega, &precomp->reduc_metadata);
    } else {
      intt_iter(nn, begin, end, itData, powomega);
    }

    if (CHECK_BOUNDS) {
      double bs __attribute__((unused)) = max_bit_size((void*)begin, (void*)end);
      LOG("Iter %3" PRIu64 " - %lf %" PRIu64 " %c\n", nn / 2, bs, itData->bs, itData->reduce ? '*' : ' ');
      assert(bs < itData->bs);
    }

    powomega += halfnn - 1;
    itData++;
  }

  // last iteration a_k . omega^k . n^-1
  if (itData->reduce) {
    ntt_iter_first_red(begin, end, itData, powomega, &precomp->reduc_metadata);
  } else {
    ntt_iter_first(begin, end, itData, powomega);
  }

  if (CHECK_BOUNDS) {
    double bs __attribute__((unused)) = max_bit_size((void*)begin, (void*)end);
    LOG("Iter %3" PRIu64 " - %lf %" PRIu64 "\n", n, bs, itData->bs);
    assert(bs < itData->bs);
  }
}
