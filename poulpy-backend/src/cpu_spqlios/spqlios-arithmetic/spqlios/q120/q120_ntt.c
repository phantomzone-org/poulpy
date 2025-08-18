#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>

#include "q120_ntt_private.h"

q120_ntt_precomp* new_precomp(const uint64_t n) {
  q120_ntt_precomp* precomp = malloc(sizeof(*precomp));
  precomp->n = n;

  assert(n && !(n & (n - 1)) && n <= (1 << 16));  // n is a power of 2 smaller than 2^16
  const uint64_t logN = ceil(log2(n));
  precomp->level_metadata = malloc((logN + 2) * sizeof(*precomp->level_metadata));

  precomp->powomega = spqlios_alloc_custom_align(32, 4 * 2 * n * sizeof(*(precomp->powomega)));

  return precomp;
}

uint32_t modq_pow(const uint32_t x, const int64_t n, const uint32_t q) {
  uint64_t np = (n % (q - 1) + q - 1) % (q - 1);

  uint64_t val_pow = x;
  uint64_t res = 1;
  while (np != 0) {
    if (np & 1) res = (res * val_pow) % q;
    val_pow = (val_pow * val_pow) % q;
    np >>= 1;
  }
  return res;
}

void fill_omegas(const uint64_t n, uint32_t omegas[4]) {
  for (uint64_t k = 0; k < 4; ++k) {
    omegas[k] = modq_pow(OMEGAS_VEC[k], (1 << 16) / n, PRIMES_VEC[k]);
  }

#ifndef NDEBUG

  const uint64_t logQ = ceil(log2(PRIMES_VEC[0]));
  for (int k = 1; k < 4; ++k) {
    if (logQ != ceil(log2(PRIMES_VEC[k]))) {
      fprintf(stderr, "The 4 primes must have the same bit-size\n");
      exit(-1);
    }
  }

  // check if each omega is a 2.n primitive root of unity
  for (uint64_t k = 0; k < 4; ++k) {
    assert(modq_pow(omegas[k], 2 * n, PRIMES_VEC[k]) == 1);
    for (uint64_t i = 1; i < 2 * n; ++i) {
      assert(modq_pow(omegas[k], i, PRIMES_VEC[k]) != 1);
    }
  }

  if (logQ > 31) {
    fprintf(stderr, "Modulus q bit-size is larger than 30 bit\n");
    exit(-1);
  }
#endif
}

uint64_t fill_reduction_meta(const uint64_t bs_start, q120_ntt_reduc_step_precomp* reduc_metadata) {
  // fill reduction metadata
  uint64_t bs_after_reduc = -1;
  {
    uint64_t min_h = -1;

    for (uint64_t h = bs_start / 2; h < bs_start; ++h) {
      uint64_t t = 0;
      for (uint64_t k = 0; k < 4; ++k) {
        const uint64_t t1 = bs_start - h + (uint64_t)ceil(log2((UINT64_C(1) << h) % PRIMES_VEC[k]));
        const uint64_t t2 = UINT64_C(1) + ((t1 > h) ? t1 : h);
        if (t < t2) t = t2;
      }
      if (t < bs_after_reduc) {
        min_h = h;
        bs_after_reduc = t;
      }
    }

    reduc_metadata->h = min_h;
    reduc_metadata->mask = (UINT64_C(1) << min_h) - 1;
    for (uint64_t k = 0; k < 4; ++k) {
      reduc_metadata->modulo_red_cst[k] = (UINT64_C(1) << min_h) % PRIMES_VEC[k];
    }

    assert(bs_after_reduc < 64);
  }

  return bs_after_reduc;
}

uint64_t round_up_half_n(const uint64_t n) { return (n + 1) / 2; }

EXPORT q120_ntt_precomp* q120_new_ntt_bb_precomp(const uint64_t n) {
  uint32_t omega_vec[4];
  fill_omegas(n, omega_vec);

  const uint64_t logQ = ceil(log2(PRIMES_VEC[0]));

  q120_ntt_precomp* precomp = new_precomp(n);

  uint64_t bs = precomp->input_bit_size = 64;

  LOG("NTT parameters:\n");
  LOG("\tsize = %" PRIu64 "\n", n)
  LOG("\tlogQ = %" PRIu64 "\n", logQ);
  LOG("\tinput bit-size = %" PRIu64 "\n", bs);

  if (n == 1) return precomp;

  // fill reduction metadata
  uint64_t bs_after_reduc = fill_reduction_meta(bs, &(precomp->reduc_metadata));

  // forward metadata
  q120_ntt_step_precomp* level_metadata_ptr = precomp->level_metadata;

  // first level a_k.omega^k
  {
    const uint64_t half_bs = (bs + 1) / 2;
    level_metadata_ptr->half_bs = half_bs;
    level_metadata_ptr->mask = (UINT64_C(1) << half_bs) - UINT64_C(1);
    level_metadata_ptr->bs = bs = half_bs + logQ + 1;
    LOG("\tlevel %6" PRIu64 " output bit-size = %" PRIu64 " (a_k.omega^k) \n", n, bs);
    level_metadata_ptr++;
  }

  for (uint64_t nn = n; nn >= 4; nn /= 2) {
    level_metadata_ptr->reduce = (bs == 64);
    if (level_metadata_ptr->reduce) {
      bs = bs_after_reduc;
      LOG("\treduce       output bit-size = %" PRIu64 "\n", bs);
    }

    for (int k = 0; k < 4; ++k) level_metadata_ptr->q2bs[k] = (uint64_t)PRIMES_VEC[k] << (bs - logQ);

    double bs_1 = bs + 1;  // bit-size of term a+b or a-b

    const uint64_t half_bs = round_up_half_n(bs_1);
    uint64_t bs_2 = half_bs + logQ + 1;  // bit-size of term (a-b).omega^k
    bs = (bs_1 > bs_2) ? bs_1 : bs_2;
    assert(bs <= 64);

    level_metadata_ptr->bs = bs;
    level_metadata_ptr->half_bs = half_bs;
    level_metadata_ptr->mask = (UINT64_C(1) << half_bs) - UINT64_C(1);
    level_metadata_ptr++;

    LOG("\tlevel %6" PRIu64 " output bit-size = %" PRIu64 "\n", nn / 2, bs);
  }

  // last level (a-b, a+b)
  {
    level_metadata_ptr->reduce = (bs == 64);
    if (level_metadata_ptr->reduce) {
      bs = bs_after_reduc;
      LOG("\treduce       output bit-size = %" PRIu64 "\n", bs);
    }
    for (int k = 0; k < 4; ++k) level_metadata_ptr->q2bs[k] = ((uint64_t)PRIMES_VEC[k] << (bs - logQ));
    level_metadata_ptr->bs = ++bs;
    level_metadata_ptr->half_bs = level_metadata_ptr->mask = UINT64_C(0);  // not used

    LOG("\tlevel %6" PRIu64 " output bit-size = %" PRIu64 "\n", UINT64_C(1), bs);
  }
  precomp->output_bit_size = bs;

  // omega powers
  uint64_t* powomega = malloc(sizeof(*powomega) * 2 * n);
  for (uint64_t k = 0; k < 4; ++k) {
    const uint64_t q = PRIMES_VEC[k];

    for (uint64_t i = 0; i < 2 * n; ++i) {
      powomega[i] = modq_pow(omega_vec[k], i, q);
    }

    uint64_t* powomega_ptr = precomp->powomega + k;
    level_metadata_ptr = precomp->level_metadata;

    {
      // const uint64_t hpow = UINT64_C(1) << level_metadata_ptr->half_bs;
      for (uint64_t i = 0; i < n; ++i) {
        uint64_t t = powomega[i];
        uint64_t t1 = (t << level_metadata_ptr->half_bs) % q;
        powomega_ptr[4 * i] = (t1 << 32) + t;
      }
      powomega_ptr += 4 * n;
      level_metadata_ptr++;
    }

    for (uint64_t nn = n; nn >= 4; nn /= 2) {
      const uint64_t halfnn = nn / 2;
      const uint64_t m = n / halfnn;

      // const uint64_t hpow = UINT64_C(1) << level_metadata_ptr->half_bs;
      for (uint64_t i = 1; i < halfnn; ++i) {
        uint64_t t = powomega[i * m];
        uint64_t t1 = (t << level_metadata_ptr->half_bs) % q;
        powomega_ptr[4 * (i - 1)] = (t1 << 32) + t;
      }
      powomega_ptr += 4 * (halfnn - 1);
      level_metadata_ptr++;
    }
  }
  free(powomega);

  return precomp;
}

EXPORT q120_ntt_precomp* q120_new_intt_bb_precomp(const uint64_t n) {
  uint32_t omega_vec[4];
  fill_omegas(n, omega_vec);

  const uint64_t logQ = ceil(log2(PRIMES_VEC[0]));

  q120_ntt_precomp* precomp = new_precomp(n);

  uint64_t bs = precomp->input_bit_size = 64;

  LOG("iNTT parameters:\n");
  LOG("\tsize = %" PRIu64 "\n", n)
  LOG("\tlogQ = %" PRIu64 "\n", logQ);
  LOG("\tinput bit-size = %" PRIu64 "\n", bs);

  if (n == 1) return precomp;

  // fill reduction metadata
  uint64_t bs_after_reduc = fill_reduction_meta(bs, &(precomp->reduc_metadata));

  // backward metadata
  q120_ntt_step_precomp* level_metadata_ptr = precomp->level_metadata;

  // first level (a+b, a-b) adds 1-bit
  {
    level_metadata_ptr->reduce = (bs == 64);
    if (level_metadata_ptr->reduce) {
      bs = bs_after_reduc;
      LOG("\treduce       output bit-size = %" PRIu64 "\n", bs);
    }

    for (int k = 0; k < 4; ++k) level_metadata_ptr->q2bs[k] = (uint64_t)PRIMES_VEC[k] << (bs - logQ);

    level_metadata_ptr->bs = ++bs;
    level_metadata_ptr->half_bs = level_metadata_ptr->mask = UINT64_C(0);  // not used
    LOG("\tlevel %6" PRIu64 " output bit-size = %" PRIu64 "\n", UINT64_C(1), bs);
    level_metadata_ptr++;
  }

  for (uint64_t nn = 4; nn <= n; nn *= 2) {
    level_metadata_ptr->reduce = (bs == 64);
    if (level_metadata_ptr->reduce) {
      bs = bs_after_reduc;
      LOG("\treduce       output bit-size = %" PRIu64 "\n", bs);
    }

    const uint64_t half_bs = round_up_half_n(bs);
    const uint64_t bs_mult = half_bs + logQ + 1;  // bit-size of term b.omega^k
    bs = 1 + ((bs > bs_mult) ? bs : bs_mult);     // bit-size of a+b.omega^k or a-b.omega^k
    assert(bs <= 64);

    level_metadata_ptr->bs = bs;
    level_metadata_ptr->half_bs = half_bs;
    level_metadata_ptr->mask = (UINT64_C(1) << half_bs) - UINT64_C(1);
    for (int k = 0; k < 4; ++k) level_metadata_ptr->q2bs[k] = (uint64_t)PRIMES_VEC[k] << (bs_mult - logQ);
    level_metadata_ptr++;

    LOG("\tlevel %6" PRIu64 " output bit-size = %" PRIu64 "\n", nn / 2, bs);
  }

  // last level a_k.omega^k
  {
    level_metadata_ptr->reduce = (bs == 64);
    if (level_metadata_ptr->reduce) {
      bs = bs_after_reduc;
      LOG("\treduce       output bit-size = %" PRIu64 "\n", bs);
    }

    const uint64_t half_bs = round_up_half_n(bs);

    bs = half_bs + logQ + 1;  // bit-size of term a.omega^k
    assert(bs <= 64);

    level_metadata_ptr->bs = bs;
    level_metadata_ptr->half_bs = half_bs;
    level_metadata_ptr->mask = (UINT64_C(1) << half_bs) - UINT64_C(1);
    for (int k = 0; k < 4; ++k) level_metadata_ptr->q2bs[k] = (uint64_t)PRIMES_VEC[k] << (bs - logQ);

    LOG("\tlevel %6" PRIu64 " output bit-size = %" PRIu64 "\n", n, bs);
  }

  // omega powers
  uint32_t* powomegabar = malloc(sizeof(*powomegabar) * 2 * n);
  for (int k = 0; k < 4; ++k) {
    const uint64_t q = PRIMES_VEC[k];

    for (uint64_t i = 0; i < 2 * n; ++i) {
      powomegabar[i] = modq_pow(omega_vec[k], -i, q);
    }

    uint64_t* powomega_ptr = precomp->powomega + k;
    level_metadata_ptr = precomp->level_metadata + 1;

    for (uint64_t nn = 4; nn <= n; nn *= 2) {
      const uint64_t halfnn = nn / 2;
      const uint64_t m = n / halfnn;

      for (uint64_t i = 1; i < halfnn; ++i) {
        uint64_t t = powomegabar[i * m];
        uint64_t t1 = (t << level_metadata_ptr->half_bs) % q;
        powomega_ptr[4 * (i - 1)] = (t1 << 32) + t;
      }
      powomega_ptr += 4 * (halfnn - 1);
      level_metadata_ptr++;
    }

    {
      const uint64_t invNmod = modq_pow(n, -1, q);
      for (uint64_t i = 0; i < n; ++i) {
        uint64_t t = (powomegabar[i] * invNmod) % q;
        uint64_t t1 = (t << level_metadata_ptr->half_bs) % q;
        powomega_ptr[4 * i] = (t1 << 32) + t;
      }
    }
  }

  free(powomegabar);

  return precomp;
}

void del_precomp(q120_ntt_precomp* precomp) {
  spqlios_free(precomp->powomega);
  free(precomp->level_metadata);
  free(precomp);
}

EXPORT void q120_del_ntt_bb_precomp(q120_ntt_precomp* precomp) { del_precomp(precomp); }

EXPORT void q120_del_intt_bb_precomp(q120_ntt_precomp* precomp) { del_precomp(precomp); }
