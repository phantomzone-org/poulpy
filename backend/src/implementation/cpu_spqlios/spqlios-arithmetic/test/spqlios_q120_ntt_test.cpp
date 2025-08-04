#include <gtest/gtest.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "spqlios/q120/q120_common.h"
#include "spqlios/q120/q120_ntt.h"
#include "testlib/mod_q120.h"

std::vector<mod_q120> q120_ntt(const std::vector<mod_q120>& x) {
  const uint64_t n = x.size();

  mod_q120 omega_2pow17{OMEGA1, OMEGA2, OMEGA3, OMEGA4};
  mod_q120 omega = pow(omega_2pow17, (1 << 16) / n);

  std::vector<mod_q120> res(n);
  for (uint64_t i = 0; i < n; ++i) {
    res[i] = x[i];
  }

  for (uint64_t i = 0; i < n; ++i) {
    res[i] = res[i] * pow(omega, i);
  }

  for (uint64_t nn = n; nn > 1; nn /= 2) {
    const uint64_t halfnn = nn / 2;
    const uint64_t m = n / halfnn;

    for (uint64_t j = 0; j < n; j += nn) {
      for (uint64_t k = 0; k < halfnn; ++k) {
        mod_q120 a = res[j + k];
        mod_q120 b = res[j + halfnn + k];

        res[j + k] = a + b;
        res[j + halfnn + k] = (a - b) * pow(omega, k * m);
      }
    }
  }

  return res;
}

std::vector<mod_q120> q120_intt(const std::vector<mod_q120>& x) {
  const uint64_t n = x.size();

  mod_q120 omega_2pow17{OMEGA1, OMEGA2, OMEGA3, OMEGA4};
  mod_q120 omega = pow(omega_2pow17, (1 << 16) / n);

  std::vector<mod_q120> res(n);
  for (uint64_t i = 0; i < n; ++i) {
    res[i] = x[i];
  }

  for (uint64_t nn = 2; nn <= n; nn *= 2) {
    const uint64_t halfnn = nn / 2;
    const uint64_t m = n / halfnn;

    for (uint64_t j = 0; j < n; j += nn) {
      for (uint64_t k = 0; k < halfnn; ++k) {
        mod_q120 a = res[j + k];
        mod_q120 b = res[j + halfnn + k];

        mod_q120 bo = b * pow(omega, -k * m);
        res[j + k] = a + bo;
        res[j + halfnn + k] = a - bo;
      }
    }
  }

  mod_q120 n_q120{(int64_t)n, (int64_t)n, (int64_t)n, (int64_t)n};
  mod_q120 n_inv_q120 = pow(n_q120, -1);

  for (uint64_t i = 0; i < n; ++i) {
    mod_q120 po = pow(omega, -i) * n_inv_q120;
    res[i] = res[i] * po;
  }

  return res;
}

class ntt : public testing::TestWithParam<uint64_t> {};

#ifdef __x86_64__

TEST_P(ntt, q120_ntt_bb_avx2) {
  const uint64_t n = GetParam();
  q120_ntt_precomp* precomp = q120_new_ntt_bb_precomp(n);

  std::vector<uint64_t> x(n * 4);
  uint64_t* px = x.data();
  for (uint64_t i = 0; i < 4 * n; i += 4) {
    uniform_q120b(px + i);
  }

  std::vector<mod_q120> x_modq(n);
  for (uint64_t i = 0; i < n; ++i) {
    x_modq[i] = mod_q120::from_q120b(px + 4 * i);
  }

  std::vector<mod_q120> y_exp = q120_ntt(x_modq);

  q120_ntt_bb_avx2(precomp, (q120b*)px);

  for (uint64_t i = 0; i < n; ++i) {
    mod_q120 comp_r = mod_q120::from_q120b(px + 4 * i);
    ASSERT_EQ(comp_r, y_exp[i]) << i;
  }

  q120_del_ntt_bb_precomp(precomp);
}

TEST_P(ntt, q120_intt_bb_avx2) {
  const uint64_t n = GetParam();
  q120_ntt_precomp* precomp = q120_new_intt_bb_precomp(n);

  std::vector<uint64_t> x(n * 4);
  uint64_t* px = x.data();
  for (uint64_t i = 0; i < 4 * n; i += 4) {
    uniform_q120b(px + i);
  }

  std::vector<mod_q120> x_modq(n);
  for (uint64_t i = 0; i < n; ++i) {
    x_modq[i] = mod_q120::from_q120b(px + 4 * i);
  }

  q120_intt_bb_avx2(precomp, (q120b*)px);

  std::vector<mod_q120> y_exp = q120_intt(x_modq);
  for (uint64_t i = 0; i < n; ++i) {
    mod_q120 comp_r = mod_q120::from_q120b(px + 4 * i);
    ASSERT_EQ(comp_r, y_exp[i]) << i;
  }

  q120_del_intt_bb_precomp(precomp);
}

TEST_P(ntt, q120_ntt_intt_bb_avx2) {
  const uint64_t n = GetParam();
  q120_ntt_precomp* precomp_ntt = q120_new_ntt_bb_precomp(n);
  q120_ntt_precomp* precomp_intt = q120_new_intt_bb_precomp(n);

  std::vector<uint64_t> x(n * 4);
  uint64_t* px = x.data();
  for (uint64_t i = 0; i < 4 * n; i += 4) {
    uniform_q120b(px + i);
  }

  std::vector<mod_q120> x_modq(n);
  for (uint64_t i = 0; i < n; ++i) {
    x_modq[i] = mod_q120::from_q120b(px + 4 * i);
  }

  q120_ntt_bb_avx2(precomp_ntt, (q120b*)px);
  q120_intt_bb_avx2(precomp_intt, (q120b*)px);

  for (uint64_t i = 0; i < n; ++i) {
    mod_q120 comp_r = mod_q120::from_q120b(px + 4 * i);
    ASSERT_EQ(comp_r, x_modq[i]) << i;
  }

  q120_del_intt_bb_precomp(precomp_intt);
  q120_del_ntt_bb_precomp(precomp_ntt);
}

INSTANTIATE_TEST_SUITE_P(q120, ntt,
                         testing::Values(1, 2, 4, 16, 256, UINT64_C(1) << 10, UINT64_C(1) << 11, UINT64_C(1) << 12,
                                         UINT64_C(1) << 13, UINT64_C(1) << 14, UINT64_C(1) << 15, UINT64_C(1) << 16),
                         testing::PrintToStringParamName());

#endif
