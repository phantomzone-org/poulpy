#include "mod_q120.h"

#include <cstdint>
#include <random>

int64_t centermod(int64_t v, int64_t q) {
  int64_t t = v % q;
  if (t >= (q + 1) / 2) return t - q;
  if (t < -q / 2) return t + q;
  return t;
}

int64_t centermod(uint64_t v, int64_t q) {
  int64_t t = int64_t(v % uint64_t(q));
  if (t >= q / 2) return t - q;
  return t;
}

mod_q120::mod_q120() {
  for (uint64_t i = 0; i < 4; ++i) {
    a[i] = 0;
  }
}

mod_q120::mod_q120(int64_t a0, int64_t a1, int64_t a2, int64_t a3) {
  a[0] = centermod(a0, Qi[0]);
  a[1] = centermod(a1, Qi[1]);
  a[2] = centermod(a2, Qi[2]);
  a[3] = centermod(a3, Qi[3]);
}

mod_q120 operator+(const mod_q120& x, const mod_q120& y) {
  mod_q120 r;
  for (uint64_t i = 0; i < 4; ++i) {
    r.a[i] = centermod(x.a[i] + y.a[i], mod_q120::Qi[i]);
  }
  return r;
}

mod_q120 operator-(const mod_q120& x, const mod_q120& y) {
  mod_q120 r;
  for (uint64_t i = 0; i < 4; ++i) {
    r.a[i] = centermod(x.a[i] - y.a[i], mod_q120::Qi[i]);
  }
  return r;
}

mod_q120 operator*(const mod_q120& x, const mod_q120& y) {
  mod_q120 r;
  for (uint64_t i = 0; i < 4; ++i) {
    r.a[i] = centermod(x.a[i] * y.a[i], mod_q120::Qi[i]);
  }
  return r;
}

mod_q120& operator+=(mod_q120& x, const mod_q120& y) {
  for (uint64_t i = 0; i < 4; ++i) {
    x.a[i] = centermod(x.a[i] + y.a[i], mod_q120::Qi[i]);
  }
  return x;
}

mod_q120& operator-=(mod_q120& x, const mod_q120& y) {
  for (uint64_t i = 0; i < 4; ++i) {
    x.a[i] = centermod(x.a[i] - y.a[i], mod_q120::Qi[i]);
  }
  return x;
}

mod_q120& operator*=(mod_q120& x, const mod_q120& y) {
  for (uint64_t i = 0; i < 4; ++i) {
    x.a[i] = centermod(x.a[i] * y.a[i], mod_q120::Qi[i]);
  }
  return x;
}

int64_t modq_pow(int64_t x, int32_t k, int64_t q) {
  k = (k % (q - 1) + q - 1) % (q - 1);

  int64_t res = 1;
  int64_t x_pow = centermod(x, q);
  while (k != 0) {
    if (k & 1) res = centermod(res * x_pow, q);
    x_pow = centermod(x_pow * x_pow, q);
    k >>= 1;
  }
  return res;
}

mod_q120 pow(const mod_q120& x, int32_t k) {
  const int64_t r0 = modq_pow(x.a[0], k, x.Qi[0]);
  const int64_t r1 = modq_pow(x.a[1], k, x.Qi[1]);
  const int64_t r2 = modq_pow(x.a[2], k, x.Qi[2]);
  const int64_t r3 = modq_pow(x.a[3], k, x.Qi[3]);
  return mod_q120{r0, r1, r2, r3};
}

static int64_t half_modq(int64_t x, int64_t q) {
  // q must be odd in this function
  if (x % 2 == 0) return x / 2;
  return centermod((x + q) / 2, q);
}

mod_q120 half(const mod_q120& x) {
  const int64_t r0 = half_modq(x.a[0], x.Qi[0]);
  const int64_t r1 = half_modq(x.a[1], x.Qi[1]);
  const int64_t r2 = half_modq(x.a[2], x.Qi[2]);
  const int64_t r3 = half_modq(x.a[3], x.Qi[3]);
  return mod_q120{r0, r1, r2, r3};
}

bool operator==(const mod_q120& x, const mod_q120& y) {
  for (uint64_t i = 0; i < 4; ++i) {
    if (x.a[i] != y.a[i]) return false;
  }
  return true;
}

std::ostream& operator<<(std::ostream& out, const mod_q120& x) {
  return out << "q120{" << x.a[0] << "," << x.a[1] << "," << x.a[2] << "," << x.a[3] << "}";
}

mod_q120 mod_q120::from_q120a(const void* addr) {
  static const uint64_t _2p32 = UINT64_C(1) << 32;
  const uint64_t* in = (const uint64_t*)addr;
  mod_q120 r;
  for (uint64_t i = 0; i < 4; ++i) {
    REQUIRE_DRAMATICALLY(in[i] < _2p32, "invalid layout a q120");
    r.a[i] = centermod(in[i], mod_q120::Qi[i]);
  }
  return r;
}

mod_q120 mod_q120::from_q120b(const void* addr) {
  const uint64_t* in = (const uint64_t*)addr;
  mod_q120 r;
  for (uint64_t i = 0; i < 4; ++i) {
    r.a[i] = centermod(in[i], mod_q120::Qi[i]);
  }
  return r;
}

mod_q120 mod_q120::from_q120c(const void* addr) {
  // static const uint64_t _mask_2p32 = (uint64_t(1) << 32) - 1;
  const uint32_t* in = (const uint32_t*)addr;
  mod_q120 r;
  for (uint64_t i = 0, k = 0; i < 8; i += 2, ++k) {
    const uint64_t q = mod_q120::Qi[k];
    uint64_t u = in[i];
    uint64_t w = in[i + 1];
    REQUIRE_DRAMATICALLY(((u << 32) % q) == (w % q),
                         "invalid layout q120c: " << u << ".2^32 != " << (w >> 32) << " mod " << q);
    r.a[k] = centermod(u, q);
  }
  return r;
}
__int128_t mod_q120::to_int128() const {
  static const __int128_t qm[] = {(__int128_t(Qi[1]) * Qi[2]) * Qi[3], (__int128_t(Qi[0]) * Qi[2]) * Qi[3],
                                  (__int128_t(Qi[0]) * Qi[1]) * Qi[3], (__int128_t(Qi[0]) * Qi[1]) * Qi[2]};
  static const int64_t CRTi[] = {Q1_CRT_CST, Q2_CRT_CST, Q3_CRT_CST, Q4_CRT_CST};
  static const __int128_t q = qm[0] * Qi[0];
  static const __int128_t qs2 = q / 2;
  __int128_t res = 0;
  for (uint64_t i = 0; i < 4; ++i) {
    res += (a[i] * CRTi[i] % Qi[i]) * qm[i];
  }
  res = (((res % q) + q + qs2) % q) - qs2;  // centermod
  return res;
}
void mod_q120::save_as_q120a(void* dest) const {
  int64_t* d = (int64_t*)dest;
  for (uint64_t i = 0; i < 4; ++i) {
    d[i] = a[i] + Qi[i];
  }
}
void mod_q120::save_as_q120b(void* dest) const {
  int64_t* d = (int64_t*)dest;
  for (uint64_t i = 0; i < 4; ++i) {
    d[i] = a[i] + (Qi[i] * (1 + uniform_u64_bits(32)));
  }
}
void mod_q120::save_as_q120c(void* dest) const {
  int32_t* d = (int32_t*)dest;
  for (uint64_t i = 0; i < 4; ++i) {
    d[2 * i] = a[i] + 3 * Qi[i];
    d[2 * i + 1] = (uint64_t(d[2 * i]) << 32) % uint64_t(Qi[i]);
  }
}

mod_q120 uniform_q120() {
  test_rng& gen = randgen();
  std::uniform_int_distribution<uint64_t> dista(0, mod_q120::Qi[0]);
  std::uniform_int_distribution<uint64_t> distb(0, mod_q120::Qi[1]);
  std::uniform_int_distribution<uint64_t> distc(0, mod_q120::Qi[2]);
  std::uniform_int_distribution<uint64_t> distd(0, mod_q120::Qi[3]);
  return mod_q120(dista(gen), distb(gen), distc(gen), distd(gen));
}

void uniform_q120a(void* dest) {
  uint64_t* res = (uint64_t*)dest;
  for (uint64_t i = 0; i < 4; ++i) {
    res[i] = uniform_u64_bits(32);
  }
}

void uniform_q120b(void* dest) {
  uint64_t* res = (uint64_t*)dest;
  for (uint64_t i = 0; i < 4; ++i) {
    res[i] = uniform_u64();
  }
}

void uniform_q120c(void* dest) {
  uint32_t* res = (uint32_t*)dest;
  static const uint64_t _2p32 = uint64_t(1) << 32;
  for (uint64_t i = 0, k = 0; i < 8; i += 2, ++k) {
    const uint64_t q = mod_q120::Qi[k];
    const uint64_t z = uniform_u64_bits(32);
    const uint64_t z_pow_red = (z << 32) % q;
    const uint64_t room = (_2p32 - z_pow_red) / q;
    const uint64_t z_pow = z_pow_red + (uniform_u64() % room) * q;
    REQUIRE_DRAMATICALLY(z < _2p32, "bug!");
    REQUIRE_DRAMATICALLY(z_pow < _2p32, "bug!");
    REQUIRE_DRAMATICALLY(z_pow % q == (z << 32) % q, "bug!");

    res[i] = (uint32_t)z;
    res[i + 1] = (uint32_t)z_pow;
  }
}
