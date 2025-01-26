#include <cstdint>
#include <random>

#include "test_commons.h"

bool is_pow2(uint64_t n) { return !(n & (n - 1)); }

test_rng& randgen() {
  static test_rng gen;
  return gen;
}
uint64_t uniform_u64() {
  static std::uniform_int_distribution<uint64_t> dist64(0, UINT64_MAX);
  return dist64(randgen());
}

uint64_t uniform_u64_bits(uint64_t nbits) {
  if (nbits >= 64) return uniform_u64();
  return uniform_u64() >> (64 - nbits);
}

int64_t uniform_i64() {
  std::uniform_int_distribution<int64_t> dist;
  return dist(randgen());
}

int64_t uniform_i64_bits(uint64_t nbits) {
  int64_t bound = int64_t(1) << nbits;
  std::uniform_int_distribution<int64_t> dist(-bound, bound);
  return dist(randgen());
}

int64_t uniform_i64_bounds(const int64_t lb, const int64_t ub) {
  std::uniform_int_distribution<int64_t> dist(lb, ub);
  return dist(randgen());
}

__int128_t uniform_i128_bounds(const __int128_t lb, const __int128_t ub) {
  std::uniform_int_distribution<__int128_t> dist(lb, ub);
  return dist(randgen());
}

double random_f64_gaussian(double stdev) {
  std::normal_distribution<double> dist(0, stdev);
  return dist(randgen());
}

double uniform_f64_bounds(const double lb, const double ub) {
  std::uniform_real_distribution<double> dist(lb, ub);
  return dist(randgen());
}

double uniform_f64_01() {
  return uniform_f64_bounds(0, 1);
}
