#ifndef SPQLIOS_TEST_COMMONS_H
#define SPQLIOS_TEST_COMMONS_H

#include <iostream>
#include <random>

#include "../../spqlios/commons.h"

/** @brief macro that crashes if the condition are not met */
#define REQUIRE_DRAMATICALLY(req_contition, error_msg)                                                        \
  do {                                                                                                        \
    if (!(req_contition)) {                                                                                   \
      std::cerr << "REQUIREMENT FAILED at " << __FILE__ << ":" << __LINE__ << ": " << error_msg << std::endl; \
      abort();                                                                                                \
    }                                                                                                         \
  } while (0)

typedef std::default_random_engine test_rng;
/** @brief reference to the default test rng */
test_rng& randgen();
/** @brief uniformly random 64-bit uint */
uint64_t uniform_u64();
/** @brief uniformly random number <= 2^nbits-1 */
uint64_t uniform_u64_bits(uint64_t nbits);
/** @brief uniformly random signed 64-bit number */
int64_t uniform_i64();
/** @brief uniformly random signed |number| <= 2^nbits */
int64_t uniform_i64_bits(uint64_t nbits);
/** @brief uniformly random signed lb <= number <= ub */
int64_t uniform_i64_bounds(const int64_t lb, const int64_t ub);
/** @brief uniformly random signed lb <= number <= ub */
__int128_t uniform_i128_bounds(const __int128_t lb, const __int128_t ub);
/** @brief uniformly random gaussian float64 */
double random_f64_gaussian(double stdev = 1);
/** @brief uniformly random signed lb <= number <= ub */
double uniform_f64_bounds(const double lb, const double ub);
/** @brief uniformly random float64 in [0,1] */
double uniform_f64_01();
/** @brief random gaussian float64 */
double random_f64_gaussian(double stdev);

bool is_pow2(uint64_t n);

void* alloc64(uint64_t size);

typedef __uint128_t thash;
/** @brief returns some pseudorandom hash of a contiguous content */
thash test_hash(const void* data, uint64_t size);
/** @brief class to return a pseudorandom hash of a piecewise-defined content */
class test_hasher {
  void* md;
 public:
  test_hasher();
  test_hasher(const test_hasher&) = delete;
  void operator=(const test_hasher&) = delete;
  /**
   * @brief append input bytes.
   * The final hash only depends on the concatenation of bytes, not on the
   * way the content was split into multiple calls to update.
   */
  void update(const void* data, uint64_t size);
  /**
   * @brief returns the final hash.
   * no more calls to update(...) shall be issued after this call.
   */
  thash hash();
  ~test_hasher();
};

// not included by default, since it makes some versions of gtest not compile
// std::ostream& operator<<(std::ostream& out, __int128_t x);
// std::ostream& operator<<(std::ostream& out, __uint128_t x);

#endif  // SPQLIOS_TEST_COMMONS_H
