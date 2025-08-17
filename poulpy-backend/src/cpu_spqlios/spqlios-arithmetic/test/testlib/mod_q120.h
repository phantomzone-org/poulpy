#ifndef SPQLIOS_MOD_Q120_H
#define SPQLIOS_MOD_Q120_H

#include <cstdint>

#include "../../spqlios/q120/q120_common.h"
#include "test_commons.h"

/** @brief centered modulo q */
int64_t centermod(int64_t v, int64_t q);
int64_t centermod(uint64_t v, int64_t q);

/** @brief this class represents an integer mod Q120 */
class mod_q120 {
 public:
  static constexpr int64_t Qi[] = {Q1, Q2, Q3, Q4};
  int64_t a[4];
  mod_q120(int64_t a1, int64_t a2, int64_t a3, int64_t a4);
  mod_q120();
  __int128_t to_int128() const;
  static mod_q120 from_q120a(const void* addr);
  static mod_q120 from_q120b(const void* addr);
  static mod_q120 from_q120c(const void* addr);
  void save_as_q120a(void* dest) const;
  void save_as_q120b(void* dest) const;
  void save_as_q120c(void* dest) const;
};

mod_q120 operator+(const mod_q120& x, const mod_q120& y);
mod_q120 operator-(const mod_q120& x, const mod_q120& y);
mod_q120 operator*(const mod_q120& x, const mod_q120& y);
mod_q120& operator+=(mod_q120& x, const mod_q120& y);
mod_q120& operator-=(mod_q120& x, const mod_q120& y);
mod_q120& operator*=(mod_q120& x, const mod_q120& y);
std::ostream& operator<<(std::ostream& out, const mod_q120& x);
bool operator==(const mod_q120& x, const mod_q120& y);
mod_q120 pow(const mod_q120& x, int32_t k);
mod_q120 half(const mod_q120& x);

/** @brief a uniformly drawn number mod Q120 */
mod_q120 uniform_q120();
/** @brief a uniformly random mod Q120 layout A (4 integers < 2^32) */
void uniform_q120a(void* dest);
/** @brief a uniformly random mod Q120 layout B (4 integers < 2^64) */
void uniform_q120b(void* dest);
/** @brief a uniformly random mod Q120 layout C (4 integers repr. x,2^32x) */
void uniform_q120c(void* dest);

#endif  // SPQLIOS_MOD_Q120_H
