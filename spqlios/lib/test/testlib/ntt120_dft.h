#ifndef SPQLIOS_NTT120_DFT_H
#define SPQLIOS_NTT120_DFT_H

#include <vector>

#include "../../spqlios/q120/q120_arithmetic.h"
#include "mod_q120.h"
#include "negacyclic_polynomial.h"
#include "test_commons.h"

class q120_nttvec {
 public:
  std::vector<mod_q120> v;
  q120_nttvec() = default;
  explicit q120_nttvec(uint64_t n);
  q120_nttvec(uint64_t n, const q120b* data);
  q120_nttvec(uint64_t n, const q120c* data);
  uint64_t nn() const;
  static q120_nttvec zero(uint64_t n);
  static q120_nttvec random(uint64_t n);
  void save_as(q120a* dest) const;
  void save_as(q120b* dest) const;
  void save_as(q120c* dest) const;
  mod_q120 get_blk(uint64_t blk) const;
};

q120_nttvec simple_ntt120(const znx_i64& polynomial);
znx_i128 simple_intt120(const q120_nttvec& fftvec);
bool operator==(const q120_nttvec& a, const q120_nttvec& b);

#endif  // SPQLIOS_NTT120_DFT_H
