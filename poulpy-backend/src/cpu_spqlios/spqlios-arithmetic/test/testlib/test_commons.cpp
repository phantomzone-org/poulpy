#include "test_commons.h"

#include <inttypes.h>

std::ostream& operator<<(std::ostream& out, __int128_t x) {
  char c[35] = {0};
  snprintf(c, 35, "0x%016" PRIx64 "%016" PRIx64, uint64_t(x >> 64), uint64_t(x));
  return out << c;
}
std::ostream& operator<<(std::ostream& out, __uint128_t x) { return out << __int128_t(x); }
