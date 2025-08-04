#include "polynomial_vector.h"

#include <cstring>

#ifdef VALGRIND_MEM_TESTS
#include "valgrind/memcheck.h"
#endif

#define CANARY_PADDING (1024)
#define GARBAGE_VALUE (242)

znx_vec_i64_layout::znx_vec_i64_layout(uint64_t n, uint64_t size, uint64_t slice) : n(n), size(size), slice(slice) {
  REQUIRE_DRAMATICALLY(is_pow2(n), "not a power of 2" << n);
  REQUIRE_DRAMATICALLY(slice >= n, "slice too small" << slice << " < " << n);
  this->region = (uint8_t*)malloc(size * slice * sizeof(int64_t) + 2 * CANARY_PADDING);
  this->data_start = (int64_t*)(region + CANARY_PADDING);
  // ensure that any invalid value is kind-of garbage
  memset(region, GARBAGE_VALUE, size * slice * sizeof(int64_t) + 2 * CANARY_PADDING);
  // mark inter-slice memory as non accessible
#ifdef VALGRIND_MEM_TESTS
  VALGRIND_MAKE_MEM_NOACCESS(region, CANARY_PADDING);
  VALGRIND_MAKE_MEM_NOACCESS(region + size * slice * sizeof(int64_t) + CANARY_PADDING, CANARY_PADDING);
  for (uint64_t i = 0; i < size; ++i) {
    VALGRIND_MAKE_MEM_UNDEFINED(data_start + i * slice, n * sizeof(int64_t));
  }
  if (size != slice) {
    for (uint64_t i = 0; i < size; ++i) {
      VALGRIND_MAKE_MEM_NOACCESS(data_start + i * slice + n, (slice - n) * sizeof(int64_t));
    }
  }
#endif
}

znx_vec_i64_layout::~znx_vec_i64_layout() { free(region); }

znx_i64 znx_vec_i64_layout::get_copy_zext(uint64_t index) const {
  if (index < size) {
    return znx_i64(n, data_start + index * slice);
  } else {
    return znx_i64::zero(n);
  }
}

znx_i64 znx_vec_i64_layout::get_copy(uint64_t index) const {
  REQUIRE_DRAMATICALLY(index < size, "index overflow: " << index << " / " << size);
  return znx_i64(n, data_start + index * slice);
}

void znx_vec_i64_layout::set(uint64_t index, const znx_i64& elem) {
  REQUIRE_DRAMATICALLY(index < size, "index overflow: " << index << " / " << size);
  REQUIRE_DRAMATICALLY(elem.nn() == n, "incompatible ring dimensions: " << elem.nn() << " / " << n);
  elem.save_as(data_start + index * slice);
}

int64_t* znx_vec_i64_layout::data() { return data_start; }
const int64_t* znx_vec_i64_layout::data() const { return data_start; }

void znx_vec_i64_layout::fill_random(uint64_t bits) {
  for (uint64_t i = 0; i < size; ++i) {
    set(i, znx_i64::random_log2bound(n, bits));
  }
}
__uint128_t znx_vec_i64_layout::content_hash() const {
  test_hasher hasher;
  for (uint64_t i = 0; i < size; ++i) {
    hasher.update(data() + i * slice, n * sizeof(int64_t));
  }
  return hasher.hash();
}
