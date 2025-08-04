#include "sha3.h"
#include "test_commons.h"

/** @brief returns some pseudorandom hash of the content */
thash test_hash(const void* data, uint64_t size) {
  thash res;
  sha3(data, size, &res, sizeof(res));
  return res;
}
/** @brief class to return a pseudorandom hash of the content */
test_hasher::test_hasher() {
  md = malloc(sizeof(sha3_ctx_t));
  sha3_init((sha3_ctx_t*)md, 16);
}

void test_hasher::update(const void* data, uint64_t size) { sha3_update((sha3_ctx_t*)md, data, size); }

thash test_hasher::hash() {
  thash res;
  sha3_final(&res, (sha3_ctx_t*)md);
  return res;
}

test_hasher::~test_hasher() { free(md); }
