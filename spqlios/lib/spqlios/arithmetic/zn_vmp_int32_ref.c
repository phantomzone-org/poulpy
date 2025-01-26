// This file is actually a template: it will be compiled multiple times with
// different INTTYPES
#ifndef INTTYPE
#define INTTYPE int32_t
#define INTSN i32
#endif

#include <memory.h>

#include "zn_arithmetic_private.h"

#define concat_inner(aa, bb, cc) aa##_##bb##_##cc
#define concat(aa, bb, cc) concat_inner(aa, bb, cc)
#define zn32_vec_fn(cc) concat(zn32_vec, INTSN, cc)

// the ref version shares the same implementation for each fixed column size
// optimized implementations may do something different.
static __always_inline void IMPL_zn32_vec_matcols_ref(
    const uint64_t NCOLS,            // fixed number of columns
    uint64_t nrows,                  // nrows of b
    int32_t* res,                    // result: size NCOLS, only the first min(b_sl, NCOLS) are relevant
    const INTTYPE* a,                // a: nrows-sized vector
    const int32_t* b, uint64_t b_sl  // b: nrows * min(b_sl, NCOLS) matrix
) {
  memset(res, 0, NCOLS * sizeof(int32_t));
  for (uint64_t row = 0; row < nrows; ++row) {
    int32_t ai = a[row];
    const int32_t* bb = b + row * b_sl;
    for (uint64_t i = 0; i < NCOLS; ++i) {
      res[i] += ai * bb[i];
    }
  }
}

void zn32_vec_fn(mat32cols_ref)(uint64_t nrows, int32_t* res, const INTTYPE* a, const int32_t* b, uint64_t b_sl) {
  IMPL_zn32_vec_matcols_ref(32, nrows, res, a, b, b_sl);
}
void zn32_vec_fn(mat24cols_ref)(uint64_t nrows, int32_t* res, const INTTYPE* a, const int32_t* b, uint64_t b_sl) {
  IMPL_zn32_vec_matcols_ref(24, nrows, res, a, b, b_sl);
}
void zn32_vec_fn(mat16cols_ref)(uint64_t nrows, int32_t* res, const INTTYPE* a, const int32_t* b, uint64_t b_sl) {
  IMPL_zn32_vec_matcols_ref(16, nrows, res, a, b, b_sl);
}
void zn32_vec_fn(mat8cols_ref)(uint64_t nrows, int32_t* res, const INTTYPE* a, const int32_t* b, uint64_t b_sl) {
  IMPL_zn32_vec_matcols_ref(8, nrows, res, a, b, b_sl);
}

typedef void (*vm_f)(uint64_t nrows,                  //
                     int32_t* res,                    //
                     const INTTYPE* a,                //
                     const int32_t* b, uint64_t b_sl  //
);
static const vm_f zn32_vec_mat8kcols_ref[4] = {  //
    zn32_vec_fn(mat8cols_ref),                   //
    zn32_vec_fn(mat16cols_ref),                  //
    zn32_vec_fn(mat24cols_ref),                  //
    zn32_vec_fn(mat32cols_ref)};

/** @brief applies a vmp product (int32_t* input) */
EXPORT void concat(default_zn32_vmp_apply, INTSN, ref)(  //
    const MOD_Z* module,                                 //
    int32_t* res, uint64_t res_size,                     //
    const INTTYPE* a, uint64_t a_size,                   //
    const ZN32_VMP_PMAT* pmat, uint64_t nrows, uint64_t ncols) {
  const uint64_t rows = a_size < nrows ? a_size : nrows;
  const uint64_t cols = res_size < ncols ? res_size : ncols;
  const uint64_t ncolblk = cols >> 5;
  const uint64_t ncolrem = cols & 31;
  // copy the first full blocks
  const uint32_t full_blk_size = nrows * 32;
  const int32_t* mat = (int32_t*)pmat;
  int32_t* rr = res;
  for (uint64_t blk = 0;  //
       blk < ncolblk;     //
       ++blk, mat += full_blk_size, rr += 32) {
    zn32_vec_fn(mat32cols_ref)(rows, rr, a, mat, 32);
  }
  // last block
  if (ncolrem) {
    uint64_t orig_rem = ncols - (ncolblk << 5);
    uint64_t b_sl = orig_rem >= 32 ? 32 : orig_rem;
    int32_t tmp[32];
    zn32_vec_mat8kcols_ref[(ncolrem - 1) >> 3](rows, tmp, a, mat, b_sl);
    memcpy(rr, tmp, ncolrem * sizeof(int32_t));
  }
  // trailing bytes
  memset(res + cols, 0, (res_size - cols) * sizeof(int32_t));
}
