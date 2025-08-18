// This file is actually a template: it will be compiled multiple times with
// different INTTYPES
#ifndef INTTYPE
#define INTTYPE int32_t
#define INTSN i32
#endif

#include <immintrin.h>
#include <memory.h>

#include "zn_arithmetic_private.h"

#define concat_inner(aa, bb, cc) aa##_##bb##_##cc
#define concat(aa, bb, cc) concat_inner(aa, bb, cc)
#define zn32_vec_fn(cc) concat(zn32_vec, INTSN, cc)

static void zn32_vec_mat32cols_avx_prefetch(uint64_t nrows, int32_t* res, const INTTYPE* a, const int32_t* b) {
  if (nrows == 0) {
    memset(res, 0, 32 * sizeof(int32_t));
    return;
  }
  const int32_t* bb = b;
  const int32_t* pref_bb = b;
  const uint64_t pref_iters = 128;
  const uint64_t pref_start = pref_iters < nrows ? pref_iters : nrows;
  const uint64_t pref_last = pref_iters > nrows ? 0 : nrows - pref_iters;
  // let's do some prefetching of the GSW key, since on some cpus,
  // it helps
  for (uint64_t i = 0; i < pref_start; ++i) {
    __builtin_prefetch(pref_bb, 0, _MM_HINT_T0);
    __builtin_prefetch(pref_bb + 16, 0, _MM_HINT_T0);
    pref_bb += 32;
  }
  // we do the first iteration
  __m256i x = _mm256_set1_epi32(a[0]);
  __m256i r0 = _mm256_mullo_epi32(x, _mm256_loadu_si256((__m256i*)(bb)));
  __m256i r1 = _mm256_mullo_epi32(x, _mm256_loadu_si256((__m256i*)(bb + 8)));
  __m256i r2 = _mm256_mullo_epi32(x, _mm256_loadu_si256((__m256i*)(bb + 16)));
  __m256i r3 = _mm256_mullo_epi32(x, _mm256_loadu_si256((__m256i*)(bb + 24)));
  bb += 32;
  uint64_t row = 1;
  for (;                 //
       row < pref_last;  //
       ++row, bb += 32) {
    // prefetch the next iteration
    __builtin_prefetch(pref_bb, 0, _MM_HINT_T0);
    __builtin_prefetch(pref_bb + 16, 0, _MM_HINT_T0);
    pref_bb += 32;
    INTTYPE ai = a[row];
    if (ai == 0) continue;
    x = _mm256_set1_epi32(ai);
    r0 = _mm256_add_epi32(r0, _mm256_mullo_epi32(x, _mm256_loadu_si256((__m256i*)(bb))));
    r1 = _mm256_add_epi32(r1, _mm256_mullo_epi32(x, _mm256_loadu_si256((__m256i*)(bb + 8))));
    r2 = _mm256_add_epi32(r2, _mm256_mullo_epi32(x, _mm256_loadu_si256((__m256i*)(bb + 16))));
    r3 = _mm256_add_epi32(r3, _mm256_mullo_epi32(x, _mm256_loadu_si256((__m256i*)(bb + 24))));
  }
  for (;             //
       row < nrows;  //
       ++row, bb += 32) {
    INTTYPE ai = a[row];
    if (ai == 0) continue;
    x = _mm256_set1_epi32(ai);
    r0 = _mm256_add_epi32(r0, _mm256_mullo_epi32(x, _mm256_loadu_si256((__m256i*)(bb))));
    r1 = _mm256_add_epi32(r1, _mm256_mullo_epi32(x, _mm256_loadu_si256((__m256i*)(bb + 8))));
    r2 = _mm256_add_epi32(r2, _mm256_mullo_epi32(x, _mm256_loadu_si256((__m256i*)(bb + 16))));
    r3 = _mm256_add_epi32(r3, _mm256_mullo_epi32(x, _mm256_loadu_si256((__m256i*)(bb + 24))));
  }
  _mm256_storeu_si256((__m256i*)(res), r0);
  _mm256_storeu_si256((__m256i*)(res + 8), r1);
  _mm256_storeu_si256((__m256i*)(res + 16), r2);
  _mm256_storeu_si256((__m256i*)(res + 24), r3);
}

void zn32_vec_fn(mat32cols_avx)(uint64_t nrows, int32_t* res, const INTTYPE* a, const int32_t* b, uint64_t b_sl) {
  if (nrows == 0) {
    memset(res, 0, 32 * sizeof(int32_t));
    return;
  }
  const INTTYPE* aa = a;
  const INTTYPE* const aaend = a + nrows;
  const int32_t* bb = b;
  __m256i x = _mm256_set1_epi32(*aa);
  __m256i r0 = _mm256_mullo_epi32(x, _mm256_loadu_si256((__m256i*)(bb)));
  __m256i r1 = _mm256_mullo_epi32(x, _mm256_loadu_si256((__m256i*)(bb + 8)));
  __m256i r2 = _mm256_mullo_epi32(x, _mm256_loadu_si256((__m256i*)(bb + 16)));
  __m256i r3 = _mm256_mullo_epi32(x, _mm256_loadu_si256((__m256i*)(bb + 24)));
  bb += b_sl;
  ++aa;
  for (;            //
       aa < aaend;  //
       bb += b_sl, ++aa) {
    INTTYPE ai = *aa;
    if (ai == 0) continue;
    x = _mm256_set1_epi32(ai);
    r0 = _mm256_add_epi32(r0, _mm256_mullo_epi32(x, _mm256_loadu_si256((__m256i*)(bb))));
    r1 = _mm256_add_epi32(r1, _mm256_mullo_epi32(x, _mm256_loadu_si256((__m256i*)(bb + 8))));
    r2 = _mm256_add_epi32(r2, _mm256_mullo_epi32(x, _mm256_loadu_si256((__m256i*)(bb + 16))));
    r3 = _mm256_add_epi32(r3, _mm256_mullo_epi32(x, _mm256_loadu_si256((__m256i*)(bb + 24))));
  }
  _mm256_storeu_si256((__m256i*)(res), r0);
  _mm256_storeu_si256((__m256i*)(res + 8), r1);
  _mm256_storeu_si256((__m256i*)(res + 16), r2);
  _mm256_storeu_si256((__m256i*)(res + 24), r3);
}

void zn32_vec_fn(mat24cols_avx)(uint64_t nrows, int32_t* res, const INTTYPE* a, const int32_t* b, uint64_t b_sl) {
  if (nrows == 0) {
    memset(res, 0, 24 * sizeof(int32_t));
    return;
  }
  const INTTYPE* aa = a;
  const INTTYPE* const aaend = a + nrows;
  const int32_t* bb = b;
  __m256i x = _mm256_set1_epi32(*aa);
  __m256i r0 = _mm256_mullo_epi32(x, _mm256_loadu_si256((__m256i*)(bb)));
  __m256i r1 = _mm256_mullo_epi32(x, _mm256_loadu_si256((__m256i*)(bb + 8)));
  __m256i r2 = _mm256_mullo_epi32(x, _mm256_loadu_si256((__m256i*)(bb + 16)));
  bb += b_sl;
  ++aa;
  for (;            //
       aa < aaend;  //
       bb += b_sl, ++aa) {
    INTTYPE ai = *aa;
    if (ai == 0) continue;
    x = _mm256_set1_epi32(ai);
    r0 = _mm256_add_epi32(r0, _mm256_mullo_epi32(x, _mm256_loadu_si256((__m256i*)(bb))));
    r1 = _mm256_add_epi32(r1, _mm256_mullo_epi32(x, _mm256_loadu_si256((__m256i*)(bb + 8))));
    r2 = _mm256_add_epi32(r2, _mm256_mullo_epi32(x, _mm256_loadu_si256((__m256i*)(bb + 16))));
  }
  _mm256_storeu_si256((__m256i*)(res), r0);
  _mm256_storeu_si256((__m256i*)(res + 8), r1);
  _mm256_storeu_si256((__m256i*)(res + 16), r2);
}
void zn32_vec_fn(mat16cols_avx)(uint64_t nrows, int32_t* res, const INTTYPE* a, const int32_t* b, uint64_t b_sl) {
  if (nrows == 0) {
    memset(res, 0, 16 * sizeof(int32_t));
    return;
  }
  const INTTYPE* aa = a;
  const INTTYPE* const aaend = a + nrows;
  const int32_t* bb = b;
  __m256i x = _mm256_set1_epi32(*aa);
  __m256i r0 = _mm256_mullo_epi32(x, _mm256_loadu_si256((__m256i*)(bb)));
  __m256i r1 = _mm256_mullo_epi32(x, _mm256_loadu_si256((__m256i*)(bb + 8)));
  bb += b_sl;
  ++aa;
  for (;            //
       aa < aaend;  //
       bb += b_sl, ++aa) {
    INTTYPE ai = *aa;
    if (ai == 0) continue;
    x = _mm256_set1_epi32(ai);
    r0 = _mm256_add_epi32(r0, _mm256_mullo_epi32(x, _mm256_loadu_si256((__m256i*)(bb))));
    r1 = _mm256_add_epi32(r1, _mm256_mullo_epi32(x, _mm256_loadu_si256((__m256i*)(bb + 8))));
  }
  _mm256_storeu_si256((__m256i*)(res), r0);
  _mm256_storeu_si256((__m256i*)(res + 8), r1);
}

void zn32_vec_fn(mat8cols_avx)(uint64_t nrows, int32_t* res, const INTTYPE* a, const int32_t* b, uint64_t b_sl) {
  if (nrows == 0) {
    memset(res, 0, 8 * sizeof(int32_t));
    return;
  }
  const INTTYPE* aa = a;
  const INTTYPE* const aaend = a + nrows;
  const int32_t* bb = b;
  __m256i x = _mm256_set1_epi32(*aa);
  __m256i r0 = _mm256_mullo_epi32(x, _mm256_loadu_si256((__m256i*)(bb)));
  bb += b_sl;
  ++aa;
  for (;            //
       aa < aaend;  //
       bb += b_sl, ++aa) {
    INTTYPE ai = *aa;
    if (ai == 0) continue;
    x = _mm256_set1_epi32(ai);
    r0 = _mm256_add_epi32(r0, _mm256_mullo_epi32(x, _mm256_loadu_si256((__m256i*)(bb))));
  }
  _mm256_storeu_si256((__m256i*)(res), r0);
}

typedef void (*vm_f)(uint64_t nrows,                  //
                     int32_t* res,                    //
                     const INTTYPE* a,                //
                     const int32_t* b, uint64_t b_sl  //
);
static const vm_f zn32_vec_mat8kcols_avx[4] = {  //
    zn32_vec_fn(mat8cols_avx),                   //
    zn32_vec_fn(mat16cols_avx),                  //
    zn32_vec_fn(mat24cols_avx),                  //
    zn32_vec_fn(mat32cols_avx)};

/** @brief applies a vmp product (int32_t* input) */
EXPORT void concat(default_zn32_vmp_apply, INTSN, avx)(  //
    const MOD_Z* module,                                 //
    int32_t* res, uint64_t res_size,                     //
    const INTTYPE* a, uint64_t a_size,                   //
    const ZN32_VMP_PMAT* pmat, uint64_t nrows, uint64_t ncols) {
  const uint64_t rows = a_size < nrows ? a_size : nrows;
  const uint64_t cols = res_size < ncols ? res_size : ncols;
  const uint64_t ncolblk = cols >> 5;
  const uint64_t ncolrem = cols & 31;
  // copy the first full blocks
  const uint64_t full_blk_size = nrows * 32;
  const int32_t* mat = (int32_t*)pmat;
  int32_t* rr = res;
  for (uint64_t blk = 0;  //
       blk < ncolblk;     //
       ++blk, mat += full_blk_size, rr += 32) {
    zn32_vec_mat32cols_avx_prefetch(rows, rr, a, mat);
  }
  // last block
  if (ncolrem) {
    uint64_t orig_rem = ncols - (ncolblk << 5);
    uint64_t b_sl = orig_rem >= 32 ? 32 : orig_rem;
    int32_t tmp[32];
    zn32_vec_mat8kcols_avx[(ncolrem - 1) >> 3](rows, tmp, a, mat, b_sl);
    memcpy(rr, tmp, ncolrem * sizeof(int32_t));
  }
  // trailing bytes
  memset(res + cols, 0, (res_size - cols) * sizeof(int32_t));
}
