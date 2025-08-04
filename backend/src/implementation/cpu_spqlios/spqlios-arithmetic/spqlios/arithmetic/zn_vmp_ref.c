#include <memory.h>

#include "zn_arithmetic_private.h"

/** @brief size in bytes of a prepared matrix (for custom allocation) */
EXPORT uint64_t bytes_of_zn32_vmp_pmat(const MOD_Z* module,            // N
                                       uint64_t nrows, uint64_t ncols  // dimensions
) {
  return (nrows * ncols + 7) * sizeof(int32_t);
}

/** @brief allocates a prepared matrix (release with delete_zn32_vmp_pmat) */
EXPORT ZN32_VMP_PMAT* new_zn32_vmp_pmat(const MOD_Z* module,  // N
                                        uint64_t nrows, uint64_t ncols) {
  return (ZN32_VMP_PMAT*)spqlios_alloc(bytes_of_zn32_vmp_pmat(module, nrows, ncols));
}

/** @brief deletes a prepared matrix (release with free) */
EXPORT void delete_zn32_vmp_pmat(ZN32_VMP_PMAT* ptr) { spqlios_free(ptr); }

/** @brief prepares a vmp matrix (contiguous row-major version) */
EXPORT void default_zn32_vmp_prepare_contiguous_ref(  //
    const MOD_Z* module,
    ZN32_VMP_PMAT* pmat,                                // output
    const int32_t* mat, uint64_t nrows, uint64_t ncols  // a
) {
  int32_t* const out = (int32_t*)pmat;
  const uint64_t nblk = ncols >> 5;
  const uint64_t ncols_rem = ncols & 31;
  const uint64_t final_elems = (8 - nrows * ncols) & 7;
  for (uint64_t blk = 0; blk < nblk; ++blk) {
    int32_t* outblk = out + blk * nrows * 32;
    const int32_t* srcblk = mat + blk * 32;
    for (uint64_t row = 0; row < nrows; ++row) {
      int32_t* dest = outblk + row * 32;
      const int32_t* src = srcblk + row * ncols;
      for (uint64_t i = 0; i < 32; ++i) {
        dest[i] = src[i];
      }
    }
  }
  // copy the last block if any
  if (ncols_rem) {
    int32_t* outblk = out + nblk * nrows * 32;
    const int32_t* srcblk = mat + nblk * 32;
    for (uint64_t row = 0; row < nrows; ++row) {
      int32_t* dest = outblk + row * ncols_rem;
      const int32_t* src = srcblk + row * ncols;
      for (uint64_t i = 0; i < ncols_rem; ++i) {
        dest[i] = src[i];
      }
    }
  }
  // zero-out the final elements that may be accessed
  if (final_elems) {
    int32_t* f = out + nrows * ncols;
    for (uint64_t i = 0; i < final_elems; ++i) {
      f[i] = 0;
    }
  }
}

/** @brief prepares a vmp matrix (mat[row]+col*N points to the item) */
EXPORT void default_zn32_vmp_prepare_dblptr_ref(  //
    const MOD_Z* module,
    ZN32_VMP_PMAT* pmat,                                 // output
    const int32_t** mat, uint64_t nrows, uint64_t ncols  // a
) {
  for (uint64_t row_i = 0; row_i < nrows; ++row_i) {
    default_zn32_vmp_prepare_row_ref(module, pmat, mat[row_i], row_i, nrows, ncols);
  }
}

/** @brief prepares the ith-row of a vmp matrix with nrows and ncols */
EXPORT void default_zn32_vmp_prepare_row_ref(  //
    const MOD_Z* module,
    ZN32_VMP_PMAT* pmat,                                                // output
    const int32_t* row, uint64_t row_i, uint64_t nrows, uint64_t ncols  // a
) {
  int32_t* const out = (int32_t*)pmat;
  const uint64_t nblk = ncols >> 5;
  const uint64_t ncols_rem = ncols & 31;
  const uint64_t final_elems = (row_i == nrows - 1) && (8 - nrows * ncols) & 7;
  for (uint64_t blk = 0; blk < nblk; ++blk) {
    int32_t* outblk = out + blk * nrows * 32;
    int32_t* dest = outblk + row_i * 32;
    const int32_t* src = row + blk * 32;
    for (uint64_t i = 0; i < 32; ++i) {
      dest[i] = src[i];
    }
  }
  // copy the last block if any
  if (ncols_rem) {
    int32_t* outblk = out + nblk * nrows * 32;
    int32_t* dest = outblk + row_i * ncols_rem;
    const int32_t* src = row + nblk * 32;
    for (uint64_t i = 0; i < ncols_rem; ++i) {
      dest[i] = src[i];
    }
  }
  // zero-out the final elements that may be accessed
  if (final_elems) {
    int32_t* f = out + nrows * ncols;
    for (uint64_t i = 0; i < final_elems; ++i) {
      f[i] = 0;
    }
  }
}

#if 0

#define IMPL_zn32_vec_ixxx_matyyycols_ref(NCOLS) \
  memset(res, 0, NCOLS * sizeof(int32_t));       \
  for (uint64_t row = 0; row < nrows; ++row) {   \
    int32_t ai = a[row];                         \
    const int32_t* bb = b + row * b_sl;          \
    for (uint64_t i = 0; i < NCOLS; ++i) {       \
      res[i] += ai * bb[i];                      \
    }                                            \
  }

#define IMPL_zn32_vec_ixxx_mat8cols_ref() IMPL_zn32_vec_ixxx_matyyycols_ref(8)
#define IMPL_zn32_vec_ixxx_mat16cols_ref() IMPL_zn32_vec_ixxx_matyyycols_ref(16)
#define IMPL_zn32_vec_ixxx_mat24cols_ref() IMPL_zn32_vec_ixxx_matyyycols_ref(24)
#define IMPL_zn32_vec_ixxx_mat32cols_ref() IMPL_zn32_vec_ixxx_matyyycols_ref(32)

void zn32_vec_i8_mat32cols_ref(uint64_t nrows, int32_t* res, const int8_t* a, const int32_t* b, uint64_t b_sl) {
  IMPL_zn32_vec_ixxx_mat32cols_ref()
}
void zn32_vec_i16_mat32cols_ref(uint64_t nrows, int32_t* res, const int16_t* a, const int32_t* b, uint64_t b_sl) {
  IMPL_zn32_vec_ixxx_mat32cols_ref()
}

void zn32_vec_i32_mat32cols_ref(uint64_t nrows, int32_t* res, const int32_t* a, const int32_t* b, uint64_t b_sl) {
  IMPL_zn32_vec_ixxx_mat32cols_ref()
}
void zn32_vec_i32_mat24cols_ref(uint64_t nrows, int32_t* res, const int32_t* a, const int32_t* b, uint64_t b_sl) {
  IMPL_zn32_vec_ixxx_mat24cols_ref()
}
void zn32_vec_i32_mat16cols_ref(uint64_t nrows, int32_t* res, const int32_t* a, const int32_t* b, uint64_t b_sl) {
  IMPL_zn32_vec_ixxx_mat16cols_ref()
}
void zn32_vec_i32_mat8cols_ref(uint64_t nrows, int32_t* res, const int32_t* a, const int32_t* b, uint64_t b_sl) {
  IMPL_zn32_vec_ixxx_mat8cols_ref()
}
typedef void (*zn32_vec_i32_mat8kcols_ref_f)(uint64_t nrows,                  //
                                             int32_t* res,                    //
                                             const int32_t* a,                //
                                             const int32_t* b, uint64_t b_sl  //
);
zn32_vec_i32_mat8kcols_ref_f zn32_vec_i32_mat8kcols_ref[4] = {  //
    zn32_vec_i32_mat8cols_ref, zn32_vec_i32_mat16cols_ref,      //
    zn32_vec_i32_mat24cols_ref, zn32_vec_i32_mat32cols_ref};

/** @brief applies a vmp product (int32_t* input) */
EXPORT void default_zn32_vmp_apply_i32_ref(const MOD_Z* module,                //
                                           int32_t* res, uint64_t res_size,    //
                                           const int32_t* a, uint64_t a_size,  //
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
    zn32_vec_i32_mat32cols_ref(rows, rr, a, mat, 32);
  }
  // last block
  if (ncolrem) {
    uint64_t orig_rem = ncols - (ncolblk << 5);
    uint64_t b_sl = orig_rem >= 32 ? 32 : orig_rem;
    int32_t tmp[32];
    zn32_vec_i32_mat8kcols_ref[(ncolrem - 1) >> 3](rows, tmp, a, mat, b_sl);
    memcpy(rr, tmp, ncolrem * sizeof(int32_t));
  }
  // trailing bytes
  memset(res + cols, 0, (res_size - cols) * sizeof(int32_t));
}

#endif
