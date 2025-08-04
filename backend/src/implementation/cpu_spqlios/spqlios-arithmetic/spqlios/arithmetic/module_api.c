#include <string.h>

#include "vec_znx_arithmetic_private.h"

static void fill_generic_virtual_table(MODULE* module) {
  // TODO add default ref handler here
  module->func.vec_znx_zero = vec_znx_zero_ref;
  module->func.vec_znx_copy = vec_znx_copy_ref;
  module->func.vec_znx_negate = vec_znx_negate_ref;
  module->func.vec_znx_add = vec_znx_add_ref;
  module->func.vec_znx_sub = vec_znx_sub_ref;
  module->func.vec_znx_rotate = vec_znx_rotate_ref;
  module->func.vec_znx_automorphism = vec_znx_automorphism_ref;
  module->func.vec_znx_normalize_base2k = vec_znx_normalize_base2k_ref;
  module->func.vec_znx_normalize_base2k_tmp_bytes = vec_znx_normalize_base2k_tmp_bytes_ref;
  if (CPU_SUPPORTS("avx2")) {
    // TODO add avx handlers here
    module->func.vec_znx_negate = vec_znx_negate_avx;
    module->func.vec_znx_add = vec_znx_add_avx;
    module->func.vec_znx_sub = vec_znx_sub_avx;
  }
}

static void fill_fft64_virtual_table(MODULE* module) {
  // TODO add default ref handler here
  // module->func.vec_znx_dft = ...;
  module->func.vec_znx_big_normalize_base2k = fft64_vec_znx_big_normalize_base2k;
  module->func.vec_znx_big_normalize_base2k_tmp_bytes = fft64_vec_znx_big_normalize_base2k_tmp_bytes;
  module->func.vec_znx_big_range_normalize_base2k = fft64_vec_znx_big_range_normalize_base2k;
  module->func.vec_znx_big_range_normalize_base2k_tmp_bytes = fft64_vec_znx_big_range_normalize_base2k_tmp_bytes;
  module->func.vec_znx_dft_automorphism = fft64_vec_znx_dft_automorphism_ref;
  module->func.vec_znx_dft_automorphism_tmp_bytes = fft64_vec_znx_dft_automorphism_tmp_bytes;
  module->func.vec_znx_dft = fft64_vec_znx_dft;
  module->func.vec_znx_idft = fft64_vec_znx_idft;
  module->func.vec_dft_add = fft64_vec_dft_add;
  module->func.vec_dft_sub = fft64_vec_dft_sub;
  module->func.vec_znx_idft_tmp_bytes = fft64_vec_znx_idft_tmp_bytes;
  module->func.vec_znx_idft_tmp_a = fft64_vec_znx_idft_tmp_a;
  module->func.vec_znx_big_add = fft64_vec_znx_big_add;
  module->func.vec_znx_big_add_small = fft64_vec_znx_big_add_small;
  module->func.vec_znx_big_add_small2 = fft64_vec_znx_big_add_small2;
  module->func.vec_znx_big_sub = fft64_vec_znx_big_sub;
  module->func.vec_znx_big_sub_small_a = fft64_vec_znx_big_sub_small_a;
  module->func.vec_znx_big_sub_small_b = fft64_vec_znx_big_sub_small_b;
  module->func.vec_znx_big_sub_small2 = fft64_vec_znx_big_sub_small2;
  module->func.vec_znx_big_rotate = fft64_vec_znx_big_rotate;
  module->func.vec_znx_big_automorphism = fft64_vec_znx_big_automorphism;
  module->func.svp_prepare = fft64_svp_prepare_ref;
  module->func.svp_apply_dft = fft64_svp_apply_dft_ref;
  module->func.svp_apply_dft_to_dft = fft64_svp_apply_dft_to_dft_ref;
  module->func.znx_small_single_product = fft64_znx_small_single_product;
  module->func.znx_small_single_product_tmp_bytes = fft64_znx_small_single_product_tmp_bytes;
  module->func.vmp_prepare_contiguous = fft64_vmp_prepare_contiguous_ref;
  module->func.vmp_prepare_tmp_bytes = fft64_vmp_prepare_tmp_bytes;
  module->func.vmp_apply_dft = fft64_vmp_apply_dft_ref;
  module->func.vmp_apply_dft_add = fft64_vmp_apply_dft_add_ref;
  module->func.vmp_apply_dft_tmp_bytes = fft64_vmp_apply_dft_tmp_bytes;
  module->func.vmp_apply_dft_to_dft = fft64_vmp_apply_dft_to_dft_ref;
  module->func.vmp_apply_dft_to_dft_add = fft64_vmp_apply_dft_to_dft_add_ref;
  module->func.vmp_apply_dft_to_dft_tmp_bytes = fft64_vmp_apply_dft_to_dft_tmp_bytes;
  module->func.bytes_of_vec_znx_dft = fft64_bytes_of_vec_znx_dft;
  module->func.bytes_of_vec_znx_big = fft64_bytes_of_vec_znx_big;
  module->func.bytes_of_svp_ppol = fft64_bytes_of_svp_ppol;
  module->func.bytes_of_vmp_pmat = fft64_bytes_of_vmp_pmat;
  if (CPU_SUPPORTS("avx2")) {
    // TODO add avx handlers here
    // TODO: enable when avx implementation is done
    module->func.vmp_prepare_contiguous = fft64_vmp_prepare_contiguous_avx;
    module->func.vmp_apply_dft = fft64_vmp_apply_dft_avx;
    module->func.vmp_apply_dft_add = fft64_vmp_apply_dft_add_avx;
    module->func.vmp_apply_dft_to_dft = fft64_vmp_apply_dft_to_dft_avx;
    module->func.vmp_apply_dft_to_dft_add = fft64_vmp_apply_dft_to_dft_add_avx;
  }
}

static void fill_ntt120_virtual_table(MODULE* module) {
  // TODO add default ref handler here
  // module->func.vec_znx_dft = ...;
  if (CPU_SUPPORTS("avx2")) {
    // TODO add avx handlers here
    module->func.vec_znx_dft = ntt120_vec_znx_dft_avx;
    module->func.vec_znx_idft = ntt120_vec_znx_idft_avx;
    module->func.vec_znx_idft_tmp_bytes = ntt120_vec_znx_idft_tmp_bytes_avx;
    module->func.vec_znx_idft_tmp_a = ntt120_vec_znx_idft_tmp_a_avx;
  }
}

static void fill_virtual_table(MODULE* module) {
  fill_generic_virtual_table(module);
  switch (module->module_type) {
    case FFT64:
      fill_fft64_virtual_table(module);
      break;
    case NTT120:
      fill_ntt120_virtual_table(module);
      break;
    default:
      NOT_SUPPORTED();  // invalid type
  }
}

static void fill_fft64_precomp(MODULE* module) {
  // fill any necessary precomp stuff
  module->mod.fft64.p_conv = new_reim_from_znx64_precomp(module->m, 50);
  module->mod.fft64.p_fft = new_reim_fft_precomp(module->m, 0);
  module->mod.fft64.p_reim_to_znx = new_reim_to_znx64_precomp(module->m, module->m, 63);
  module->mod.fft64.p_ifft = new_reim_ifft_precomp(module->m, 0);
  module->mod.fft64.p_addmul = new_reim_fftvec_addmul_precomp(module->m);
  module->mod.fft64.p_automorphism = new_reim_fftvec_automorphism_precomp(module->m);
  module->mod.fft64.mul_fft = new_reim_fftvec_mul_precomp(module->m);
  module->mod.fft64.add_fft = new_reim_fftvec_add_precomp(module->m);
  module->mod.fft64.sub_fft = new_reim_fftvec_sub_precomp(module->m);
}
static void fill_ntt120_precomp(MODULE* module) {
  // fill any necessary precomp stuff
  if (CPU_SUPPORTS("avx2")) {
    module->mod.q120.p_ntt = q120_new_ntt_bb_precomp(module->nn);
    module->mod.q120.p_intt = q120_new_intt_bb_precomp(module->nn);
  }
}

static void fill_module_precomp(MODULE* module) {
  switch (module->module_type) {
    case FFT64:
      fill_fft64_precomp(module);
      break;
    case NTT120:
      fill_ntt120_precomp(module);
      break;
    default:
      NOT_SUPPORTED();  // invalid type
  }
}

static void fill_module(MODULE* module, uint64_t nn, MODULE_TYPE mtype) {
  // init to zero to ensure that any non-initialized field bug is detected
  // by at least a "proper" segfault
  memset(module, 0, sizeof(MODULE));
  module->module_type = mtype;
  module->nn = nn;
  module->m = nn >> 1;
  fill_module_precomp(module);
  fill_virtual_table(module);
}

EXPORT MODULE* new_module_info(uint64_t N, MODULE_TYPE mtype) {
  MODULE* m = (MODULE*)malloc(sizeof(MODULE));
  fill_module(m, N, mtype);
  return m;
}

EXPORT void delete_module_info(MODULE* mod) {
  switch (mod->module_type) {
    case FFT64:
      free(mod->mod.fft64.p_conv);
      free(mod->mod.fft64.p_fft);
      free(mod->mod.fft64.p_ifft);
      free(mod->mod.fft64.p_reim_to_znx);
      free(mod->mod.fft64.mul_fft);
      free(mod->mod.fft64.p_addmul);
      free(mod->mod.fft64.p_automorphism);
      break;
    case NTT120:
      if (CPU_SUPPORTS("avx2")) {
        q120_del_ntt_bb_precomp(mod->mod.q120.p_ntt);
        q120_del_intt_bb_precomp(mod->mod.q120.p_intt);
      }
      break;
    default:
      break;
  }
  free(mod);
}

EXPORT uint64_t module_get_n(const MODULE* module) { return module->nn; }
