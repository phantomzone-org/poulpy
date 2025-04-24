use crate::ciphertext::Ciphertext;
use crate::elem::ElemCommon;
use base2k::{Module, VecZnx, VecZnxBig, VecZnxBigOps, VecZnxDft, VecZnxDftOps, VmpPMat, VmpPMatOps, assert_alignement};
use std::cmp::min;

pub fn key_switch_tmp_bytes(module: &Module, log_base2k: usize, res_logq: usize, in_logq: usize, gct_logq: usize) -> usize {
    let gct_cols: usize = (gct_logq + log_base2k - 1) / log_base2k;
    let in_cols: usize = (in_logq + log_base2k - 1) / log_base2k;
    let res_cols: usize = (res_logq + log_base2k - 1) / log_base2k;
    return module.vmp_apply_dft_to_dft_tmp_bytes(res_cols, in_cols, in_cols, gct_cols)
        + module.bytes_of_vec_znx_dft(std::cmp::min(res_cols, in_cols))
        + module.bytes_of_vec_znx_dft(gct_cols);
}

pub fn key_switch_rlwe(
    module: &Module,
    c: &mut Ciphertext<VecZnx>,
    a: &Ciphertext<VecZnx>,
    b: &Ciphertext<VmpPMat>,
    b_cols: usize,
    tmp_bytes: &mut [u8],
) {
    key_switch_rlwe_core(module, c, a, b, b_cols, tmp_bytes);
}

pub fn key_switch_rlwe_inplace(
    module: &Module,
    a: &mut Ciphertext<VecZnx>,
    b: &Ciphertext<VmpPMat>,
    b_cols: usize,
    tmp_bytes: &mut [u8],
) {
    key_switch_rlwe_core(module, a, a, b, b_cols, tmp_bytes);
}

fn key_switch_rlwe_core(
    module: &Module,
    c: *mut Ciphertext<VecZnx>,
    a: *const Ciphertext<VecZnx>,
    b: &Ciphertext<VmpPMat>,
    b_cols: usize,
    tmp_bytes: &mut [u8],
) {
    // SAFETY WARNING: must ensure `c` and `a` are valid for read/write
    let c: &mut Ciphertext<VecZnx> = unsafe { &mut *c };
    let a: &Ciphertext<VecZnx> = unsafe { &*a };

    let cols: usize = min(min(c.cols(), a.cols()), b.rows());

    #[cfg(debug_assertions)]
    {
        assert!(b_cols <= b.cols());
        assert!(tmp_bytes.len() >= key_switch_tmp_bytes(module, c.cols(), a.cols(), b.rows(), b.cols()));
        assert_alignement(tmp_bytes.as_ptr());
    }

    let (tmp_bytes_a1_dft, tmp_bytes) = tmp_bytes.split_at_mut(module.bytes_of_vec_znx_dft(cols));
    let (tmp_bytes_res_dft, tmp_bytes) = tmp_bytes.split_at_mut(module.bytes_of_vec_znx_dft(b_cols));

    let mut a1_dft = module.new_vec_znx_dft_from_bytes_borrow(cols, tmp_bytes_a1_dft);
    let mut res_dft = module.new_vec_znx_dft_from_bytes_borrow(b_cols, tmp_bytes_res_dft);
    let mut res_big = res_dft.as_vec_znx_big();

    module.vec_znx_dft(&mut a1_dft, a.at(1));
    module.vmp_apply_dft_to_dft(&mut res_dft, &a1_dft, b.at(0), tmp_bytes);
    module.vec_znx_idft_tmp_a(&mut res_big, &mut res_dft);

    module.vec_znx_big_add_small_inplace(&mut res_big, a.at(0));
    module.vec_znx_big_normalize(c.log_base2k(), c.at_mut(0), &mut res_big, tmp_bytes);

    module.vmp_apply_dft_to_dft(&mut res_dft, &a1_dft, b.at(1), tmp_bytes);
    module.vec_znx_idft_tmp_a(&mut res_big, &mut res_dft);

    module.vec_znx_big_normalize(c.log_base2k(), c.at_mut(1), &mut res_big, tmp_bytes);
}

pub fn key_switch_grlwe(module: &Module, c: &mut Ciphertext<VecZnx>, a: &Ciphertext<VecZnx>, b: &Ciphertext<VmpPMat>) {}

pub fn key_switch_rgsw(module: &Module, c: &mut Ciphertext<VecZnx>, a: &Ciphertext<VecZnx>, b: &Ciphertext<VmpPMat>) {}
