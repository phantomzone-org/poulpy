use crate::{
    ciphertext::Ciphertext,
    elem::{Elem, ElemCommon, ElemVecZnx, VecZnxCommon},
};
use base2k::{Module, VecZnxBig, VecZnxBigOps, VecZnxDft, VecZnxDftOps, VmpPMat, VmpPMatOps};
use std::cmp::min;

pub fn rgsw_product<T>(
    module: &Module,
    _res: &mut Elem<T>,
    a: &Ciphertext<T>,
    b: &Ciphertext<VmpPMat>,
    tmp_bytes: &mut [u8],
) where
    T: VecZnxCommon<Owned = T>,
    Elem<T>: ElemVecZnx<T>,
{
    let _log_base2k: usize = b.log_base2k();
    let rows: usize = min(b.rows(), a.cols());
    let cols: usize = b.cols();
    let in_cols = a.cols();
    let out_cols: usize = a.cols();

    let bytes_of_vec_znx_dft = module.bytes_of_vec_znx_dft(cols);
    let bytes_of_vmp_apply_dft_to_dft =
        module.vmp_apply_dft_to_dft_tmp_bytes(out_cols, in_cols, rows, cols);

    let (tmp_bytes_c0_dft, tmp_bytes) = tmp_bytes.split_at_mut(bytes_of_vec_znx_dft);
    let (tmp_bytes_c1_dft, tmp_bytes) = tmp_bytes.split_at_mut(bytes_of_vec_znx_dft);
    let (tmp_bytes_tmp_dft, tmp_bytes) = tmp_bytes.split_at_mut(bytes_of_vec_znx_dft);
    let (tmp_bytes_r1_dft, tmp_bytes) = tmp_bytes.split_at_mut(bytes_of_vec_znx_dft);
    let (tmp_bytes_r2_dft, tmp_bytes) = tmp_bytes.split_at_mut(bytes_of_vec_znx_dft);
    let (bytes_of_vmp_apply_dft_to_dft, tmp_bytes) =
        tmp_bytes.split_at_mut(bytes_of_vmp_apply_dft_to_dft);

    let mut c0_dft: VecZnxDft = module.new_vec_znx_dft_from_bytes(cols, tmp_bytes_c0_dft);
    let mut c1_dft: VecZnxDft = module.new_vec_znx_dft_from_bytes(cols, tmp_bytes_c1_dft);
    let mut _tmp_dft: VecZnxDft = module.new_vec_znx_dft_from_bytes(cols, tmp_bytes_tmp_dft);
    let mut r1_dft: VecZnxDft = module.new_vec_znx_dft_from_bytes(cols, tmp_bytes_r1_dft);
    let mut _r2_dft: VecZnxDft = module.new_vec_znx_dft_from_bytes(cols, tmp_bytes_r2_dft);

    // c0_dft <- DFT(a[0])
    module.vec_znx_dft(&mut c0_dft, a.at(0), in_cols);

    // r_dft <- sum[rows] c0_dft[cols] x RGSW[0][cols]
    module.vmp_apply_dft_to_dft(
        &mut r1_dft,
        &c1_dft,
        &b.0.value[0],
        bytes_of_vmp_apply_dft_to_dft,
    );

    // c1_dft <- DFT(a[1])
    module.vec_znx_dft(&mut c1_dft, a.at(1), in_cols);
}
