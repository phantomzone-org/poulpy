use crate::{
    ciphertext::Ciphertext,
    elem::{Elem, ElemVecZnx, VecZnxCommon},
};
use base2k::{
    Infos, Module, VecZnxApi, VecZnxBig, VecZnxBigOps, VecZnxDft, VecZnxDftOps, VmpPMat, VmpPMatOps,
};
use std::cmp::min;

pub fn gadget_product_tmp_bytes(
    module: &Module,
    log_base2k: usize,
    out_log_q: usize,
    in_log_q: usize,
    gct_rows: usize,
    gct_log_q: usize,
) -> usize {
    let gct_cols: usize = (gct_log_q + log_base2k - 1) / log_base2k;
    let in_limbs: usize = (in_log_q + log_base2k - 1) / log_base2k;
    let out_limbs: usize = (out_log_q + log_base2k - 1) / log_base2k;
    module.vmp_apply_dft_to_dft_tmp_bytes(out_limbs, in_limbs, gct_rows, gct_cols)
        + 2 * module.bytes_of_vec_znx_dft(gct_cols)
}

pub fn gadget_product_inplace_thread_safe<const OVERWRITE: bool, T: VecZnxApi<Owned = T> + Infos>(
    module: &Module,
    res: &mut Elem<T>,
    b: &Ciphertext<VmpPMat>,
    tmp_bytes: &mut [u8],
) where
    T: VecZnxCommon,
    Elem<T>: Infos + ElemVecZnx<T>,
{
    unsafe {
        let a_ptr: *const T = res.at(1) as *const T;
        gadget_product_thread_safe::<OVERWRITE, T>(module, res, &*a_ptr, b, tmp_bytes);
    }
}

/// Evaluates the gadget product res <- a x b.
///
/// # Arguments
///
/// * `module`: backend support for operations mod (X^N + 1).
/// * `res`: an [Elem] to store (-cs + m * a + e, c) with res_ncols limbs.
/// * `a`: a [VecZnx] of a_ncols limbs.
/// * `b`: a [GadgetCiphertext] as a vector of (-Bs + m * 2^{-k} + E, B)
///       containing b_nrows [VecZnx], each of b_ncols limbs.
///
/// # Computation
///
/// res = sum[min(a_ncols, b_nrows)] decomp(a, i) * (-B[i]s + m * 2^{-k*i} + E[i], B[i])
///     = (cs + m * a + e, c) with min(res_limbs, b_cols) limbs.
pub fn gadget_product_thread_safe<const OVERWRITE: bool, T: VecZnxApi<Owned = T> + Infos>(
    module: &Module,
    res: &mut Elem<T>,
    a: &T,
    b: &Ciphertext<VmpPMat>,
    tmp_bytes: &mut [u8],
) where
    T: VecZnxCommon,
    Elem<T>: Infos + ElemVecZnx<T>,
{
    let log_base2k: usize = b.log_base2k();
    let rows: usize = min(b.rows(), a.cols());
    let cols: usize = b.cols();

    let bytes_vmp_apply_dft: usize =
        module.vmp_apply_dft_to_dft_tmp_bytes(cols, a.cols(), rows, cols);
    let bytes_vec_znx_dft: usize = module.bytes_of_vec_znx_dft(cols);

    let (tmp_bytes_vmp_apply_dft, tmp_bytes) = tmp_bytes.split_at_mut(bytes_vmp_apply_dft);
    let (tmp_bytes_c1_dft, tmp_bytes_res_dft) = tmp_bytes.split_at_mut(bytes_vec_znx_dft);

    let mut c1_dft: VecZnxDft = module.new_vec_znx_dft_from_bytes(cols, tmp_bytes_c1_dft);
    let mut res_dft: VecZnxDft = module.new_vec_znx_dft_from_bytes(cols, tmp_bytes_res_dft);
    let mut res_big: VecZnxBig = res_dft.as_vec_znx_big();

    // Alias c0 and c1 part of res_big
    let (tmp_bytes_res_dft_c0, tmp_bytes_res_dft_c1) =
        tmp_bytes_res_dft.split_at_mut(bytes_vec_znx_dft >> 1);
    let res_big_c0: VecZnxBig = module.new_vec_znx_big_from_bytes(cols >> 1, tmp_bytes_res_dft_c0);
    let mut res_big_c1: VecZnxBig =
        module.new_vec_znx_big_from_bytes(cols >> 1, tmp_bytes_res_dft_c1);

    // a_dft <- DFT(a)
    module.vec_znx_dft(&mut c1_dft, a, a.cols());

    // (n x cols) <- (n x limbs=rows) x (rows x cols)
    // res_dft[a * (G0|G1)] <- sum[rows] DFT(a) x (DFT(G0)|DFT(G1))
    module.vmp_apply_dft_to_dft(
        &mut res_dft,
        &c1_dft,
        &b.0.value[0],
        tmp_bytes_vmp_apply_dft,
    );

    // res_big[a * (G0|G1)] <- IDFT(res_dft[a * (G0|G1)])
    module.vec_znx_idft_tmp_a(&mut res_big, &mut res_dft, cols);

    // res_big <- res[0] + res_big[a*G0]
    module.vec_znx_big_add_small_inplace(&mut res_big, res.at(0));
    module.vec_znx_big_normalize(log_base2k, res.at_mut(0), &res_big_c0, tmp_bytes_c1_dft);

    if OVERWRITE {
        // res[1] = normalize(res_big[a*G1])
        module.vec_znx_big_normalize(log_base2k, res.at_mut(1), &res_big_c1, tmp_bytes_c1_dft);
    } else {
        // res[1] = normalize(res_big[a*G1] + res[1])
        module.vec_znx_big_add_small_inplace(&mut res_big_c1, res.at(1));
        module.vec_znx_big_normalize(log_base2k, res.at_mut(1), &res_big_c1, tmp_bytes_c1_dft);
    }
}

pub fn rgsw_product_thread_safe<T: VecZnxApi<Owned = T> + Infos>(
    module: &Module,
    res: &mut Elem<T>,
    a: &Ciphertext<T>,
    b: &Ciphertext<VmpPMat>,
    tmp_bytes: &mut [u8],
) where
    T: VecZnxCommon,
    Elem<T>: Infos + ElemVecZnx<T>,
{
    let log_base2k: usize = b.log_base2k();
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
    let mut tmp_dft: VecZnxDft = module.new_vec_znx_dft_from_bytes(cols, tmp_bytes_tmp_dft);
    let mut r1_dft: VecZnxDft = module.new_vec_znx_dft_from_bytes(cols, tmp_bytes_r1_dft);
    let mut r2_dft: VecZnxDft = module.new_vec_znx_dft_from_bytes(cols, tmp_bytes_r2_dft);

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
