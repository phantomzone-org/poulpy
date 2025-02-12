use crate::{
    ciphertext::{Ciphertext, GadgetCiphertext},
    elem::Elem,
};
use base2k::{Infos, Module, VecZnx, VecZnxBig, VecZnxDft, VmpPMatOps};

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

/// Evaluates the gadget product res <- a x b.
///
/// # Arguments
///
/// * `module`: backend support for operations mod (X^N + 1)
/// * `res`: an [Elem] to store (-cs + m * a + e, c) with res_ncols limbs.
/// * `a`: a [VecZnx] of a_ncols limbs.
/// * `b`: a [GadgetCiphertext] as a vector of (-Bs + m * 2^{-k} + E, B)
///       containing b_nrows [VecZnx], each of b_ncols limbs.
///
/// # Computation
///
/// res = sum[min(a_ncols, b_nrows)] decomp(a, i) * (-B[i]s + m * 2^{-k*i} + E[i], B[i])
///     = (cs + m * a + e, c) with min(res_limbs, b_cols) limbs.
pub fn gadget_product_thread_safe(
    module: &Module,
    res: &mut Elem,
    a: &VecZnx,
    b: &GadgetCiphertext,
    tmp_bytes: &mut [u8],
) {
    let log_base2k: usize = b.log_base2k();
    let cols: usize = b.cols();

    let (tmp_bytes_vmp_apply_dft, tmp_bytes) =
        tmp_bytes.split_at_mut(module.bytes_of_vec_znx_dft(cols));

    let (tmp_bytes_c1_dft, tmp_bytes_res_dft) = tmp_bytes.split_at_mut(tmp_bytes.len() >> 1);

    let mut c1_dft: VecZnxDft = module.new_vec_znx_from_bytes(cols, tmp_bytes_c1_dft);
    let mut res_dft: VecZnxDft = module.new_vec_znx_from_bytes(cols, tmp_bytes_res_dft);
    let mut res_big: VecZnxBig = res_dft.as_vec_znx_big();

    // c1_dft <- DFT(c1) [cols]
    module.vec_znx_dft(&mut c1_dft, a, a.limbs());

    // res_dft <- sum[rows] DFT(c1)[cols] x GadgetCiphertext[0][cols]
    module.vmp_apply_dft_to_dft(&mut res_dft, &c1_dft, &b.value[0], tmp_bytes_vmp_apply_dft);

    // res_big <- IDFT(DFT(c1) x GadgetCiphertext[0])
    module.vec_znx_idft_tmp_a(&mut res_big, &mut res_dft, cols);

    // res[0] = normalize(c0 + c1_dft x GadgetCiphertext[0])
    module.vec_znx_big_normalize(log_base2k, res.at_mut(0), &res_big, tmp_bytes_vmp_apply_dft);

    // res_dft <- DFT(c1) x GadgetCiphertext[1]
    module.vmp_apply_dft_to_dft(&mut res_dft, &c1_dft, &b.value[1], tmp_bytes_vmp_apply_dft);

    // res_big <- IDFT(DFT(c1) x GadgetCiphertext[1])
    module.vec_znx_idft_tmp_a(&mut res_big, &mut res_dft, cols);

    // res[1] = normalize(c1_dft x GadgetCiphertext[1])
    module.vec_znx_big_normalize(log_base2k, res.at_mut(1), &res_big, tmp_bytes_vmp_apply_dft);
}
