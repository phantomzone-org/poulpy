use crate::ciphertext::{Ciphertext, GadgetCiphertext};
use base2k::{Module, VecZnxBig, VecZnxDft, VmpPMatOps};

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

pub fn gadget_product_inplace(
    module: &Module,
    a: &mut Ciphertext,
    b: &GadgetCiphertext,
    tmp_bytes: &mut [u8],
) {
    // This is safe to do because the relevant values of a are copied to a buffer before being
    // overwritten.
    unsafe {
        let a_ptr: *mut Ciphertext = a;
        gadget_product(module, a, &*a_ptr, b, tmp_bytes)
    }
}

pub fn gadget_product(
    module: &Module,
    res: &mut Ciphertext,
    a: &Ciphertext,
    b: &GadgetCiphertext,
    tmp_bytes: &mut [u8],
) {
    assert!(
        a.log_base2k() == b.log_base2k(),
        "invalid inputs: a.log_base2k={} != b.log_base2k={}",
        a.log_base2k(),
        b.log_base2k()
    );

    let log_base2k: usize = b.log_base2k();
    let cols: usize = b.cols();

    let (tmp_bytes_vmp_apply_dft, tmp_bytes) =
        tmp_bytes.split_at_mut(module.bytes_of_vec_znx_dft(cols));

    let (tmp_bytes_c1_dft, tmp_bytes_res_dft) = tmp_bytes.split_at_mut(tmp_bytes.len() >> 1);

    let mut c1_dft: VecZnxDft = module.new_vec_znx_from_bytes(cols, tmp_bytes_c1_dft);
    let mut res_dft: VecZnxDft = module.new_vec_znx_from_bytes(cols, tmp_bytes_res_dft);
    let mut res_big: VecZnxBig = res_dft.as_vec_znx_big();

    // c1_dft <- DFT(b[1])
    module.vec_znx_dft(&mut c1_dft, a.at(1), a.limbs());

    // res_dft <- DFT(c1) x GadgetCiphertext[0]
    module.vmp_apply_dft_to_dft(&mut res_dft, &c1_dft, &b.value[0], tmp_bytes_vmp_apply_dft);

    // res_big <- IDFT(DFT(c1) x GadgetCiphertext[0])
    module.vec_znx_idft_tmp_a(&mut res_big, &mut res_dft, cols);

    // res_big <- c0 + c1_dft x GadgetCiphertext[0]
    module.vec_znx_big_add_small_inplace(&mut res_big, a.at(0), cols);

    // res[0] = normalize(c0 + c1_dft x GadgetCiphertext[0])
    module.vec_znx_big_normalize(log_base2k, res.at_mut(0), &res_big, tmp_bytes_vmp_apply_dft);

    // res_dft <- DFT(c1) x GadgetCiphertext[1]
    module.vmp_apply_dft_to_dft(&mut res_dft, &c1_dft, &b.value[1], tmp_bytes_vmp_apply_dft);

    // res_big <- IDFT(DFT(c1) x GadgetCiphertext[1])
    module.vec_znx_idft_tmp_a(&mut res_big, &mut res_dft, cols);

    // res[1] = normalize(c1_dft x GadgetCiphertext[1])
    module.vec_znx_big_normalize(log_base2k, res.at_mut(1), &res_big, tmp_bytes_vmp_apply_dft);
}
