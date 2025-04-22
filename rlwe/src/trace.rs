use crate::{automorphism::AutomorphismKey, ciphertext::Ciphertext, elem::ElemCommon};
use base2k::{
    Module, VecZnx, VecZnxBig, VecZnxBigOps, VecZnxDft, VecZnxDftOps, VecZnxOps, VmpPMatOps,
    assert_alignement,
};
use std::collections::HashMap;

pub fn trace_tmp_bytes(
    module: &Module,
    c_cols: usize,
    a_cols: usize,
    b_rows: usize,
    b_cols: usize,
) -> usize {
    return module.vmp_apply_dft_to_dft_tmp_bytes(c_cols, a_cols, b_rows, b_cols)
        + 2 * module.bytes_of_vec_znx_dft(std::cmp::min(c_cols, a_cols));
}

pub fn trace_inplace(
    module: &Module,
    a: &mut Ciphertext<VecZnx>,
    start: usize,
    end: usize,
    b: HashMap<i64, AutomorphismKey>,
    tmp_bytes: &mut [u8],
) {
    let cols: usize = a.cols();

    let b_rows: usize;
    let b_cols: usize;

    if let Some((_, key)) = b.iter().next() {
        b_rows = key.value.rows();
        b_cols = key.value.cols();
    } else {
        panic!("b: HashMap<i64, AutomorphismKey>, is empty")
    }

    #[cfg(debug_assertions)]
    {
        assert!(start <= end);
        assert!(end <= module.n());
        assert!(tmp_bytes.len() >= trace_tmp_bytes(module, cols, cols, b_rows, b_cols));
        assert_alignement(tmp_bytes.as_ptr());
    }

    let cols: usize = std::cmp::min(b_cols, a.cols());

    let (tmp_bytes_b1_dft, tmp_bytes) = tmp_bytes.split_at_mut(module.bytes_of_vec_znx_dft(cols));
    let (tmp_bytes_res_dft, tmp_bytes) = tmp_bytes.split_at_mut(module.bytes_of_vec_znx_dft(cols));

    let mut a1_dft: VecZnxDft = module.new_vec_znx_dft_from_bytes_borrow(cols, tmp_bytes_b1_dft);
    let mut res_dft: VecZnxDft = module.new_vec_znx_dft_from_bytes_borrow(cols, tmp_bytes_res_dft);
    let mut res_big: VecZnxBig = res_dft.as_vec_znx_big();

    let log_base2k: usize = a.log_base2k();

    (start..end).for_each(|i| {
        a.at_mut(0).rsh(log_base2k, 1, tmp_bytes);
        a.at_mut(1).rsh(log_base2k, 1, tmp_bytes);

        let p: i64;
        if i == 0 {
            p = -1;
        } else {
            p = module.galois_element(1 << (i - 1));
        }

        if let Some(key) = b.get(&p) {
            module.vec_znx_dft(&mut a1_dft, a.at(1));

            // a[0] = NORMALIZE(a[0] + AUTO(a[0] + IDFT(<DFT(a[1]), key[0]>)))
            module.vmp_apply_dft_to_dft(&mut res_dft, &a1_dft, key.value.at(0), tmp_bytes);
            module.vec_znx_idft_tmp_a(&mut res_big, &mut res_dft);
            module.vec_znx_big_add_small_inplace(&mut res_big, a.at(0));
            module.vec_znx_big_automorphism_inplace(p, &mut res_big);
            module.vec_znx_big_add_small_inplace(&mut res_big, a.at(0));
            module.vec_znx_big_normalize(a.log_base2k(), a.at_mut(0), &mut res_big, tmp_bytes);

            // a[1] = NORMALIZE(a[1] + AUTO(IDFT(<DFT(a[1]), key[1]>)))
            module.vmp_apply_dft_to_dft(&mut res_dft, &a1_dft, key.value.at(1), tmp_bytes);
            module.vec_znx_idft_tmp_a(&mut res_big, &mut res_dft);
            module.vec_znx_big_automorphism_inplace(p, &mut res_big);
            module.vec_znx_big_add_small_inplace(&mut res_big, a.at(1));
            module.vec_znx_big_normalize(a.log_base2k(), a.at_mut(1), &mut res_big, tmp_bytes);
        } else {
            panic!("b[{}] is empty", p)
        }
    })
}

#[cfg(test)]
mod test {
    use crate::{
        ciphertext::{Ciphertext, new_gadget_ciphertext},
        decryptor::decrypt_rlwe,
        elem::{Elem, ElemCommon, ElemVecZnx},
        encryptor::encrypt_grlwe_sk,
        gadget_product::gadget_product_core,
        keys::SecretKey,
        parameters::{Parameters, ParametersLiteral},
        plaintext::Plaintext,
    };
    use base2k::{
        BACKEND, Infos, Sampling, SvpPPolOps, VecZnx, VecZnxBig, VecZnxBigOps, VecZnxDft,
        VecZnxDftOps, VecZnxOps, VmpPMat, alloc_aligned_u8,
    };
    use sampling::source::{Source, new_seed};

    #[test]
    fn test_trace_inplace() {}
}
