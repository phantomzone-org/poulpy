use crate::{
    ciphertext::Ciphertext,
    elem::{Elem, ElemCommon},
    keys::SecretKey,
    parameters::Parameters,
    plaintext::Plaintext,
};
use base2k::{Module, SvpPPol, SvpPPolOps, VecZnx, VecZnxBigOps, VecZnxDft, VecZnxDftOps};
use std::cmp::min;

pub struct Decryptor {
    sk: SvpPPol,
}

impl Decryptor {
    pub fn new(params: &Parameters, sk: &SecretKey) -> Self {
        let mut sk_svp_ppol: SvpPPol = params.module().new_svp_ppol();
        sk.prepare(params.module(), &mut sk_svp_ppol);
        Self { sk: sk_svp_ppol }
    }
}

pub fn decrypt_rlwe_tmp_byte(module: &Module, limbs: usize) -> usize {
    module.bytes_of_vec_znx_dft(limbs) + module.vec_znx_big_normalize_tmp_bytes()
}

impl Parameters {
    pub fn decrypt_rlwe_tmp_byte(&self, log_q: usize) -> usize {
        decrypt_rlwe_tmp_byte(
            self.module(),
            (log_q + self.log_base2k() - 1) / self.log_base2k(),
        )
    }

    pub fn decrypt_rlwe(&self, res: &mut Plaintext, ct: &Ciphertext<VecZnx>, sk: &SvpPPol, tmp_bytes: &mut [u8]) {
        decrypt_rlwe(self.module(), &mut res.0, &ct.0, sk, tmp_bytes)
    }
}

pub fn decrypt_rlwe(module: &Module, res: &mut Elem<VecZnx>, a: &Elem<VecZnx>, sk: &SvpPPol, tmp_bytes: &mut [u8]) {
    let cols: usize = a.cols();

    assert!(
        tmp_bytes.len() >= decrypt_rlwe_tmp_byte(module, cols),
        "invalid tmp_bytes: tmp_bytes.len()={} < decrypt_rlwe_tmp_byte={}",
        tmp_bytes.len(),
        decrypt_rlwe_tmp_byte(module, cols)
    );

    let (tmp_bytes_vec_znx_dft, tmp_bytes_normalize) = tmp_bytes.split_at_mut(module.bytes_of_vec_znx_dft(cols));

    let mut res_dft: VecZnxDft = VecZnxDft::from_bytes_borrow(module, cols, tmp_bytes_vec_znx_dft);
    let mut res_big: base2k::VecZnxBig = res_dft.as_vec_znx_big();

    // res_dft <- DFT(ct[1]) * DFT(sk)
    module.svp_apply_dft(&mut res_dft, sk, a.at(1));
    // res_big <- ct[1] x sk
    module.vec_znx_idft_tmp_a(&mut res_big, &mut res_dft);
    // res_big <- ct[1] x sk + ct[0]
    module.vec_znx_big_add_small_inplace(&mut res_big, a.at(0));
    // res <- normalize(ct[1] x sk + ct[0])
    module.vec_znx_big_normalize(a.log_base2k(), res.at_mut(0), &res_big, tmp_bytes_normalize);

    res.log_base2k = a.log_base2k();
    res.log_q = min(res.log_q(), a.log_q());
    res.log_scale = a.log_scale();
}
