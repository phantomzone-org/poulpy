use crate::{
    ciphertext::Ciphertext, keys::SecretKey, parameters::Parameters, plaintext::Plaintext,
};
use base2k::{Module, SvpPPol, SvpPPolOps, VecZnxDft};
use std::cmp::min;

pub struct Decryptor {
    sk: SvpPPol,
}

impl Decryptor {
    pub fn new(params: &Parameters, sk: &SecretKey) -> Self {
        let mut sk_svp_ppol: SvpPPol = params.module().svp_new_ppol();
        sk.prepare(params.module(), &mut sk_svp_ppol);
        Self { sk: sk_svp_ppol }
    }
}

pub fn decrypt_rlwe_thread_safe_tmp_byte(module: &Module, limbs: usize) -> usize {
    module.bytes_of_vec_znx_dft(limbs) + module.vec_znx_big_normalize_tmp_bytes()
}

impl Parameters {
    pub fn decrypt_rlwe_thread_safe_tmp_byte(&self, log_q: usize) -> usize {
        decrypt_rlwe_thread_safe_tmp_byte(
            self.module(),
            (log_q + self.log_base2k() - 1) / self.log_base2k(),
        )
    }

    pub fn decrypt_rlwe_thread_safe(
        &self,
        res: &mut Plaintext,
        ct: &Ciphertext,
        sk: &SvpPPol,
        tmp_bytes: &mut [u8],
    ) {
        decrypt_rlwe_thread_safe(self.module(), res, ct, sk, tmp_bytes)
    }
}

pub fn decrypt_rlwe_thread_safe(
    module: &Module,
    res: &mut Plaintext,
    ct: &Ciphertext,
    sk: &SvpPPol,
    tmp_bytes: &mut [u8],
) {
    let limbs: usize = min(res.limbs(), ct.limbs());

    assert!(
        tmp_bytes.len() >= decrypt_rlwe_thread_safe_tmp_byte(module, limbs),
        "invalid tmp_bytes: tmp_bytes.len()={} < decrypt_rlwe_thread_safe_tmp_byte={}",
        tmp_bytes.len(),
        decrypt_rlwe_thread_safe_tmp_byte(module, limbs)
    );

    let res_dft_bytes: usize = module.bytes_of_vec_znx_dft(limbs);

    let mut res_dft: VecZnxDft = VecZnxDft::from_bytes(limbs, tmp_bytes);
    let mut res_big: base2k::VecZnxBig = res_dft.as_vec_znx_big();

    // res_dft <- DFT(ct[1]) * DFT(sk)
    module.svp_apply_dft(&mut res_dft, sk, &ct.0.value[1], limbs);
    // res_big <- ct[1] x sk
    module.vec_znx_idft_tmp_a(&mut res_big, &mut res_dft, limbs);
    // res_big <- ct[1] x sk + ct[0]
    module.vec_znx_big_add_small_inplace(&mut res_big, &ct.0.value[0], limbs);
    // res <- normalize(ct[1] x sk + ct[0])
    module.vec_znx_big_normalize(
        ct.log_base2k(),
        res.at_mut(0),
        &res_big,
        &mut tmp_bytes[res_dft_bytes..],
    );

    res.0.log_base2k = ct.log_base2k();
    res.0.log_q = min(res.log_q(), ct.log_q());
    res.0.log_scale = ct.log_scale();
}
