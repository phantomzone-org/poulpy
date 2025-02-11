use crate::ciphertext::GadgetCiphertext;
use crate::elem::Elem;
use crate::encryptor::{encrypt_rlwe_sk_thread_safe, encrypt_rlwe_sk_tmp_bytes};
use crate::parameters::Parameters;
use base2k::{Module, Scalar, SvpPPol, SvpPPolOps, VmpPMat, VmpPMatOps};
use sampling::source::Source;

pub struct SecretKey(pub Scalar);

impl SecretKey {
    pub fn new(params: &Module) -> Self {
        SecretKey(Scalar::new(params.n()))
    }

    pub fn fill_ternary_prob(&mut self, prob: f64, source: &mut Source) {
        self.0.fill_ternary_prob(prob, source);
    }

    pub fn fill_ternary_hw(&mut self, hw: usize, source: &mut Source) {
        self.0.fill_ternary_hw(hw, source);
    }

    pub fn prepare(&self, module: &Module, sk_ppol: &mut SvpPPol) {
        module.svp_prepare(sk_ppol, &self.0)
    }
}

pub struct PublicKey(pub Elem);

impl PublicKey {
    pub fn new(module: &Module, log_base2k: usize, log_q: usize) -> PublicKey {
        PublicKey(Elem::new(module, log_base2k, log_q, 1, 0))
    }

    pub fn gen_thread_safe(
        &mut self,
        module: &Module,
        sk: &SvpPPol,
        xe: f64,
        xa_source: &mut Source,
        xe_source: &mut Source,
        tmp_bytes: &mut [u8],
    ) {
        encrypt_rlwe_sk_thread_safe(
            module,
            &mut self.0,
            None,
            sk,
            xa_source,
            xe_source,
            xe,
            tmp_bytes,
        );
    }

    pub fn gen_thread_safe_tmp_bytes(module: &Module, log_base2k: usize, log_q: usize) -> usize {
        encrypt_rlwe_sk_tmp_bytes(module, log_base2k, log_q)
    }
}

pub struct SwitchingKey(GadgetCiphertext);

impl SwitchingKey {
    pub fn new(
        module: &Module,
        log_base2k: usize,
        rows: usize,
        log_q: usize,
        log_scale: usize,
    ) -> SwitchingKey {
        SwitchingKey(GadgetCiphertext::new(
            module, log_base2k, rows, log_q, log_scale,
        ))
    }

    pub fn gen_thread_safe(
        &mut self,
        params: &mut Parameters,
        sk_in: &SvpPPol,
        sk_out: &SvpPPol,
        xa_source: &mut Source,
        xe_source: &mut Source,
        tmp_bytes: &mut [u8],
    ) {
    }
}
