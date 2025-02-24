use crate::encryptor::{encrypt_grlwe_sk, encrypt_grlwe_sk_tmp_bytes};
use crate::keys::{PublicKey, SecretKey, SwitchingKey};
use crate::parameters::Parameters;
use base2k::{Module, SvpPPol};
use sampling::source::Source;

pub struct KeyGenerator {}

impl KeyGenerator {
    pub fn gen_secret_key_thread_safe(
        &self,
        params: &Parameters,
        source: &mut Source,
    ) -> SecretKey {
        let mut sk: SecretKey = SecretKey::new(params.module());
        sk.fill_ternary_hw(params.xs(), source);
        sk
    }

    pub fn gen_public_key_thread_safe(
        &self,
        params: &Parameters,
        sk_ppol: &SvpPPol,
        source: &mut Source,
        tmp_bytes: &mut [u8],
    ) -> PublicKey {
        let mut xa_source: Source = source.branch();
        let mut xe_source: Source = source.branch();
        let mut pk: PublicKey =
            PublicKey::new(params.module(), params.log_base2k(), params.log_qp());
        pk.gen_thread_safe(
            params.module(),
            sk_ppol,
            params.xe(),
            &mut xa_source,
            &mut xe_source,
            tmp_bytes,
        );
        pk
    }
}

pub fn gen_switching_key_tmp_bytes(
    module: &Module,
    log_base2k: usize,
    rows: usize,
    log_q: usize,
) -> usize {
    encrypt_grlwe_sk_tmp_bytes(module, log_base2k, rows, log_q)
}

pub fn gen_switching_key(
    module: &Module,
    swk: &mut SwitchingKey,
    sk_in: &SecretKey,
    sk_out: &SvpPPol,
    source_xa: &mut Source,
    source_xe: &mut Source,
    sigma: f64,
    tmp_bytes: &mut [u8],
) {
    encrypt_grlwe_sk(
        module, &mut swk.0, &sk_in.0, sk_out, source_xa, source_xe, sigma, tmp_bytes,
    );
}
