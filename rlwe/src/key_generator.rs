use crate::encryptor::encrypt_grlwe_sk_thread_safe;
use crate::keys::{PublicKey, SecretKey, SwitchingKey};
use crate::parameters::Parameters;
use base2k::SvpPPol;
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

    pub fn gen_switching_key_thread_safe(
        &self,
        params: &Parameters,
        sk_in: &SecretKey,
        sk_out: &SecretKey,
        rows: usize,
        log_q: usize,
        tmp_bytes: &mut [u8],
    ) -> SwitchingKey {
        let swk: SwitchingKey = SwitchingKey::new(params.module(), params.log_base2k(), rows, log_q, 0);

        let module: &base2k::Module = params.module();

        encrypt_grlwe_sk_thread_safe(module, swk.0, &sk_in.0, sk_out, source_xa, source_xe, sigma, tmp_bytes);
        swk
    }
}
