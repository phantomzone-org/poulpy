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

    pub fn gen_switching_key(
        &self,
        params: &Parameters,
        sk_in: &SecretKey,
        sk_out: &SecretKey,
        rows: usize,
        log_q: usize,
    ) -> SwitchingKey {
        let swk = SwitchingKey::new(params.module(), params.log_base2k(), rows, log_q, 0);

        swk
    }
}
