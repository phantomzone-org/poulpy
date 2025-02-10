use crate::elem::Elem;
use crate::encryptor::{encrypt_rlwe_sk, encrypt_rlwe_sk_tmp_bytes};
use crate::parameters::Parameters;
use base2k::{Module, Sampling, Scalar, SvpPPol, SvpPPolOps, VecZnx};
use sampling::source::Source;

pub struct SecretKey(pub Scalar);

impl SecretKey {
    pub fn new_ternary_prob(module: &Module, limbs: usize, prob: f64, source: &mut Source) -> Self {
        let mut sk: Scalar = Scalar::new(module.n());
        sk.fill_ternary_prob(prob, source);
        SecretKey(sk)
    }
}

pub struct PublicKey(pub Elem);

impl PublicKey {
    pub fn new(
        params: &Parameters,
        sk: &SvpPPol,
        xa_source: &mut Source,
        xe_source: &mut Source,
        tmp_bytes: &mut [u8],
    ) -> Self {
        let mut pk: Elem = Elem::new(params.n(), params.log_base2k(), params.log_qp(), 1);
        encrypt_rlwe_sk(
            params.module(),
            &mut pk,
            None,
            sk,
            xa_source,
            xe_source,
            params.xe(),
            tmp_bytes,
        );
        PublicKey(pk)
    }

    pub fn new_tmp_bytes(params: &Parameters) -> usize {
        encrypt_rlwe_sk_tmp_bytes(params.module(), params.limbs_qp())
    }
}
