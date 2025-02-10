use crate::ciphertext::Ciphertext;
use crate::elem::Elem;
use crate::keys::SecretKey;
use crate::parameters::Parameters;
use crate::plaintext::Plaintext;
use base2k::sampling::Sampling;
use base2k::{Module, SvpPPol, SvpPPolOps, VecZnx, VecZnxBig, VecZnxDft};
use sampling::source::Source;

pub struct EncryptorSk {
    pub sk: SvpPPol,
}

impl EncryptorSk {
    pub fn new(params: &Parameters, sk: &SecretKey) -> Self {
        let mut sk_svp_ppol: SvpPPol = params.module().svp_new_ppol();
        params.module().svp_prepare(&mut sk_svp_ppol, &sk.0);
        Self { sk: sk_svp_ppol }
    }

    pub fn encrypt_rlwe_sk(
        &self,
        params: &Parameters,
        ct: &mut Ciphertext,
        pt: Option<&Plaintext>,
        xa_source: &mut Source,
        xe_source: &mut Source,
        tmp_bytes: &mut [u8],
    ) {
        params.encrypt_rlwe_sk(ct, pt, &self.sk, xa_source, xe_source, tmp_bytes);
    }
}

impl Parameters {
    pub fn encrypt_rlwe_sk_tmp_bytes(&self, limbs: usize) -> usize {
        encrypt_rlwe_sk_tmp_bytes(self.module(), limbs)
    }

    pub fn encrypt_rlwe_sk(
        &self,
        ct: &mut Ciphertext,
        pt: Option<&Plaintext>,
        sk: &SvpPPol,
        xa_source: &mut Source,
        xe_source: &mut Source,
        tmp_bytes: &mut [u8],
    ) {
        encrypt_rlwe_sk(
            self.module(),
            &mut ct.0,
            pt.map(|pt| &pt.0),
            sk,
            xa_source,
            xe_source,
            self.xe(),
            tmp_bytes,
        )
    }
}

pub fn encrypt_rlwe_sk_tmp_bytes(module: &Module, limbs: usize) -> usize {
    module.bytes_of_vec_znx_dft(limbs) + module.vec_znx_big_normalize_tmp_bytes()
}

pub fn encrypt_rlwe_sk(
    module: &Module,
    ct: &mut Elem,
    pt: Option<&Elem>,
    sk: &SvpPPol,
    xa_source: &mut Source,
    xe_source: &mut Source,
    sigma: f64,
    tmp_bytes: &mut [u8],
) {
    let limbs: usize = ct.limbs();

    assert!(
        tmp_bytes.len() >= encrypt_rlwe_sk_tmp_bytes(module, limbs),
        "invalid tmp_bytes: tmp_bytes={} < encrypt_rlwe_sk_tmp_bytes={}",
        tmp_bytes.len(),
        encrypt_rlwe_sk_tmp_bytes(module, limbs)
    );

    let log_q: usize = ct.log_q();
    let log_base2k: usize = ct.log_base2k();
    let c1: &mut VecZnx = ct.at_mut(1);

    // c1 <- Z_{2^prec}[X]/(X^{N}+1)
    c1.fill_uniform(limbs, log_base2k, xa_source);

    let bytes_of_vec_znx_dft = module.bytes_of_vec_znx_dft(limbs);

    // Scratch space for DFT values
    let mut buf_dft: VecZnxDft =
        VecZnxDft::from_bytes(limbs, &mut tmp_bytes[..bytes_of_vec_znx_dft]);

    // Applies buf_dft <- s * c1
    module.svp_apply_dft(&mut buf_dft, sk, c1);

    // Alias scratch space
    let mut buf_big: VecZnxBig = buf_dft.as_vec_znx_big();

    if let Some(pt) = pt {
        // buf_big <- m - buf_big
        module.vec_znx_big_sub_small_a_inplace(&mut buf_big, pt.at(0));
    };

    let carry: &mut [u8] = &mut tmp_bytes[bytes_of_vec_znx_dft..];

    // c0 <- normalize(buf_big) + e
    let c0: &mut VecZnx = ct.at_mut(0);
    module.vec_znx_big_normalize(log_base2k, c0, &buf_big, carry);
    c0.add_normal(log_base2k, log_q, xe_source, sigma, (sigma * 6.0).ceil());
}
