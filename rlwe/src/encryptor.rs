use crate::ciphertext::{Ciphertext, GadgetCiphertext};
use crate::elem::Elem;
use crate::keys::SecretKey;
use crate::parameters::Parameters;
use crate::plaintext::Plaintext;
use base2k::ffi::znx::znx_zero_i64_ref;
use base2k::sampling::Sampling;
use base2k::{
    Module, Scalar, SvpPPol, SvpPPolOps, VecZnx, VecZnxBig, VecZnxDft, VecZnxOps, VmpPMatOps,
};
use sampling::source::{Source, new_seed};

pub struct EncryptorSk {
    sk: SvpPPol,
    source_xa: Source,
    source_xe: Source,
    initialized: bool,
    tmp_bytes: Vec<u8>,
}

impl EncryptorSk {
    pub fn new(params: &Parameters, sk: Option<&SecretKey>) -> Self {
        let mut sk_svp_ppol: SvpPPol = params.module().svp_new_ppol();
        let mut initialized: bool = false;
        if let Some(sk) = sk {
            sk.prepare(params.module(), &mut sk_svp_ppol);
            initialized = true;
        }
        Self {
            sk: sk_svp_ppol,
            initialized,
            source_xa: Source::new(new_seed()),
            source_xe: Source::new(new_seed()),
            tmp_bytes: vec![0u8; params.encrypt_rlwe_sk_tmp_bytes(params.limbs_qp())],
        }
    }

    pub fn set_sk(&mut self, module: &Module, sk: &SecretKey) {
        sk.prepare(module, &mut self.sk);
        self.initialized = true;
    }

    pub fn seed_source_xa(&mut self, seed: [u8; 32]) {
        self.source_xa = Source::new(seed)
    }

    pub fn seed_source_xe(&mut self, seed: [u8; 32]) {
        self.source_xe = Source::new(seed)
    }

    pub fn encrypt_rlwe_sk(
        &mut self,
        params: &Parameters,
        ct: &mut Ciphertext,
        pt: Option<&Plaintext>,
    ) {
        assert!(
            self.initialized == true,
            "invalid call to [EncryptorSk.encrypt_rlwe_sk]: [EncryptorSk] has not been initialized with a [SecretKey]"
        );
        params.encrypt_rlwe_sk_thread_safe(
            ct,
            pt,
            &self.sk,
            &mut self.source_xa,
            &mut self.source_xe,
            &mut self.tmp_bytes,
        );
    }

    pub fn encrypt_rlwe_sk_thread_safe(
        &self,
        params: &Parameters,
        ct: &mut Ciphertext,
        pt: Option<&Plaintext>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        tmp_bytes: &mut [u8],
    ) {
        assert!(
            self.initialized == true,
            "invalid call to [EncryptorSk.encrypt_rlwe_sk_thread_safe]: [EncryptorSk] has not been initialized with a [SecretKey]"
        );
        params.encrypt_rlwe_sk_thread_safe(ct, pt, &self.sk, source_xa, source_xe, tmp_bytes);
    }
}

impl Parameters {
    pub fn encrypt_rlwe_sk_tmp_bytes(&self, log_q: usize) -> usize {
        encrypt_rlwe_sk_tmp_bytes(self.module(), self.log_base2k(), log_q)
    }

    pub fn encrypt_rlwe_sk_thread_safe(
        &self,
        ct: &mut Ciphertext,
        pt: Option<&Plaintext>,
        sk: &SvpPPol,
        source_xa: &mut Source,
        source_xe: &mut Source,
        tmp_bytes: &mut [u8],
    ) {
        encrypt_rlwe_sk_thread_safe(
            self.module(),
            &mut ct.0,
            pt.map(|pt| &pt.0),
            sk,
            source_xa,
            source_xe,
            self.xe(),
            tmp_bytes,
        )
    }
}

pub fn encrypt_rlwe_sk_tmp_bytes(module: &Module, log_base2k: usize, log_q: usize) -> usize {
    module.bytes_of_vec_znx_dft((log_q + log_base2k - 1) / log_base2k)
        + module.vec_znx_big_normalize_tmp_bytes()
}

pub fn encrypt_rlwe_sk_thread_safe(
    module: &Module,
    ct: &mut Elem,
    pt: Option<&Elem>,
    sk: &SvpPPol,
    source_xa: &mut Source,
    source_xe: &mut Source,
    sigma: f64,
    tmp_bytes: &mut [u8],
) {
    let limbs: usize = ct.limbs();
    let log_base2k: usize = ct.log_base2k();
    let log_q: usize = ct.log_q();

    assert!(
        tmp_bytes.len() >= encrypt_rlwe_sk_tmp_bytes(module, log_base2k, log_q),
        "invalid tmp_bytes: tmp_bytes={} < encrypt_rlwe_sk_tmp_bytes={}",
        tmp_bytes.len(),
        encrypt_rlwe_sk_tmp_bytes(module, log_base2k, log_q)
    );

    let log_q: usize = ct.log_q();
    let log_base2k: usize = ct.log_base2k();
    let c1: &mut VecZnx = ct.at_mut(1);

    // c1 <- Z_{2^prec}[X]/(X^{N}+1)
    c1.fill_uniform(log_base2k, limbs, source_xa);

    let bytes_of_vec_znx_dft: usize = module.bytes_of_vec_znx_dft(limbs);

    // Scratch space for DFT values
    let mut buf_dft: VecZnxDft =
        VecZnxDft::from_bytes(limbs, &mut tmp_bytes[..bytes_of_vec_znx_dft]);

    // Applies buf_dft <- DFT(s) * DFT(c1)
    module.svp_apply_dft(&mut buf_dft, sk, c1, limbs);

    // Alias scratch space
    let mut buf_big: VecZnxBig = buf_dft.as_vec_znx_big();

    // buf_big = s x c1
    module.vec_znx_idft_tmp_a(&mut buf_big, &mut buf_dft, limbs);

    let carry: &mut [u8] = &mut tmp_bytes[bytes_of_vec_znx_dft..];

    // c0 <- -s x c1 + m
    let c0: &mut VecZnx = ct.at_mut(0);

    if let Some(pt) = pt {
        module.vec_znx_big_sub_small_a_inplace(&mut buf_big, pt.at(0));
        module.vec_znx_big_normalize(log_base2k, c0, &buf_big, carry);
    } else {
        module.vec_znx_big_normalize(log_base2k, c0, &buf_big, carry);
        module.vec_znx_negate_inplace(c0);
    }

    // c0 <- -s x c1 + m + e
    c0.add_normal(log_base2k, log_q, source_xe, sigma, (sigma * 6.0).ceil());
}

pub fn encrypt_grlwe_sk_tmp_bytes(
    module: &Module,
    log_base2k: usize,
    rows: usize,
    log_q: usize,
) -> usize {
    let cols = (log_q + log_base2k - 1) / log_base2k;
    Elem::bytes_of(module, log_base2k, log_q, 1)
        + Plaintext::bytes_of(module, log_base2k, log_q)
        + encrypt_rlwe_sk_tmp_bytes(module, log_base2k, log_q)
        + module.vmp_prepare_tmp_bytes(rows, cols)
}

pub fn encrypt_grlwe_sk_thread_safe(
    module: &Module,
    ct: &mut GadgetCiphertext,
    m: &Scalar,
    sk: &SvpPPol,
    source_xa: &mut Source,
    source_xe: &mut Source,
    sigma: f64,
    tmp_bytes: &mut [u8],
) {
    let rows: usize = ct.rows();
    let log_q: usize = ct.log_q();
    let log_base2k: usize = ct.log_base2k();

    let min_tmp_bytes_len = encrypt_grlwe_sk_tmp_bytes(module, log_base2k, rows, log_q);

    assert!(
        tmp_bytes.len() >= min_tmp_bytes_len,
        "invalid tmp_bytes: tmp_bytes.len()={} < encrypt_grlwe_sk_tmp_bytes={}",
        tmp_bytes.len(),
        min_tmp_bytes_len
    );

    let mut ptr: usize = 0;
    let mut tmp_elem: Elem = Elem::from_bytes(module, log_base2k, ct.log_q(), 1, tmp_bytes);
    let bytes_of_elem: usize = Elem::bytes_of(module, log_base2k, log_q, 1);
    ptr += bytes_of_elem;

    let mut tmp_pt: Plaintext =
        Plaintext::from_bytes(module, log_base2k, log_q, &mut tmp_bytes[ptr..]);
    ptr += Plaintext::bytes_of(module, log_base2k, log_q);

    let (tmp_bytes_encrypt_sk, tmp_bytes_vmp_prepare_row) =
        tmp_bytes[ptr..].split_at_mut(encrypt_rlwe_sk_tmp_bytes(module, log_base2k, log_q));

    (0..rows).for_each(|row_i| {
        // Sets the i-th row of the RLWE sample to m (i.e. m * 2^{-log_base2k*i})
        tmp_pt.0.value[0].at_mut(row_i).copy_from_slice(&m.0);

        // Encrypts RLWE(m * 2^{-log_base2k*i})
        encrypt_rlwe_sk_thread_safe(
            module,
            &mut tmp_elem,
            Some(&tmp_pt.0),
            sk,
            source_xa,
            source_xe,
            sigma,
            tmp_bytes_encrypt_sk,
        );

        // Zeroes the ith-row of tmp_pt
        tmp_pt.0.value[0].at_mut(row_i).fill(0);

        // GRLWE[row_i][0] = -as + m * 2^{-i*log_base2k} + e*2^{-log_q}
        module.vmp_prepare_row(
            &mut ct.value[0],
            tmp_elem.at(0),
            row_i,
            tmp_bytes_vmp_prepare_row,
        );

        // GRLWE[row_i][1] = a
        module.vmp_prepare_row(
            &mut ct.value[1],
            tmp_elem.at(1),
            row_i,
            tmp_bytes_vmp_prepare_row,
        );
    })
}
