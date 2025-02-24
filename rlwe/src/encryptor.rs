use crate::ciphertext::Ciphertext;
use crate::elem::{Elem, ElemCommon, ElemVecZnx, VecZnxCommon};
use crate::keys::SecretKey;
use crate::parameters::Parameters;
use crate::plaintext::Plaintext;
use base2k::sampling::Sampling;
use base2k::{
    Module, Scalar, SvpPPol, SvpPPolOps, VecZnx, VecZnxApi, VecZnxBig, VecZnxBigOps, VecZnxBorrow,
    VecZnxDft, VecZnxDftOps, VecZnxOps, VmpPMat, VmpPMatOps,
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
        let mut sk_svp_ppol: SvpPPol = params.module().new_svp_ppol();
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
            tmp_bytes: vec![0u8; params.encrypt_rlwe_sk_tmp_bytes(params.cols_qp())],
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

    pub fn encrypt_rlwe_sk<T>(
        &mut self,
        params: &Parameters,
        ct: &mut Ciphertext<T>,
        pt: Option<&Plaintext<T>>,
    ) where
        T: VecZnxCommon<Owned = T>,
        Elem<T>: ElemCommon<T>,
    {
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

    pub fn encrypt_rlwe_sk_thread_safe<T>(
        &self,
        params: &Parameters,
        ct: &mut Ciphertext<T>,
        pt: Option<&Plaintext<T>>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        tmp_bytes: &mut [u8],
    ) where
        T: VecZnxCommon<Owned = T>,
        Elem<T>: ElemCommon<T>,
    {
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

    pub fn encrypt_rlwe_sk_thread_safe<T>(
        &self,
        ct: &mut Ciphertext<T>,
        pt: Option<&Plaintext<T>>,
        sk: &SvpPPol,
        source_xa: &mut Source,
        source_xe: &mut Source,
        tmp_bytes: &mut [u8],
    ) where
        T: VecZnxCommon<Owned = T>,
        Elem<T>: ElemCommon<T>,
    {
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

pub fn encrypt_rlwe_sk_thread_safe<T>(
    module: &Module,
    ct: &mut Elem<T>,
    pt: Option<&Elem<T>>,
    sk: &SvpPPol,
    source_xa: &mut Source,
    source_xe: &mut Source,
    sigma: f64,
    tmp_bytes: &mut [u8],
) where
    T: VecZnxCommon<Owned = T>,
    Elem<T>: ElemCommon<T>,
{
    let cols: usize = ct.cols();
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
    let c1: &mut T = ct.at_mut(1);

    // c1 <- Z_{2^prec}[X]/(X^{N}+1)
    module.fill_uniform(log_base2k, c1, cols, source_xa);

    let bytes_of_vec_znx_dft: usize = module.bytes_of_vec_znx_dft(cols);

    // Scratch space for DFT values
    let mut buf_dft: VecZnxDft =
        VecZnxDft::from_bytes(cols, &mut tmp_bytes[..bytes_of_vec_znx_dft]);

    // Applies buf_dft <- DFT(s) * DFT(c1)
    module.svp_apply_dft(&mut buf_dft, sk, c1, cols);

    // Alias scratch space
    let mut buf_big: VecZnxBig = buf_dft.as_vec_znx_big();

    // buf_big = s x c1
    module.vec_znx_idft_tmp_a(&mut buf_big, &mut buf_dft, cols);

    let carry: &mut [u8] = &mut tmp_bytes[bytes_of_vec_znx_dft..];

    // c0 <- -s x c1 + m
    let c0: &mut T = ct.at_mut(0);

    if let Some(pt) = pt {
        module.vec_znx_big_sub_small_a_inplace(&mut buf_big, pt.at(0));
        module.vec_znx_big_normalize(log_base2k, c0, &buf_big, carry);
    } else {
        module.vec_znx_big_normalize(log_base2k, c0, &buf_big, carry);
        module.vec_znx_negate_inplace(c0);
    }

    // c0 <- -s x c1 + m + e
    module.add_normal(
        log_base2k,
        c0,
        log_q,
        source_xe,
        sigma,
        (sigma * 6.0).ceil(),
    );
}

impl Parameters {
    pub fn encrypt_grlwe_sk_tmp_bytes(&self, rows: usize, log_q: usize) -> usize {
        encrypt_grlwe_sk_tmp_bytes(self.module(), self.log_base2k(), rows, log_q)
    }
}

pub fn encrypt_grlwe_sk_tmp_bytes(
    module: &Module,
    log_base2k: usize,
    rows: usize,
    log_q: usize,
) -> usize {
    let cols = (log_q + log_base2k - 1) / log_base2k;
    Elem::<VecZnx>::bytes_of(module, log_base2k, log_q, 2)
        + Plaintext::<VecZnx>::bytes_of(module, log_base2k, log_q)
        + encrypt_rlwe_sk_tmp_bytes(module, log_base2k, log_q)
        + module.vmp_prepare_tmp_bytes(rows, cols)
}

pub fn encrypt_grlwe_sk(
    module: &Module,
    ct: &mut Ciphertext<VmpPMat>,
    m: &Scalar,
    sk: &SvpPPol,
    source_xa: &mut Source,
    source_xe: &mut Source,
    sigma: f64,
    tmp_bytes: &mut [u8],
) {
    let rows: usize = ct.rows();
    let log_q: usize = ct.log_q();
    //let cols: usize = (log_q + ct.log_base2k() - 1) / ct.log_base2k();
    let log_base2k: usize = ct.log_base2k();

    let min_tmp_bytes_len = encrypt_grlwe_sk_tmp_bytes(module, log_base2k, rows, log_q);

    assert!(
        tmp_bytes.len() >= min_tmp_bytes_len,
        "invalid tmp_bytes: tmp_bytes.len()={} < encrypt_grlwe_sk_tmp_bytes={}",
        tmp_bytes.len(),
        min_tmp_bytes_len
    );

    let bytes_of_elem: usize = Elem::<VecZnxBorrow>::bytes_of(module, log_base2k, log_q, 2);
    let bytes_of_pt: usize = Plaintext::<VecZnx>::bytes_of(module, log_base2k, log_q);
    let bytes_of_enc_sk: usize = encrypt_rlwe_sk_tmp_bytes(module, log_base2k, log_q);

    let (tmp_bytes_pt, tmp_bytes) = tmp_bytes.split_at_mut(bytes_of_pt);
    let (tmp_bytes_enc_sk, tmp_bytes) = tmp_bytes.split_at_mut(bytes_of_enc_sk);
    let (tmp_bytes_elem, tmp_bytes_vmp_prepare_row) = tmp_bytes.split_at_mut(bytes_of_elem);

    let mut tmp_elem: Elem<VecZnxBorrow> =
        Elem::<VecZnxBorrow>::from_bytes(module, log_base2k, ct.log_q(), 2, tmp_bytes_elem);
    let mut tmp_pt: Plaintext<VecZnxBorrow> =
        Plaintext::<VecZnxBorrow>::from_bytes(module, log_base2k, log_q, tmp_bytes_pt);

    (0..rows).for_each(|row_i| {
        // Sets the i-th row of the RLWE sample to m (i.e. m * 2^{-log_base2k*i})
        tmp_pt.at_mut(0).at_mut(row_i).copy_from_slice(&m.0);

        // Encrypts RLWE(m * 2^{-log_base2k*i})
        encrypt_rlwe_sk_thread_safe(
            module,
            &mut tmp_elem,
            Some(&tmp_pt.0),
            sk,
            source_xa,
            source_xe,
            sigma,
            tmp_bytes_enc_sk,
        );

        //tmp_pt.at(0).print(tmp_pt.cols(), 16);
        //println!();

        // Zeroes the ith-row of tmp_pt
        tmp_pt.at_mut(0).at_mut(row_i).fill(0);

        //println!("row:{}/{}", row_i, rows);
        //tmp_elem.at(0).print(tmp_elem.cols(), tmp_elem.n());
        //tmp_elem.at(1).print(tmp_elem.cols(), tmp_elem.n());
        //println!();
        //println!(">>>");

        // GRLWE[row_i][0||1] = [-as + m * 2^{-i*log_base2k} + e*2^{-log_q} || a]
        module.vmp_prepare_row(
            &mut ct.at_mut(0),
            tmp_elem.at(0).raw(),
            row_i,
            tmp_bytes_vmp_prepare_row,
        );
        module.vmp_prepare_row(
            &mut ct.at_mut(1),
            tmp_elem.at(1).raw(),
            row_i,
            tmp_bytes_vmp_prepare_row,
        );
    });
    //println!("DONE");
}
