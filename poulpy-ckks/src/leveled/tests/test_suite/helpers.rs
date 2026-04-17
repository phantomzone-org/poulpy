//! Test context builders and precision assertion helpers.
//!
//! [`TestContext`] owns the backend module, prepared secret key, optional
//! evaluation keys, and two pairs of test messages.  It provides convenience
//! methods for encrypt, decrypt-and-decode, and scratch allocation.

use std::{collections::HashMap, f64::consts::TAU};

use super::CKKSTestParams;
use crate::{
    CKKS, CKKSCompositionError, CKKSInfos,
    encoding::reim::Encoder,
    layouts::{
        CKKSCiphertext,
        ciphertext::CKKSOffset,
        plaintext::{CKKSPlaintextConversion, CKKSPlaintextRnx, CKKSPlaintextZnx, alloc_pt_znx},
    },
    leveled::{
        encryption::{CKKSDecrypt, CKKSEncrypt},
        operations::{mul::CKKSMulOps, pt_znx::CKKSPlaintextZnxOps},
    },
};
use poulpy_core::{
    EncryptionLayout, GLWEAdd, GLWEAutomorphism, GLWEAutomorphismKeyEncryptSk, GLWECopy, GLWEDecrypt, GLWEMulConst, GLWEMulPlain,
    GLWENegate, GLWENormalize, GLWERotate, GLWEShift, GLWESub, GLWETensorKeyEncryptSk, GLWETensoring, ScratchTakeCore,
    layouts::{
        Base2K, Degree, GLWEAutomorphismKey, GLWEAutomorphismKeyPrepared, GLWEAutomorphismKeyPreparedFactory, GLWESecret,
        GLWESecretPreparedFactory, GLWETensorKey, GLWETensorKeyPrepared, GLWETensorKeyPreparedFactory, LWEInfos,
        prepared::GLWESecretPrepared,
    },
    oep::CoreImpl,
};
use poulpy_cpu_ref::FFT64Ref;

use poulpy_hal::{
    api::{ModuleN, ModuleNew, ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxRshAddInto, VecZnxRshSub},
    layouts::{Backend, DataRef, DeviceBuf, GaloisElement, Module, Scratch, ScratchOwned, ZnxView},
    oep::HalImpl,
    source::Source,
};

pub trait TestBackend: Backend + CoreImpl<Self> + HalImpl<Self> {}

impl<T> TestBackend for T where T: Backend + CoreImpl<T> + HalImpl<T> {}

pub trait TestContextBackend: TestBackend
where
    Module<Self>: ModuleNew<Self>
        + ModuleN
        + GLWESecretPreparedFactory<Self>
        + GLWETensorKeyEncryptSk<Self>
        + GLWETensorKeyPreparedFactory<Self>
        + GLWEAutomorphismKeyEncryptSk<Self>
        + GLWEAutomorphism<Self>
        + GLWETensoring<Self>
        + CKKSEncrypt<Self>
        + CKKSDecrypt<Self>,
    ScratchOwned<Self>: ScratchOwnedAlloc<Self> + ScratchOwnedBorrow<Self>,
    Scratch<Self>: ScratchTakeCore<Self>,
{
}

impl<T> TestContextBackend for T
where
    T: TestBackend,
    Module<T>: ModuleNew<T>
        + ModuleN
        + GLWESecretPreparedFactory<T>
        + GLWETensorKeyEncryptSk<T>
        + GLWETensorKeyPreparedFactory<T>
        + GLWEAutomorphismKeyEncryptSk<T>
        + GLWEAutomorphism<T>
        + GLWETensoring<T>
        + CKKSEncrypt<T>
        + CKKSDecrypt<T>,
    ScratchOwned<T>: ScratchOwnedAlloc<T> + ScratchOwnedBorrow<T>,
    Scratch<T>: ScratchTakeCore<T>,
{
}

pub trait TestCiphertextBackend: TestBackend
where
    Module<Self>: CKKSEncrypt<Self> + CKKSDecrypt<Self>,
    ScratchOwned<Self>: ScratchOwnedAlloc<Self> + ScratchOwnedBorrow<Self>,
    Scratch<Self>: ScratchTakeCore<Self>,
{
}

impl<T> TestCiphertextBackend for T
where
    T: TestBackend,
    Module<T>: CKKSEncrypt<T> + CKKSDecrypt<T>,
    ScratchOwned<T>: ScratchOwnedAlloc<T> + ScratchOwnedBorrow<T>,
    Scratch<T>: ScratchTakeCore<T>,
{
}

pub trait TestAddBackend: TestCiphertextBackend
where
    Module<Self>: GLWEAdd + GLWEShift<Self> + VecZnxRshAddInto<Self>,
    Scratch<Self>: ScratchAvailable,
{
}

impl<T> TestAddBackend for T
where
    T: TestCiphertextBackend,
    Module<T>: GLWEAdd + GLWEShift<T> + VecZnxRshAddInto<T>,
    Scratch<T>: ScratchAvailable,
{
}

pub trait TestSubBackend: TestCiphertextBackend
where
    Module<Self>: GLWESub + GLWEShift<Self> + VecZnxRshSub<Self>,
    Scratch<Self>: ScratchAvailable,
{
}

impl<T> TestSubBackend for T
where
    T: TestCiphertextBackend,
    Module<T>: GLWESub + GLWEShift<T> + VecZnxRshSub<T>,
    Scratch<T>: ScratchAvailable,
{
}

pub trait TestNegBackend: TestCiphertextBackend
where
    Module<Self>: GLWENegate + GLWEShift<Self>,
    Scratch<Self>: ScratchAvailable,
{
}

impl<T> TestNegBackend for T
where
    T: TestCiphertextBackend,
    Module<T>: GLWENegate + GLWEShift<T>,
    Scratch<T>: ScratchAvailable,
{
}

pub trait TestRotateBackend: TestCiphertextBackend
where
    Module<Self>: GLWEAutomorphism<Self>,
    Scratch<Self>: ScratchAvailable,
{
}

impl<T> TestRotateBackend for T
where
    T: TestCiphertextBackend,
    Module<T>: GLWEAutomorphism<T>,
    Scratch<T>: ScratchAvailable,
{
}

pub trait TestMulBackend: TestCiphertextBackend
where
    Module<Self>: GLWEMulConst<Self> + GLWERotate<Self> + GLWETensoring<Self> + GLWEShift<Self>,
    Scratch<Self>: ScratchAvailable,
{
}

impl<T> TestMulBackend for T
where
    T: TestCiphertextBackend,
    Module<T>: GLWEMulConst<T> + GLWERotate<T> + GLWETensoring<T> + GLWEShift<T>,
    Scratch<T>: ScratchAvailable,
{
}

pub trait TestPow2Backend: TestCiphertextBackend
where
    Module<Self>: GLWEShift<Self> + GLWECopy,
{
}

impl<T> TestPow2Backend for T
where
    T: TestCiphertextBackend,
    Module<T>: GLWEShift<T> + GLWECopy,
{
}

pub trait TestLevelBackend: TestCiphertextBackend
where
    Module<Self>: GLWEShift<Self> + GLWECopy,
{
}

impl<T> TestLevelBackend for T
where
    T: TestCiphertextBackend,
    Module<T>: GLWEShift<T> + GLWECopy,
{
}

pub trait TestCompositionBackend: TestCiphertextBackend
where
    Module<Self>: GLWEAdd + GLWEMulPlain<Self> + GLWENormalize<Self> + GLWEShift<Self> + GLWETensoring<Self>,
{
}

impl<T> TestCompositionBackend for T
where
    T: TestCiphertextBackend,
    Module<T>: GLWEAdd + GLWEMulPlain<T> + GLWENormalize<T> + GLWEShift<T> + GLWETensoring<T>,
{
}

#[derive(Clone, Copy)]
pub enum TestVector {
    First,
    Second,
}

/// Shared test state: module, keys, and two complex test messages.
///
/// Constructed via [`TestContext::new`] (base), [`TestContext::new_with_tsk`]
/// (adds tensor key for multiplication), or [`TestContext::new_with_atk`]
/// (adds automorphism keys for rotation and conjugation).
pub struct TestContext<BE: TestBackend> {
    pub module: Module<BE>,
    pub encoder: Encoder<FFT64Ref>,
    pub params: CKKSTestParams,
    pub sk: GLWESecretPrepared<DeviceBuf<BE>, BE>,
    pub tsk: GLWETensorKeyPrepared<DeviceBuf<BE>, BE>,
    pub atks: HashMap<i64, GLWEAutomorphismKeyPrepared<DeviceBuf<BE>, BE>>,
    pub scratch_size: usize,
    pub re1: Vec<f64>,
    pub im1: Vec<f64>,
    pub re2: Vec<f64>,
    pub im2: Vec<f64>,
}

impl<BE: TestBackend> TestContext<BE> {
    pub fn degree(&self) -> Degree {
        self.params.n.into()
    }

    pub fn base2k(&self) -> Base2K {
        self.params.base2k.into()
    }

    pub fn meta(&self) -> CKKS {
        self.params.prec
    }

    pub fn max_k(&self) -> usize {
        self.params.k
    }

    pub fn precision_at(&self, log_decimal: usize) -> CKKS {
        CKKS {
            log_decimal,
            log_hom_rem: 0,
        }
    }
}

impl<BE: TestContextBackend> TestContext<BE> {
    /// Creates a base context with a prepared secret key and two test messages.
    pub fn new(params: CKKSTestParams, rotations: &[i64]) -> Self {
        let module = Module::<BE>::new(params.n as u64);
        let m = module.n() / 2;
        let glwe_infos = params.glwe_layout();
        let tsk_infos = params.tsk_layout();
        let atk_infos = params.atk_layout();

        let mut xa = Source::new([1u8; 32]);
        let mut xe = Source::new([2u8; 32]);

        let mut source_xs = Source::new([0u8; 32]);
        let mut sk_raw = GLWESecret::alloc_from_infos(&glwe_infos);
        sk_raw.fill_ternary_hw(params.hw, &mut source_xs);
        let mut sk = module.glwe_secret_prepared_alloc_from_infos(&glwe_infos);
        module.glwe_secret_prepare(&mut sk, &sk_raw);

        let mut scratch = ScratchOwned::<BE>::alloc(
            module
                .ckks_encrypt_sk_tmp_bytes(&params.glwe_layout())
                .max(module.ckks_decrypt_tmp_bytes(&params.glwe_layout()))
                .max(module.prepare_tensor_key_tmp_bytes(&tsk_infos))
                .max(module.glwe_tensor_key_encrypt_sk_tmp_bytes(&tsk_infos)),
        );

        let mut tsk = GLWETensorKey::alloc_from_infos(&tsk_infos);
        module.glwe_tensor_key_encrypt_sk(&mut tsk, &sk_raw, &tsk_infos, &mut xa, &mut xe, scratch.borrow());
        let mut tsk_prepared = module.alloc_tensor_key_prepared_from_infos(&tsk_infos);
        module.prepare_tensor_key(&mut tsk_prepared, &tsk, scratch.borrow());

        // Store keys by the public index used by operations/tests:
        // rotation shift `k` for slot rotations, and `-1` for conjugation.
        let mut automorphism_indices: Vec<i64> = rotations.to_vec();
        automorphism_indices.push(-1);
        automorphism_indices.sort();
        automorphism_indices.dedup();

        let mut atks = HashMap::new();
        for &index in &automorphism_indices {
            let mut atk = GLWEAutomorphismKey::alloc_from_infos(&atk_infos);
            let galois_element = if index == -1 { -1 } else { module.galois_element(index) };
            module.glwe_automorphism_key_encrypt_sk(
                &mut atk,
                galois_element,
                &sk_raw,
                &atk_infos,
                &mut xa,
                &mut xe,
                scratch.borrow(),
            );
            let mut atk_prepared = module.glwe_automorphism_key_prepared_alloc_from_infos(&atk_infos);
            module.glwe_automorphism_key_prepare(&mut atk_prepared, &atk, scratch.borrow());
            atks.insert(index, atk_prepared);
        }

        let ct_infos = params.glwe_layout();
        let scratch_size = module
            .ckks_encrypt_sk_tmp_bytes(&params.glwe_layout())
            .max(module.ckks_decrypt_tmp_bytes(&params.glwe_layout()))
            .max(module.glwe_shift_tmp_bytes())
            .max(module.ckks_mul_tmp_bytes(&ct_infos, &tsk_infos))
            .max(module.ckks_mul_const_tmp_bytes(&ct_infos, &ct_infos, &params.prec))
            .max(module.ckks_square_tmp_bytes(&ct_infos, &tsk_infos))
            .max(module.glwe_automorphism_tmp_bytes(&ct_infos, &ct_infos, &atk_infos));

        Self {
            module,
            encoder: Encoder::<FFT64Ref>::new(m).unwrap(),
            params,
            sk,
            tsk: tsk_prepared,
            atks,
            scratch_size,
            re1: (0..m).map(|i| (TAU * (i as f64 + 0.25) / m as f64).cos()).collect(),
            im1: (0..m).map(|i| (TAU * (i as f64 + 0.25) / m as f64).sin()).collect(),
            re2: (0..m)
                .map(|i| (TAU * (5.0 * i as f64 + 3.0) / (2.0 * m as f64)).cos())
                .collect(),
            im2: (0..m)
                .map(|i| (TAU * (5.0 * i as f64 + 3.0) / (2.0 * m as f64)).sin())
                .collect(),
        }
    }
}

impl<BE: TestBackend> TestContext<BE> {
    pub fn test_vector(&self, which: TestVector) -> (&[f64], &[f64]) {
        match which {
            TestVector::First => (&self.re1, &self.im1),
            TestVector::Second => (&self.re2, &self.im2),
        }
    }

    pub fn tsk(&self) -> &GLWETensorKeyPrepared<DeviceBuf<BE>, BE> {
        &self.tsk
    }

    pub fn atks(&self) -> &HashMap<i64, GLWEAutomorphismKeyPrepared<DeviceBuf<BE>, BE>> {
        &self.atks
    }

    pub fn atk(&self, index: i64) -> &GLWEAutomorphismKeyPrepared<DeviceBuf<BE>, BE> {
        self.atks()
            .get(&index)
            .unwrap_or_else(|| panic!("missing automorphism key for index {index}"))
    }

    /// Encodes and encrypts complex slot values into a fresh ciphertext.
    pub fn encrypt(&self, k: usize, re: &[f64], im: &[f64], scratch: &mut Scratch<BE>) -> CKKSCiphertext<Vec<u8>>
    where
        Module<BE>: CKKSEncrypt<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        self.encrypt_with_prec(k, re, im, self.meta(), scratch)
    }

    pub fn encrypt_with_prec(
        &self,
        k: usize,
        re: &[f64],
        im: &[f64],
        prec: CKKS,
        scratch: &mut Scratch<BE>,
    ) -> CKKSCiphertext<Vec<u8>>
    where
        Module<BE>: CKKSEncrypt<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let mut pt_rnx = CKKSPlaintextRnx::alloc(self.params.n).unwrap();

        self.encoder.encode_reim(&mut pt_rnx, re, im).unwrap();

        let mut pt_znx = alloc_pt_znx(self.degree(), self.base2k(), prec);
        pt_rnx.to_znx::<BE>(&mut pt_znx).unwrap();

        let mut ct = self.alloc_ct(k);
        let mut xa = Source::new([3u8; 32]);
        let mut xe = Source::new([4u8; 32]);

        let mut layout = self.params.glwe_layout().layout;
        layout.k = k.into();
        let enc_infos = EncryptionLayout::new_from_default_sigma(layout).unwrap();

        self.module
            .ckks_encrypt_sk(&mut ct, &pt_znx, &self.sk, &enc_infos, &mut xa, &mut xe, scratch)
            .unwrap();
        ct
    }

    /// Decrypts and decodes a ciphertext back to complex slot values.
    pub fn decrypt_decode(&self, ct: &CKKSCiphertext<impl DataRef>, scratch: &mut Scratch<BE>) -> (Vec<f64>, Vec<f64>)
    where
        Module<BE>: CKKSDecrypt<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let mut pt_znx = alloc_pt_znx(self.degree(), ct.base2k(), self.precision_at(ct.log_decimal()));
        let (full_pt, scratch_rest) = scratch.take_glwe_plaintext(ct);
        let mut full_pt = CKKSPlaintextZnx::from_plaintext_with_meta(full_pt, ct.meta());
        self.module.glwe_decrypt(ct, &mut full_pt, &self.sk, scratch_rest);
        //println!("full_pt: {full_pt}");
        let top_limb_msb_mask = 1u64 << (ct.base2k().as_usize() - 1);
        assert!(
            full_pt
                .data()
                .at(0, 0)
                .iter()
                .all(|&x| (x.unsigned_abs() & top_limb_msb_mask) == 0),
            "invalid decryption, plaintext overflow: {full_pt}"
        );
        self.module.ckks_extract_pt_znx(&mut pt_znx, &full_pt, scratch_rest).unwrap();

        let mut pt_rnx = CKKSPlaintextRnx::alloc(self.params.n).unwrap();
        pt_rnx.decode_from_znx::<BE>(&pt_znx).unwrap();

        let m = self.params.n / 2;
        let mut re = vec![0.0; m];
        let mut im = vec![0.0; m];
        self.encoder.decode_reim(&pt_rnx, &mut re, &mut im).unwrap();

        (re, im)
    }

    /// Allocates enough scratch for encrypt + decrypt.
    pub fn alloc_scratch(&self) -> ScratchOwned<BE>
    where
        ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    {
        ScratchOwned::<BE>::alloc(self.scratch_size)
    }

    /// Returns element-wise (re1+re2, im1+im2).
    pub fn want_add(&self) -> (Vec<f64>, Vec<f64>) {
        self.want_add_from(&self.re1, &self.im1, &self.re2, &self.im2)
    }

    pub fn want_add_from(&self, a_re: &[f64], a_im: &[f64], b_re: &[f64], b_im: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let m = self.params.n / 2;
        let re = (0..m).map(|j| a_re[j] + b_re[j]).collect();
        let im = (0..m).map(|j| a_im[j] + b_im[j]).collect();
        (re, im)
    }

    pub fn want_sub(&self) -> (Vec<f64>, Vec<f64>) {
        self.want_sub_from(&self.re1, &self.im1, &self.re2, &self.im2)
    }

    pub fn want_sub_from(&self, a_re: &[f64], a_im: &[f64], b_re: &[f64], b_im: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let m = self.params.n / 2;
        let re = (0..m).map(|j| a_re[j] - b_re[j]).collect();
        let im = (0..m).map(|j| a_im[j] - b_im[j]).collect();
        (re, im)
    }

    pub fn want_neg(&self) -> (Vec<f64>, Vec<f64>) {
        let m = self.params.n / 2;
        let re = (0..m).map(|j| -self.re1[j]).collect();
        let im = (0..m).map(|j| -self.im1[j]).collect();
        (re, im)
    }

    pub fn want_mul_pow2(&self, bits: usize) -> (Vec<f64>, Vec<f64>) {
        let m = self.params.n / 2;
        let scale = (1u64 << bits) as f64;
        let re = (0..m).map(|j| self.re1[j] * scale).collect();
        let im = (0..m).map(|j| self.im1[j] * scale).collect();
        (re, im)
    }

    pub fn want_div_pow2(&self, bits: usize) -> (Vec<f64>, Vec<f64>) {
        let m = self.params.n / 2;
        let scale = (1u64 << bits) as f64;
        let re = (0..m).map(|j| self.re1[j] / scale).collect();
        let im = (0..m).map(|j| self.im1[j] / scale).collect();
        (re, im)
    }

    pub fn scale_slots(&self, re: &[f64], im: &[f64], bits: isize) -> (Vec<f64>, Vec<f64>) {
        let scale = (2.0f64).powi(bits as i32);
        let re = re.iter().map(|x| x * scale).collect();
        let im = im.iter().map(|x| x * scale).collect();
        (re, im)
    }

    pub fn want_conjugate(&self) -> (Vec<f64>, Vec<f64>) {
        let m = self.params.n / 2;
        let re = (0..m).map(|j| self.re1[j]).collect();
        let im = (0..m).map(|j| -self.im1[j]).collect();
        (re, im)
    }

    pub fn want_rotate(&self, k: i64) -> (Vec<f64>, Vec<f64>) {
        let m = self.params.n / 2;
        let re = (0..m)
            .map(|j| self.re1[((j as i64 + k).rem_euclid(m as i64)) as usize])
            .collect();
        let im = (0..m)
            .map(|j| self.im1[((j as i64 + k).rem_euclid(m as i64)) as usize])
            .collect();
        (re, im)
    }

    pub fn want_square(&self) -> (Vec<f64>, Vec<f64>) {
        self.want_square_from(&self.re1, &self.im1)
    }

    pub fn want_square_from(&self, re_in: &[f64], im_in: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let m = self.params.n / 2;

        let mut re = vec![0.0f64; m];
        let mut im = vec![0.0f64; m];

        for i in 0..m {
            let re1 = re_in[i];
            let im1 = im_in[i];

            re[i] = re1 * re1 - im1 * im1;
            im[i] = 2.0 * re1 * im1;
        }
        (re, im)
    }

    pub fn want_mul(&self) -> (Vec<f64>, Vec<f64>) {
        self.want_mul_from(&self.re1, &self.im1, &self.re2, &self.im2)
    }

    pub fn want_mul_from(&self, a_re: &[f64], a_im: &[f64], b_re: &[f64], b_im: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let m = self.params.n / 2;

        let mut re = vec![0.0f64; m];
        let mut im = vec![0.0f64; m];

        for i in 0..m {
            let re1 = a_re[i];
            let im1 = a_im[i];
            let re2 = b_re[i];
            let im2 = b_im[i];

            re[i] = re1 * re2 - im1 * im2;
            im[i] = re1 * im2 + re2 * im1;
        }
        (re, im)
    }

    pub fn quantized_const(&self, re: f64, im: f64, log_decimal: usize) -> (f64, f64) {
        let scale = (log_decimal as f64).exp2();
        ((re * scale).round() / scale, (im * scale).round() / scale)
    }

    pub fn want_add_const_from(&self, a_re: &[f64], a_im: &[f64], c_re: f64, c_im: f64) -> (Vec<f64>, Vec<f64>) {
        let m = self.params.n / 2;
        let re = (0..m).map(|j| a_re[j] + c_re).collect();
        let im = (0..m).map(|j| a_im[j] + c_im).collect();
        (re, im)
    }

    pub fn want_mul_const_from(&self, a_re: &[f64], a_im: &[f64], c_re: f64, c_im: f64) -> (Vec<f64>, Vec<f64>) {
        let m = self.params.n / 2;
        let mut re = vec![0.0f64; m];
        let mut im = vec![0.0f64; m];

        for i in 0..m {
            re[i] = a_re[i] * c_re - a_im[i] * c_im;
            im[i] = a_re[i] * c_im + a_im[i] * c_re;
        }

        (re, im)
    }

    /// Allocates a ciphertext with one fewer limb than the default (k − base2k).
    pub fn alloc_ct(&self, k: usize) -> CKKSCiphertext<Vec<u8>> {
        let mut layout = self.params.glwe_layout();
        layout.layout.k = k.into();
        CKKSCiphertext::alloc_from_infos(&layout).unwrap()
    }

    /// Encodes (re2, im2) into an RNX plaintext via IFFT.
    pub fn encode_pt_rnx(&self, re: &[f64], im: &[f64]) -> CKKSPlaintextRnx<f64> {
        let mut pt_rnx = CKKSPlaintextRnx::<f64>::alloc(self.params.n).unwrap();
        self.encoder.encode_reim(&mut pt_rnx, re, im).unwrap();
        pt_rnx
    }

    /// Encodes (re2, im2) into a ZNX plaintext (IFFT + quantise).
    pub fn encode_pt_znx(&self, re: &[f64], im: &[f64]) -> CKKSPlaintextZnx<Vec<u8>> {
        self.encode_pt_znx_with_prec(re, im, self.meta())
    }

    pub fn encode_pt_znx_with_prec(&self, re: &[f64], im: &[f64], prec: CKKS) -> CKKSPlaintextZnx<Vec<u8>> {
        let pt_rnx = self.encode_pt_rnx(re, im);
        let mut pt_znx = alloc_pt_znx(self.degree(), self.base2k(), prec);
        pt_rnx.to_znx::<BE>(&mut pt_znx).unwrap();
        pt_znx
    }

    pub fn quantized_slots(&self, re: &[f64], im: &[f64], prec: CKKS) -> (Vec<f64>, Vec<f64>) {
        let pt_znx = self.encode_pt_znx_with_prec(re, im, prec);
        let mut pt_rnx = CKKSPlaintextRnx::alloc(self.params.n).unwrap();
        pt_rnx.decode_from_znx::<BE>(&pt_znx).unwrap();

        let m = self.params.n / 2;
        let mut re_out = vec![0.0; m];
        let mut im_out = vec![0.0; m];
        self.encoder.decode_reim(&pt_rnx, &mut re_out, &mut im_out).unwrap();
        (re_out, im_out)
    }

    pub fn quantized_vector(&self, which: TestVector, log_decimal: usize) -> (Vec<f64>, Vec<f64>) {
        let (re, im) = self.test_vector(which);
        let scale = ((log_decimal as isize) - (self.meta().log_decimal as isize)) as i32;
        let re_scaled: Vec<f64> = re.iter().map(|x| x * (2.0f64).powi(scale)).collect();
        let im_scaled: Vec<f64> = im.iter().map(|x| x * (2.0f64).powi(scale)).collect();
        self.quantized_slots(&re_scaled, &im_scaled, self.precision_at(log_decimal))
    }

    /// Decrypts `ct`, decodes, and asserts both channels meet `min_bits` of precision.
    pub fn assert_decrypt_precision(
        &self,
        label: &str,
        ct: &CKKSCiphertext<impl DataRef>,
        want_re: &[f64],
        want_im: &[f64],
        min_bits: f64,
        scratch: &mut Scratch<BE>,
    ) where
        Module<BE>: CKKSDecrypt<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let (re_out, im_out) = self.decrypt_decode(ct, scratch);
        assert_precision(&format!("{label} re"), &re_out, want_re, min_bits);
        assert_precision(&format!("{label} im"), &im_out, want_im, min_bits);
    }
}

/// Asserts that `got` and `want` agree to at least `min_bits` of precision.
///
/// The precision is measured as `-log2(max_err)`.  The assertion message
/// includes the worst-case slot index and values.
pub fn assert_precision(label: &str, got: &[f64], want: &[f64], min_bits: f64) {
    let mut max_err: f64 = 0.0;
    let mut sample = (0usize, 0.0f64, 0.0f64);
    for (idx, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let err = (g - w).abs();
        if err > max_err {
            max_err = err;
            sample = (idx, *g, *w);
        }
    }
    let prec = -(max_err.log2());
    assert!(
        prec > min_bits,
        "{label}: precision {prec:.1} bits < {min_bits} (max_err={max_err}, sample_idx={}, got={}, want={})",
        sample.0.saturating_sub(1),
        sample.1,
        sample.2
    );
}

pub fn assert_ct_meta(label: &str, ct: &CKKSCiphertext<impl DataRef>, log_decimal: usize, log_hom_rem: usize) {
    assert_eq!(ct.log_decimal(), log_decimal, "{label}: unexpected log_decimal");
    assert_eq!(ct.log_hom_rem(), log_hom_rem, "{label}: unexpected log_hom_rem");
}

pub fn assert_ckks_error(label: &str, err: &anyhow::Error, want: CKKSCompositionError) {
    let got = err.downcast_ref::<CKKSCompositionError>();
    assert_eq!(got, Some(&want), "{label}: unexpected error: {err}");
}

pub fn assert_unary_output_meta(label: &str, ct: &CKKSCiphertext<impl DataRef>, input: &CKKSCiphertext<impl DataRef>) {
    assert_ct_meta(label, ct, input.log_decimal(), input.log_hom_rem() - ct.offset_unary(input));
}

pub fn assert_binary_output_meta(
    label: &str,
    ct: &CKKSCiphertext<impl DataRef>,
    a: &CKKSCiphertext<impl DataRef>,
    b: &CKKSCiphertext<impl DataRef>,
) {
    assert_ct_meta(
        label,
        ct,
        a.log_decimal().max(b.log_decimal()),
        a.log_hom_rem().min(b.log_hom_rem()) - ct.offset_binary(a, b),
    );
}

pub fn assert_mul_ct_output_meta(label: &str, ct: &CKKSCiphertext<impl DataRef>, a: &impl CKKSInfos, b: &impl CKKSInfos) {
    let log_hom_rem = a.log_hom_rem().min(b.log_hom_rem()) - a.log_decimal().min(b.log_decimal());
    let log_decimal = a.log_decimal().max(b.log_decimal());
    let offset = (log_hom_rem + log_decimal).saturating_sub(ct.max_k().as_usize());
    assert_ct_meta(label, ct, log_decimal, log_hom_rem - offset);
}

pub fn assert_mul_pt_output_meta(label: &str, ct: &CKKSCiphertext<impl DataRef>, a: &impl CKKSInfos, b: &impl CKKSInfos) {
    let log_hom_rem = a.log_hom_rem() - a.log_decimal().min(b.log_decimal());
    let log_decimal = a.log_decimal().max(b.log_decimal());
    let offset = (log_hom_rem + log_decimal).saturating_sub(ct.max_k().as_usize());
    assert_ct_meta(label, ct, log_decimal, log_hom_rem - offset);
}
