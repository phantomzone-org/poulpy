//! Test context builders and precision assertion helpers.
//!
//! [`TestContext`] owns the backend module, prepared secret key, optional
//! evaluation keys, and two pairs of test messages.  It provides convenience
//! methods for encrypt, decrypt-and-decode, and scratch allocation.

use std::collections::HashMap;

use super::CKKSTestParams;
use crate::{
    CKKS,
    layouts::plaintext::{CKKSPlaintextConversion, CKKSPlaintextRnx, CKKSPlaintextZnx, Encoder, alloc_znx},
    leveled::{
        encryption::{CKKSDecrypt, CKKSEncrypt},
        operations::mul::CKKSMulOps,
    },
};
use poulpy_core::{
    GLWEAutomorphism, GLWEAutomorphismKeyEncryptSk, GLWEDecrypt, GLWEEncryptSk, GLWEShift, GLWETensorKeyEncryptSk, GLWETensoring,
    ScratchTakeCore,
    layouts::{
        Base2K, Degree, GLWE, GLWEAutomorphismKey, GLWEAutomorphismKeyPrepared, GLWEAutomorphismKeyPreparedFactory, GLWEInfos,
        GLWESecret, GLWESecretPreparedFactory, GLWETensorKey, GLWETensorKeyPrepared, GLWETensorKeyPreparedFactory, LWEInfos,
        TorusPrecision, prepared::GLWESecretPrepared,
    },
    oep::CoreImpl,
};
use poulpy_cpu_ref::FFT64Ref;

use poulpy_hal::{
    api::{
        ModuleN, ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxCopy, VecZnxLsh, VecZnxLshInplace, VecZnxNormalize,
        VecZnxNormalizeTmpBytes, VecZnxRshAddInto,
    },
    layouts::{Backend, DeviceBuf, GaloisElement, Module, Scratch, ScratchOwned},
    oep::HalImpl,
    source::Source,
};

pub trait TestBackend: Backend + CoreImpl<Self> + HalImpl<Self> {}

impl<T> TestBackend for T where T: Backend + CoreImpl<T> + HalImpl<T> {}

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

impl<BE: TestBackend> TestContext<BE>
where
    Module<BE>: VecZnxNormalize<BE> + VecZnxNormalizeTmpBytes,
{
    pub fn degree(&self) -> Degree {
        self.params.n.into()
    }

    pub fn base2k(&self) -> Base2K {
        self.params.base2k.into()
    }

    pub fn rank(&self) -> poulpy_core::layouts::Rank {
        self.params.glwe_layout().rank()
    }

    pub fn meta(&self) -> CKKS {
        self.params.prec
    }

    /// Creates a base context with a prepared secret key and two test messages.
    pub fn new(params: CKKSTestParams, rotations: &[i64]) -> Self
    where
        Module<BE>: poulpy_hal::api::ModuleNew<BE>
            + GLWESecretPreparedFactory<BE>
            + GLWETensorKeyEncryptSk<BE>
            + GLWETensorKeyPreparedFactory<BE>
            + GLWEAutomorphismKeyEncryptSk<BE>
            + GLWEAutomorphism<BE>
            + GLWEShift<BE>
            + GLWETensoring<BE>
            + CKKSEncrypt<BE>
            + CKKSDecrypt<BE>,
        ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
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
            .max(GLWE::<Vec<u8>, CKKS>::mul_tmp_bytes(&module, &ct_infos, &tsk_infos))
            .max(GLWE::<Vec<u8>, CKKS>::square_tmp_bytes(&module, &ct_infos, &tsk_infos))
            .max(module.glwe_automorphism_tmp_bytes(&ct_infos, &ct_infos, &atk_infos));

        Self {
            module,
            encoder: Encoder::<FFT64Ref>::new(m).unwrap(),
            params,
            sk,
            tsk: tsk_prepared,
            atks,
            scratch_size,
            re1: (0..m).map(|i| 16.0 * (i as f64) / (m as f64) - 8.0).collect(),
            im1: (0..m).map(|i| 16.0 - 2.0 * (i as f64) / (m as f64)).collect(),
            re2: (0..m).map(|i| 0.8 * (1.0 - (i as f64) / (m as f64)) - 0.4).collect(),
            im2: (0..m).map(|i| -0.6 * (i as f64) / (m as f64)).collect(),
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
    pub fn encrypt(&self, re: &[f64], im: &[f64], scratch: &mut Scratch<BE>) -> GLWE<Vec<u8>, CKKS>
    where
        Module<BE>: ModuleN + GLWEEncryptSk<BE> + GLWEShift<BE> + VecZnxRshAddInto<BE>,
        ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let mut pt_rnx = CKKSPlaintextRnx::alloc(self.params.n).unwrap();

        self.encoder.encode_reim(&mut pt_rnx, re, im).unwrap();

        let mut pt_znx = alloc_znx(self.degree(), self.base2k(), self.meta());
        pt_rnx.to_znx::<BE>(&mut pt_znx).unwrap();

        let mut ct = CKKS::alloc_from_infos(&self.params.glwe_layout()).unwrap();
        let mut xa = Source::new([3u8; 32]);
        let mut xe = Source::new([4u8; 32]);
        self.module
            .ckks_encrypt_sk(
                &mut ct,
                &pt_znx,
                &self.sk,
                &self.params.glwe_layout(),
                &mut xa,
                &mut xe,
                scratch,
            )
            .unwrap();
        ct
    }

    /// Decrypts and decodes a ciphertext back to complex slot values.
    pub fn decrypt_decode(
        &self,
        ct: &GLWE<impl poulpy_hal::layouts::DataRef, CKKS>,
        scratch: &mut Scratch<BE>,
    ) -> (Vec<f64>, Vec<f64>)
    where
        Module<BE>: ModuleN + GLWEDecrypt<BE> + GLWEShift<BE> + VecZnxLsh<BE> + VecZnxLshInplace<BE> + VecZnxCopy,
        ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let mut pt_znx = alloc_znx(self.degree(), ct.base2k(), self.params.prec);
        self.module.ckks_decrypt(&mut pt_znx, ct, &self.sk, scratch).unwrap();

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
        let m = self.params.n / 2;
        let re = (0..m).map(|j| self.re1[j] + self.re2[j]).collect();
        let im = (0..m).map(|j| self.im1[j] + self.im2[j]).collect();
        (re, im)
    }

    pub fn want_sub(&self) -> (Vec<f64>, Vec<f64>) {
        let m = self.params.n / 2;
        let re = (0..m).map(|j| self.re1[j] - self.re2[j]).collect();
        let im = (0..m).map(|j| self.im1[j] - self.im2[j]).collect();
        (re, im)
    }

    pub fn want_neg(&self) -> (Vec<f64>, Vec<f64>) {
        let m = self.params.n / 2;
        let re = (0..m).map(|j| -self.re1[j]).collect();
        let im = (0..m).map(|j| -self.im1[j]).collect();
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
        let m = self.params.n / 2;

        let mut re = vec![0.0f64; m];
        let mut im = vec![0.0f64; m];

        for i in 0..m {
            let re1 = self.re1[i];
            let im1 = self.im1[i];

            re[i] = re1 * re1 - im1 * im1;
            im[i] = 2.0 * re1 * im1;
        }
        (re, im)
    }

    pub fn want_mul(&self) -> (Vec<f64>, Vec<f64>) {
        let m = self.params.n / 2;

        let mut re = vec![0.0f64; m];
        let mut im = vec![0.0f64; m];

        for i in 0..m {
            let re1 = self.re1[i];
            let im1 = self.im1[i];
            let re2 = self.re2[i];
            let im2 = self.im2[i];

            re[i] = re1 * re2 - im1 * im2;
            im[i] = re1 * im2 + re2 * im1;
        }
        (re, im)
    }

    /// Allocates a fresh full-precision result ciphertext.
    pub fn alloc_ct(&self) -> GLWE<Vec<u8>, CKKS> {
        CKKS::alloc_from_infos(&self.params.glwe_layout()).unwrap()
    }

    /// Allocates a ciphertext with one fewer limb than the default (k − base2k).
    pub fn alloc_ct_reduced_k(&self) -> GLWE<Vec<u8>, CKKS> {
        let mut layout = self.params.glwe_layout();
        layout.layout.k = TorusPrecision(layout.layout.k.as_u32() - self.params.base2k as u32);
        CKKS::alloc_from_infos(&layout).unwrap()
    }

    /// Encodes (re2, im2) into an RNX plaintext via IFFT.
    pub fn encode_pt_rnx(&self) -> CKKSPlaintextRnx<f64> {
        let mut pt_rnx = CKKSPlaintextRnx::<f64>::alloc(self.params.n).unwrap();
        self.encoder.encode_reim(&mut pt_rnx, &self.re2, &self.im2).unwrap();
        pt_rnx
    }

    /// Encodes (re2, im2) into a ZNX plaintext (IFFT + quantise).
    pub fn encode_pt_znx(&self) -> CKKSPlaintextZnx<Vec<u8>> {
        let pt_rnx = self.encode_pt_rnx();
        let mut pt_znx = alloc_znx(self.degree(), self.base2k(), self.meta());
        pt_rnx.to_znx::<BE>(&mut pt_znx).unwrap();
        pt_znx
    }

    /// Decrypts `ct`, decodes, and asserts both channels meet `min_bits` of precision.
    pub fn assert_decrypt_precision(
        &self,
        label: &str,
        ct: &GLWE<impl poulpy_hal::layouts::DataRef, CKKS>,
        want_re: &[f64],
        want_im: &[f64],
        min_bits: f64,
        scratch: &mut Scratch<BE>,
    ) where
        Module<BE>: ModuleN + GLWEDecrypt<BE> + VecZnxLsh<BE> + VecZnxLshInplace<BE> + VecZnxCopy,
        ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
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
