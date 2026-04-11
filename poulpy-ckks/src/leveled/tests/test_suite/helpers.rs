//! Test context builders and precision assertion helpers.
//!
//! [`TestContext`] owns the backend module, prepared secret key, optional
//! evaluation keys, and two pairs of test messages.  It provides convenience
//! methods for encrypt, decrypt-and-decode, and scratch allocation.

use std::collections::HashMap;

use super::CKKSTestParams;
use crate::{
    layouts::{
        Metadata,
        ciphertext::CKKSCiphertext,
        plaintext::{CKKSPlaintextConversion, CKKSPlaintextRnx, CKKSPlaintextZnx},
    },
    leveled::encryption::{decrypt, decrypt_tmp_bytes, encrypt_sk, encrypt_sk_tmp_bytes},
};
use poulpy_core::{
    GLWEAutomorphism, GLWEAutomorphismKeyEncryptSk, GLWEDecrypt, GLWEEncryptSk, GLWEShift, GLWETensorKeyEncryptSk, GLWETensoring,
    ScratchTakeCore,
    layouts::{
        Base2K, Degree, GLWEAutomorphismKey, GLWEAutomorphismKeyPrepared, GLWEInfos, GLWESecret, GLWESecretPreparedFactory,
        GLWETensorKey, GLWETensorKeyPrepared, GLWETensorKeyPreparedFactory, LWEInfos, TorusPrecision,
        prepared::GLWESecretPrepared,
    },
};
use poulpy_cpu_ref::{FFT64Ref, fft64::FFT64ModuleHandle};

use poulpy_hal::{
    api::{
        ModuleN, ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxCopy, VecZnxLsh, VecZnxLshInplace, VecZnxNormalize,
        VecZnxNormalizeTmpBytes, VecZnxRshAdd,
    },
    layouts::{Backend, GaloisElement, Module, Scratch, ScratchOwned},
    source::Source,
};

/// Shared test state: module, keys, and two complex test messages.
///
/// Constructed via [`TestContext::new`] (base), [`TestContext::new_with_tsk`]
/// (adds tensor key for multiplication), or [`TestContext::new_with_atk`]
/// (adds automorphism keys for rotation and conjugation).
pub struct TestContext<BE: Backend> {
    pub module: Module<BE>,
    pub params: CKKSTestParams,
    pub sk: GLWESecretPrepared<Vec<u8>, BE>,
    pub tsk: GLWETensorKeyPrepared<Vec<u8>, BE>,
    pub atks: HashMap<i64, GLWEAutomorphismKeyPrepared<Vec<u8>, BE>>,
    pub scratch_size: usize,
    pub re1: Vec<f64>,
    pub im1: Vec<f64>,
    pub re2: Vec<f64>,
    pub im2: Vec<f64>,
}

impl<BE: Backend> TestContext<BE>
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

    pub fn prec(&self) -> Metadata {
        self.params.prec
    }

    /// Creates a base context with a prepared secret key and two test messages.
    pub fn new(params: CKKSTestParams, rotations: &[i64]) -> Self
    where
        Module<BE>: poulpy_hal::api::ModuleNew<BE>
            + GLWEEncryptSk<BE>
            + GLWEDecrypt<BE>
            + GLWESecretPreparedFactory<BE>
            + GLWETensorKeyEncryptSk<BE>
            + GLWETensorKeyPreparedFactory<BE>
            + GLWEAutomorphismKeyEncryptSk<BE>
            + GLWEAutomorphism<BE>
            + GLWEShift<BE>
            + GLWETensoring<BE>,
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
        let mut sk = GLWESecretPrepared::alloc_from_infos(&module, &glwe_infos);
        sk.prepare(&module, &sk_raw);

        let mut scratch = ScratchOwned::<BE>::alloc(
            encrypt_sk_tmp_bytes(&module, &params.glwe_layout())
                .max(decrypt_tmp_bytes(&module, &params.glwe_layout()))
                .max(GLWETensorKeyPrepared::prepare_tmp_bytes(&module, &tsk_infos))
                .max(GLWETensorKey::encrypt_sk_tmp_bytes(&module, &tsk_infos)),
        );

        let mut tsk = GLWETensorKey::alloc_from_infos(&tsk_infos);
        tsk.encrypt_sk(&module, &sk_raw, &tsk_infos, &mut xa, &mut xe, scratch.borrow());
        let mut tsk_prepared = GLWETensorKeyPrepared::alloc_from_infos(&module, &tsk_infos);
        tsk_prepared.prepare(&module, &tsk, scratch.borrow());

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
            atk.encrypt_sk(
                &module,
                galois_element,
                &sk_raw,
                &atk_infos,
                &mut xa,
                &mut xe,
                scratch.borrow(),
            );
            let mut atk_prepared = GLWEAutomorphismKeyPrepared::alloc_from_infos(&module, &atk_infos);
            atk_prepared.prepare(&module, &atk, scratch.borrow());
            atks.insert(index, atk_prepared);
        }

        let ct_infos = params.glwe_layout();
        let scratch_size = encrypt_sk_tmp_bytes(&module, &ct_infos)
            .max(decrypt_tmp_bytes(&module, &ct_infos))
            .max(module.glwe_shift_tmp_bytes())
            .max(CKKSCiphertext::mul_relin_tmp_bytes(&module, &ct_infos, &tsk_infos))
            .max(CKKSCiphertext::automorphism_tmp_bytes(&module, &ct_infos, &atk_infos));

        Self {
            module,
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

    pub fn tsk(&self) -> &GLWETensorKeyPrepared<Vec<u8>, BE> {
        &self.tsk
    }

    pub fn atks(&self) -> &HashMap<i64, GLWEAutomorphismKeyPrepared<Vec<u8>, BE>> {
        &self.atks
    }

    pub fn atk(&self, index: i64) -> &GLWEAutomorphismKeyPrepared<Vec<u8>, BE> {
        self.atks()
            .get(&index)
            .unwrap_or_else(|| panic!("missing automorphism key for index {index}"))
    }

    /// Encodes and encrypts complex slot values into a fresh ciphertext.
    pub fn encrypt(&self, re: &[f64], im: &[f64], scratch: &mut Scratch<BE>) -> CKKSCiphertext<Vec<u8>>
    where
        Module<BE>: ModuleN + GLWEEncryptSk<BE> + VecZnxRshAdd<BE>,
        ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let fft_module = Module::<FFT64Ref>::new(self.params.n as u64);
        let mut pt_rnx = CKKSPlaintextRnx::alloc(self.params.n);

        pt_rnx.encode_reim(fft_module.get_ifft_table(), re, im);

        let mut pt_znx = CKKSPlaintextZnx::alloc(self.degree(), self.base2k(), self.prec());
        pt_rnx.to_znx::<BE>(&mut pt_znx).unwrap();

        let mut ct = CKKSCiphertext::alloc_from_infos(&self.params.glwe_layout()).unwrap();
        let mut xa = Source::new([3u8; 32]);
        let mut xe = Source::new([4u8; 32]);
        encrypt_sk(
            &self.module,
            &mut ct,
            &pt_znx,
            &self.sk,
            &self.params.glwe_layout(),
            &mut xa,
            &mut xe,
            scratch,
        );
        ct
    }

    /// Decrypts and decodes a ciphertext back to complex slot values.
    pub fn decrypt_decode(
        &self,
        ct: &CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> (Vec<f64>, Vec<f64>)
    where
        Module<BE>: ModuleN + GLWEDecrypt<BE> + VecZnxLsh<BE> + VecZnxLshInplace<BE> + VecZnxCopy,
        ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let fft_module = Module::<FFT64Ref>::new(self.params.n as u64);
        let mut pt_znx = CKKSPlaintextZnx::alloc(self.degree(), ct.inner.base2k(), self.params.prec);
        decrypt(&self.module, &mut pt_znx, ct, &self.sk, scratch);

        let mut pt_rnx = CKKSPlaintextRnx::alloc(self.params.n);
        pt_rnx.decode_from_znx::<BE>(&pt_znx).unwrap();

        let m = self.params.n / 2;
        let mut re = vec![0.0; m];
        let mut im = vec![0.0; m];
        pt_rnx.decode_reim(fft_module.get_fft_table(), &mut re, &mut im);

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
    pub fn alloc_ct(&self) -> CKKSCiphertext<Vec<u8>> {
        CKKSCiphertext::alloc_from_infos(&self.params.glwe_layout()).unwrap()
    }

    /// Allocates a ciphertext with one fewer limb than the default (k − base2k).
    pub fn alloc_ct_reduced_k(&self) -> CKKSCiphertext<Vec<u8>> {
        let mut layout = self.params.glwe_layout();
        layout.layout.k = TorusPrecision(layout.layout.k.as_u32() - self.params.base2k as u32);
        CKKSCiphertext::alloc_from_infos(&layout).unwrap()
    }

    /// Encodes (re2, im2) into an RNX plaintext via IFFT.
    pub fn encode_pt_rnx(&self) -> CKKSPlaintextRnx<f64> {
        let fft_module = Module::<FFT64Ref>::new(self.params.n as u64);
        let mut pt_rnx = CKKSPlaintextRnx::<f64>::alloc(self.params.n);
        pt_rnx.encode_reim(fft_module.get_ifft_table(), &self.re2, &self.im2);
        pt_rnx
    }

    /// Encodes (re2, im2) into a ZNX plaintext (IFFT + quantise).
    pub fn encode_pt_znx(&self) -> CKKSPlaintextZnx<Vec<u8>> {
        let pt_rnx = self.encode_pt_rnx();
        let mut pt_znx = CKKSPlaintextZnx::alloc(self.degree(), self.base2k(), self.prec());
        pt_rnx.to_znx::<BE>(&mut pt_znx).unwrap();
        pt_znx
    }

    /// Decrypts `ct`, decodes, and asserts both channels meet `min_bits` of precision.
    pub fn assert_decrypt_precision(
        &self,
        label: &str,
        ct: &CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
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
