//! Pre-expanded CKKS plaintext for allocation-free leveled operations.
//!
//! A [`CKKSPlaintextPrepared`] stores a [`GLWEPlaintext`] already expanded to
//! match a target ciphertext's torus layout. Preparing a plaintext once lets
//! [`add_prepared_pt`], [`sub_prepared_pt`], and [`mul_prepared_pt`] operate
//! without per-call scratch overhead for plaintext expansion.

use crate::layouts::plaintext::CKKSPlaintext;
use poulpy_core::layouts::{Base2K, Degree, GLWEPlaintext, GLWEPlaintextToMut, GLWEPlaintextToRef, LWEInfos, TorusPrecision};
use poulpy_hal::{
    api::{VecZnxNormalize, VecZnxNormalizeTmpBytes},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch, ZnxInfos},
};

/// Pre-expanded CKKS plaintext matching a target ciphertext geometry.
///
/// A prepared plaintext caches the torus placement of a compact
/// [`CKKSPlaintext`] so that [`add_prepared_pt`], [`sub_prepared_pt`],
/// and [`mul_prepared_pt`] can operate without per-call placement overhead.
///
/// ## Lifecycle
///
/// 1. Allocate with [`CKKSPlaintextPrepared::alloc`] or
///    [`CKKSPlaintextPrepared::alloc_and_prepare`].
/// 2. If allocated empty, call [`prepare`](CKKSPlaintextPrepared::prepare)
///    with the compact plaintext.
/// 3. Pass to prepared-plaintext arithmetic functions.
///
/// [`add_prepared_pt`]: crate::leveled::operations::add::add_prepared_pt
/// [`sub_prepared_pt`]: crate::leveled::operations::sub::sub_prepared_pt
/// [`mul_prepared_pt`]: crate::leveled::operations::mul::mul_prepared_pt
pub struct CKKSPlaintextPrepared<D: Data> {
    pub inner: GLWEPlaintext<D>,
    pub embed_bits: u32,
}

impl CKKSPlaintextPrepared<Vec<u8>> {
    /// Allocates an empty prepared plaintext matching the target ciphertext parameters.
    pub fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision, embed_bits: u32) -> Self {
        Self {
            inner: GLWEPlaintext::alloc(n, base2k, k),
            embed_bits,
        }
    }

    /// Allocates and prepares a plaintext in one step.
    pub fn alloc_and_prepare<BE: Backend>(
        module: &Module<BE>,
        ct_n: Degree,
        ct_base2k: Base2K,
        ct_k: TorusPrecision,
        pt: &CKKSPlaintext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Self
    where
        Module<BE>: VecZnxNormalize<BE>,
    {
        let mut prepared = Self::alloc(ct_n, ct_base2k, ct_k, pt.embed_bits);
        prepared.prepare(module, pt, scratch);
        prepared
    }

    /// Encodes a complex constant and prepares it for allocation-free operations.
    pub fn from_const<BE: Backend>(
        module: &Module<BE>,
        n: Degree,
        base2k: Base2K,
        k: TorusPrecision,
        embed_bits: u32,
        c: (f64, f64),
        scratch: &mut Scratch<BE>,
    ) -> Self
    where
        Module<BE>: VecZnxNormalize<BE>,
    {
        let mut compact = CKKSPlaintext::alloc(n, base2k, embed_bits);
        let pt_k = compact.inner.max_k();
        let delta = (2.0f64).powi(embed_bits as i32);
        compact.inner.encode_coeff_i64((delta * c.0).round() as i64, pt_k, 0);
        compact
            .inner
            .encode_coeff_i64((delta * c.1).round() as i64, pt_k, n.0 as usize / 2);
        let mut prepared = Self::alloc(n, base2k, k, embed_bits);
        prepared.prepare(module, &compact, scratch);
        prepared
    }
}

impl<D: Data> CKKSPlaintextPrepared<D> {
    pub fn embed_bits(&self) -> u32 {
        self.embed_bits
    }
}

impl<D: DataMut> CKKSPlaintextPrepared<D> {
    pub fn prepare_tmp_bytes<BE: Backend>(module: &Module<BE>) -> usize
    where
        Module<BE>: VecZnxNormalizeTmpBytes,
    {
        module.vec_znx_normalize_tmp_bytes()
    }

    /// Fills this prepared plaintext from a compact [`CKKSPlaintext`] using the
    /// same explicit torus placement as encryption.
    pub fn prepare<BE: Backend>(&mut self, module: &Module<BE>, pt: &CKKSPlaintext<impl DataRef>, scratch: &mut Scratch<BE>)
    where
        Module<BE>: VecZnxNormalize<BE>,
    {
        assert_eq!(
            self.inner.base2k.0, pt.inner.base2k.0,
            "prepare: base2k mismatch ({} != {})",
            self.inner.base2k.0, pt.inner.base2k.0
        );
        assert!(
            pt.inner.data.size() <= self.inner.data.size(),
            "plaintext has more limbs than prepared target"
        );

        self.embed_bits = pt.embed_bits;
        let target_k = self.inner.max_k();
        crate::leveled::operations::utils::fill_offset_pt(module, &mut self.inner, target_k, pt, scratch);
    }
}

pub trait CKKSPlaintextPreparedToRef {
    fn to_ref(&self) -> CKKSPlaintextPrepared<&[u8]>;
}

impl<D: DataRef> CKKSPlaintextPreparedToRef for CKKSPlaintextPrepared<D> {
    fn to_ref(&self) -> CKKSPlaintextPrepared<&[u8]> {
        CKKSPlaintextPrepared {
            inner: self.inner.to_ref(),
            embed_bits: self.embed_bits,
        }
    }
}

pub trait CKKSPlaintextPreparedToMut {
    fn to_mut(&mut self) -> CKKSPlaintextPrepared<&mut [u8]>;
}

impl<D: DataMut> CKKSPlaintextPreparedToMut for CKKSPlaintextPrepared<D> {
    fn to_mut(&mut self) -> CKKSPlaintextPrepared<&mut [u8]> {
        CKKSPlaintextPrepared {
            inner: self.inner.to_mut(),
            embed_bits: self.embed_bits,
        }
    }
}
