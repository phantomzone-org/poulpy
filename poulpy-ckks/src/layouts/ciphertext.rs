//! CKKS ciphertext layout.
//!
//! A [`CKKSCiphertext`] pairs a rank-1 [`GLWE`] ciphertext with scaling and
//! precision metadata.

use poulpy_core::layouts::{
    Base2K, Degree, GLWE, GLWEToMut, GLWEToRef, LWEInfos, Rank, SetGLWEInfos, TorusPrecision,
};
use poulpy_hal::layouts::{Data, DataMut, DataRef, ZnxViewMut};

/// Encrypted CKKS value carrying a GLWE ciphertext and CKKS metadata.
///
/// ## Three-level precision hierarchy
///
/// | Field | Meaning | Changes on |
/// |-------|---------|------------|
/// | `inner.k()` / `size` | Physical precision — how many torus limbs are stored | `drop_torus_precision`, `rescale`, `mul` |
/// | `offset_bits` | Message position — where the message sits in the torus | `drop_torus_precision`, `rescale`, `mul` |
/// | `torus_scale_bits` | Torus scaling factor carried by the ciphertext | `drop_scaling_precision`, `rescale`, `mul` |
///
/// The **prefix property** of the bivariate representation allows
/// torus precision to be reduced without destroying the message: the dominant
/// limbs are preserved, and only the least-significant limbs are dropped.
///
/// ## Invariants
///
/// - `offset_bits >= torus_scale_bits`
/// - `inner.size() == div_ceil(inner.k(), base2k)` — the active limb count is
///   always consistent with `k`. All `(k, size)` updates must go through
///   [`CKKSCiphertext::set_active_k`].
/// - **Storage invariant:** every inactive tail limb `inner.data[col, size..max_size]`
///   is zero across every column. A shrunk ciphertext is therefore
///   operationally identical to a freshly allocated one at the same `k`:
///   downstream operations whose cost depends on `size()` see no stale data,
///   and the storage cost of a leveled operation scales with the reduced
///   `size`, not the original `max_size`. The invariant is re-established by
///   [`CKKSCiphertext::zero_inactive_tail`] after any operation that shrinks
///   `size`.
///
/// Message extraction: `message ≈ phase · 2^{offset_bits} / 2^{torus_scale_bits}`.
pub struct CKKSCiphertext<D: Data> {
    pub inner: GLWE<D>,
    pub offset_bits: u32,
    pub torus_scale_bits: u32,
}

impl CKKSCiphertext<Vec<u8>> {
    pub fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision, torus_scale_bits: u32) -> Self {
        Self {
            inner: GLWE::alloc(n, base2k, k, Rank(1)),
            offset_bits: k.0,
            torus_scale_bits,
        }
    }
}

impl<D: Data> CKKSCiphertext<D> {
    pub fn prefix_bits(&self) -> u32 {
        self.inner.k().0
    }

    pub fn torus_scale_bits(&self) -> u32 {
        self.torus_scale_bits
    }

    pub fn offset_bits(&self) -> u32 {
        self.offset_bits
    }
}

impl<D: DataRef> CKKSCiphertext<D> {
    pub fn assert_valid(&self, label: &str) {
        let prefix_bits = self.prefix_bits();
        let limb_bits = self.inner.base2k().0;
        let expected_size = prefix_bits.div_ceil(limb_bits) as usize;
        let stored_bits = self.inner.size() as u32 * limb_bits;

        assert!(
            self.offset_bits <= prefix_bits,
            "{label}: offset_bits ({}) exceeds prefix_bits ({prefix_bits})",
            self.offset_bits
        );
        assert!(
            self.torus_scale_bits <= self.offset_bits,
            "{label}: torus_scale_bits ({}) exceeds offset_bits ({})",
            self.torus_scale_bits,
            self.offset_bits
        );
        assert!(
            stored_bits >= prefix_bits,
            "{label}: active storage ({stored_bits} bits) does not cover prefix_bits ({prefix_bits})"
        );
        assert_eq!(
            self.inner.size(),
            expected_size,
            "{label}: active limb count ({}) does not match prefix_bits ({prefix_bits}) and base2k ({limb_bits})",
            self.inner.size()
        );

        #[cfg(debug_assertions)]
        {
            // Storage invariant: every inactive tail limb must be zero.
            let data = self.inner.data();
            let active_size = data.size;
            let max_size = data.max_size;
            if active_size < max_size {
                let n = data.n;
                let cols = data.cols;
                let total_i64 = n * cols * max_size;
                let ptr = data.data.as_ref().as_ptr() as *const i64;
                // SAFETY: `VecZnx`'s backing buffer stores at least
                // `n * cols * max_size` i64s laid out limb-major
                // (see `VecZnx` docs). Reading up to `max_size` limbs is
                // therefore always in-bounds regardless of the current
                // active `size`.
                let full: &[i64] = unsafe { std::slice::from_raw_parts(ptr, total_i64) };
                let tail_start = active_size * n * cols;
                for (offset, &v) in full[tail_start..].iter().enumerate() {
                    assert!(
                        v == 0,
                        "{label}: inactive tail limb not zeroed (limb {}, i64 offset {}, value {})",
                        active_size + offset / (n * cols),
                        tail_start + offset,
                        v,
                    );
                }
            }
        }
    }
}

impl<D: DataMut> CKKSCiphertext<D> {
    /// Sets the active torus precision `k` and synchronizes `data.size` to
    /// `div_ceil(k, base2k)`.
    ///
    /// This is the sanctioned way to move the `(k, size)` pair together.
    /// It does not touch `offset_bits` / `torus_scale_bits`, does not
    /// sign-extend the new MSB limb, and does not zero any inactive tail
    /// limbs; callers that shrink `k` must follow up with
    /// [`Self::zero_inactive_tail`] so the storage invariant holds.
    pub(crate) fn set_active_k(&mut self, k: TorusPrecision) {
        let base2k = self.inner.base2k().0;
        let new_size = k.0.div_ceil(base2k) as usize;
        let max_size = self.inner.data_mut().max_size;
        assert!(
            new_size <= max_size,
            "set_active_k: size {new_size} for k={} base2k={base2k} exceeds max_size {max_size}",
            k.0
        );
        self.inner.set_k(k);
        self.inner.data_mut().size = new_size;
    }

    /// Zeros all inactive tail limbs `data[col, size..max_size]` across
    /// every column, re-establishing the storage invariant after a shrink.
    pub(crate) fn zero_inactive_tail(&mut self) {
        let data = self.inner.data_mut();
        if data.size >= data.max_size {
            return;
        }
        let active_size = data.size;
        let tail_start = active_size * data.n * data.cols;
        // Temporarily widen so `raw_mut()` covers the full allocated buffer.
        data.size = data.max_size;
        data.raw_mut()[tail_start..].fill(0);
        data.size = active_size;
    }
}

pub trait CKKSCiphertextToRef {
    fn to_ref(&self) -> CKKSCiphertext<&[u8]>;
}

impl<D: DataRef> CKKSCiphertextToRef for CKKSCiphertext<D> {
    fn to_ref(&self) -> CKKSCiphertext<&[u8]> {
        CKKSCiphertext {
            inner: self.inner.to_ref(),
            offset_bits: self.offset_bits,
            torus_scale_bits: self.torus_scale_bits,
        }
    }
}

pub trait CKKSCiphertextToMut {
    fn to_mut(&mut self) -> CKKSCiphertext<&mut [u8]>;
}

impl<D: DataMut> CKKSCiphertextToMut for CKKSCiphertext<D> {
    fn to_mut(&mut self) -> CKKSCiphertext<&mut [u8]> {
        CKKSCiphertext {
            inner: self.inner.to_mut(),
            offset_bits: self.offset_bits,
            torus_scale_bits: self.torus_scale_bits,
        }
    }
}
