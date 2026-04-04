//! CKKS ciphertext layout.
//!
//! A [`CKKSCiphertext`] pairs a rank-1 [`GLWE`] ciphertext with scaling and
//! precision metadata.

use poulpy_core::layouts::{Base2K, Degree, GLWE, GLWELayout, GLWEToMut, GLWEToRef, LWEInfos, Rank, TorusPrecision};
use poulpy_hal::layouts::{Data, DataMut, DataRef};

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
/// **Invariant:** `offset_bits >= torus_scale_bits`.
///
/// Message extraction: `message ≈ phase · 2^{offset_bits} / 2^{torus_scale_bits}`.
pub struct CKKSCiphertext<D: Data> {
    pub inner: GLWE<D>,
    pub offset_bits: u32,
    pub torus_scale_bits: u32,
}

impl CKKSCiphertext<Vec<u8>> {
    pub fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision, torus_scale_bits: u32) -> Self {
        let infos = GLWELayout {
            n,
            base2k,
            k,
            rank: Rank(1),
        };
        Self {
            inner: GLWE::alloc_from_infos(&infos),
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
