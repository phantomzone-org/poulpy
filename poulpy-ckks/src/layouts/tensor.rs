//! CKKS tensor product result (intermediate between tensor and relinearize).

use poulpy_core::layouts::{Base2K, Degree, GLWEInfos, GLWETensor, LWEInfos, Rank, TorusPrecision};
use poulpy_hal::layouts::Data;

/// Intermediate tensor product of two rank-1 CKKS ciphertexts.
///
/// Produced by [`tensor`](crate::leveled::operations::mul::tensor) and consumed
/// by [`relinearize`](crate::leveled::operations::mul::relinearize) to yield a
/// standard [`CKKSCiphertext`](super::ciphertext::CKKSCiphertext).  The
/// `torus_scale_bits` is the sum of the two operands' scaling factors.
pub struct CKKSTensor<D: Data> {
    pub inner: GLWETensor<D>,
    pub offset_bits: u32,
    pub torus_scale_bits: u32,
}

impl CKKSTensor<Vec<u8>> {
    /// Allocates a tensor for the product of two rank-1 CKKS ciphertexts.
    pub fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision, torus_scale_bits: u32) -> Self {
        CKKSTensor {
            inner: GLWETensor::alloc(n, base2k, k, Rank(1)),
            offset_bits: k.0,
            torus_scale_bits,
        }
    }

    /// Allocates a tensor from a GLWE layout, setting `torus_scale_bits`.
    pub fn alloc_from_infos<A: GLWEInfos>(infos: &A, torus_scale_bits: u32) -> Self {
        CKKSTensor {
            inner: GLWETensor::alloc_from_infos(infos),
            offset_bits: infos.k().0,
            torus_scale_bits,
        }
    }
}

impl<D: Data> CKKSTensor<D> {
    pub fn prefix_bits(&self) -> u32 {
        self.inner.k().0
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
