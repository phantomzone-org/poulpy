//! CKKS ciphertext layout.
//!
//! A [`CKKSCiphertext`] pairs a rank-1 [`GLWE`] ciphertext with scaling and
//! precision metadata.

use poulpy_core::layouts::{Base2K, Degree, GLWE, GLWEInfos, LWEInfos, Rank, TorusPrecision};
use poulpy_hal::{
    api::VecZnxLshInplace,
    layouts::{Backend, Data, DataMut, Module, Scratch},
};

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
    pub log_delta: usize,
}

impl CKKSCiphertext<Vec<u8>> {
    pub fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank) -> Self {
        Self {
            inner: GLWE::alloc(n, base2k, k, rank),
            log_delta: 0,
        }
    }

    pub fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GLWEInfos,
    {
        Self {
            inner: GLWE::alloc_from_infos(infos),
            log_delta: 0,
        }
    }
}

impl<D: Data> CKKSCiphertext<D> {
    pub fn delta(&self) -> usize {
        self.log_delta
    }
}

impl<D: DataMut> CKKSCiphertext<D> {
    pub fn rescale_inplace<BE: Backend>(&mut self, module: &Module<BE>, k: usize, scratch: &mut Scratch<BE>)
    where
        Module<BE>: VecZnxLshInplace<BE>,
    {
        if k == 0 {
            return;
        }

        assert!(self.log_delta >= k);

        let base2k = self.inner.base2k().as_usize();
        let cols = self.inner.rank().as_usize() + 1;
        for col_i in 0..cols {
            module.vec_znx_lsh_inplace(base2k, k, self.inner.data_mut(), col_i, scratch);
        }

        self.log_delta -= k;
    }

    pub fn align_inplace<BE: Backend>(&mut self, module: &Module<BE>, other: &mut Self, scratch: &mut Scratch<BE>)
    where
        Module<BE>: VecZnxLshInplace<BE>,
    {
        if self.log_delta < other.log_delta {
            other.rescale_inplace(module, other.log_delta - self.log_delta, scratch);
        } else {
            self.rescale_inplace(module, self.log_delta - other.log_delta, scratch);
        }
    }
}
