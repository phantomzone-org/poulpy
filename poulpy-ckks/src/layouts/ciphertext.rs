//! CKKS ciphertext layout.
//!
//! A [`CKKSCiphertext`] pairs a rank-1 [`GLWE`] ciphertext with scaling and
//! precision metadata.

use poulpy_core::{
    GLWECopy,
    layouts::{Base2K, Degree, GLWE, GLWEInfos, LWEInfos, Rank, TorusPrecision},
};
use poulpy_hal::{
    api::{VecZnxLsh, VecZnxLshInplace},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch},
};

use crate::layouts::{Metadata, PrecisionInfos};
use anyhow::Result;

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
    pub prec: Metadata,
}

impl<D: Data> LWEInfos for CKKSCiphertext<D> {
    fn base2k(&self) -> Base2K {
        self.inner.base2k()
    }

    fn max_k(&self) -> TorusPrecision {
        self.inner.max_k()
    }

    fn n(&self) -> Degree {
        self.inner.n()
    }

    fn size(&self) -> usize {
        self.inner.size()
    }
}

impl<D: Data> GLWEInfos for CKKSCiphertext<D> {
    fn rank(&self) -> Rank {
        self.inner.rank()
    }
}

impl<D: Data> PrecisionInfos for CKKSCiphertext<D> {
    fn log_decimal(&self) -> usize {
        self.prec.log_decimal
    }

    fn log_hom_rem(&self) -> usize {
        self.prec.log_hom_rem
    }

    fn set_log_decimal(&mut self, log_decimal: usize) -> Result<()> {
        anyhow::ensure!(self.max_k().as_usize() - self.log_hom_rem() >= log_decimal);
        self.prec.log_decimal = log_decimal;
        Ok(())
    }

    fn set_log_hom_rem(&mut self, log_integer: usize) -> Result<()> {
        anyhow::ensure!(self.max_k().as_usize() - self.log_decimal() >= log_integer);
        self.prec.log_hom_rem = log_integer;
        Ok(())
    }
}
impl CKKSCiphertext<Vec<u8>> {
    pub fn alloc_from_infos<A>(infos: &A) -> Result<Self>
    where
        A: GLWEInfos,
    {
        Ok(Self {
            inner: GLWE::alloc_from_infos(infos),
            prec: Metadata::default(),
        })
    }
}

impl CKKSCiphertext<Vec<u8>> {
    /// Reallocates the owned backing buffer so capacity matches `size` limb count.
    /// Fails if dropping limbs would reduce the gap between log_integer and max_k
    /// below log_decimal.
    pub fn reallocate_limbs(&mut self, size: usize) -> Result<()> {
        anyhow::ensure!(self.max_k().as_usize() - self.log_decimal() >= size * self.base2k().as_usize());
        self.inner.reallocate_limbs(size);
        Ok(())
    }

    /// Reallocates the owned backing buffer such that [Self::max_k()] >= [Self::log_decimal()] + [Self::log_integer()].
    pub fn compact_limbs(&mut self) -> Result<()> {
        let size = (self.max_k().as_usize() - self.log_decimal() + self.log_hom_rem()).div_ceil(self.base2k().as_usize());
        self.reallocate_limbs(size)
    }
}

impl<D: Data> CKKSCiphertext<D> {
    /// Returns the number of bits that a and b need to be commonly shifted to match the receiver [TorusPrecision].
    /// If the receiver has more [TorusPrecision] than either a or b, returns 0.
    pub(crate) fn offset_binary(&self, a: &CKKSCiphertext<impl Data>, b: &CKKSCiphertext<impl Data>) -> usize {
        a.inner
            .max_k()
            .min(b.inner.max_k())
            .as_usize()
            .saturating_sub(self.inner.max_k().as_usize())
    }

    pub(crate) fn offset_unary(&self, a: &CKKSCiphertext<impl Data>) -> usize {
        a.inner.max_k().as_usize().saturating_sub(self.inner.max_k().as_usize())
    }
}

impl<D: DataMut> CKKSCiphertext<D> {
    pub fn rescale_inplace<BE: Backend>(&mut self, module: &Module<BE>, k: usize, scratch: &mut Scratch<BE>) -> Result<()>
    where
        Module<BE>: VecZnxLshInplace<BE>,
    {
        if k == 0 {
            return Ok(());
        }

        anyhow::ensure!(self.log_hom_rem() >= k);

        let base2k = self.inner.base2k().as_usize();
        let cols = self.inner.rank().as_usize() + 1;
        for col_i in 0..cols {
            module.vec_znx_lsh_inplace(base2k, k, self.inner.data_mut(), col_i, scratch);
        }

        self.prec.log_hom_rem -= k;

        Ok(())
    }

    pub fn rescale<BE: Backend>(
        &mut self,
        module: &Module<BE>,
        k: usize,
        other: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: VecZnxLsh<BE> + GLWECopy,
    {
        anyhow::ensure!(self.log_hom_rem() >= k);

        let base2k = self.inner.base2k().as_usize();
        let cols = self.inner.rank().as_usize() + 1;
        for col_i in 0..cols {
            module.vec_znx_lsh(base2k, k, self.inner.data_mut(), col_i, other.inner.data(), col_i, scratch);
        }
        self.prec = other.prec;
        self.prec.log_hom_rem -= k;

        Ok(())
    }

    pub fn align_inplace<BE: Backend>(&mut self, module: &Module<BE>, other: &mut Self, scratch: &mut Scratch<BE>) -> Result<()>
    where
        Module<BE>: VecZnxLshInplace<BE>,
    {
        if self.log_hom_rem() < other.log_hom_rem() {
            other.rescale_inplace(module, other.log_hom_rem() - self.log_hom_rem(), scratch)
        } else {
            self.rescale_inplace(module, self.log_hom_rem() - other.log_hom_rem(), scratch)
        }
    }
}
