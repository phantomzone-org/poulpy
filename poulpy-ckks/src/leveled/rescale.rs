//! CKKS rescaling and level alignment.
//!
//! Rescale shifts the GLWE payload while consuming homomorphic capacity.
//! Alignment rescales one of two ciphertexts so their `log_hom_rem` matches.

use anyhow::Result;
use poulpy_core::{GLWEShift, ScratchTakeCore};
use poulpy_hal::layouts::{Backend, DataMut, DataRef, Module, Scratch};

use crate::{CKKSInfos, checked_log_hom_rem_sub, layouts::CKKSCiphertext};

/// CKKS rescaling and level-alignment APIs.
///
/// Rescale lowers `log_hom_rem` by shifting the torus representation. Align
/// equalizes the `log_hom_rem` of two ciphertexts by rescaling the one with
/// more remaining capacity.
pub trait CKKSRescaleOps<BE: Backend> {
    /// Returns scratch bytes required by [`Self::ckks_rescale`].
    fn ckks_rescale_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>;

    /// Rescales a ciphertext in place by `k` bits.
    ///
    /// Errors include `InsufficientHomomorphicCapacity` if `k` exceeds the
    /// available `log_hom_rem`.
    fn ckks_rescale_inplace(&self, ct: &mut CKKSCiphertext<impl DataMut>, k: usize, scratch: &mut Scratch<BE>) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;

    /// Computes a rescaled copy of `src` into `dst`.
    ///
    /// Errors include `InsufficientHomomorphicCapacity`.
    fn ckks_rescale(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        k: usize,
        src: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;

    /// Rescales either `a` or `b` in place so both ciphertexts end up with the
    /// same `log_hom_rem`.
    ///
    /// Errors propagate from the underlying rescale operation.
    fn ckks_align_inplace(
        &self,
        a: &mut CKKSCiphertext<impl DataMut>,
        b: &mut CKKSCiphertext<impl DataMut>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;

    /// Returns scratch bytes required by [`Self::ckks_align_inplace`].
    fn ckks_align_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>;
}

#[doc(hidden)]
pub trait CKKSRescaleOpsDefault<BE: Backend> {
    fn ckks_rescale_tmp_bytes_default(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        self.glwe_shift_tmp_bytes()
    }

    fn ckks_align_tmp_bytes_default(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        self.glwe_shift_tmp_bytes()
    }

    fn ckks_rescale_inplace_default(
        &self,
        ct: &mut CKKSCiphertext<impl DataMut>,
        k: usize,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let log_hom_rem = checked_log_hom_rem_sub("rescale_inplace", ct.log_hom_rem(), k)?;
        self.glwe_lsh_inplace(ct, k, scratch);
        ct.meta.log_hom_rem = log_hom_rem;
        Ok(())
    }

    fn ckks_rescale_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        k: usize,
        src: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let log_hom_rem = checked_log_hom_rem_sub("rescale", src.log_hom_rem(), k)?;
        self.glwe_lsh(dst, src, k, scratch);
        dst.meta = src.meta();
        dst.meta.log_hom_rem = log_hom_rem;
        Ok(())
    }

    fn ckks_align_inplace_default(
        &self,
        a: &mut CKKSCiphertext<impl DataMut>,
        b: &mut CKKSCiphertext<impl DataMut>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        if a.log_hom_rem() < b.log_hom_rem() {
            self.ckks_rescale_inplace_default(b, b.log_hom_rem() - a.log_hom_rem(), scratch)
        } else {
            self.ckks_rescale_inplace_default(a, a.log_hom_rem() - b.log_hom_rem(), scratch)
        }
    }
}

impl<BE: Backend> CKKSRescaleOpsDefault<BE> for Module<BE> {}

impl<BE: Backend> CKKSRescaleOps<BE> for Module<BE>
where
    Module<BE>: CKKSRescaleOpsDefault<BE>,
{
    fn ckks_rescale_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        self.ckks_rescale_tmp_bytes_default()
    }

    fn ckks_rescale_inplace(&self, ct: &mut CKKSCiphertext<impl DataMut>, k: usize, scratch: &mut Scratch<BE>) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        self.ckks_rescale_inplace_default(ct, k, scratch)
    }

    fn ckks_rescale(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        k: usize,
        src: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        self.ckks_rescale_default(dst, k, src, scratch)
    }

    fn ckks_align_inplace(
        &self,
        a: &mut CKKSCiphertext<impl DataMut>,
        b: &mut CKKSCiphertext<impl DataMut>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        self.ckks_align_inplace_default(a, b, scratch)
    }

    fn ckks_align_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        self.ckks_align_tmp_bytes_default()
    }
}
