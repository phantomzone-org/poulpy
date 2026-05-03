//! CKKS metadata attached to ciphertext storage.
//!
//! A CKKS ciphertext is represented as [`CKKSCiphertext<D>`], a thin wrapper
//! over `poulpy-core`'s `GLWE<D, CKKS>`.

use std::ops::{Deref, DerefMut};

use anyhow::Result;
use poulpy_core::layouts::{Base2K, Degree, GLWE, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, Rank};
use poulpy_hal::layouts::{Backend, Data, HostBackend, HostDataRef, Module};

use crate::{CKKSInfos, CKKSMeta, SetCKKSInfos, error::CKKSCompositionError, layouts::CKKSModuleAlloc};

/// CKKS ciphertext storage plus semantic precision metadata.
///
/// `inner` contains the raw GLWE torus digits while `meta` describes the
/// semantic decimal scaling and remaining homomorphic capacity of the value.
pub struct CKKSCiphertext<D: Data> {
    /// Raw GLWE ciphertext storage.
    pub(crate) inner: GLWE<D>,
    /// Semantic CKKS metadata associated with `inner`.
    pub(crate) meta: CKKSMeta,
}

impl<D: Data> CKKSCiphertext<D> {
    pub(crate) fn from_inner(inner: GLWE<D>, meta: CKKSMeta) -> Self {
        Self { inner, meta }
    }

    pub fn to_ref<BE: Backend>(&self) -> GLWE<BE::BufRef<'_>>
    where
        GLWE<D>: GLWEToBackendRef<BE>,
    {
        GLWEToBackendRef::to_backend_ref(&self.inner)
    }

    pub fn to_mut<BE: Backend>(&mut self) -> GLWE<BE::BufMut<'_>>
    where
        GLWE<D>: GLWEToBackendMut<BE>,
    {
        GLWEToBackendMut::to_backend_mut(&mut self.inner)
    }

    /// Replaces the semantic metadata after checking that the current storage
    /// can represent it.
    ///
    /// This is intended for callers that build ciphertext buffers manually.
    /// Normal CKKS operations update metadata themselves.
    pub fn set_meta_checked(&mut self, meta: CKKSMeta) -> Result<()> {
        anyhow::ensure!(
            meta.effective_k() <= self.max_k().as_usize(),
            CKKSCompositionError::LimbReallocationShrinksBelowMetadata {
                max_k: self.max_k().as_usize(),
                log_delta: meta.log_delta(),
                base2k: self.base2k().as_usize(),
                requested_limbs: self.size(),
            }
        );
        self.meta = meta;
        Ok(())
    }
}

impl<D: Data> Deref for CKKSCiphertext<D> {
    type Target = GLWE<D>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<D: Data> DerefMut for CKKSCiphertext<D> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<D: Data> LWEInfos for CKKSCiphertext<D> {
    fn base2k(&self) -> Base2K {
        self.inner.base2k()
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

impl<D: Data> CKKSInfos for CKKSCiphertext<D> {
    fn meta(&self) -> CKKSMeta {
        self.meta
    }

    fn log_delta(&self) -> usize {
        self.meta.log_delta()
    }

    fn log_budget(&self) -> usize {
        self.meta.log_budget()
    }
}

impl<D: Data> SetCKKSInfos for CKKSCiphertext<D> {
    fn set_meta(&mut self, meta: CKKSMeta) {
        self.meta = meta;
    }
}

impl<BE: Backend, D: Data> GLWEToBackendRef<BE> for CKKSCiphertext<D>
where
    GLWE<D>: GLWEToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GLWE<BE::BufRef<'_>> {
        GLWEToBackendRef::to_backend_ref(&self.inner)
    }
}

impl<BE: Backend, D: Data> GLWEToBackendMut<BE> for CKKSCiphertext<D>
where
    GLWE<D>: GLWEToBackendMut<BE>,
{
    fn to_backend_mut(&mut self) -> GLWE<BE::BufMut<'_>> {
        GLWEToBackendMut::to_backend_mut(&mut self.inner)
    }
}

/// Maintenance operations for resizing ciphertext limb storage.
pub trait CKKSMaintainOps {
    /// Reallocates the owned backing buffer to exactly `size` limbs.
    ///
    /// Inputs:
    /// - `ct`: ciphertext whose owned limb buffer should be resized
    /// - `size`: requested number of limbs
    ///
    /// Output:
    /// - returns `Ok(())` after resizing `ct`
    ///
    /// Behavior:
    /// - preserves ciphertext metadata
    /// - rejects shrink operations that would make the buffer too small for the
    ///   current semantic precision
    ///
    /// Errors:
    /// - `LimbReallocationShrinksBelowMetadata` if the requested limb count
    ///   cannot represent the current metadata
    fn ckks_reallocate_limbs_checked(&self, ct: &mut CKKSCiphertext<Vec<u8>>, size: usize) -> Result<()>;

    /// Shrinks an owned ciphertext buffer to the minimum limb count that still
    /// preserves its current metadata.
    ///
    /// Inputs:
    /// - `ct`: ciphertext whose limb storage should be compacted
    ///
    /// Output:
    /// - returns `Ok(())` after compacting `ct`
    ///
    /// Errors:
    /// - propagates `ckks_reallocate_limbs_checked` if the computed compact
    ///   size would violate metadata constraints
    fn ckks_compact_limbs(&self, ct: &mut CKKSCiphertext<Vec<u8>>) -> Result<()>;

    /// Returns a newly allocated owned ciphertext holding a compacted copy of
    /// `ct`.
    ///
    /// Inputs:
    /// - `ct`: ciphertext to copy and compact
    ///
    /// Output:
    /// - a fresh owned ciphertext with the same metadata and the minimum limb
    ///   count needed to preserve it
    ///
    /// Errors:
    /// - propagates allocation failures from the underlying GLWE type
    fn ckks_compact_limbs_copy<D>(&self, ct: &CKKSCiphertext<D>) -> Result<CKKSCiphertext<Vec<u8>>>
    where
        D: HostDataRef;
}

#[doc(hidden)]
pub trait CKKSMaintainOpsDefault {
    fn ckks_reallocate_limbs_checked_default(&self, ct: &mut CKKSCiphertext<Vec<u8>>, size: usize) -> Result<()> {
        let base2k = ct.base2k().as_usize();
        let required_limbs = ct.effective_k().div_ceil(base2k);
        anyhow::ensure!(
            size >= required_limbs,
            CKKSCompositionError::LimbReallocationShrinksBelowMetadata {
                max_k: ct.max_k().as_usize(),
                log_delta: ct.log_delta(),
                base2k,
                requested_limbs: size,
            }
        );
        ct.data_mut().reallocate_limbs(size);
        Ok(())
    }

    fn ckks_compact_limbs_default(&self, ct: &mut CKKSCiphertext<Vec<u8>>) -> Result<()> {
        let size = ct.effective_k().div_ceil(ct.base2k().as_usize());
        self.ckks_reallocate_limbs_checked_default(ct, size)?;
        Ok(())
    }
}

impl<BE: Backend> CKKSMaintainOpsDefault for Module<BE> {}

impl<BE: Backend> CKKSMaintainOps for Module<BE>
where
    BE: HostBackend<OwnedBuf = Vec<u8>>,
    Module<BE>: CKKSMaintainOpsDefault + CKKSModuleAlloc<BE>,
{
    fn ckks_reallocate_limbs_checked(&self, ct: &mut CKKSCiphertext<Vec<u8>>, size: usize) -> Result<()> {
        self.ckks_reallocate_limbs_checked_default(ct, size)
    }

    fn ckks_compact_limbs(&self, ct: &mut CKKSCiphertext<Vec<u8>>) -> Result<()> {
        self.ckks_compact_limbs_default(ct)
    }

    fn ckks_compact_limbs_copy<D>(&self, ct: &CKKSCiphertext<D>) -> Result<CKKSCiphertext<Vec<u8>>>
    where
        D: HostDataRef,
    {
        let size = ct.effective_k().div_ceil(ct.base2k().as_usize());
        let mut compact = self.ckks_ciphertext_alloc_from_infos(ct);
        compact.meta = ct.meta();
        self.ckks_reallocate_limbs_checked_default(&mut compact, size)?;
        let dst_len = compact.data().data.len();
        compact.data_mut().data.copy_from_slice(&ct.data().data.as_ref()[..dst_len]);
        Ok(compact)
    }
}

pub(crate) trait CKKSOffset: LWEInfos + CKKSInfos {
    #[allow(dead_code)]
    fn offset_binary<A, B>(&self, a: &A, b: &B) -> usize
    where
        A: LWEInfos + CKKSInfos,
        B: LWEInfos + CKKSInfos,
    {
        crate::ckks_offset_binary(self, a, b)
    }

    fn offset_unary<A>(&self, a: &A) -> usize
    where
        A: LWEInfos + CKKSInfos,
    {
        crate::ckks_offset_unary(self, a)
    }
}

impl<T> CKKSOffset for T where T: LWEInfos + CKKSInfos {}
