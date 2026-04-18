//! CKKS ciphertext slot rotation.
//!
//! Slot rotation is an automorphism whose Galois element is determined by
//! the rotation amount. The public API is expressed in terms of the slot
//! shift `k`, and the provided key store is indexed by that same shift.

use crate::{
    CKKSInfos, checked_log_hom_rem_sub,
    layouts::{CKKSCiphertext, ciphertext::CKKSOffset},
};
use anyhow::Result;
use poulpy_core::{
    GLWEAutomorphism, GLWEShift, ScratchTakeCore,
    layouts::{GGLWEInfos, GGLWEPreparedToRef, GLWEAutomorphismKeyHelper, GLWEInfos, GetGaloisElement},
};
use poulpy_hal::layouts::{Backend, DataMut, DataRef, Module, Scratch};

/// CKKS slot-rotation APIs.
///
/// Rotation uses GLWE automorphisms indexed by the user-visible slot shift
/// `k`.
pub trait CKKSRotateOps<BE: Backend> {
    /// Returns scratch bytes required by [`Self::ckks_rotate`].
    fn ckks_rotate_tmp_bytes<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        C: GLWEInfos,
        K: GGLWEInfos,
        Self: GLWEAutomorphism<BE>;

    /// Rotates the slots of `src` by `k` positions into `dst`.
    ///
    /// Inputs:
    /// - `keys`: automorphism-key helper indexed by slot rotation amount
    ///
    /// Errors:
    /// - backend automorphism/shift failures
    /// - panics if the required rotation key is missing from `keys`
    fn ckks_rotate<H, K>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        src: &CKKSCiphertext<impl DataRef>,
        k: i64,
        keys: &H,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAutomorphism<BE> + GLWEShift<BE>,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        Scratch<BE>: ScratchTakeCore<BE>;

    /// Rotates the slots of `dst` by `k` positions in place.
    ///
    /// Panics if the required rotation key is missing from `keys`.
    fn ckks_rotate_inplace<H, K>(&self, dst: &mut CKKSCiphertext<impl DataMut>, k: i64, keys: &H, scratch: &mut Scratch<BE>)
    where
        Self: GLWEAutomorphism<BE>,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        Scratch<BE>: ScratchTakeCore<BE>;
}

#[doc(hidden)]
pub trait CKKSRotateOpsDefault<BE: Backend> {
    fn ckks_rotate_tmp_bytes_default<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        C: GLWEInfos,
        K: GGLWEInfos,
        Self: GLWEAutomorphism<BE>,
    {
        self.glwe_automorphism_tmp_bytes(ct_infos, ct_infos, key_infos)
    }

    fn ckks_rotate_default<H, K>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        src: &CKKSCiphertext<impl DataRef>,
        k: i64,
        keys: &H,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAutomorphism<BE> + GLWEShift<BE>,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let key = keys
            .get_automorphism_key(k)
            .unwrap_or_else(|| panic!("missing automorphism key for rotation {k}"));

        let offset = dst.offset_unary(src);

        if offset != 0 {
            self.glwe_lsh(dst, src, offset, scratch);
            self.glwe_automorphism_inplace(dst, key, scratch);
        } else {
            self.glwe_automorphism(dst, src, key, scratch);
        }

        dst.meta = src.meta();
        dst.meta.log_hom_rem = checked_log_hom_rem_sub("rotate", dst.log_hom_rem(), offset)?;
        Ok(())
    }

    fn ckks_rotate_inplace_default<H, K>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        k: i64,
        keys: &H,
        scratch: &mut Scratch<BE>,
    ) where
        Self: GLWEAutomorphism<BE>,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let key = keys
            .get_automorphism_key(k)
            .unwrap_or_else(|| panic!("missing automorphism key for rotation {k}"));
        self.glwe_automorphism_inplace(dst, key, scratch);
    }
}

impl<BE: Backend> CKKSRotateOpsDefault<BE> for Module<BE> {}

impl<BE: Backend> CKKSRotateOps<BE> for Module<BE>
where
    Module<BE>: CKKSRotateOpsDefault<BE>,
{
    fn ckks_rotate_tmp_bytes<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        C: GLWEInfos,
        K: GGLWEInfos,
        Self: GLWEAutomorphism<BE>,
    {
        self.ckks_rotate_tmp_bytes_default(ct_infos, key_infos)
    }

    fn ckks_rotate<H, K>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        src: &CKKSCiphertext<impl DataRef>,
        k: i64,
        keys: &H,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAutomorphism<BE> + GLWEShift<BE>,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        self.ckks_rotate_default(dst, src, k, keys, scratch)
    }

    fn ckks_rotate_inplace<H, K>(&self, dst: &mut CKKSCiphertext<impl DataMut>, k: i64, keys: &H, scratch: &mut Scratch<BE>)
    where
        Self: GLWEAutomorphism<BE>,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        self.ckks_rotate_inplace_default(dst, k, keys, scratch)
    }
}
