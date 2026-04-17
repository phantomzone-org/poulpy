//! CKKS ciphertext complex conjugation.
//!
//! Complex conjugation is the automorphism for Galois element `-1`.
//! The caller must supply the prepared automorphism key for that element.

use anyhow::Result;
use poulpy_core::{
    GLWEAutomorphism, GLWEShift, ScratchTakeCore,
    layouts::{GGLWEInfos, GLWEAutomorphismKeyPrepared, GLWEInfos},
};
use poulpy_hal::layouts::{Backend, DataMut, DataRef, Module, Scratch};

use crate::{
    CKKSInfos, checked_log_hom_rem_sub,
    layouts::{CKKSCiphertext, ciphertext::CKKSOffset},
};

pub trait CKKSConjugateOps<BE: Backend> {
    fn ckks_conjugate_tmp_bytes<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        C: GLWEInfos,
        K: GGLWEInfos,
        Self: GLWEAutomorphism<BE>;

    fn ckks_conjugate(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        src: &CKKSCiphertext<impl DataRef>,
        key: &GLWEAutomorphismKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAutomorphism<BE> + GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn ckks_conjugate_inplace(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        key: &GLWEAutomorphismKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) where
        Self: GLWEAutomorphism<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;
}

#[doc(hidden)]
pub trait CKKSConjugateOpsDefault<BE: Backend> {
    fn ckks_conjugate_tmp_bytes_default<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        C: GLWEInfos,
        K: GGLWEInfos,
        Self: GLWEAutomorphism<BE>,
    {
        self.glwe_automorphism_tmp_bytes(ct_infos, ct_infos, key_infos)
    }

    fn ckks_conjugate_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        src: &CKKSCiphertext<impl DataRef>,
        key: &GLWEAutomorphismKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAutomorphism<BE> + GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let offset = dst.offset_unary(src);
        if offset != 0 {
            self.glwe_lsh(dst, src, offset, scratch);
            self.glwe_automorphism_inplace(dst, key, scratch);
        } else {
            self.glwe_automorphism(dst, src, key, scratch);
        }

        dst.meta = src.meta();
        dst.meta.log_hom_rem = checked_log_hom_rem_sub("conjugate", dst.log_hom_rem(), offset)?;
        Ok(())
    }

    fn ckks_conjugate_inplace_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        key: &GLWEAutomorphismKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) where
        Self: GLWEAutomorphism<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        self.glwe_automorphism_inplace(dst, key, scratch);
    }
}

impl<BE: Backend> CKKSConjugateOpsDefault<BE> for Module<BE> {}

impl<BE: Backend> CKKSConjugateOps<BE> for Module<BE>
where
    Module<BE>: CKKSConjugateOpsDefault<BE>,
{
    fn ckks_conjugate_tmp_bytes<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        C: GLWEInfos,
        K: GGLWEInfos,
        Self: GLWEAutomorphism<BE>,
    {
        self.ckks_conjugate_tmp_bytes_default(ct_infos, key_infos)
    }

    fn ckks_conjugate(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        src: &CKKSCiphertext<impl DataRef>,
        key: &GLWEAutomorphismKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAutomorphism<BE> + GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        self.ckks_conjugate_default(dst, src, key, scratch)
    }

    fn ckks_conjugate_inplace(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        key: &GLWEAutomorphismKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) where
        Self: GLWEAutomorphism<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        self.ckks_conjugate_inplace_default(dst, key, scratch)
    }
}
