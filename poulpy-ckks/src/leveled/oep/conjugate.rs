use anyhow::Result;
use poulpy_core::{
    GLWEAutomorphism, GLWEShift, ScratchArenaTakeCore,
    layouts::{GGLWEInfos, GLWEInfos, prepared::GLWEAutomorphismKeyPreparedBackendRef},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSCiphertextMut, CKKSCiphertextRef, oep::CKKSImpl};

pub(crate) trait CKKSConjugateOep<BE: Backend + CKKSImpl<BE>> {
    fn ckks_conjugate_tmp_bytes<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        C: GLWEInfos,
        K: GGLWEInfos,
        Self: GLWEAutomorphism<BE>;

    fn ckks_conjugate_into(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        src: &CKKSCiphertextRef<'_, BE>,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAutomorphism<BE> + GLWEShift<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;

    fn ckks_conjugate_assign(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAutomorphism<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;
}

impl<BE: Backend + CKKSImpl<BE>> CKKSConjugateOep<BE> for Module<BE> {
    fn ckks_conjugate_tmp_bytes<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        C: GLWEInfos,
        K: GGLWEInfos,
        Self: GLWEAutomorphism<BE>,
    {
        BE::ckks_conjugate_tmp_bytes(self, ct_infos, key_infos)
    }

    fn ckks_conjugate_into(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        src: &CKKSCiphertextRef<'_, BE>,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAutomorphism<BE> + GLWEShift<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_conjugate_into(self, dst, src, key, scratch)
    }

    fn ckks_conjugate_assign(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAutomorphism<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_conjugate_assign(self, dst, key, scratch)
    }
}
