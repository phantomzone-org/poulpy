use anyhow::Result;
use poulpy_core::{
    GLWEAutomorphism, GLWEShift, ScratchArenaTakeCore,
    layouts::{GGLWEInfos, GLWEAutomorphismKeyPrepared, GLWEInfos},
};
use poulpy_hal::layouts::{Backend, Data, Module, ScratchArena};

use crate::{layouts::CKKSCiphertext, oep::CKKSImpl};

use crate::leveled::{api::CKKSConjugateOps, oep::CKKSConjugateOep};

impl<BE: Backend + CKKSImpl<BE>> CKKSConjugateOps<BE> for Module<BE> {
    fn ckks_conjugate_tmp_bytes<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        C: GLWEInfos,
        K: GGLWEInfos,
        Self: GLWEAutomorphism<BE>,
    {
        CKKSConjugateOep::ckks_conjugate_tmp_bytes(self, ct_infos, key_infos)
    }

    fn ckks_conjugate_into<Dst: Data, Src: Data, K: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        src: &CKKSCiphertext<Src>,
        key: &GLWEAutomorphismKeyPrepared<K, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAutomorphism<BE> + GLWEShift<BE>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        CKKSCiphertext<Src>: poulpy_core::layouts::GLWEToBackendRef<BE>,
        GLWEAutomorphismKeyPrepared<K, BE>:
            poulpy_core::layouts::GGLWEPreparedToBackendRef<BE> + poulpy_core::layouts::GGLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        CKKSConjugateOep::ckks_conjugate_into(self, dst, src, key, scratch)
    }

    fn ckks_conjugate_assign<Dst: Data, K: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        key: &GLWEAutomorphismKeyPrepared<K, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAutomorphism<BE>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        GLWEAutomorphismKeyPrepared<K, BE>:
            poulpy_core::layouts::GGLWEPreparedToBackendRef<BE> + poulpy_core::layouts::GGLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        CKKSConjugateOep::ckks_conjugate_assign(self, dst, key, scratch)
    }
}
