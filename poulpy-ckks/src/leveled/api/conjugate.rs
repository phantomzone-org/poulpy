use anyhow::Result;
use poulpy_core::{
    GLWEAutomorphism, GLWEShift, ScratchArenaTakeCore,
    layouts::{GGLWEInfos, GLWEAutomorphismKeyPrepared, GLWEInfos},
};
use poulpy_hal::layouts::{Backend, Data, ScratchArena};

use crate::{layouts::CKKSCiphertext, oep::CKKSImpl};

pub trait CKKSConjugateOps<BE: Backend + CKKSImpl<BE>> {
    fn ckks_conjugate_tmp_bytes<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        C: GLWEInfos,
        K: GGLWEInfos,
        Self: GLWEAutomorphism<BE>;

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
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;

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
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;
}
