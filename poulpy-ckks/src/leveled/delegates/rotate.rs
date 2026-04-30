use anyhow::Result;
use poulpy_core::{
    GLWEAutomorphism, GLWEShift, ScratchArenaTakeCore,
    layouts::{GGLWEInfos, GGLWEPreparedToBackendRef, GLWEAutomorphismKeyHelper, GLWEInfos, GetGaloisElement},
};
use poulpy_hal::layouts::{Backend, Data, Module, ScratchArena};

use crate::{layouts::CKKSCiphertext, oep::CKKSImpl};

use crate::leveled::{api::CKKSRotateOps, oep::CKKSRotateOep};

impl<BE: Backend + CKKSImpl<BE>> CKKSRotateOps<BE> for Module<BE> {
    fn ckks_rotate_tmp_bytes<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        C: GLWEInfos,
        K: GGLWEInfos,
        Self: GLWEAutomorphism<BE>,
    {
        CKKSRotateOep::ckks_rotate_tmp_bytes(self, ct_infos, key_infos)
    }

    fn ckks_rotate_into<Dst: Data, Src: Data, H, K>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        src: &CKKSCiphertext<Src>,
        k: i64,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAutomorphism<BE> + GLWEShift<BE>,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        CKKSCiphertext<Src>: poulpy_core::layouts::GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        CKKSRotateOep::ckks_rotate_into(self, dst, src, k, keys, scratch)
    }

    fn ckks_rotate_assign<Dst: Data, H, K>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        k: i64,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAutomorphism<BE>,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        CKKSRotateOep::ckks_rotate_assign(self, dst, k, keys, scratch)
    }
}
