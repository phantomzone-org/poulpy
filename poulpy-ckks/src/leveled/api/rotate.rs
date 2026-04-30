use anyhow::Result;
use poulpy_core::{
    GLWEAutomorphism, GLWEShift, ScratchArenaTakeCore,
    layouts::{GGLWEInfos, GGLWEPreparedToBackendRef, GLWEAutomorphismKeyHelper, GLWEInfos, GetGaloisElement},
};
use poulpy_hal::layouts::{Backend, Data, ScratchArena};

use crate::{layouts::CKKSCiphertext, oep::CKKSImpl};

pub trait CKKSRotateOps<BE: Backend + CKKSImpl<BE>> {
    fn ckks_rotate_tmp_bytes<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        C: GLWEInfos,
        K: GGLWEInfos,
        Self: GLWEAutomorphism<BE>;

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
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;

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
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;
}
