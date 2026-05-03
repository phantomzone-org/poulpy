use anyhow::Result;
use poulpy_core::layouts::{
    GGLWEInfos, GGLWEPreparedToBackendRef, GLWEAutomorphismKeyHelper, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef,
    GetGaloisElement, LWEInfos, prepared::GLWEAutomorphismKeyPreparedToBackendRef,
};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{CKKSInfos, SetCKKSInfos, oep::CKKSImpl};

pub trait CKKSRotateOps<BE: Backend + CKKSImpl<BE>> {
    fn ckks_rotate_tmp_bytes<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        C: GLWEInfos + CKKSInfos,
        K: GGLWEInfos;

    fn ckks_rotate_into<Dst, Src, H, K>(
        &self,
        dst: &mut Dst,
        src: &Src,
        k: i64,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + LWEInfos + CKKSInfos;

    fn ckks_rotate_assign<Dst, H, K>(&self, dst: &mut Dst, k: i64, keys: &H, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos;
}
