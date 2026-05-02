use anyhow::Result;
use poulpy_core::{
    GLWEAutomorphism, GLWEShift, ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GLWEAutomorphismKeyHelper, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement,
        prepared::GLWEAutomorphismKeyPreparedToBackendRef,
    },
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSCiphertextToBackendMut, CKKSCiphertextToBackendRef};

use crate::{CKKSCompositionError, CKKSInfos, SetCKKSInfos, layouts::CKKSCiphertext, oep::CKKSImpl};

use crate::leveled::{api::CKKSRotateOps, oep::CKKSRotateOep};

impl<BE: Backend + CKKSImpl<BE>> CKKSRotateOps<BE> for Module<BE>
where
    Module<BE>: GLWEAutomorphism<BE> + GLWEShift<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_rotate_tmp_bytes<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        C: GLWEInfos + CKKSInfos,
        K: GGLWEInfos,
    {
        CKKSRotateOep::ckks_rotate_tmp_bytes(self, ct_infos, key_infos)
    }

    fn ckks_rotate_into<Dst, Src, H, K>(
        &self,
        dst: &mut Dst,
        src: &Src,
        k: i64,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        Src: CKKSCiphertextToBackendRef<BE> + CKKSInfos,
    {
        let key = keys
            .get_automorphism_key(k)
            .ok_or(CKKSCompositionError::MissingAutomorphismKey {
                op: "rotate",
                rotation: k,
            })?;
        let key_ref = key.to_backend_ref();
        let dst_meta = dst.meta();
        let mut dst_ct = CKKSCiphertext::from_inner(GLWEToBackendMut::to_backend_mut(dst), dst_meta);
        let src_ct = CKKSCiphertext::from_inner(GLWEToBackendRef::to_backend_ref(src), src.meta());
        let res = CKKSRotateOep::ckks_rotate_into(self, &mut dst_ct, &src_ct, &key_ref, scratch);
        let new_meta = dst_ct.meta();
        drop(dst_ct);
        dst.set_meta(new_meta);
        res
    }

    fn ckks_rotate_assign<Dst, H, K>(&self, dst: &mut Dst, k: i64, keys: &H, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
    {
        let key = keys
            .get_automorphism_key(k)
            .ok_or(CKKSCompositionError::MissingAutomorphismKey {
                op: "rotate_assign",
                rotation: k,
            })?;
        let key_ref = key.to_backend_ref();
        let dst_meta = dst.meta();
        let mut dst_ct = CKKSCiphertext::from_inner(GLWEToBackendMut::to_backend_mut(dst), dst_meta);
        let res = CKKSRotateOep::ckks_rotate_assign(self, &mut dst_ct, &key_ref, scratch);
        let new_meta = dst_ct.meta();
        drop(dst_ct);
        dst.set_meta(new_meta);
        res
    }
}
