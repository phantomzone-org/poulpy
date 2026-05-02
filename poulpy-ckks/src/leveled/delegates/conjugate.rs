use anyhow::Result;
use poulpy_core::{
    GLWEAutomorphism, GLWEShift, ScratchArenaTakeCore,
    layouts::{GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, prepared::GLWEAutomorphismKeyPreparedToBackendRef},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSCiphertextToBackendMut, CKKSCiphertextToBackendRef};

use crate::{CKKSInfos, SetCKKSInfos, layouts::CKKSCiphertext, oep::CKKSImpl};

use crate::leveled::{api::CKKSConjugateOps, oep::CKKSConjugateOep};

impl<BE: Backend + CKKSImpl<BE>> CKKSConjugateOps<BE> for Module<BE>
where
    Module<BE>: GLWEAutomorphism<BE> + GLWEShift<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_conjugate_tmp_bytes<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        C: GLWEInfos + CKKSInfos,
        K: GGLWEInfos,
    {
        CKKSConjugateOep::ckks_conjugate_tmp_bytes(self, ct_infos, key_infos)
    }

    fn ckks_conjugate_into<Dst, Src, K>(
        &self,
        dst: &mut Dst,
        src: &Src,
        key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        Src: CKKSCiphertextToBackendRef<BE> + CKKSInfos,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEInfos,
    {
        let key_ref = key.to_backend_ref();
        let dst_meta = dst.meta();
        let mut dst_ct = CKKSCiphertext::from_inner(GLWEToBackendMut::to_backend_mut(dst), dst_meta);
        let src_ct = CKKSCiphertext::from_inner(GLWEToBackendRef::to_backend_ref(src), src.meta());
        let res = CKKSConjugateOep::ckks_conjugate_into(self, &mut dst_ct, &src_ct, &key_ref, scratch);
        let new_meta = dst_ct.meta();
        drop(dst_ct);
        dst.set_meta(new_meta);
        res
    }

    fn ckks_conjugate_assign<Dst, K>(&self, dst: &mut Dst, key: &K, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEInfos,
    {
        let key_ref = key.to_backend_ref();
        let dst_meta = dst.meta();
        let mut dst_ct = CKKSCiphertext::from_inner(GLWEToBackendMut::to_backend_mut(dst), dst_meta);
        let res = CKKSConjugateOep::ckks_conjugate_assign(self, &mut dst_ct, &key_ref, scratch);
        let new_meta = dst_ct.meta();
        drop(dst_ct);
        dst.set_meta(new_meta);
        res
    }
}
