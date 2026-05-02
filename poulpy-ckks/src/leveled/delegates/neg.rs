use anyhow::Result;
use poulpy_core::{
    GLWENegate, GLWEShift, ScratchArenaTakeCore,
    layouts::{GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, Module, ScratchArena},
};

use crate::{CKKSCiphertextToBackendMut, CKKSCiphertextToBackendRef};

use crate::{CKKSInfos, SetCKKSInfos, layouts::CKKSCiphertext, oep::CKKSImpl};

use crate::leveled::{api::CKKSNegOps, oep::CKKSNegOep};

impl<BE: Backend + CKKSImpl<BE>> CKKSNegOps<BE> for Module<BE>
where
    Module<BE>: GLWENegate<BE> + GLWEShift<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_neg_tmp_bytes(&self) -> usize {
        CKKSNegOep::ckks_neg_tmp_bytes(self)
    }

    fn ckks_neg_into<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        Src: CKKSCiphertextToBackendRef<BE> + CKKSInfos,
    {
        let dst_meta = dst.meta();
        let mut dst_ct = CKKSCiphertext::from_inner(GLWEToBackendMut::to_backend_mut(dst), dst_meta);
        let src_ct = CKKSCiphertext::from_inner(GLWEToBackendRef::to_backend_ref(src), src.meta());
        let res = CKKSNegOep::ckks_neg_into(self, &mut dst_ct, &src_ct, scratch);
        let new_meta = dst_ct.meta();
        drop(dst_ct);
        dst.set_meta(new_meta);
        res
    }

    fn ckks_neg_assign<Dst>(&self, dst: &mut Dst) -> Result<()>
    where
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
    {
        let dst_meta = dst.meta();
        let mut dst_ct = CKKSCiphertext::from_inner(GLWEToBackendMut::to_backend_mut(dst), dst_meta);
        let res = CKKSNegOep::ckks_neg_assign(self, &mut dst_ct);
        let new_meta = dst_ct.meta();
        drop(dst_ct);
        dst.set_meta(new_meta);
        res
    }
}
