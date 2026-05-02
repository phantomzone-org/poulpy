use anyhow::Result;
use poulpy_core::{
    GLWEShift, ScratchArenaTakeCore,
    layouts::{GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSCiphertextToBackendMut, CKKSCiphertextToBackendRef};

use crate::{
    CKKSInfos, SetCKKSInfos,
    layouts::CKKSCiphertext,
    leveled::{api::CKKSRescaleOps, oep::CKKSRescaleOep},
    oep::CKKSImpl,
};

impl<BE: Backend + CKKSImpl<BE>> CKKSRescaleOps<BE> for Module<BE>
where
    Module<BE>: CKKSRescaleOep<BE> + GLWEShift<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_rescale_tmp_bytes(&self) -> usize {
        CKKSRescaleOep::ckks_rescale_tmp_bytes(self)
    }

    fn ckks_rescale_assign<Dst>(&self, ct: &mut Dst, k: usize, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
    {
        let ct_meta = ct.meta();
        let mut ct_ckks = CKKSCiphertext::from_inner(GLWEToBackendMut::to_backend_mut(ct), ct_meta);
        let res = CKKSRescaleOep::ckks_rescale_assign(self, &mut ct_ckks, k, scratch);
        let new_meta = ct_ckks.meta();
        drop(ct_ckks);
        ct.set_meta(new_meta);
        res
    }

    fn ckks_rescale_into<Dst, Src>(&self, dst: &mut Dst, k: usize, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        Src: CKKSCiphertextToBackendRef<BE> + CKKSInfos,
    {
        let dst_meta = dst.meta();
        let mut dst_ct = CKKSCiphertext::from_inner(GLWEToBackendMut::to_backend_mut(dst), dst_meta);
        let src_ct = CKKSCiphertext::from_inner(GLWEToBackendRef::to_backend_ref(src), src.meta());
        let res = CKKSRescaleOep::ckks_rescale_into(self, &mut dst_ct, k, &src_ct, scratch);
        let new_meta = dst_ct.meta();
        drop(dst_ct);
        dst.set_meta(new_meta);
        res
    }

    fn ckks_align_assign<A, B>(&self, a: &mut A, b: &mut B, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        A: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        B: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
    {
        let a_meta = a.meta();
        let b_meta = b.meta();
        let mut a_ct = CKKSCiphertext::from_inner(GLWEToBackendMut::to_backend_mut(a), a_meta);
        let mut b_ct = CKKSCiphertext::from_inner(GLWEToBackendMut::to_backend_mut(b), b_meta);
        let res = CKKSRescaleOep::ckks_align_assign(self, &mut a_ct, &mut b_ct, scratch);
        let new_a_meta = a_ct.meta();
        let new_b_meta = b_ct.meta();
        drop(a_ct);
        drop(b_ct);
        a.set_meta(new_a_meta);
        b.set_meta(new_b_meta);
        res
    }

    fn ckks_align_tmp_bytes(&self) -> usize {
        CKKSRescaleOep::ckks_align_tmp_bytes(self)
    }
}
