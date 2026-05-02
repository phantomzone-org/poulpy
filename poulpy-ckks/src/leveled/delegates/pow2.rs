use anyhow::Result;
use poulpy_core::{
    GLWECopy, GLWEShift, ScratchArenaTakeCore,
    layouts::{GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSCiphertextToBackendMut, CKKSCiphertextToBackendRef};

use crate::{CKKSInfos, SetCKKSInfos, layouts::CKKSCiphertext, oep::CKKSImpl};

use crate::leveled::{api::CKKSPow2Ops, oep::CKKSPow2Oep};

impl<BE: Backend + CKKSImpl<BE>> CKKSPow2Ops<BE> for Module<BE>
where
    Module<BE>: GLWECopy<BE> + GLWEShift<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_mul_pow2_tmp_bytes(&self) -> usize {
        CKKSPow2Oep::ckks_mul_pow2_tmp_bytes(self)
    }

    fn ckks_mul_pow2_into<Dst, Src>(
        &self,
        dst: &mut Dst,
        src: &Src,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        Src: CKKSCiphertextToBackendRef<BE> + CKKSInfos,
    {
        let dst_meta = dst.meta();
        let mut dst_ct = CKKSCiphertext::from_inner(GLWEToBackendMut::to_backend_mut(dst), dst_meta);
        let src_ct = CKKSCiphertext::from_inner(GLWEToBackendRef::to_backend_ref(src), src.meta());
        let res = CKKSPow2Oep::ckks_mul_pow2_into(self, &mut dst_ct, &src_ct, bits, scratch);
        let new_meta = dst_ct.meta();
        drop(dst_ct);
        dst.set_meta(new_meta);
        res
    }

    fn ckks_mul_pow2_assign<Dst>(&self, dst: &mut Dst, bits: usize, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
    {
        let dst_meta = dst.meta();
        let mut dst_ct = CKKSCiphertext::from_inner(GLWEToBackendMut::to_backend_mut(dst), dst_meta);
        let res = CKKSPow2Oep::ckks_mul_pow2_assign(self, &mut dst_ct, bits, scratch);
        let new_meta = dst_ct.meta();
        drop(dst_ct);
        dst.set_meta(new_meta);
        res
    }

    fn ckks_div_pow2_tmp_bytes(&self) -> usize {
        CKKSPow2Oep::ckks_div_pow2_tmp_bytes(self)
    }

    fn ckks_div_pow2_into<Dst, Src>(
        &self,
        dst: &mut Dst,
        src: &Src,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        Src: CKKSCiphertextToBackendRef<BE> + CKKSInfos,
    {
        let dst_meta = dst.meta();
        let mut dst_ct = CKKSCiphertext::from_inner(GLWEToBackendMut::to_backend_mut(dst), dst_meta);
        let src_ct = CKKSCiphertext::from_inner(GLWEToBackendRef::to_backend_ref(src), src.meta());
        let res = CKKSPow2Oep::ckks_div_pow2_into(self, &mut dst_ct, &src_ct, bits, scratch);
        let new_meta = dst_ct.meta();
        drop(dst_ct);
        dst.set_meta(new_meta);
        res
    }

    fn ckks_div_pow2_assign<Dst>(&self, dst: &mut Dst, bits: usize) -> Result<()>
    where
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
    {
        let dst_meta = dst.meta();
        let mut dst_ct = CKKSCiphertext::from_inner(GLWEToBackendMut::to_backend_mut(dst), dst_meta);
        let res = CKKSPow2Oep::ckks_div_pow2_assign(self, &mut dst_ct, bits);
        let new_meta = dst_ct.meta();
        drop(dst_ct);
        dst.set_meta(new_meta);
        res
    }
}
