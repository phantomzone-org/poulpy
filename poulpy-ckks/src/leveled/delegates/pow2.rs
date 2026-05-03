use anyhow::Result;
use poulpy_core::{
    GLWECopy, GLWEShift, ScratchArenaTakeCore,
    layouts::{GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSInfos, SetCKKSInfos, oep::CKKSImpl};

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
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + LWEInfos + CKKSInfos,
    {
        CKKSPow2Oep::ckks_mul_pow2_into(self, dst, src, bits, scratch)
    }

    fn ckks_mul_pow2_assign<Dst>(&self, dst: &mut Dst, bits: usize, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
    {
        CKKSPow2Oep::ckks_mul_pow2_assign(self, dst, bits, scratch)
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
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + LWEInfos + CKKSInfos,
    {
        CKKSPow2Oep::ckks_div_pow2_into(self, dst, src, bits, scratch)
    }

    fn ckks_div_pow2_assign<Dst>(&self, dst: &mut Dst, bits: usize) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
    {
        CKKSPow2Oep::ckks_div_pow2_assign(self, dst, bits)
    }
}
