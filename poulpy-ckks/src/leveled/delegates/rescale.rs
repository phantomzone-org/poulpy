use anyhow::Result;
use poulpy_core::{
    GLWEShift, ScratchArenaTakeCore,
    layouts::{GLWEToBackendMut, GLWEToBackendRef, LWEInfos},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSInfos, SetCKKSInfos,
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
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
    {
        CKKSRescaleOep::ckks_rescale_assign(self, ct, k, scratch)
    }

    fn ckks_rescale_into<Dst, Src>(&self, dst: &mut Dst, k: usize, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
    {
        CKKSRescaleOep::ckks_rescale_into(self, dst, k, src, scratch)
    }

    fn ckks_align_assign<A, B>(&self, a: &mut A, b: &mut B, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        A: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        B: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
    {
        CKKSRescaleOep::ckks_align_assign(self, a, b, scratch)
    }

    fn ckks_align_tmp_bytes(&self) -> usize {
        CKKSRescaleOep::ckks_align_tmp_bytes(self)
    }
}
