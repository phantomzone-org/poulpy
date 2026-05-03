use anyhow::Result;
use poulpy_core::{
    GLWENegate, GLWEShift, ScratchArenaTakeCore,
    layouts::{GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos},
};
use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, Module, ScratchArena},
};

use crate::{CKKSInfos, SetCKKSInfos, oep::CKKSImpl};

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
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + LWEInfos + CKKSInfos,
    {
        CKKSNegOep::ckks_neg_into(self, dst, src, scratch)
    }

    fn ckks_neg_assign<Dst>(&self, dst: &mut Dst) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
    {
        CKKSNegOep::ckks_neg_assign(self, dst)
    }
}
