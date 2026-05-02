use anyhow::Result;
use poulpy_core::{GLWENegate, GLWEShift, ScratchArenaTakeCore};
use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, Module, ScratchArena},
};

use crate::{CKKSCiphertextMut, CKKSCiphertextRef, oep::CKKSImpl};

pub(crate) trait CKKSNegOep<BE: Backend + CKKSImpl<BE>> {
    fn ckks_neg_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>;

    fn ckks_neg_into(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        src: &CKKSCiphertextRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWENegate<BE> + GLWEShift<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_neg_assign(&self, dst: &mut CKKSCiphertextMut<'_, BE>) -> Result<()>
    where
        Self: GLWENegate<BE>;
}

impl<BE: Backend + CKKSImpl<BE>> CKKSNegOep<BE> for Module<BE> {
    fn ckks_neg_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        BE::ckks_neg_tmp_bytes(self)
    }

    fn ckks_neg_into(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        src: &CKKSCiphertextRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWENegate<BE> + GLWEShift<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_neg_into(self, dst, src, scratch)
    }

    fn ckks_neg_assign(&self, dst: &mut CKKSCiphertextMut<'_, BE>) -> Result<()>
    where
        Self: GLWENegate<BE>,
    {
        BE::ckks_neg_assign(self, dst)
    }
}
