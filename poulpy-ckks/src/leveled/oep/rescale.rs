use anyhow::Result;
use poulpy_core::{GLWEShift, ScratchArenaTakeCore};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSCiphertextMut, CKKSCiphertextRef, oep::CKKSImpl};

pub(crate) trait CKKSRescaleOep<BE: Backend + CKKSImpl<BE>> {
    fn ckks_rescale_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>;

    fn ckks_rescale_assign(&self, ct: &mut CKKSCiphertextMut<'_, BE>, k: usize, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWEShift<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;

    fn ckks_rescale_into(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        k: usize,
        src: &CKKSCiphertextRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;

    fn ckks_align_assign(
        &self,
        a: &mut CKKSCiphertextMut<'_, BE>,
        b: &mut CKKSCiphertextMut<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;

    fn ckks_align_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>;
}

impl<BE: Backend + CKKSImpl<BE>> CKKSRescaleOep<BE> for Module<BE> {
    fn ckks_rescale_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        BE::ckks_rescale_tmp_bytes(self)
    }

    fn ckks_rescale_assign(&self, ct: &mut CKKSCiphertextMut<'_, BE>, k: usize, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWEShift<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_rescale_assign(self, ct, k, scratch)
    }

    fn ckks_rescale_into(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        k: usize,
        src: &CKKSCiphertextRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_rescale_into(self, dst, k, src, scratch)
    }

    fn ckks_align_assign(
        &self,
        a: &mut CKKSCiphertextMut<'_, BE>,
        b: &mut CKKSCiphertextMut<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_align_assign(self, a, b, scratch)
    }

    fn ckks_align_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        BE::ckks_align_tmp_bytes(self)
    }
}
