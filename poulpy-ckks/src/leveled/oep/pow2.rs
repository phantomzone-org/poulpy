use anyhow::Result;
use poulpy_core::{GLWECopy, GLWEShift, ScratchArenaTakeCore};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSCiphertextMut, CKKSCiphertextRef, oep::CKKSImpl};

pub(crate) trait CKKSPow2Oep<BE: Backend + CKKSImpl<BE>> {
    fn ckks_mul_pow2_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>;

    fn ckks_mul_pow2_into(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        src: &CKKSCiphertextRef<'_, BE>,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;

    fn ckks_mul_pow2_assign(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;

    fn ckks_div_pow2_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>;

    fn ckks_div_pow2_into(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        src: &CKKSCiphertextRef<'_, BE>,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE> + GLWECopy<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;

    fn ckks_div_pow2_assign(&self, dst: &mut CKKSCiphertextMut<'_, BE>, bits: usize) -> Result<()>;
}

impl<BE: Backend + CKKSImpl<BE>> CKKSPow2Oep<BE> for Module<BE> {
    fn ckks_mul_pow2_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        BE::ckks_mul_pow2_tmp_bytes(self)
    }

    fn ckks_mul_pow2_into(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        src: &CKKSCiphertextRef<'_, BE>,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_mul_pow2_into(self, dst, src, bits, scratch)
    }

    fn ckks_mul_pow2_assign(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_mul_pow2_assign(self, dst, bits, scratch)
    }

    fn ckks_div_pow2_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        BE::ckks_div_pow2_tmp_bytes(self)
    }

    fn ckks_div_pow2_into(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        src: &CKKSCiphertextRef<'_, BE>,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE> + GLWECopy<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_div_pow2_into(self, dst, src, bits, scratch)
    }

    fn ckks_div_pow2_assign(&self, dst: &mut CKKSCiphertextMut<'_, BE>, bits: usize) -> Result<()> {
        BE::ckks_div_pow2_assign(self, dst, bits)
    }
}
