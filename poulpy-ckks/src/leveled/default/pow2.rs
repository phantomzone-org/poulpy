use anyhow::Result;
use poulpy_core::{
    GLWECopy, GLWEShift, ScratchArenaTakeCore,
    layouts::{GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::layouts::{Backend, Data, Module, ScratchArena};

use crate::{
    CKKSInfos, checked_log_budget_sub,
    layouts::{CKKSCiphertext, ciphertext::CKKSOffset},
};

pub(crate) trait CKKSPow2Default<BE: Backend> {
    fn ckks_mul_pow2_tmp_bytes_default(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        self.glwe_shift_tmp_bytes()
    }

    fn ckks_div_pow2_tmp_bytes_default(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        self.glwe_shift_tmp_bytes()
    }

    fn ckks_mul_pow2_into_default<Dst, Src>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        src: &CKKSCiphertext<Src>,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        Src: Data,
        Self: GLWEShift<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<Src>: GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        let offset = dst.offset_unary(src);
        self.glwe_lsh(dst, src, bits + offset, scratch);
        dst.meta = src.meta();
        dst.meta.log_budget = checked_log_budget_sub("mul_pow2", dst.log_budget(), offset)?;
        Ok(())
    }

    fn ckks_mul_pow2_assign_default<Dst>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        Self: GLWEShift<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        self.glwe_lsh_assign(dst, bits, scratch);
        Ok(())
    }

    fn ckks_div_pow2_into_default<Dst, Src>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        src: &CKKSCiphertext<Src>,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        Src: Data,
        Self: GLWEShift<BE> + GLWECopy<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<Src>: GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        let offset = dst.offset_unary(src);
        self.glwe_lsh(dst, src, offset, scratch);
        dst.meta = src.meta();
        dst.meta.log_budget = checked_log_budget_sub("div_pow2", dst.log_budget(), bits + offset)?;
        dst.meta.log_delta += bits;
        Ok(())
    }

    fn ckks_div_pow2_assign_default<Dst>(&self, dst: &mut CKKSCiphertext<Dst>, bits: usize) -> Result<()>
    where
        Dst: Data,
    {
        dst.meta.log_budget = checked_log_budget_sub("div_pow2_assign", dst.log_budget(), bits)?;
        Ok(())
    }
}

impl<BE: Backend> CKKSPow2Default<BE> for Module<BE> {}
