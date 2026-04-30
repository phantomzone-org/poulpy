use anyhow::Result;
use poulpy_core::{
    GLWEShift, ScratchArenaTakeCore,
    layouts::{GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::layouts::{Backend, Data, Module, ScratchArena};

use crate::{CKKSInfos, checked_log_budget_sub, layouts::CKKSCiphertext};

#[doc(hidden)]
pub(crate) trait CKKSRescaleOpsDefault<BE: Backend> {
    fn ckks_rescale_tmp_bytes_default(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        self.glwe_shift_tmp_bytes()
    }

    fn ckks_align_tmp_bytes_default(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        self.glwe_shift_tmp_bytes()
    }

    fn ckks_rescale_assign_default<Dst>(
        &self,
        ct: &mut CKKSCiphertext<Dst>,
        k: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        Self: GLWEShift<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        let log_budget = checked_log_budget_sub("rescale_assign", ct.log_budget(), k)?;
        self.glwe_lsh_assign(ct, k, scratch);
        ct.meta.log_budget = log_budget;
        Ok(())
    }

    fn ckks_rescale_into_default<Dst, Src>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        k: usize,
        src: &CKKSCiphertext<Src>,
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
        let log_budget = checked_log_budget_sub("rescale", src.log_budget(), k)?;
        self.glwe_lsh(dst, src, k, scratch);
        dst.meta = src.meta();
        dst.meta.log_budget = log_budget;
        Ok(())
    }

    fn ckks_align_assign_default<A, B>(
        &self,
        a: &mut CKKSCiphertext<A>,
        b: &mut CKKSCiphertext<B>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        A: Data,
        B: Data,
        Self: GLWEShift<BE>,
        CKKSCiphertext<A>: GLWEToBackendMut<BE>,
        CKKSCiphertext<B>: GLWEToBackendMut<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        if a.log_budget() < b.log_budget() {
            self.ckks_rescale_assign_default(b, b.log_budget() - a.log_budget(), scratch)
        } else {
            self.ckks_rescale_assign_default(a, a.log_budget() - b.log_budget(), scratch)
        }
    }
}

impl<BE: Backend> CKKSRescaleOpsDefault<BE> for Module<BE> {}
