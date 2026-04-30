use anyhow::Result;
use poulpy_core::{
    GLWEShift, ScratchArenaTakeCore,
    layouts::{GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::layouts::{Backend, Data, Module, ScratchArena};

use crate::{
    layouts::CKKSCiphertext,
    leveled::{api::CKKSRescaleOps, default::CKKSRescaleOpsDefault},
};

impl<BE: Backend> CKKSRescaleOps<BE> for Module<BE>
where
    Module<BE>: CKKSRescaleOpsDefault<BE>,
{
    fn ckks_rescale_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        self.ckks_rescale_tmp_bytes_default()
    }

    fn ckks_rescale_assign<D: Data>(&self, ct: &mut CKKSCiphertext<D>, k: usize, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWEShift<BE>,
        CKKSCiphertext<D>: GLWEToBackendMut<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        self.ckks_rescale_assign_default(ct, k, scratch)
    }

    fn ckks_rescale_into<Dst: Data, Src: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        k: usize,
        src: &CKKSCiphertext<Src>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<Src>: GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        self.ckks_rescale_into_default(dst, k, src, scratch)
    }

    fn ckks_align_assign<A: Data, B: Data>(
        &self,
        a: &mut CKKSCiphertext<A>,
        b: &mut CKKSCiphertext<B>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        CKKSCiphertext<A>: GLWEToBackendMut<BE>,
        CKKSCiphertext<B>: GLWEToBackendMut<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        self.ckks_align_assign_default(a, b, scratch)
    }

    fn ckks_align_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        self.ckks_align_tmp_bytes_default()
    }
}
