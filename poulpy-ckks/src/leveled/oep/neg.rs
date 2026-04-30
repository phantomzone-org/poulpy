use anyhow::Result;
use poulpy_core::{GLWENegate, GLWEShift, ScratchArenaTakeCore};
use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, Data, Module, ScratchArena},
};

use crate::{layouts::CKKSCiphertext, oep::CKKSImpl};

pub(crate) trait CKKSNegOep<BE: Backend + CKKSImpl<BE>> {
    fn ckks_neg_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>;

    fn ckks_neg_into<Dst: Data, Src: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        src: &CKKSCiphertext<Src>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWENegate<BE> + GLWEShift<BE>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        CKKSCiphertext<Src>: poulpy_core::layouts::GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_neg_assign<Dst: Data>(&self, dst: &mut CKKSCiphertext<Dst>) -> Result<()>
    where
        Self: GLWENegate<BE>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>;
}

impl<BE: Backend + CKKSImpl<BE>> CKKSNegOep<BE> for Module<BE> {
    fn ckks_neg_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        BE::ckks_neg_tmp_bytes(self)
    }

    fn ckks_neg_into<Dst: Data, Src: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        src: &CKKSCiphertext<Src>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWENegate<BE> + GLWEShift<BE>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        CKKSCiphertext<Src>: poulpy_core::layouts::GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_neg_into(self, dst, src, scratch)
    }

    fn ckks_neg_assign<Dst: Data>(&self, dst: &mut CKKSCiphertext<Dst>) -> Result<()>
    where
        Self: GLWENegate<BE>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
    {
        BE::ckks_neg_assign(self, dst)
    }
}
