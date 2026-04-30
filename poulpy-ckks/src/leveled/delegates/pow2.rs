use anyhow::Result;
use poulpy_core::{GLWECopy, GLWEShift, ScratchArenaTakeCore};
use poulpy_hal::layouts::{Backend, Data, Module, ScratchArena};

use crate::{layouts::CKKSCiphertext, oep::CKKSImpl};

use crate::leveled::{api::CKKSPow2Ops, oep::CKKSPow2Oep};

impl<BE: Backend + CKKSImpl<BE>> CKKSPow2Ops<BE> for Module<BE> {
    fn ckks_mul_pow2_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        CKKSPow2Oep::ckks_mul_pow2_tmp_bytes(self)
    }

    fn ckks_mul_pow2_into<Dst: Data, Src: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        src: &CKKSCiphertext<Src>,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        CKKSCiphertext<Src>: poulpy_core::layouts::GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        CKKSPow2Oep::ckks_mul_pow2_into(self, dst, src, bits, scratch)
    }

    fn ckks_mul_pow2_assign<Dst: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        CKKSPow2Oep::ckks_mul_pow2_assign(self, dst, bits, scratch)
    }

    fn ckks_div_pow2_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        CKKSPow2Oep::ckks_div_pow2_tmp_bytes(self)
    }

    fn ckks_div_pow2_into<Dst: Data, Src: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        src: &CKKSCiphertext<Src>,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE> + GLWECopy<BE>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        CKKSCiphertext<Src>: poulpy_core::layouts::GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        CKKSPow2Oep::ckks_div_pow2_into(self, dst, src, bits, scratch)
    }

    fn ckks_div_pow2_assign<Dst: Data>(&self, dst: &mut CKKSCiphertext<Dst>, bits: usize) -> Result<()> {
        CKKSPow2Oep::ckks_div_pow2_assign(self, dst, bits)
    }
}
