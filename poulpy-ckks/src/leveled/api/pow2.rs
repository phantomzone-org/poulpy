use anyhow::Result;
use poulpy_core::{GLWECopy, GLWEShift, ScratchArenaTakeCore};
use poulpy_hal::layouts::{Backend, Data, ScratchArena};

use crate::{layouts::CKKSCiphertext, oep::CKKSImpl};

pub trait CKKSPow2Ops<BE: Backend + CKKSImpl<BE>> {
    fn ckks_mul_pow2_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>;

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
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;

    fn ckks_mul_pow2_assign<Dst: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;

    fn ckks_div_pow2_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>;

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
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;

    fn ckks_div_pow2_assign<Dst: Data>(&self, dst: &mut CKKSCiphertext<Dst>, bits: usize) -> Result<()>;
}
