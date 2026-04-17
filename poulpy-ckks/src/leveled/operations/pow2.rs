//! CKKS ciphertext multiplication and division by a power of two.
//!
//! These operations shift the bivariate GLWE payload without changing CKKS
//! metadata (`log_decimal`, `log_hom_rem`).  The decoded message becomes
//! `message * 2^bits` (for `mul_pow2`) or `message / 2^bits` (for `div_pow2`).

use crate::{
    CKKSInfos, checked_log_hom_rem_sub,
    layouts::{CKKSCiphertext, ciphertext::CKKSOffset},
};
use anyhow::Result;
use poulpy_core::{GLWECopy, GLWEShift, ScratchTakeCore};
use poulpy_hal::layouts::{Backend, DataMut, DataRef, Module, Scratch};

pub trait CKKSPow2Ops<BE: Backend> {
    fn ckks_mul_pow2(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        src: &CKKSCiphertext<impl DataRef>,
        bits: usize,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn ckks_mul_pow2_inplace(&self, dst: &mut CKKSCiphertext<impl DataMut>, bits: usize, scratch: &mut Scratch<BE>) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn ckks_div_pow2(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        src: &CKKSCiphertext<impl DataRef>,
        bits: usize,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE> + GLWECopy,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn ckks_div_pow2_inplace(&self, dst: &mut CKKSCiphertext<impl DataMut>, bits: usize) -> Result<()>;
}

#[doc(hidden)]
pub trait CKKSPow2OpsDefault<BE: Backend> {
    fn ckks_mul_pow2_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        src: &CKKSCiphertext<impl DataRef>,
        bits: usize,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let offset = dst.offset_unary(src);
        self.glwe_lsh(dst, src, bits + offset, scratch);
        dst.meta = src.meta();
        dst.meta.log_hom_rem = checked_log_hom_rem_sub("mul_pow2", dst.log_hom_rem(), offset)?;
        Ok(())
    }

    fn ckks_mul_pow2_inplace_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        bits: usize,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        self.glwe_lsh_inplace(dst, bits, scratch);
        Ok(())
    }

    fn ckks_div_pow2_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        src: &CKKSCiphertext<impl DataRef>,
        bits: usize,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE> + GLWECopy,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let offset = dst.offset_unary(src);
        self.glwe_lsh(dst, src, offset, scratch);
        dst.meta = src.meta();
        dst.meta.log_hom_rem = checked_log_hom_rem_sub("div_pow2", dst.log_hom_rem(), bits + offset)?;
        Ok(())
    }

    fn ckks_div_pow2_inplace_default(&self, dst: &mut CKKSCiphertext<impl DataMut>, bits: usize) -> Result<()> {
        dst.meta.log_hom_rem = checked_log_hom_rem_sub("div_pow2_inplace", dst.log_hom_rem(), bits)?;
        Ok(())
    }
}

impl<BE: Backend> CKKSPow2OpsDefault<BE> for Module<BE> {}

impl<BE: Backend> CKKSPow2Ops<BE> for Module<BE>
where
    Module<BE>: CKKSPow2OpsDefault<BE>,
{
    fn ckks_mul_pow2(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        src: &CKKSCiphertext<impl DataRef>,
        bits: usize,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        self.ckks_mul_pow2_default(dst, src, bits, scratch)
    }

    fn ckks_mul_pow2_inplace(&self, dst: &mut CKKSCiphertext<impl DataMut>, bits: usize, scratch: &mut Scratch<BE>) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        self.ckks_mul_pow2_inplace_default(dst, bits, scratch)
    }

    fn ckks_div_pow2(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        src: &CKKSCiphertext<impl DataRef>,
        bits: usize,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE> + GLWECopy,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        self.ckks_div_pow2_default(dst, src, bits, scratch)
    }

    fn ckks_div_pow2_inplace(&self, dst: &mut CKKSCiphertext<impl DataMut>, bits: usize) -> Result<()> {
        self.ckks_div_pow2_inplace_default(dst, bits)
    }
}
