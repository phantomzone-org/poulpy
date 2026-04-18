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

/// Multiplication and division of CKKS ciphertexts by powers of two.
///
/// These operations act on the torus placement of the ciphertext and update
/// metadata consistently with the consumed output precision.
pub trait CKKSPow2Ops<BE: Backend> {
    /// Returns scratch bytes required by [`Self::ckks_mul_pow2`].
    fn ckks_mul_pow2_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>;

    /// Computes `dst = src * 2^bits`.
    ///
    /// Errors occur if the destination buffer is too small for the aligned
    /// output metadata.
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

    /// Multiplies a ciphertext by `2^bits` in place.
    ///
    /// This only shifts the stored torus digits; semantic metadata stays the
    /// same.
    fn ckks_mul_pow2_inplace(&self, dst: &mut CKKSCiphertext<impl DataMut>, bits: usize, scratch: &mut Scratch<BE>) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;

    /// Returns scratch bytes required by [`Self::ckks_div_pow2`].
    fn ckks_div_pow2_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>;

    /// Computes `dst = src / 2^bits`.
    ///
    /// Errors include `InsufficientHomomorphicCapacity` if the division would
    /// consume more `log_hom_rem` than available.
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

    /// Divides a ciphertext by `2^bits` in place by reducing its
    /// `log_hom_rem`.
    ///
    /// Errors include `InsufficientHomomorphicCapacity`.
    fn ckks_div_pow2_inplace(&self, dst: &mut CKKSCiphertext<impl DataMut>, bits: usize) -> Result<()>;
}

#[doc(hidden)]
pub trait CKKSPow2OpsDefault<BE: Backend> {
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
    fn ckks_mul_pow2_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        self.ckks_mul_pow2_tmp_bytes_default()
    }

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

    fn ckks_div_pow2_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        self.ckks_div_pow2_tmp_bytes_default()
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
