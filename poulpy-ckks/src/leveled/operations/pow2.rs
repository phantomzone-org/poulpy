//! CKKS ciphertext multiplication and division by a power of two.
//!
//! These operations shift the bivariate GLWE payload without changing CKKS
//! metadata (`log_decimal`, `log_hom_rem`).  The decoded message becomes
//! `message * 2^bits` (for `mul_pow2`) or `message / 2^bits` (for `div_pow2`).

use crate::{CKKS, CKKSInfos, layouts::ciphertext::CKKSOffset};
use anyhow::Result;
use poulpy_core::{GLWECopy, GLWEShift, ScratchTakeCore, layouts::GLWE};
use poulpy_hal::layouts::{Backend, DataMut, DataRef, Module, Scratch};

pub trait CKKSPow2Ops {
    fn mul_pow2<BE: Backend>(
        &mut self,
        module: &Module<BE>,
        a: &GLWE<impl DataRef, CKKS>,
        bits: usize,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn mul_pow2_inplace<BE: Backend>(&mut self, module: &Module<BE>, bits: usize, scratch: &mut Scratch<BE>) -> Result<()>
    where
        Module<BE>: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn div_pow2<BE: Backend>(
        &mut self,
        module: &Module<BE>,
        a: &GLWE<impl DataRef, CKKS>,
        bits: usize,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEShift<BE> + GLWECopy,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn div_pow2_inplace(&mut self, bits: usize) -> Result<()>;
}

impl<D: DataMut> CKKSPow2Ops for GLWE<D, CKKS> {
    /// Out-of-place: `self = a * 2^bits` (left-shift the GLWE payload).
    fn mul_pow2<BE: Backend>(
        &mut self,
        module: &Module<BE>,
        other: &GLWE<impl DataRef, CKKS>,
        bits: usize,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let offset = self.offset_unary(other);
        module.glwe_lsh(self, other, bits + offset, scratch);
        self.meta = other.meta();
        self.set_log_hom_rem(self.log_hom_rem() - offset)?;
        Ok(())
    }

    /// In-place: `self *= 2^bits` (left-shift the GLWE payload).
    fn mul_pow2_inplace<BE: Backend>(&mut self, module: &Module<BE>, bits: usize, scratch: &mut Scratch<BE>) -> Result<()>
    where
        Module<BE>: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.glwe_lsh_inplace(self, bits, scratch);
        Ok(())
    }

    /// Out-of-place: `self = a / 2^bits` (right-shift the GLWE payload).
    fn div_pow2<BE: Backend>(
        &mut self,
        module: &Module<BE>,
        other: &GLWE<impl DataRef, CKKS>,
        bits: usize,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEShift<BE> + GLWECopy,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let offset = self.offset_unary(other);
        module.glwe_lsh(self, other, offset, scratch);
        self.meta = other.meta();
        self.set_log_hom_rem(self.log_hom_rem() - bits - offset)?;
        Ok(())
    }

    /// In-place: `self /= 2^bits`.
    fn div_pow2_inplace(&mut self, bits: usize) -> Result<()> {
        self.set_log_hom_rem(self.log_hom_rem() - bits)?;
        Ok(())
    }
}
