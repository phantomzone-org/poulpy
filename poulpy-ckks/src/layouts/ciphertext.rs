//! CKKS metadata attached to [`GLWE`].
//!
//! A CKKS ciphertext is represented as `GLWE<D, CKKS>`, where `CKKS` carries
//! scheme-specific metadata (precision, offsets, scale).

use poulpy_core::{
    GLWEShift, ScratchTakeCore,
    layouts::{Base2K, Degree, GLWE, GLWEInfos, GLWEToRef, LWEInfos, Rank, TorusPrecision},
};
use poulpy_hal::layouts::{Backend, Data, DataMut, Module, Scratch};

use crate::{CKKS, CKKSInfos, checked_log_hom_rem_sub, ensure_limb_count_fits, ensure_log_decimal_fits, ensure_log_hom_rem_fits};
use anyhow::Result;

impl CKKS {
    pub fn alloc(n: Degree, k: TorusPrecision, base2k: Base2K) -> GLWE<Vec<u8>, CKKS> {
        GLWE::alloc_with_meta(n, base2k, k, Rank(1), CKKS::default())
    }

    pub fn alloc_from_infos<A>(infos: &A) -> Result<GLWE<Vec<u8>, CKKS>>
    where
        A: GLWEInfos,
    {
        Ok(GLWE::alloc_with_meta(
            infos.n(),
            infos.base2k(),
            infos.max_k(),
            infos.rank(),
            CKKS::default(),
        ))
    }
}

impl<D: Data> CKKSInfos for GLWE<D, CKKS> {
    fn meta(&self) -> CKKS {
        self.meta
    }

    fn log_decimal(&self) -> usize {
        self.meta.log_decimal
    }

    fn log_hom_rem(&self) -> usize {
        self.meta.log_hom_rem
    }

    fn set_log_decimal(&mut self, log_decimal: usize) -> Result<()> {
        ensure_log_decimal_fits(self.max_k().as_usize(), self.log_hom_rem(), log_decimal)?;
        self.meta.log_decimal = log_decimal;
        Ok(())
    }

    fn set_log_hom_rem(&mut self, log_hom_rem: usize) -> Result<()> {
        ensure_log_hom_rem_fits(self.max_k().as_usize(), self.log_decimal(), log_hom_rem)?;
        self.meta.log_hom_rem = log_hom_rem;
        Ok(())
    }
}

pub trait CKKSMaintainOps {
    /// Reallocates the owned backing buffer so capacity matches `size` limb count.
    /// Fails if dropping limbs would reduce the gap between log_hom_rem and max_k
    /// below log_decimal.
    fn reallocate_limbs_checked(&mut self, size: usize) -> Result<()>;
    /// Reallocates the owned backing buffer such that [Self::max_k()] >= [Self::log_decimal()] + [Self::log_hom_rem()].
    fn compact_limbs(&mut self) -> Result<()>;
}

impl CKKSMaintainOps for GLWE<Vec<u8>, CKKS> {
    fn reallocate_limbs_checked(&mut self, size: usize) -> Result<()> {
        ensure_limb_count_fits(self.max_k().as_usize(), self.log_decimal(), self.base2k().as_usize(), size)?;
        self.data_mut().reallocate_limbs(size);
        Ok(())
    }

    fn compact_limbs(&mut self) -> Result<()> {
        let size = (self.max_k().as_usize() - self.log_decimal() + self.log_hom_rem()).div_ceil(self.base2k().as_usize());
        self.reallocate_limbs_checked(size)?;
        Ok(())
    }
}

pub trait CKKSRescaleOps {
    fn rescale_inplace<BE: Backend>(&mut self, module: &Module<BE>, k: usize, scratch: &mut Scratch<BE>) -> Result<()>
    where
        Module<BE>: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;
    fn rescale<O, BE: Backend>(&mut self, module: &Module<BE>, k: usize, other: &O, scratch: &mut Scratch<BE>) -> Result<()>
    where
        Module<BE>: GLWEShift<BE>,
        O: GLWEToRef + CKKSInfos,
        Scratch<BE>: ScratchTakeCore<BE>;
    fn align_inplace<BE: Backend>(
        &mut self,
        module: &Module<BE>,
        other: &mut GLWE<impl DataMut, CKKS>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;
}

impl<D: DataMut> CKKSRescaleOps for GLWE<D, CKKS> {
    fn rescale_inplace<BE: Backend>(&mut self, module: &Module<BE>, k: usize, scratch: &mut Scratch<BE>) -> Result<()>
    where
        Module<BE>: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let log_hom_rem = checked_log_hom_rem_sub("rescale_inplace", self.log_hom_rem(), k)?;
        module.glwe_lsh_inplace(self, k, scratch);
        self.set_log_hom_rem(log_hom_rem)?;
        Ok(())
    }

    fn rescale<O, BE: Backend>(&mut self, module: &Module<BE>, k: usize, other: &O, scratch: &mut Scratch<BE>) -> Result<()>
    where
        Module<BE>: GLWEShift<BE>,
        O: GLWEToRef + CKKSInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let log_hom_rem = checked_log_hom_rem_sub("rescale", other.log_hom_rem(), k)?;
        module.glwe_lsh(self, other, k, scratch);
        self.meta = other.meta();
        self.set_log_hom_rem(log_hom_rem)?;
        Ok(())
    }

    fn align_inplace<BE: Backend>(
        &mut self,
        module: &Module<BE>,
        other: &mut GLWE<impl DataMut, CKKS>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        if self.log_hom_rem() < other.log_hom_rem() {
            other.rescale_inplace(module, other.log_hom_rem() - self.log_hom_rem(), scratch)
        } else {
            self.rescale_inplace(module, self.log_hom_rem() - other.log_hom_rem(), scratch)
        }
    }
}

pub(crate) trait CKKSOffset {
    fn offset_binary<A, B>(&self, a: &A, b: &B) -> usize
    where
        A: LWEInfos + CKKSInfos,
        B: LWEInfos + CKKSInfos;
    fn offset_unary<A>(&self, a: &A) -> usize
    where
        A: LWEInfos + CKKSInfos;
}

impl<D: Data> CKKSOffset for GLWE<D, CKKS> {
    fn offset_binary<A, B>(&self, a: &A, b: &B) -> usize
    where
        A: LWEInfos + CKKSInfos,
        B: LWEInfos + CKKSInfos,
    {
        a.effective_k().min(b.effective_k()).saturating_sub(self.max_k().as_usize())
    }

    fn offset_unary<A>(&self, a: &A) -> usize
    where
        A: LWEInfos + CKKSInfos,
    {
        a.effective_k().saturating_sub(self.max_k().as_usize())
    }
}
