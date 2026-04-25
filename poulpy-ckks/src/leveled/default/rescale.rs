use anyhow::Result;
use poulpy_core::{GLWEShift, ScratchTakeCore};
use poulpy_hal::layouts::{Backend, DataMut, DataRef, Module, Scratch};

use crate::{CKKSInfos, checked_log_hom_rem_sub, layouts::CKKSCiphertext};

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

    fn ckks_rescale_inplace_default(
        &self,
        ct: &mut CKKSCiphertext<impl DataMut>,
        k: usize,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let log_hom_rem = checked_log_hom_rem_sub("rescale_inplace", ct.log_hom_rem(), k)?;
        self.glwe_lsh_inplace(ct, k, scratch);
        ct.meta.log_hom_rem = log_hom_rem;
        Ok(())
    }

    fn ckks_rescale_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        k: usize,
        src: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let log_hom_rem = checked_log_hom_rem_sub("rescale", src.log_hom_rem(), k)?;
        self.glwe_lsh(dst, src, k, scratch);
        dst.meta = src.meta();
        dst.meta.log_hom_rem = log_hom_rem;
        Ok(())
    }

    fn ckks_align_inplace_default(
        &self,
        a: &mut CKKSCiphertext<impl DataMut>,
        b: &mut CKKSCiphertext<impl DataMut>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        if a.log_hom_rem() < b.log_hom_rem() {
            self.ckks_rescale_inplace_default(b, b.log_hom_rem() - a.log_hom_rem(), scratch)
        } else {
            self.ckks_rescale_inplace_default(a, a.log_hom_rem() - b.log_hom_rem(), scratch)
        }
    }
}

impl<BE: Backend> CKKSRescaleOpsDefault<BE> for Module<BE> {}
