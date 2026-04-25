use anyhow::Result;
use poulpy_core::{GLWENegate, GLWEShift, ScratchTakeCore};
use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};

use crate::{
    CKKSInfos, checked_log_hom_rem_sub,
    layouts::{CKKSCiphertext, ciphertext::CKKSOffset},
};

pub(crate) trait CKKSNegDefault<BE: Backend> {
    fn ckks_neg_tmp_bytes_default(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        self.glwe_shift_tmp_bytes()
    }

    fn ckks_neg_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        src: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWENegate + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let offset = dst.offset_unary(src);
        if offset != 0 {
            self.glwe_lsh(dst, src, offset, scratch);
            dst.meta = src.meta();
            dst.meta.log_hom_rem = checked_log_hom_rem_sub("neg", src.log_hom_rem(), offset)?;
            self.glwe_negate_inplace(dst);
        } else {
            self.glwe_negate(dst, src);
            dst.meta = src.meta();
        }
        Ok(())
    }

    fn ckks_neg_inplace_default(&self, dst: &mut CKKSCiphertext<impl DataMut>)
    where
        Self: GLWENegate,
    {
        self.glwe_negate_inplace(dst);
    }
}

impl<BE: Backend> CKKSNegDefault<BE> for Module<BE> {}
