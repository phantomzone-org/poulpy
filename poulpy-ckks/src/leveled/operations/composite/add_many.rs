//! Sum of many CKKS ciphertexts.

use anyhow::{Result, bail};
use poulpy_core::{GLWEAdd, GLWENormalize, GLWEShift, ScratchTakeCore};
use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};

use crate::{
    CKKSInfos, checked_log_hom_rem_sub,
    layouts::CKKSCiphertext,
    layouts::ciphertext::CKKSOffset,
    leveled::operations::{
        add::{CKKSAddOps, CKKSAddOpsWithoutNormalization},
        composite::ensure_accumulation_fits,
    },
    oep::CKKSImpl,
};

pub trait CKKSAddManyOps<BE: Backend + CKKSImpl<BE>> {
    fn ckks_add_many_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE> + CKKSAddOps<BE>;

    fn ckks_add_many<D: DataRef>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        inputs: &[&CKKSCiphertext<D>],
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd + GLWEShift<BE> + GLWENormalize<BE> + CKKSAddOpsWithoutNormalization<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;
}

impl<BE: Backend + CKKSImpl<BE>> CKKSAddManyOps<BE> for Module<BE> {
    fn ckks_add_many_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE> + CKKSAddOps<BE>,
    {
        self.ckks_add_tmp_bytes()
    }

    fn ckks_add_many<D: DataRef>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        inputs: &[&CKKSCiphertext<D>],
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd + GLWEShift<BE> + GLWENormalize<BE> + CKKSAddOpsWithoutNormalization<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        match inputs.len() {
            0 => bail!("ckks_add_many: inputs must contain at least one ciphertext"),
            1 => {
                let offset = dst.offset_unary(inputs[0]);
                self.glwe_lsh(dst, inputs[0], offset, scratch);
                dst.meta = inputs[0].meta();
                dst.meta.log_hom_rem = checked_log_hom_rem_sub("ckks_add_many", inputs[0].log_hom_rem(), offset)?;
            }
            _ => {
                ensure_accumulation_fits("ckks_add_many", dst, inputs.len())?;
                unsafe {
                    self.ckks_add_without_normalization(dst, inputs[0], inputs[1], scratch)?;
                    for ct in &inputs[2..] {
                        self.ckks_add_inplace_without_normalization(dst, ct, scratch)?;
                    }
                }
                self.glwe_normalize_inplace(dst, scratch);
            }
        }
        Ok(())
    }
}
