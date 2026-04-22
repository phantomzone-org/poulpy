//! Sum of many CKKS ciphertexts.

use anyhow::{Result, bail};
use poulpy_core::{GLWEAdd, GLWECopy, GLWENormalize, GLWEShift, ScratchTakeCore, layouts::LWEInfos};
use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};

use crate::{
    CKKSInfos,
    layouts::CKKSCiphertext,
    leveled::operations::add::{CKKSAddOps, CKKSAddOpsWithoutNormalization},
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
        Self: GLWEAdd + GLWECopy + GLWEShift<BE> + GLWENormalize<BE> + CKKSAddOpsWithoutNormalization<BE>,
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
        Self: GLWEAdd + GLWECopy + GLWEShift<BE> + GLWENormalize<BE> + CKKSAddOpsWithoutNormalization<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        match inputs.len() {
            0 => bail!("ckks_add_many: inputs must contain at least one ciphertext"),
            1 => {
                self.glwe_copy(dst, inputs[0]);
                dst.meta = inputs[0].meta();
            }
            _ => {
                // Overflow guard: each unnormalized add on K-normalized inputs
                // can grow limb magnitudes by ≤ 2^(base2k-1); i64 overflow
                // requires `n · 2^(base2k-1) ≤ 2^63`. See §3.3 of
                // [eprint 2023/771](https://eprint.iacr.org/2023/771) for the
                // accumulation bound.
                let base2k: usize = dst.base2k().as_usize();
                debug_assert!(
                    base2k < 64 && inputs.len() <= (1usize << (63 - base2k)),
                    "ckks_add_many: {} terms risks i64 overflow at base2k={base2k}",
                    inputs.len(),
                );
                // SAFETY: intermediates stay inside this function; the trailing
                // glwe_normalize_inplace restores K-normalization.
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
