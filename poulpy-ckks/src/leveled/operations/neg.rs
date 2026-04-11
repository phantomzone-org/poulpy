//! CKKS ciphertext negation.
//!
//! Negates each column of the GLWE ciphertext.

use crate::layouts::{PrecisionInfos, ciphertext::CKKSCiphertext};
use anyhow::Result;
use poulpy_core::{GLWECopy, GLWENegate, ScratchTakeCore, layouts::LWEInfos};
use poulpy_hal::{
    api::{ScratchAvailable, VecZnxLsh},
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};

impl<D: DataMut> CKKSCiphertext<D> {
    pub fn neg<BE: Backend>(
        &mut self,
        module: &Module<BE>,
        other: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWENegate + GLWECopy + VecZnxLsh<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        if self.max_k() < other.effective_k() {
            self.rescale(module, (other.max_k() - self.effective_k()).into(), other, scratch)?;
            module.glwe_negate_inplace(&mut self.inner);
        } else {
            module.glwe_negate(&mut self.inner, &other.inner);
            self.prec = other.prec
        }
        Ok(())
    }

    pub fn neg_inplace<BE: Backend>(&mut self, module: &Module<BE>)
    where
        Module<BE>: GLWENegate,
    {
        module.glwe_negate_inplace(&mut self.inner);
    }
}
