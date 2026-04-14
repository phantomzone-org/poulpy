//! CKKS ciphertext negation.
//!
//! Negates each column of the GLWE ciphertext.

use crate::{CKKS, CKKSInfos, layouts::CKKSRescaleOps};
use anyhow::Result;
use poulpy_core::{
    GLWENegate, GLWEShift, ScratchTakeCore,
    layouts::{GLWE, GLWEToRef, LWEInfos},
};
use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, DataMut, Module, Scratch},
};

pub trait CKKSNegOps {
    fn neg<O, BE: Backend>(&mut self, module: &Module<BE>, other: &O, scratch: &mut Scratch<BE>) -> Result<()>
    where
        Module<BE>: GLWENegate + GLWEShift<BE>,
        O: GLWEToRef + LWEInfos + CKKSInfos,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn neg_inplace<BE: Backend>(&mut self, module: &Module<BE>)
    where
        Module<BE>: GLWENegate;
}

impl<D: DataMut> CKKSNegOps for GLWE<D, CKKS> {
    fn neg<O, BE: Backend>(&mut self, module: &Module<BE>, other: &O, scratch: &mut Scratch<BE>) -> Result<()>
    where
        Module<BE>: GLWENegate + GLWEShift<BE>,
        O: GLWEToRef + LWEInfos + CKKSInfos,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        if self.max_k() < other.effective_k() {
            self.rescale(module, (other.max_k() - self.effective_k()).into(), other, scratch)?;
            module.glwe_negate_inplace(self);
        } else {
            module.glwe_negate(self, other);
            self.meta = other.meta();
        }
        Ok(())
    }

    fn neg_inplace<BE: Backend>(&mut self, module: &Module<BE>)
    where
        Module<BE>: GLWENegate,
    {
        module.glwe_negate_inplace(self);
    }
}
