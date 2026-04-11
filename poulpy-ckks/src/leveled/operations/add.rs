//! CKKS ciphertext addition.
//!
//! Provides ct+ct, ct+pt in ZNX form, and ct+pt in RNX form, each in
//! out-of-place and in-place form.

use std::fmt::Debug;

use crate::layouts::{
    Metadata, PrecisionInfos,
    ciphertext::CKKSCiphertext,
    plaintext::{CKKSPlaintextConversion, CKKSPlaintextRnx, CKKSPlaintextZnx},
};
use poulpy_core::{
    GLWEAdd, GLWECopy, GLWEShift, ScratchTakeCore,
    layouts::{GLWEPlaintextLayout, LWEInfos},
};
use poulpy_hal::{
    api::{ScratchAvailable, VecZnxRshAdd},
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};

use anyhow::Result;
use rand_distr::num_traits::{Float, FloatConst};

impl<D: DataMut> CKKSCiphertext<D> {
    pub fn add<BE: Backend>(
        &mut self,
        module: &Module<BE>,
        a: &CKKSCiphertext<impl DataRef>,
        b: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEAdd + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        // If the destination has less precision than the aligned inputs, shift
        // the computation down by `offset` bits before writing the result.
        let offset = self.offset_binary(a, b);

        if offset == 0 && a.log_hom_rem() == b.log_hom_rem() {
            module.glwe_add(&mut self.inner, &a.inner, &b.inner);
        } else if a.log_hom_rem() <= b.log_hom_rem() {
            module.glwe_lsh(&mut self.inner, &a.inner, offset, scratch);
            module.glwe_lsh_add(&mut self.inner, &b.inner, b.log_hom_rem() - a.log_hom_rem() + offset, scratch);
        } else {
            module.glwe_lsh(&mut self.inner, &b.inner, offset, scratch);
            module.glwe_lsh_add(&mut self.inner, &a.inner, a.log_hom_rem() - b.log_hom_rem() + offset, scratch);
        }

        self.set_log_hom_rem(a.log_hom_rem().min(b.log_hom_rem()) - offset)?;

        Ok(())
    }

    pub fn add_inplace<BE: Backend>(
        &mut self,
        module: &Module<BE>,
        a: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEAdd + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let self_log_ingeter = self.log_hom_rem();

        if self_log_ingeter < a.log_hom_rem() {
            module.glwe_lsh_add(&mut self.inner, &a.inner, a.log_hom_rem() - self_log_ingeter, scratch);
        } else if self_log_ingeter > a.log_hom_rem() {
            module.glwe_lsh_inplace(&mut self.inner, self_log_ingeter - a.log_hom_rem(), scratch);
            module.glwe_add_inplace(&mut self.inner, &a.inner);
        } else {
            module.glwe_add_inplace(&mut self.inner, &a.inner);
        }

        self.set_log_hom_rem(self_log_ingeter.min(a.log_hom_rem()))?;

        Ok(())
    }

    pub fn add_pt_znx<BE: Backend>(
        &mut self,
        module: &Module<BE>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_znx: &CKKSPlaintextZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: VecZnxRshAdd<BE> + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let offset = self.offset_unary(a);
        module.glwe_lsh(&mut self.inner, &a.inner, offset, scratch);
        self.set_log_hom_rem(a.log_hom_rem() - offset)?;
        self.add_pt_znx_inplace(module, pt_znx, scratch)?;
        Ok(())
    }

    pub fn add_pt_znx_inplace<BE: Backend>(
        &mut self,
        module: &Module<BE>,
        pt_znx: &CKKSPlaintextZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: VecZnxRshAdd<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let log_integer = self.log_hom_rem();
        pt_znx.add_to(module, self.inner.data_mut(), log_integer, scratch);
        Ok(())
    }

    pub fn add_pt_rnx<F, BE: Backend>(
        &mut self,
        module: &Module<BE>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_rnx: &CKKSPlaintextRnx<F>,
        prec: Metadata,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        F: Float + FloatConst + Debug,
        Module<BE>: VecZnxRshAdd<BE> + GLWECopy + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextRnx<F>: CKKSPlaintextConversion,
    {
        let (pt_glwe, scratch_1) = scratch.take_glwe_plaintext(&GLWEPlaintextLayout {
            n: module.n().into(),
            base2k: self.inner.base2k(),
            k: prec.min_k(self.inner.base2k()),
        });
        let mut pt_znx = CKKSPlaintextZnx { inner: pt_glwe, prec };
        pt_rnx.to_znx::<BE>(&mut pt_znx).unwrap();
        self.add_pt_znx(module, a, &pt_znx, scratch_1)?;
        Ok(())
    }

    pub fn add_pt_rnx_inplace<F, BE: Backend>(
        &mut self,
        module: &Module<BE>,
        pt_rnx: &CKKSPlaintextRnx<F>,
        prec: Metadata,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        F: Float + FloatConst + Debug,
        Module<BE>: VecZnxRshAdd<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextRnx<F>: CKKSPlaintextConversion,
    {
        let (pt_glwe, scratch_1) = scratch.take_glwe_plaintext(&GLWEPlaintextLayout {
            n: module.n().into(),
            base2k: self.inner.base2k(),
            k: prec.min_k(self.inner.base2k()),
        });
        let mut pt_znx = CKKSPlaintextZnx { inner: pt_glwe, prec };
        pt_rnx.to_znx::<BE>(&mut pt_znx).unwrap();
        self.add_pt_znx_inplace(module, &pt_znx, scratch_1)?;
        Ok(())
    }
}
