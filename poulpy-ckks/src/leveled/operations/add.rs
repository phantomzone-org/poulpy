//! CKKS ciphertext addition.
//!
//! Provides ct+ct, ct+pt (compact and prepared), and ct+constant variants,
//! each in out-of-place and in-place form.  Compact plaintext operands are
//! expanded into the ciphertext torus layout via [`fill_offset_pt`] before
//! the underlying GLWE addition.
//!
//! [`fill_offset_pt`]: super::utils::fill_offset_pt

use std::fmt::Debug;

use crate::layouts::{
    ciphertext::CKKSCiphertext,
    plaintext::{CKKSPlaintextConversion, CKKSPlaintextRnx, CKKSPlaintextZnx, PrecisionLayout},
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
        // Case where the receiver ciphertext has less maximum precision than the minimum precision between the two
        // input ciphertexts. In this case we need to shift the addition by `offset` bits.
        let offset = a
            .inner
            .max_k()
            .min(b.inner.max_k())
            .as_usize()
            .saturating_sub(self.inner.max_k().as_usize());

        if offset == 0 && a.log_delta == b.log_delta {
            module.glwe_add(&mut self.inner, &a.inner, &b.inner);
        } else if a.log_delta <= b.log_delta {
            module.glwe_lsh(&mut self.inner, &a.inner, offset, scratch);
            module.glwe_lsh_add(&mut self.inner, &b.inner, b.log_delta - a.log_delta + offset, scratch);
        } else {
            module.glwe_lsh(&mut self.inner, &b.inner, offset, scratch);
            module.glwe_lsh_add(&mut self.inner, &a.inner, a.log_delta - b.log_delta + offset, scratch);
        }

        self.log_delta = a.log_delta.min(b.log_delta) - offset;

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
        if self.log_delta < a.log_delta {
            module.glwe_lsh_add(&mut self.inner, &a.inner, a.log_delta - self.log_delta, scratch);
        } else if self.log_delta > a.log_delta {
            module.glwe_lsh_inplace(&mut self.inner, self.log_delta - a.log_delta, scratch);
            module.glwe_add_inplace(&mut self.inner, &a.inner);
        } else {
            module.glwe_add_inplace(&mut self.inner, &a.inner);
        }

        self.log_delta = self.log_delta.min(a.log_delta);

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
        let offset = a.inner.max_k().as_usize().saturating_sub(self.inner.max_k().as_usize());
        module.glwe_lsh(&mut self.inner, &a.inner, offset, scratch);
        self.log_delta = a.log_delta - offset;
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
        let offset = self.log_delta + pt_znx.log_decimal_prec() - self.inner.base2k().as_usize();
        pt_znx.add_to(module, self.inner.data_mut(), offset, scratch);
        Ok(())
    }

    pub fn add_pt_rnx<F, BE: Backend>(
        &mut self,
        module: &Module<BE>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_rnx: &CKKSPlaintextRnx<F>,
        prec: PrecisionLayout,
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
            k: prec.k(self.inner.base2k()),
        });
        let mut pt_znx = CKKSPlaintextZnx { data: pt_glwe, prec };
        pt_rnx.to_znx::<BE>(&mut pt_znx).unwrap();
        self.add_pt_znx(module, a, &pt_znx, scratch_1)?;
        Ok(())
    }

    pub fn add_pt_rnx_inplace<F, BE: Backend>(
        &mut self,
        module: &Module<BE>,
        pt_rnx: &CKKSPlaintextRnx<F>,
        prec: PrecisionLayout,
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
            k: prec.k(self.inner.base2k()),
        });
        let mut pt_znx = CKKSPlaintextZnx { data: pt_glwe, prec };
        pt_rnx.to_znx::<BE>(&mut pt_znx).unwrap();
        self.add_pt_znx_inplace(module, &pt_znx, scratch_1)?;
        Ok(())
    }
}
