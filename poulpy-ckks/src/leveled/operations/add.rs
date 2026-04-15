//! CKKS ciphertext addition.
//!
//! Provides ct+ct, ct+pt in ZNX form, and ct+pt in RNX form, each in
//! out-of-place and in-place form.

use crate::{
    CKKS, CKKSInfos,
    layouts::{
        ciphertext::CKKSOffset,
        plaintext::{CKKSPlaintextConversion, CKKSPlaintextRnx, CKKSPlaintextZnx, attach_meta},
    },
    leveled::operations::pt_znx::CKKSPlaintextZnxOps,
};
use poulpy_core::{
    GLWEAdd, GLWEShift, ScratchTakeCore,
    layouts::{GLWE, GLWEPlaintextLayout, LWEInfos},
};
use poulpy_hal::{
    api::{ScratchAvailable, VecZnxRshAddInto},
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};

use anyhow::Result;

pub trait CKKSAddOps {
    fn add<BE: Backend>(
        &mut self,
        module: &Module<BE>,
        a: &GLWE<impl DataRef, CKKS>,
        b: &GLWE<impl DataRef, CKKS>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEAdd + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn add_inplace<BE: Backend>(
        &mut self,
        module: &Module<BE>,
        a: &GLWE<impl DataRef, CKKS>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEAdd + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn add_pt_znx<BE: Backend>(
        &mut self,
        module: &Module<BE>,
        a: &GLWE<impl DataRef, CKKS>,
        pt_znx: &CKKSPlaintextZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: VecZnxRshAddInto<BE> + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn add_pt_znx_inplace<BE: Backend>(
        &mut self,
        module: &Module<BE>,
        pt_znx: &CKKSPlaintextZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: VecZnxRshAddInto<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn add_pt_rnx<F, BE: Backend>(
        &mut self,
        module: &Module<BE>,
        a: &GLWE<impl DataRef, CKKS>,
        pt_rnx: &CKKSPlaintextRnx<F>,
        prec: CKKS,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: VecZnxRshAddInto<BE> + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextRnx<F>: CKKSPlaintextConversion;

    fn add_pt_rnx_inplace<F, BE: Backend>(
        &mut self,
        module: &Module<BE>,
        pt_rnx: &CKKSPlaintextRnx<F>,
        prec: CKKS,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: VecZnxRshAddInto<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextRnx<F>: CKKSPlaintextConversion;
}

impl<D: DataMut> CKKSAddOps for GLWE<D, CKKS> {
    fn add<BE: Backend>(
        &mut self,
        module: &Module<BE>,
        a: &GLWE<impl DataRef, CKKS>,
        b: &GLWE<impl DataRef, CKKS>,
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
            module.glwe_add_into(self, a, b);
        } else if a.log_hom_rem() <= b.log_hom_rem() {
            module.glwe_lsh(self, a, offset, scratch);
            module.glwe_lsh_add(self, b, b.log_hom_rem() - a.log_hom_rem() + offset, scratch);
        } else {
            module.glwe_lsh(self, b, offset, scratch);
            module.glwe_lsh_add(self, a, a.log_hom_rem() - b.log_hom_rem() + offset, scratch);
        }

        self.set_log_decimal(a.log_decimal().max(b.log_decimal()))?;
        self.set_log_hom_rem(a.log_hom_rem().min(b.log_hom_rem()) - offset)?;

        Ok(())
    }

    fn add_inplace<BE: Backend>(
        &mut self,
        module: &Module<BE>,
        a: &GLWE<impl DataRef, CKKS>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEAdd + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let self_log_hom_rem = self.log_hom_rem();

        if self_log_hom_rem < a.log_hom_rem() {
            module.glwe_lsh_add(self, a, a.log_hom_rem() - self_log_hom_rem, scratch);
        } else if self_log_hom_rem > a.log_hom_rem() {
            module.glwe_lsh_inplace(self, self_log_hom_rem - a.log_hom_rem(), scratch);
            module.glwe_add_assign(self, a);
        } else {
            module.glwe_add_assign(self, a);
        }

        self.set_log_hom_rem(self_log_hom_rem.min(a.log_hom_rem()))?;

        Ok(())
    }

    fn add_pt_znx<BE: Backend>(
        &mut self,
        module: &Module<BE>,
        a: &GLWE<impl DataRef, CKKS>,
        pt_znx: &CKKSPlaintextZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: VecZnxRshAddInto<BE> + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let offset = self.offset_unary(a);
        module.glwe_lsh(self, a, offset, scratch);
        self.meta = a.meta();
        self.set_log_hom_rem(a.log_hom_rem() - offset)?;
        self.add_pt_znx_inplace(module, pt_znx, scratch)?;
        Ok(())
    }

    fn add_pt_znx_inplace<BE: Backend>(
        &mut self,
        module: &Module<BE>,
        pt_znx: &CKKSPlaintextZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: VecZnxRshAddInto<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        module.ckks_add_pt_znx(self, pt_znx, scratch)?;
        Ok(())
    }

    fn add_pt_rnx<F, BE: Backend>(
        &mut self,
        module: &Module<BE>,
        a: &GLWE<impl DataRef, CKKS>,
        pt_rnx: &CKKSPlaintextRnx<F>,
        prec: CKKS,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: VecZnxRshAddInto<BE> + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextRnx<F>: CKKSPlaintextConversion,
    {
        let (pt_glwe, scratch_1) = scratch.take_glwe_plaintext(&GLWEPlaintextLayout {
            n: module.n().into(),
            base2k: self.base2k(),
            k: prec.min_k(self.base2k()),
        });
        let mut pt_znx = attach_meta(pt_glwe, prec);
        pt_rnx.to_znx::<BE>(&mut pt_znx).unwrap();
        self.add_pt_znx(module, a, &pt_znx, scratch_1)?;
        Ok(())
    }

    fn add_pt_rnx_inplace<F, BE: Backend>(
        &mut self,
        module: &Module<BE>,
        pt_rnx: &CKKSPlaintextRnx<F>,
        prec: CKKS,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: VecZnxRshAddInto<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextRnx<F>: CKKSPlaintextConversion,
    {
        let (pt_glwe, scratch_1) = scratch.take_glwe_plaintext(&GLWEPlaintextLayout {
            n: module.n().into(),
            base2k: self.base2k(),
            k: prec.min_k(self.base2k()),
        });
        let mut pt_znx = attach_meta(pt_glwe, prec);
        pt_rnx.to_znx::<BE>(&mut pt_znx).unwrap();
        self.add_pt_znx_inplace(module, &pt_znx, scratch_1)?;
        Ok(())
    }
}
