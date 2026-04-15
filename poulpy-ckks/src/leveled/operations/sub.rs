//! CKKS ciphertext subtraction.
//!
//! Provides ct-ct, ct-pt in ZNX form, and ct-pt in RNX form, each in
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
    GLWEShift, GLWESub, ScratchTakeCore,
    layouts::{GLWE, GLWEPlaintextLayout, GLWEToRef, LWEInfos},
};
use poulpy_hal::{
    api::{ScratchAvailable, VecZnxRshSub, VecZnxRshTmpBytes},
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};

use anyhow::Result;

pub trait CKKSSubOps {
    fn sub_tmp_bytes<BE: Backend>(module: &Module<BE>) -> usize
    where
        Module<BE>: GLWEShift<BE> + VecZnxRshTmpBytes;

    fn sub<A, B, BE: Backend>(&mut self, module: &Module<BE>, a: &A, b: &B, scratch: &mut Scratch<BE>) -> Result<()>
    where
        Module<BE>: GLWESub + GLWEShift<BE>,
        A: GLWEToRef + LWEInfos + CKKSInfos,
        B: GLWEToRef + LWEInfos + CKKSInfos,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn sub_inplace<A, BE: Backend>(&mut self, module: &Module<BE>, a: &A, scratch: &mut Scratch<BE>) -> Result<()>
    where
        Module<BE>: GLWESub + GLWEShift<BE>,
        A: GLWEToRef + CKKSInfos,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn sub_pt_znx<A, BE: Backend>(
        &mut self,
        module: &Module<BE>,
        a: &A,
        pt_znx: &CKKSPlaintextZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: VecZnxRshSub<BE> + GLWEShift<BE>,
        A: GLWEToRef + LWEInfos + CKKSInfos,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn sub_pt_znx_inplace<BE: Backend>(
        &mut self,
        module: &Module<BE>,
        pt_znx: &CKKSPlaintextZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: VecZnxRshSub<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn sub_pt_rnx<A, F, BE: Backend>(
        &mut self,
        module: &Module<BE>,
        a: &A,
        pt_rnx: &CKKSPlaintextRnx<F>,
        prec: CKKS,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: VecZnxRshSub<BE> + GLWEShift<BE>,
        A: GLWEToRef + LWEInfos + CKKSInfos,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextRnx<F>: CKKSPlaintextConversion;

    fn sub_pt_rnx_inplace<F, BE: Backend>(
        &mut self,
        module: &Module<BE>,
        pt_rnx: &CKKSPlaintextRnx<F>,
        prec: CKKS,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: VecZnxRshSub<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextRnx<F>: CKKSPlaintextConversion;
}

impl<D: DataMut> CKKSSubOps for GLWE<D, CKKS> {
    fn sub_tmp_bytes<BE: Backend>(module: &Module<BE>) -> usize
    where
        Module<BE>: GLWEShift<BE> + VecZnxRshTmpBytes,
    {
        module.glwe_shift_tmp_bytes().max(module.vec_znx_rsh_tmp_bytes())
    }

    fn sub<A, B, BE: Backend>(&mut self, module: &Module<BE>, a: &A, b: &B, scratch: &mut Scratch<BE>) -> Result<()>
    where
        Module<BE>: GLWESub + GLWEShift<BE>,
        A: GLWEToRef + LWEInfos + CKKSInfos,
        B: GLWEToRef + LWEInfos + CKKSInfos,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        // If the destination has less precision than the aligned inputs, shift
        // the computation down by `offset` bits before writing the result.
        let offset = self.offset_binary(a, b);

        if offset == 0 && a.log_hom_rem() == b.log_hom_rem() {
            module.glwe_sub(self, a, b);
        } else if a.log_hom_rem() <= b.log_hom_rem() {
            module.glwe_lsh(self, a, offset, scratch);
            module.glwe_lsh_sub(self, b, b.log_hom_rem() - a.log_hom_rem() + offset, scratch);
        } else {
            module.glwe_lsh(self, a, a.log_hom_rem() - b.log_hom_rem() + offset, scratch);
            module.glwe_lsh_sub(self, b, offset, scratch);
        }

        self.set_log_decimal(a.log_decimal().max(b.log_decimal()))?;
        self.set_log_hom_rem(a.log_hom_rem().min(b.log_hom_rem()) - offset)?;

        Ok(())
    }

    fn sub_inplace<A, BE: Backend>(&mut self, module: &Module<BE>, a: &A, scratch: &mut Scratch<BE>) -> Result<()>
    where
        Module<BE>: GLWESub + GLWEShift<BE>,
        A: GLWEToRef + CKKSInfos,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let self_log_hom_rem = self.log_hom_rem();

        if self_log_hom_rem < a.log_hom_rem() {
            module.glwe_lsh_sub(self, a, a.log_hom_rem() - self_log_hom_rem, scratch);
        } else if self_log_hom_rem > a.log_hom_rem() {
            module.glwe_lsh_inplace(self, self_log_hom_rem - a.log_hom_rem(), scratch);
            module.glwe_sub_inplace(self, a);
        } else {
            module.glwe_sub_inplace(self, a);
        }

        self.set_log_hom_rem(self_log_hom_rem.min(a.log_hom_rem()))?;

        Ok(())
    }

    fn sub_pt_znx<A, BE: Backend>(
        &mut self,
        module: &Module<BE>,
        a: &A,
        pt_znx: &CKKSPlaintextZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: VecZnxRshSub<BE> + GLWEShift<BE>,
        A: GLWEToRef + LWEInfos + CKKSInfos,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let offset = self.offset_unary(a);
        module.glwe_lsh(self, a, offset, scratch);
        self.meta = a.meta();
        self.set_log_hom_rem(a.log_hom_rem() - offset)?;
        self.sub_pt_znx_inplace(module, pt_znx, scratch)?;
        Ok(())
    }

    fn sub_pt_znx_inplace<BE: Backend>(
        &mut self,
        module: &Module<BE>,
        pt_znx: &CKKSPlaintextZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: VecZnxRshSub<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        module.ckks_sub_pt_znx(self, pt_znx, scratch)?;
        Ok(())
    }

    fn sub_pt_rnx<A, F, BE: Backend>(
        &mut self,
        module: &Module<BE>,
        a: &A,
        pt_rnx: &CKKSPlaintextRnx<F>,
        prec: CKKS,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: VecZnxRshSub<BE> + GLWEShift<BE>,
        A: GLWEToRef + LWEInfos + CKKSInfos,
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
        self.sub_pt_znx(module, a, &pt_znx, scratch_1)?;
        Ok(())
    }

    fn sub_pt_rnx_inplace<F, BE: Backend>(
        &mut self,
        module: &Module<BE>,
        pt_rnx: &CKKSPlaintextRnx<F>,
        prec: CKKS,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: VecZnxRshSub<BE>,
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
        self.sub_pt_znx_inplace(module, &pt_znx, scratch_1)?;
        Ok(())
    }
}
