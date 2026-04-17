//! CKKS ciphertext subtraction.
//!
//! Provides ct-ct, ct-pt in ZNX form, and ct-pt in RNX form, each in
//! out-of-place and in-place form.

use crate::{
    CKKS, CKKSInfos, checked_log_hom_rem_sub,
    layouts::{
        CKKSCiphertext,
        ciphertext::CKKSOffset,
        plaintext::{CKKSPlaintextConversion, CKKSPlaintextRnx, CKKSPlaintextZnx},
    },
    leveled::operations::pt_znx::CKKSPlaintextZnxOps,
};
use anyhow::Result;
use poulpy_core::{
    GLWEShift, GLWESub, ScratchTakeCore,
    layouts::{GLWEPlaintextLayout, LWEInfos},
};
use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, VecZnxRshSub, VecZnxRshTmpBytes},
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};

pub trait CKKSSubOps<BE: Backend> {
    fn ckks_sub_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE> + VecZnxRshTmpBytes;

    fn ckks_sub(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        b: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWESub + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_sub_inplace(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWESub + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_sub_pt_znx(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_znx: &CKKSPlaintextZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshSub<BE> + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_sub_pt_znx_inplace(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_znx: &CKKSPlaintextZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshSub<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_sub_pt_rnx<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_rnx: &CKKSPlaintextRnx<F>,
        prec: CKKS,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + VecZnxRshSub<BE> + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextRnx<F>: CKKSPlaintextConversion;

    fn ckks_sub_pt_rnx_inplace<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_rnx: &CKKSPlaintextRnx<F>,
        prec: CKKS,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + VecZnxRshSub<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextRnx<F>: CKKSPlaintextConversion;
}

#[doc(hidden)]
pub trait CKKSSubOpsDefault<BE: Backend> {
    fn ckks_sub_tmp_bytes_default(&self) -> usize
    where
        Self: GLWEShift<BE> + VecZnxRshTmpBytes,
    {
        self.glwe_shift_tmp_bytes().max(self.vec_znx_rsh_tmp_bytes())
    }

    fn ckks_sub_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        b: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWESub + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        // If the destination has less precision than the aligned inputs, shift
        // the computation down by `offset` bits before writing the result.
        let offset = dst.offset_binary(a, b);

        if offset == 0 && a.log_hom_rem() == b.log_hom_rem() {
            self.glwe_sub(dst, a, b);
        } else if a.log_hom_rem() <= b.log_hom_rem() {
            self.glwe_lsh(dst, a, offset, scratch);
            self.glwe_lsh_sub(dst, b, b.log_hom_rem() - a.log_hom_rem() + offset, scratch);
        } else {
            self.glwe_lsh(dst, a, a.log_hom_rem() - b.log_hom_rem() + offset, scratch);
            self.glwe_lsh_sub(dst, b, offset, scratch);
        }

        let log_hom_rem = checked_log_hom_rem_sub("sub", a.log_hom_rem().min(b.log_hom_rem()), offset)?;
        dst.meta.log_decimal = a.log_decimal().max(b.log_decimal());
        dst.meta.log_hom_rem = log_hom_rem;

        Ok(())
    }

    fn ckks_sub_inplace_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWESub + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let dst_log_hom_rem = dst.log_hom_rem();

        if dst_log_hom_rem < a.log_hom_rem() {
            self.glwe_lsh_sub(dst, a, a.log_hom_rem() - dst_log_hom_rem, scratch);
        } else if dst_log_hom_rem > a.log_hom_rem() {
            self.glwe_lsh_inplace(dst, dst_log_hom_rem - a.log_hom_rem(), scratch);
            self.glwe_sub_inplace(dst, a);
        } else {
            self.glwe_sub_inplace(dst, a);
        }

        dst.meta.log_hom_rem = dst_log_hom_rem.min(a.log_hom_rem());

        Ok(())
    }

    fn ckks_sub_pt_znx_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_znx: &CKKSPlaintextZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshSub<BE> + GLWEShift<BE> + CKKSPlaintextZnxOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let offset = dst.offset_unary(a);
        self.glwe_lsh(dst, a, offset, scratch);
        dst.meta = a.meta();
        dst.meta.log_hom_rem = checked_log_hom_rem_sub("sub_pt_znx", a.log_hom_rem(), offset)?;
        self.ckks_sub_pt_znx_inplace_default(dst, pt_znx, scratch)?;
        Ok(())
    }

    fn ckks_sub_pt_znx_inplace_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_znx: &CKKSPlaintextZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshSub<BE> + CKKSPlaintextZnxOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        CKKSPlaintextZnxOps::ckks_sub_pt_znx(self, dst, pt_znx, scratch)?;
        Ok(())
    }

    fn ckks_sub_pt_rnx_default<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_rnx: &CKKSPlaintextRnx<F>,
        prec: CKKS,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + VecZnxRshSub<BE> + GLWEShift<BE> + CKKSPlaintextZnxOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextRnx<F>: CKKSPlaintextConversion,
    {
        let (pt_glwe, scratch_1) = scratch.take_glwe_plaintext(&GLWEPlaintextLayout {
            n: self.n().into(),
            base2k: dst.base2k(),
            k: prec.min_k(dst.base2k()),
        });
        let mut pt_znx = CKKSPlaintextZnx::from_plaintext_with_meta(pt_glwe, prec);
        pt_rnx.to_znx::<BE>(&mut pt_znx)?;
        self.ckks_sub_pt_znx_default(dst, a, &pt_znx, scratch_1)?;
        Ok(())
    }

    fn ckks_sub_pt_rnx_inplace_default<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_rnx: &CKKSPlaintextRnx<F>,
        prec: CKKS,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + VecZnxRshSub<BE> + CKKSPlaintextZnxOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextRnx<F>: CKKSPlaintextConversion,
    {
        let (pt_glwe, scratch_1) = scratch.take_glwe_plaintext(&GLWEPlaintextLayout {
            n: self.n().into(),
            base2k: dst.base2k(),
            k: prec.min_k(dst.base2k()),
        });
        let mut pt_znx = CKKSPlaintextZnx::from_plaintext_with_meta(pt_glwe, prec);
        pt_rnx.to_znx::<BE>(&mut pt_znx)?;
        self.ckks_sub_pt_znx_inplace_default(dst, &pt_znx, scratch_1)?;
        Ok(())
    }
}

impl<BE: Backend> CKKSSubOpsDefault<BE> for Module<BE> {}

impl<BE: Backend> CKKSSubOps<BE> for Module<BE>
where
    Module<BE>: CKKSSubOpsDefault<BE>,
{
    fn ckks_sub_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE> + VecZnxRshTmpBytes,
    {
        self.ckks_sub_tmp_bytes_default()
    }

    fn ckks_sub(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        b: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWESub + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        self.ckks_sub_default(dst, a, b, scratch)
    }

    fn ckks_sub_inplace(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWESub + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        self.ckks_sub_inplace_default(dst, a, scratch)
    }

    fn ckks_sub_pt_znx(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_znx: &CKKSPlaintextZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshSub<BE> + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        self.ckks_sub_pt_znx_default(dst, a, pt_znx, scratch)
    }

    fn ckks_sub_pt_znx_inplace(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_znx: &CKKSPlaintextZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshSub<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        self.ckks_sub_pt_znx_inplace_default(dst, pt_znx, scratch)
    }

    fn ckks_sub_pt_rnx<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_rnx: &CKKSPlaintextRnx<F>,
        prec: CKKS,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + VecZnxRshSub<BE> + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextRnx<F>: CKKSPlaintextConversion,
    {
        self.ckks_sub_pt_rnx_default(dst, a, pt_rnx, prec, scratch)
    }

    fn ckks_sub_pt_rnx_inplace<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_rnx: &CKKSPlaintextRnx<F>,
        prec: CKKS,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + VecZnxRshSub<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextRnx<F>: CKKSPlaintextConversion,
    {
        self.ckks_sub_pt_rnx_inplace_default(dst, pt_rnx, prec, scratch)
    }
}
