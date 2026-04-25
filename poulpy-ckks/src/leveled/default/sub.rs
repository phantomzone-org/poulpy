use anyhow::Result;
use poulpy_core::{
    GLWENormalize, GLWEShift, GLWESub, ScratchTakeCore,
    layouts::{GLWEInfos, GLWEPlaintext, GLWEPlaintextLayout, LWEInfos},
};
use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, VecZnxRshSub, VecZnxRshTmpBytes},
    layouts::{Backend, DataMut, DataRef, Module, Scratch, ZnxViewMut},
};

use crate::{
    CKKSInfos, CKKSMeta, checked_log_hom_rem_sub,
    layouts::{
        CKKSCiphertext,
        ciphertext::CKKSOffset,
        plaintext::{
            CKKSConstPlaintextConversion, CKKSPlaintextConversion, CKKSPlaintextCstRnx, CKKSPlaintextCstZnx, CKKSPlaintextVecRnx,
            CKKSPlaintextVecZnx,
        },
    },
    leveled::default::CKKSPlaintextZnxDefault,
};

pub(crate) trait CKKSSubDefault<BE: Backend> {
    fn ckks_sub_tmp_bytes_default(&self) -> usize
    where
        Self: GLWEShift<BE> + GLWENormalize<BE> + VecZnxRshTmpBytes,
    {
        self.glwe_shift_tmp_bytes()
            .max(self.vec_znx_rsh_tmp_bytes())
            .max(self.glwe_normalize_tmp_bytes())
    }

    fn ckks_sub_pt_vec_znx_tmp_bytes_default(&self) -> usize
    where
        Self: GLWEShift<BE> + GLWENormalize<BE> + VecZnxRshTmpBytes,
    {
        self.ckks_sub_tmp_bytes_default()
    }

    fn ckks_sub_pt_vec_rnx_tmp_bytes_default<R, A>(&self, res: &R, _a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: ModuleN + GLWEShift<BE> + GLWENormalize<BE> + VecZnxRshTmpBytes,
    {
        let b_infos = GLWEPlaintextLayout {
            n: self.n().into(),
            base2k: res.base2k(),
            k: b.min_k(res.base2k()),
        };
        GLWEPlaintext::<Vec<u8>>::bytes_of_from_infos(&b_infos) + self.ckks_sub_pt_vec_znx_tmp_bytes_default()
    }

    fn ckks_sub_const_tmp_bytes_default(&self) -> usize
    where
        Self: GLWEShift<BE> + GLWENormalize<BE>,
    {
        self.glwe_shift_tmp_bytes().max(self.glwe_normalize_tmp_bytes())
    }

    fn ckks_sub_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        b: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWESub + GLWEShift<BE> + GLWENormalize<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        self.ckks_sub_without_normalization_default(dst, a, b, scratch)?;
        self.glwe_normalize_inplace(dst, scratch);
        Ok(())
    }

    fn ckks_sub_without_normalization_default(
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
        Self: GLWESub + GLWEShift<BE> + GLWENormalize<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        self.ckks_sub_inplace_without_normalization_default(dst, a, scratch)?;
        self.glwe_normalize_inplace(dst, scratch);
        Ok(())
    }

    fn ckks_sub_inplace_without_normalization_default(
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

    fn ckks_sub_pt_vec_znx_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshSub<BE> + GLWEShift<BE> + GLWENormalize<BE> + CKKSPlaintextZnxDefault<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        self.ckks_sub_pt_vec_znx_without_normalization_default(dst, a, pt_znx, scratch)?;
        self.glwe_normalize_inplace(dst, scratch);
        Ok(())
    }

    fn ckks_sub_pt_vec_znx_without_normalization_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshSub<BE> + GLWEShift<BE> + CKKSPlaintextZnxDefault<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let offset = dst.offset_unary(a);
        self.glwe_lsh(dst, a, offset, scratch);
        dst.meta = a.meta();
        dst.meta.log_hom_rem = checked_log_hom_rem_sub("sub_pt_znx", a.log_hom_rem(), offset)?;
        self.ckks_sub_pt_vec_znx_inplace_without_normalization_default(dst, pt_znx, scratch)?;
        Ok(())
    }

    fn ckks_sub_pt_vec_znx_inplace_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshSub<BE> + GLWENormalize<BE> + CKKSPlaintextZnxDefault<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        self.ckks_sub_pt_vec_znx_inplace_without_normalization_default(dst, pt_znx, scratch)?;
        self.glwe_normalize_inplace(dst, scratch);
        Ok(())
    }

    fn ckks_sub_pt_vec_znx_inplace_without_normalization_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshSub<BE> + CKKSPlaintextZnxDefault<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        CKKSPlaintextZnxDefault::ckks_sub_pt_vec_znx_default(self, dst, pt_znx, scratch)?;
        Ok(())
    }

    fn ckks_sub_pt_vec_rnx_default<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + VecZnxRshSub<BE> + GLWEShift<BE> + GLWENormalize<BE> + CKKSPlaintextZnxDefault<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
    {
        self.ckks_sub_pt_vec_rnx_without_normalization_default(dst, a, pt_rnx, prec, scratch)?;
        self.glwe_normalize_inplace(dst, scratch);
        Ok(())
    }

    fn ckks_sub_pt_vec_rnx_without_normalization_default<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + VecZnxRshSub<BE> + GLWEShift<BE> + CKKSPlaintextZnxDefault<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
    {
        let (pt_glwe, scratch_1) = scratch.take_glwe_plaintext(&GLWEPlaintextLayout {
            n: self.n().into(),
            base2k: dst.base2k(),
            k: prec.min_k(dst.base2k()),
        });
        let mut pt_znx = CKKSPlaintextVecZnx::from_plaintext_with_meta(pt_glwe, prec);
        pt_rnx.to_znx(&mut pt_znx)?;
        CKKSSubDefault::ckks_sub_pt_vec_znx_without_normalization_default(self, dst, a, &pt_znx, scratch_1)?;
        Ok(())
    }

    fn ckks_sub_pt_vec_rnx_inplace_default<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + VecZnxRshSub<BE> + GLWENormalize<BE> + CKKSPlaintextZnxDefault<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
    {
        self.ckks_sub_pt_vec_rnx_inplace_without_normalization_default(dst, pt_rnx, prec, scratch)?;
        self.glwe_normalize_inplace(dst, scratch);
        Ok(())
    }

    fn ckks_sub_pt_vec_rnx_inplace_without_normalization_default<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + VecZnxRshSub<BE> + CKKSPlaintextZnxDefault<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
    {
        let (pt_glwe, scratch_1) = scratch.take_glwe_plaintext(&GLWEPlaintextLayout {
            n: self.n().into(),
            base2k: dst.base2k(),
            k: prec.min_k(dst.base2k()),
        });
        let mut pt_znx = CKKSPlaintextVecZnx::from_plaintext_with_meta(pt_glwe, prec);
        pt_rnx.to_znx(&mut pt_znx)?;
        self.ckks_sub_pt_vec_znx_inplace_without_normalization_default(dst, &pt_znx, scratch_1)?;
        Ok(())
    }

    fn ckks_sub_const_znx_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE> + GLWENormalize<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        self.ckks_sub_const_znx_without_normalization_default(dst, a, cst_znx, scratch)?;
        self.glwe_normalize_inplace(dst, scratch);
        Ok(())
    }

    fn ckks_sub_const_znx_without_normalization_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let offset = dst.offset_unary(a);
        self.glwe_lsh(dst, a, offset, scratch);
        dst.meta = a.meta();
        dst.meta.log_hom_rem = checked_log_hom_rem_sub("sub_const_znx", a.log_hom_rem(), offset)?;
        self.ckks_sub_const_znx_inplace_without_normalization_default(dst, cst_znx, scratch)
    }

    fn ckks_sub_const_znx_inplace_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWENormalize<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        self.ckks_sub_const_znx_inplace_without_normalization_default(dst, cst_znx, scratch)?;
        self.glwe_normalize_inplace(dst, scratch);
        Ok(())
    }

    fn ckks_sub_const_znx_inplace_without_normalization_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        cst_znx: &CKKSPlaintextCstZnx,
        _scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        if cst_znx.re().is_none() && cst_znx.im().is_none() {
            return Ok(());
        }

        let _offset = crate::ensure_plaintext_alignment(
            "ckks_sub_const_znx",
            dst.log_hom_rem(),
            cst_znx.log_decimal(),
            cst_znx.effective_k(),
        )?;
        let n = dst.n().as_usize();
        if let Some(coeff) = cst_znx.re() {
            for (limb, digit) in coeff.iter().enumerate() {
                dst.data_mut().at_mut(0, limb)[0] -= *digit;
            }
        }
        if let Some(coeff) = cst_znx.im() {
            for (limb, digit) in coeff.iter().enumerate() {
                dst.data_mut().at_mut(0, limb)[n / 2] -= *digit;
            }
        }
        Ok(())
    }

    fn ckks_sub_const_rnx_default<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE> + GLWENormalize<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion,
    {
        self.ckks_sub_const_rnx_without_normalization_default(dst, a, cst_rnx, prec, scratch)?;
        self.glwe_normalize_inplace(dst, scratch);
        Ok(())
    }

    fn ckks_sub_const_rnx_without_normalization_default<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion,
    {
        let offset = dst.offset_unary(a);
        if cst_rnx.re().is_none() && cst_rnx.im().is_none() {
            self.glwe_lsh(dst, a, offset, scratch);
            dst.meta = a.meta();
            dst.meta.log_hom_rem = checked_log_hom_rem_sub("sub_const_rnx", a.log_hom_rem(), offset)?;
            return Ok(());
        }

        let res_log_hom_rem = checked_log_hom_rem_sub("sub_const_rnx", a.log_hom_rem(), offset)?;
        let cst_znx = cst_rnx.to_znx_at_k(
            dst.base2k(),
            res_log_hom_rem
                .checked_add(prec.log_decimal)
                .expect("aligned precision overflow"),
            prec.log_decimal,
        )?;
        self.ckks_sub_const_znx_without_normalization_default(dst, a, &cst_znx, scratch)
    }

    fn ckks_sub_const_rnx_inplace_default<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWENormalize<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion,
    {
        self.ckks_sub_const_rnx_inplace_without_normalization_default(dst, cst_rnx, prec, scratch)?;
        self.glwe_normalize_inplace(dst, scratch);
        Ok(())
    }

    fn ckks_sub_const_rnx_inplace_without_normalization_default<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion,
    {
        if cst_rnx.re().is_none() && cst_rnx.im().is_none() {
            return Ok(());
        }

        let cst_znx = cst_rnx.to_znx_at_k(
            dst.base2k(),
            dst.log_hom_rem()
                .checked_add(prec.log_decimal)
                .expect("aligned precision overflow"),
            prec.log_decimal,
        )?;
        self.ckks_sub_const_znx_inplace_without_normalization_default(dst, &cst_znx, scratch)
    }
}

impl<BE: Backend> CKKSSubDefault<BE> for Module<BE> {}
