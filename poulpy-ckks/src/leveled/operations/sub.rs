//! CKKS ciphertext subtraction.
//!
//! Provides ct-ct, ct-pt in ZNX form, and ct-pt in RNX form, each in
//! out-of-place and in-place form.

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
    leveled::operations::pt_znx::CKKSPlaintextZnxOps,
};
use anyhow::Result;
use poulpy_core::{
    GLWEShift, GLWESub, ScratchTakeCore,
    layouts::{GLWEInfos, GLWEPlaintext, GLWEPlaintextLayout, LWEInfos},
};
use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, VecZnxRshSub, VecZnxRshTmpBytes},
    layouts::{Backend, DataMut, DataRef, Module, Scratch, ZnxViewMut},
};

/// CKKS ciphertext subtraction APIs.
///
/// This trait covers ciphertext-ciphertext subtraction plus plaintext and
/// constant subtraction in both ZNX and RNX forms.
pub trait CKKSSubOps<BE: Backend> {
    /// Returns scratch bytes required by [`Self::ckks_sub`].
    fn ckks_sub_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE> + VecZnxRshTmpBytes;

    /// Returns scratch bytes required by [`Self::ckks_sub_pt_vec_znx`].
    fn ckks_sub_pt_vec_znx_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE> + VecZnxRshTmpBytes;

    /// Computes `dst = a - b`.
    ///
    /// Errors include backend shift/sub failures and metadata truncation caused
    /// by a too-small destination buffer.
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

    /// Computes `dst -= a`.
    fn ckks_sub_inplace(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWESub + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    /// Computes `dst = a - pt_znx` for a quantized vector plaintext.
    ///
    /// Errors include `PlaintextBase2KMismatch` and
    /// `PlaintextAlignmentImpossible`.
    fn ckks_sub_pt_vec_znx(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshSub<BE> + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    /// Computes `dst -= pt_znx` for a quantized vector plaintext.
    fn ckks_sub_pt_vec_znx_inplace(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshSub<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    /// Returns scratch bytes required by [`Self::ckks_sub_pt_vec_rnx`].
    fn ckks_sub_pt_vec_rnx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: ModuleN + GLWEShift<BE> + VecZnxRshTmpBytes;

    /// Computes `dst = a - pt_rnx` for an RNX plaintext vector.
    ///
    /// `prec` describes the plaintext metadata used during conversion.
    fn ckks_sub_pt_vec_rnx<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + VecZnxRshSub<BE> + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion;

    /// Computes `dst -= pt_rnx` for an RNX plaintext vector.
    fn ckks_sub_pt_vec_rnx_inplace<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + VecZnxRshSub<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion;

    /// Returns scratch bytes required by constant-subtraction APIs.
    fn ckks_sub_const_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>;

    /// Computes `dst = a - cst_znx` for a quantized constant plaintext.
    fn ckks_sub_pt_const_znx(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    /// Computes `dst -= cst_znx` for a quantized constant plaintext.
    fn ckks_sub_pt_const_znx_inplace(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    /// Computes `dst = a - cst_rnx` for an RNX constant.
    fn ckks_sub_pt_const_rnx<F>(
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
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion;

    /// Computes `dst -= cst_rnx` for an RNX constant.
    fn ckks_sub_pt_const_rnx_inplace<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion;

    /// Alias of [`Self::ckks_sub_pt_const_znx`].
    fn ckks_sub_const_znx(
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
        self.ckks_sub_pt_const_znx(dst, a, cst_znx, scratch)
    }

    /// Alias of [`Self::ckks_sub_pt_const_znx_inplace`].
    fn ckks_sub_const_znx_inplace(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        self.ckks_sub_pt_const_znx_inplace(dst, cst_znx, scratch)
    }

    /// Alias of [`Self::ckks_sub_pt_const_rnx`].
    fn ckks_sub_const_rnx<F>(
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
        self.ckks_sub_pt_const_rnx(dst, a, cst_rnx, prec, scratch)
    }

    /// Alias of [`Self::ckks_sub_pt_const_rnx_inplace`].
    fn ckks_sub_const_rnx_inplace<F>(
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
        self.ckks_sub_pt_const_rnx_inplace(dst, cst_rnx, prec, scratch)
    }
}

#[doc(hidden)]
pub trait CKKSSubOpsDefault<BE: Backend> {
    fn ckks_sub_tmp_bytes_default(&self) -> usize
    where
        Self: GLWEShift<BE> + VecZnxRshTmpBytes,
    {
        self.glwe_shift_tmp_bytes().max(self.vec_znx_rsh_tmp_bytes())
    }

    fn ckks_sub_pt_vec_znx_tmp_bytes_default(&self) -> usize
    where
        Self: GLWEShift<BE> + VecZnxRshTmpBytes,
    {
        self.ckks_sub_tmp_bytes_default()
    }

    fn ckks_sub_pt_vec_rnx_tmp_bytes_default<R, A>(&self, res: &R, _a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: ModuleN + GLWEShift<BE> + VecZnxRshTmpBytes,
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
        Self: GLWEShift<BE>,
    {
        self.glwe_shift_tmp_bytes()
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

    fn ckks_sub_pt_vec_znx_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
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
        self.ckks_sub_pt_vec_znx_inplace_default(dst, pt_znx, scratch)?;
        Ok(())
    }

    fn ckks_sub_pt_vec_znx_inplace_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshSub<BE> + CKKSPlaintextZnxOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        CKKSPlaintextZnxOps::ckks_sub_pt_vec_znx(self, dst, pt_znx, scratch)?;
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
        Self: ModuleN + VecZnxRshSub<BE> + GLWEShift<BE> + CKKSPlaintextZnxOps<BE>,
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
        self.ckks_sub_pt_vec_znx_default(dst, a, &pt_znx, scratch_1)?;
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
        Self: ModuleN + VecZnxRshSub<BE> + CKKSPlaintextZnxOps<BE>,
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
        self.ckks_sub_pt_vec_znx_inplace_default(dst, &pt_znx, scratch_1)?;
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
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let offset = dst.offset_unary(a);
        self.glwe_lsh(dst, a, offset, scratch);
        dst.meta = a.meta();
        dst.meta.log_hom_rem = checked_log_hom_rem_sub("sub_const_znx", a.log_hom_rem(), offset)?;
        self.ckks_sub_const_znx_inplace_default(dst, cst_znx, scratch)
    }

    fn ckks_sub_const_znx_inplace_default(
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
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion,
    {
        let offset = dst.offset_unary(a);
        let res_log_hom_rem = checked_log_hom_rem_sub("sub_const_rnx", a.log_hom_rem(), offset)?;
        let cst_znx = cst_rnx.to_znx_at_k(
            dst.base2k(),
            res_log_hom_rem
                .checked_add(prec.log_decimal)
                .expect("aligned precision overflow"),
            prec.log_decimal,
        )?;
        self.ckks_sub_const_znx_default(dst, a, &cst_znx, scratch)
    }

    fn ckks_sub_const_rnx_inplace_default<F>(
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
        let cst_znx = cst_rnx.to_znx_at_k(
            dst.base2k(),
            dst.log_hom_rem()
                .checked_add(prec.log_decimal)
                .expect("aligned precision overflow"),
            prec.log_decimal,
        )?;
        self.ckks_sub_const_znx_inplace_default(dst, &cst_znx, scratch)
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

    fn ckks_sub_pt_vec_znx_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE> + VecZnxRshTmpBytes,
    {
        self.ckks_sub_pt_vec_znx_tmp_bytes_default()
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

    fn ckks_sub_pt_vec_znx(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshSub<BE> + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        self.ckks_sub_pt_vec_znx_default(dst, a, pt_znx, scratch)
    }

    fn ckks_sub_pt_vec_znx_inplace(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshSub<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        self.ckks_sub_pt_vec_znx_inplace_default(dst, pt_znx, scratch)
    }

    fn ckks_sub_pt_vec_rnx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: ModuleN + GLWEShift<BE> + VecZnxRshTmpBytes,
    {
        self.ckks_sub_pt_vec_rnx_tmp_bytes_default(res, a, b)
    }

    fn ckks_sub_pt_vec_rnx<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + VecZnxRshSub<BE> + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
    {
        self.ckks_sub_pt_vec_rnx_default(dst, a, pt_rnx, prec, scratch)
    }

    fn ckks_sub_pt_vec_rnx_inplace<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + VecZnxRshSub<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
    {
        self.ckks_sub_pt_vec_rnx_inplace_default(dst, pt_rnx, prec, scratch)
    }

    fn ckks_sub_const_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        self.ckks_sub_const_tmp_bytes_default()
    }

    fn ckks_sub_pt_const_znx(
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
        self.ckks_sub_const_znx_default(dst, a, cst_znx, scratch)
    }

    fn ckks_sub_pt_const_znx_inplace(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        self.ckks_sub_const_znx_inplace_default(dst, cst_znx, scratch)
    }

    fn ckks_sub_pt_const_rnx<F>(
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
        self.ckks_sub_const_rnx_default(dst, a, cst_rnx, prec, scratch)
    }

    fn ckks_sub_pt_const_rnx_inplace<F>(
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
        self.ckks_sub_const_rnx_inplace_default(dst, cst_rnx, prec, scratch)
    }
}
