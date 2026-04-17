//! CKKS ciphertext multiplication.

use crate::{
    CKKS, CKKSInfos, checked_log_hom_rem_sub, checked_mul_ct_log_hom_rem,
    error::checked_mul_pt_log_hom_rem,
    layouts::{
        CKKSCiphertext,
        plaintext::{CKKSPlaintextConversion, CKKSPlaintextRnx, CKKSPlaintextZnx},
    },
};
use anyhow::Result;
use poulpy_core::{
    GLWEMulPlain, GLWETensoring, ScratchTakeCore,
    layouts::{
        GGLWEInfos, GLWEInfos, GLWELayout, GLWEPlaintext, GLWEPlaintextLayout, GLWETensor, GLWETensorKeyPrepared, GLWEToMut,
        GLWEToRef, LWEInfos, TorusPrecision,
    },
};
use poulpy_hal::{
    api::{ModuleN, ScratchAvailable},
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};

pub trait CKKSMulOps<BE: Backend> {
    fn ckks_mul_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Self: GLWETensoring<BE>;

    fn ckks_square_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Self: GLWETensoring<BE>;

    fn ckks_mul_pt_znx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKS) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulPlain<BE>;

    fn ckks_mul_pt_rnx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKS) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: ModuleN + GLWEMulPlain<BE>;

    fn ckks_mul(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        b: &CKKSCiphertext<impl DataRef>,
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWETensoring<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_mul_inplace(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWETensoring<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_square(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWETensoring<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_square_inplace(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWETensoring<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_mul_pt_znx(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_znx: &CKKSPlaintextZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEMulPlain<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_mul_pt_znx_inplace(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_znx: &CKKSPlaintextZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEMulPlain<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_mul_pt_rnx<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_rnx: &CKKSPlaintextRnx<F>,
        prec: CKKS,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + GLWEMulPlain<BE>,
        CKKSPlaintextRnx<F>: CKKSPlaintextConversion,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_mul_pt_rnx_inplace<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_rnx: &CKKSPlaintextRnx<F>,
        prec: CKKS,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + GLWEMulPlain<BE>,
        CKKSPlaintextRnx<F>: CKKSPlaintextConversion,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;
}

#[doc(hidden)]
pub trait CKKSMulOpsDefault<BE: Backend> {
    fn ckks_mul_tmp_bytes_default<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Self: GLWETensoring<BE>,
    {
        let glwe_layout = GLWELayout {
            n: res.n(),
            base2k: res.base2k(),
            k: TorusPrecision(res.max_k().as_u32()),
            rank: res.rank(),
        };

        let lvl_0 = GLWETensor::bytes_of_from_infos(&glwe_layout);
        let lvl_1 = self
            .glwe_tensor_apply_tmp_bytes(&glwe_layout, res, res)
            .max(self.glwe_tensor_relinearize_tmp_bytes(res, &glwe_layout, tsk));

        lvl_0 + lvl_1
    }

    fn ckks_mul_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        b: &CKKSCiphertext<impl DataRef>,
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWETensoring<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let (res_log_hom_rem, res_log_decimal, cnv_offset) = get_mul_ct_params(dst, a, b)?;

        let tensor_layout = GLWELayout {
            n: dst.n(),
            base2k: dst.base2k(),
            k: a.max_k().max(b.max_k()),
            rank: dst.rank(),
        };

        let (mut tmp, scratch_1) = scratch.take_glwe_tensor(&tensor_layout);

        let a_ref = a.to_ref();
        let b_ref = b.to_ref();
        self.glwe_tensor_apply(
            cnv_offset,
            &mut tmp,
            &a_ref,
            a.effective_k(),
            &b_ref,
            b.effective_k(),
            scratch_1,
        );

        let mut dst_view = dst.to_mut();
        self.glwe_tensor_relinearize(&mut dst_view, &tmp, tsk, tsk.size(), scratch_1);

        dst.meta.log_hom_rem = res_log_hom_rem;
        dst.meta.log_decimal = res_log_decimal;

        Ok(())
    }

    fn ckks_mul_inplace_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWETensoring<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let (res_log_hom_rem, res_log_decimal, cnv_offset) = get_mul_ct_params(dst, dst, a)?;

        let tensor_layout = GLWELayout {
            n: dst.n(),
            base2k: dst.base2k(),
            k: dst.max_k().max(a.max_k()),
            rank: dst.rank(),
        };

        let (mut tmp, scratch_1) = scratch.take_glwe_tensor(&tensor_layout);

        let dst_ref = dst.to_ref();
        let a_ref = a.to_ref();
        self.glwe_tensor_apply(
            cnv_offset,
            &mut tmp,
            &dst_ref,
            dst.effective_k(),
            &a_ref,
            a.effective_k(),
            scratch_1,
        );

        let mut dst_view = dst.to_mut();
        self.glwe_tensor_relinearize(&mut dst_view, &tmp, tsk, tsk.size(), scratch_1);

        dst.meta.log_hom_rem = res_log_hom_rem;
        dst.meta.log_decimal = res_log_decimal;

        Ok(())
    }

    fn ckks_square_tmp_bytes_default<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Self: GLWETensoring<BE>,
    {
        let glwe_layout = GLWELayout {
            n: res.n(),
            base2k: res.base2k(),
            k: TorusPrecision(res.max_k().as_u32()),
            rank: res.rank(),
        };

        let lvl_0 = GLWETensor::bytes_of_from_infos(&glwe_layout);
        let lvl_1 = self
            .glwe_tensor_square_apply_tmp_bytes(&glwe_layout, res)
            .max(self.glwe_tensor_relinearize_tmp_bytes(res, &glwe_layout, tsk));

        lvl_0 + lvl_1
    }

    fn ckks_square_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWETensoring<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let (res_log_hom_rem, res_log_decimal, cnv_offset) = get_mul_ct_params(dst, a, a)?;

        let tensor_layout = GLWELayout {
            n: dst.n(),
            base2k: dst.base2k(),
            k: a.max_k(),
            rank: dst.rank(),
        };

        let (mut tmp, scratch_1) = scratch.take_glwe_tensor(&tensor_layout);

        let a_ref = a.to_ref();
        self.glwe_tensor_square_apply(cnv_offset, &mut tmp, &a_ref, a.effective_k(), scratch_1);

        let mut dst_view = dst.to_mut();
        self.glwe_tensor_relinearize(&mut dst_view, &tmp, tsk, tsk.size(), scratch_1);

        dst.meta.log_hom_rem = res_log_hom_rem;
        dst.meta.log_decimal = res_log_decimal;
        Ok(())
    }

    fn ckks_square_inplace_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWETensoring<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let (res_log_hom_rem, res_log_decimal, cnv_offset) = get_mul_ct_params(dst, dst, dst)?;

        let tensor_layout = GLWELayout {
            n: dst.n(),
            base2k: dst.base2k(),
            k: dst.max_k(),
            rank: dst.rank(),
        };

        let (mut tmp, scratch_1) = scratch.take_glwe_tensor(&tensor_layout);

        self.glwe_tensor_square_apply(cnv_offset, &mut tmp, &dst.to_ref(), dst.effective_k(), scratch_1);

        let mut dst_view = dst.to_mut();
        self.glwe_tensor_relinearize(&mut dst_view, &tmp, tsk, tsk.size(), scratch_1);

        dst.meta.log_hom_rem = res_log_hom_rem;
        dst.meta.log_decimal = res_log_decimal;
        Ok(())
    }

    fn ckks_mul_pt_znx_tmp_bytes_default<R, A>(&self, res: &R, a: &A, b: &CKKS) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulPlain<BE>,
    {
        let b_infos = GLWEPlaintextLayout {
            n: res.n(),
            base2k: res.base2k(),
            k: b.min_k(res.base2k()),
        };
        self.glwe_mul_plain_tmp_bytes(res, a, &b_infos)
    }

    fn ckks_mul_pt_rnx_tmp_bytes_default<R, A>(&self, res: &R, a: &A, b: &CKKS) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: ModuleN + GLWEMulPlain<BE>,
    {
        let b_infos = GLWEPlaintextLayout {
            n: self.n().into(),
            base2k: res.base2k(),
            k: b.min_k(res.base2k()),
        };
        GLWEPlaintext::<Vec<u8>, ()>::bytes_of_from_infos(&b_infos) + self.glwe_mul_plain_tmp_bytes(res, a, &b_infos)
    }

    fn ckks_mul_pt_znx_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_znx: &CKKSPlaintextZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEMulPlain<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let (res_log_hom_rem, res_log_decimal, cnv_offset) = get_mul_pt_params(dst, a, pt_znx)?;

        self.glwe_mul_plain(
            cnv_offset,
            &mut dst.to_mut(),
            &a.to_ref(),
            a.effective_k(),
            pt_znx,
            pt_znx.max_k().as_usize(),
            scratch,
        );

        dst.meta.log_hom_rem = res_log_hom_rem;
        dst.meta.log_decimal = res_log_decimal;

        Ok(())
    }

    fn ckks_mul_pt_znx_inplace_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_znx: &CKKSPlaintextZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEMulPlain<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let (res_log_hom_rem, res_log_decimal, cnv_offset) = get_mul_pt_params(dst, dst, pt_znx)?;

        let dst_effective_k = dst.effective_k();

        self.glwe_mul_plain_inplace(
            cnv_offset,
            &mut dst.to_mut(),
            dst_effective_k,
            pt_znx,
            pt_znx.max_k().as_usize(),
            scratch,
        );

        dst.meta.log_hom_rem = res_log_hom_rem;
        dst.meta.log_decimal = res_log_decimal;

        Ok(())
    }

    fn ckks_mul_pt_rnx_default<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_rnx: &CKKSPlaintextRnx<F>,
        prec: CKKS,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + GLWEMulPlain<BE>,
        CKKSPlaintextRnx<F>: CKKSPlaintextConversion,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let (pt_glwe, scratch_1) = scratch.take_glwe_plaintext(&GLWEPlaintextLayout {
            n: self.n().into(),
            base2k: dst.base2k(),
            k: prec.min_k(dst.base2k()),
        });

        let mut pt_znx = CKKSPlaintextZnx::from_plaintext_with_meta(pt_glwe, prec);
        pt_rnx.to_znx::<BE>(&mut pt_znx)?;
        self.ckks_mul_pt_znx_default(dst, a, &pt_znx, scratch_1)
    }

    fn ckks_mul_pt_rnx_inplace_default<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_rnx: &CKKSPlaintextRnx<F>,
        prec: CKKS,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + GLWEMulPlain<BE>,
        CKKSPlaintextRnx<F>: CKKSPlaintextConversion,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let (pt_glwe, scratch_1) = scratch.take_glwe_plaintext(&GLWEPlaintextLayout {
            n: self.n().into(),
            base2k: dst.base2k(),
            k: prec.min_k(dst.base2k()),
        });

        let mut pt_znx = CKKSPlaintextZnx::from_plaintext_with_meta(pt_glwe, prec);
        pt_rnx.to_znx::<BE>(&mut pt_znx)?;
        self.ckks_mul_pt_znx_inplace_default(dst, &pt_znx, scratch_1)
    }
}

impl<BE: Backend> CKKSMulOpsDefault<BE> for Module<BE> {}

impl<BE: Backend> CKKSMulOps<BE> for Module<BE>
where
    Module<BE>: CKKSMulOpsDefault<BE>,
{
    fn ckks_mul_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Self: GLWETensoring<BE>,
    {
        self.ckks_mul_tmp_bytes_default(res, tsk)
    }

    fn ckks_square_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Self: GLWETensoring<BE>,
    {
        self.ckks_square_tmp_bytes_default(res, tsk)
    }

    fn ckks_mul_pt_znx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKS) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulPlain<BE>,
    {
        self.ckks_mul_pt_znx_tmp_bytes_default(res, a, b)
    }

    fn ckks_mul_pt_rnx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKS) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: ModuleN + GLWEMulPlain<BE>,
    {
        self.ckks_mul_pt_rnx_tmp_bytes_default(res, a, b)
    }

    fn ckks_mul(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        b: &CKKSCiphertext<impl DataRef>,
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWETensoring<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        self.ckks_mul_default(dst, a, b, tsk, scratch)
    }

    fn ckks_mul_inplace(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWETensoring<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        self.ckks_mul_inplace_default(dst, a, tsk, scratch)
    }

    fn ckks_square(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWETensoring<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        self.ckks_square_default(dst, a, tsk, scratch)
    }

    fn ckks_square_inplace(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWETensoring<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        self.ckks_square_inplace_default(dst, tsk, scratch)
    }

    fn ckks_mul_pt_znx(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_znx: &CKKSPlaintextZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEMulPlain<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        self.ckks_mul_pt_znx_default(dst, a, pt_znx, scratch)
    }

    fn ckks_mul_pt_znx_inplace(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_znx: &CKKSPlaintextZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEMulPlain<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        self.ckks_mul_pt_znx_inplace_default(dst, pt_znx, scratch)
    }

    fn ckks_mul_pt_rnx<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_rnx: &CKKSPlaintextRnx<F>,
        prec: CKKS,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + GLWEMulPlain<BE>,
        CKKSPlaintextRnx<F>: CKKSPlaintextConversion,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        self.ckks_mul_pt_rnx_default(dst, a, pt_rnx, prec, scratch)
    }

    fn ckks_mul_pt_rnx_inplace<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_rnx: &CKKSPlaintextRnx<F>,
        prec: CKKS,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + GLWEMulPlain<BE>,
        CKKSPlaintextRnx<F>: CKKSPlaintextConversion,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        self.ckks_mul_pt_rnx_inplace_default(dst, pt_rnx, prec, scratch)
    }
}

fn get_mul_ct_params<R, A, B>(res: &R, a: &A, b: &B) -> Result<(usize, usize, usize)>
where
    R: LWEInfos + CKKSInfos,
    A: LWEInfos + CKKSInfos,
    B: LWEInfos + CKKSInfos,
{
    let res_log_hom_rem = checked_mul_ct_log_hom_rem("mul", a.log_hom_rem(), b.log_hom_rem(), a.log_decimal(), b.log_decimal())?;
    let res_log_decimal = a.log_decimal().max(b.log_decimal());

    let res_offset = (res_log_hom_rem + res_log_decimal).saturating_sub(res.max_k().as_usize());

    let cnv_offset = a.effective_k().max(b.effective_k()) + res_offset;

    Ok((
        checked_log_hom_rem_sub("mul", res_log_hom_rem, res_offset)?,
        res_log_decimal,
        cnv_offset,
    ))
}

fn get_mul_pt_params<R, A, B>(res: &R, a: &A, b: &B) -> Result<(usize, usize, usize)>
where
    R: LWEInfos + CKKSInfos,
    A: LWEInfos + CKKSInfos,
    B: LWEInfos + CKKSInfos,
{
    let res_log_hom_rem = checked_mul_pt_log_hom_rem("mul", a.log_hom_rem(), b.log_hom_rem(), a.log_decimal(), b.log_decimal())?;
    let res_log_decimal = a.log_decimal();

    let res_offset = (res_log_hom_rem + res_log_decimal).saturating_sub(res.max_k().as_usize());

    let cnv_offset = b.max_k().as_usize() + res_offset;

    Ok((
        checked_log_hom_rem_sub("mul", res_log_hom_rem, res_offset)?,
        res_log_decimal,
        cnv_offset,
    ))
}
