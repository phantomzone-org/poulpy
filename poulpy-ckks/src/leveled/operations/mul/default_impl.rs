use anyhow::Result;
use poulpy_core::{
    GLWEAdd, GLWEMulConst, GLWEMulPlain, GLWERotate, GLWETensoring, ScratchTakeCore,
    layouts::{
        GGLWEInfos, GLWE, GLWEInfos, GLWELayout, GLWEPlaintext, GLWEPlaintextLayout, GLWETensor, GLWETensorKeyPrepared,
        GLWEToMut, GLWEToRef, LWEInfos, TorusPrecision,
    },
};
use poulpy_hal::{
    api::{ModuleN, ScratchAvailable},
    layouts::{Backend, DataMut, DataRef, Module, Scratch, ZnxZero},
};

use crate::{
    CKKSInfos, CKKSMeta, checked_log_hom_rem_sub, checked_mul_ct_log_hom_rem,
    error::checked_mul_pt_log_hom_rem,
    layouts::{
        CKKSCiphertext,
        plaintext::{
            CKKSConstPlaintextConversion, CKKSPlaintextConversion, CKKSPlaintextCstRnx, CKKSPlaintextCstZnx, CKKSPlaintextVecRnx,
            CKKSPlaintextVecZnx,
        },
    },
};

pub(crate) trait CKKSMulDefault<BE: Backend> {
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

        let sequential = GLWETensor::bytes_of_from_infos(&glwe_layout)
            + self
                .glwe_tensor_apply_tmp_bytes(&glwe_layout, res, res)
                .max(self.glwe_tensor_relinearize_tmp_bytes(res, &glwe_layout, tsk));

        let fused = if res.rank().as_usize() == 1 {
            self.glwe_mul_ct_rank1_fused_tmp_bytes(res, res, res, tsk)
        } else {
            0
        };

        sequential.max(fused)
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

        let a_ref = a.to_ref();
        let b_ref = b.to_ref();

        let fused_eligible = dst.rank().as_usize() == 1
            && a.rank().as_usize() == 1
            && b.rank().as_usize() == 1
            && a.base2k() == b.base2k()
            && a.base2k() == tsk.base2k()
            && a.base2k() == dst.base2k();
        if fused_eligible {
            let mut dst_view = dst.to_mut();
            self.glwe_mul_ct_rank1_fused(
                cnv_offset,
                &mut dst_view,
                &a_ref,
                a.effective_k(),
                &b_ref,
                b.effective_k(),
                tsk,
                tsk.size(),
                scratch,
            );
        } else {
            let tensor_layout = GLWELayout {
                n: dst.n(),
                base2k: dst.base2k(),
                k: a.max_k().max(b.max_k()),
                rank: dst.rank(),
            };

            let (mut tmp, scratch_1) = scratch.take_glwe_tensor(&tensor_layout);

            self.glwe_tensor_apply(
                cnv_offset,
                &mut tmp,
                &a_ref,
                a.effective_k(),
                &b_ref,
                b.effective_k(),
                scratch_1,
            );
            self.glwe_tensor_relinearize(&mut dst.to_mut(), &tmp, tsk, tsk.size(), scratch_1);
        }

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

        self.glwe_tensor_apply(
            cnv_offset,
            &mut tmp,
            &dst.to_ref(),
            dst.effective_k(),
            &a.to_ref(),
            a.effective_k(),
            scratch_1,
        );
        self.glwe_tensor_relinearize(&mut dst.to_mut(), &tmp, tsk, tsk.size(), scratch_1);

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
        let sequential = lvl_0 + lvl_1;
        let fused = if res.rank().as_usize() == 1 {
            self.glwe_square_ct_rank1_fused_tmp_bytes(res, res, tsk)
        } else {
            0
        };

        sequential.max(fused)
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
        let a_ref = a.to_ref();
        let fused_eligible =
            dst.rank().as_usize() == 1 && a.rank().as_usize() == 1 && a.base2k() == tsk.base2k() && a.base2k() == dst.base2k();
        if fused_eligible {
            let mut dst_view = dst.to_mut();
            self.glwe_square_ct_rank1_fused(cnv_offset, &mut dst_view, &a_ref, a.effective_k(), tsk, tsk.size(), scratch);
        } else {
            let tensor_layout = GLWELayout {
                n: dst.n(),
                base2k: dst.base2k(),
                k: a.max_k(),
                rank: dst.rank(),
            };

            let (mut tmp, scratch_1) = scratch.take_glwe_tensor(&tensor_layout);
            self.glwe_tensor_square_apply(cnv_offset, &mut tmp, &a_ref, a.effective_k(), scratch_1);
            self.glwe_tensor_relinearize(&mut dst.to_mut(), &tmp, tsk, tsk.size(), scratch_1);
        }

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
        self.glwe_tensor_relinearize(&mut dst.to_mut(), &tmp, tsk, tsk.size(), scratch_1);

        dst.meta.log_hom_rem = res_log_hom_rem;
        dst.meta.log_decimal = res_log_decimal;
        Ok(())
    }

    fn ckks_mul_pt_vec_znx_tmp_bytes_default<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
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

    fn ckks_mul_pt_vec_rnx_tmp_bytes_default<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
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
        GLWEPlaintext::<Vec<u8>>::bytes_of_from_infos(&b_infos) + self.glwe_mul_plain_tmp_bytes(res, a, &b_infos)
    }

    fn ckks_mul_const_tmp_bytes_default<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulConst<BE> + GLWERotate<BE>,
    {
        let b_size = b.min_k(res.base2k()).as_usize().div_ceil(res.base2k().as_usize());
        GLWE::<Vec<u8>>::bytes_of_from_infos(res)
            + self
                .glwe_mul_const_tmp_bytes(res, a, b_size)
                .max(self.glwe_rotate_tmp_bytes())
    }

    fn ckks_mul_pt_vec_znx_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
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

    fn ckks_mul_pt_vec_znx_inplace_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
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

    fn ckks_mul_pt_vec_rnx_default<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + GLWEMulPlain<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let (pt_glwe, scratch_1) = scratch.take_glwe_plaintext(&GLWEPlaintextLayout {
            n: self.n().into(),
            base2k: dst.base2k(),
            k: prec.min_k(dst.base2k()),
        });

        let mut pt_znx = CKKSPlaintextVecZnx::from_plaintext_with_meta(pt_glwe, prec);
        pt_rnx.to_znx(&mut pt_znx)?;
        self.ckks_mul_pt_vec_znx_default(dst, a, &pt_znx, scratch_1)
    }

    fn ckks_mul_pt_vec_rnx_inplace_default<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + GLWEMulPlain<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let (pt_glwe, scratch_1) = scratch.take_glwe_plaintext(&GLWEPlaintextLayout {
            n: self.n().into(),
            base2k: dst.base2k(),
            k: prec.min_k(dst.base2k()),
        });

        let mut pt_znx = CKKSPlaintextVecZnx::from_plaintext_with_meta(pt_glwe, prec);
        pt_rnx.to_znx(&mut pt_znx)?;
        self.ckks_mul_pt_vec_znx_inplace_default(dst, &pt_znx, scratch_1)
    }

    fn ckks_mul_const_znx_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd + GLWEMulConst<BE> + GLWERotate<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let (res_log_hom_rem, res_log_decimal, cnv_offset) = get_mul_const_params(dst, a, cst_znx.meta())?;
        match (cst_znx.re(), cst_znx.im()) {
            (None, None) => dst.data_mut().zero(),
            (Some(re_const), None) => self.glwe_mul_const(cnv_offset, &mut dst.to_mut(), &a.to_ref(), re_const, scratch),
            (None, Some(im_const)) => {
                self.glwe_mul_const(cnv_offset, &mut dst.to_mut(), &a.to_ref(), im_const, scratch);
                self.glwe_rotate_inplace((dst.n().as_usize() / 2) as i64, dst, scratch);
            }
            (Some(re_const), Some(im_const)) => {
                let (mut tmp, scratch_1) = scratch.take_glwe(dst);

                self.glwe_mul_const(cnv_offset, &mut dst.to_mut(), &a.to_ref(), re_const, scratch_1);
                self.glwe_mul_const(cnv_offset, &mut tmp, &a.to_ref(), im_const, scratch_1);
                self.glwe_rotate_inplace((dst.n().as_usize() / 2) as i64, &mut tmp, scratch_1);
                self.glwe_add_assign(dst, &tmp);
            }
        }

        dst.meta.log_hom_rem = res_log_hom_rem;
        dst.meta.log_decimal = res_log_decimal;
        Ok(())
    }

    fn ckks_mul_const_znx_inplace_default(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd + GLWEMulConst<BE> + GLWERotate<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let (res_log_hom_rem, res_log_decimal, cnv_offset) = get_mul_const_params(dst, dst, cst_znx.meta())?;
        match (cst_znx.re(), cst_znx.im()) {
            (None, None) => dst.data_mut().zero(),
            (Some(re_const), None) => self.glwe_mul_const_inplace(cnv_offset, &mut dst.to_mut(), re_const, scratch),
            (None, Some(im_const)) => {
                self.glwe_mul_const_inplace(cnv_offset, &mut dst.to_mut(), im_const, scratch);
                self.glwe_rotate_inplace((dst.n().as_usize() / 2) as i64, dst, scratch);
            }
            (Some(re_const), Some(im_const)) => {
                let (mut tmp, scratch_1) = scratch.take_glwe(dst);

                self.glwe_mul_const(cnv_offset, &mut tmp, &dst.to_ref(), im_const, scratch_1);
                self.glwe_mul_const_inplace(cnv_offset, &mut dst.to_mut(), re_const, scratch_1);
                self.glwe_rotate_inplace((dst.n().as_usize() / 2) as i64, &mut tmp, scratch_1);
                self.glwe_add_assign(dst, &tmp);
            }
        }

        dst.meta.log_hom_rem = res_log_hom_rem;
        dst.meta.log_decimal = res_log_decimal;
        Ok(())
    }

    fn ckks_mul_const_rnx_default<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd + GLWEMulConst<BE> + GLWERotate<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion,
    {
        if cst_rnx.re().is_none() && cst_rnx.im().is_none() {
            let (res_log_hom_rem, res_log_decimal, _) = get_mul_const_params(dst, a, prec)?;
            dst.data_mut().zero();
            dst.meta.log_hom_rem = res_log_hom_rem;
            dst.meta.log_decimal = res_log_decimal;
            return Ok(());
        }

        let cst_znx = cst_rnx.to_znx(dst.base2k(), prec)?;
        self.ckks_mul_const_znx_default(dst, a, &cst_znx, scratch)
    }

    fn ckks_mul_const_rnx_inplace_default<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd + GLWEMulConst<BE> + GLWERotate<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion,
    {
        if cst_rnx.re().is_none() && cst_rnx.im().is_none() {
            let (res_log_hom_rem, res_log_decimal, _) = get_mul_const_params(dst, dst, prec)?;
            dst.data_mut().zero();
            dst.meta.log_hom_rem = res_log_hom_rem;
            dst.meta.log_decimal = res_log_decimal;
            return Ok(());
        }

        let cst_znx = cst_rnx.to_znx(dst.base2k(), prec)?;
        self.ckks_mul_const_znx_inplace_default(dst, &cst_znx, scratch)
    }
}

impl<BE: Backend> CKKSMulDefault<BE> for Module<BE> {}

fn get_mul_ct_params<R, A, B>(res: &R, a: &A, b: &B) -> Result<(usize, usize, usize)>
where
    R: poulpy_core::layouts::LWEInfos + CKKSInfos,
    A: poulpy_core::layouts::LWEInfos + CKKSInfos,
    B: poulpy_core::layouts::LWEInfos + CKKSInfos,
{
    let res_log_hom_rem = checked_mul_ct_log_hom_rem("mul", a.log_hom_rem(), b.log_hom_rem(), a.log_decimal(), b.log_decimal())?;
    let res_log_decimal = a.log_decimal().min(b.log_decimal());

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
    R: poulpy_core::layouts::LWEInfos + CKKSInfos,
    A: poulpy_core::layouts::LWEInfos + CKKSInfos,
    B: poulpy_core::layouts::LWEInfos + CKKSInfos,
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

fn get_mul_const_params<R, A>(res: &R, a: &A, prec: CKKSMeta) -> Result<(usize, usize, usize)>
where
    R: poulpy_core::layouts::LWEInfos + CKKSInfos,
    A: poulpy_core::layouts::LWEInfos + CKKSInfos,
{
    let res_log_hom_rem = checked_mul_pt_log_hom_rem(
        "mul_const",
        a.log_hom_rem(),
        prec.log_hom_rem,
        a.log_decimal(),
        prec.log_decimal,
    )?;
    let res_log_decimal = a.log_decimal();
    let res_offset = (res_log_hom_rem + res_log_decimal).saturating_sub(res.max_k().as_usize());
    let cnv_offset = prec.min_k(res.base2k()).as_usize() + res_offset;

    Ok((
        checked_log_hom_rem_sub("mul_const", res_log_hom_rem, res_offset)?,
        res_log_decimal,
        cnv_offset,
    ))
}
