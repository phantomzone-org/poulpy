//! CKKS ciphertext multiplication.

use poulpy_core::{
    GLWEMulPlain, GLWETensoring, ScratchTakeCore,
    layouts::{
        GGLWEInfos, GLWE, GLWEInfos, GLWELayout, GLWEPlaintext, GLWEPlaintextLayout, GLWETensor, GLWETensorKeyPrepared,
        GLWEToMut, GLWEToRef, LWEInfos, TorusPrecision,
    },
};
use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};

use crate::{
    CKKS, CKKSInfos, checked_log_hom_rem_sub, checked_mul_ct_log_hom_rem,
    error::checked_mul_pt_log_hom_rem,
    layouts::plaintext::{CKKSPlaintextConversion, CKKSPlaintextRnx, attach_meta},
};
use anyhow::Result;

pub trait CKKSMulOps {
    fn mul_tmp_bytes<R, T, BE: Backend>(module: &Module<BE>, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Module<BE>: GLWETensoring<BE>;

    fn square_tmp_bytes<R, T, BE: Backend>(module: &Module<BE>, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Module<BE>: GLWETensoring<BE>;

    fn mul_pt_znx_tmp_bytes<R, A, BE: Backend>(module: &Module<BE>, res: &R, a: &A, b: &CKKS) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Module<BE>: GLWEMulPlain<BE>;

    fn mul_pt_rnx_tmp_bytes<R, A, BE: Backend>(module: &Module<BE>, res: &R, a: &A, b: &CKKS) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Module<BE>: GLWEMulPlain<BE>;

    fn mul<A, B, BE: Backend>(
        &mut self,
        module: &Module<BE>,
        a: &A,
        b: &B,
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWETensoring<BE>,
        A: GLWEToRef + LWEInfos + CKKSInfos,
        B: GLWEToRef + LWEInfos + CKKSInfos,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn mul_inplace<A, BE: Backend>(
        &mut self,
        module: &Module<BE>,
        a: &A,
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWETensoring<BE>,
        A: GLWEToRef + LWEInfos + CKKSInfos,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn square<A, BE: Backend>(
        &mut self,
        module: &Module<BE>,
        a: &A,
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWETensoring<BE>,
        A: GLWEToRef + LWEInfos + CKKSInfos,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn square_inplace<BE: Backend>(
        &mut self,
        module: &Module<BE>,
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWETensoring<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn mul_pt_znx<A, BE: Backend>(
        &mut self,
        module: &Module<BE>,
        a: &A,
        pt_znx: &GLWEPlaintext<impl DataRef, CKKS>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEMulPlain<BE>,
        A: GLWEToRef + LWEInfos + CKKSInfos,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn mul_pt_znx_inplace<BE: Backend>(
        &mut self,
        module: &Module<BE>,
        pt_znx: &GLWEPlaintext<impl DataRef, CKKS>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEMulPlain<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn mul_pt_rnx<A, F, BE: Backend>(
        &mut self,
        module: &Module<BE>,
        a: &A,
        pt_rnx: &CKKSPlaintextRnx<F>,
        prec: CKKS,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEMulPlain<BE>,
        A: GLWEToRef + LWEInfos + CKKSInfos,
        CKKSPlaintextRnx<F>: CKKSPlaintextConversion,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn mul_pt_rnx_inplace<F, BE: Backend>(
        &mut self,
        module: &Module<BE>,
        pt_rnx: &CKKSPlaintextRnx<F>,
        prec: CKKS,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEMulPlain<BE>,
        CKKSPlaintextRnx<F>: CKKSPlaintextConversion,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;
}

impl<D: DataMut> CKKSMulOps for GLWE<D, CKKS> {
    fn mul_tmp_bytes<R, T, BE: Backend>(module: &Module<BE>, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Module<BE>: GLWETensoring<BE>,
    {
        let glwe_layout = GLWELayout {
            n: res.n(),
            base2k: res.base2k(),
            k: TorusPrecision(res.max_k().as_u32()),
            rank: res.rank(),
        };

        let lvl_0 = GLWETensor::bytes_of_from_infos(&glwe_layout);
        let lvl_1 = module
            .glwe_tensor_apply_tmp_bytes(&glwe_layout, res, res)
            .max(module.glwe_tensor_relinearize_tmp_bytes(res, &glwe_layout, tsk));

        lvl_0 + lvl_1
    }
    fn mul<A, B, BE: Backend>(
        &mut self,
        module: &Module<BE>,
        a: &A,
        b: &B,
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWETensoring<BE>,
        A: GLWEToRef + LWEInfos + CKKSInfos,
        B: GLWEToRef + LWEInfos + CKKSInfos,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let (res_log_hom_rem, res_log_decimal, cnv_offset) = get_mul_ct_params(self, a, b)?;

        let tensor_layout = GLWELayout {
            n: self.n(),
            base2k: self.base2k(),
            k: a.max_k().max(b.max_k()), //TODO: optimize
            rank: self.rank(),
        };

        let (mut tmp, scratch_1) = scratch.take_glwe_tensor(&tensor_layout);

        let a_ref = a.to_ref();
        let b_ref = b.to_ref();
        module.glwe_tensor_apply(
            cnv_offset,
            &mut tmp,
            &a_ref,
            a.effective_k(),
            &b_ref,
            b.effective_k(),
            scratch_1,
        );

        // TODO: Chose correct optimal size based on noise
        let mut self_view = self.to_mut();
        module.glwe_tensor_relinearize(&mut self_view, &tmp, tsk, tsk.size(), scratch_1);

        self.set_log_hom_rem(res_log_hom_rem)?;
        self.set_log_decimal(res_log_decimal)?;

        Ok(())
    }

    fn mul_inplace<A, BE: Backend>(
        &mut self,
        module: &Module<BE>,
        a: &A,
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWETensoring<BE>,
        A: GLWEToRef + LWEInfos + CKKSInfos,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let (res_log_hom_rem, res_log_decimal, cnv_offset) = get_mul_ct_params(self, self, a)?;

        let tensor_layout = GLWELayout {
            n: self.n(),
            base2k: self.base2k(),
            k: self.max_k().max(a.max_k()), //TODO: optimize
            rank: self.rank(),
        };

        let (mut tmp, scratch_1) = scratch.take_glwe_tensor(&tensor_layout);

        let self_ref = self.to_ref();
        let a_ref = a.to_ref();
        module.glwe_tensor_apply(
            cnv_offset,
            &mut tmp,
            &self_ref,
            self.effective_k(),
            &a_ref,
            a.effective_k(),
            scratch_1,
        );

        // TODO: Chose correct optimal size based on noise
        let mut self_view = self.to_mut();
        module.glwe_tensor_relinearize(&mut self_view, &tmp, tsk, tsk.size(), scratch_1);

        self.set_log_hom_rem(res_log_hom_rem)?;
        self.set_log_decimal(res_log_decimal)?;

        Ok(())
    }

    fn square_tmp_bytes<R, T, BE: Backend>(module: &Module<BE>, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Module<BE>: GLWETensoring<BE>,
    {
        let glwe_layout = GLWELayout {
            n: res.n(),
            base2k: res.base2k(),
            k: TorusPrecision(res.max_k().as_u32()),
            rank: res.rank(),
        };

        let lvl_0 = GLWETensor::bytes_of_from_infos(&glwe_layout);
        let lvl_1 = module
            .glwe_tensor_square_apply_tmp_bytes(&glwe_layout, res)
            .max(module.glwe_tensor_relinearize_tmp_bytes(res, &glwe_layout, tsk));

        lvl_0 + lvl_1
    }

    fn square<A, BE: Backend>(
        &mut self,
        module: &Module<BE>,
        a: &A,
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWETensoring<BE>,
        A: GLWEToRef + LWEInfos + CKKSInfos,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let (res_log_hom_rem, res_log_decimal, cnv_offset) = get_mul_ct_params(self, a, a)?;

        let tensor_layout = GLWELayout {
            n: self.n(),
            base2k: self.base2k(),
            k: a.max_k(), //TODO: optimize
            rank: self.rank(),
        };

        let (mut tmp, scratch_1) = scratch.take_glwe_tensor(&tensor_layout);

        let a_ref = a.to_ref();
        module.glwe_tensor_square_apply(cnv_offset, &mut tmp, &a_ref, a.effective_k(), scratch_1);

        // TODO: Chose correct optimal size based on noise
        let mut self_view = self.to_mut();
        module.glwe_tensor_relinearize(&mut self_view, &tmp, tsk, tsk.size(), scratch_1);

        self.set_log_hom_rem(res_log_hom_rem)?;
        self.set_log_decimal(res_log_decimal)?;
        Ok(())
    }

    fn square_inplace<BE: Backend>(
        &mut self,
        module: &Module<BE>,
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWETensoring<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let (res_log_hom_rem, res_log_decimal, cnv_offset) = get_mul_ct_params(self, self, self)?;

        let tensor_layout = GLWELayout {
            n: self.n(),
            base2k: self.base2k(),
            k: self.max_k(), //TODO: optimize
            rank: self.rank(),
        };

        let (mut tmp, scratch_1) = scratch.take_glwe_tensor(&tensor_layout);

        module.glwe_tensor_square_apply(cnv_offset, &mut tmp, &self.to_ref(), self.effective_k(), scratch_1);

        // TODO: Chose correct optimal size based on noise
        let mut self_view = self.to_mut();
        module.glwe_tensor_relinearize(&mut self_view, &tmp, tsk, tsk.size(), scratch_1);

        self.set_log_hom_rem(res_log_hom_rem)?;
        self.set_log_decimal(res_log_decimal)?;
        Ok(())
    }

    fn mul_pt_znx_tmp_bytes<R, A, BE: Backend>(module: &Module<BE>, res: &R, a: &A, b: &CKKS) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Module<BE>: GLWEMulPlain<BE>,
    {
        let b_infos = GLWEPlaintextLayout {
            n: res.n(),
            base2k: res.base2k(),
            k: b.min_k(res.base2k()),
        };
        module.glwe_mul_plain_tmp_bytes(res, a, &b_infos)
    }

    fn mul_pt_rnx_tmp_bytes<R, A, BE: Backend>(module: &Module<BE>, res: &R, a: &A, b: &CKKS) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Module<BE>: GLWEMulPlain<BE>,
    {
        let b_infos = GLWEPlaintextLayout {
            n: module.n().into(),
            base2k: res.base2k(),
            k: b.min_k(res.base2k()),
        };
        GLWEPlaintext::<Vec<u8>, ()>::bytes_of_from_infos(&b_infos) + module.glwe_mul_plain_tmp_bytes(res, a, &b_infos)
    }

    fn mul_pt_znx<A, BE: Backend>(
        &mut self,
        module: &Module<BE>,
        a: &A,
        pt_znx: &GLWEPlaintext<impl DataRef, CKKS>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEMulPlain<BE>,
        A: GLWEToRef + LWEInfos + CKKSInfos,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let (res_log_hom_rem, res_log_decimal, cnv_offset) = get_mul_pt_params(self, a, pt_znx)?;

        module.glwe_mul_plain(
            cnv_offset,
            &mut self.to_mut(),
            &a.to_ref(),
            a.effective_k(),
            pt_znx,
            pt_znx.max_k().as_usize(),
            scratch,
        );

        self.set_log_hom_rem(res_log_hom_rem)?;
        self.set_log_decimal(res_log_decimal)?;

        Ok(())
    }

    fn mul_pt_znx_inplace<BE: Backend>(
        &mut self,
        module: &Module<BE>,
        pt_znx: &GLWEPlaintext<impl DataRef, CKKS>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEMulPlain<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let (res_log_hom_rem, res_log_decimal, cnv_offset) = get_mul_pt_params(self, self, pt_znx)?;

        let self_effective_k = self.effective_k();

        module.glwe_mul_plain_inplace(
            cnv_offset,
            &mut self.to_mut(),
            self_effective_k,
            pt_znx,
            pt_znx.max_k().as_usize(),
            scratch,
        );

        self.set_log_hom_rem(res_log_hom_rem)?;
        self.set_log_decimal(res_log_decimal)?;

        Ok(())
    }

    fn mul_pt_rnx<A, F, BE: Backend>(
        &mut self,
        module: &Module<BE>,
        a: &A,
        pt_rnx: &CKKSPlaintextRnx<F>,
        prec: CKKS,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEMulPlain<BE>,
        A: GLWEToRef + LWEInfos + CKKSInfos,
        CKKSPlaintextRnx<F>: CKKSPlaintextConversion,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let (pt_glwe, scratch_1) = scratch.take_glwe_plaintext(&GLWEPlaintextLayout {
            n: module.n().into(),
            base2k: self.base2k(),
            k: prec.min_k(self.base2k()),
        });

        let mut pt_znx = attach_meta(pt_glwe, prec);
        pt_rnx.to_znx::<BE>(&mut pt_znx).unwrap();
        self.mul_pt_znx(module, a, &pt_znx, scratch_1)
    }

    fn mul_pt_rnx_inplace<F, BE: Backend>(
        &mut self,
        module: &Module<BE>,
        pt_rnx: &CKKSPlaintextRnx<F>,
        prec: CKKS,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEMulPlain<BE>,
        CKKSPlaintextRnx<F>: CKKSPlaintextConversion,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let (pt_glwe, scratch_1) = scratch.take_glwe_plaintext(&GLWEPlaintextLayout {
            n: module.n().into(),
            base2k: self.base2k(),
            k: prec.min_k(self.base2k()),
        });

        let mut pt_znx = attach_meta(pt_glwe, prec);
        pt_rnx.to_znx::<BE>(&mut pt_znx).unwrap();
        self.mul_pt_znx_inplace(module, &pt_znx, scratch_1)
    }
}

fn get_mul_ct_params<R, A, B>(res: &R, a: &A, b: &B) -> Result<(usize, usize, usize)>
where
    R: LWEInfos + CKKSInfos,
    A: LWEInfos + CKKSInfos,
    B: LWEInfos + CKKSInfos,
{
    // Value before considering res size
    let res_log_hom_rem = checked_mul_ct_log_hom_rem("mul", a.log_hom_rem(), b.log_hom_rem(), a.log_decimal(), b.log_decimal())?;
    let res_log_decimal = a.log_decimal().max(b.log_decimal());

    // Offset to accomodate `res_log_hom_rem` to `res.max_k()`
    let res_offset = (res_log_hom_rem + res_log_decimal).saturating_sub(res.max_k().as_usize());

    // cnv_offset that takes into account `res_offset`
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
    // Value before considering res size
    let res_log_hom_rem = checked_mul_pt_log_hom_rem("mul", a.log_hom_rem(), b.log_hom_rem(), a.log_decimal(), b.log_decimal())?;
    let res_log_decimal = a.log_decimal();

    // Offset to accomodate `res_log_hom_rem` to `res.max_k()`
    let res_offset = (res_log_hom_rem + res_log_decimal).saturating_sub(res.max_k().as_usize());

    // cnv_offset that takes into account `res_offset`
    let cnv_offset = b.max_k().as_usize() + res_offset;

    Ok((
        checked_log_hom_rem_sub("mul", res_log_hom_rem, res_offset)?,
        res_log_decimal,
        cnv_offset,
    ))
}
