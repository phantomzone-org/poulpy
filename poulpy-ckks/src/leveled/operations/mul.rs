//! CKKS ciphertext multiplication.

use poulpy_core::{
    GLWETensoring, ScratchTakeCore,
    layouts::{
        GGLWEInfos, GLWE, GLWEInfos, GLWELayout, GLWETensor, GLWETensorKeyPrepared, GLWEToMut, GLWEToRef, LWEInfos,
        TorusPrecision,
    },
};
use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};

use crate::{CKKS, CKKSInfos, checked_log_hom_rem_sub, checked_mul_log_hom_rem};
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

    #[allow(clippy::too_many_arguments)]
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
}

fn get_mul_params<R, A, B>(res: &R, a: &A, b: &B) -> Result<(usize, usize, usize)>
where
    R: LWEInfos + CKKSInfos,
    A: LWEInfos + CKKSInfos,
    B: LWEInfos + CKKSInfos,
{
    // Value before considering res size
    let res_log_hom_rem = checked_mul_log_hom_rem("mul", a.log_hom_rem(), b.log_hom_rem(), a.log_decimal(), b.log_decimal())?;
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
            k: TorusPrecision(res.max_k().as_u32() + 4 * res.base2k().as_u32()),
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
        let (res_log_hom_rem, res_log_decimal, cnv_offset) = get_mul_params(self, a, b)?;

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

    fn square_tmp_bytes<R, T, BE: Backend>(module: &Module<BE>, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Module<BE>: GLWETensoring<BE>,
    {
        let glwe_layout = GLWELayout {
            n: res.n(),
            base2k: res.base2k(),
            k: TorusPrecision(res.max_k().as_u32() + 4 * res.base2k().as_u32()),
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
        let (res_log_hom_rem, res_log_decimal, cnv_offset) = get_mul_params(self, a, a)?;

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
}
