use anyhow::Result;
use poulpy_core::{
    GLWEAdd, GLWECopy, GLWEMulConst, GLWEMulPlain, GLWERotate, GLWETensoring, ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GLWE, GLWEInfos, GLWELayout, GLWEPlaintextLayout, GLWETensor, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
        ModuleCoreAlloc, TorusPrecision, glwe_backend_data_mut, glwe_backend_mut_from_mut,
        prepared::GLWETensorKeyPreparedToBackendRef,
    },
};
use poulpy_hal::{
    api::{ScratchAvailable, VecZnxCopyBackend, VecZnxZeroBackend},
    layouts::{Backend, Module, ScratchArena},
};

use crate::{CKKSInfos, CKKSMeta, SetCKKSInfos, checked_log_budget_sub, checked_mul_ct_log_budget, checked_mul_pt_log_budget};

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

        let lvl_0 = GLWETensor::bytes_of_from_infos(&glwe_layout);
        let lvl_1 = self
            .glwe_tensor_apply_tmp_bytes(&glwe_layout, res, res)
            .max(self.glwe_tensor_relinearize_tmp_bytes(res, &glwe_layout, tsk));

        lvl_0 + lvl_1
    }

    fn ckks_mul_into_default<Dst, A, B, T>(
        &self,
        dst: &mut Dst,
        a: &A,
        b: &B,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWETensoring<BE> + GLWECopy<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        B: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_ct_params(dst, a, b)?;

        let tensor_layout = GLWELayout {
            n: dst.n(),
            base2k: dst.base2k(),
            k: a.max_k().max(b.max_k()),
            rank: dst.rank(),
        };
        let scratch_local = scratch.borrow();
        let (mut tmp, mut scratch_local) = scratch_local.take_glwe_tensor(&tensor_layout);
        {
            let mut tmp_arg = &mut tmp;
            self.glwe_tensor_apply(
                cnv_offset,
                &mut tmp_arg,
                a,
                a.effective_k(),
                b,
                b.effective_k(),
                &mut scratch_local,
            );
            self.glwe_tensor_relinearize(dst, &tmp_arg, tsk, tsk.size(), &mut scratch_local);
        }

        dst.set_log_budget(res_log_budget);
        dst.set_log_delta(res_log_delta);
        Ok(())
    }

    fn ckks_mul_assign_default<Dst, A, T>(&self, dst: &mut Dst, a: &A, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWETensoring<BE> + GLWECopy<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_ct_params(dst, dst, a)?;

        let tensor_layout = GLWELayout {
            n: dst.n(),
            base2k: dst.base2k(),
            k: dst.max_k().max(a.max_k()),
            rank: dst.rank(),
        };
        let scratch_local = scratch.borrow();
        let (mut tmp, mut scratch_local) = scratch_local.take_glwe_tensor(&tensor_layout);
        {
            let mut tmp_arg = &mut tmp;
            self.glwe_tensor_apply(
                cnv_offset,
                &mut tmp_arg,
                &*dst,
                dst.effective_k(),
                a,
                a.effective_k(),
                &mut scratch_local,
            );
            self.glwe_tensor_relinearize(dst, &tmp_arg, tsk, tsk.size(), &mut scratch_local);
        }

        dst.set_log_budget(res_log_budget);
        dst.set_log_delta(res_log_delta);
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

    fn ckks_square_into_default<Dst, A, T>(&self, dst: &mut Dst, a: &A, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWETensoring<BE> + GLWECopy<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_ct_params(dst, a, a)?;

        let tensor_layout = GLWELayout {
            n: dst.n(),
            base2k: dst.base2k(),
            k: a.max_k(),
            rank: dst.rank(),
        };
        let scratch_local = scratch.borrow();
        let (mut tmp, mut scratch_local) = scratch_local.take_glwe_tensor(&tensor_layout);
        {
            let mut tmp_arg = &mut tmp;
            self.glwe_tensor_square_apply(cnv_offset, &mut tmp_arg, a, a.effective_k(), &mut scratch_local);
            self.glwe_tensor_relinearize(dst, &tmp_arg, tsk, tsk.size(), &mut scratch_local);
        }

        dst.set_log_budget(res_log_budget);
        dst.set_log_delta(res_log_delta);
        Ok(())
    }

    fn ckks_square_assign_default<Dst, T>(&self, dst: &mut Dst, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWETensoring<BE> + GLWECopy<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_ct_params(dst, dst, dst)?;

        let tensor_layout = GLWELayout {
            n: dst.n(),
            base2k: dst.base2k(),
            k: dst.max_k(),
            rank: dst.rank(),
        };
        let scratch_local = scratch.borrow();
        let (mut tmp, mut scratch_local) = scratch_local.take_glwe_tensor(&tensor_layout);
        {
            let mut tmp_arg = &mut tmp;
            self.glwe_tensor_square_apply(cnv_offset, &mut tmp_arg, &*dst, dst.effective_k(), &mut scratch_local);
            self.glwe_tensor_relinearize(dst, &tmp_arg, tsk, tsk.size(), &mut scratch_local);
        }

        dst.set_log_budget(res_log_budget);
        dst.set_log_delta(res_log_delta);
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

    fn ckks_mul_pt_const_tmp_bytes_default<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulConst<BE> + GLWERotate<BE>,
    {
        let b_infos = GLWEPlaintextLayout {
            n: res.n(),
            base2k: res.base2k(),
            k: b.min_k(res.base2k()),
        };
        GLWE::<Vec<u8>>::bytes_of_from_infos(res)
            + self
                .glwe_mul_const_tmp_bytes(res, a, &b_infos)
                .max(self.glwe_rotate_tmp_bytes())
    }

    fn ckks_mul_pt_vec_znx_into_default<Dst, A, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos + CKKSInfos,
        Self: GLWECopy<BE> + GLWEMulPlain<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf> + VecZnxCopyBackend<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_pt_params(dst, a, pt_znx)?;
        self.glwe_mul_plain(
            cnv_offset,
            dst,
            a,
            a.effective_k(),
            pt_znx,
            pt_znx.max_k().as_usize(),
            scratch,
        );
        dst.set_log_budget(res_log_budget);
        dst.set_log_delta(res_log_delta);
        Ok(())
    }

    fn ckks_mul_pt_vec_znx_assign_default<Dst, P>(
        &self,
        dst: &mut Dst,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos + CKKSInfos,
        Self: GLWECopy<BE> + GLWEMulPlain<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf> + VecZnxCopyBackend<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_pt_params(dst, dst, pt_znx)?;
        let dst_effective_k = dst.effective_k();
        self.glwe_mul_plain_assign(cnv_offset, dst, dst_effective_k, pt_znx, pt_znx.max_k().as_usize(), scratch);
        dst.set_log_budget(res_log_budget);
        dst.set_log_delta(res_log_delta);
        Ok(())
    }

    fn ckks_mul_pt_const_znx_into_default<Dst, A, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos + CKKSInfos,
        Self: GLWEAdd<BE> + GLWECopy<BE> + GLWEMulConst<BE> + GLWERotate<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_pt_params(dst, a, pt_znx)?;
        let rotate_by = (dst.n().as_usize() / 2) as i64;
        let scratch_local = scratch.borrow();
        let (mut tmp, mut scratch_local) = scratch_local.take_glwe(dst);
        self.glwe_mul_const(cnv_offset, dst, a, pt_znx, 0, &mut scratch_local);
        {
            let mut tmp_arg = &mut tmp;
            self.glwe_mul_const(cnv_offset, &mut tmp_arg, a, pt_znx, rotate_by as usize, &mut scratch_local);
        }
        {
            let mut tmp_ref = glwe_backend_mut_from_mut::<BE>(&mut tmp);
            self.glwe_rotate_assign(rotate_by, &mut tmp_ref, &mut scratch_local);
        }
        {
            let tmp_arg = &mut tmp;
            self.glwe_add_assign(dst, &tmp_arg);
        }

        dst.set_log_budget(res_log_budget);
        dst.set_log_delta(res_log_delta);
        Ok(())
    }

    fn ckks_mul_pt_const_znx_assign_default<Dst, P>(
        &self,
        dst: &mut Dst,
        cnst: &P,
        cnst_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos + CKKSInfos,
        Self: GLWEMulConst<BE>,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_pt_params(dst, dst, cnst)?;

        self.glwe_mul_const_assign(cnv_offset, dst, cnst, cnst_coeff, scratch);

        dst.set_log_budget(res_log_budget);
        dst.set_log_delta(res_log_delta);
        Ok(())
    }
}
impl<BE: Backend> CKKSMulDefault<BE> for Module<BE> {}

fn zero_ciphertext<BE: Backend, Dst, M>(module: &M, dst: &mut Dst)
where
    M: VecZnxZeroBackend<BE> + ?Sized,
    Dst: GLWEToBackendMut<BE> + GLWEInfos,
{
    let cols = dst.rank().as_usize() + 1;
    let mut dst_ref = GLWEToBackendMut::<BE>::to_backend_mut(dst);
    let mut dst_data = glwe_backend_data_mut::<BE>(&mut dst_ref);
    for col in 0..cols {
        module.vec_znx_zero_backend(&mut dst_data, col);
    }
}

fn get_mul_ct_params<R, A, B>(res: &R, a: &A, b: &B) -> Result<(usize, usize, usize)>
where
    R: LWEInfos + CKKSInfos,
    A: LWEInfos + CKKSInfos,
    B: LWEInfos + CKKSInfos,
{
    let res_log_budget = checked_mul_ct_log_budget("mul", a.log_budget(), b.log_budget(), a.log_delta(), b.log_delta())?;
    let res_log_delta = a.log_delta().min(b.log_delta());

    let res_offset = (res_log_budget + res_log_delta).saturating_sub(res.max_k().as_usize());
    let cnv_offset = a.effective_k().max(b.effective_k()) + res_offset;

    Ok((
        checked_log_budget_sub("mul", res_log_budget, res_offset)?,
        res_log_delta,
        cnv_offset,
    ))
}

fn get_mul_pt_params<R, A, B>(res: &R, a: &A, b: &B) -> Result<(usize, usize, usize)>
where
    R: LWEInfos + CKKSInfos,
    A: LWEInfos + CKKSInfos,
    B: LWEInfos + CKKSInfos,
{
    let res_log_budget = checked_mul_pt_log_budget("mul", a.log_budget(), b.log_budget(), a.log_delta(), b.log_delta())?;
    let res_log_delta = a.log_delta();
    let res_offset = (res_log_budget + res_log_delta).saturating_sub(res.max_k().as_usize());
    let cnv_offset = b.max_k().as_usize() + res_offset;

    Ok((
        checked_log_budget_sub("mul", res_log_budget, res_offset)?,
        res_log_delta,
        cnv_offset,
    ))
}
