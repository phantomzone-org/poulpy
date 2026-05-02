use anyhow::Result;
use poulpy_core::{
    GLWEAdd, GLWECopy, GLWEMulConst, GLWEMulPlain, GLWERotate, GLWETensoring, ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GLWE, GLWEInfos, GLWELayout, GLWEPlaintext, GLWEPlaintextLayout, GLWEPlaintextToBackendMut,
        GLWEPlaintextToBackendRef, GLWETensor, GLWEToBackendMut, GLWEToBackendRef,
        ModuleCoreAlloc, TorusPrecision, glwe_backend_data_mut, prepared::GLWETensorKeyPreparedToBackendRef,
    },
};
use poulpy_hal::{
    api::{ScratchAvailable, VecZnxCopyBackend, VecZnxZeroBackend},
    layouts::{Backend, Module, ScratchArena},
};

use crate::{
    CKKSCiphertextToBackendMut, CKKSCiphertextToBackendRef, CKKSInfos, CKKSMeta, CKKSPlaintexToBackendRef, SetCKKSInfos,
    checked_log_budget_sub, checked_mul_ct_log_budget, checked_mul_pt_log_budget,
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
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: CKKSCiphertextToBackendRef<BE> + CKKSInfos + GLWEInfos,
        B: CKKSCiphertextToBackendRef<BE> + CKKSInfos + GLWEInfos,
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

        let a_owned = copy_ct_to_owned::<BE, _, A>(self, a);
        let b_owned = copy_ct_to_owned::<BE, _, B>(self, b);
        let mut dst_owned = self.glwe_alloc_from_infos(dst);
        let mut tmp = self.glwe_tensor_alloc_from_infos(&tensor_layout);
        self.glwe_tensor_apply(
            cnv_offset,
            &mut tmp,
            &a_owned,
            a.effective_k(),
            &b_owned,
            b.effective_k(),
            scratch,
        );
        self.glwe_tensor_relinearize(&mut dst_owned, &tmp, tsk, tsk.size(), scratch);
        copy_ct_from_owned::<BE, _, Dst>(self, dst, &dst_owned);

        dst.set_log_budget(res_log_budget);
        dst.set_log_delta(res_log_delta);
        Ok(())
    }

    fn ckks_mul_assign_default<Dst, A, T>(
        &self,
        dst: &mut Dst,
        a: &A,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWETensoring<BE> + GLWECopy<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSCiphertextToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: CKKSCiphertextToBackendRef<BE> + CKKSInfos + GLWEInfos,
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

        let dst_owned = copy_ct_to_owned::<BE, _, Dst>(self, dst);
        let a_owned = copy_ct_to_owned::<BE, _, A>(self, a);
        let mut res_owned = self.glwe_alloc_from_infos(dst);
        let mut tmp = self.glwe_tensor_alloc_from_infos(&tensor_layout);
        self.glwe_tensor_apply(
            cnv_offset,
            &mut tmp,
            &dst_owned,
            dst.effective_k(),
            &a_owned,
            a.effective_k(),
            scratch,
        );
        self.glwe_tensor_relinearize(&mut res_owned, &tmp, tsk, tsk.size(), scratch);
        copy_ct_from_owned::<BE, _, Dst>(self, dst, &res_owned);

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

    fn ckks_square_into_default<Dst, A, T>(
        &self,
        dst: &mut Dst,
        a: &A,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWETensoring<BE> + GLWECopy<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: CKKSCiphertextToBackendRef<BE> + CKKSInfos + GLWEInfos,
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

        let a_owned = copy_ct_to_owned::<BE, _, A>(self, a);
        let mut dst_owned = self.glwe_alloc_from_infos(dst);
        let mut tmp = self.glwe_tensor_alloc_from_infos(&tensor_layout);
        self.glwe_tensor_square_apply(cnv_offset, &mut tmp, &a_owned, a.effective_k(), scratch);
        self.glwe_tensor_relinearize(&mut dst_owned, &tmp, tsk, tsk.size(), scratch);
        copy_ct_from_owned::<BE, _, Dst>(self, dst, &dst_owned);

        dst.set_log_budget(res_log_budget);
        dst.set_log_delta(res_log_delta);
        Ok(())
    }

    fn ckks_square_assign_default<Dst, T>(
        &self,
        dst: &mut Dst,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWETensoring<BE> + GLWECopy<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSCiphertextToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
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

        let dst_owned = copy_ct_to_owned::<BE, _, Dst>(self, dst);
        let mut res_owned = self.glwe_alloc_from_infos(dst);
        let mut tmp = self.glwe_tensor_alloc_from_infos(&tensor_layout);
        self.glwe_tensor_square_apply(cnv_offset, &mut tmp, &dst_owned, dst.effective_k(), scratch);
        self.glwe_tensor_relinearize(&mut res_owned, &tmp, tsk, tsk.size(), scratch);
        copy_ct_from_owned::<BE, _, Dst>(self, dst, &res_owned);

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
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos,
        Self: GLWECopy<BE> + GLWEMulPlain<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf> + VecZnxCopyBackend<BE>,
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: CKKSCiphertextToBackendRef<BE> + CKKSInfos + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_pt_params(dst, a, pt_znx)?;
        let a_owned = copy_ct_to_owned::<BE, _, A>(self, a);
        let pt_owned = copy_pt_to_owned::<BE, _, P>(self, pt_znx);
        let mut dst_owned = self.glwe_alloc_from_infos(dst);
        self.glwe_mul_plain(
            cnv_offset,
            &mut dst_owned,
            &a_owned,
            a.effective_k(),
            &pt_owned,
            pt_znx.max_k().as_usize(),
            scratch,
        );
        copy_ct_from_owned::<BE, _, Dst>(self, dst, &dst_owned);
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
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos,
        Self: GLWECopy<BE> + GLWEMulPlain<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf> + VecZnxCopyBackend<BE>,
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSCiphertextToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_pt_params(dst, dst, pt_znx)?;
        let dst_effective_k = dst.effective_k();
        let mut dst_owned = copy_ct_to_owned::<BE, _, Dst>(self, dst);
        let pt_owned = copy_pt_to_owned::<BE, _, P>(self, pt_znx);
        self.glwe_mul_plain_assign(
            cnv_offset,
            &mut dst_owned,
            dst_effective_k,
            &pt_owned,
            pt_znx.max_k().as_usize(),
            scratch,
        );
        copy_ct_from_owned::<BE, _, Dst>(self, dst, &dst_owned);
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
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos,
        Self: GLWEAdd<BE> + GLWECopy<BE> + GLWEMulConst<BE> + GLWERotate<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: CKKSCiphertextToBackendRef<BE> + CKKSInfos + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_const_params(dst, a, pt_znx.meta())?;
        let rotate_by = (dst.n().as_usize() / 2) as i64;
        let a_owned = copy_ct_to_owned::<BE, _, A>(self, a);
        let mut dst_owned = self.glwe_alloc_from_infos(dst);
        let pt_owned = copy_pt_to_owned::<BE, _, P>(self, pt_znx);
        self.glwe_mul_const(cnv_offset, &mut dst_owned, &a_owned, &pt_owned, 0, scratch);
        let mut tmp = self.glwe_alloc_from_infos(dst);
        self.glwe_mul_const(cnv_offset, &mut tmp, &a_owned, &pt_owned, rotate_by as usize, scratch);
        {
            let mut tmp_ref = GLWEToBackendMut::<BE>::to_backend_mut(&mut tmp);
            self.glwe_rotate_assign(rotate_by, &mut tmp_ref, scratch);
        }
        self.glwe_add_assign(&mut dst_owned, &tmp);
        copy_ct_from_owned::<BE, _, Dst>(self, dst, &dst_owned);

        dst.set_log_budget(res_log_budget);
        dst.set_log_delta(res_log_delta);
        Ok(())
    }

    fn ckks_mul_pt_const_znx_assign_default<Dst, P>(
        &self,
        dst: &mut Dst,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos,
        Self: GLWEAdd<BE> + GLWECopy<BE> + GLWEMulConst<BE> + GLWERotate<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSCiphertextToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_const_params(dst, dst, pt_znx.meta())?;
        let rotate_by = (dst.n().as_usize() / 2) as i64;
        let mut dst_owned = copy_ct_to_owned::<BE, _, Dst>(self, dst);
        let pt_owned = copy_pt_to_owned::<BE, _, P>(self, pt_znx);
        let mut tmp = self.glwe_alloc_from_infos(dst);
        let dst_src = copy_ct_to_owned::<BE, _, Dst>(self, dst);
        self.glwe_mul_const(cnv_offset, &mut tmp, &dst_src, &pt_owned, rotate_by as usize, scratch);
        self.glwe_mul_const_assign(cnv_offset, &mut dst_owned, &pt_owned, 0, scratch);
        {
            let mut tmp_ref = GLWEToBackendMut::<BE>::to_backend_mut(&mut tmp);
            self.glwe_rotate_assign(rotate_by, &mut tmp_ref, scratch);
        }
        self.glwe_add_assign(&mut dst_owned, &tmp);
        copy_ct_from_owned::<BE, _, Dst>(self, dst, &dst_owned);

        dst.set_log_budget(res_log_budget);
        dst.set_log_delta(res_log_delta);
        Ok(())
    }
}

impl<BE: Backend> CKKSMulDefault<BE> for Module<BE> {}

fn copy_ct_to_owned<BE: Backend, M, A>(module: &M, src: &A) -> GLWE<BE::OwnedBuf>
where
    M: GLWECopy<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf> + ?Sized,
    A: CKKSCiphertextToBackendRef<BE> + GLWEInfos,
{
    let mut dst = module.glwe_alloc_from_infos(src);
    let src_ref = GLWEToBackendRef::<BE>::to_backend_ref(src);
    module.glwe_copy(&mut GLWEToBackendMut::<BE>::to_backend_mut(&mut dst), &src_ref);
    dst
}

fn copy_ct_from_owned<BE: Backend, M, Dst>(module: &M, dst: &mut Dst, src: &GLWE<BE::OwnedBuf>)
where
    M: GLWECopy<BE> + ?Sized,
    Dst: CKKSCiphertextToBackendMut<BE>,
{
    let src_ref = GLWEToBackendRef::<BE>::to_backend_ref(src);
    module.glwe_copy(&mut GLWEToBackendMut::<BE>::to_backend_mut(dst), &src_ref);
}

fn copy_pt_to_owned<BE: Backend, M, P>(module: &M, src: &P) -> GLWEPlaintext<BE::OwnedBuf>
where
    M: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf> + VecZnxCopyBackend<BE> + ?Sized,
    P: CKKSPlaintexToBackendRef<BE> + GLWEInfos,
{
    let mut dst = module.glwe_plaintext_alloc_from_infos(src);
    let src_ref = GLWEPlaintextToBackendRef::<BE>::to_backend_ref(src);
    {
        let mut dst_mut = GLWEPlaintextToBackendMut::<BE>::to_backend_mut(&mut dst);
        module.vec_znx_copy_backend(&mut dst_mut.data, 0, &src_ref.data, 0);
    }
    dst
}

fn zero_ciphertext<BE: Backend, Dst, M>(module: &M, dst: &mut Dst)
where
    M: VecZnxZeroBackend<BE> + ?Sized,
    Dst: CKKSCiphertextToBackendMut<BE> + GLWEInfos,
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
    R: poulpy_core::layouts::LWEInfos + CKKSInfos,
    A: poulpy_core::layouts::LWEInfos + CKKSInfos,
    B: poulpy_core::layouts::LWEInfos + CKKSInfos,
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
    R: poulpy_core::layouts::LWEInfos + CKKSInfos,
    A: poulpy_core::layouts::LWEInfos + CKKSInfos,
    B: poulpy_core::layouts::LWEInfos + CKKSInfos,
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

fn get_mul_const_params<R, A>(res: &R, a: &A, prec: CKKSMeta) -> Result<(usize, usize, usize)>
where
    R: poulpy_core::layouts::LWEInfos + CKKSInfos,
    A: poulpy_core::layouts::LWEInfos + CKKSInfos,
{
    let res_log_budget = checked_mul_pt_log_budget("mul_const", a.log_budget(), prec.log_budget, a.log_delta(), prec.log_delta)?;
    let res_log_delta = a.log_delta();
    let res_offset = (res_log_budget + res_log_delta).saturating_sub(res.max_k().as_usize());
    let cnv_offset = prec.min_k(res.base2k()).as_usize() + res_offset;

    Ok((
        checked_log_budget_sub("mul_const", res_log_budget, res_offset)?,
        res_log_delta,
        cnv_offset,
    ))
}
