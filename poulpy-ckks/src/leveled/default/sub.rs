use anyhow::Result;
use poulpy_core::{
    GLWENormalize, GLWEShift, GLWESub, ScratchArenaTakeCore,
    layouts::{GLWEPlaintext, GLWEPlaintextLayout, GLWEToBackendMut, LWEInfos},
};
use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, VecZnxRshSubBackend, VecZnxRshSubCoeffIntoBackend, VecZnxRshTmpBytes},
    layouts::{Backend, Module, ScratchArena},
};

use crate::{
    CKKSInfos, CKKSMeta, GLWEToBackendRef, SetCKKSInfos, checked_log_budget_sub, ckks_offset_binary, ckks_offset_unary,
    leveled::default::CKKSPlaintextDefault,
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
        R: LWEInfos,
        A: LWEInfos,
        Self: ModuleN + GLWEShift<BE> + GLWENormalize<BE> + VecZnxRshTmpBytes,
    {
        let b_infos = GLWEPlaintextLayout {
            n: self.n().into(),
            base2k: res.base2k(),
            k: b.min_k(res.base2k()),
        };
        GLWEPlaintext::<Vec<u8>>::bytes_of_from_infos(&b_infos) + self.ckks_sub_pt_vec_znx_tmp_bytes_default()
    }

    fn ckks_sub_pt_const_tmp_bytes_default(&self) -> usize
    where
        Self: GLWEShift<BE> + GLWENormalize<BE> + VecZnxRshTmpBytes,
    {
        self.glwe_shift_tmp_bytes()
            .max(self.glwe_normalize_tmp_bytes())
            .max(self.vec_znx_rsh_tmp_bytes())
    }

    fn ckks_sub_into_default<Dst, A, B>(&self, dst: &mut Dst, a: &A, b: &B, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWESub<BE> + GLWEShift<BE> + GLWENormalize<BE>,
        Dst: GLWEToBackendMut<BE> + LWEInfos + SetCKKSInfos + CKKSInfos,
        A: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        B: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        self.ckks_sub_into_unsafe_default(dst, a, b, scratch)?;
        self.glwe_normalize_assign(&mut GLWEToBackendMut::to_backend_mut(dst), scratch);
        Ok(())
    }

    fn ckks_sub_into_unsafe_default<Dst, A, B>(
        &self,
        dst: &mut Dst,
        a: &A,
        b: &B,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWESub<BE> + GLWEShift<BE>,
        Dst: GLWEToBackendMut<BE> + LWEInfos + SetCKKSInfos + CKKSInfos,
        A: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        B: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        let offset = ckks_offset_binary(dst, a, b);

        if offset == 0 && a.log_budget() == b.log_budget() {
            self.glwe_sub(dst, a, b);
        } else if a.log_budget() <= b.log_budget() {
            self.glwe_lsh(dst, a, offset, scratch);
            self.glwe_lsh_sub(dst, b, b.log_budget() - a.log_budget() + offset, scratch);
        } else {
            self.glwe_lsh(dst, a, a.log_budget() - b.log_budget() + offset, scratch);
            self.glwe_lsh_sub(dst, b, offset, scratch);
        }

        let log_budget = checked_log_budget_sub("sub", a.log_budget().min(b.log_budget()), offset)?;
        dst.set_log_delta(a.log_delta().min(b.log_delta()));
        dst.set_log_budget(log_budget);
        Ok(())
    }

    fn ckks_sub_assign_default<Dst, A>(&self, dst: &mut Dst, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWESub<BE> + GLWEShift<BE> + GLWENormalize<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        self.ckks_sub_assign_unsafe_default(dst, a, scratch)?;
        self.glwe_normalize_assign(&mut GLWEToBackendMut::to_backend_mut(dst), scratch);
        Ok(())
    }

    fn ckks_sub_assign_unsafe_default<Dst, A>(&self, dst: &mut Dst, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWESub<BE> + GLWEShift<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        let dst_log_budget = dst.log_budget();

        if dst_log_budget < a.log_budget() {
            self.glwe_lsh_sub(dst, a, a.log_budget() - dst_log_budget, scratch);
        } else if dst_log_budget > a.log_budget() {
            self.glwe_lsh_assign(dst, dst_log_budget - a.log_budget(), scratch);
            self.glwe_sub_assign(dst, a);
        } else {
            self.glwe_sub_assign(dst, a);
        }

        dst.set_log_budget(dst_log_budget.min(a.log_budget()));
        dst.set_log_delta(dst.log_delta().min(a.log_delta()));
        Ok(())
    }

    fn ckks_sub_pt_vec_znx_into_default<Dst, A, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshSubBackend<BE> + GLWEShift<BE> + GLWENormalize<BE> + CKKSPlaintextDefault<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + LWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + LWEInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        self.ckks_sub_pt_vec_znx_into_unsafe_default(dst, a, pt_znx, scratch)?;
        self.glwe_normalize_assign(&mut GLWEToBackendMut::to_backend_mut(dst), scratch);
        Ok(())
    }

    fn ckks_sub_pt_vec_znx_into_unsafe_default<Dst, A, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshSubBackend<BE> + GLWEShift<BE> + CKKSPlaintextDefault<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + LWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + LWEInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        let offset = ckks_offset_unary(dst, a);
        self.glwe_lsh(dst, a, offset, scratch);
        dst.set_meta(a.meta());
        dst.set_log_budget(checked_log_budget_sub("sub_pt_vec_znx", a.log_budget(), offset)?);
        self.ckks_sub_pt_vec_znx_assign_unsafe_default(dst, pt_znx, scratch)?;
        Ok(())
    }

    fn ckks_sub_pt_vec_znx_assign_default<Dst, P>(
        &self,
        dst: &mut Dst,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshSubBackend<BE> + GLWENormalize<BE> + CKKSPlaintextDefault<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + LWEInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        self.ckks_sub_pt_vec_znx_assign_unsafe_default(dst, pt_znx, scratch)?;
        self.glwe_normalize_assign(&mut GLWEToBackendMut::to_backend_mut(dst), scratch);
        Ok(())
    }

    fn ckks_sub_pt_vec_znx_assign_unsafe_default<Dst, P>(
        &self,
        dst: &mut Dst,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshSubBackend<BE> + CKKSPlaintextDefault<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + LWEInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        CKKSPlaintextDefault::ckks_sub_pt_vec_znx_into_default(self, dst, pt_znx, scratch)?;
        Ok(())
    }

    fn ckks_sub_pt_const_znx_into_default<Dst, A, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        dst_coeff: usize,
        cst_znx: &P,
        cst_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE> + GLWENormalize<BE> + VecZnxRshSubCoeffIntoBackend<BE> + CKKSPlaintextDefault<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + LWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + LWEInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        self.ckks_sub_pt_const_znx_into_unsafe_default(dst, a, dst_coeff, cst_znx, cst_coeff, scratch)?;
        self.glwe_normalize_assign(&mut GLWEToBackendMut::to_backend_mut(dst), scratch);
        Ok(())
    }

    fn ckks_sub_pt_const_znx_into_unsafe_default<Dst, A, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        dst_coeff: usize,
        cst_znx: &P,
        cst_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE> + VecZnxRshSubCoeffIntoBackend<BE> + CKKSPlaintextDefault<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + LWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + LWEInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        let offset = ckks_offset_unary(dst, a);
        self.glwe_lsh(dst, a, offset, scratch);
        dst.set_meta(a.meta());
        dst.set_log_budget(checked_log_budget_sub("sub_pt_const_znx", a.log_budget(), offset)?);
        self.ckks_sub_pt_const_znx_assign_unsafe_default(dst, dst_coeff, cst_znx, cst_coeff, scratch)
    }

    fn ckks_sub_pt_const_znx_assign_default<Dst, P>(
        &self,
        dst: &mut Dst,
        dst_coeff: usize,
        cst_znx: &P,
        cst_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshSubCoeffIntoBackend<BE> + CKKSPlaintextDefault<BE> + GLWENormalize<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + LWEInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        self.ckks_sub_pt_const_znx_assign_unsafe_default(dst, dst_coeff, cst_znx, cst_coeff, scratch)?;
        self.glwe_normalize_assign(&mut GLWEToBackendMut::to_backend_mut(dst), scratch);
        Ok(())
    }

    fn ckks_sub_pt_const_znx_assign_unsafe_default<Dst, P>(
        &self,
        dst: &mut Dst,
        dst_coeff: usize,
        cst_znx: &P,
        cst_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshSubCoeffIntoBackend<BE> + CKKSPlaintextDefault<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + LWEInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        CKKSPlaintextDefault::ckks_sub_pt_cst_znx_into_default(self, dst, dst_coeff, cst_znx, cst_coeff, scratch)?;
        Ok(())
    }
}

impl<BE: Backend> CKKSSubDefault<BE> for Module<BE> {}
