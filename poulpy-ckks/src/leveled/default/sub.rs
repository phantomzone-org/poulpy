use anyhow::Result;
use poulpy_core::{
    GLWENormalize, GLWEShift, GLWESub, ScratchArenaTakeCore,
    layouts::{
        GLWEInfos, GLWEPlaintext, GLWEPlaintextLayout, GLWEPlaintextToBackendRef, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
        glwe_backend_data_mut,
    },
};
use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, VecZnxAddConstAssignBackend, VecZnxRshSubBackend, VecZnxRshTmpBytes},
    layouts::{Backend, Data, HostBackend, Module, ScratchArena},
};

use crate::{
    CKKSInfos, CKKSMeta, checked_log_budget_sub,
    layouts::{
        CKKSCiphertext, CKKSModuleAlloc,
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

    fn ckks_sub_pt_const_tmp_bytes_default(&self) -> usize
    where
        Self: GLWEShift<BE> + GLWENormalize<BE>,
    {
        self.glwe_shift_tmp_bytes().max(self.glwe_normalize_tmp_bytes())
    }

    fn ckks_sub_into_default<Dst, A, B>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        b: &CKKSCiphertext<B>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        A: Data,
        B: Data,
        Self: GLWESub<BE> + GLWEShift<BE> + GLWENormalize<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        CKKSCiphertext<B>: GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        self.ckks_sub_into_unsafe_default(dst, a, b, scratch)?;
        self.glwe_normalize_assign(&mut GLWEToBackendMut::to_backend_mut(dst), scratch);
        Ok(())
    }

    fn ckks_sub_into_unsafe_default<Dst, A, B>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        b: &CKKSCiphertext<B>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        A: Data,
        B: Data,
        Self: GLWESub<BE> + GLWEShift<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        CKKSCiphertext<B>: GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        let offset = dst.offset_binary(a, b);

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
        dst.meta.log_delta = a.log_delta().min(b.log_delta());
        dst.meta.log_budget = log_budget;
        Ok(())
    }

    fn ckks_sub_assign_default<Dst, A>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        A: Data,
        Self: GLWESub<BE> + GLWEShift<BE> + GLWENormalize<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        self.ckks_sub_assign_unsafe_default(dst, a, scratch)?;
        self.glwe_normalize_assign(&mut GLWEToBackendMut::to_backend_mut(dst), scratch);
        Ok(())
    }

    fn ckks_sub_assign_unsafe_default<Dst, A>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        A: Data,
        Self: GLWESub<BE> + GLWEShift<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
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

        dst.meta.log_budget = dst_log_budget.min(a.log_budget());
        dst.meta.log_delta = dst.log_delta().min(a.log_delta());
        Ok(())
    }

    fn ckks_sub_pt_vec_znx_into_default<Dst, A, P>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt_znx: &CKKSPlaintextVecZnx<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        A: Data,
        P: Data,
        Self: VecZnxRshSubBackend<BE> + GLWEShift<BE> + GLWENormalize<BE> + CKKSPlaintextZnxDefault<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        CKKSPlaintextVecZnx<P>: GLWEPlaintextToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        self.ckks_sub_pt_vec_znx_into_unsafe_default(dst, a, pt_znx, scratch)?;
        self.glwe_normalize_assign(&mut GLWEToBackendMut::to_backend_mut(dst), scratch);
        Ok(())
    }

    fn ckks_sub_pt_vec_znx_into_unsafe_default<Dst, A, P>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt_znx: &CKKSPlaintextVecZnx<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        A: Data,
        P: Data,
        Self: VecZnxRshSubBackend<BE> + GLWEShift<BE> + CKKSPlaintextZnxDefault<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        CKKSPlaintextVecZnx<P>: GLWEPlaintextToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        let offset = dst.offset_unary(a);
        self.glwe_lsh(dst, a, offset, scratch);
        dst.meta = a.meta();
        dst.meta.log_budget = checked_log_budget_sub("sub_pt_vec_znx", a.log_budget(), offset)?;
        self.ckks_sub_pt_vec_znx_assign_unsafe_default(dst, pt_znx, scratch)?;
        Ok(())
    }

    fn ckks_sub_pt_vec_znx_assign_default<Dst, P>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        pt_znx: &CKKSPlaintextVecZnx<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        P: Data,
        Self: VecZnxRshSubBackend<BE> + GLWENormalize<BE> + CKKSPlaintextZnxDefault<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSPlaintextVecZnx<P>: GLWEPlaintextToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        self.ckks_sub_pt_vec_znx_assign_unsafe_default(dst, pt_znx, scratch)?;
        self.glwe_normalize_assign(&mut GLWEToBackendMut::to_backend_mut(dst), scratch);
        Ok(())
    }

    fn ckks_sub_pt_vec_znx_assign_unsafe_default<Dst, P>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        pt_znx: &CKKSPlaintextVecZnx<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        P: Data,
        Self: VecZnxRshSubBackend<BE> + CKKSPlaintextZnxDefault<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSPlaintextVecZnx<P>: GLWEPlaintextToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        CKKSPlaintextZnxDefault::ckks_sub_pt_vec_znx_into_default(self, dst, pt_znx, scratch)?;
        Ok(())
    }

    fn ckks_sub_pt_vec_rnx_into_default<Dst, A, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        A: Data,
        Self: ModuleN
            + VecZnxRshSubBackend<BE>
            + GLWEShift<BE>
            + GLWENormalize<BE>
            + CKKSPlaintextZnxDefault<BE>
            + CKKSModuleAlloc<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        BE: HostBackend<OwnedBuf = Vec<u8>>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
    {
        self.ckks_sub_pt_vec_rnx_into_unsafe_default(dst, a, pt_rnx, prec, scratch)?;
        self.glwe_normalize_assign(&mut GLWEToBackendMut::to_backend_mut(dst), scratch);
        Ok(())
    }

    fn ckks_sub_pt_vec_rnx_into_unsafe_default<Dst, A, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        A: Data,
        Self: ModuleN + VecZnxRshSubBackend<BE> + GLWEShift<BE> + CKKSPlaintextZnxDefault<BE> + CKKSModuleAlloc<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        BE: HostBackend<OwnedBuf = Vec<u8>>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
    {
        let mut pt_znx_host = self.ckks_pt_vec_znx_alloc(dst.base2k(), prec);
        pt_rnx.to_znx(&mut pt_znx_host)?;
        CKKSSubDefault::ckks_sub_pt_vec_znx_into_unsafe_default(self, dst, a, &pt_znx_host, scratch)?;
        Ok(())
    }

    fn ckks_sub_pt_vec_rnx_assign_default<Dst, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        Self: ModuleN + VecZnxRshSubBackend<BE> + GLWENormalize<BE> + CKKSPlaintextZnxDefault<BE> + CKKSModuleAlloc<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        BE: HostBackend<OwnedBuf = Vec<u8>>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
    {
        self.ckks_sub_pt_vec_rnx_assign_unsafe_default(dst, pt_rnx, prec, scratch)?;
        self.glwe_normalize_assign(&mut GLWEToBackendMut::to_backend_mut(dst), scratch);
        Ok(())
    }

    fn ckks_sub_pt_vec_rnx_assign_unsafe_default<Dst, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        Self: ModuleN + VecZnxRshSubBackend<BE> + CKKSPlaintextZnxDefault<BE> + CKKSModuleAlloc<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        BE: HostBackend<OwnedBuf = Vec<u8>>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
    {
        let mut pt_znx_host = self.ckks_pt_vec_znx_alloc(dst.base2k(), prec);
        pt_rnx.to_znx(&mut pt_znx_host)?;
        self.ckks_sub_pt_vec_znx_assign_unsafe_default(dst, &pt_znx_host, scratch)?;
        Ok(())
    }

    fn ckks_sub_pt_const_znx_into_default<Dst, A>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        A: Data,
        Self: GLWEShift<BE> + GLWENormalize<BE> + VecZnxAddConstAssignBackend<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        self.ckks_sub_pt_const_znx_into_unsafe_default(dst, a, cst_znx, scratch)?;
        self.glwe_normalize_assign(&mut GLWEToBackendMut::to_backend_mut(dst), scratch);
        Ok(())
    }

    fn ckks_sub_pt_const_znx_into_unsafe_default<Dst, A>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        A: Data,
        Self: GLWEShift<BE> + VecZnxAddConstAssignBackend<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        let offset = dst.offset_unary(a);
        self.glwe_lsh(dst, a, offset, scratch);
        dst.meta = a.meta();
        dst.meta.log_budget = checked_log_budget_sub("sub_pt_const_znx", a.log_budget(), offset)?;
        self.ckks_sub_pt_const_znx_assign_unsafe_default(dst, cst_znx, scratch)
    }

    fn ckks_sub_pt_const_znx_assign_default<Dst>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        Self: VecZnxAddConstAssignBackend<BE> + GLWENormalize<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        self.ckks_sub_pt_const_znx_assign_unsafe_default(dst, cst_znx, scratch)?;
        self.glwe_normalize_assign(&mut GLWEToBackendMut::to_backend_mut(dst), scratch);
        Ok(())
    }

    fn ckks_sub_pt_const_znx_assign_unsafe_default<Dst>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        cst_znx: &CKKSPlaintextCstZnx,
        _scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        Self: VecZnxAddConstAssignBackend<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        if cst_znx.re().is_none() && cst_znx.im().is_none() {
            return Ok(());
        }

        let _offset = crate::ensure_plaintext_alignment(
            "ckks_sub_pt_const_znx_into",
            dst.log_budget(),
            cst_znx.log_delta(),
            cst_znx.effective_k(),
        )?;
        let n = dst.n().as_usize();
        let mut dst_backend = GLWEToBackendMut::to_backend_mut(dst);
        let mut dst_data = glwe_backend_data_mut::<BE>(&mut dst_backend);
        if let Some(coeff) = cst_znx.re() {
            let neg_coeff: Vec<i64> = coeff.iter().map(|digit| digit.wrapping_neg()).collect();
            self.vec_znx_add_const_assign_backend(&mut dst_data, 0, &neg_coeff, 0, 0);
        }
        if let Some(coeff) = cst_znx.im() {
            let neg_coeff: Vec<i64> = coeff.iter().map(|digit| digit.wrapping_neg()).collect();
            self.vec_znx_add_const_assign_backend(&mut dst_data, 0, &neg_coeff, 0, n / 2);
        }
        Ok(())
    }

    fn ckks_sub_pt_const_rnx_into_default<Dst, A, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        A: Data,
        Self: GLWEShift<BE> + GLWENormalize<BE> + VecZnxAddConstAssignBackend<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion,
    {
        self.ckks_sub_pt_const_rnx_into_unsafe_default(dst, a, cst_rnx, prec, scratch)?;
        self.glwe_normalize_assign(&mut GLWEToBackendMut::to_backend_mut(dst), scratch);
        Ok(())
    }

    fn ckks_sub_pt_const_rnx_into_unsafe_default<Dst, A, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        A: Data,
        Self: GLWEShift<BE> + VecZnxAddConstAssignBackend<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion,
    {
        let offset = dst.offset_unary(a);
        if cst_rnx.re().is_none() && cst_rnx.im().is_none() {
            self.glwe_lsh(dst, a, offset, scratch);
            dst.meta = a.meta();
            dst.meta.log_budget = checked_log_budget_sub("sub_pt_const_rnx", a.log_budget(), offset)?;
            return Ok(());
        }

        let res_log_budget = checked_log_budget_sub("sub_pt_const_rnx", a.log_budget(), offset)?;
        let cst_znx = cst_rnx.to_znx_at_k(
            dst.base2k(),
            res_log_budget
                .checked_add(prec.log_delta)
                .ok_or_else(|| anyhow::anyhow!("sub_pt_const_rnx: aligned precision overflow"))?,
            prec.log_delta,
        )?;
        self.ckks_sub_pt_const_znx_into_unsafe_default(dst, a, &cst_znx, scratch)
    }

    fn ckks_sub_pt_const_rnx_assign_default<Dst, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        Self: VecZnxAddConstAssignBackend<BE> + GLWENormalize<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion,
    {
        self.ckks_sub_pt_const_rnx_assign_unsafe_default(dst, cst_rnx, prec, scratch)?;
        self.glwe_normalize_assign(&mut GLWEToBackendMut::to_backend_mut(dst), scratch);
        Ok(())
    }

    fn ckks_sub_pt_const_rnx_assign_unsafe_default<Dst, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        Self: VecZnxAddConstAssignBackend<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion,
    {
        if cst_rnx.re().is_none() && cst_rnx.im().is_none() {
            return Ok(());
        }

        let cst_znx = cst_rnx.to_znx_at_k(
            dst.base2k(),
            dst.log_budget()
                .checked_add(prec.log_delta)
                .ok_or_else(|| anyhow::anyhow!("sub_pt_const_rnx_assign: aligned precision overflow"))?,
            prec.log_delta,
        )?;
        CKKSSubDefault::ckks_sub_pt_const_znx_assign_unsafe_default::<Dst>(self, dst, &cst_znx, scratch)
    }
}

impl<BE: Backend> CKKSSubDefault<BE> for Module<BE> {}
