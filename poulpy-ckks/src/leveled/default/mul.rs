use anyhow::Result;
use poulpy_core::{
    GLWEAdd, GLWECopy, GLWEMulConst, GLWEMulPlain, GLWERotate, GLWETensoring, ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GLWE, GLWEInfos, GLWELayout, GLWEPlaintext, GLWEPlaintextLayout, GLWEPlaintextToBackendMut,
        GLWEPlaintextToBackendRef, GLWETensor, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
        ModuleCoreAlloc, TorusPrecision, glwe_backend_data_mut, prepared::GLWETensorKeyPreparedToBackendRef,
    },
};
use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, VecZnxCopyBackend, VecZnxZeroBackend},
    layouts::{Backend, Data, HostBackend, Module, ScratchArena},
};

use crate::{
    CKKSInfos, CKKSMeta, checked_log_budget_sub, checked_mul_ct_log_budget, checked_mul_pt_log_budget,
    layouts::{
        CKKSCiphertext, CKKSModuleAlloc,
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

        let lvl_0 = GLWETensor::bytes_of_from_infos(&glwe_layout);
        let lvl_1 = self
            .glwe_tensor_apply_tmp_bytes(&glwe_layout, res, res)
            .max(self.glwe_tensor_relinearize_tmp_bytes(res, &glwe_layout, tsk));

        lvl_0 + lvl_1
    }

    fn ckks_mul_into_default<Dst, A, B, T>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        b: &CKKSCiphertext<B>,
        tsk: &GLWETensorKeyPrepared<T, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        A: Data,
        B: Data,
        T: Data,
        Self: GLWETensoring<BE> + GLWECopy<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + GLWEInfos,
        CKKSCiphertext<B>: GLWEToBackendRef<BE> + GLWEInfos,
        GLWETensorKeyPrepared<T, BE>: GLWETensorKeyPreparedToBackendRef<BE>,
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

        dst.meta.log_budget = res_log_budget;
        dst.meta.log_delta = res_log_delta;
        Ok(())
    }

    fn ckks_mul_assign_default<Dst, A, T>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        tsk: &GLWETensorKeyPrepared<T, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        A: Data,
        T: Data,
        Self: GLWETensoring<BE> + GLWECopy<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + GLWEInfos,
        GLWETensorKeyPrepared<T, BE>: GLWETensorKeyPreparedToBackendRef<BE>,
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

        dst.meta.log_budget = res_log_budget;
        dst.meta.log_delta = res_log_delta;
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
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        tsk: &GLWETensorKeyPrepared<T, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        A: Data,
        T: Data,
        Self: GLWETensoring<BE> + GLWECopy<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + GLWEInfos,
        GLWETensorKeyPrepared<T, BE>: GLWETensorKeyPreparedToBackendRef<BE>,
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

        dst.meta.log_budget = res_log_budget;
        dst.meta.log_delta = res_log_delta;
        Ok(())
    }

    fn ckks_square_assign_default<Dst, T>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        tsk: &GLWETensorKeyPrepared<T, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        T: Data,
        Self: GLWETensoring<BE> + GLWECopy<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
        GLWETensorKeyPrepared<T, BE>: GLWETensorKeyPreparedToBackendRef<BE>,
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

        dst.meta.log_budget = res_log_budget;
        dst.meta.log_delta = res_log_delta;
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

    fn ckks_mul_pt_const_tmp_bytes_default<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
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

    fn ckks_mul_pt_vec_znx_into_default<Dst, A, P>(
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
        Self: GLWECopy<BE> + GLWEMulPlain<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf> + VecZnxCopyBackend<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + GLWEInfos,
        CKKSPlaintextVecZnx<P>: poulpy_core::layouts::GLWEPlaintextToBackendRef<BE>,
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
        dst.meta.log_budget = res_log_budget;
        dst.meta.log_delta = res_log_delta;
        Ok(())
    }

    fn ckks_mul_pt_vec_znx_assign_default<Dst, P>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        pt_znx: &CKKSPlaintextVecZnx<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        P: Data,
        Self: GLWECopy<BE> + GLWEMulPlain<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf> + VecZnxCopyBackend<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
        CKKSPlaintextVecZnx<P>: poulpy_core::layouts::GLWEPlaintextToBackendRef<BE>,
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
        dst.meta.log_budget = res_log_budget;
        dst.meta.log_delta = res_log_delta;
        Ok(())
    }

    fn ckks_mul_pt_vec_rnx_into_default<Dst, A, F>(
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
        Self: GLWECopy<BE>
            + GLWEMulPlain<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
            + ModuleN
            + VecZnxCopyBackend<BE>
            + CKKSModuleAlloc<BE>,
        BE: HostBackend<OwnedBuf = Vec<u8>>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + GLWEInfos,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        let mut pt_znx_host = self.ckks_pt_vec_znx_alloc(dst.base2k(), prec);
        pt_rnx.to_znx(&mut pt_znx_host)?;
        self.ckks_mul_pt_vec_znx_into_default(dst, a, &pt_znx_host, scratch)
    }

    fn ckks_mul_pt_vec_rnx_assign_default<Dst, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        Self: GLWECopy<BE>
            + GLWEMulPlain<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
            + ModuleN
            + VecZnxCopyBackend<BE>
            + CKKSModuleAlloc<BE>,
        BE: HostBackend<OwnedBuf = Vec<u8>>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        let mut pt_znx_host = self.ckks_pt_vec_znx_alloc(dst.base2k(), prec);
        pt_rnx.to_znx(&mut pt_znx_host)?;
        self.ckks_mul_pt_vec_znx_assign_default(dst, &pt_znx_host, scratch)
    }

    fn ckks_mul_pt_const_znx_into_default<Dst, A>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        A: Data,
        Self: GLWEAdd<BE> + GLWECopy<BE> + GLWEMulConst<BE> + GLWERotate<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_const_params(dst, a, cst_znx.meta())?;
        let rotate_by = (dst.n().as_usize() / 2) as i64;
        let a_owned = copy_ct_to_owned::<BE, _, A>(self, a);
        let mut dst_owned = self.glwe_alloc_from_infos(dst);
        match (cst_znx.re(), cst_znx.im()) {
            (None, None) => zero_ciphertext::<BE, Dst, Self>(self, dst),
            (Some(re_const), None) => {
                self.glwe_mul_const(cnv_offset, &mut dst_owned, &a_owned, re_const, scratch);
                copy_ct_from_owned::<BE, _, Dst>(self, dst, &dst_owned);
            }
            (None, Some(im_const)) => {
                self.glwe_mul_const(cnv_offset, &mut dst_owned, &a_owned, im_const, scratch);
                {
                    let mut dst_owned_ref = GLWEToBackendMut::<BE>::to_backend_mut(&mut dst_owned);
                    self.glwe_rotate_assign(rotate_by, &mut dst_owned_ref, scratch);
                }
                copy_ct_from_owned::<BE, _, Dst>(self, dst, &dst_owned);
            }
            (Some(re_const), Some(im_const)) => {
                let mut tmp = self.glwe_alloc_from_infos(dst);
                self.glwe_mul_const(cnv_offset, &mut dst_owned, &a_owned, re_const, scratch);
                self.glwe_mul_const(cnv_offset, &mut tmp, &a_owned, im_const, scratch);
                {
                    let mut tmp_ref = GLWEToBackendMut::<BE>::to_backend_mut(&mut tmp);
                    self.glwe_rotate_assign(rotate_by, &mut tmp_ref, scratch);
                }
                self.glwe_add_assign(&mut dst_owned, &tmp);
                copy_ct_from_owned::<BE, _, Dst>(self, dst, &dst_owned);
            }
        }

        dst.meta.log_budget = res_log_budget;
        dst.meta.log_delta = res_log_delta;
        Ok(())
    }

    fn ckks_mul_pt_const_znx_assign_default<Dst>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        Self: GLWEAdd<BE> + GLWECopy<BE> + GLWEMulConst<BE> + GLWERotate<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_const_params(dst, dst, cst_znx.meta())?;
        let rotate_by = (dst.n().as_usize() / 2) as i64;
        let mut dst_owned = copy_ct_to_owned::<BE, _, Dst>(self, dst);
        match (cst_znx.re(), cst_znx.im()) {
            (None, None) => zero_ciphertext::<BE, Dst, Self>(self, dst),
            (Some(re_const), None) => {
                self.glwe_mul_const_assign(cnv_offset, &mut dst_owned, re_const, scratch);
                copy_ct_from_owned::<BE, _, Dst>(self, dst, &dst_owned);
            }
            (None, Some(im_const)) => {
                self.glwe_mul_const_assign(cnv_offset, &mut dst_owned, im_const, scratch);
                {
                    let mut dst_owned_ref = GLWEToBackendMut::<BE>::to_backend_mut(&mut dst_owned);
                    self.glwe_rotate_assign(rotate_by, &mut dst_owned_ref, scratch);
                }
                copy_ct_from_owned::<BE, _, Dst>(self, dst, &dst_owned);
            }
            (Some(re_const), Some(im_const)) => {
                let mut tmp = self.glwe_alloc_from_infos(dst);
                let dst_src = copy_ct_to_owned::<BE, _, Dst>(self, dst);
                self.glwe_mul_const(cnv_offset, &mut tmp, &dst_src, im_const, scratch);
                self.glwe_mul_const_assign(cnv_offset, &mut dst_owned, re_const, scratch);
                {
                    let mut tmp_ref = GLWEToBackendMut::<BE>::to_backend_mut(&mut tmp);
                    self.glwe_rotate_assign(rotate_by, &mut tmp_ref, scratch);
                }
                self.glwe_add_assign(&mut dst_owned, &tmp);
                copy_ct_from_owned::<BE, _, Dst>(self, dst, &dst_owned);
            }
        }

        dst.meta.log_budget = res_log_budget;
        dst.meta.log_delta = res_log_delta;
        Ok(())
    }

    fn ckks_mul_pt_const_rnx_into_default<Dst, A, F>(
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
        Self: GLWEAdd<BE> + GLWECopy<BE> + GLWEMulConst<BE> + GLWERotate<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion,
    {
        if cst_rnx.re().is_none() && cst_rnx.im().is_none() {
            let (res_log_budget, res_log_delta, _) = get_mul_const_params(dst, a, prec)?;
            zero_ciphertext::<BE, Dst, Self>(self, dst);
            dst.meta.log_budget = res_log_budget;
            dst.meta.log_delta = res_log_delta;
            return Ok(());
        }

        let cst_znx = cst_rnx.to_znx(dst.base2k(), prec)?;
        self.ckks_mul_pt_const_znx_into_default(dst, a, &cst_znx, scratch)
    }

    fn ckks_mul_pt_const_rnx_assign_default<Dst, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: Data,
        Self: GLWEAdd<BE> + GLWECopy<BE> + GLWEMulConst<BE> + GLWERotate<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion,
    {
        if cst_rnx.re().is_none() && cst_rnx.im().is_none() {
            let (res_log_budget, res_log_delta, _) = get_mul_const_params(dst, dst, prec)?;
            zero_ciphertext::<BE, Dst, Self>(self, dst);
            dst.meta.log_budget = res_log_budget;
            dst.meta.log_delta = res_log_delta;
            return Ok(());
        }

        let cst_znx = cst_rnx.to_znx(dst.base2k(), prec)?;
        self.ckks_mul_pt_const_znx_assign_default(dst, &cst_znx, scratch)
    }
}

impl<BE: Backend> CKKSMulDefault<BE> for Module<BE> {}

fn copy_ct_to_owned<BE: Backend, M, D>(module: &M, src: &CKKSCiphertext<D>) -> GLWE<BE::OwnedBuf>
where
    M: GLWECopy<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf> + ?Sized,
    D: Data,
    CKKSCiphertext<D>: GLWEToBackendRef<BE> + GLWEInfos,
{
    let mut dst = module.glwe_alloc_from_infos(src);
    let src_ref = GLWEToBackendRef::<BE>::to_backend_ref(src);
    module.glwe_copy(&mut GLWEToBackendMut::<BE>::to_backend_mut(&mut dst), &src_ref);
    dst
}

fn copy_ct_from_owned<BE: Backend, M, D>(module: &M, dst: &mut CKKSCiphertext<D>, src: &GLWE<BE::OwnedBuf>)
where
    M: GLWECopy<BE> + ?Sized,
    D: Data,
    CKKSCiphertext<D>: GLWEToBackendMut<BE>,
{
    let src_ref = GLWEToBackendRef::<BE>::to_backend_ref(src);
    module.glwe_copy(&mut GLWEToBackendMut::<BE>::to_backend_mut(dst), &src_ref);
}

fn copy_pt_to_owned<BE: Backend, M, D>(module: &M, src: &CKKSPlaintextVecZnx<D>) -> GLWEPlaintext<BE::OwnedBuf>
where
    M: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf> + VecZnxCopyBackend<BE> + ?Sized,
    D: Data,
    CKKSPlaintextVecZnx<D>: GLWEPlaintextToBackendRef<BE>,
{
    let mut dst = module.glwe_plaintext_alloc_from_infos(src);
    let src_ref = GLWEPlaintextToBackendRef::<BE>::to_backend_ref(src);
    {
        let mut dst_mut = GLWEPlaintextToBackendMut::<BE>::to_backend_mut(&mut dst);
        module.vec_znx_copy_backend(&mut dst_mut.data, 0, &src_ref.data, 0);
    }
    dst
}

fn zero_ciphertext<BE: Backend, Dst: Data, M>(module: &M, dst: &mut CKKSCiphertext<Dst>)
where
    M: VecZnxZeroBackend<BE> + ?Sized,
    CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
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
