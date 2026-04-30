use anyhow::{Result, bail, ensure};
use poulpy_core::{
    GLWEAdd, GLWECopy, GLWEMulConst, GLWEMulPlain, GLWENormalize, GLWERotate, GLWEShift, GLWESub, GLWETensoring,
    ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GLWE, GLWEInfos, GLWELayout, GLWETensor, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
        ModuleCoreAlloc, TorusPrecision,
    },
};
use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, VecZnxCopyBackend, VecZnxRshAddIntoBackend},
    layouts::{Backend, Data, HostBackend, Module, ScratchArena},
};

use crate::{
    CKKSInfos, CKKSMeta,
    layouts::{
        CKKSCiphertext, CKKSModuleAlloc,
        ciphertext::CKKSOffset,
        plaintext::{
            CKKSConstPlaintextConversion, CKKSPlaintextConversion, CKKSPlaintextCstRnx, CKKSPlaintextCstZnx, CKKSPlaintextVecRnx,
            CKKSPlaintextVecZnx,
        },
    },
    leveled::api::{
        CKKSAddManyOps, CKKSAddOps, CKKSAddOpsUnsafe, CKKSDotProductOps, CKKSMulAddOps, CKKSMulManyOps, CKKSMulOps,
        CKKSMulSubOps, CKKSRescaleOps, CKKSSubOps,
    },
    oep::CKKSImpl,
};

fn take_mul_tmp<BE: Backend, D: Data>(module: &Module<BE>, dst: &CKKSCiphertext<D>) -> CKKSCiphertext<BE::OwnedBuf>
where
    Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
{
    module.ckks_ciphertext_alloc_from_infos(dst)
}

fn ensure_accumulation_fits<D: Data>(op: &'static str, dst: &CKKSCiphertext<D>, n: usize) -> Result<()> {
    let base2k: usize = dst.base2k().as_usize();
    ensure!(base2k < 64, "{op}: unsupported base2k={base2k}");
    ensure!(
        n <= (1usize << (63 - base2k)),
        "{op}: {n} terms risks i64 overflow at base2k={base2k}",
    );
    Ok(())
}

// --- CKKSAddManyOps ---

impl<BE: Backend + CKKSImpl<BE>> CKKSAddManyOps<BE> for Module<BE> {
    fn ckks_add_many_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE> + CKKSAddOps<BE>,
    {
        self.ckks_add_tmp_bytes()
    }

    fn ckks_add_many<Dst: Data, Src: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        inputs: &[&CKKSCiphertext<Src>],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSAddOps<BE> + CKKSRescaleOps<BE> + GLWEAdd<BE> + GLWEShift<BE> + GLWENormalize<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<Src>: GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        match inputs.len() {
            0 => bail!("ckks_add_many: inputs must contain at least one ciphertext"),
            1 => {
                self.ckks_rescale_into(dst, dst.offset_unary(inputs[0]), inputs[0], scratch)?;
            }
            _ => {
                ensure_accumulation_fits("ckks_add_many", dst, inputs.len())?;
                self.ckks_add_into(dst, inputs[0], inputs[1], scratch)?;
                for ct in &inputs[2..] {
                    self.ckks_add_assign(dst, ct, scratch)?;
                }
            }
        }
        Ok(())
    }
}

// --- CKKSMulManyOps ---

fn ceil_log2(n: usize) -> usize {
    debug_assert!(n >= 1);
    if n <= 1 { 0 } else { (n - 1).ilog2() as usize + 1 }
}

impl<BE: Backend + CKKSImpl<BE>> CKKSMulManyOps<BE> for Module<BE> {
    fn ckks_mul_many_tmp_bytes<R, T>(&self, n: usize, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Self: GLWETensoring<BE> + CKKSMulOps<BE>,
    {
        let mul_scratch: usize = self.ckks_mul_tmp_bytes(res, tsk);
        if n <= 2 {
            return mul_scratch;
        }
        let depth: usize = ceil_log2(n);
        2 * depth * GLWE::<Vec<u8>>::bytes_of_from_infos(res) + mul_scratch
    }

    fn ckks_mul_many<Dst: Data, Src: Data, T: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        inputs: &[&CKKSCiphertext<Src>],
        tsk: &GLWETensorKeyPrepared<T, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWECopy<BE>
            + GLWETensoring<BE>
            + CKKSMulOps<BE>
            + CKKSRescaleOps<BE>
            + GLWEShift<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
        CKKSCiphertext<Src>: GLWEToBackendRef<BE> + GLWEInfos,
        GLWETensorKeyPrepared<T, BE>: poulpy_core::layouts::prepared::GLWETensorKeyPreparedToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        if inputs.is_empty() {
            bail!("ckks_mul_many: inputs must contain at least one ciphertext");
        }
        anyhow::ensure!(
            inputs.iter().all(|c| c.log_delta() == inputs[0].log_delta()),
            "ckks_mul_many: all inputs must have the same log_delta"
        );
        if inputs.len() == 1 {
            self.ckks_rescale_into(dst, dst.offset_unary(inputs[0]), inputs[0], scratch)?;
            return Ok(());
        }

        let mut acc = take_mul_tmp(self, inputs[0]);
        self.ckks_mul_into::<BE::OwnedBuf, Src, Src, T>(&mut acc, inputs[0], inputs[1], tsk, scratch)?;

        for ct in &inputs[2..] {
            let mut compact = self.ckks_ciphertext_alloc(acc.base2k(), acc.effective_k().into());
            self.ckks_rescale_into::<BE::OwnedBuf, BE::OwnedBuf>(&mut compact, 0, &acc, scratch)?;

            let mut next = take_mul_tmp(self, inputs[0]);
            self.ckks_mul_into::<BE::OwnedBuf, BE::OwnedBuf, Src, T>(&mut next, &compact, ct, tsk, scratch)?;
            acc = next;
        }

        self.ckks_rescale_into::<Dst, BE::OwnedBuf>(dst, dst.offset_unary(&acc), &acc, scratch)
    }
}

// --- CKKSMulAddOps ---

impl<BE: Backend + CKKSImpl<BE>> CKKSMulAddOps<BE> for Module<BE> {
    fn ckks_mul_add_ct_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Self: GLWEShift<BE> + GLWETensoring<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_tmp_bytes(res, tsk).max(self.ckks_add_tmp_bytes())
    }

    fn ckks_mul_add_pt_vec_znx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulPlain<BE> + GLWEShift<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_pt_vec_znx_tmp_bytes(res, a, b).max(self.ckks_add_tmp_bytes())
    }

    fn ckks_mul_add_pt_vec_rnx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: ModuleN + GLWEMulPlain<BE> + GLWEShift<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_pt_vec_rnx_tmp_bytes(res, a, b).max(self.ckks_add_tmp_bytes())
    }

    fn ckks_mul_add_pt_const_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulConst<BE> + GLWERotate<BE> + GLWEShift<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_pt_const_tmp_bytes(res, a, b).max(self.ckks_add_tmp_bytes())
    }

    fn ckks_mul_add_ct_into<Dst: Data, A: Data, B: Data, T: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        b: &CKKSCiphertext<B>,
        tsk: &GLWETensorKeyPrepared<T, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd<BE>
            + GLWECopy<BE>
            + GLWEShift<BE>
            + GLWETensoring<BE>
            + CKKSAddOps<BE>
            + CKKSMulOps<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + GLWEInfos,
        CKKSCiphertext<B>: GLWEToBackendRef<BE> + GLWEInfos,
        GLWETensorKeyPrepared<T, BE>: poulpy_core::layouts::prepared::GLWETensorKeyPreparedToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        let mut tmp = take_mul_tmp(self, dst);
        self.ckks_mul_into::<BE::OwnedBuf, A, B, T>(&mut tmp, a, b, tsk, scratch)?;
        self.ckks_add_assign::<Dst, BE::OwnedBuf>(dst, &tmp, scratch)
    }

    fn ckks_mul_add_pt_vec_znx_into<Dst: Data, A: Data, P: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt_znx: &CKKSPlaintextVecZnx<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd<BE>
            + GLWECopy<BE>
            + GLWEMulPlain<BE>
            + GLWEShift<BE>
            + CKKSAddOps<BE>
            + CKKSMulOps<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
            + VecZnxCopyBackend<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + GLWEInfos,
        CKKSPlaintextVecZnx<P>: poulpy_core::layouts::GLWEPlaintextToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        let mut tmp = take_mul_tmp(self, dst);
        self.ckks_mul_pt_vec_znx_into::<BE::OwnedBuf, A, P>(&mut tmp, a, pt_znx, scratch)?;
        self.ckks_add_assign::<Dst, BE::OwnedBuf>(dst, &tmp, scratch)
    }

    fn ckks_mul_add_pt_vec_rnx_into<Dst: Data, A: Data, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: ModuleN
            + GLWEAdd<BE>
            + GLWECopy<BE>
            + GLWEMulPlain<BE>
            + GLWEShift<BE>
            + CKKSAddOps<BE>
            + CKKSMulOps<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
            + VecZnxCopyBackend<BE>,
        BE: HostBackend<OwnedBuf = Vec<u8>>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
    {
        let mut tmp = take_mul_tmp(self, dst);
        self.ckks_mul_pt_vec_rnx_into::<BE::OwnedBuf, A, F>(&mut tmp, a, pt_rnx, prec, scratch)?;
        self.ckks_add_assign::<Dst, BE::OwnedBuf>(dst, &tmp, scratch)
    }

    fn ckks_mul_add_pt_const_znx_into<Dst: Data, A: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd<BE>
            + GLWECopy<BE>
            + GLWEMulConst<BE>
            + GLWERotate<BE>
            + GLWEShift<BE>
            + CKKSAddOps<BE>
            + CKKSMulOps<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        if cst_znx.re().is_none() && cst_znx.im().is_none() {
            return Ok(());
        }

        let mut tmp = take_mul_tmp(self, dst);
        self.ckks_mul_pt_const_znx_into::<BE::OwnedBuf, A>(&mut tmp, a, cst_znx, scratch)?;
        self.ckks_add_assign::<Dst, BE::OwnedBuf>(dst, &tmp, scratch)
    }

    fn ckks_mul_add_pt_const_rnx_into<Dst: Data, A: Data, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd<BE>
            + GLWECopy<BE>
            + GLWEMulConst<BE>
            + GLWERotate<BE>
            + GLWEShift<BE>
            + CKKSAddOps<BE>
            + CKKSMulOps<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion,
    {
        if cst_rnx.re().is_none() && cst_rnx.im().is_none() {
            return Ok(());
        }

        let mut tmp = take_mul_tmp(self, dst);
        self.ckks_mul_pt_const_rnx_into::<BE::OwnedBuf, A, F>(&mut tmp, a, cst_rnx, prec, scratch)?;
        self.ckks_add_assign::<Dst, BE::OwnedBuf>(dst, &tmp, scratch)
    }
}

// --- CKKSMulSubOps ---

impl<BE: Backend + CKKSImpl<BE>> CKKSMulSubOps<BE> for Module<BE> {
    fn ckks_mul_sub_ct_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Self: GLWEShift<BE> + GLWETensoring<BE> + CKKSSubOps<BE> + CKKSMulOps<BE>,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_tmp_bytes(res, tsk).max(self.ckks_sub_tmp_bytes())
    }

    fn ckks_mul_sub_pt_vec_znx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulPlain<BE> + GLWEShift<BE> + CKKSSubOps<BE> + CKKSMulOps<BE>,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_pt_vec_znx_tmp_bytes(res, a, b).max(self.ckks_sub_tmp_bytes())
    }

    fn ckks_mul_sub_pt_vec_rnx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: ModuleN + GLWEMulPlain<BE> + GLWEShift<BE> + CKKSSubOps<BE> + CKKSMulOps<BE>,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_pt_vec_rnx_tmp_bytes(res, a, b).max(self.ckks_sub_tmp_bytes())
    }

    fn ckks_mul_sub_pt_const_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulConst<BE> + GLWERotate<BE> + GLWEShift<BE> + CKKSSubOps<BE> + CKKSMulOps<BE>,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_pt_const_tmp_bytes(res, a, b).max(self.ckks_sub_tmp_bytes())
    }

    fn ckks_mul_sub_ct_into<Dst: Data, A: Data, B: Data, T: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        b: &CKKSCiphertext<B>,
        tsk: &GLWETensorKeyPrepared<T, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWECopy<BE>
            + GLWESub<BE>
            + GLWEShift<BE>
            + GLWETensoring<BE>
            + CKKSSubOps<BE>
            + CKKSMulOps<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + GLWEInfos,
        CKKSCiphertext<B>: GLWEToBackendRef<BE> + GLWEInfos,
        GLWETensorKeyPrepared<T, BE>: poulpy_core::layouts::prepared::GLWETensorKeyPreparedToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        let mut tmp = take_mul_tmp(self, dst);
        self.ckks_mul_into::<BE::OwnedBuf, A, B, T>(&mut tmp, a, b, tsk, scratch)?;
        self.ckks_sub_assign::<Dst, BE::OwnedBuf>(dst, &tmp, scratch)
    }

    fn ckks_mul_sub_pt_vec_znx_into<Dst: Data, A: Data, P: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt_znx: &CKKSPlaintextVecZnx<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWECopy<BE>
            + GLWESub<BE>
            + GLWEMulPlain<BE>
            + GLWEShift<BE>
            + CKKSSubOps<BE>
            + CKKSMulOps<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
            + VecZnxCopyBackend<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + GLWEInfos,
        CKKSPlaintextVecZnx<P>: poulpy_core::layouts::GLWEPlaintextToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        let mut tmp = take_mul_tmp(self, dst);
        self.ckks_mul_pt_vec_znx_into::<BE::OwnedBuf, A, P>(&mut tmp, a, pt_znx, scratch)?;
        self.ckks_sub_assign::<Dst, BE::OwnedBuf>(dst, &tmp, scratch)
    }

    fn ckks_mul_sub_pt_vec_rnx_into<Dst: Data, A: Data, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: ModuleN
            + GLWECopy<BE>
            + GLWESub<BE>
            + GLWEMulPlain<BE>
            + GLWEShift<BE>
            + CKKSSubOps<BE>
            + CKKSMulOps<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
            + VecZnxCopyBackend<BE>,
        BE: HostBackend<OwnedBuf = Vec<u8>>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
    {
        let mut tmp = take_mul_tmp(self, dst);
        self.ckks_mul_pt_vec_rnx_into::<BE::OwnedBuf, A, F>(&mut tmp, a, pt_rnx, prec, scratch)?;
        self.ckks_sub_assign::<Dst, BE::OwnedBuf>(dst, &tmp, scratch)
    }

    fn ckks_mul_sub_pt_const_znx_into<Dst: Data, A: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd<BE>
            + GLWECopy<BE>
            + GLWESub<BE>
            + GLWEMulConst<BE>
            + GLWERotate<BE>
            + GLWEShift<BE>
            + CKKSSubOps<BE>
            + CKKSMulOps<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        if cst_znx.re().is_none() && cst_znx.im().is_none() {
            return Ok(());
        }

        let mut tmp = take_mul_tmp(self, dst);
        self.ckks_mul_pt_const_znx_into::<BE::OwnedBuf, A>(&mut tmp, a, cst_znx, scratch)?;
        self.ckks_sub_assign::<Dst, BE::OwnedBuf>(dst, &tmp, scratch)
    }

    fn ckks_mul_sub_pt_const_rnx_into<Dst: Data, A: Data, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd<BE>
            + GLWECopy<BE>
            + GLWESub<BE>
            + GLWEMulConst<BE>
            + GLWERotate<BE>
            + GLWEShift<BE>
            + CKKSSubOps<BE>
            + CKKSMulOps<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion,
    {
        if cst_rnx.re().is_none() && cst_rnx.im().is_none() {
            return Ok(());
        }

        let mut tmp = take_mul_tmp(self, dst);
        self.ckks_mul_pt_const_rnx_into::<BE::OwnedBuf, A, F>(&mut tmp, a, cst_rnx, prec, scratch)?;
        self.ckks_sub_assign::<Dst, BE::OwnedBuf>(dst, &tmp, scratch)
    }
}

// --- CKKSDotProductOps ---

fn check_lengths(op: &'static str, a_len: usize, b_len: usize) -> Result<()> {
    if a_len == 0 {
        bail!("{op}: inputs must contain at least one pair");
    }
    if a_len != b_len {
        bail!("{op}: length mismatch between ct vector ({a_len}) and weight vector ({b_len})");
    }
    Ok(())
}

fn accumulate_unnormalized<BE, D, F>(
    module: &Module<BE>,
    dst: &mut CKKSCiphertext<D>,
    n: usize,
    scratch: &mut ScratchArena<'_, BE>,
    mut mul_term_into_tmp: F,
) -> Result<()>
where
    BE: Backend + CKKSImpl<BE>,
    D: Data,
    Module<BE>: CKKSAddOps<BE> + GLWEAdd<BE> + GLWEShift<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
    CKKSCiphertext<D>: GLWEToBackendMut<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    F: FnMut(&mut CKKSCiphertext<BE::OwnedBuf>, usize, &mut ScratchArena<'_, BE>) -> Result<()>,
{
    if n <= 1 {
        return Ok(());
    }
    let mut tmp = take_mul_tmp(module, dst);
    for i in 1..n {
        mul_term_into_tmp(&mut tmp, i, scratch)?;
        <Module<BE> as CKKSAddOps<BE>>::ckks_add_assign::<D, BE::OwnedBuf>(module, dst, &tmp, scratch)?;
    }
    Ok(())
}

impl<BE: Backend + CKKSImpl<BE>> CKKSDotProductOps<BE> for Module<BE> {
    fn ckks_dot_product_ct_tmp_bytes<R, T>(&self, n: usize, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Self: GLWEShift<BE> + GLWETensoring<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>,
    {
        let mul_scratch: usize = self.ckks_mul_tmp_bytes(res, tsk);
        if n <= 1 {
            return mul_scratch;
        }
        let ct_bytes: usize = GLWE::<Vec<u8>>::bytes_of_from_infos(res);
        let fallback: usize = ct_bytes + mul_scratch.max(self.ckks_add_tmp_bytes());
        let tensor_layout = GLWELayout {
            n: res.n(),
            base2k: res.base2k(),
            k: TorusPrecision(res.max_k().as_u32()),
            rank: res.rank(),
        };
        let tensor_bytes: usize = GLWETensor::bytes_of_from_infos(&tensor_layout);
        let inner: usize = self
            .ckks_rescale_tmp_bytes()
            .max(self.glwe_tensor_apply_tmp_bytes(&tensor_layout, res, res))
            .max(self.glwe_tensor_relinearize_tmp_bytes(res, &tensor_layout, tsk));
        let fast: usize = 2 * n * ct_bytes + tensor_bytes + inner;
        fallback.max(fast)
    }

    fn ckks_dot_product_pt_vec_znx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulPlain<BE> + GLWEShift<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_pt_vec_znx_tmp_bytes(res, a, b).max(self.ckks_add_tmp_bytes())
    }

    fn ckks_dot_product_pt_vec_rnx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: ModuleN + GLWEMulPlain<BE> + GLWEShift<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_pt_vec_rnx_tmp_bytes(res, a, b).max(self.ckks_add_tmp_bytes())
    }

    fn ckks_dot_product_pt_const_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulConst<BE> + GLWERotate<BE> + GLWEShift<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>,
    {
        GLWE::<Vec<u8>>::bytes_of_from_infos(res) + self.ckks_mul_pt_const_tmp_bytes(res, a, b).max(self.ckks_add_tmp_bytes())
    }

    fn ckks_dot_product_ct<Dst: Data, D: Data, E: Data, T: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &[&CKKSCiphertext<D>],
        b: &[&CKKSCiphertext<E>],
        tsk: &GLWETensorKeyPrepared<T, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd<BE>
            + GLWECopy<BE>
            + GLWEShift<BE>
            + GLWENormalize<BE>
            + GLWETensoring<BE>
            + CKKSAddOpsUnsafe<BE>
            + CKKSMulOps<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<D>: GLWEToBackendRef<BE> + GLWEInfos,
        CKKSCiphertext<E>: GLWEToBackendRef<BE> + GLWEInfos,
        GLWETensorKeyPrepared<T, BE>: poulpy_core::layouts::prepared::GLWETensorKeyPreparedToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        check_lengths("ckks_dot_product_ct", a.len(), b.len())?;
        let n: usize = a.len();
        ensure_accumulation_fits("ckks_dot_product_ct", dst, n)?;
        self.ckks_mul_into::<Dst, D, E, T>(dst, a[0], b[0], tsk, scratch)?;
        accumulate_unnormalized(self, dst, n, scratch, |tmp, i, s| {
            self.ckks_mul_into::<BE::OwnedBuf, D, E, T>(tmp, a[i], b[i], tsk, s)
        })
    }

    fn ckks_dot_product_pt_vec_znx<Dst: Data, D: Data, E: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &[&CKKSCiphertext<D>],
        b: &[&CKKSPlaintextVecZnx<E>],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd<BE>
            + GLWECopy<BE>
            + GLWEMulPlain<BE>
            + GLWEShift<BE>
            + GLWENormalize<BE>
            + VecZnxRshAddIntoBackend<BE>
            + CKKSAddOpsUnsafe<BE>
            + CKKSMulOps<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
            + VecZnxCopyBackend<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<D>: GLWEToBackendRef<BE> + GLWEInfos,
        CKKSPlaintextVecZnx<E>: poulpy_core::layouts::GLWEPlaintextToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        check_lengths("ckks_dot_product_pt_vec_znx", a.len(), b.len())?;
        let n: usize = a.len();
        ensure_accumulation_fits("ckks_dot_product_pt_vec_znx", dst, n)?;
        self.ckks_mul_pt_vec_znx_into::<Dst, D, E>(dst, a[0], b[0], scratch)?;
        accumulate_unnormalized(self, dst, n, scratch, |tmp, i, s| {
            self.ckks_mul_pt_vec_znx_into::<BE::OwnedBuf, D, E>(tmp, a[i], b[i], s)
        })
    }

    fn ckks_dot_product_pt_vec_rnx<Dst: Data, D: Data, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &[&CKKSCiphertext<D>],
        b: &[&CKKSPlaintextVecRnx<F>],
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: ModuleN
            + GLWEAdd<BE>
            + GLWECopy<BE>
            + GLWEMulPlain<BE>
            + GLWEShift<BE>
            + GLWENormalize<BE>
            + VecZnxRshAddIntoBackend<BE>
            + CKKSAddOpsUnsafe<BE>
            + CKKSMulOps<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
            + VecZnxCopyBackend<BE>,
        BE: HostBackend<OwnedBuf = Vec<u8>>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<D>: GLWEToBackendRef<BE> + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
    {
        check_lengths("ckks_dot_product_pt_vec_rnx", a.len(), b.len())?;
        let n: usize = a.len();
        ensure_accumulation_fits("ckks_dot_product_pt_vec_rnx", dst, n)?;
        self.ckks_mul_pt_vec_rnx_into::<Dst, D, F>(dst, a[0], b[0], prec, scratch)?;
        accumulate_unnormalized(self, dst, n, scratch, |tmp, i, s| {
            self.ckks_mul_pt_vec_rnx_into::<BE::OwnedBuf, D, F>(tmp, a[i], b[i], prec, s)
        })
    }

    fn ckks_dot_product_pt_const_znx<Dst: Data, D: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &[&CKKSCiphertext<D>],
        b: &[&CKKSPlaintextCstZnx],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd<BE>
            + GLWECopy<BE>
            + GLWEMulConst<BE>
            + GLWERotate<BE>
            + GLWEShift<BE>
            + GLWENormalize<BE>
            + CKKSAddOpsUnsafe<BE>
            + CKKSMulOps<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<D>: GLWEToBackendRef<BE> + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        check_lengths("ckks_dot_product_pt_const_znx", a.len(), b.len())?;
        let n: usize = a.len();
        ensure_accumulation_fits("ckks_dot_product_pt_const_znx", dst, n)?;
        self.ckks_mul_pt_const_znx_into::<Dst, D>(dst, a[0], b[0], scratch)?;
        accumulate_unnormalized(self, dst, n, scratch, |tmp, i, s| {
            self.ckks_mul_pt_const_znx_into::<BE::OwnedBuf, D>(tmp, a[i], b[i], s)
        })
    }

    fn ckks_dot_product_pt_const_rnx<Dst: Data, D: Data, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &[&CKKSCiphertext<D>],
        b: &[&CKKSPlaintextCstRnx<F>],
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd<BE>
            + GLWECopy<BE>
            + GLWEMulConst<BE>
            + GLWERotate<BE>
            + GLWEShift<BE>
            + GLWENormalize<BE>
            + CKKSAddOpsUnsafe<BE>
            + CKKSMulOps<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<D>: GLWEToBackendRef<BE> + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion,
    {
        check_lengths("ckks_dot_product_pt_const_rnx", a.len(), b.len())?;
        let n: usize = a.len();
        ensure_accumulation_fits("ckks_dot_product_pt_const_rnx", dst, n)?;
        self.ckks_mul_pt_const_rnx_into::<Dst, D, F>(dst, a[0], b[0], prec, scratch)?;
        accumulate_unnormalized(self, dst, n, scratch, |tmp, i, s| {
            self.ckks_mul_pt_const_rnx_into::<BE::OwnedBuf, D, F>(tmp, a[i], b[i], prec, s)
        })
    }
}
