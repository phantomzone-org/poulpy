use anyhow::Result;
use poulpy_core::{
    GLWEAdd, GLWEShift, ScratchArenaTakeCore,
    layouts::{GLWEInfos, GLWEPlaintextToBackendRef, GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, VecZnxAddConstAssignBackend, VecZnxRshAddIntoBackend, VecZnxRshTmpBytes},
    layouts::{Backend, Data, HostBackend, Module, ScratchArena},
};

use crate::{
    CKKSMeta,
    layouts::{
        CKKSCiphertext,
        plaintext::{
            CKKSConstPlaintextConversion, CKKSPlaintextConversion, CKKSPlaintextCstRnx, CKKSPlaintextCstZnx, CKKSPlaintextVecRnx,
            CKKSPlaintextVecZnx,
        },
    },
    oep::CKKSImpl,
};

use crate::leveled::{api::CKKSAddOps, oep::CKKSAddOep};

impl<BE: Backend + CKKSImpl<BE>> CKKSAddOps<BE> for Module<BE> {
    fn ckks_add_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        CKKSAddOep::ckks_add_tmp_bytes(self)
    }

    fn ckks_add_into<Dst: Data, A: Data, B: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        b: &CKKSCiphertext<B>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd<BE> + GLWEShift<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        CKKSCiphertext<B>: GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        CKKSAddOep::ckks_add_into(self, dst, a, b, scratch)
    }

    fn ckks_add_assign<Dst: Data, A: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd<BE> + GLWEShift<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        CKKSAddOep::ckks_add_assign(self, dst, a, scratch)
    }

    fn ckks_add_pt_vec_znx_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE> + VecZnxRshTmpBytes,
    {
        CKKSAddOep::ckks_add_pt_vec_znx_tmp_bytes(self)
    }

    fn ckks_add_pt_vec_znx_into<Dst: Data, A: Data, P: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt_znx: &CKKSPlaintextVecZnx<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshAddIntoBackend<BE> + GLWEShift<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        CKKSPlaintextVecZnx<P>: GLWEPlaintextToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        CKKSAddOep::ckks_add_pt_vec_znx_into(self, dst, a, pt_znx, scratch)
    }

    fn ckks_add_pt_vec_znx_assign<Dst: Data, P: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        pt_znx: &CKKSPlaintextVecZnx<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshAddIntoBackend<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSPlaintextVecZnx<P>: GLWEPlaintextToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        CKKSAddOep::ckks_add_pt_vec_znx_assign(self, dst, pt_znx, scratch)
    }

    fn ckks_add_pt_vec_rnx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: ModuleN + GLWEShift<BE> + VecZnxRshTmpBytes,
    {
        CKKSAddOep::ckks_add_pt_vec_rnx_tmp_bytes(self, res, a, b)
    }

    fn ckks_add_pt_vec_rnx_into<Dst: Data, A: Data, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: ModuleN + VecZnxRshAddIntoBackend<BE> + GLWEShift<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        BE: HostBackend<OwnedBuf = Vec<u8>>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
    {
        CKKSAddOep::ckks_add_pt_vec_rnx_into(self, dst, a, pt_rnx, prec, scratch)
    }

    fn ckks_add_pt_vec_rnx_assign<Dst: Data, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: ModuleN + VecZnxRshAddIntoBackend<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        BE: HostBackend<OwnedBuf = Vec<u8>>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
    {
        CKKSAddOep::ckks_add_pt_vec_rnx_assign(self, dst, pt_rnx, prec, scratch)
    }

    fn ckks_add_pt_const_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        CKKSAddOep::ckks_add_pt_const_tmp_bytes(self)
    }

    fn ckks_add_pt_const_znx_into<Dst: Data, A: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE> + VecZnxAddConstAssignBackend<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        CKKSAddOep::ckks_add_pt_const_znx_into(self, dst, a, cst_znx, scratch)
    }

    fn ckks_add_pt_const_znx_assign<Dst: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: VecZnxAddConstAssignBackend<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        CKKSAddOep::ckks_add_pt_const_znx_assign(self, dst, cst_znx, scratch)
    }

    fn ckks_add_pt_const_rnx_into<Dst: Data, A: Data, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE> + VecZnxAddConstAssignBackend<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion,
    {
        CKKSAddOep::ckks_add_pt_const_rnx_into(self, dst, a, cst_rnx, prec, scratch)
    }

    fn ckks_add_pt_const_rnx_assign<Dst: Data, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion,
    {
        CKKSAddOep::ckks_add_pt_const_rnx_assign(self, dst, cst_rnx, prec, scratch)
    }
}

use crate::leveled::api::CKKSAddOpsUnsafe;
use crate::leveled::default::CKKSAddDefault;

unsafe impl<BE: Backend + poulpy_hal::oep::HalVecZnxImpl<BE>> CKKSAddOpsUnsafe<BE> for Module<BE>
where
    Module<BE>: CKKSAddDefault<BE>,
{
    unsafe fn ckks_add_into_unsafe<Dst: Data, A: Data, B: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        b: &CKKSCiphertext<B>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd<BE> + GLWEShift<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        CKKSCiphertext<B>: GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        self.ckks_add_into_unsafe_default(dst, a, b, scratch)
    }

    unsafe fn ckks_add_assign_unsafe<Dst: Data, A: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd<BE> + GLWEShift<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        self.ckks_add_assign_unsafe_default(dst, a, scratch)
    }

    unsafe fn ckks_add_pt_vec_znx_into_unsafe<Dst: Data, A: Data, P: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt_znx: &CKKSPlaintextVecZnx<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshAddIntoBackend<BE> + GLWEShift<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        CKKSPlaintextVecZnx<P>: GLWEPlaintextToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        self.ckks_add_pt_vec_znx_into_unsafe_default(dst, a, pt_znx, scratch)
    }

    unsafe fn ckks_add_pt_vec_znx_assign_unsafe<Dst: Data, P: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        pt_znx: &CKKSPlaintextVecZnx<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshAddIntoBackend<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSPlaintextVecZnx<P>: GLWEPlaintextToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        self.ckks_add_pt_vec_znx_assign_unsafe_default(dst, pt_znx, scratch)
    }

    unsafe fn ckks_add_pt_vec_rnx_into_unsafe<Dst: Data, A: Data, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: ModuleN + VecZnxRshAddIntoBackend<BE> + GLWEShift<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        BE: HostBackend<OwnedBuf = Vec<u8>>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
    {
        self.ckks_add_pt_vec_rnx_into_unsafe_default(dst, a, pt_rnx, prec, scratch)
    }

    unsafe fn ckks_add_pt_vec_rnx_assign_unsafe<Dst: Data, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: ModuleN + VecZnxRshAddIntoBackend<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        BE: HostBackend<OwnedBuf = Vec<u8>>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
    {
        self.ckks_add_pt_vec_rnx_assign_unsafe_default(dst, pt_rnx, prec, scratch)
    }

    unsafe fn ckks_add_pt_const_znx_into_unsafe<Dst: Data, A: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        self.ckks_add_pt_const_znx_into_unsafe_default(dst, a, cst_znx, scratch)
    }

    unsafe fn ckks_add_pt_const_znx_assign_unsafe<Dst: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: VecZnxAddConstAssignBackend<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        self.ckks_add_pt_const_znx_assign_unsafe_default(dst, cst_znx, scratch)
    }

    unsafe fn ckks_add_pt_const_rnx_into_unsafe<Dst: Data, A: Data, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE> + VecZnxAddConstAssignBackend<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion,
    {
        self.ckks_add_pt_const_rnx_into_unsafe_default(dst, a, cst_rnx, prec, scratch)
    }

    unsafe fn ckks_add_pt_const_rnx_assign_unsafe<Dst: Data, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: VecZnxAddConstAssignBackend<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion,
    {
        self.ckks_add_pt_const_rnx_assign_unsafe_default(dst, cst_rnx, prec, scratch)
    }
}
