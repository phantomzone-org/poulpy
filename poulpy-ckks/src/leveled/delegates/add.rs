use anyhow::Result;
use poulpy_core::{
    GLWEAdd, GLWENormalize, GLWEShift, ScratchArenaTakeCore,
    layouts::{GLWEToBackendMut, GLWEToBackendRef, LWEInfos},
};
use poulpy_hal::{
    api::{ScratchAvailable, VecZnxRshAddCoeffIntoBackend, VecZnxRshAddIntoBackend, VecZnxRshTmpBytes},
    layouts::{Backend, Module, ScratchArena},
    oep::HalVecZnxImpl,
};

use crate::leveled::{
    api::{CKKSAddOps, CKKSAddOpsUnsafe},
    default::{CKKSAddDefault, CKKSPlaintextDefault},
    oep::CKKSAddOep,
};

use crate::{CKKSInfos, SetCKKSInfos, oep::CKKSImpl};

impl<BE: Backend + CKKSImpl<BE>> CKKSAddOps<BE> for Module<BE>
where
    Module<BE>: GLWEShift<BE>
        + GLWEAdd<BE>
        + GLWENormalize<BE>
        + CKKSPlaintextDefault<BE>
        + VecZnxRshAddCoeffIntoBackend<BE>
        + VecZnxRshAddIntoBackend<BE>
        + VecZnxRshTmpBytes,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_add_tmp_bytes(&self) -> usize {
        CKKSAddOep::ckks_add_tmp_bytes(self)
    }

    fn ckks_add_into<Dst, A, B>(&self, dst: &mut Dst, a: &A, b: &B, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        B: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
    {
        CKKSAddOep::ckks_add_into(self, dst, a, b, scratch)
    }

    fn ckks_add_assign<Dst, A>(&self, dst: &mut Dst, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
    {
        CKKSAddOep::ckks_add_assign(self, dst, a, scratch)
    }

    fn ckks_add_pt_vec_znx_tmp_bytes(&self) -> usize {
        CKKSAddOep::ckks_add_pt_vec_znx_tmp_bytes(self)
    }

    fn ckks_add_pt_vec_znx_into<Dst, A, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
    {
        CKKSAddOep::ckks_add_pt_vec_znx_into(self, dst, a, pt_znx, scratch)
    }

    fn ckks_add_pt_vec_znx_assign<Dst, P>(&self, dst: &mut Dst, pt_znx: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
    {
        CKKSAddOep::ckks_add_pt_vec_znx_assign(self, dst, pt_znx, scratch)
    }

    fn ckks_add_pt_const_tmp_bytes(&self) -> usize {
        CKKSAddOep::ckks_add_pt_const_tmp_bytes(self)
    }

    fn ckks_add_pt_const_znx_into<Dst, A, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        dst_coeff: usize,
        pt_znx: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
    {
        CKKSAddOep::ckks_add_pt_const_znx_into(self, dst, a, dst_coeff, pt_znx, pt_coeff, scratch)
    }

    fn ckks_add_pt_const_znx_assign<Dst, P>(
        &self,
        dst: &mut Dst,
        dst_coeff: usize,
        pt_znx: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
    {
        CKKSAddOep::ckks_add_pt_const_znx_assign(self, dst, dst_coeff, pt_znx, pt_coeff, scratch)
    }
}

unsafe impl<BE: Backend + HalVecZnxImpl<BE>> CKKSAddOpsUnsafe<BE> for Module<BE>
where
    Module<BE>: CKKSAddDefault<BE>
        + GLWEShift<BE>
        + GLWEAdd<BE>
        + CKKSPlaintextDefault<BE>
        + VecZnxRshAddCoeffIntoBackend<BE>
        + VecZnxRshAddIntoBackend<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
{
    unsafe fn ckks_add_into_unsafe<Dst, A, B>(
        &self,
        dst: &mut Dst,
        a: &A,
        b: &B,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        B: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
    {
        self.ckks_add_into_unsafe_default(dst, a, b, scratch)
    }

    unsafe fn ckks_add_assign_unsafe<Dst, A>(&self, dst: &mut Dst, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
    {
        self.ckks_add_assign_unsafe_default(dst, a, scratch)
    }

    unsafe fn ckks_add_pt_vec_znx_into_unsafe<Dst, A, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
    {
        self.ckks_add_pt_vec_znx_into_unsafe_default(dst, a, pt_znx, scratch)
    }

    unsafe fn ckks_add_pt_vec_znx_assign_unsafe<Dst, P>(
        &self,
        dst: &mut Dst,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
    {
        self.ckks_add_pt_vec_znx_assign_unsafe_default(dst, pt_znx, scratch)
    }

    unsafe fn ckks_add_pt_const_znx_into_unsafe<Dst, A, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        dst_coeff: usize,
        pt_znx: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
    {
        self.ckks_add_pt_const_znx_into_unsafe_default(dst, a, dst_coeff, pt_znx, pt_coeff, scratch)
    }

    unsafe fn ckks_add_pt_const_znx_assign_unsafe<Dst, P>(
        &self,
        dst: &mut Dst,
        dst_coeff: usize,
        pt_znx: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
    {
        self.ckks_add_pt_const_znx_assign_unsafe_default(dst, dst_coeff, pt_znx, pt_coeff, scratch)
    }
}
