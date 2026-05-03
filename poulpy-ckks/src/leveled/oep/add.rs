use anyhow::Result;
use poulpy_core::{
    GLWEAdd, GLWENormalize, GLWEShift, ScratchArenaTakeCore,
    layouts::{GLWEToBackendMut, LWEInfos},
};
use poulpy_hal::{
    api::{ScratchAvailable, VecZnxRshAddCoeffIntoBackend, VecZnxRshAddIntoBackend, VecZnxRshTmpBytes},
    layouts::{Backend, Module, ScratchArena},
};

use crate::{CKKSInfos, GLWEToBackendRef, SetCKKSInfos, leveled::default::pt_znx::CKKSPlaintextDefault, oep::CKKSImpl};

pub(crate) trait CKKSAddOep<BE: Backend + CKKSImpl<BE>> {
    fn ckks_add_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE> + GLWENormalize<BE>;

    fn ckks_add_into<Dst, A, B>(&self, dst: &mut Dst, a: &A, b: &B, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        B: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        Self: GLWEAdd<BE> + GLWEShift<BE> + GLWENormalize<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_add_assign<Dst, A>(&self, dst: &mut Dst, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos,
        Self: GLWEAdd<BE> + GLWEShift<BE> + GLWENormalize<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_add_pt_vec_znx_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE> + GLWENormalize<BE> + VecZnxRshTmpBytes;

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
        Self: VecZnxRshAddIntoBackend<BE> + GLWEShift<BE> + GLWENormalize<BE> + CKKSPlaintextDefault<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_add_pt_vec_znx_assign<Dst, P>(&self, dst: &mut Dst, pt_znx: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        Self: VecZnxRshAddIntoBackend<BE> + GLWENormalize<BE> + CKKSPlaintextDefault<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_add_pt_const_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE> + GLWENormalize<BE> + VecZnxRshTmpBytes;

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
        Self: GLWEShift<BE> + GLWENormalize<BE> + VecZnxRshAddCoeffIntoBackend<BE> + CKKSPlaintextDefault<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_add_pt_const_znx_assign<Dst, P>(
        &self,
        dst: &mut Dst,
        dst_coeff: usize,
        pt_znx: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        Self: GLWENormalize<BE> + VecZnxRshAddCoeffIntoBackend<BE> + CKKSPlaintextDefault<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;
}

impl<BE: Backend + CKKSImpl<BE>> CKKSAddOep<BE> for Module<BE> {
    fn ckks_add_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE> + GLWENormalize<BE>,
    {
        BE::ckks_add_tmp_bytes(self)
    }

    fn ckks_add_into<Dst, A, B>(&self, dst: &mut Dst, a: &A, b: &B, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        B: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        Self: GLWEAdd<BE> + GLWEShift<BE> + GLWENormalize<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_add_into(self, dst, a, b, scratch)
    }

    fn ckks_add_assign<Dst, A>(&self, dst: &mut Dst, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos,
        Self: GLWEAdd<BE> + GLWEShift<BE> + GLWENormalize<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_add_assign(self, dst, a, scratch)
    }

    fn ckks_add_pt_vec_znx_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE> + GLWENormalize<BE> + VecZnxRshTmpBytes,
    {
        BE::ckks_add_pt_vec_znx_tmp_bytes(self)
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
        Self: VecZnxRshAddIntoBackend<BE> + GLWEShift<BE> + GLWENormalize<BE> + CKKSPlaintextDefault<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_add_pt_vec_znx_into(self, dst, a, pt_znx, scratch)
    }

    fn ckks_add_pt_vec_znx_assign<Dst, P>(&self, dst: &mut Dst, pt_znx: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        Self: VecZnxRshAddIntoBackend<BE> + GLWENormalize<BE> + CKKSPlaintextDefault<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_add_pt_vec_znx_assign(self, dst, pt_znx, scratch)
    }

    fn ckks_add_pt_const_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE> + GLWENormalize<BE> + VecZnxRshTmpBytes,
    {
        BE::ckks_add_pt_const_tmp_bytes(self)
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
        Self: GLWEShift<BE> + GLWENormalize<BE> + VecZnxRshAddCoeffIntoBackend<BE> + CKKSPlaintextDefault<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_add_pt_const_znx_into(self, dst, a, dst_coeff, pt_znx, pt_coeff, scratch)
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
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        Self: GLWENormalize<BE> + VecZnxRshAddCoeffIntoBackend<BE> + CKKSPlaintextDefault<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_add_pt_const_znx_assign(self, dst, dst_coeff, pt_znx, pt_coeff, scratch)
    }
}
