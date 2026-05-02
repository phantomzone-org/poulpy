use anyhow::Result;
use poulpy_core::{GLWEAdd, GLWEShift, ScratchArenaTakeCore};
use poulpy_hal::{
    api::{ScratchAvailable, VecZnxRshAddIntoBackend, VecZnxRshTmpBytes},
    layouts::{Backend, Module, ScratchArena},
};

use crate::{CKKSCiphertextMut, CKKSCiphertextRef, CKKSInfos, CKKSPlaintexToBackendRef, oep::CKKSImpl};

pub(crate) trait CKKSAddOep<BE: Backend + CKKSImpl<BE>> {
    fn ckks_add_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>;

    fn ckks_add_into(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        a: &CKKSCiphertextRef<'_, BE>,
        b: &CKKSCiphertextRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd<BE> + GLWEShift<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_add_assign(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        a: &CKKSCiphertextRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd<BE> + GLWEShift<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_add_pt_vec_znx_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE> + VecZnxRshTmpBytes;

    fn ckks_add_pt_vec_znx_into<P>(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        a: &CKKSCiphertextRef<'_, BE>,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos,
        Self: VecZnxRshAddIntoBackend<BE> + GLWEShift<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_add_pt_vec_znx_assign<P>(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos,
        Self: VecZnxRshAddIntoBackend<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_add_pt_const_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>;

    fn ckks_add_pt_const_znx_into<P>(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        a: &CKKSCiphertextRef<'_, BE>,
        dst_coeff: usize,
        pt_znx: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos,
        Self: GLWEShift<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_add_pt_const_znx_assign<P>(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        dst_coeff: usize,
        pt_znx: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;
}

impl<BE: Backend + CKKSImpl<BE>> CKKSAddOep<BE> for Module<BE> {
    fn ckks_add_tmp_bytes(&self) -> usize {
        BE::ckks_add_tmp_bytes(self)
    }

    fn ckks_add_into(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        a: &CKKSCiphertextRef<'_, BE>,
        b: &CKKSCiphertextRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd<BE> + GLWEShift<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_add_into(self, dst, a, b, scratch)
    }

    fn ckks_add_assign(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        a: &CKKSCiphertextRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd<BE> + GLWEShift<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_add_assign(self, dst, a, scratch)
    }

    fn ckks_add_pt_vec_znx_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE> + VecZnxRshTmpBytes,
    {
        BE::ckks_add_pt_vec_znx_tmp_bytes(self)
    }

    fn ckks_add_pt_vec_znx_into<P>(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        a: &CKKSCiphertextRef<'_, BE>,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos,
        Self: VecZnxRshAddIntoBackend<BE> + GLWEShift<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_add_pt_vec_znx_into(self, dst, a, pt_znx, scratch)
    }

    fn ckks_add_pt_vec_znx_assign<P>(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos,
        Self: VecZnxRshAddIntoBackend<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_add_pt_vec_znx_assign(self, dst, pt_znx, scratch)
    }

    fn ckks_add_pt_const_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        BE::ckks_add_pt_const_tmp_bytes(self)
    }

    fn ckks_add_pt_const_znx_into<P>(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        a: &CKKSCiphertextRef<'_, BE>,
        dst_coeff: usize,
        pt_znx: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos,
        Self: GLWEShift<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_add_pt_const_znx_into(self, dst, a, dst_coeff, pt_znx, pt_coeff, scratch)
    }

    fn ckks_add_pt_const_znx_assign<P>(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        dst_coeff: usize,
        pt_znx: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_add_pt_const_znx_assign(self, dst, dst_coeff, pt_znx, pt_coeff, scratch)
    }
}
