use anyhow::Result;
use poulpy_core::{GLWEShift, GLWESub, ScratchArenaTakeCore};
use poulpy_hal::{
    api::{ScratchAvailable, VecZnxRshSubBackend, VecZnxRshTmpBytes},
    layouts::{Backend, Module, ScratchArena},
};

use crate::{CKKSCiphertextMut, CKKSCiphertextRef, CKKSInfos, CKKSPlaintexToBackendRef, oep::CKKSImpl};

pub(crate) trait CKKSSubOep<BE: Backend + CKKSImpl<BE>> {
    fn ckks_sub_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE> + VecZnxRshTmpBytes;

    fn ckks_sub_pt_vec_znx_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE> + VecZnxRshTmpBytes;

    fn ckks_sub_into(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        a: &CKKSCiphertextRef<'_, BE>,
        b: &CKKSCiphertextRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWESub<BE> + GLWEShift<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_sub_assign(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        a: &CKKSCiphertextRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWESub<BE> + GLWEShift<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_sub_pt_vec_znx_into<P>(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        a: &CKKSCiphertextRef<'_, BE>,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos,
        Self: VecZnxRshSubBackend<BE> + GLWEShift<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_sub_pt_vec_znx_assign<P>(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos,
        Self: VecZnxRshSubBackend<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_sub_pt_const_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>;

    fn ckks_sub_pt_const_znx_into<P>(
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

    fn ckks_sub_pt_const_znx_assign<P>(
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

impl<BE: Backend + CKKSImpl<BE>> CKKSSubOep<BE> for Module<BE> {
    fn ckks_sub_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE> + VecZnxRshTmpBytes,
    {
        BE::ckks_sub_tmp_bytes(self)
    }

    fn ckks_sub_pt_vec_znx_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE> + VecZnxRshTmpBytes,
    {
        BE::ckks_sub_pt_vec_znx_tmp_bytes(self)
    }

    fn ckks_sub_into(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        a: &CKKSCiphertextRef<'_, BE>,
        b: &CKKSCiphertextRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWESub<BE> + GLWEShift<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_sub_into(self, dst, a, b, scratch)
    }

    fn ckks_sub_assign(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        a: &CKKSCiphertextRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWESub<BE> + GLWEShift<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_sub_assign(self, dst, a, scratch)
    }

    fn ckks_sub_pt_vec_znx_into<P>(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        a: &CKKSCiphertextRef<'_, BE>,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos,
        Self: VecZnxRshSubBackend<BE> + GLWEShift<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_sub_pt_vec_znx_into(self, dst, a, pt_znx, scratch)
    }

    fn ckks_sub_pt_vec_znx_assign<P>(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos,
        Self: VecZnxRshSubBackend<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_sub_pt_vec_znx_assign(self, dst, pt_znx, scratch)
    }

    fn ckks_sub_pt_const_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        BE::ckks_sub_pt_const_tmp_bytes(self)
    }

    fn ckks_sub_pt_const_znx_into<P>(
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
        BE::ckks_sub_pt_const_znx_into(self, dst, a, dst_coeff, pt_znx, pt_coeff, scratch)
    }

    fn ckks_sub_pt_const_znx_assign<P>(
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
        BE::ckks_sub_pt_const_znx_assign(self, dst, dst_coeff, pt_znx, pt_coeff, scratch)
    }
}
