use anyhow::Result;
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{CKKSCiphertextToBackendMut, CKKSCiphertextToBackendRef, CKKSPlaintexToBackendRef};

use crate::{CKKSInfos, SetCKKSInfos, oep::CKKSImpl};

pub trait CKKSSubOps<BE: Backend + CKKSImpl<BE>> {
    fn ckks_sub_tmp_bytes(&self) -> usize;
    fn ckks_sub_pt_vec_znx_tmp_bytes(&self) -> usize;

    fn ckks_sub_into<Dst, A, B>(&self, dst: &mut Dst, a: &A, b: &B, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        A: CKKSCiphertextToBackendRef<BE> + CKKSInfos,
        B: CKKSCiphertextToBackendRef<BE> + CKKSInfos;

    fn ckks_sub_assign<Dst, A>(&self, dst: &mut Dst, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        A: CKKSCiphertextToBackendRef<BE> + CKKSInfos;

    fn ckks_sub_pt_vec_znx_into<Dst, A, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        A: CKKSCiphertextToBackendRef<BE> + CKKSInfos,
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos;

    fn ckks_sub_pt_vec_znx_assign<Dst, P>(&self, dst: &mut Dst, pt_znx: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos;

    fn ckks_sub_pt_const_tmp_bytes(&self) -> usize;

    fn ckks_sub_pt_const_znx_into<Dst, A, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        dst_coeff: usize,
        pt_znx: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        A: CKKSCiphertextToBackendRef<BE> + CKKSInfos,
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos;

    fn ckks_sub_pt_const_znx_assign<Dst, P>(
        &self,
        dst: &mut Dst,
        dst_coeff: usize,
        pt_znx: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos;
}

#[allow(clippy::missing_safety_doc)]
pub unsafe trait CKKSSubOpsUnsafe<BE: Backend> {
    unsafe fn ckks_sub_into_unsafe<Dst, A, B>(
        &self,
        dst: &mut Dst,
        a: &A,
        b: &B,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        A: CKKSCiphertextToBackendRef<BE> + CKKSInfos,
        B: CKKSCiphertextToBackendRef<BE> + CKKSInfos;

    unsafe fn ckks_sub_assign_unsafe<Dst, A>(&self, dst: &mut Dst, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        A: CKKSCiphertextToBackendRef<BE> + CKKSInfos;

    unsafe fn ckks_sub_pt_vec_znx_into_unsafe<Dst, A, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        A: CKKSCiphertextToBackendRef<BE> + CKKSInfos,
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos;

    unsafe fn ckks_sub_pt_vec_znx_assign_unsafe<Dst, P>(
        &self,
        dst: &mut Dst,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos;

    unsafe fn ckks_sub_pt_const_znx_into_unsafe<Dst, A, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        dst_coeff: usize,
        pt_znx: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        A: CKKSCiphertextToBackendRef<BE> + CKKSInfos,
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos;

    unsafe fn ckks_sub_pt_const_znx_assign_unsafe<Dst, P>(
        &self,
        dst: &mut Dst,
        dst_coeff: usize,
        pt_znx: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos;
}
