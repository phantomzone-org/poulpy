use anyhow::Result;
use poulpy_core::{
    GLWEAdd, GLWEShift, ScratchArenaTakeCore,
    layouts::{GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::{
    api::{ScratchAvailable, VecZnxRshAddCoeffIntoBackend, VecZnxRshAddIntoBackend, VecZnxRshTmpBytes},
    layouts::{Backend, Module, ScratchArena},
};

use crate::leveled::{
    api::{CKKSAddOps, CKKSAddOpsUnsafe},
    default::CKKSAddDefault,
    oep::CKKSAddOep,
};
use crate::{CKKSCiphertextToBackendMut, CKKSCiphertextToBackendRef, CKKSPlaintexToBackendRef};
use crate::{CKKSInfos, SetCKKSInfos, layouts::CKKSCiphertext, oep::CKKSImpl};

impl<BE: Backend + CKKSImpl<BE>> CKKSAddOps<BE> for Module<BE>
where
    Module<BE>: GLWEShift<BE> + GLWEAdd<BE> + VecZnxRshAddCoeffIntoBackend<BE> + VecZnxRshAddIntoBackend<BE> + VecZnxRshTmpBytes,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_add_tmp_bytes(&self) -> usize {
        CKKSAddOep::ckks_add_tmp_bytes(self)
    }

    fn ckks_add_into<Dst, A, B>(&self, dst: &mut Dst, a: &A, b: &B, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        A: CKKSCiphertextToBackendRef<BE> + CKKSInfos,
        B: CKKSCiphertextToBackendRef<BE> + CKKSInfos,
    {
        let dst_meta = dst.meta();
        let mut dst_ct = CKKSCiphertext::from_inner(GLWEToBackendMut::to_backend_mut(dst), dst_meta);
        let a_ct = CKKSCiphertext::from_inner(GLWEToBackendRef::to_backend_ref(a), a.meta());
        let b_ct = CKKSCiphertext::from_inner(GLWEToBackendRef::to_backend_ref(b), b.meta());
        let res = CKKSAddOep::ckks_add_into(self, &mut dst_ct, &a_ct, &b_ct, scratch);
        let new_meta = dst_ct.meta();
        drop(dst_ct);
        dst.set_meta(new_meta);
        res
    }

    fn ckks_add_assign<Dst, A>(&self, dst: &mut Dst, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        A: CKKSCiphertextToBackendRef<BE> + CKKSInfos,
    {
        let dst_meta = dst.meta();
        let mut dst_ct = CKKSCiphertext::from_inner(GLWEToBackendMut::to_backend_mut(dst), dst_meta);
        let a_ct = CKKSCiphertext::from_inner(GLWEToBackendRef::to_backend_ref(a), a.meta());
        let res = CKKSAddOep::ckks_add_assign(self, &mut dst_ct, &a_ct, scratch);
        let new_meta = dst_ct.meta();
        drop(dst_ct);
        dst.set_meta(new_meta);
        res
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
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        A: CKKSCiphertextToBackendRef<BE> + CKKSInfos,
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos,
    {
        let dst_meta = dst.meta();
        let mut dst_ct = CKKSCiphertext::from_inner(GLWEToBackendMut::to_backend_mut(dst), dst_meta);
        let a_ct = CKKSCiphertext::from_inner(GLWEToBackendRef::to_backend_ref(a), a.meta());
        let res = CKKSAddOep::ckks_add_pt_vec_znx_into(self, &mut dst_ct, &a_ct, pt_znx, scratch);
        let new_meta = dst_ct.meta();
        drop(dst_ct);
        dst.set_meta(new_meta);
        res
    }

    fn ckks_add_pt_vec_znx_assign<Dst, P>(&self, dst: &mut Dst, pt_znx: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos,
    {
        let dst_meta = dst.meta();
        let mut dst_ct = CKKSCiphertext::from_inner(GLWEToBackendMut::to_backend_mut(dst), dst_meta);
        let res = CKKSAddOep::ckks_add_pt_vec_znx_assign(self, &mut dst_ct, pt_znx, scratch);
        let new_meta = dst_ct.meta();
        drop(dst_ct);
        dst.set_meta(new_meta);
        res
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
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        A: CKKSCiphertextToBackendRef<BE> + CKKSInfos,
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos,
    {
        let dst_meta = dst.meta();
        let mut dst_ct = CKKSCiphertext::from_inner(GLWEToBackendMut::to_backend_mut(dst), dst_meta);
        let a_ct = CKKSCiphertext::from_inner(GLWEToBackendRef::to_backend_ref(a), a.meta());
        let res = CKKSAddOep::ckks_add_pt_const_znx_into(self, &mut dst_ct, &a_ct, dst_coeff, pt_znx, pt_coeff, scratch);
        let new_meta = dst_ct.meta();
        drop(dst_ct);
        dst.set_meta(new_meta);
        res
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
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos,
    {
        let dst_meta = dst.meta();
        let mut dst_ct = CKKSCiphertext::from_inner(GLWEToBackendMut::to_backend_mut(dst), dst_meta);
        let res = CKKSAddOep::ckks_add_pt_const_znx_assign(self, &mut dst_ct, dst_coeff, pt_znx, pt_coeff, scratch);
        let new_meta = dst_ct.meta();
        drop(dst_ct);
        dst.set_meta(new_meta);
        res
    }
}

unsafe impl<BE: Backend + poulpy_hal::oep::HalVecZnxImpl<BE>> CKKSAddOpsUnsafe<BE> for Module<BE>
where
    Module<BE>: CKKSAddDefault<BE> + GLWEShift<BE> + GLWEAdd<BE> + VecZnxRshAddCoeffIntoBackend<BE> + VecZnxRshAddIntoBackend<BE>,
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
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        A: CKKSCiphertextToBackendRef<BE> + CKKSInfos,
        B: CKKSCiphertextToBackendRef<BE> + CKKSInfos,
    {
        let dst_meta = dst.meta();
        let mut dst_ct = CKKSCiphertext::from_inner(GLWEToBackendMut::to_backend_mut(dst), dst_meta);
        let a_ct = CKKSCiphertext::from_inner(GLWEToBackendRef::to_backend_ref(a), a.meta());
        let b_ct = CKKSCiphertext::from_inner(GLWEToBackendRef::to_backend_ref(b), b.meta());
        let mut dst_ct_ref = &mut dst_ct;
        let a_ct_ref = &a_ct;
        let b_ct_ref = &b_ct;
        let res = self.ckks_add_into_unsafe_default(&mut dst_ct_ref, &a_ct_ref, &b_ct_ref, scratch);
        let new_meta = dst_ct.meta();
        drop(dst_ct);
        dst.set_meta(new_meta);
        res
    }

    unsafe fn ckks_add_assign_unsafe<Dst, A>(&self, dst: &mut Dst, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        A: CKKSCiphertextToBackendRef<BE> + CKKSInfos,
    {
        let dst_meta = dst.meta();
        let mut dst_ct = CKKSCiphertext::from_inner(GLWEToBackendMut::to_backend_mut(dst), dst_meta);
        let a_ct = CKKSCiphertext::from_inner(GLWEToBackendRef::to_backend_ref(a), a.meta());
        let mut dst_ct_ref = &mut dst_ct;
        let a_ct_ref = &a_ct;
        let res = self.ckks_add_assign_unsafe_default(&mut dst_ct_ref, &a_ct_ref, scratch);
        let new_meta = dst_ct.meta();
        drop(dst_ct);
        dst.set_meta(new_meta);
        res
    }

    unsafe fn ckks_add_pt_vec_znx_into_unsafe<Dst, A, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        A: CKKSCiphertextToBackendRef<BE> + CKKSInfos,
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos,
    {
        let dst_meta = dst.meta();
        let mut dst_ct = CKKSCiphertext::from_inner(GLWEToBackendMut::to_backend_mut(dst), dst_meta);
        let a_ct = CKKSCiphertext::from_inner(GLWEToBackendRef::to_backend_ref(a), a.meta());
        let mut dst_ct_ref = &mut dst_ct;
        let a_ct_ref = &a_ct;
        let res = self.ckks_add_pt_vec_znx_into_unsafe_default(&mut dst_ct_ref, &a_ct_ref, pt_znx, scratch);
        let new_meta = dst_ct.meta();
        drop(dst_ct);
        dst.set_meta(new_meta);
        res
    }

    unsafe fn ckks_add_pt_vec_znx_assign_unsafe<Dst, P>(
        &self,
        dst: &mut Dst,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos,
    {
        let dst_meta = dst.meta();
        let mut dst_ct = CKKSCiphertext::from_inner(GLWEToBackendMut::to_backend_mut(dst), dst_meta);
        let mut dst_ct_ref = &mut dst_ct;
        let res = self.ckks_add_pt_vec_znx_assign_unsafe_default(&mut dst_ct_ref, pt_znx, scratch);
        let new_meta = dst_ct.meta();
        drop(dst_ct);
        dst.set_meta(new_meta);
        res
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
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        A: CKKSCiphertextToBackendRef<BE> + CKKSInfos,
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos,
    {
        let dst_meta = dst.meta();
        let mut dst_ct = CKKSCiphertext::from_inner(GLWEToBackendMut::to_backend_mut(dst), dst_meta);
        let a_ct = CKKSCiphertext::from_inner(GLWEToBackendRef::to_backend_ref(a), a.meta());
        let mut dst_ct_ref = &mut dst_ct;
        let a_ct_ref = &a_ct;
        let res = self.ckks_add_pt_const_znx_into_unsafe_default(&mut dst_ct_ref, &a_ct_ref, dst_coeff, pt_znx, pt_coeff, scratch);
        let new_meta = dst_ct.meta();
        drop(dst_ct);
        dst.set_meta(new_meta);
        res
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
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos,
    {
        let dst_meta = dst.meta();
        let mut dst_ct = CKKSCiphertext::from_inner(GLWEToBackendMut::to_backend_mut(dst), dst_meta);
        let mut dst_ct_ref = &mut dst_ct;
        let res = self.ckks_add_pt_const_znx_assign_unsafe_default(&mut dst_ct_ref, dst_coeff, pt_znx, pt_coeff, scratch);
        let new_meta = dst_ct.meta();
        drop(dst_ct);
        dst.set_meta(new_meta);
        res
    }
}
