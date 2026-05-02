use anyhow::Result;
use poulpy_core::{
    GLWEAdd, GLWECopy, GLWEMulConst, GLWEMulPlain, GLWERotate, GLWETensoring, ScratchArenaTakeCore,
    layouts::{GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, prepared::GLWETensorKeyPreparedToBackendRef},
};
use poulpy_hal::{
    api::{ModuleN, ScratchAvailable},
    layouts::{Backend, Module, ScratchArena},
};

use crate::{CKKSCiphertextToBackendMut, CKKSCiphertextToBackendRef, CKKSPlaintexToBackendRef};

use crate::{CKKSInfos, SetCKKSInfos, layouts::CKKSCiphertext, oep::CKKSImpl};

use crate::leveled::{api::CKKSMulOps, oep::CKKSMulOep};

impl<BE: Backend + CKKSImpl<BE>> CKKSMulOps<BE> for Module<BE>
where
    Module<BE>: GLWEAdd<BE>
        + GLWECopy<BE>
        + GLWEMulConst<BE>
        + GLWEMulPlain<BE>
        + GLWERotate<BE>
        + GLWETensoring<BE>
        + ModuleN
        + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
        + poulpy_hal::api::VecZnxCopyBackend<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_mul_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
    {
        CKKSMulOep::ckks_mul_tmp_bytes(self, res, tsk)
    }

    fn ckks_square_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
    {
        CKKSMulOep::ckks_square_tmp_bytes(self, res, tsk)
    }

    fn ckks_mul_pt_vec_znx_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: GLWEInfos + CKKSInfos,
        A: GLWEInfos + CKKSInfos,
        P: CKKSInfos,
    {
        let b = b.meta();
        CKKSMulOep::ckks_mul_pt_vec_znx_tmp_bytes(self, res, a, &b)
    }

    fn ckks_mul_pt_const_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: GLWEInfos + CKKSInfos,
        A: GLWEInfos + CKKSInfos,
        P: CKKSInfos,
    {
        let b = b.meta();
        CKKSMulOep::ckks_mul_pt_const_tmp_bytes(self, res, a, &b)
    }

    fn ckks_mul_into<Dst, A, B, T>(&self, dst: &mut Dst, a: &A, b: &B, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        A: CKKSCiphertextToBackendRef<BE> + CKKSInfos,
        B: CKKSCiphertextToBackendRef<BE> + CKKSInfos,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
    {
        let dst_meta = dst.meta();
        let mut dst_ct = CKKSCiphertext::from_inner(GLWEToBackendMut::to_backend_mut(dst), dst_meta);
        let a_ct = CKKSCiphertext::from_inner(GLWEToBackendRef::to_backend_ref(a), a.meta());
        let b_ct = CKKSCiphertext::from_inner(GLWEToBackendRef::to_backend_ref(b), b.meta());
        let tsk_ref = tsk.to_backend_ref();
        let res = CKKSMulOep::ckks_mul_into(self, &mut dst_ct, &a_ct, &b_ct, &tsk_ref, scratch);
        let new_meta = dst_ct.meta();
        drop(dst_ct);
        dst.set_meta(new_meta);
        res
    }

    fn ckks_mul_assign<Dst, A, T>(&self, dst: &mut Dst, a: &A, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSCiphertextToBackendRef<BE> + CKKSInfos + SetCKKSInfos,
        A: CKKSCiphertextToBackendRef<BE> + CKKSInfos,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
    {
        let dst_meta = dst.meta();
        let mut dst_ct = CKKSCiphertext::from_inner(GLWEToBackendMut::to_backend_mut(dst), dst_meta);
        let a_ct = CKKSCiphertext::from_inner(GLWEToBackendRef::to_backend_ref(a), a.meta());
        let tsk_ref = tsk.to_backend_ref();
        let res = CKKSMulOep::ckks_mul_assign(self, &mut dst_ct, &a_ct, &tsk_ref, scratch);
        let new_meta = dst_ct.meta();
        drop(dst_ct);
        dst.set_meta(new_meta);
        res
    }

    fn ckks_square_into<Dst, A, T>(&self, dst: &mut Dst, a: &A, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        A: CKKSCiphertextToBackendRef<BE> + CKKSInfos,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
    {
        let dst_meta = dst.meta();
        let mut dst_ct = CKKSCiphertext::from_inner(GLWEToBackendMut::to_backend_mut(dst), dst_meta);
        let a_ct = CKKSCiphertext::from_inner(GLWEToBackendRef::to_backend_ref(a), a.meta());
        let tsk_ref = tsk.to_backend_ref();
        let res = CKKSMulOep::ckks_square_into(self, &mut dst_ct, &a_ct, &tsk_ref, scratch);
        let new_meta = dst_ct.meta();
        drop(dst_ct);
        dst.set_meta(new_meta);
        res
    }

    fn ckks_square_assign<Dst, T>(&self, dst: &mut Dst, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSCiphertextToBackendRef<BE> + CKKSInfos + SetCKKSInfos,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
    {
        let dst_meta = dst.meta();
        let mut dst_ct = CKKSCiphertext::from_inner(GLWEToBackendMut::to_backend_mut(dst), dst_meta);
        let tsk_ref = tsk.to_backend_ref();
        let res = CKKSMulOep::ckks_square_assign(self, &mut dst_ct, &tsk_ref, scratch);
        let new_meta = dst_ct.meta();
        drop(dst_ct);
        dst.set_meta(new_meta);
        res
    }

    fn ckks_mul_pt_vec_znx_into<Dst, A, P>(
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
        let res = CKKSMulOep::ckks_mul_pt_vec_znx_into(self, &mut dst_ct, &a_ct, pt_znx, scratch);
        let new_meta = dst_ct.meta();
        drop(dst_ct);
        dst.set_meta(new_meta);
        res
    }

    fn ckks_mul_pt_vec_znx_assign<Dst, P>(&self, dst: &mut Dst, pt_znx: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSCiphertextToBackendRef<BE> + CKKSInfos + SetCKKSInfos,
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos,
    {
        let dst_meta = dst.meta();
        let mut dst_ct = CKKSCiphertext::from_inner(GLWEToBackendMut::to_backend_mut(dst), dst_meta);
        let res = CKKSMulOep::ckks_mul_pt_vec_znx_assign(self, &mut dst_ct, pt_znx, scratch);
        let new_meta = dst_ct.meta();
        drop(dst_ct);
        dst.set_meta(new_meta);
        res
    }

    fn ckks_mul_pt_const_znx_into<Dst, A, P>(
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
        let res = CKKSMulOep::ckks_mul_pt_const_znx_into(self, &mut dst_ct, &a_ct, pt_znx, scratch);
        let new_meta = dst_ct.meta();
        drop(dst_ct);
        dst.set_meta(new_meta);
        res
    }

    fn ckks_mul_pt_const_znx_assign<Dst, P>(&self, dst: &mut Dst, pt_znx: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: CKKSCiphertextToBackendMut<BE> + CKKSCiphertextToBackendRef<BE> + CKKSInfos + SetCKKSInfos,
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos,
    {
        let dst_meta = dst.meta();
        let mut dst_ct = CKKSCiphertext::from_inner(GLWEToBackendMut::to_backend_mut(dst), dst_meta);
        let res = CKKSMulOep::ckks_mul_pt_const_znx_assign(self, &mut dst_ct, pt_znx, scratch);
        let new_meta = dst_ct.meta();
        drop(dst_ct);
        dst.set_meta(new_meta);
        res
    }
}
