use anyhow::Result;
use poulpy_core::{
    GLWEAdd, GLWECopy, GLWEMulConst, GLWEMulPlain, GLWERotate, GLWETensoring, ScratchArenaTakeCore,
    layouts::{GGLWEInfos, GLWEInfos, LWEInfos, ModuleCoreAlloc, prepared::GLWETensorKeyPreparedToBackendRef},
};
use poulpy_hal::{
    api::{ScratchAvailable, VecZnxCopyBackend},
    layouts::{Backend, Module, ScratchArena},
};

use crate::{CKKSInfos, CKKSMeta, GLWEToBackendMut, GLWEToBackendRef, SetCKKSInfos, oep::CKKSImpl};

pub(crate) trait CKKSMulOep<BE: Backend + CKKSImpl<BE>> {
    fn ckks_mul_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Self: GLWETensoring<BE>;

    fn ckks_square_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Self: GLWETensoring<BE>;

    fn ckks_mul_pt_vec_znx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulPlain<BE>;

    fn ckks_mul_pt_const_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulConst<BE> + GLWERotate<BE>;

    fn ckks_mul_into<Dst, A, B, T>(&self, dst: &mut Dst, a: &A, b: &B, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        B: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
        Self: GLWECopy<BE> + GLWETensoring<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_mul_assign<Dst, A, T>(&self, dst: &mut Dst, a: &A, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
        Self: GLWECopy<BE> + GLWETensoring<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_square_into<Dst, A, T>(&self, dst: &mut Dst, a: &A, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
        Self: GLWECopy<BE> + GLWETensoring<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_square_assign<Dst, T>(&self, dst: &mut Dst, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
        Self: GLWECopy<BE> + GLWETensoring<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_mul_pt_vec_znx_into<Dst, A, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos + CKKSInfos,
        Self: GLWECopy<BE> + GLWEMulPlain<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf> + VecZnxCopyBackend<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_mul_pt_vec_znx_assign<Dst, P>(&self, dst: &mut Dst, pt_znx: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos + CKKSInfos,
        Self: GLWECopy<BE> + GLWEMulPlain<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf> + VecZnxCopyBackend<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_mul_pt_const_znx_into<Dst, A, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos + CKKSInfos,
        Self: GLWEAdd<BE> + GLWECopy<BE> + GLWEMulConst<BE> + GLWERotate<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_mul_pt_const_znx_assign<Dst, P>(
        &self,
        dst: &mut Dst,
        pt_znx: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos + CKKSInfos,
        Self: GLWEAdd<BE> + GLWECopy<BE> + GLWEMulConst<BE> + GLWERotate<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;
}

impl<BE: Backend + CKKSImpl<BE>> CKKSMulOep<BE> for Module<BE> {
    fn ckks_mul_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Self: GLWETensoring<BE>,
    {
        BE::ckks_mul_tmp_bytes(self, res, tsk)
    }

    fn ckks_square_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Self: GLWETensoring<BE>,
    {
        BE::ckks_square_tmp_bytes(self, res, tsk)
    }

    fn ckks_mul_pt_vec_znx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulPlain<BE>,
    {
        BE::ckks_mul_pt_vec_znx_tmp_bytes(self, res, a, b)
    }

    fn ckks_mul_pt_const_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulConst<BE> + GLWERotate<BE>,
    {
        BE::ckks_mul_pt_const_tmp_bytes(self, res, a, b)
    }

    fn ckks_mul_into<Dst, A, B, T>(&self, dst: &mut Dst, a: &A, b: &B, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        B: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
        Self: GLWECopy<BE> + GLWETensoring<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_mul_into(self, dst, a, b, tsk, scratch)
    }

    fn ckks_mul_assign<Dst, A, T>(&self, dst: &mut Dst, a: &A, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
        Self: GLWECopy<BE> + GLWETensoring<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_mul_assign(self, dst, a, tsk, scratch)
    }

    fn ckks_square_into<Dst, A, T>(&self, dst: &mut Dst, a: &A, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
        Self: GLWECopy<BE> + GLWETensoring<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_square_into(self, dst, a, tsk, scratch)
    }

    fn ckks_square_assign<Dst, T>(&self, dst: &mut Dst, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
        Self: GLWECopy<BE> + GLWETensoring<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_square_assign(self, dst, tsk, scratch)
    }

    fn ckks_mul_pt_vec_znx_into<Dst, A, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos + CKKSInfos,
        Self: GLWECopy<BE> + GLWEMulPlain<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf> + VecZnxCopyBackend<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_mul_pt_vec_znx_into(self, dst, a, pt_znx, scratch)
    }

    fn ckks_mul_pt_vec_znx_assign<Dst, P>(&self, dst: &mut Dst, pt_znx: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos + CKKSInfos,
        Self: GLWECopy<BE> + GLWEMulPlain<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf> + VecZnxCopyBackend<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_mul_pt_vec_znx_assign(self, dst, pt_znx, scratch)
    }

    fn ckks_mul_pt_const_znx_into<Dst, A, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos + CKKSInfos,
        Self: GLWEAdd<BE> + GLWECopy<BE> + GLWEMulConst<BE> + GLWERotate<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_mul_pt_const_znx_into(self, dst, a, pt_znx, scratch)
    }

    fn ckks_mul_pt_const_znx_assign<Dst, P>(
        &self,
        dst: &mut Dst,
        pt_znx: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos + CKKSInfos,
        Self: GLWEAdd<BE> + GLWECopy<BE> + GLWEMulConst<BE> + GLWERotate<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_mul_pt_const_znx_assign(self, dst, pt_znx, pt_coeff, scratch)
    }
}
