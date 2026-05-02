use anyhow::Result;
use poulpy_core::{
    GLWEAdd, GLWECopy, GLWEMulConst, GLWEMulPlain, GLWERotate, GLWETensoring, ScratchArenaTakeCore,
    layouts::{GGLWEInfos, GLWEInfos, prepared::GLWETensorKeyPreparedBackendRef},
};
use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, Module, ScratchArena},
};

use crate::{CKKSCiphertextMut, CKKSCiphertextRef, CKKSInfos, CKKSMeta, CKKSPlaintexToBackendRef, oep::CKKSImpl};

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

    fn ckks_mul_into(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        a: &CKKSCiphertextRef<'_, BE>,
        b: &CKKSCiphertextRef<'_, BE>,
        tsk: &GLWETensorKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWECopy<BE> + GLWETensoring<BE> + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_mul_assign(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        a: &CKKSCiphertextRef<'_, BE>,
        tsk: &GLWETensorKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWECopy<BE> + GLWETensoring<BE> + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_square_into(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        a: &CKKSCiphertextRef<'_, BE>,
        tsk: &GLWETensorKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWECopy<BE> + GLWETensoring<BE> + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_square_assign(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        tsk: &GLWETensorKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWECopy<BE> + GLWETensoring<BE> + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_mul_pt_vec_znx_into<P>(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        a: &CKKSCiphertextRef<'_, BE>,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos,
        Self: GLWECopy<BE>
            + GLWEMulPlain<BE>
            + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
            + poulpy_hal::api::VecZnxCopyBackend<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_mul_pt_vec_znx_assign<P>(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos,
        Self: GLWECopy<BE>
            + GLWEMulPlain<BE>
            + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
            + poulpy_hal::api::VecZnxCopyBackend<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_mul_pt_const_znx_into<P>(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        a: &CKKSCiphertextRef<'_, BE>,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos,
        Self: GLWEAdd<BE>
            + GLWECopy<BE>
            + GLWEMulConst<BE>
            + GLWERotate<BE>
            + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_mul_pt_const_znx_assign<P>(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos,
        Self: GLWEAdd<BE>
            + GLWECopy<BE>
            + GLWEMulConst<BE>
            + GLWERotate<BE>
            + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
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

    fn ckks_mul_into(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        a: &CKKSCiphertextRef<'_, BE>,
        b: &CKKSCiphertextRef<'_, BE>,
        tsk: &GLWETensorKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWECopy<BE> + GLWETensoring<BE> + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_mul_into(self, dst, a, b, tsk, scratch)
    }

    fn ckks_mul_assign(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        a: &CKKSCiphertextRef<'_, BE>,
        tsk: &GLWETensorKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWECopy<BE> + GLWETensoring<BE> + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_mul_assign(self, dst, a, tsk, scratch)
    }

    fn ckks_square_into(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        a: &CKKSCiphertextRef<'_, BE>,
        tsk: &GLWETensorKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWECopy<BE> + GLWETensoring<BE> + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_square_into(self, dst, a, tsk, scratch)
    }

    fn ckks_square_assign(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        tsk: &GLWETensorKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWECopy<BE> + GLWETensoring<BE> + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_square_assign(self, dst, tsk, scratch)
    }

    fn ckks_mul_pt_vec_znx_into<P>(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        a: &CKKSCiphertextRef<'_, BE>,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos,
        Self: GLWECopy<BE>
            + GLWEMulPlain<BE>
            + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
            + poulpy_hal::api::VecZnxCopyBackend<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_mul_pt_vec_znx_into(self, dst, a, pt_znx, scratch)
    }

    fn ckks_mul_pt_vec_znx_assign<P>(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos,
        Self: GLWECopy<BE>
            + GLWEMulPlain<BE>
            + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
            + poulpy_hal::api::VecZnxCopyBackend<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_mul_pt_vec_znx_assign(self, dst, pt_znx, scratch)
    }

    fn ckks_mul_pt_const_znx_into<P>(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        a: &CKKSCiphertextRef<'_, BE>,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos,
        Self: GLWEAdd<BE>
            + GLWECopy<BE>
            + GLWEMulConst<BE>
            + GLWERotate<BE>
            + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_mul_pt_const_znx_into(self, dst, a, pt_znx, scratch)
    }

    fn ckks_mul_pt_const_znx_assign<P>(
        &self,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos,
        Self: GLWEAdd<BE>
            + GLWECopy<BE>
            + GLWEMulConst<BE>
            + GLWERotate<BE>
            + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_mul_pt_const_znx_assign(self, dst, pt_znx, scratch)
    }
}
