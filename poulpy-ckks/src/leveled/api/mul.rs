use anyhow::Result;
use poulpy_core::{
    GLWEAdd, GLWECopy, GLWEMulConst, GLWEMulPlain, GLWERotate, GLWETensoring, ScratchArenaTakeCore,
    layouts::{GGLWEInfos, GLWEInfos, GLWETensorKeyPrepared},
};
use poulpy_hal::{
    api::{ModuleN, ScratchAvailable},
    layouts::{Backend, Data, HostBackend, ScratchArena},
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

pub trait CKKSMulOps<BE: Backend + CKKSImpl<BE>> {
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

    fn ckks_mul_pt_vec_rnx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: ModuleN + GLWEMulPlain<BE>;

    fn ckks_mul_pt_const_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulConst<BE> + GLWERotate<BE>;

    fn ckks_mul_into<Dst: Data, A: Data, B: Data, T: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        b: &CKKSCiphertext<B>,
        tsk: &GLWETensorKeyPrepared<T, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWECopy<BE> + GLWETensoring<BE> + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: poulpy_core::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        CKKSCiphertext<B>: poulpy_core::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        GLWETensorKeyPrepared<T, BE>: poulpy_core::layouts::prepared::GLWETensorKeyPreparedToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_mul_assign<Dst: Data, A: Data, T: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        tsk: &GLWETensorKeyPrepared<T, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWECopy<BE> + GLWETensoring<BE> + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE> + poulpy_core::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        CKKSCiphertext<A>: poulpy_core::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        GLWETensorKeyPrepared<T, BE>: poulpy_core::layouts::prepared::GLWETensorKeyPreparedToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_square_into<Dst: Data, A: Data, T: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        tsk: &GLWETensorKeyPrepared<T, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWECopy<BE> + GLWETensoring<BE> + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: poulpy_core::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        GLWETensorKeyPrepared<T, BE>: poulpy_core::layouts::prepared::GLWETensorKeyPreparedToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_square_assign<Dst: Data, T: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        tsk: &GLWETensorKeyPrepared<T, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWECopy<BE> + GLWETensoring<BE> + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE> + poulpy_core::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        GLWETensorKeyPrepared<T, BE>: poulpy_core::layouts::prepared::GLWETensorKeyPreparedToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_mul_pt_vec_znx_into<Dst: Data, A: Data, P: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt_znx: &CKKSPlaintextVecZnx<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWECopy<BE>
            + GLWEMulPlain<BE>
            + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
            + poulpy_hal::api::VecZnxCopyBackend<BE>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: poulpy_core::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        CKKSPlaintextVecZnx<P>: poulpy_core::layouts::GLWEPlaintextToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_mul_pt_vec_znx_assign<Dst: Data, P: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        pt_znx: &CKKSPlaintextVecZnx<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWECopy<BE>
            + GLWEMulPlain<BE>
            + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
            + poulpy_hal::api::VecZnxCopyBackend<BE>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE> + poulpy_core::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        CKKSPlaintextVecZnx<P>: poulpy_core::layouts::GLWEPlaintextToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_mul_pt_vec_rnx_into<Dst: Data, A: Data, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: HostBackend<OwnedBuf = Vec<u8>>,
        Self: GLWECopy<BE>
            + GLWEMulPlain<BE>
            + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
            + ModuleN
            + poulpy_hal::api::VecZnxCopyBackend<BE>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: poulpy_core::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_mul_pt_vec_rnx_assign<Dst: Data, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: HostBackend<OwnedBuf = Vec<u8>>,
        Self: GLWECopy<BE>
            + GLWEMulPlain<BE>
            + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
            + ModuleN
            + poulpy_hal::api::VecZnxCopyBackend<BE>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE> + poulpy_core::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_mul_pt_const_znx_into<Dst: Data, A: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd<BE>
            + GLWECopy<BE>
            + GLWEMulConst<BE>
            + GLWERotate<BE>
            + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: poulpy_core::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_mul_pt_const_znx_assign<Dst: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd<BE>
            + GLWECopy<BE>
            + GLWEMulConst<BE>
            + GLWERotate<BE>
            + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE> + poulpy_core::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_mul_pt_const_rnx_into<Dst: Data, A: Data, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd<BE>
            + GLWECopy<BE>
            + GLWEMulConst<BE>
            + GLWERotate<BE>
            + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: poulpy_core::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion;

    fn ckks_mul_pt_const_rnx_assign<Dst: Data, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd<BE>
            + GLWECopy<BE>
            + GLWEMulConst<BE>
            + GLWERotate<BE>
            + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE> + poulpy_core::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion;
}
