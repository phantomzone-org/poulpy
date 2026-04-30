use anyhow::Result;
use poulpy_core::{
    GLWEAdd, GLWECopy, GLWEMulConst, GLWEMulPlain, GLWENormalize, GLWERotate, GLWEShift, GLWESub, GLWETensoring,
    ScratchArenaTakeCore,
    layouts::{GGLWEInfos, GLWEInfos, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, ModuleCoreAlloc},
};
use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, VecZnxAddAssignBackend, VecZnxCopyBackend, VecZnxRshAddIntoBackend},
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
    leveled::api::{
        add::{CKKSAddOps, CKKSAddOpsUnsafe},
        mul::CKKSMulOps,
        sub::CKKSSubOps,
    },
    oep::CKKSImpl,
};

pub trait CKKSAddManyOps<BE: Backend + CKKSImpl<BE>> {
    fn ckks_add_many_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE> + CKKSAddOps<BE>;

    fn ckks_add_many<Dst: Data, Src: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        inputs: &[&CKKSCiphertext<Src>],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd<BE> + GLWEShift<BE> + GLWENormalize<BE> + CKKSAddOpsUnsafe<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<Src>: GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;
}

pub trait CKKSMulManyOps<BE: Backend + CKKSImpl<BE>> {
    fn ckks_mul_many_tmp_bytes<R, T>(&self, n: usize, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Self: GLWETensoring<BE> + CKKSMulOps<BE>;

    fn ckks_mul_many<Dst: Data, Src: Data, T: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        inputs: &[&CKKSCiphertext<Src>],
        tsk: &GLWETensorKeyPrepared<T, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWECopy<BE> + GLWEShift<BE> + GLWETensoring<BE> + CKKSMulOps<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
        CKKSCiphertext<Src>: GLWEToBackendRef<BE> + GLWEInfos,
        GLWETensorKeyPrepared<T, BE>: poulpy_core::layouts::prepared::GLWETensorKeyPreparedToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;
}

pub trait CKKSMulAddOps<BE: Backend + CKKSImpl<BE>> {
    fn ckks_mul_add_ct_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Self: GLWEShift<BE> + GLWETensoring<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>;

    fn ckks_mul_add_pt_vec_znx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulPlain<BE> + GLWEShift<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>;

    fn ckks_mul_add_pt_vec_rnx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: ModuleN + GLWEMulPlain<BE> + GLWEShift<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>;

    fn ckks_mul_add_pt_const_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulConst<BE> + GLWERotate<BE> + GLWEShift<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>;

    fn ckks_mul_add_ct_into<Dst: Data, A: Data, B: Data, T: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        b: &CKKSCiphertext<B>,
        tsk: &GLWETensorKeyPrepared<T, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd<BE>
            + GLWECopy<BE>
            + GLWEShift<BE>
            + GLWETensoring<BE>
            + CKKSAddOps<BE>
            + CKKSMulOps<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + GLWEInfos,
        CKKSCiphertext<B>: GLWEToBackendRef<BE> + GLWEInfos,
        GLWETensorKeyPrepared<T, BE>: poulpy_core::layouts::prepared::GLWETensorKeyPreparedToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_mul_add_pt_vec_znx_into<Dst: Data, A: Data, P: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt_znx: &CKKSPlaintextVecZnx<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd<BE>
            + GLWECopy<BE>
            + GLWEMulPlain<BE>
            + GLWEShift<BE>
            + CKKSAddOps<BE>
            + CKKSMulOps<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
            + VecZnxCopyBackend<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + GLWEInfos,
        CKKSPlaintextVecZnx<P>: poulpy_core::layouts::GLWEPlaintextToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_mul_add_pt_vec_rnx_into<Dst: Data, A: Data, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: ModuleN
            + GLWEAdd<BE>
            + GLWECopy<BE>
            + GLWEMulPlain<BE>
            + GLWEShift<BE>
            + CKKSAddOps<BE>
            + CKKSMulOps<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
            + VecZnxCopyBackend<BE>,
        BE: HostBackend<OwnedBuf = Vec<u8>>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion;

    fn ckks_mul_add_pt_const_znx_into<Dst: Data, A: Data>(
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
            + GLWEShift<BE>
            + CKKSAddOps<BE>
            + CKKSMulOps<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_mul_add_pt_const_rnx_into<Dst: Data, A: Data, F>(
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
            + GLWEShift<BE>
            + CKKSAddOps<BE>
            + CKKSMulOps<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion;
}

pub trait CKKSMulSubOps<BE: Backend + CKKSImpl<BE>> {
    fn ckks_mul_sub_ct_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Self: GLWEShift<BE> + GLWETensoring<BE> + CKKSSubOps<BE> + CKKSMulOps<BE>;

    fn ckks_mul_sub_pt_vec_znx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulPlain<BE> + GLWEShift<BE> + CKKSSubOps<BE> + CKKSMulOps<BE>;

    fn ckks_mul_sub_pt_vec_rnx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: ModuleN + GLWEMulPlain<BE> + GLWEShift<BE> + CKKSSubOps<BE> + CKKSMulOps<BE>;

    fn ckks_mul_sub_pt_const_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulConst<BE> + GLWERotate<BE> + GLWEShift<BE> + CKKSSubOps<BE> + CKKSMulOps<BE>;

    fn ckks_mul_sub_ct_into<Dst: Data, A: Data, B: Data, T: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        b: &CKKSCiphertext<B>,
        tsk: &GLWETensorKeyPrepared<T, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWECopy<BE>
            + GLWESub<BE>
            + GLWEShift<BE>
            + GLWETensoring<BE>
            + CKKSSubOps<BE>
            + CKKSMulOps<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + GLWEInfos,
        CKKSCiphertext<B>: GLWEToBackendRef<BE> + GLWEInfos,
        GLWETensorKeyPrepared<T, BE>: poulpy_core::layouts::prepared::GLWETensorKeyPreparedToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_mul_sub_pt_vec_znx_into<Dst: Data, A: Data, P: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt_znx: &CKKSPlaintextVecZnx<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWECopy<BE>
            + GLWESub<BE>
            + GLWEMulPlain<BE>
            + GLWEShift<BE>
            + CKKSSubOps<BE>
            + CKKSMulOps<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
            + VecZnxCopyBackend<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + GLWEInfos,
        CKKSPlaintextVecZnx<P>: poulpy_core::layouts::GLWEPlaintextToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_mul_sub_pt_vec_rnx_into<Dst: Data, A: Data, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: ModuleN
            + GLWECopy<BE>
            + GLWESub<BE>
            + GLWEMulPlain<BE>
            + GLWEShift<BE>
            + CKKSSubOps<BE>
            + CKKSMulOps<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
            + VecZnxCopyBackend<BE>,
        BE: HostBackend<OwnedBuf = Vec<u8>>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion;

    fn ckks_mul_sub_pt_const_znx_into<Dst: Data, A: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd<BE>
            + GLWECopy<BE>
            + GLWESub<BE>
            + GLWEMulConst<BE>
            + GLWERotate<BE>
            + GLWEShift<BE>
            + CKKSSubOps<BE>
            + CKKSMulOps<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_mul_sub_pt_const_rnx_into<Dst: Data, A: Data, F>(
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
            + GLWESub<BE>
            + GLWEMulConst<BE>
            + GLWERotate<BE>
            + GLWEShift<BE>
            + CKKSSubOps<BE>
            + CKKSMulOps<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: GLWEToBackendRef<BE> + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion;
}

pub trait CKKSDotProductOps<BE: Backend + CKKSImpl<BE>> {
    fn ckks_dot_product_ct_tmp_bytes<R, T>(&self, n: usize, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Self: GLWEShift<BE> + GLWETensoring<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>;

    fn ckks_dot_product_pt_vec_znx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulPlain<BE> + GLWEShift<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>;

    fn ckks_dot_product_pt_vec_rnx_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: ModuleN + GLWEMulPlain<BE> + GLWEShift<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>;

    fn ckks_dot_product_pt_const_tmp_bytes<R, A>(&self, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulConst<BE> + GLWERotate<BE> + GLWEShift<BE> + CKKSAddOps<BE> + CKKSMulOps<BE>;

    fn ckks_dot_product_ct<Dst: Data, D: Data, E: Data, T: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &[&CKKSCiphertext<D>],
        b: &[&CKKSCiphertext<E>],
        tsk: &GLWETensorKeyPrepared<T, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd<BE>
            + GLWECopy<BE>
            + GLWEShift<BE>
            + GLWENormalize<BE>
            + GLWETensoring<BE>
            + VecZnxAddAssignBackend<BE>
            + CKKSAddOpsUnsafe<BE>
            + CKKSMulOps<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<D>: GLWEToBackendRef<BE> + GLWEInfos,
        CKKSCiphertext<E>: GLWEToBackendRef<BE> + GLWEInfos,
        GLWETensorKeyPrepared<T, BE>: poulpy_core::layouts::prepared::GLWETensorKeyPreparedToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_dot_product_pt_vec_znx<Dst: Data, D: Data, E: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &[&CKKSCiphertext<D>],
        b: &[&CKKSPlaintextVecZnx<E>],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd<BE>
            + GLWECopy<BE>
            + GLWEMulPlain<BE>
            + GLWEShift<BE>
            + GLWENormalize<BE>
            + VecZnxRshAddIntoBackend<BE>
            + CKKSAddOpsUnsafe<BE>
            + CKKSMulOps<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
            + VecZnxCopyBackend<BE>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<D>: GLWEToBackendRef<BE> + GLWEInfos,
        CKKSPlaintextVecZnx<E>: poulpy_core::layouts::GLWEPlaintextToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_dot_product_pt_vec_rnx<Dst: Data, D: Data, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &[&CKKSCiphertext<D>],
        b: &[&CKKSPlaintextVecRnx<F>],
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: ModuleN
            + GLWEAdd<BE>
            + GLWECopy<BE>
            + GLWEMulPlain<BE>
            + GLWEShift<BE>
            + GLWENormalize<BE>
            + VecZnxRshAddIntoBackend<BE>
            + CKKSAddOpsUnsafe<BE>
            + CKKSMulOps<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
            + VecZnxCopyBackend<BE>,
        BE: HostBackend<OwnedBuf = Vec<u8>>,
        CKKSCiphertext<D>: GLWEToBackendRef<BE> + GLWEInfos,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion;

    fn ckks_dot_product_pt_const_znx<Dst: Data, D: Data>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &[&CKKSCiphertext<D>],
        b: &[&CKKSPlaintextCstZnx],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd<BE>
            + GLWECopy<BE>
            + GLWEMulConst<BE>
            + GLWERotate<BE>
            + GLWEShift<BE>
            + GLWENormalize<BE>
            + CKKSAddOpsUnsafe<BE>
            + CKKSMulOps<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<D>: GLWEToBackendRef<BE> + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_dot_product_pt_const_rnx<Dst: Data, D: Data, F>(
        &self,
        dst: &mut CKKSCiphertext<Dst>,
        a: &[&CKKSCiphertext<D>],
        b: &[&CKKSPlaintextCstRnx<F>],
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd<BE>
            + GLWECopy<BE>
            + GLWEMulConst<BE>
            + GLWERotate<BE>
            + GLWEShift<BE>
            + GLWENormalize<BE>
            + CKKSAddOpsUnsafe<BE>
            + CKKSMulOps<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: GLWEToBackendMut<BE>,
        CKKSCiphertext<D>: GLWEToBackendRef<BE> + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion;
}

// Suppress unused import warnings
#[allow(unused_imports)]
use poulpy_core::layouts::LWEInfos;
