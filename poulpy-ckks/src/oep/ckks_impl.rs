#![allow(clippy::too_many_arguments)]

use anyhow::Result;
use poulpy_core::{
    GLWEAdd, GLWEAutomorphism, GLWECopy, GLWEMulConst, GLWEMulPlain, GLWENegate, GLWERotate, GLWEShift, GLWESub, GLWETensoring,
    ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GGLWEPreparedToBackendRef, GLWEAutomorphismKeyHelper, GLWEAutomorphismKeyPrepared, GLWEInfos, GLWEPlaintext,
        GLWETensorKeyPrepared, GetGaloisElement,
    },
};
use poulpy_hal::{
    api::{
        ModuleN, ScratchAvailable, VecZnxLshBackend, VecZnxLshTmpBytes, VecZnxRshAddIntoBackend, VecZnxRshBackend,
        VecZnxRshSubBackend, VecZnxRshTmpBytes,
    },
    layouts::{Backend, Data, HostBackend, Module, ScratchArena},
};

use crate::{
    CKKSInfos, CKKSMeta,
    layouts::{
        CKKSCiphertext,
        plaintext::{
            CKKSConstPlaintextConversion, CKKSPlaintextConversion, CKKSPlaintextCstRnx, CKKSPlaintextCstZnx, CKKSPlaintextVecRnx,
            CKKSPlaintextVecZnx,
        },
    },
};

/// Backend-owned CKKS leveled-operations extension point.
///
/// `Module<BE>` remains the public execution surface. Backend crates can
/// implement this trait on their backend marker type to override CKKS-level
/// algorithms while preserving the existing module-facing API.
///
/// # Safety
/// Implementors must preserve all CKKS metadata invariants and must obey the
/// scratch, sizing, aliasing, and layout contracts required by the underlying
/// `poulpy-core` and `poulpy-hal` operations they call.
pub unsafe trait CKKSImpl<BE: Backend>: Backend {
    fn ckks_extract_pt_znx_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: VecZnxLshTmpBytes + VecZnxRshTmpBytes;

    fn ckks_extract_pt_znx<Dst: Data, Src: Data, S: CKKSInfos>(
        module: &Module<BE>,
        dst: &mut CKKSPlaintextVecZnx<Dst>,
        src: &GLWEPlaintext<Src>,
        src_meta: &S,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        Module<BE>: VecZnxLshBackend<BE> + VecZnxRshBackend<BE>,
        CKKSPlaintextVecZnx<Dst>: poulpy_core::layouts::GLWEPlaintextToBackendMut<BE>,
        GLWEPlaintext<Src>: poulpy_core::layouts::GLWEPlaintextToBackendRef<BE>;

    fn ckks_add_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: GLWEShift<BE>;

    fn ckks_add_into<Dst: Data, A: Data, B: Data>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        b: &CKKSCiphertext<B>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEAdd<BE> + GLWEShift<BE>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: poulpy_core::layouts::GLWEToBackendRef<BE>,
        CKKSCiphertext<B>: poulpy_core::layouts::GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_add_assign<Dst: Data, A: Data>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEAdd<BE> + GLWEShift<BE>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: poulpy_core::layouts::GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_add_pt_vec_znx_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: GLWEShift<BE> + VecZnxRshTmpBytes;

    fn ckks_add_pt_vec_znx_into<Dst: Data, A: Data, P: Data>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt_znx: &CKKSPlaintextVecZnx<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: VecZnxRshAddIntoBackend<BE> + GLWEShift<BE>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: poulpy_core::layouts::GLWEToBackendRef<BE>,
        CKKSPlaintextVecZnx<P>: poulpy_core::layouts::GLWEPlaintextToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_add_pt_vec_znx_assign<Dst: Data, P: Data>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<Dst>,
        pt_znx: &CKKSPlaintextVecZnx<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: VecZnxRshAddIntoBackend<BE>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        CKKSPlaintextVecZnx<P>: poulpy_core::layouts::GLWEPlaintextToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_add_pt_vec_rnx_tmp_bytes<R: GLWEInfos, A: GLWEInfos>(module: &Module<BE>, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        Module<BE>: ModuleN + GLWEShift<BE> + VecZnxRshTmpBytes;

    fn ckks_add_pt_vec_rnx_into<Dst: Data, A: Data, F>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: ModuleN + VecZnxRshAddIntoBackend<BE> + GLWEShift<BE>,
        BE: HostBackend<OwnedBuf = Vec<u8>>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: poulpy_core::layouts::GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion;

    fn ckks_add_pt_vec_rnx_assign<Dst: Data, F>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<Dst>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: ModuleN + VecZnxRshAddIntoBackend<BE>,
        BE: HostBackend<OwnedBuf = Vec<u8>>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion;

    fn ckks_add_pt_const_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: GLWEShift<BE>;

    fn ckks_add_pt_const_znx_into<Dst: Data, A: Data>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEShift<BE>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: poulpy_core::layouts::GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_add_pt_const_znx_assign<Dst: Data>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<Dst>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_add_pt_const_rnx_into<Dst: Data, A: Data, F>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEShift<BE>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: poulpy_core::layouts::GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion;

    fn ckks_add_pt_const_rnx_assign<Dst: Data, F>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<Dst>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion;

    fn ckks_sub_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: GLWEShift<BE> + VecZnxRshTmpBytes;

    fn ckks_sub_pt_vec_znx_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: GLWEShift<BE> + VecZnxRshTmpBytes;

    fn ckks_sub_into<Dst: Data, A: Data, B: Data>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        b: &CKKSCiphertext<B>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWESub<BE> + GLWEShift<BE>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: poulpy_core::layouts::GLWEToBackendRef<BE>,
        CKKSCiphertext<B>: poulpy_core::layouts::GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_sub_assign<Dst: Data, A: Data>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWESub<BE> + GLWEShift<BE>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: poulpy_core::layouts::GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_sub_pt_vec_znx_into<Dst: Data, A: Data, P: Data>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt_znx: &CKKSPlaintextVecZnx<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: VecZnxRshSubBackend<BE> + GLWEShift<BE>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: poulpy_core::layouts::GLWEToBackendRef<BE>,
        CKKSPlaintextVecZnx<P>: poulpy_core::layouts::GLWEPlaintextToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_sub_pt_vec_znx_assign<Dst: Data, P: Data>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<Dst>,
        pt_znx: &CKKSPlaintextVecZnx<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: VecZnxRshSubBackend<BE>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        CKKSPlaintextVecZnx<P>: poulpy_core::layouts::GLWEPlaintextToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_sub_pt_vec_rnx_tmp_bytes<R: GLWEInfos, A: GLWEInfos>(module: &Module<BE>, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        Module<BE>: ModuleN + GLWEShift<BE> + VecZnxRshTmpBytes;

    fn ckks_sub_pt_vec_rnx_into<Dst: Data, A: Data, F>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: ModuleN + VecZnxRshSubBackend<BE> + GLWEShift<BE>,
        BE: HostBackend<OwnedBuf = Vec<u8>>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: poulpy_core::layouts::GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion;

    fn ckks_sub_pt_vec_rnx_assign<Dst: Data, F>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<Dst>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: ModuleN + VecZnxRshSubBackend<BE>,
        BE: HostBackend<OwnedBuf = Vec<u8>>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion;

    fn ckks_sub_pt_const_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: GLWEShift<BE>;

    fn ckks_sub_pt_const_znx_into<Dst: Data, A: Data>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEShift<BE>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: poulpy_core::layouts::GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_sub_pt_const_znx_assign<Dst: Data>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<Dst>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_sub_pt_const_rnx_into<Dst: Data, A: Data, F>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEShift<BE>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: poulpy_core::layouts::GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion;

    fn ckks_sub_pt_const_rnx_assign<Dst: Data, F>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<Dst>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion;

    fn ckks_neg_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: GLWEShift<BE>;

    fn ckks_neg_into<Dst: Data, Src: Data>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<Dst>,
        src: &CKKSCiphertext<Src>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWENegate<BE> + GLWEShift<BE>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        CKKSCiphertext<Src>: poulpy_core::layouts::GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_neg_assign<Dst: Data>(module: &Module<BE>, dst: &mut CKKSCiphertext<Dst>) -> Result<()>
    where
        Module<BE>: GLWENegate<BE>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>;

    fn ckks_mul_pow2_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: GLWEShift<BE>;

    fn ckks_mul_pow2_into<Dst: Data, Src: Data>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<Dst>,
        src: &CKKSCiphertext<Src>,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEShift<BE>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        CKKSCiphertext<Src>: poulpy_core::layouts::GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;

    fn ckks_mul_pow2_assign<Dst: Data>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<Dst>,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEShift<BE>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;

    fn ckks_div_pow2_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: GLWEShift<BE>;

    fn ckks_div_pow2_into<Dst: Data, Src: Data>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<Dst>,
        src: &CKKSCiphertext<Src>,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEShift<BE> + GLWECopy<BE>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        CKKSCiphertext<Src>: poulpy_core::layouts::GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;

    fn ckks_div_pow2_assign<Dst: Data>(module: &Module<BE>, dst: &mut CKKSCiphertext<Dst>, bits: usize) -> Result<()>;

    fn ckks_rotate_tmp_bytes<C: GLWEInfos, K: GGLWEInfos>(module: &Module<BE>, ct_infos: &C, key_infos: &K) -> usize
    where
        Module<BE>: GLWEAutomorphism<BE>;

    fn ckks_rotate_into<Dst: Data, Src: Data, H, K>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<Dst>,
        src: &CKKSCiphertext<Src>,
        k: i64,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEAutomorphism<BE> + GLWEShift<BE>,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        CKKSCiphertext<Src>: poulpy_core::layouts::GLWEToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;

    fn ckks_rotate_assign<Dst: Data, H, K>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<Dst>,
        k: i64,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEAutomorphism<BE>,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;

    fn ckks_conjugate_tmp_bytes<C: GLWEInfos, K: GGLWEInfos>(module: &Module<BE>, ct_infos: &C, key_infos: &K) -> usize
    where
        Module<BE>: GLWEAutomorphism<BE>;

    fn ckks_conjugate_into<Dst: Data, Src: Data, K: Data>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<Dst>,
        src: &CKKSCiphertext<Src>,
        key: &GLWEAutomorphismKeyPrepared<K, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEAutomorphism<BE> + GLWEShift<BE>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        CKKSCiphertext<Src>: poulpy_core::layouts::GLWEToBackendRef<BE>,
        GLWEAutomorphismKeyPrepared<K, BE>: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;

    fn ckks_conjugate_assign<Dst: Data, K: Data>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<Dst>,
        key: &GLWEAutomorphismKeyPrepared<K, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEAutomorphism<BE>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        GLWEAutomorphismKeyPrepared<K, BE>: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;

    fn ckks_mul_tmp_bytes<R: GLWEInfos, T: GGLWEInfos>(module: &Module<BE>, res: &R, tsk: &T) -> usize
    where
        Module<BE>: GLWETensoring<BE>;

    fn ckks_square_tmp_bytes<R: GLWEInfos, T: GGLWEInfos>(module: &Module<BE>, res: &R, tsk: &T) -> usize
    where
        Module<BE>: GLWETensoring<BE>;

    fn ckks_mul_pt_vec_znx_tmp_bytes<R: GLWEInfos, A: GLWEInfos>(module: &Module<BE>, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        Module<BE>: GLWEMulPlain<BE>;

    fn ckks_mul_pt_vec_rnx_tmp_bytes<R: GLWEInfos, A: GLWEInfos>(module: &Module<BE>, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        Module<BE>: ModuleN + GLWEMulPlain<BE>;

    fn ckks_mul_pt_const_tmp_bytes<R: GLWEInfos, A: GLWEInfos>(module: &Module<BE>, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        Module<BE>: GLWEMulConst<BE> + GLWERotate<BE>;

    fn ckks_mul_into<Dst: Data, A: Data, B: Data, T: Data>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        b: &CKKSCiphertext<B>,
        tsk: &GLWETensorKeyPrepared<T, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWECopy<BE> + GLWETensoring<BE> + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: poulpy_core::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        CKKSCiphertext<B>: poulpy_core::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        GLWETensorKeyPrepared<T, BE>: poulpy_core::layouts::prepared::GLWETensorKeyPreparedToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_mul_assign<Dst: Data, A: Data, T: Data>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        tsk: &GLWETensorKeyPrepared<T, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWECopy<BE> + GLWETensoring<BE> + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE> + poulpy_core::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        CKKSCiphertext<A>: poulpy_core::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        GLWETensorKeyPrepared<T, BE>: poulpy_core::layouts::prepared::GLWETensorKeyPreparedToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_square_into<Dst: Data, A: Data, T: Data>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        tsk: &GLWETensorKeyPrepared<T, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWECopy<BE> + GLWETensoring<BE> + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: poulpy_core::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        GLWETensorKeyPrepared<T, BE>: poulpy_core::layouts::prepared::GLWETensorKeyPreparedToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_square_assign<Dst: Data, T: Data>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<Dst>,
        tsk: &GLWETensorKeyPrepared<T, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWECopy<BE> + GLWETensoring<BE> + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE> + poulpy_core::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        GLWETensorKeyPrepared<T, BE>: poulpy_core::layouts::prepared::GLWETensorKeyPreparedToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_mul_pt_vec_znx_into<Dst: Data, A: Data, P: Data>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt_znx: &CKKSPlaintextVecZnx<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWECopy<BE>
            + GLWEMulPlain<BE>
            + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
            + poulpy_hal::api::VecZnxCopyBackend<BE>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: poulpy_core::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        CKKSPlaintextVecZnx<P>: poulpy_core::layouts::GLWEPlaintextToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_mul_pt_vec_znx_assign<Dst: Data, P: Data>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<Dst>,
        pt_znx: &CKKSPlaintextVecZnx<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWECopy<BE>
            + GLWEMulPlain<BE>
            + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
            + poulpy_hal::api::VecZnxCopyBackend<BE>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE> + poulpy_core::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        CKKSPlaintextVecZnx<P>: poulpy_core::layouts::GLWEPlaintextToBackendRef<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_mul_pt_vec_rnx_into<Dst: Data, A: Data, F>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWECopy<BE>
            + GLWEMulPlain<BE>
            + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
            + ModuleN
            + poulpy_hal::api::VecZnxCopyBackend<BE>,
        BE: HostBackend<OwnedBuf = Vec<u8>>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: poulpy_core::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_mul_pt_vec_rnx_assign<Dst: Data, F>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<Dst>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWECopy<BE>
            + GLWEMulPlain<BE>
            + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
            + ModuleN
            + poulpy_hal::api::VecZnxCopyBackend<BE>,
        BE: HostBackend<OwnedBuf = Vec<u8>>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE> + poulpy_core::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_mul_pt_const_znx_into<Dst: Data, A: Data>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEAdd<BE>
            + GLWECopy<BE>
            + GLWEMulConst<BE>
            + GLWERotate<BE>
            + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: poulpy_core::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_mul_pt_const_znx_assign<Dst: Data>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<Dst>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEAdd<BE>
            + GLWECopy<BE>
            + GLWEMulConst<BE>
            + GLWERotate<BE>
            + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE> + poulpy_core::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_mul_pt_const_rnx_into<Dst: Data, A: Data, F>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<Dst>,
        a: &CKKSCiphertext<A>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEAdd<BE>
            + GLWECopy<BE>
            + GLWEMulConst<BE>
            + GLWERotate<BE>
            + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        CKKSCiphertext<A>: poulpy_core::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion;

    fn ckks_mul_pt_const_rnx_assign<Dst: Data, F>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<Dst>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEAdd<BE>
            + GLWECopy<BE>
            + GLWEMulConst<BE>
            + GLWERotate<BE>
            + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<BE> + poulpy_core::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion;
}

#[macro_export]
macro_rules! impl_ckks_default_methods {
    ($backend:ty) => {
        $crate::impl_ckks_pt_znx_default_methods!($backend);
        $crate::impl_ckks_add_default_methods!($backend);
        $crate::impl_ckks_sub_default_methods!($backend);
        $crate::impl_ckks_neg_default_methods!($backend);
        $crate::impl_ckks_pow2_default_methods!($backend);
        $crate::impl_ckks_rotate_default_methods!($backend);
        $crate::impl_ckks_conjugate_default_methods!($backend);
        $crate::impl_ckks_mul_default_methods!($backend);
    };
}

pub use crate::impl_ckks_default_methods;
