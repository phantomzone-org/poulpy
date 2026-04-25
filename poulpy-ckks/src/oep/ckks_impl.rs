#![allow(clippy::too_many_arguments)]

use anyhow::Result;
use poulpy_core::{
    GLWEAdd, GLWEAutomorphism, GLWECopy, GLWEMulConst, GLWEMulPlain, GLWENegate, GLWERotate, GLWEShift, GLWESub, GLWETensoring,
    ScratchTakeCore,
    layouts::{
        GGLWEInfos, GGLWEPreparedToRef, GLWEAutomorphismKeyHelper, GLWEAutomorphismKeyPrepared, GLWEInfos, GLWEPlaintext,
        GLWETensorKeyPrepared, GetGaloisElement,
    },
};
use poulpy_hal::{
    api::{
        ModuleN, ScratchAvailable, VecZnxLsh, VecZnxLshTmpBytes, VecZnxRsh, VecZnxRshAddInto, VecZnxRshSub, VecZnxRshTmpBytes,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
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
    fn ckks_add_pt_znx_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: VecZnxRshTmpBytes;

    fn ckks_add_pt_vec_znx(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        src: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Scratch<BE>: ScratchTakeCore<BE>,
        Module<BE>: VecZnxRshAddInto<BE>;

    fn ckks_sub_pt_znx_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: VecZnxRshTmpBytes;

    fn ckks_sub_pt_vec_znx(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        src: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Scratch<BE>: ScratchTakeCore<BE>,
        Module<BE>: VecZnxRshSub<BE>;

    fn ckks_extract_pt_znx_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: VecZnxLshTmpBytes + VecZnxRshTmpBytes;

    fn ckks_extract_pt_znx<S: CKKSInfos>(
        module: &Module<BE>,
        dst: &mut CKKSPlaintextVecZnx<impl DataMut>,
        src: &GLWEPlaintext<impl DataRef>,
        src_meta: &S,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Scratch<BE>: ScratchTakeCore<BE>,
        Module<BE>: VecZnxLsh<BE> + VecZnxRsh<BE>;

    fn ckks_add_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: GLWEShift<BE>;

    fn ckks_add(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        b: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEAdd + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_add_inplace(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEAdd + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_add_pt_vec_znx_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: GLWEShift<BE> + VecZnxRshTmpBytes;

    fn ckks_add_pt_vec_znx_out(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: VecZnxRshAddInto<BE> + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_add_pt_vec_znx_inplace(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: VecZnxRshAddInto<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_add_pt_vec_rnx_tmp_bytes<R: GLWEInfos, A: GLWEInfos>(module: &Module<BE>, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        Module<BE>: ModuleN + GLWEShift<BE> + VecZnxRshTmpBytes;

    fn ckks_add_pt_vec_rnx<F>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: ModuleN + VecZnxRshAddInto<BE> + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion;

    fn ckks_add_pt_vec_rnx_inplace<F>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: ModuleN + VecZnxRshAddInto<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion;

    fn ckks_add_const_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: GLWEShift<BE>;

    fn ckks_add_pt_const_znx(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_add_pt_const_znx_inplace(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_add_pt_const_rnx<F>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion;

    fn ckks_add_pt_const_rnx_inplace<F>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion;

    fn ckks_sub_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: GLWEShift<BE> + VecZnxRshTmpBytes;

    fn ckks_sub_pt_vec_znx_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: GLWEShift<BE> + VecZnxRshTmpBytes;

    fn ckks_sub(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        b: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWESub + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_sub_inplace(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWESub + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_sub_pt_vec_znx_out(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: VecZnxRshSub<BE> + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_sub_pt_vec_znx_inplace(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: VecZnxRshSub<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_sub_pt_vec_rnx_tmp_bytes<R: GLWEInfos, A: GLWEInfos>(module: &Module<BE>, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        Module<BE>: ModuleN + GLWEShift<BE> + VecZnxRshTmpBytes;

    fn ckks_sub_pt_vec_rnx<F>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: ModuleN + VecZnxRshSub<BE> + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion;

    fn ckks_sub_pt_vec_rnx_inplace<F>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: ModuleN + VecZnxRshSub<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion;

    fn ckks_sub_const_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: GLWEShift<BE>;

    fn ckks_sub_pt_const_znx(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_sub_pt_const_znx_inplace(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_sub_pt_const_rnx<F>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion;

    fn ckks_sub_pt_const_rnx_inplace<F>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion;

    fn ckks_neg_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: GLWEShift<BE>;

    fn ckks_neg(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        src: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWENegate + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_neg_inplace(module: &Module<BE>, dst: &mut CKKSCiphertext<impl DataMut>)
    where
        Module<BE>: GLWENegate;

    fn ckks_mul_pow2_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: GLWEShift<BE>;

    fn ckks_mul_pow2(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        src: &CKKSCiphertext<impl DataRef>,
        bits: usize,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn ckks_mul_pow2_inplace(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        bits: usize,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn ckks_div_pow2_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: GLWEShift<BE>;

    fn ckks_div_pow2(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        src: &CKKSCiphertext<impl DataRef>,
        bits: usize,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEShift<BE> + GLWECopy,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn ckks_div_pow2_inplace(module: &Module<BE>, dst: &mut CKKSCiphertext<impl DataMut>, bits: usize) -> Result<()>;

    fn ckks_rotate_tmp_bytes<C: GLWEInfos, K: GGLWEInfos>(module: &Module<BE>, ct_infos: &C, key_infos: &K) -> usize
    where
        Module<BE>: GLWEAutomorphism<BE>;

    fn ckks_rotate<H, K>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        src: &CKKSCiphertext<impl DataRef>,
        k: i64,
        keys: &H,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEAutomorphism<BE> + GLWEShift<BE>,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn ckks_rotate_inplace<H, K>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        k: i64,
        keys: &H,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEAutomorphism<BE>,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn ckks_conjugate_tmp_bytes<C: GLWEInfos, K: GGLWEInfos>(module: &Module<BE>, ct_infos: &C, key_infos: &K) -> usize
    where
        Module<BE>: GLWEAutomorphism<BE>;

    fn ckks_conjugate(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        src: &CKKSCiphertext<impl DataRef>,
        key: &GLWEAutomorphismKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEAutomorphism<BE> + GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn ckks_conjugate_inplace(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        key: &GLWEAutomorphismKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) where
        Module<BE>: GLWEAutomorphism<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;

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

    fn ckks_mul_const_tmp_bytes<R: GLWEInfos, A: GLWEInfos>(module: &Module<BE>, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        Module<BE>: GLWEMulConst<BE> + GLWERotate<BE>;

    fn ckks_mul(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        b: &CKKSCiphertext<impl DataRef>,
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWETensoring<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_mul_inplace(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWETensoring<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_square(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWETensoring<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_square_inplace(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWETensoring<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_mul_pt_vec_znx(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEMulPlain<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_mul_pt_vec_znx_inplace(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEMulPlain<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_mul_pt_vec_rnx<F>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: ModuleN + GLWEMulPlain<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_mul_pt_vec_rnx_inplace<F>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: ModuleN + GLWEMulPlain<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_mul_pt_const_znx(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEAdd + GLWEMulConst<BE> + GLWERotate<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_mul_pt_const_znx_inplace(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEAdd + GLWEMulConst<BE> + GLWERotate<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_mul_pt_const_rnx<F>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEAdd + GLWEMulConst<BE> + GLWERotate<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion;

    fn ckks_mul_pt_const_rnx_inplace<F>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertext<impl DataMut>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWEAdd + GLWEMulConst<BE> + GLWERotate<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion;
}

#[macro_export]
macro_rules! impl_ckks_default_methods {
    ($backend:ty) => {
        fn ckks_add_pt_znx_tmp_bytes(module: &poulpy_hal::layouts::Module<$backend>) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_hal::api::VecZnxRshTmpBytes,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::pt_znx::CKKSPlaintextZnxDefault<$backend>>::ckks_add_pt_znx_tmp_bytes_default(module)
        }

        fn ckks_add_pt_vec_znx(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            src: &$crate::layouts::plaintext::CKKSPlaintextVecZnx<impl poulpy_hal::layouts::DataRef>,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Scratch<$backend>: poulpy_core::ScratchTakeCore<$backend>,
            poulpy_hal::layouts::Module<$backend>: poulpy_hal::api::VecZnxRshAddInto<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::pt_znx::CKKSPlaintextZnxDefault<$backend>>::ckks_add_pt_vec_znx_default(module, dst, src, scratch)
        }

        fn ckks_sub_pt_znx_tmp_bytes(module: &poulpy_hal::layouts::Module<$backend>) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_hal::api::VecZnxRshTmpBytes,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::pt_znx::CKKSPlaintextZnxDefault<$backend>>::ckks_sub_pt_znx_tmp_bytes_default(module)
        }

        fn ckks_sub_pt_vec_znx(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            src: &$crate::layouts::plaintext::CKKSPlaintextVecZnx<impl poulpy_hal::layouts::DataRef>,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Scratch<$backend>: poulpy_core::ScratchTakeCore<$backend>,
            poulpy_hal::layouts::Module<$backend>: poulpy_hal::api::VecZnxRshSub<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::pt_znx::CKKSPlaintextZnxDefault<$backend>>::ckks_sub_pt_vec_znx_default(module, dst, src, scratch)
        }

        fn ckks_extract_pt_znx_tmp_bytes(module: &poulpy_hal::layouts::Module<$backend>) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_hal::api::VecZnxLshTmpBytes + poulpy_hal::api::VecZnxRshTmpBytes,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::pt_znx::CKKSPlaintextZnxDefault<$backend>>::ckks_extract_pt_znx_tmp_bytes_default(module)
        }

        fn ckks_extract_pt_znx<S: $crate::CKKSInfos>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::plaintext::CKKSPlaintextVecZnx<impl poulpy_hal::layouts::DataMut>,
            src: &poulpy_core::layouts::GLWEPlaintext<impl poulpy_hal::layouts::DataRef>,
            src_meta: &S,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Scratch<$backend>: poulpy_core::ScratchTakeCore<$backend>,
            poulpy_hal::layouts::Module<$backend>: poulpy_hal::api::VecZnxLsh<$backend> + poulpy_hal::api::VecZnxRsh<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::pt_znx::CKKSPlaintextZnxDefault<$backend>>::ckks_extract_pt_znx_default(module, dst, src, src_meta, scratch)
        }

        fn ckks_add_tmp_bytes(module: &poulpy_hal::layouts::Module<$backend>) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::add::CKKSAddDefault<$backend>>::ckks_add_tmp_bytes_default(module)
        }

        fn ckks_add(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            a: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            b: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEAdd + poulpy_core::GLWEShift<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::add::CKKSAddDefault<$backend>>::ckks_add_default(module, dst, a, b, scratch)
        }

        fn ckks_add_inplace(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            a: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEAdd + poulpy_core::GLWEShift<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::add::CKKSAddDefault<$backend>>::ckks_add_inplace_default(module, dst, a, scratch)
        }

        fn ckks_add_pt_vec_znx_tmp_bytes(module: &poulpy_hal::layouts::Module<$backend>) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend> + poulpy_hal::api::VecZnxRshTmpBytes,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::add::CKKSAddDefault<$backend>>::ckks_add_pt_vec_znx_tmp_bytes_default(module)
        }

        fn ckks_add_pt_vec_znx_out(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            a: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            pt_znx: &$crate::layouts::plaintext::CKKSPlaintextVecZnx<impl poulpy_hal::layouts::DataRef>,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_hal::api::VecZnxRshAddInto<$backend> + poulpy_core::GLWEShift<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::add::CKKSAddDefault<$backend>>::ckks_add_pt_vec_znx_default(module, dst, a, pt_znx, scratch)
        }

        fn ckks_add_pt_vec_znx_inplace(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            pt_znx: &$crate::layouts::plaintext::CKKSPlaintextVecZnx<impl poulpy_hal::layouts::DataRef>,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_hal::api::VecZnxRshAddInto<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::add::CKKSAddDefault<$backend>>::ckks_add_pt_vec_znx_inplace_default(module, dst, pt_znx, scratch)
        }

        fn ckks_add_pt_vec_rnx_tmp_bytes<R: poulpy_core::layouts::GLWEInfos, A: poulpy_core::layouts::GLWEInfos>(
            module: &poulpy_hal::layouts::Module<$backend>,
            res: &R,
            a: &A,
            b: &$crate::CKKSMeta,
        ) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_hal::api::ModuleN + poulpy_core::GLWEShift<$backend> + poulpy_hal::api::VecZnxRshTmpBytes,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::add::CKKSAddDefault<$backend>>::ckks_add_pt_vec_rnx_tmp_bytes_default(module, res, a, b)
        }

        fn ckks_add_pt_vec_rnx<F>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            a: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            pt_rnx: &$crate::layouts::plaintext::CKKSPlaintextVecRnx<F>,
            prec: $crate::CKKSMeta,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_hal::api::ModuleN + poulpy_hal::api::VecZnxRshAddInto<$backend> + poulpy_core::GLWEShift<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
            $crate::layouts::plaintext::CKKSPlaintextVecRnx<F>: $crate::layouts::plaintext::CKKSPlaintextConversion,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::add::CKKSAddDefault<$backend>>::ckks_add_pt_vec_rnx_default(module, dst, a, pt_rnx, prec, scratch)
        }

        fn ckks_add_pt_vec_rnx_inplace<F>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            pt_rnx: &$crate::layouts::plaintext::CKKSPlaintextVecRnx<F>,
            prec: $crate::CKKSMeta,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_hal::api::ModuleN + poulpy_hal::api::VecZnxRshAddInto<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
            $crate::layouts::plaintext::CKKSPlaintextVecRnx<F>: $crate::layouts::plaintext::CKKSPlaintextConversion,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::add::CKKSAddDefault<$backend>>::ckks_add_pt_vec_rnx_inplace_default(module, dst, pt_rnx, prec, scratch)
        }

        fn ckks_add_const_tmp_bytes(module: &poulpy_hal::layouts::Module<$backend>) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::add::CKKSAddDefault<$backend>>::ckks_add_const_tmp_bytes_default(module)
        }

        fn ckks_add_pt_const_znx(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            a: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            cst_znx: &$crate::layouts::plaintext::CKKSPlaintextCstZnx,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::add::CKKSAddDefault<$backend>>::ckks_add_const_znx_default(module, dst, a, cst_znx, scratch)
        }

        fn ckks_add_pt_const_znx_inplace(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            cst_znx: &$crate::layouts::plaintext::CKKSPlaintextCstZnx,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::add::CKKSAddDefault<$backend>>::ckks_add_const_znx_inplace_default(module, dst, cst_znx, scratch)
        }

        fn ckks_add_pt_const_rnx<F>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            a: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            cst_rnx: &$crate::layouts::plaintext::CKKSPlaintextCstRnx<F>,
            prec: $crate::CKKSMeta,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
            $crate::layouts::plaintext::CKKSPlaintextCstRnx<F>: $crate::layouts::plaintext::CKKSConstPlaintextConversion,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::add::CKKSAddDefault<$backend>>::ckks_add_const_rnx_default(module, dst, a, cst_rnx, prec, scratch)
        }

        fn ckks_add_pt_const_rnx_inplace<F>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            cst_rnx: &$crate::layouts::plaintext::CKKSPlaintextCstRnx<F>,
            prec: $crate::CKKSMeta,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
            $crate::layouts::plaintext::CKKSPlaintextCstRnx<F>: $crate::layouts::plaintext::CKKSConstPlaintextConversion,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::add::CKKSAddDefault<$backend>>::ckks_add_const_rnx_inplace_default(module, dst, cst_rnx, prec, scratch)
        }

        fn ckks_sub_tmp_bytes(module: &poulpy_hal::layouts::Module<$backend>) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend> + poulpy_hal::api::VecZnxRshTmpBytes,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::sub::CKKSSubDefault<$backend>>::ckks_sub_tmp_bytes_default(module)
        }

        fn ckks_sub_pt_vec_znx_tmp_bytes(module: &poulpy_hal::layouts::Module<$backend>) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend> + poulpy_hal::api::VecZnxRshTmpBytes,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::sub::CKKSSubDefault<$backend>>::ckks_sub_pt_vec_znx_tmp_bytes_default(module)
        }

        fn ckks_sub(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            a: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            b: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWESub + poulpy_core::GLWEShift<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::sub::CKKSSubDefault<$backend>>::ckks_sub_default(module, dst, a, b, scratch)
        }

        fn ckks_sub_inplace(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            a: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWESub + poulpy_core::GLWEShift<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::sub::CKKSSubDefault<$backend>>::ckks_sub_inplace_default(module, dst, a, scratch)
        }

        fn ckks_sub_pt_vec_znx_out(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            a: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            pt_znx: &$crate::layouts::plaintext::CKKSPlaintextVecZnx<impl poulpy_hal::layouts::DataRef>,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_hal::api::VecZnxRshSub<$backend> + poulpy_core::GLWEShift<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::sub::CKKSSubDefault<$backend>>::ckks_sub_pt_vec_znx_default(module, dst, a, pt_znx, scratch)
        }

        fn ckks_sub_pt_vec_znx_inplace(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            pt_znx: &$crate::layouts::plaintext::CKKSPlaintextVecZnx<impl poulpy_hal::layouts::DataRef>,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_hal::api::VecZnxRshSub<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::sub::CKKSSubDefault<$backend>>::ckks_sub_pt_vec_znx_inplace_default(module, dst, pt_znx, scratch)
        }

        fn ckks_sub_pt_vec_rnx_tmp_bytes<R: poulpy_core::layouts::GLWEInfos, A: poulpy_core::layouts::GLWEInfos>(
            module: &poulpy_hal::layouts::Module<$backend>,
            res: &R,
            a: &A,
            b: &$crate::CKKSMeta,
        ) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_hal::api::ModuleN + poulpy_core::GLWEShift<$backend> + poulpy_hal::api::VecZnxRshTmpBytes,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::sub::CKKSSubDefault<$backend>>::ckks_sub_pt_vec_rnx_tmp_bytes_default(module, res, a, b)
        }

        fn ckks_sub_pt_vec_rnx<F>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            a: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            pt_rnx: &$crate::layouts::plaintext::CKKSPlaintextVecRnx<F>,
            prec: $crate::CKKSMeta,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_hal::api::ModuleN + poulpy_hal::api::VecZnxRshSub<$backend> + poulpy_core::GLWEShift<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
            $crate::layouts::plaintext::CKKSPlaintextVecRnx<F>: $crate::layouts::plaintext::CKKSPlaintextConversion,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::sub::CKKSSubDefault<$backend>>::ckks_sub_pt_vec_rnx_default(module, dst, a, pt_rnx, prec, scratch)
        }

        fn ckks_sub_pt_vec_rnx_inplace<F>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            pt_rnx: &$crate::layouts::plaintext::CKKSPlaintextVecRnx<F>,
            prec: $crate::CKKSMeta,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_hal::api::ModuleN + poulpy_hal::api::VecZnxRshSub<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
            $crate::layouts::plaintext::CKKSPlaintextVecRnx<F>: $crate::layouts::plaintext::CKKSPlaintextConversion,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::sub::CKKSSubDefault<$backend>>::ckks_sub_pt_vec_rnx_inplace_default(module, dst, pt_rnx, prec, scratch)
        }

        fn ckks_sub_const_tmp_bytes(module: &poulpy_hal::layouts::Module<$backend>) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::sub::CKKSSubDefault<$backend>>::ckks_sub_const_tmp_bytes_default(module)
        }

        fn ckks_sub_pt_const_znx(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            a: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            cst_znx: &$crate::layouts::plaintext::CKKSPlaintextCstZnx,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::sub::CKKSSubDefault<$backend>>::ckks_sub_const_znx_default(module, dst, a, cst_znx, scratch)
        }

        fn ckks_sub_pt_const_znx_inplace(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            cst_znx: &$crate::layouts::plaintext::CKKSPlaintextCstZnx,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::sub::CKKSSubDefault<$backend>>::ckks_sub_const_znx_inplace_default(module, dst, cst_znx, scratch)
        }

        fn ckks_sub_pt_const_rnx<F>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            a: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            cst_rnx: &$crate::layouts::plaintext::CKKSPlaintextCstRnx<F>,
            prec: $crate::CKKSMeta,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
            $crate::layouts::plaintext::CKKSPlaintextCstRnx<F>: $crate::layouts::plaintext::CKKSConstPlaintextConversion,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::sub::CKKSSubDefault<$backend>>::ckks_sub_const_rnx_default(module, dst, a, cst_rnx, prec, scratch)
        }

        fn ckks_sub_pt_const_rnx_inplace<F>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            cst_rnx: &$crate::layouts::plaintext::CKKSPlaintextCstRnx<F>,
            prec: $crate::CKKSMeta,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
            $crate::layouts::plaintext::CKKSPlaintextCstRnx<F>: $crate::layouts::plaintext::CKKSConstPlaintextConversion,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::sub::CKKSSubDefault<$backend>>::ckks_sub_const_rnx_inplace_default(module, dst, cst_rnx, prec, scratch)
        }

        fn ckks_neg_tmp_bytes(module: &poulpy_hal::layouts::Module<$backend>) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::neg::CKKSNegDefault<$backend>>::ckks_neg_tmp_bytes_default(module)
        }

        fn ckks_neg(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            src: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWENegate + poulpy_core::GLWEShift<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::neg::CKKSNegDefault<$backend>>::ckks_neg_default(module, dst, src, scratch)
        }

        fn ckks_neg_inplace(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
        ) where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWENegate,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::neg::CKKSNegDefault<$backend>>::ckks_neg_inplace_default(module, dst)
        }

        fn ckks_mul_pow2_tmp_bytes(module: &poulpy_hal::layouts::Module<$backend>) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::pow2::CKKSPow2Default<$backend>>::ckks_mul_pow2_tmp_bytes_default(module)
        }

        fn ckks_mul_pow2(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            src: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            bits: usize,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::pow2::CKKSPow2Default<$backend>>::ckks_mul_pow2_default(module, dst, src, bits, scratch)
        }

        fn ckks_mul_pow2_inplace(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            bits: usize,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::pow2::CKKSPow2Default<$backend>>::ckks_mul_pow2_inplace_default(module, dst, bits, scratch)
        }

        fn ckks_div_pow2_tmp_bytes(module: &poulpy_hal::layouts::Module<$backend>) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::pow2::CKKSPow2Default<$backend>>::ckks_div_pow2_tmp_bytes_default(module)
        }

        fn ckks_div_pow2(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            src: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            bits: usize,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend> + poulpy_core::GLWECopy,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::pow2::CKKSPow2Default<$backend>>::ckks_div_pow2_default(module, dst, src, bits, scratch)
        }

        fn ckks_div_pow2_inplace(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            bits: usize,
        ) -> anyhow::Result<()> {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::pow2::CKKSPow2Default<$backend>>::ckks_div_pow2_inplace_default(module, dst, bits)
        }

        fn ckks_rotate_tmp_bytes<C: poulpy_core::layouts::GLWEInfos, K: poulpy_core::layouts::GGLWEInfos>(
            module: &poulpy_hal::layouts::Module<$backend>,
            ct_infos: &C,
            key_infos: &K,
        ) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEAutomorphism<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::rotate::CKKSRotateDefault<$backend>>::ckks_rotate_tmp_bytes_default(module, ct_infos, key_infos)
        }

        fn ckks_rotate<H, K>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            src: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            k: i64,
            keys: &H,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEAutomorphism<$backend> + poulpy_core::GLWEShift<$backend>,
            K: poulpy_core::layouts::GGLWEPreparedToRef<$backend> + poulpy_core::layouts::GetGaloisElement + poulpy_core::layouts::GGLWEInfos,
            H: poulpy_core::layouts::GLWEAutomorphismKeyHelper<K, $backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::rotate::CKKSRotateDefault<$backend>>::ckks_rotate_default(module, dst, src, k, keys, scratch)
        }

        fn ckks_rotate_inplace<H, K>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            k: i64,
            keys: &H,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEAutomorphism<$backend>,
            K: poulpy_core::layouts::GGLWEPreparedToRef<$backend> + poulpy_core::layouts::GetGaloisElement + poulpy_core::layouts::GGLWEInfos,
            H: poulpy_core::layouts::GLWEAutomorphismKeyHelper<K, $backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::rotate::CKKSRotateDefault<$backend>>::ckks_rotate_inplace_default(module, dst, k, keys, scratch)
        }

        fn ckks_conjugate_tmp_bytes<C: poulpy_core::layouts::GLWEInfos, K: poulpy_core::layouts::GGLWEInfos>(
            module: &poulpy_hal::layouts::Module<$backend>,
            ct_infos: &C,
            key_infos: &K,
        ) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEAutomorphism<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::conjugate::CKKSConjugateDefault<$backend>>::ckks_conjugate_tmp_bytes_default(module, ct_infos, key_infos)
        }

        fn ckks_conjugate(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            src: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            key: &poulpy_core::layouts::GLWEAutomorphismKeyPrepared<impl poulpy_hal::layouts::DataRef, $backend>,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEAutomorphism<$backend> + poulpy_core::GLWEShift<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::conjugate::CKKSConjugateDefault<$backend>>::ckks_conjugate_default(module, dst, src, key, scratch)
        }

        fn ckks_conjugate_inplace(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            key: &poulpy_core::layouts::GLWEAutomorphismKeyPrepared<impl poulpy_hal::layouts::DataRef, $backend>,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEAutomorphism<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::conjugate::CKKSConjugateDefault<$backend>>::ckks_conjugate_inplace_default(module, dst, key, scratch)
        }

        fn ckks_mul_tmp_bytes<R: poulpy_core::layouts::GLWEInfos, T: poulpy_core::layouts::GGLWEInfos>(
            module: &poulpy_hal::layouts::Module<$backend>,
            res: &R,
            tsk: &T,
        ) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWETensoring<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::mul::CKKSMulDefault<$backend>>::ckks_mul_tmp_bytes_default(module, res, tsk)
        }

        fn ckks_square_tmp_bytes<R: poulpy_core::layouts::GLWEInfos, T: poulpy_core::layouts::GGLWEInfos>(
            module: &poulpy_hal::layouts::Module<$backend>,
            res: &R,
            tsk: &T,
        ) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWETensoring<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::mul::CKKSMulDefault<$backend>>::ckks_square_tmp_bytes_default(module, res, tsk)
        }

        fn ckks_mul_pt_vec_znx_tmp_bytes<R: poulpy_core::layouts::GLWEInfos, A: poulpy_core::layouts::GLWEInfos>(
            module: &poulpy_hal::layouts::Module<$backend>,
            res: &R,
            a: &A,
            b: &$crate::CKKSMeta,
        ) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEMulPlain<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::mul::CKKSMulDefault<$backend>>::ckks_mul_pt_vec_znx_tmp_bytes_default(module, res, a, b)
        }

        fn ckks_mul_pt_vec_rnx_tmp_bytes<R: poulpy_core::layouts::GLWEInfos, A: poulpy_core::layouts::GLWEInfos>(
            module: &poulpy_hal::layouts::Module<$backend>,
            res: &R,
            a: &A,
            b: &$crate::CKKSMeta,
        ) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_hal::api::ModuleN + poulpy_core::GLWEMulPlain<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::mul::CKKSMulDefault<$backend>>::ckks_mul_pt_vec_rnx_tmp_bytes_default(module, res, a, b)
        }

        fn ckks_mul_const_tmp_bytes<R: poulpy_core::layouts::GLWEInfos, A: poulpy_core::layouts::GLWEInfos>(
            module: &poulpy_hal::layouts::Module<$backend>,
            res: &R,
            a: &A,
            b: &$crate::CKKSMeta,
        ) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEMulConst<$backend> + poulpy_core::GLWERotate<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::mul::CKKSMulDefault<$backend>>::ckks_mul_const_tmp_bytes_default(module, res, a, b)
        }

        fn ckks_mul(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            a: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            b: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            tsk: &poulpy_core::layouts::GLWETensorKeyPrepared<impl poulpy_hal::layouts::DataRef, $backend>,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWETensoring<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::mul::CKKSMulDefault<$backend>>::ckks_mul_default(module, dst, a, b, tsk, scratch)
        }

        fn ckks_mul_inplace(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            a: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            tsk: &poulpy_core::layouts::GLWETensorKeyPrepared<impl poulpy_hal::layouts::DataRef, $backend>,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWETensoring<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::mul::CKKSMulDefault<$backend>>::ckks_mul_inplace_default(module, dst, a, tsk, scratch)
        }

        fn ckks_square(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            a: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            tsk: &poulpy_core::layouts::GLWETensorKeyPrepared<impl poulpy_hal::layouts::DataRef, $backend>,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWETensoring<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::mul::CKKSMulDefault<$backend>>::ckks_square_default(module, dst, a, tsk, scratch)
        }

        fn ckks_square_inplace(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            tsk: &poulpy_core::layouts::GLWETensorKeyPrepared<impl poulpy_hal::layouts::DataRef, $backend>,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWETensoring<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::mul::CKKSMulDefault<$backend>>::ckks_square_inplace_default(module, dst, tsk, scratch)
        }

        fn ckks_mul_pt_vec_znx(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            a: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            pt_znx: &$crate::layouts::plaintext::CKKSPlaintextVecZnx<impl poulpy_hal::layouts::DataRef>,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEMulPlain<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::mul::CKKSMulDefault<$backend>>::ckks_mul_pt_vec_znx_default(module, dst, a, pt_znx, scratch)
        }

        fn ckks_mul_pt_vec_znx_inplace(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            pt_znx: &$crate::layouts::plaintext::CKKSPlaintextVecZnx<impl poulpy_hal::layouts::DataRef>,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEMulPlain<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::mul::CKKSMulDefault<$backend>>::ckks_mul_pt_vec_znx_inplace_default(module, dst, pt_znx, scratch)
        }

        fn ckks_mul_pt_vec_rnx<F>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            a: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            pt_rnx: &$crate::layouts::plaintext::CKKSPlaintextVecRnx<F>,
            prec: $crate::CKKSMeta,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_hal::api::ModuleN + poulpy_core::GLWEMulPlain<$backend>,
            $crate::layouts::plaintext::CKKSPlaintextVecRnx<F>: $crate::layouts::plaintext::CKKSPlaintextConversion,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::mul::CKKSMulDefault<$backend>>::ckks_mul_pt_vec_rnx_default(module, dst, a, pt_rnx, prec, scratch)
        }

        fn ckks_mul_pt_vec_rnx_inplace<F>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            pt_rnx: &$crate::layouts::plaintext::CKKSPlaintextVecRnx<F>,
            prec: $crate::CKKSMeta,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_hal::api::ModuleN + poulpy_core::GLWEMulPlain<$backend>,
            $crate::layouts::plaintext::CKKSPlaintextVecRnx<F>: $crate::layouts::plaintext::CKKSPlaintextConversion,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::mul::CKKSMulDefault<$backend>>::ckks_mul_pt_vec_rnx_inplace_default(module, dst, pt_rnx, prec, scratch)
        }

        fn ckks_mul_pt_const_znx(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            a: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            cst_znx: &$crate::layouts::plaintext::CKKSPlaintextCstZnx,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEAdd + poulpy_core::GLWEMulConst<$backend> + poulpy_core::GLWERotate<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::mul::CKKSMulDefault<$backend>>::ckks_mul_const_znx_default(module, dst, a, cst_znx, scratch)
        }

        fn ckks_mul_pt_const_znx_inplace(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            cst_znx: &$crate::layouts::plaintext::CKKSPlaintextCstZnx,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEAdd + poulpy_core::GLWEMulConst<$backend> + poulpy_core::GLWERotate<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::mul::CKKSMulDefault<$backend>>::ckks_mul_const_znx_inplace_default(module, dst, cst_znx, scratch)
        }

        fn ckks_mul_pt_const_rnx<F>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            a: &$crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataRef>,
            cst_rnx: &$crate::layouts::plaintext::CKKSPlaintextCstRnx<F>,
            prec: $crate::CKKSMeta,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEAdd + poulpy_core::GLWEMulConst<$backend> + poulpy_core::GLWERotate<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
            $crate::layouts::plaintext::CKKSPlaintextCstRnx<F>: $crate::layouts::plaintext::CKKSConstPlaintextConversion,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::mul::CKKSMulDefault<$backend>>::ckks_mul_const_rnx_default(module, dst, a, cst_rnx, prec, scratch)
        }

        fn ckks_mul_pt_const_rnx_inplace<F>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<impl poulpy_hal::layouts::DataMut>,
            cst_rnx: &$crate::layouts::plaintext::CKKSPlaintextCstRnx<F>,
            prec: $crate::CKKSMeta,
            scratch: &mut poulpy_hal::layouts::Scratch<$backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEAdd + poulpy_core::GLWEMulConst<$backend> + poulpy_core::GLWERotate<$backend>,
            poulpy_hal::layouts::Scratch<$backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchTakeCore<$backend>,
            $crate::layouts::plaintext::CKKSPlaintextCstRnx<F>: $crate::layouts::plaintext::CKKSConstPlaintextConversion,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::operations::mul::CKKSMulDefault<$backend>>::ckks_mul_const_rnx_inplace_default(module, dst, cst_rnx, prec, scratch)
        }
    };
}
