#![allow(clippy::too_many_arguments)]

use anyhow::Result;
use poulpy_core::layouts::{GGLWEInfos, GLWEInfos, GLWESecretPreparedToBackendRef, GLWETensorKeyPreparedBackendRef, LWEInfos};
use poulpy_core::{
    EncryptionInfos, GLWEAdd, GLWECopy, GLWEDecrypt, GLWEMulConst, GLWEMulPlain, GLWENegate, GLWENormalize, GLWERotate, GLWESub,
    GLWETensoring,
};
use poulpy_core::{
    GLWEAutomorphism, GLWEEncryptSk, GLWEShift, ScratchArenaTakeCore,
    layouts::{GGLWEPreparedToBackendRef, GetGaloisElement},
};
use poulpy_hal::api::{
    VecZnxAddConstAssignBackend, VecZnxCopyBackend, VecZnxLshBackend, VecZnxLshTmpBytes, VecZnxRshBackend, VecZnxRshCoeffBackend,
    VecZnxRshSubBackend, VecZnxRshSubCoeffIntoBackend, VecZnxSubAssignBackend,
};
use poulpy_hal::{
    api::{ScratchAvailable, VecZnxRshAddCoeffIntoBackend, VecZnxRshAddIntoBackend, VecZnxRshTmpBytes},
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::{
    CKKSCiphertextMut, CKKSCiphertextRef, CKKSInfos, CKKSMeta, CKKSPlaintexToBackendRef, GLWEToBackendMut, GLWEToBackendRef,
    SetCKKSInfos, leveled::default::pt_znx::CKKSPlaintextDefault,
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

    fn ckks_extract_pt_znx<Dst, Src>(
        module: &Module<BE>,
        dst: &mut Dst,
        src: &Src,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        Module<BE>: VecZnxLshBackend<BE> + VecZnxRshBackend<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;

    fn ckks_add_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: GLWEShift<BE> + GLWENormalize<BE>;

    fn ckks_add_into<Dst, A, B>(
        module: &Module<BE>,
        dst: &mut Dst,
        a: &A,
        b: &B,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        B: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_add_assign<Dst, A>(module: &Module<BE>, dst: &mut Dst, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos,
        Module<BE>: GLWEAdd<BE> + GLWEShift<BE> + GLWENormalize<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_add_pt_vec_znx_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: GLWEShift<BE> + GLWENormalize<BE> + VecZnxRshTmpBytes;

    fn ckks_add_pt_vec_znx_into<Dst, A, P>(
        module: &Module<BE>,
        dst: &mut Dst,
        a: &A,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        Module<BE>: VecZnxRshAddIntoBackend<BE> + GLWEShift<BE> + GLWENormalize<BE> + CKKSPlaintextDefault<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_add_pt_vec_znx_assign<Dst, P>(
        module: &Module<BE>,
        dst: &mut Dst,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        Module<BE>: VecZnxRshAddIntoBackend<BE> + GLWENormalize<BE> + CKKSPlaintextDefault<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_add_pt_const_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: GLWEShift<BE> + GLWENormalize<BE> + VecZnxRshTmpBytes;

    fn ckks_add_pt_const_znx_into<Dst, A, P>(
        module: &Module<BE>,
        dst: &mut Dst,
        a: &A,
        dst_coeff: usize,
        pt_znx: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        Module<BE>: GLWEShift<BE> + GLWENormalize<BE> + VecZnxRshAddCoeffIntoBackend<BE> + CKKSPlaintextDefault<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_add_pt_const_znx_assign<Dst, P>(
        module: &Module<BE>,
        dst: &mut Dst,
        dst_coeff: usize,
        pt_znx: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        Module<BE>: GLWENormalize<BE> + VecZnxRshAddCoeffIntoBackend<BE> + CKKSPlaintextDefault<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_encrypt_sk_tmp_bytes<A>(module: &Module<BE>, ct_infos: &A) -> usize
    where
        A: GLWEInfos + CKKSInfos,
        Module<BE>: GLWEEncryptSk<BE> + VecZnxRshAddIntoBackend<BE> + VecZnxRshTmpBytes;

    #[allow(clippy::too_many_arguments)]
    fn ckks_encrypt_sk<'s, Ct, S, E, Pt>(
        module: &Module<BE>,
        ct: &mut Ct,
        pt: &Pt,
        sk: &S,
        enc_infos: &E,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) -> Result<()>
    where
        Pt: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        Ct: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        E: EncryptionInfos,
        Module<BE>: GLWEEncryptSk<BE> + VecZnxRshAddIntoBackend<BE> + CKKSPlaintextDefault<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_decrypt_tmp_bytes<A>(module: &Module<BE>, ct_infos: &A) -> usize
    where
        A: GLWEInfos + CKKSInfos,
        Module<BE>: GLWEDecrypt<BE> + VecZnxLshBackend<BE> + VecZnxLshTmpBytes + VecZnxRshBackend<BE> + VecZnxRshTmpBytes;

    fn ckks_decrypt<S, C, Pt>(module: &Module<BE>, pt: &mut Pt, ct: &C, sk: &S, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Pt: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos + LWEInfos + CKKSInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        Module<BE>: GLWEDecrypt<BE>
            + VecZnxLshBackend<BE>
            + VecZnxLshTmpBytes
            + VecZnxRshBackend<BE>
            + VecZnxRshTmpBytes
            + CKKSPlaintextDefault<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;

    fn ckks_sub_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: GLWEShift<BE> + GLWENormalize<BE> + VecZnxRshTmpBytes;

    fn ckks_sub_pt_vec_znx_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: GLWEShift<BE> + GLWENormalize<BE> + VecZnxRshTmpBytes;

    fn ckks_sub_into<Dst, A, B>(
        module: &Module<BE>,
        dst: &mut Dst,
        a: &A,
        b: &B,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        B: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        Module<BE>: GLWESub<BE> + GLWEShift<BE> + GLWENormalize<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_sub_assign<Dst, A>(module: &Module<BE>, dst: &mut Dst, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos,
        Module<BE>: GLWESub<BE> + GLWEShift<BE> + GLWENormalize<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_sub_pt_vec_znx_into<Dst, A, P>(
        module: &Module<BE>,
        dst: &mut Dst,
        a: &A,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        Module<BE>: VecZnxRshSubBackend<BE> + GLWEShift<BE> + GLWENormalize<BE> + CKKSPlaintextDefault<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_sub_pt_vec_znx_assign<Dst, P>(
        module: &Module<BE>,
        dst: &mut Dst,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        Module<BE>: VecZnxRshSubBackend<BE> + GLWENormalize<BE> + CKKSPlaintextDefault<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_sub_pt_const_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: GLWEShift<BE> + GLWENormalize<BE> + VecZnxRshTmpBytes;

    fn ckks_sub_pt_const_znx_into<Dst, A, P>(
        module: &Module<BE>,
        dst: &mut Dst,
        a: &A,
        dst_coeff: usize,
        pt_znx: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        Module<BE>: GLWEShift<BE> + GLWENormalize<BE> + VecZnxRshSubCoeffIntoBackend<BE> + CKKSPlaintextDefault<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_sub_pt_const_znx_assign<Dst, P>(
        module: &Module<BE>,
        dst: &mut Dst,
        dst_coeff: usize,
        pt_znx: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        Module<BE>: VecZnxRshSubCoeffIntoBackend<BE> + GLWENormalize<BE> + CKKSPlaintextDefault<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_neg_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: GLWEShift<BE>;

    fn ckks_neg_into<Dst, Src>(module: &Module<BE>, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        Module<BE>: GLWENegate<BE> + GLWEShift<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_neg_assign<Dst>(module: &Module<BE>, dst: &mut Dst) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Module<BE>: GLWENegate<BE>;

    fn ckks_mul_pow2_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: GLWEShift<BE>;

    fn ckks_mul_pow2_into<Dst, Src>(
        module: &Module<BE>,
        dst: &mut Dst,
        src: &Src,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        Module<BE>: GLWEShift<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;

    fn ckks_mul_pow2_assign<Dst>(
        module: &Module<BE>,
        dst: &mut Dst,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Module<BE>: GLWEShift<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;

    fn ckks_div_pow2_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: GLWEShift<BE>;

    fn ckks_div_pow2_into<Dst, Src>(
        module: &Module<BE>,
        dst: &mut Dst,
        src: &Src,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        Module<BE>: GLWEShift<BE> + GLWECopy<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;

    fn ckks_div_pow2_assign<Dst>(module: &Module<BE>, dst: &mut Dst, bits: usize) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos;

    fn ckks_rescale_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: GLWEShift<BE>;

    fn ckks_rescale_assign<Dst>(module: &Module<BE>, ct: &mut Dst, k: usize, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Module<BE>: GLWEShift<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;

    fn ckks_rescale_into<Dst, Src>(
        module: &Module<BE>,
        dst: &mut Dst,
        k: usize,
        src: &Src,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        Module<BE>: GLWEShift<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;

    fn ckks_align_assign<A, B>(module: &Module<BE>, a: &mut A, b: &mut B, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        A: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        B: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Module<BE>: GLWEShift<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;

    fn ckks_align_tmp_bytes(module: &Module<BE>) -> usize
    where
        Module<BE>: GLWEShift<BE>;

    fn ckks_rotate_tmp_bytes<C: GLWEInfos, K: GGLWEInfos>(module: &Module<BE>, ct_infos: &C, key_infos: &K) -> usize
    where
        Module<BE>: GLWEAutomorphism<BE>;

    fn ckks_rotate_into<Dst, Src, K>(
        module: &Module<BE>,
        dst: &mut Dst,
        src: &Src,
        key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        Module<BE>: GLWEAutomorphism<BE> + GLWEShift<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;

    fn ckks_rotate_assign<Dst, K>(module: &Module<BE>, dst: &mut Dst, key: &K, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        Module<BE>: GLWEAutomorphism<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;

    fn ckks_conjugate_tmp_bytes<C: GLWEInfos, K: GGLWEInfos>(module: &Module<BE>, ct_infos: &C, key_infos: &K) -> usize
    where
        Module<BE>: GLWEAutomorphism<BE>;

    fn ckks_conjugate_into<Dst, Src, K>(
        module: &Module<BE>,
        dst: &mut Dst,
        src: &Src,
        key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        Module<BE>: GLWEAutomorphism<BE> + GLWEShift<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;

    fn ckks_conjugate_assign<Dst, K>(
        module: &Module<BE>,
        dst: &mut Dst,
        key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + SetCKKSInfos,
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        Module<BE>: GLWEAutomorphism<BE>,
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

    fn ckks_mul_pt_const_tmp_bytes<R: GLWEInfos, A: GLWEInfos>(module: &Module<BE>, res: &R, a: &A, b: &CKKSMeta) -> usize
    where
        Module<BE>: GLWEMulConst<BE> + GLWERotate<BE>;

    fn ckks_mul_into<Dst, A, B, T>(
        module: &Module<BE>,
        dst: &mut Dst,
        a: &A,
        b: &B,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        B: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        T: poulpy_core::layouts::prepared::GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
        Module<BE>: GLWECopy<BE> + GLWETensoring<BE> + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_mul_assign<Dst, A, T>(
        module: &Module<BE>,
        dst: &mut Dst,
        a: &A,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        T: poulpy_core::layouts::prepared::GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
        Module<BE>: GLWECopy<BE> + GLWETensoring<BE> + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_square_into<Dst, A, T>(
        module: &Module<BE>,
        dst: &mut Dst,
        a: &A,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        T: poulpy_core::layouts::prepared::GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
        Module<BE>: GLWECopy<BE> + GLWETensoring<BE> + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_square_assign<Dst, T>(module: &Module<BE>, dst: &mut Dst, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        T: poulpy_core::layouts::prepared::GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
        Module<BE>: GLWECopy<BE> + GLWETensoring<BE> + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_mul_pt_vec_znx_into<Dst, A, P>(
        module: &Module<BE>,
        dst: &mut Dst,
        a: &A,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos + CKKSInfos,
        Module<BE>: GLWECopy<BE>
            + GLWEMulPlain<BE>
            + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
            + VecZnxCopyBackend<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_mul_pt_vec_znx_assign<Dst, P>(
        module: &Module<BE>,
        dst: &mut Dst,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos + CKKSInfos,
        Module<BE>: GLWECopy<BE>
            + GLWEMulPlain<BE>
            + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
            + VecZnxCopyBackend<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_mul_pt_const_znx_into<Dst, A, P>(
        module: &Module<BE>,
        dst: &mut Dst,
        a: &A,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos + CKKSInfos,
        Module<BE>: GLWEAdd<BE>
            + GLWECopy<BE>
            + GLWEMulConst<BE>
            + GLWERotate<BE>
            + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_mul_pt_const_znx_assign<Dst, P>(
        module: &Module<BE>,
        dst: &mut Dst,
        pt_znx: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos + CKKSInfos,
        Module<BE>: GLWEAdd<BE>
            + GLWECopy<BE>
            + GLWEMulConst<BE>
            + GLWERotate<BE>
            + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;
}

#[macro_export]
macro_rules! impl_ckks_default_methods {
    ($backend:ty) => {
        $crate::impl_ckks_pt_znx_default_methods!($backend);
        $crate::impl_ckks_add_default_methods!($backend);
        $crate::impl_ckks_encryption_default_methods!($backend);
        $crate::impl_ckks_sub_default_methods!($backend);
        $crate::impl_ckks_neg_default_methods!($backend);
        $crate::impl_ckks_pow2_default_methods!($backend);
        $crate::impl_ckks_rescale_default_methods!($backend);
        $crate::impl_ckks_rotate_default_methods!($backend);
        $crate::impl_ckks_conjugate_default_methods!($backend);
        $crate::impl_ckks_mul_default_methods!($backend);
    };
}

pub use crate::impl_ckks_default_methods;
