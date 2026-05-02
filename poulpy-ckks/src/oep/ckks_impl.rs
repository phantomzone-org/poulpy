#![allow(clippy::too_many_arguments)]

use anyhow::Result;
use poulpy_core::layouts::{
    GGLWEInfos, GLWEInfos, GLWEPlaintext, GLWEPlaintextToBackendRef, LWEInfos,
    prepared::{GLWEAutomorphismKeyPreparedBackendRef, GLWETensorKeyPreparedBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, Data, Module, ScratchArena},
    source::Source,
};

use crate::{CKKSPlaintexToBackendRef, CKKSPlaintextVecZnxToBackendMut};

use crate::{CKKSCiphertextMut, CKKSCiphertextRef, CKKSInfos, CKKSMeta, SetCKKSInfos};

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
    fn ckks_extract_pt_znx_tmp_bytes(module: &Module<BE>) -> usize;

    fn ckks_extract_pt_znx<Dst, Src: Data, S: CKKSInfos>(
        module: &Module<BE>,
        dst: &mut Dst,
        src: &GLWEPlaintext<Src>,
        src_meta: &S,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: CKKSPlaintextVecZnxToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        GLWEPlaintext<Src>: GLWEPlaintextToBackendRef<BE>;

    fn ckks_add_tmp_bytes(module: &Module<BE>) -> usize;

    fn ckks_add_into(
        module: &Module<BE>,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        a: &CKKSCiphertextRef<'_, BE>,
        b: &CKKSCiphertextRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>;

    fn ckks_add_assign(
        module: &Module<BE>,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        a: &CKKSCiphertextRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>;

    fn ckks_add_pt_vec_znx_tmp_bytes(module: &Module<BE>) -> usize;

    fn ckks_add_pt_vec_znx_into<P>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        a: &CKKSCiphertextRef<'_, BE>,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos;

    fn ckks_add_pt_vec_znx_assign<P>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos;

    fn ckks_add_pt_const_tmp_bytes(module: &Module<BE>) -> usize;

    fn ckks_add_pt_const_znx_into<P>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        a: &CKKSCiphertextRef<'_, BE>,
        dst_coeff: usize,
        pt_znx: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos;

    fn ckks_add_pt_const_znx_assign<P>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        dst_coeff: usize,
        pt_znx: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos;

    fn ckks_encrypt_sk_tmp_bytes<A>(module: &Module<BE>, ct_infos: &A) -> usize
    where
        A: GLWEInfos + CKKSInfos;

    #[allow(clippy::too_many_arguments)]
    fn ckks_encrypt_sk<'s, S, E, Pt>(
        module: &Module<BE>,
        ct: &mut CKKSCiphertextMut<'_, BE>,
        pt: &Pt,
        sk: &S,
        enc_infos: &E,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) -> Result<()>
    where
        Pt: CKKSPlaintexToBackendRef<BE> + CKKSInfos,
        S: poulpy_core::layouts::GLWESecretPreparedToBackendRef<BE>,
        E: poulpy_core::EncryptionInfos;

    fn ckks_decrypt_tmp_bytes<A>(module: &Module<BE>, ct_infos: &A) -> usize
    where
        A: GLWEInfos + CKKSInfos;

    fn ckks_decrypt<S, Pt>(
        module: &Module<BE>,
        pt: &mut Pt,
        ct: &CKKSCiphertextRef<'_, BE>,
        sk: &S,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Pt: CKKSPlaintextVecZnxToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        S: poulpy_core::layouts::GLWESecretPreparedToBackendRef<BE> + GLWEInfos;

    fn ckks_sub_tmp_bytes(module: &Module<BE>) -> usize;

    fn ckks_sub_pt_vec_znx_tmp_bytes(module: &Module<BE>) -> usize;

    fn ckks_sub_into(
        module: &Module<BE>,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        a: &CKKSCiphertextRef<'_, BE>,
        b: &CKKSCiphertextRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>;

    fn ckks_sub_assign(
        module: &Module<BE>,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        a: &CKKSCiphertextRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>;

    fn ckks_sub_pt_vec_znx_into<P>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        a: &CKKSCiphertextRef<'_, BE>,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos;

    fn ckks_sub_pt_vec_znx_assign<P>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos;

    fn ckks_sub_pt_const_tmp_bytes(module: &Module<BE>) -> usize;

    fn ckks_sub_pt_const_znx_into<P>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        a: &CKKSCiphertextRef<'_, BE>,
        dst_coeff: usize,
        pt_znx: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos;

    fn ckks_sub_pt_const_znx_assign<P>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        dst_coeff: usize,
        pt_znx: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos;

    fn ckks_neg_tmp_bytes(module: &Module<BE>) -> usize;

    fn ckks_neg_into(
        module: &Module<BE>,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        src: &CKKSCiphertextRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>;

    fn ckks_neg_assign(module: &Module<BE>, dst: &mut CKKSCiphertextMut<'_, BE>) -> Result<()>;

    fn ckks_mul_pow2_tmp_bytes(module: &Module<BE>) -> usize;

    fn ckks_mul_pow2_into(
        module: &Module<BE>,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        src: &CKKSCiphertextRef<'_, BE>,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>;

    fn ckks_mul_pow2_assign(
        module: &Module<BE>,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>;

    fn ckks_div_pow2_tmp_bytes(module: &Module<BE>) -> usize;

    fn ckks_div_pow2_into(
        module: &Module<BE>,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        src: &CKKSCiphertextRef<'_, BE>,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>;

    fn ckks_div_pow2_assign(module: &Module<BE>, dst: &mut CKKSCiphertextMut<'_, BE>, bits: usize) -> Result<()>;

    fn ckks_rescale_tmp_bytes(module: &Module<BE>) -> usize;

    fn ckks_rescale_assign(
        module: &Module<BE>,
        ct: &mut CKKSCiphertextMut<'_, BE>,
        k: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>;

    fn ckks_rescale_into(
        module: &Module<BE>,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        k: usize,
        src: &CKKSCiphertextRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>;

    fn ckks_align_assign(
        module: &Module<BE>,
        a: &mut CKKSCiphertextMut<'_, BE>,
        b: &mut CKKSCiphertextMut<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>;

    fn ckks_align_tmp_bytes(module: &Module<BE>) -> usize;

    fn ckks_rotate_tmp_bytes<C: GLWEInfos, K: GGLWEInfos>(module: &Module<BE>, ct_infos: &C, key_infos: &K) -> usize;

    fn ckks_rotate_into(
        module: &Module<BE>,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        src: &CKKSCiphertextRef<'_, BE>,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>;

    fn ckks_rotate_assign(
        module: &Module<BE>,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>;

    fn ckks_conjugate_tmp_bytes<C: GLWEInfos, K: GGLWEInfos>(module: &Module<BE>, ct_infos: &C, key_infos: &K) -> usize;

    fn ckks_conjugate_into(
        module: &Module<BE>,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        src: &CKKSCiphertextRef<'_, BE>,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>;

    fn ckks_conjugate_assign(
        module: &Module<BE>,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>;

    fn ckks_mul_tmp_bytes<R: GLWEInfos, T: GGLWEInfos>(module: &Module<BE>, res: &R, tsk: &T) -> usize;
    fn ckks_square_tmp_bytes<R: GLWEInfos, T: GGLWEInfos>(module: &Module<BE>, res: &R, tsk: &T) -> usize;
    fn ckks_mul_pt_vec_znx_tmp_bytes<R: GLWEInfos, A: GLWEInfos>(module: &Module<BE>, res: &R, a: &A, b: &CKKSMeta) -> usize;
    fn ckks_mul_pt_const_tmp_bytes<R: GLWEInfos, A: GLWEInfos>(module: &Module<BE>, res: &R, a: &A, b: &CKKSMeta) -> usize;

    fn ckks_mul_into(
        module: &Module<BE>,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        a: &CKKSCiphertextRef<'_, BE>,
        b: &CKKSCiphertextRef<'_, BE>,
        tsk: &GLWETensorKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>;

    fn ckks_mul_assign(
        module: &Module<BE>,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        a: &CKKSCiphertextRef<'_, BE>,
        tsk: &GLWETensorKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>;

    fn ckks_square_into(
        module: &Module<BE>,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        a: &CKKSCiphertextRef<'_, BE>,
        tsk: &GLWETensorKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>;

    fn ckks_square_assign(
        module: &Module<BE>,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        tsk: &GLWETensorKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>;

    fn ckks_mul_pt_vec_znx_into<P>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        a: &CKKSCiphertextRef<'_, BE>,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos;

    fn ckks_mul_pt_vec_znx_assign<P>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos;

    fn ckks_mul_pt_const_znx_into<P>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        a: &CKKSCiphertextRef<'_, BE>,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos;

    fn ckks_mul_pt_const_znx_assign<P>(
        module: &Module<BE>,
        dst: &mut CKKSCiphertextMut<'_, BE>,
        pt_znx: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: CKKSPlaintexToBackendRef<BE> + CKKSInfos;
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
