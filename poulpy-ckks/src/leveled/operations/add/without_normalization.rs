//! Unnormalized variants of CKKS addition.
//!
//! These methods skip the trailing `glwe_normalize_inplace` that the safe
//! [`CKKSAddOps`] variants perform after each operation. They exist because
//! the bivariate representation is linear over `Z`: additions and
//! subtractions compose termwise, and carry propagation can be deferred and
//! paid for only once, just before an operation that actually requires
//! K-normalized input.
//!
//! # Safety
//!
//! Not a memory-safety hazard; `unsafe` flags a correctness invariant the
//! compiler cannot check. Misuse yields silently-wrong plaintexts on
//! decryption, never UB.
//!
//! A ciphertext produced by any `*_without_normalization` call is still a
//! valid bivariate representation of the same plaintext, but its limbs are
//! no longer K-normalized: coefficients may lie outside
//! `[-2^(base2k-1), 2^(base2k-1))` and keep growing with each subsequent
//! raw add/sub. Starting from normalized operands, §3.3 of [eprint 2023/771]
//! shows that on the order of `ℓ·N·2^K` raw linear ops (well over `10^5`
//! in practice) can be chained before limbs risk overflowing their `i64`
//! storage. The caller must apply a normalizing operation at *some point*
//! in the chain — either `glwe_normalize_inplace`, a subsequent normalized
//! CKKS op, or any op that internally normalizes (external product,
//! relinearization, keyswitch, automorphism, …) — before limbs overflow
//! and before handing the ciphertext to code that assumes a K-normalized
//! layout (e.g. serialization, noise analysis, hot-path operations that
//! rely on bounded coefficients).
//!
//! [eprint 2023/771]: https://eprint.iacr.org/2023/771

use anyhow::Result;
use poulpy_core::{GLWEAdd, GLWEShift, ScratchTakeCore};
use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, VecZnxRshAddInto},
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
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
};

use super::default_impl::CKKSAddDefault;

/// Opt-in API exposing unnormalized CKKS addition variants.
///
/// # Safety
/// Not a memory-safety hazard; `unsafe` flags a correctness invariant. Every
/// method may leave the destination with non-K-normalized limbs — a
/// normalizing op must eventually run before limbs overflow `i64` or the
/// ciphertext is consumed by code requiring K-normalized inputs. Misuse
/// yields silently-wrong plaintexts on decryption, never UB.
#[allow(clippy::missing_safety_doc)]
pub unsafe trait CKKSAddOpsWithoutNormalization<BE: Backend> {
    unsafe fn ckks_add_without_normalization(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        b: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    unsafe fn ckks_add_inplace_without_normalization(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    unsafe fn ckks_add_pt_vec_znx_without_normalization(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshAddInto<BE> + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    unsafe fn ckks_add_pt_vec_znx_inplace_without_normalization(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshAddInto<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    unsafe fn ckks_add_pt_vec_rnx_without_normalization<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + VecZnxRshAddInto<BE> + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion;

    unsafe fn ckks_add_pt_vec_rnx_inplace_without_normalization<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + VecZnxRshAddInto<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion;

    unsafe fn ckks_add_pt_const_znx_without_normalization(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    unsafe fn ckks_add_pt_const_znx_inplace_without_normalization(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    unsafe fn ckks_add_pt_const_rnx_without_normalization<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion;

    unsafe fn ckks_add_pt_const_rnx_inplace_without_normalization<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion;
}

unsafe impl<BE: Backend> CKKSAddOpsWithoutNormalization<BE> for Module<BE>
where
    Module<BE>: CKKSAddDefault<BE>,
{
    unsafe fn ckks_add_without_normalization(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        b: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        self.ckks_add_without_normalization_default(dst, a, b, scratch)
    }

    unsafe fn ckks_add_inplace_without_normalization(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEAdd + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        self.ckks_add_inplace_without_normalization_default(dst, a, scratch)
    }

    unsafe fn ckks_add_pt_vec_znx_without_normalization(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshAddInto<BE> + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        self.ckks_add_pt_vec_znx_without_normalization_default(dst, a, pt_znx, scratch)
    }

    unsafe fn ckks_add_pt_vec_znx_inplace_without_normalization(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_znx: &CKKSPlaintextVecZnx<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: VecZnxRshAddInto<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        self.ckks_add_pt_vec_znx_inplace_without_normalization_default(dst, pt_znx, scratch)
    }

    unsafe fn ckks_add_pt_vec_rnx_without_normalization<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + VecZnxRshAddInto<BE> + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
    {
        self.ckks_add_pt_vec_rnx_without_normalization_default(dst, a, pt_rnx, prec, scratch)
    }

    unsafe fn ckks_add_pt_vec_rnx_inplace_without_normalization<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        pt_rnx: &CKKSPlaintextVecRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: ModuleN + VecZnxRshAddInto<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextVecRnx<F>: CKKSPlaintextConversion,
    {
        self.ckks_add_pt_vec_rnx_inplace_without_normalization_default(dst, pt_rnx, prec, scratch)
    }

    unsafe fn ckks_add_pt_const_znx_without_normalization(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        self.ckks_add_const_znx_without_normalization_default(dst, a, cst_znx, scratch)
    }

    unsafe fn ckks_add_pt_const_znx_inplace_without_normalization(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        cst_znx: &CKKSPlaintextCstZnx,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        self.ckks_add_const_znx_inplace_without_normalization_default(dst, cst_znx, scratch)
    }

    unsafe fn ckks_add_pt_const_rnx_without_normalization<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        a: &CKKSCiphertext<impl DataRef>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion,
    {
        self.ckks_add_const_rnx_without_normalization_default(dst, a, cst_rnx, prec, scratch)
    }

    unsafe fn ckks_add_pt_const_rnx_inplace_without_normalization<F>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        cst_rnx: &CKKSPlaintextCstRnx<F>,
        prec: CKKSMeta,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
        CKKSPlaintextCstRnx<F>: CKKSConstPlaintextConversion,
    {
        self.ckks_add_const_rnx_inplace_without_normalization_default(dst, cst_rnx, prec, scratch)
    }
}
