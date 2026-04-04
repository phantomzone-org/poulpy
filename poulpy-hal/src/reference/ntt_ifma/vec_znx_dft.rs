//! NTT-domain vector polynomial operations for the IFMA backend.
//!
//! Provides:
//! - [`NttIfmaModuleHandle`] / [`NttIfmaHandleProvider`] traits for table access
//! - Forward/inverse DFT apply functions
//!
//! DFT-domain arithmetic (add, sub, negate, copy, zero) is handled by
//! having the backend implement the same `Ntt{Add,Sub,...}` traits from
//! [`crate::reference::ntt120`] and reusing the `ntt120_vec_znx_dft_*`
//! generic functions directly.

use bytemuck::{cast_slice, cast_slice_mut};

use crate::{
    layouts::{
        Backend, Module, VecZnxBig, VecZnxBigToMut, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef, VecZnxToRef, ZnxInfos, ZnxView,
        ZnxViewMut,
    },
    reference::ntt120::types::Q120bScalar,
};

use super::{
    NttIfmaCopy, NttIfmaDFTExecute, NttIfmaFromZnx64, NttIfmaToZnx128, NttIfmaZero,
    mat_vec::{BbbIfmaMeta, BbcIfmaMeta},
    ntt::{NttIfmaTable, NttIfmaTableInv},
    primes::Primes40,
};

// ──────────────────────────────────────────────────────────────────────────────
// NttIfmaModuleHandle trait + NttIfmaHandleProvider blanket impl
// ──────────────────────────────────────────────────────────────────────────────

/// Access to precomputed NTT/iNTT tables for the IFMA backend.
pub trait NttIfmaModuleHandle {
    fn get_ntt_ifma_table(&self) -> &NttIfmaTable<Primes40>;
    fn get_intt_ifma_table(&self) -> &NttIfmaTableInv<Primes40>;
    fn get_bbc_ifma_meta(&self) -> &BbcIfmaMeta<Primes40>;
    fn get_bbb_ifma_meta(&self) -> &BbbIfmaMeta<Primes40>;
}

/// Implemented by backend `Handle` types that store IFMA NTT tables.
///
/// # Safety
///
/// Implementors must ensure the returned references are valid for the
/// lifetime of `&self` and that the tables were fully initialised.
pub unsafe trait NttIfmaHandleProvider {
    fn get_ntt_ifma_table(&self) -> &NttIfmaTable<Primes40>;
    fn get_intt_ifma_table(&self) -> &NttIfmaTableInv<Primes40>;
    fn get_bbc_ifma_meta(&self) -> &BbcIfmaMeta<Primes40>;
    fn get_bbb_ifma_meta(&self) -> &BbbIfmaMeta<Primes40>;
}

/// Blanket impl: any `Module<B>` whose handle implements `NttIfmaHandleProvider`
/// automatically satisfies `NttIfmaModuleHandle`.
impl<B> NttIfmaModuleHandle for Module<B>
where
    B: Backend,
    B::Handle: NttIfmaHandleProvider,
{
    fn get_ntt_ifma_table(&self) -> &NttIfmaTable<Primes40> {
        unsafe { (&*self.ptr()).get_ntt_ifma_table() }
    }
    fn get_intt_ifma_table(&self) -> &NttIfmaTableInv<Primes40> {
        unsafe { (&*self.ptr()).get_intt_ifma_table() }
    }
    fn get_bbc_ifma_meta(&self) -> &BbcIfmaMeta<Primes40> {
        unsafe { (&*self.ptr()).get_bbc_ifma_meta() }
    }
    fn get_bbb_ifma_meta(&self) -> &BbbIfmaMeta<Primes40> {
        unsafe { (&*self.ptr()).get_bbb_ifma_meta() }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Helper: cast VecZnxDft limb to &[u64]
// ──────────────────────────────────────────────────────────────────────────────

#[inline(always)]
fn limb_u64<D: crate::layouts::DataRef, BE: Backend<ScalarPrep = Q120bScalar>>(
    v: &VecZnxDft<D, BE>,
    col: usize,
    limb: usize,
) -> &[u64] {
    cast_slice(v.at(col, limb))
}

#[inline(always)]
fn limb_u64_mut<D: crate::layouts::DataMut, BE: Backend<ScalarPrep = Q120bScalar>>(
    v: &mut VecZnxDft<D, BE>,
    col: usize,
    limb: usize,
) -> &mut [u64] {
    cast_slice_mut(v.at_mut(col, limb))
}

// ──────────────────────────────────────────────────────────────────────────────
// Forward DFT
// ──────────────────────────────────────────────────────────────────────────────

/// Forward NTT for the IFMA backend.
pub fn ntt_ifma_vec_znx_dft_apply<R, A, BE>(
    module: &impl NttIfmaModuleHandle,
    step: usize,
    offset: usize,
    res: &mut R,
    res_col: usize,
    a: &A,
    a_col: usize,
) where
    BE: Backend<ScalarPrep = Q120bScalar> + NttIfmaDFTExecute<NttIfmaTable<Primes40>> + NttIfmaFromZnx64 + NttIfmaZero,
    R: VecZnxDftToMut<BE>,
    A: VecZnxToRef,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a = a.to_ref();

    let a_size = a.size();
    let res_size = res.size();

    let table = module.get_ntt_ifma_table();

    let steps = a_size.div_ceil(step);
    let min_steps = res_size.min(steps);

    for j in 0..min_steps {
        let limb = offset + j * step;
        if limb < a_size {
            let res_slice: &mut [u64] = limb_u64_mut(&mut res, res_col, j);
            BE::ntt_ifma_from_znx64(res_slice, a.at(a_col, limb));
            BE::ntt_ifma_dft_execute(table, res_slice);
        } else {
            BE::ntt_ifma_zero(limb_u64_mut(&mut res, res_col, j));
        }
    }

    for j in min_steps..res_size {
        BE::ntt_ifma_zero(limb_u64_mut(&mut res, res_col, j));
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Inverse DFT
// ──────────────────────────────────────────────────────────────────────────────

/// Scratch space (in bytes) for [`ntt_ifma_vec_znx_idft_apply`].
pub fn ntt_ifma_vec_znx_idft_apply_tmp_bytes(n: usize) -> usize {
    use std::mem::size_of;
    4 * n * size_of::<u64>()
}

/// Inverse NTT (non-destructive) for the IFMA backend.
pub fn ntt_ifma_vec_znx_idft_apply<R, A, BE>(
    module: &impl NttIfmaModuleHandle,
    res: &mut R,
    res_col: usize,
    a: &A,
    a_col: usize,
    tmp: &mut [u64],
) where
    BE: Backend<ScalarPrep = Q120bScalar, ScalarBig = i128>
        + NttIfmaDFTExecute<NttIfmaTableInv<Primes40>>
        + NttIfmaToZnx128
        + NttIfmaCopy,
    R: VecZnxBigToMut<BE>,
    A: VecZnxDftToRef<BE>,
{
    let mut res: VecZnxBig<&mut [u8], BE> = res.to_mut();
    let a: VecZnxDft<&[u8], BE> = a.to_ref();

    let n = res.n();
    let res_size = res.size();
    let min_size = res_size.min(a.size());

    let table = module.get_intt_ifma_table();

    for j in 0..min_size {
        let a_slice: &[u64] = limb_u64(&a, a_col, j);
        let tmp_n: &mut [u64] = &mut tmp[..4 * n];
        BE::ntt_ifma_copy(tmp_n, a_slice);
        BE::ntt_ifma_dft_execute(table, tmp_n);
        BE::ntt_ifma_to_znx128(res.at_mut(res_col, j), n, tmp_n);
    }

    for j in min_size..res_size {
        res.at_mut(res_col, j).fill(0i128);
    }
}

/// Inverse NTT (destructive — modifies `a` in place).
pub fn ntt_ifma_vec_znx_idft_apply_tmpa<R, A, BE>(
    module: &impl NttIfmaModuleHandle,
    res: &mut R,
    res_col: usize,
    a: &mut A,
    a_col: usize,
) where
    BE: Backend<ScalarPrep = Q120bScalar, ScalarBig = i128> + NttIfmaDFTExecute<NttIfmaTableInv<Primes40>> + NttIfmaToZnx128,
    R: VecZnxBigToMut<BE>,
    A: VecZnxDftToMut<BE>,
{
    let mut res: VecZnxBig<&mut [u8], BE> = res.to_mut();
    let mut a: VecZnxDft<&mut [u8], BE> = a.to_mut();

    let n = res.n();
    let res_size = res.size();
    let min_size = res_size.min(a.size());

    let table = module.get_intt_ifma_table();

    for j in 0..min_size {
        BE::ntt_ifma_dft_execute(table, limb_u64_mut(&mut a, a_col, j));
        let a_slice: &[u64] = limb_u64(&a, a_col, j);
        BE::ntt_ifma_to_znx128(res.at_mut(res_col, j), n, a_slice);
    }

    for j in min_size..res_size {
        res.at_mut(res_col, j).fill(0i128);
    }
}
