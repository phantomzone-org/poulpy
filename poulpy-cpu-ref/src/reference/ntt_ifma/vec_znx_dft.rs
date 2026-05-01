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
    NttIfmaAdd, NttIfmaAddAssign, NttIfmaCopy, NttIfmaDFTExecute, NttIfmaFromZnx64, NttIfmaNegate, NttIfmaNegateAssign,
    NttIfmaSub, NttIfmaSubAssign, NttIfmaSubNegateAssign, NttIfmaToZnx128, NttIfmaZero,
    mat_vec::BbcIfmaMeta,
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

// ──────────────────────────────────────────────────────────────────────────────
// NttIfmaHandleFactory
// ──────────────────────────────────────────────────────────────────────────────

/// Construct IFMA backend handles for [`Module::new`](crate::api::ModuleNew::new).
///
/// # Safety
///
/// Implementors must return a fully initialized handle for the requested `n`.
/// The handle is boxed and stored inside the `Module`, so it must be safe to
/// drop via [`crate::layouts::Backend::destroy`].
pub unsafe trait NttIfmaHandleFactory: Sized {
    /// Builds a fully initialized handle for ring dimension `n`.
    fn create_ntt_ifma_handle(n: usize) -> Self;

    /// Optional runtime capability check (default: no-op).
    fn assert_ntt_ifma_runtime_support() {}
}

// ──────────────────────────────────────────────────────────────────────────────
// DFT-domain arithmetic (IFMA-specific versions)
// ──────────────────────────────────────────────────────────────────────────────

/// DFT-domain add: `res[res_col] = a[a_col] + b[b_col]`.
pub fn ntt_ifma_vec_znx_dft_add_into<R, A, B, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend<ScalarPrep = Q120bScalar> + NttIfmaAdd + NttIfmaCopy + NttIfmaZero,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
    B: VecZnxDftToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: VecZnxDft<&[u8], BE> = a.to_ref();
    let b: VecZnxDft<&[u8], BE> = b.to_ref();

    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();

    if a_size <= b_size {
        let sum_size = a_size.min(res_size);
        let cpy_size = b_size.min(res_size);
        for j in 0..sum_size {
            BE::ntt_ifma_add(
                limb_u64_mut(&mut res, res_col, j),
                limb_u64(&a, a_col, j),
                limb_u64(&b, b_col, j),
            );
        }
        for j in sum_size..cpy_size {
            BE::ntt_ifma_copy(limb_u64_mut(&mut res, res_col, j), limb_u64(&b, b_col, j));
        }
        for j in cpy_size..res_size {
            BE::ntt_ifma_zero(limb_u64_mut(&mut res, res_col, j));
        }
    } else {
        let sum_size = b_size.min(res_size);
        let cpy_size = a_size.min(res_size);
        for j in 0..sum_size {
            BE::ntt_ifma_add(
                limb_u64_mut(&mut res, res_col, j),
                limb_u64(&a, a_col, j),
                limb_u64(&b, b_col, j),
            );
        }
        for j in sum_size..cpy_size {
            BE::ntt_ifma_copy(limb_u64_mut(&mut res, res_col, j), limb_u64(&a, a_col, j));
        }
        for j in cpy_size..res_size {
            BE::ntt_ifma_zero(limb_u64_mut(&mut res, res_col, j));
        }
    }
}

/// DFT-domain in-place add: `res[res_col] += a[a_col]`.
pub fn ntt_ifma_vec_znx_dft_add_assign<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarPrep = Q120bScalar> + NttIfmaAddAssign,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: VecZnxDft<&[u8], BE> = a.to_ref();

    let sum_size = res.size().min(a.size());
    for j in 0..sum_size {
        BE::ntt_ifma_add_assign(limb_u64_mut(&mut res, res_col, j), limb_u64(&a, a_col, j));
    }
}

/// DFT-domain scaled in-place add: `res[res_col] += a[a_col] >> (a_scale * base2k)`.
pub fn ntt_ifma_vec_znx_dft_add_scaled_assign<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, a_scale: i64)
where
    BE: Backend<ScalarPrep = Q120bScalar> + NttIfmaAddAssign,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: VecZnxDft<&[u8], BE> = a.to_ref();

    let res_size = res.size();
    let a_size = a.size();

    if a_scale > 0 {
        let shift = (a_scale as usize).min(a_size);
        let sum_size = a_size.min(res_size).saturating_sub(shift);
        for j in 0..sum_size {
            BE::ntt_ifma_add_assign(limb_u64_mut(&mut res, res_col, j), limb_u64(&a, a_col, j + shift));
        }
    } else if a_scale < 0 {
        let shift = (a_scale.unsigned_abs() as usize).min(res_size);
        let sum_size = a_size.min(res_size.saturating_sub(shift));
        for j in 0..sum_size {
            BE::ntt_ifma_add_assign(limb_u64_mut(&mut res, res_col, j + shift), limb_u64(&a, a_col, j));
        }
    } else {
        let sum_size = a_size.min(res_size);
        for j in 0..sum_size {
            BE::ntt_ifma_add_assign(limb_u64_mut(&mut res, res_col, j), limb_u64(&a, a_col, j));
        }
    }
}

/// DFT-domain sub: `res[res_col] = a[a_col] - b[b_col]`.
pub fn ntt_ifma_vec_znx_dft_sub<R, A, B, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend<ScalarPrep = Q120bScalar> + NttIfmaSub + NttIfmaNegate + NttIfmaCopy + NttIfmaZero,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
    B: VecZnxDftToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: VecZnxDft<&[u8], BE> = a.to_ref();
    let b: VecZnxDft<&[u8], BE> = b.to_ref();

    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();

    if a_size <= b_size {
        let sum_size = a_size.min(res_size);
        let cpy_size = b_size.min(res_size);
        for j in 0..sum_size {
            BE::ntt_ifma_sub(
                limb_u64_mut(&mut res, res_col, j),
                limb_u64(&a, a_col, j),
                limb_u64(&b, b_col, j),
            );
        }
        for j in sum_size..cpy_size {
            BE::ntt_ifma_negate(limb_u64_mut(&mut res, res_col, j), limb_u64(&b, b_col, j));
        }
        for j in cpy_size..res_size {
            BE::ntt_ifma_zero(limb_u64_mut(&mut res, res_col, j));
        }
    } else {
        let sum_size = b_size.min(res_size);
        let cpy_size = a_size.min(res_size);
        for j in 0..sum_size {
            BE::ntt_ifma_sub(
                limb_u64_mut(&mut res, res_col, j),
                limb_u64(&a, a_col, j),
                limb_u64(&b, b_col, j),
            );
        }
        for j in sum_size..cpy_size {
            BE::ntt_ifma_copy(limb_u64_mut(&mut res, res_col, j), limb_u64(&a, a_col, j));
        }
        for j in cpy_size..res_size {
            BE::ntt_ifma_zero(limb_u64_mut(&mut res, res_col, j));
        }
    }
}

/// DFT-domain in-place sub: `res[res_col] -= a[a_col]`.
pub fn ntt_ifma_vec_znx_dft_sub_assign<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarPrep = Q120bScalar> + NttIfmaSubAssign,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: VecZnxDft<&[u8], BE> = a.to_ref();

    let sum_size = res.size().min(a.size());
    for j in 0..sum_size {
        BE::ntt_ifma_sub_assign(limb_u64_mut(&mut res, res_col, j), limb_u64(&a, a_col, j));
    }
}

/// DFT-domain in-place swap-sub: `res[res_col] = a[a_col] - res[res_col]`.
pub fn ntt_ifma_vec_znx_dft_sub_negate_assign<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarPrep = Q120bScalar> + NttIfmaSubNegateAssign + NttIfmaNegateAssign,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: VecZnxDft<&[u8], BE> = a.to_ref();

    let res_size = res.size();
    let sum_size = res_size.min(a.size());
    for j in 0..sum_size {
        BE::ntt_ifma_sub_negate_assign(limb_u64_mut(&mut res, res_col, j), limb_u64(&a, a_col, j));
    }
    for j in sum_size..res_size {
        BE::ntt_ifma_negate_assign(limb_u64_mut(&mut res, res_col, j));
    }
}

/// DFT-domain copy with stride: `res[res_col][j] = a[a_col][offset + j*step]`.
pub fn ntt_ifma_vec_znx_dft_copy<R, A, BE>(step: usize, offset: usize, res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarPrep = Q120bScalar> + NttIfmaCopy + NttIfmaZero,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: VecZnxDft<&[u8], BE> = a.to_ref();

    #[cfg(debug_assertions)]
    {
        assert_eq!(res.n(), a.n())
    }

    let steps: usize = a.size().div_ceil(step);
    let min_steps: usize = res.size().min(steps);

    for j in 0..min_steps {
        let limb = offset + j * step;
        if limb < a.size() {
            BE::ntt_ifma_copy(limb_u64_mut(&mut res, res_col, j), limb_u64(&a, a_col, limb));
        } else {
            BE::ntt_ifma_zero(limb_u64_mut(&mut res, res_col, j));
        }
    }
    for j in min_steps..res.size() {
        BE::ntt_ifma_zero(limb_u64_mut(&mut res, res_col, j));
    }
}

/// Zero all limbs of `res[res_col]`.
pub fn ntt_ifma_vec_znx_dft_zero<R, BE>(res: &mut R, res_col: usize)
where
    BE: Backend<ScalarPrep = Q120bScalar> + NttIfmaZero,
    R: VecZnxDftToMut<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    for j in 0..res.size() {
        BE::ntt_ifma_zero(limb_u64_mut(&mut res, res_col, j));
    }
}
