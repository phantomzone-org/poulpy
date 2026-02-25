//! NTT-domain vector polynomial operations for the NTT120 backend.
//!
//! This module provides:
//!
//! - The [`NttModuleHandle`] trait, which exposes precomputed NTT/iNTT
//!   tables and multiply–accumulate metadata from a module handle.
//! - Forward (`ntt120_vec_znx_dft_apply`) and inverse
//!   (`ntt120_vec_znx_idft_apply`, `ntt120_vec_znx_idft_apply_tmpa`) DFT
//!   operations.
//! - Component-wise DFT-domain arithmetic (add, sub, negate, copy, zero).
//!
//! # Scalar layout
//!
//! `VecZnxDft<_, NTT120Ref>` stores [`Q120bScalar`] values (32 bytes each).
//! Each `Q120bScalar` holds four `u64` CRT residues for one ring coefficient.
//! A `bytemuck::cast_slice` converts a `&[Q120bScalar]` limb slice to
//! `&[u64]` for use with the primitive NTT arithmetic functions.
//!
//! # Prime set
//!
//! All arithmetic is hardcoded to [`Primes30`] (the spqlios-arithmetic
//! default, Q ≈ 2^120).  Generalisation to `Primes29` / `Primes31`
//! is future work.

use bytemuck::{cast_slice, cast_slice_mut};

use crate::{
    layouts::{
        Backend, Module, VecZnxBig, VecZnxBigToMut, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef, VecZnxToRef, ZnxInfos, ZnxView,
        ZnxViewMut,
    },
    reference::ntt120::{
        NttAdd, NttAddInplace, NttCopy, NttDFTExecute, NttFromZnx64, NttNegate, NttNegateInplace, NttSub, NttSubInplace,
        NttSubNegateInplace, NttToZnx128, NttZero,
        mat_vec::BbcMeta,
        ntt::{NttTable, NttTableInv},
        primes::Primes30,
        types::Q120bScalar,
    },
};

// ──────────────────────────────────────────────────────────────────────────────
// NttModuleHandle trait + NttHandleProvider blanket impl
// ──────────────────────────────────────────────────────────────────────────────

/// Access to the precomputed NTT/iNTT tables and lazy-accumulation metadata
/// stored inside a `Module<B>` handle.
///
/// Automatically implemented for any `Module<B>` whose `B::Handle` implements
/// [`NttHandleProvider`].  Backend crates (e.g. `poulpy-cpu-ref`) implement
/// `NttHandleProvider` for their concrete handle type; they do *not* implement
/// this trait directly (which would violate the orphan rule).
///
/// <!-- DOCUMENTED EXCEPTION: Primes30 hardcoded for spqlios compatibility.
///   Generalisation path: add `type PrimeSet: PrimeSet` as an associated type here,
///   then parameterise NttTable/NttTableInv/BbcMeta accordingly. -->
pub trait NttModuleHandle {
    /// Precomputed forward NTT twiddle table (Primes30, size `n`).
    fn get_ntt_table(&self) -> &NttTable<Primes30>;
    /// Precomputed inverse NTT twiddle table (Primes30, size `n`).
    fn get_intt_table(&self) -> &NttTableInv<Primes30>;
    /// Precomputed metadata for `q120b × q120c` lazy multiply–accumulate.
    fn get_bbc_meta(&self) -> &BbcMeta<Primes30>;
}

/// Implemented by backend `Handle` types that store NTT/iNTT tables and BBC
/// metadata.
///
/// Implement this trait for your concrete handle struct (e.g. `NTT120RefHandle`)
/// in the backend crate.  A blanket `impl NttModuleHandle for Module<B>` is
/// provided here in `poulpy-hal`, so no orphan-rule violation occurs.
///
/// # Safety
///
/// Implementors must ensure the returned references are valid for the lifetime
/// of `&self` and that the tables were fully initialised before first use.
pub unsafe trait NttHandleProvider {
    /// Returns a reference to the forward NTT twiddle table.
    fn get_ntt_table(&self) -> &NttTable<Primes30>;
    /// Returns a reference to the inverse NTT twiddle table.
    fn get_intt_table(&self) -> &NttTableInv<Primes30>;
    /// Returns a reference to the lazy multiply–accumulate metadata.
    fn get_bbc_meta(&self) -> &BbcMeta<Primes30>;
}

/// Blanket impl: any `Module<B>` whose handle implements `NttHandleProvider`
/// automatically satisfies `NttModuleHandle`.
impl<B> NttModuleHandle for Module<B>
where
    B: Backend,
    B::Handle: NttHandleProvider,
{
    fn get_ntt_table(&self) -> &NttTable<Primes30> {
        // SAFETY: `ptr()` returns a valid, non-null pointer to `B::Handle`
        // that was initialised by `ModuleNewImpl::new_impl` and is kept
        // alive by the `Module`.
        unsafe { (&*self.ptr()).get_ntt_table() }
    }

    fn get_intt_table(&self) -> &NttTableInv<Primes30> {
        unsafe { (&*self.ptr()).get_intt_table() }
    }

    fn get_bbc_meta(&self) -> &BbcMeta<Primes30> {
        unsafe { (&*self.ptr()).get_bbc_meta() }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Helper: cast VecZnxDft limb to &[u64]
// ──────────────────────────────────────────────────────────────────────────────

/// Returns the q120b u64 slice for limb `(col, limb)` of a VecZnxDft.
///
/// `at(col, limb)` returns `&[Q120bScalar]` of length `n`; we cast to
/// `&[u64]` of length `4*n`.
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

/// Forward NTT: encode `a[a_col]` into `res[res_col]`.
///
/// For each output limb `j`:
/// - Input limb index `= offset + j * step` from `a[a_col]`.
/// - Converts i64 coefficients to q120b with [`NttFromZnx64`],
///   then applies the forward NTT in-place via [`NttDFTExecute`].
/// - Missing input limbs (out of range) are zeroed in `res`.
pub fn ntt120_vec_znx_dft_apply<R, A, BE>(
    module: &impl NttModuleHandle,
    step: usize,
    offset: usize,
    res: &mut R,
    res_col: usize,
    a: &A,
    a_col: usize,
) where
    BE: Backend<ScalarPrep = Q120bScalar> + NttDFTExecute<NttTable<Primes30>> + NttFromZnx64 + NttZero,
    R: VecZnxDftToMut<BE>,
    A: VecZnxToRef,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a = a.to_ref();

    let a_size = a.size();
    let res_size = res.size();

    let table = module.get_ntt_table();

    let steps = a_size.div_ceil(step);
    let min_steps = res_size.min(steps);

    for j in 0..min_steps {
        let limb = offset + j * step;
        if limb < a_size {
            let res_slice: &mut [u64] = limb_u64_mut(&mut res, res_col, j);
            BE::ntt_from_znx64(res_slice, a.at(a_col, limb));
            BE::ntt_dft_execute(table, res_slice);
        } else {
            BE::ntt_zero(limb_u64_mut(&mut res, res_col, j));
        }
    }

    for j in min_steps..res_size {
        BE::ntt_zero(limb_u64_mut(&mut res, res_col, j));
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Inverse DFT
// ──────────────────────────────────────────────────────────────────────────────

/// Returns the scratch space (in bytes) for [`ntt120_vec_znx_idft_apply`].
///
/// Requires one q120b buffer of length `n` (4 u64 per coefficient).
pub fn ntt120_vec_znx_idft_apply_tmp_bytes(n: usize) -> usize {
    4 * n * size_of::<u64>()
}

/// Inverse NTT (non-destructive): decode `a[a_col]` into `res[res_col]`.
///
/// For each output limb `j`:
/// 1. Copies `a.at(a_col, j)` into `tmp` via [`NttCopy`].
/// 2. Applies the inverse NTT to `tmp` in place via [`NttDFTExecute`].
/// 3. CRT-reconstructs the `i128` coefficients via [`NttToZnx128`].
///
/// `tmp` must hold at least `4 * n` `u64` values.
pub fn ntt120_vec_znx_idft_apply<R, A, BE>(
    module: &impl NttModuleHandle,
    res: &mut R,
    res_col: usize,
    a: &A,
    a_col: usize,
    tmp: &mut [u64],
) where
    BE: Backend<ScalarPrep = Q120bScalar, ScalarBig = i128> + NttDFTExecute<NttTableInv<Primes30>> + NttToZnx128 + NttCopy,
    R: VecZnxBigToMut<BE>,
    A: VecZnxDftToRef<BE>,
{
    let mut res: VecZnxBig<&mut [u8], BE> = res.to_mut();
    let a: VecZnxDft<&[u8], BE> = a.to_ref();

    let n = res.n();
    let res_size = res.size();
    let min_size = res_size.min(a.size());

    let table = module.get_intt_table();

    for j in 0..min_size {
        let a_slice: &[u64] = limb_u64(&a, a_col, j);
        let tmp_n: &mut [u64] = &mut tmp[..4 * n];
        BE::ntt_copy(tmp_n, a_slice);
        BE::ntt_dft_execute(table, tmp_n);
        BE::ntt_to_znx128(res.at_mut(res_col, j), n, tmp_n);
    }

    for j in min_size..res_size {
        res.at_mut(res_col, j).fill(0i128);
    }
}

/// Inverse NTT (destructive): decode `a[a_col]` into `res[res_col]`.
///
/// Like [`ntt120_vec_znx_idft_apply`] but applies the inverse NTT
/// **in place** to `a`, modifying it.  Requires no scratch space.
pub fn ntt120_vec_znx_idft_apply_tmpa<R, A, BE>(
    module: &impl NttModuleHandle,
    res: &mut R,
    res_col: usize,
    a: &mut A,
    a_col: usize,
) where
    BE: Backend<ScalarPrep = Q120bScalar, ScalarBig = i128> + NttDFTExecute<NttTableInv<Primes30>> + NttToZnx128,
    R: VecZnxBigToMut<BE>,
    A: VecZnxDftToMut<BE>,
{
    let mut res: VecZnxBig<&mut [u8], BE> = res.to_mut();
    let mut a: VecZnxDft<&mut [u8], BE> = a.to_mut();

    let n = res.n();
    let res_size = res.size();
    let min_size = res_size.min(a.size());

    let table = module.get_intt_table();

    for j in 0..min_size {
        BE::ntt_dft_execute(table, limb_u64_mut(&mut a, a_col, j));
        let a_slice: &[u64] = limb_u64(&a, a_col, j);
        BE::ntt_to_znx128(res.at_mut(res_col, j), n, a_slice);
    }

    for j in min_size..res_size {
        res.at_mut(res_col, j).fill(0i128);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// DFT-domain arithmetic
// ──────────────────────────────────────────────────────────────────────────────

/// DFT-domain add: `res[res_col] = a[a_col] + b[b_col]`.
///
/// Uses lazy q120b addition; out-of-range limbs are copied or zeroed.
pub fn ntt120_vec_znx_dft_add<R, A, B, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend<ScalarPrep = Q120bScalar> + NttAdd + NttCopy + NttZero,
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
            BE::ntt_add(
                limb_u64_mut(&mut res, res_col, j),
                limb_u64(&a, a_col, j),
                limb_u64(&b, b_col, j),
            );
        }
        for j in sum_size..cpy_size {
            BE::ntt_copy(limb_u64_mut(&mut res, res_col, j), limb_u64(&b, b_col, j));
        }
        for j in cpy_size..res_size {
            BE::ntt_zero(limb_u64_mut(&mut res, res_col, j));
        }
    } else {
        let sum_size = b_size.min(res_size);
        let cpy_size = a_size.min(res_size);
        for j in 0..sum_size {
            BE::ntt_add(
                limb_u64_mut(&mut res, res_col, j),
                limb_u64(&a, a_col, j),
                limb_u64(&b, b_col, j),
            );
        }
        for j in sum_size..cpy_size {
            BE::ntt_copy(limb_u64_mut(&mut res, res_col, j), limb_u64(&a, a_col, j));
        }
        for j in cpy_size..res_size {
            BE::ntt_zero(limb_u64_mut(&mut res, res_col, j));
        }
    }
}

/// DFT-domain in-place add: `res[res_col] += a[a_col]`.
pub fn ntt120_vec_znx_dft_add_inplace<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarPrep = Q120bScalar> + NttAddInplace,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: VecZnxDft<&[u8], BE> = a.to_ref();

    let sum_size = res.size().min(a.size());
    for j in 0..sum_size {
        BE::ntt_add_inplace(limb_u64_mut(&mut res, res_col, j), limb_u64(&a, a_col, j));
    }
}

/// DFT-domain scaled in-place add: `res[res_col] += a[a_col] >> (a_scale * base2k)`.
///
/// `a_scale > 0` shifts `a` down by `a_scale` limbs (drops low limbs);
/// `a_scale < 0` shifts `a` up by `|a_scale|` limbs (adds into higher limbs).
pub fn ntt120_vec_znx_dft_add_scaled_inplace<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, a_scale: i64)
where
    BE: Backend<ScalarPrep = Q120bScalar> + NttAddInplace,
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
            BE::ntt_add_inplace(limb_u64_mut(&mut res, res_col, j), limb_u64(&a, a_col, j + shift));
        }
    } else if a_scale < 0 {
        let shift = (a_scale.unsigned_abs() as usize).min(res_size);
        let sum_size = a_size.min(res_size.saturating_sub(shift));
        for j in 0..sum_size {
            BE::ntt_add_inplace(limb_u64_mut(&mut res, res_col, j + shift), limb_u64(&a, a_col, j));
        }
    } else {
        let sum_size = a_size.min(res_size);
        for j in 0..sum_size {
            BE::ntt_add_inplace(limb_u64_mut(&mut res, res_col, j), limb_u64(&a, a_col, j));
        }
    }
}

/// DFT-domain sub: `res[res_col] = a[a_col] - b[b_col]`.
pub fn ntt120_vec_znx_dft_sub<R, A, B, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend<ScalarPrep = Q120bScalar> + NttSub + NttNegate + NttCopy + NttZero,
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
            BE::ntt_sub(
                limb_u64_mut(&mut res, res_col, j),
                limb_u64(&a, a_col, j),
                limb_u64(&b, b_col, j),
            );
        }
        for j in sum_size..cpy_size {
            BE::ntt_negate(limb_u64_mut(&mut res, res_col, j), limb_u64(&b, b_col, j));
        }
        for j in cpy_size..res_size {
            BE::ntt_zero(limb_u64_mut(&mut res, res_col, j));
        }
    } else {
        let sum_size = b_size.min(res_size);
        let cpy_size = a_size.min(res_size);
        for j in 0..sum_size {
            BE::ntt_sub(
                limb_u64_mut(&mut res, res_col, j),
                limb_u64(&a, a_col, j),
                limb_u64(&b, b_col, j),
            );
        }
        for j in sum_size..cpy_size {
            BE::ntt_copy(limb_u64_mut(&mut res, res_col, j), limb_u64(&a, a_col, j));
        }
        for j in cpy_size..res_size {
            BE::ntt_zero(limb_u64_mut(&mut res, res_col, j));
        }
    }
}

/// DFT-domain in-place sub: `res[res_col] -= a[a_col]`.
pub fn ntt120_vec_znx_dft_sub_inplace<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarPrep = Q120bScalar> + NttSubInplace,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: VecZnxDft<&[u8], BE> = a.to_ref();

    let sum_size = res.size().min(a.size());
    for j in 0..sum_size {
        BE::ntt_sub_inplace(limb_u64_mut(&mut res, res_col, j), limb_u64(&a, a_col, j));
    }
}

/// DFT-domain in-place swap-sub: `res[res_col] = a[a_col] - res[res_col]`.
///
/// Extra `res` limbs beyond `a.size()` are negated.
pub fn ntt120_vec_znx_dft_sub_negate_inplace<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarPrep = Q120bScalar> + NttSubNegateInplace + NttNegateInplace,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: VecZnxDft<&[u8], BE> = a.to_ref();

    let res_size = res.size();
    let sum_size = res_size.min(a.size());
    for j in 0..sum_size {
        BE::ntt_sub_negate_inplace(limb_u64_mut(&mut res, res_col, j), limb_u64(&a, a_col, j));
    }
    for j in sum_size..res_size {
        BE::ntt_negate_inplace(limb_u64_mut(&mut res, res_col, j));
    }
}

/// DFT-domain copy with stride: `res[res_col][j] = a[a_col][offset + j*step]`.
///
/// Mirrors `vec_znx_dft_copy` from the FFT64 backend.
pub fn ntt120_vec_znx_dft_copy<R, A, BE>(step: usize, offset: usize, res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarPrep = Q120bScalar> + NttCopy + NttZero,
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
            BE::ntt_copy(limb_u64_mut(&mut res, res_col, j), limb_u64(&a, a_col, limb));
        } else {
            BE::ntt_zero(limb_u64_mut(&mut res, res_col, j));
        }
    }
    for j in min_steps..res.size() {
        BE::ntt_zero(limb_u64_mut(&mut res, res_col, j));
    }
}

/// Zero all limbs of `res[res_col]`.
pub fn ntt120_vec_znx_dft_zero<R, BE>(res: &mut R, res_col: usize)
where
    BE: Backend<ScalarPrep = Q120bScalar> + NttZero,
    R: VecZnxDftToMut<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    for j in 0..res.size() {
        BE::ntt_zero(limb_u64_mut(&mut res, res_col, j));
    }
}
