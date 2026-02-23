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
        arithmetic::{add_bbb_ref, b_from_znx64_ref, b_to_znx128_ref},
        mat_vec::BbcMeta,
        ntt::{NttTable, NttTableInv, intt_ref, ntt_ref},
        primes::{PrimeSet, Primes30},
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
// Private arithmetic helpers for Primes30 q120b
// ──────────────────────────────────────────────────────────────────────────────

const Q_SHIFTED: [u64; 4] = [
    (Primes30::Q[0] as u64) << 33,
    (Primes30::Q[1] as u64) << 33,
    (Primes30::Q[2] as u64) << 33,
    (Primes30::Q[3] as u64) << 33,
];

/// In-place add: `res[i] = (res[i] % q_shifted[k]) + (a[i] % q_shifted[k])`.
#[inline(always)]
fn add_bbb_inplace(n: usize, res: &mut [u64], a: &[u64]) {
    debug_assert!(res.len() >= 4 * n);
    debug_assert!(a.len() >= 4 * n);
    for j in 0..n {
        for (k, &q_s) in Q_SHIFTED.iter().enumerate() {
            let idx = 4 * j + k;
            res[idx] = res[idx] % q_s + a[idx] % q_s;
        }
    }
}

/// Out-of-place sub: `res = a - b` in lazy q120b arithmetic.
#[inline(always)]
fn sub_bbb(n: usize, res: &mut [u64], a: &[u64], b: &[u64]) {
    debug_assert!(res.len() >= 4 * n);
    debug_assert!(a.len() >= 4 * n);
    debug_assert!(b.len() >= 4 * n);
    for j in 0..n {
        for (k, &q_s) in Q_SHIFTED.iter().enumerate() {
            let idx = 4 * j + k;
            res[idx] = a[idx] % q_s + (q_s - b[idx] % q_s);
        }
    }
}

/// In-place sub: `res -= a` in lazy q120b arithmetic.
#[inline(always)]
fn sub_bbb_inplace(n: usize, res: &mut [u64], a: &[u64]) {
    debug_assert!(res.len() >= 4 * n);
    debug_assert!(a.len() >= 4 * n);
    for j in 0..n {
        for (k, &q_s) in Q_SHIFTED.iter().enumerate() {
            let idx = 4 * j + k;
            res[idx] = res[idx] % q_s + (q_s - a[idx] % q_s);
        }
    }
}

/// In-place swap-sub: `res = a - res` in lazy q120b arithmetic.
#[inline(always)]
fn sub_negate_bbb_inplace(n: usize, res: &mut [u64], a: &[u64]) {
    debug_assert!(res.len() >= 4 * n);
    debug_assert!(a.len() >= 4 * n);
    for j in 0..n {
        for (k, &q_s) in Q_SHIFTED.iter().enumerate() {
            let idx = 4 * j + k;
            res[idx] = a[idx] % q_s + (q_s - res[idx] % q_s);
        }
    }
}

/// Out-of-place negate: `res = -a` in lazy q120b arithmetic.
#[inline(always)]
fn negate_bbb(n: usize, res: &mut [u64], a: &[u64]) {
    debug_assert!(res.len() >= 4 * n);
    debug_assert!(a.len() >= 4 * n);
    for j in 0..n {
        for (k, &q_s) in Q_SHIFTED.iter().enumerate() {
            let idx = 4 * j + k;
            res[idx] = q_s - a[idx] % q_s;
        }
    }
}

/// In-place negate: `res = -res` in lazy q120b arithmetic.
#[inline(always)]
fn negate_bbb_inplace(n: usize, res: &mut [u64]) {
    debug_assert!(res.len() >= 4 * n);
    for j in 0..n {
        for (k, &q_s) in Q_SHIFTED.iter().enumerate() {
            let idx = 4 * j + k;
            res[idx] = q_s - res[idx] % q_s;
        }
    }
}

/// Copy `a` into `res`.
#[inline(always)]
fn copy_bbb(n: usize, res: &mut [u64], a: &[u64]) {
    res[..4 * n].copy_from_slice(&a[..4 * n]);
}

/// Zero `res`.
#[inline(always)]
fn zero_bbb(n: usize, res: &mut [u64]) {
    res[..4 * n].fill(0);
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
/// - Converts i64 coefficients to q120b with [`b_from_znx64_ref`],
///   then applies the forward NTT in-place.
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
    BE: Backend<ScalarPrep = Q120bScalar>,
    R: VecZnxDftToMut<BE>,
    A: VecZnxToRef,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a = a.to_ref();

    let n = res.n();
    let a_size = a.size();
    let res_size = res.size();

    let table = module.get_ntt_table();

    let steps = a_size.div_ceil(step);
    let min_steps = res_size.min(steps);

    for j in 0..min_steps {
        let limb = offset + j * step;
        if limb < a_size {
            let res_slice: &mut [u64] = limb_u64_mut(&mut res, res_col, j);
            b_from_znx64_ref::<Primes30>(n, res_slice, a.at(a_col, limb));
            ntt_ref::<Primes30>(table, res_slice);
        } else {
            zero_bbb(n, limb_u64_mut(&mut res, res_col, j));
        }
    }

    for j in min_steps..res_size {
        zero_bbb(n, limb_u64_mut(&mut res, res_col, j));
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
/// 1. Copies `a.at(a_col, j)` into `tmp`.
/// 2. Applies the inverse NTT to `tmp` in place (with 1/n scaling baked in).
/// 3. CRT-reconstructs the `i128` coefficients via [`b_to_znx128_ref`].
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
    BE: Backend<ScalarPrep = Q120bScalar, ScalarBig = i128>,
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
        tmp_n.copy_from_slice(a_slice);
        intt_ref::<Primes30>(table, tmp_n);
        b_to_znx128_ref::<Primes30>(n, res.at_mut(res_col, j), tmp_n);
    }

    for j in min_size..res_size {
        res.at_mut(res_col, j).iter_mut().for_each(|x| *x = 0i128);
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
    BE: Backend<ScalarPrep = Q120bScalar, ScalarBig = i128>,
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
        intt_ref::<Primes30>(table, limb_u64_mut(&mut a, a_col, j));
        let a_slice: &[u64] = limb_u64(&a, a_col, j);
        b_to_znx128_ref::<Primes30>(n, res.at_mut(res_col, j), a_slice);
    }

    for j in min_size..res_size {
        res.at_mut(res_col, j).iter_mut().for_each(|x| *x = 0i128);
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
    BE: Backend<ScalarPrep = Q120bScalar>,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
    B: VecZnxDftToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: VecZnxDft<&[u8], BE> = a.to_ref();
    let b: VecZnxDft<&[u8], BE> = b.to_ref();

    let n = res.n();
    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();

    if a_size <= b_size {
        let sum_size = a_size.min(res_size);
        let cpy_size = b_size.min(res_size);
        for j in 0..sum_size {
            add_bbb_ref::<Primes30>(
                n,
                limb_u64_mut(&mut res, res_col, j),
                limb_u64(&a, a_col, j),
                limb_u64(&b, b_col, j),
            );
        }
        for j in sum_size..cpy_size {
            copy_bbb(n, limb_u64_mut(&mut res, res_col, j), limb_u64(&b, b_col, j));
        }
        for j in cpy_size..res_size {
            zero_bbb(n, limb_u64_mut(&mut res, res_col, j));
        }
    } else {
        let sum_size = b_size.min(res_size);
        let cpy_size = a_size.min(res_size);
        for j in 0..sum_size {
            add_bbb_ref::<Primes30>(
                n,
                limb_u64_mut(&mut res, res_col, j),
                limb_u64(&a, a_col, j),
                limb_u64(&b, b_col, j),
            );
        }
        for j in sum_size..cpy_size {
            copy_bbb(n, limb_u64_mut(&mut res, res_col, j), limb_u64(&a, a_col, j));
        }
        for j in cpy_size..res_size {
            zero_bbb(n, limb_u64_mut(&mut res, res_col, j));
        }
    }
}

/// DFT-domain in-place add: `res[res_col] += a[a_col]`.
pub fn ntt120_vec_znx_dft_add_inplace<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarPrep = Q120bScalar>,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: VecZnxDft<&[u8], BE> = a.to_ref();

    let n = res.n();
    let sum_size = res.size().min(a.size());
    for j in 0..sum_size {
        add_bbb_inplace(n, limb_u64_mut(&mut res, res_col, j), limb_u64(&a, a_col, j));
    }
}

/// DFT-domain scaled in-place add: `res[res_col] += a[a_col] >> (a_scale * base2k)`.
///
/// `a_scale > 0` shifts `a` down by `a_scale` limbs (drops low limbs);
/// `a_scale < 0` shifts `a` up by `|a_scale|` limbs (adds into higher limbs).
pub fn ntt120_vec_znx_dft_add_scaled_inplace<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, a_scale: i64)
where
    BE: Backend<ScalarPrep = Q120bScalar>,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: VecZnxDft<&[u8], BE> = a.to_ref();

    let n = res.n();
    let res_size = res.size();
    let a_size = a.size();

    if a_scale > 0 {
        let shift = (a_scale as usize).min(a_size);
        let sum_size = a_size.min(res_size).saturating_sub(shift);
        for j in 0..sum_size {
            add_bbb_inplace(n, limb_u64_mut(&mut res, res_col, j), limb_u64(&a, a_col, j + shift));
        }
    } else if a_scale < 0 {
        let shift = (a_scale.unsigned_abs() as usize).min(res_size);
        let sum_size = a_size.min(res_size.saturating_sub(shift));
        for j in 0..sum_size {
            add_bbb_inplace(n, limb_u64_mut(&mut res, res_col, j + shift), limb_u64(&a, a_col, j));
        }
    } else {
        let sum_size = a_size.min(res_size);
        for j in 0..sum_size {
            add_bbb_inplace(n, limb_u64_mut(&mut res, res_col, j), limb_u64(&a, a_col, j));
        }
    }
}

/// DFT-domain sub: `res[res_col] = a[a_col] - b[b_col]`.
pub fn ntt120_vec_znx_dft_sub<R, A, B, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend<ScalarPrep = Q120bScalar>,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
    B: VecZnxDftToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: VecZnxDft<&[u8], BE> = a.to_ref();
    let b: VecZnxDft<&[u8], BE> = b.to_ref();

    let n = res.n();
    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();

    if a_size <= b_size {
        let sum_size = a_size.min(res_size);
        let cpy_size = b_size.min(res_size);
        for j in 0..sum_size {
            sub_bbb(
                n,
                limb_u64_mut(&mut res, res_col, j),
                limb_u64(&a, a_col, j),
                limb_u64(&b, b_col, j),
            );
        }
        for j in sum_size..cpy_size {
            negate_bbb(n, limb_u64_mut(&mut res, res_col, j), limb_u64(&b, b_col, j));
        }
        for j in cpy_size..res_size {
            zero_bbb(n, limb_u64_mut(&mut res, res_col, j));
        }
    } else {
        let sum_size = b_size.min(res_size);
        let cpy_size = a_size.min(res_size);
        for j in 0..sum_size {
            sub_bbb(
                n,
                limb_u64_mut(&mut res, res_col, j),
                limb_u64(&a, a_col, j),
                limb_u64(&b, b_col, j),
            );
        }
        for j in sum_size..cpy_size {
            copy_bbb(n, limb_u64_mut(&mut res, res_col, j), limb_u64(&a, a_col, j));
        }
        for j in cpy_size..res_size {
            zero_bbb(n, limb_u64_mut(&mut res, res_col, j));
        }
    }
}

/// DFT-domain in-place sub: `res[res_col] -= a[a_col]`.
pub fn ntt120_vec_znx_dft_sub_inplace<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarPrep = Q120bScalar>,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: VecZnxDft<&[u8], BE> = a.to_ref();

    let n = res.n();
    let sum_size = res.size().min(a.size());
    for j in 0..sum_size {
        sub_bbb_inplace(n, limb_u64_mut(&mut res, res_col, j), limb_u64(&a, a_col, j));
    }
}

/// DFT-domain in-place swap-sub: `res[res_col] = a[a_col] - res[res_col]`.
///
/// Extra `res` limbs beyond `a.size()` are negated.
pub fn ntt120_vec_znx_dft_sub_negate_inplace<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarPrep = Q120bScalar>,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: VecZnxDft<&[u8], BE> = a.to_ref();

    let n = res.n();
    let res_size = res.size();
    let sum_size = res_size.min(a.size());
    for j in 0..sum_size {
        sub_negate_bbb_inplace(n, limb_u64_mut(&mut res, res_col, j), limb_u64(&a, a_col, j));
    }
    for j in sum_size..res_size {
        negate_bbb_inplace(n, limb_u64_mut(&mut res, res_col, j));
    }
}

/// DFT-domain copy with stride: `res[res_col][j] = a[a_col][offset + j*step]`.
///
/// Mirrors `vec_znx_dft_copy` from the FFT64 backend.
pub fn ntt120_vec_znx_dft_copy<R, A, BE>(step: usize, offset: usize, res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarPrep = Q120bScalar>,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: VecZnxDft<&[u8], BE> = a.to_ref();

    let n = res.n();
    let steps = a.size().div_ceil(step);
    let min_steps = res.size().min(steps);

    for j in 0..min_steps {
        let limb = offset + j * step;
        if limb < a.size() {
            copy_bbb(n, limb_u64_mut(&mut res, res_col, j), limb_u64(&a, a_col, limb));
        }
    }
    for j in min_steps..res.size() {
        zero_bbb(n, limb_u64_mut(&mut res, res_col, j));
    }
}

/// Zero all limbs of `res[res_col]`.
pub fn ntt120_vec_znx_dft_zero<R, BE>(res: &mut R, res_col: usize)
where
    BE: Backend<ScalarPrep = Q120bScalar>,
    R: VecZnxDftToMut<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let n = res.n();
    for j in 0..res.size() {
        zero_bbb(n, limb_u64_mut(&mut res, res_col, j));
    }
}
