//! SVP (scalar-vector product) operations for the NTT120 backend.
//!
//! This module provides the NTT-domain SVP primitives used by
//! `poulpy-cpu-ref`.  The workflow is:
//!
//! 1. **Prepare** — encode a `ScalarZnx` (i64 coefficients) into the
//!    [`SvpPPol`] prepared format (q120c, NTT domain) via
//!    [`ntt120_svp_prepare`].
//! 2. **Apply** — multiply a [`VecZnxDft`] (q120b) by a prepared
//!    [`SvpPPol`] (q120c) to obtain a new [`VecZnxDft`] (q120b) via
//!    one of the `ntt120_svp_apply_dft_to_dft*` functions.
//!
//! # Storage formats
//!
//! | Layout | Scalar type | u64/u32 view | Bytes/coeff |
//! |--------|-------------|--------------|-------------|
//! | `VecZnxDft` (q120b) | `Q120bScalar` | 4 u64 | 32 |
//! | `SvpPPol` (q120c)   | `Q120bScalar` | 8 u32 | 32 |
//!
//! Both layouts share the same [`Q120bScalar`] element type but differ in
//! their arithmetic interpretation.  Use [`bytemuck::cast_slice`] /
//! [`bytemuck::cast_slice_mut`] to obtain the appropriate `&[u32]` or
//! `&[u64]` view.

use bytemuck::{cast_slice, cast_slice_mut};

use crate::{
    layouts::{
        Backend, ScalarZnx, ScalarZnxToRef, SvpPPol, SvpPPolToMut, SvpPPolToRef, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef,
        ZnxInfos, ZnxView, ZnxViewMut,
    },
    reference::ntt120::{
        arithmetic::{b_from_znx64_ref, c_from_b_ref},
        mat_vec::{BbcMeta, vec_mat1col_product_bbc_ref},
        ntt::ntt_ref,
        primes::{PrimeSet, Primes30},
        types::Q120bScalar,
        vec_znx_dft::NttModuleHandle,
    },
};

// Lazy-reduction bound: Q[k] << 33 is the modulus for q120b lazy arithmetic.
const Q_SHIFTED: [u64; 4] = [
    (Primes30::Q[0] as u64) << 33,
    (Primes30::Q[1] as u64) << 33,
    (Primes30::Q[2] as u64) << 33,
    (Primes30::Q[3] as u64) << 33,
];

// ──────────────────────────────────────────────────────────────────────────────
// Prepare
// ──────────────────────────────────────────────────────────────────────────────

/// Encode a scalar polynomial into the q120c NTT-domain prepared format.
///
/// Steps:
/// 1. Map i64 coefficients of `a` to q120b (via [`b_from_znx64_ref`]).
/// 2. Apply the forward NTT (via [`ntt_ref`]).
/// 3. Convert q120b → q120c (via [`c_from_b_ref`]) and store in `res`.
///
/// `res` must be a [`SvpPPol`] with `ScalarPrep = Q120bScalar`.
/// A temporary heap buffer of `4 * n` u64 values is allocated internally
/// (this is a setup/key-preparation function, not a hot path).
pub fn ntt120_svp_prepare<R, A, BE>(module: &impl NttModuleHandle, res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarPrep = Q120bScalar>,
    R: SvpPPolToMut<BE>,
    A: ScalarZnxToRef,
{
    let mut res: SvpPPol<&mut [u8], BE> = res.to_mut();
    let a: ScalarZnx<&[u8]> = a.to_ref();
    let n = res.n();

    // Temporary q120b working buffer (heap-allocated; prepare is not hot).
    let mut tmp = vec![0u64; 4 * n];
    b_from_znx64_ref::<Primes30>(n, &mut tmp, a.at(a_col, 0));
    ntt_ref(module.get_ntt_table(), &mut tmp);

    // Write q120c into the SvpPPol buffer.
    let res_u32: &mut [u32] = cast_slice_mut(res.at_mut(res_col, 0));
    c_from_b_ref::<Primes30>(n, res_u32, &tmp);
}

// ──────────────────────────────────────────────────────────────────────────────
// Private pointwise multiply helper
// ──────────────────────────────────────────────────────────────────────────────

/// Compute one q120b output coefficient = q120b input × q120c prepared coeff.
///
/// Calls `vec_mat1col_product_bbc_ref` with `ell = 1` (single-term product).
///
/// - `res4`:      `&mut [u64; 4]` output (q120b)
/// - `b_u32_ni`:  8-element u32 slice at position `n_i` in the q120b input
///   (bytemuck view of a `Q120bScalar`).
/// - `a_u32_ni`:  8-element u32 slice at position `n_i` in the q120c prepared
///   polynomial.
/// - `meta`:      precomputed `BbcMeta` for `Primes30`.
#[inline(always)]
fn pointwise_mul_bbc(meta: &BbcMeta<Primes30>, res4: &mut [u64], b_u32_ni: &[u32], a_u32_ni: &[u32]) {
    vec_mat1col_product_bbc_ref::<Primes30>(meta, 1, res4, b_u32_ni, a_u32_ni);
}

// ──────────────────────────────────────────────────────────────────────────────
// Apply: overwrite
// ──────────────────────────────────────────────────────────────────────────────

/// Pointwise DFT-domain multiply: `res = a ⊙ b`.
///
/// For each active limb `j` and each NTT coefficient index `n_i`:
/// ```text
/// res[res_col, j, n_i]  =  a[a_col, n_i]  ×  b[b_col, j, n_i]   (mod Q)
/// ```
/// Limbs of `res` beyond `b.size()` are zeroed.
///
/// `a`: prepared [`SvpPPol`] in q120c format.
/// `b`: input [`VecZnxDft`] in q120b format.
/// `res`: output [`VecZnxDft`] in q120b format.
pub fn ntt120_svp_apply_dft_to_dft<R, A, C, BE>(
    module: &impl NttModuleHandle,
    res: &mut R,
    res_col: usize,
    a: &A,
    a_col: usize,
    b: &C,
    b_col: usize,
) where
    BE: Backend<ScalarPrep = Q120bScalar>,
    R: VecZnxDftToMut<BE>,
    A: SvpPPolToRef<BE>,
    C: VecZnxDftToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: SvpPPol<&[u8], BE> = a.to_ref();
    let b: VecZnxDft<&[u8], BE> = b.to_ref();

    let meta = module.get_bbc_meta();
    let n = res.n();
    let res_size = res.size();
    let b_size = b.size();
    let min_size = res_size.min(b_size);

    // q120c view of the prepared polynomial (constant across all limbs).
    let a_u32: &[u32] = cast_slice(a.at(a_col, 0));

    // Active limbs: pointwise multiply.
    for j in 0..min_size {
        let res_u64: &mut [u64] = cast_slice_mut(res.at_mut(res_col, j));
        let b_u32: &[u32] = cast_slice(b.at(b_col, j));
        for n_i in 0..n {
            pointwise_mul_bbc(
                meta,
                &mut res_u64[4 * n_i..4 * n_i + 4],
                &b_u32[8 * n_i..8 * n_i + 8],
                &a_u32[8 * n_i..8 * n_i + 8],
            );
        }
    }

    // Remaining limbs: zero.
    for j in min_size..res_size {
        for x in cast_slice_mut::<_, u64>(res.at_mut(res_col, j)).iter_mut() {
            *x = 0;
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Apply: accumulate
// ──────────────────────────────────────────────────────────────────────────────

/// Pointwise DFT-domain multiply-accumulate: `res += a ⊙ b`.
///
/// For each active limb `j` and each NTT coefficient index `n_i`:
/// ```text
/// res[res_col, j, n_i]  +=  a[a_col, n_i]  ×  b[b_col, j, n_i]   (mod Q, lazy)
/// ```
/// Addition uses the Q120b lazy-reduction bound `Q[k] << 33`.
/// Limbs of `res` beyond `b.size()` are zeroed (not accumulated).
pub fn ntt120_svp_apply_dft_to_dft_add<R, A, C, BE>(
    module: &impl NttModuleHandle,
    res: &mut R,
    res_col: usize,
    a: &A,
    a_col: usize,
    b: &C,
    b_col: usize,
) where
    BE: Backend<ScalarPrep = Q120bScalar>,
    R: VecZnxDftToMut<BE>,
    A: SvpPPolToRef<BE>,
    C: VecZnxDftToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: SvpPPol<&[u8], BE> = a.to_ref();
    let b: VecZnxDft<&[u8], BE> = b.to_ref();

    let meta = module.get_bbc_meta();
    let n = res.n();
    let res_size = res.size();
    let b_size = b.size();
    let min_size = res_size.min(b_size);

    let a_u32: &[u32] = cast_slice(a.at(a_col, 0));

    for j in 0..min_size {
        let res_u64: &mut [u64] = cast_slice_mut(res.at_mut(res_col, j));
        let b_u32: &[u32] = cast_slice(b.at(b_col, j));
        let mut product = [0u64; 4];
        for n_i in 0..n {
            pointwise_mul_bbc(meta, &mut product, &b_u32[8 * n_i..8 * n_i + 8], &a_u32[8 * n_i..8 * n_i + 8]);
            for k in 0..4 {
                let idx = 4 * n_i + k;
                res_u64[idx] = res_u64[idx] % Q_SHIFTED[k] + product[k] % Q_SHIFTED[k];
            }
        }
    }

    // Limbs beyond b.size(): zero out (clear any stale data).
    for j in min_size..res_size {
        for x in cast_slice_mut::<_, u64>(res.at_mut(res_col, j)).iter_mut() {
            *x = 0;
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Apply: in-place overwrite
// ──────────────────────────────────────────────────────────────────────────────

/// Pointwise DFT-domain multiply in place: `res = a ⊙ res`.
///
/// For each active limb `j` and each NTT coefficient index `n_i`:
/// ```text
/// res[res_col, j, n_i]  =  a[a_col, n_i]  ×  res[res_col, j, n_i]   (mod Q)
/// ```
///
/// Processes each q120b coefficient by copying it (since [`Q120bScalar`] is
/// `Copy`) before overwriting to avoid aliasing conflicts.
pub fn ntt120_svp_apply_dft_to_dft_inplace<R, A, BE>(
    module: &impl NttModuleHandle,
    res: &mut R,
    res_col: usize,
    a: &A,
    a_col: usize,
) where
    BE: Backend<ScalarPrep = Q120bScalar>,
    R: VecZnxDftToMut<BE>,
    A: SvpPPolToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: SvpPPol<&[u8], BE> = a.to_ref();

    let meta = module.get_bbc_meta();
    let n = res.n();
    let res_size = res.size();

    // Borrow a's q120c data once; it is valid for the entire loop.
    let a_u32: &[u32] = cast_slice(a.at(a_col, 0));

    for j in 0..res_size {
        let res_slice: &mut [Q120bScalar] = res.at_mut(res_col, j);
        let mut product = [0u64; 4];
        for n_i in 0..n {
            // Copy the coefficient (Q120bScalar is Copy) so we can reborrow res_slice.
            let x_elem: Q120bScalar = res_slice[n_i];
            let x_u32: &[u32] = cast_slice(std::slice::from_ref(&x_elem));
            pointwise_mul_bbc(meta, &mut product, x_u32, &a_u32[8 * n_i..8 * n_i + 8]);
            res_slice[n_i] = Q120bScalar(product);
        }
    }
}
