//! Bivariate convolution operations for the NTT120 backend.
//!
//! Implements the five-function convolution pipeline used by
//! `poulpy-cpu-ref`:
//!
//! | Step | Function | Description |
//! |------|----------|-------------|
//! | 1a | [`ntt120_cnv_prepare_left`]  | Encode `VecZnx` → `CnvPVecL` (q120b, NTT domain) |
//! | 1b | [`ntt120_cnv_prepare_right`] | Encode `VecZnx` → `CnvPVecR` (q120c, NTT domain) |
//! | 2  | [`ntt120_cnv_apply_dft`]     | `res[k] = Σ a[j] ⊙ b[k−j]` (bbc product) |
//! | 2p | [`ntt120_cnv_pairwise_apply_dft`] | `res = a[:,i]⊙b[:,i] + a[:,j]⊙b[:,j]` |
//! | 3  | [`ntt120_cnv_by_const_apply`] | Coefficient-domain negacyclic convolution into i128 |
//!
//! # Prepared-format asymmetry
//!
//! The bbc kernel (`accum_mul_q120_bc` in `mat_vec`) expects its
//! left operand in **q120b** (4 × u64 per NTT coefficient) and its right
//! operand in **q120c** (8 × u32: `(r mod Qₖ, r·2³² mod Qₖ)` per prime).
//! `CnvPVecL` stores q120b; `CnvPVecR` stores q120c.  Both are 32 bytes
//! per NTT coefficient — the same as `size_of::<Q120bScalar>()`.
//!
//! # Memory layout (Option A)
//!
//! Both `CnvPVecL` and `CnvPVecR` use the same flat layout as
//! `vec_znx_dft`: for column `col`, limb `j`, and NTT
//! coefficient index `n_i`, the element lives at
//! `(col * size + j) * n + n_i` in [`Q120bScalar`] units.  Access via
//! [`ZnxView::at`] / `at_mut` is identical
//! to `VecZnxDft`.

use bytemuck::{cast_slice, cast_slice_mut};

use crate::{
    layouts::{
        Backend, CnvPVecL, CnvPVecLToMut, CnvPVecLToRef, CnvPVecR, CnvPVecRToMut, CnvPVecRToRef, VecZnx, VecZnxBig,
        VecZnxBigToMut, VecZnxDft, VecZnxDftToMut, VecZnxToRef, ZnxInfos, ZnxView, ZnxViewMut,
    },
    reference::ntt120::{
        arithmetic::{b_from_znx64_ref, c_from_b_ref},
        mat_vec::{accum_mul_q120_bc, accum_to_q120b},
        ntt::ntt_ref,
        primes::{PrimeSet, Primes30},
        types::Q120bScalar,
        vec_znx_dft::NttModuleHandle,
    },
};

/// Lazy-reduction bound used when adding two q120b values pointwise.
///
/// `Q_SHIFTED[k] = Q[k] << 33`.  Any q120b residue produced by
/// [`accum_to_q120b`] satisfies `x < 2·Q_SHIFTED[k]`, so reducing
/// modulo `Q_SHIFTED[k]` before adding two such values keeps the result
/// below `4·Q_SHIFTED[k]`, which is safe for a subsequent NTT.
const Q_SHIFTED: [u64; 4] = [
    (Primes30::Q[0] as u64) << 33,
    (Primes30::Q[1] as u64) << 33,
    (Primes30::Q[2] as u64) << 33,
    (Primes30::Q[3] as u64) << 33,
];

// ──────────────────────────────────────────────────────────────────────────────
// Prepare left  (VecZnx → CnvPVecL, q120b)
// ──────────────────────────────────────────────────────────────────────────────

/// Scratch bytes required by [`ntt120_cnv_prepare_left`].
///
/// Returns 0: the function writes the NTT directly into the output buffer.
pub fn ntt120_cnv_prepare_left_tmp_bytes(_n: usize) -> usize {
    0
}

/// Encode a `VecZnx` (i64 coefficients) into a `CnvPVecL` (q120b, NTT domain).
///
/// For each column `col` and each limb `j` of the input `a`:
/// 1. Map i64 coefficients → q120b via [`b_from_znx64_ref`].
/// 2. Apply the forward NTT in-place via [`ntt_ref`].
/// 3. Store the result directly in `res[col, j]` as q120b.
///
/// Limbs of `res` beyond `a.size()` are zeroed.
/// No scratch buffer is needed; `_tmp` is unused.
pub fn ntt120_cnv_prepare_left<R, A, BE>(module: &impl NttModuleHandle, res: &mut R, a: &A, _tmp: &mut [u8])
where
    BE: Backend<ScalarPrep = Q120bScalar>,
    R: CnvPVecLToMut<BE>,
    A: VecZnxToRef,
{
    let mut res: CnvPVecL<&mut [u8], BE> = res.to_mut();
    let a: VecZnx<&[u8]> = a.to_ref();
    let n = res.n();
    let table = module.get_ntt_table();
    let cols = res.cols();
    let res_size = res.size();
    let min_size = res_size.min(a.size());

    for col in 0..cols {
        for j in 0..min_size {
            let res_u64: &mut [u64] = cast_slice_mut(res.at_mut(col, j));
            b_from_znx64_ref::<Primes30>(n, res_u64, a.at(col, j));
            ntt_ref(table, res_u64);
        }
        for j in min_size..res_size {
            for x in cast_slice_mut::<_, u64>(res.at_mut(col, j)).iter_mut() {
                *x = 0;
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Prepare right  (VecZnx → CnvPVecR, q120c)
// ──────────────────────────────────────────────────────────────────────────────

/// Scratch bytes required by [`ntt120_cnv_prepare_right`].
///
/// Returns `4 * n * 8` bytes — one intermediate q120b buffer of `4 * n` u64
/// values used for the NTT before the q120b → q120c conversion.
pub fn ntt120_cnv_prepare_right_tmp_bytes(n: usize) -> usize {
    4 * n * size_of::<u64>()
}

/// Encode a `VecZnx` (i64 coefficients) into a `CnvPVecR` (q120c, NTT domain).
///
/// For each column `col` and each limb `j` of the input `a`:
/// 1. Map i64 coefficients → q120b via [`b_from_znx64_ref`] into `tmp`.
/// 2. Apply the forward NTT in-place via [`ntt_ref`].
/// 3. Convert q120b → q120c via [`c_from_b_ref`] into `res[col, j]`.
///
/// `tmp` must hold at least [`ntt120_cnv_prepare_right_tmp_bytes`]`(n)` bytes,
/// properly aligned to `u64` (8 bytes).
/// Limbs of `res` beyond `a.size()` are zeroed.
pub fn ntt120_cnv_prepare_right<R, A, BE>(module: &impl NttModuleHandle, res: &mut R, a: &A, tmp: &mut [u8])
where
    BE: Backend<ScalarPrep = Q120bScalar>,
    R: CnvPVecRToMut<BE>,
    A: VecZnxToRef,
{
    let mut res: CnvPVecR<&mut [u8], BE> = res.to_mut();
    let a: VecZnx<&[u8]> = a.to_ref();
    let n = res.n();
    let table = module.get_ntt_table();
    let tmp_u64: &mut [u64] = cast_slice_mut(tmp);
    let cols = res.cols();
    let res_size = res.size();
    let min_size = res_size.min(a.size());

    for col in 0..cols {
        for j in 0..min_size {
            b_from_znx64_ref::<Primes30>(n, tmp_u64, a.at(col, j));
            ntt_ref(table, tmp_u64);
            let res_u32: &mut [u32] = cast_slice_mut(res.at_mut(col, j));
            c_from_b_ref::<Primes30>(n, res_u32, tmp_u64);
        }
        for j in min_size..res_size {
            for x in cast_slice_mut::<_, u32>(res.at_mut(col, j)).iter_mut() {
                *x = 0;
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Apply DFT  (CnvPVecL × CnvPVecR → VecZnxDft, bbc product)
// ──────────────────────────────────────────────────────────────────────────────

/// Scratch bytes required by [`ntt120_cnv_apply_dft`].
///
/// Returns 0: accumulators are kept on the stack.
pub fn ntt120_cnv_apply_dft_tmp_bytes(_res_size: usize, _a_size: usize, _b_size: usize) -> usize {
    0
}

/// Compute the DFT-domain bivariate convolution `res[k] = Σ a[j] ⊙ b[k−j]`.
///
/// For each output limb `k ∈ [0, min_size)` and each NTT coefficient `n_i`:
///
/// ```text
/// res[res_col, k, n_i] = Σ_{j=j_min}^{j_max-1}  bbc( a[a_col, k_abs−j, n_i],
///                                                      b[b_col,       j, n_i] )
/// ```
///
/// where `k_abs = k + res_offset`, `j_min = max(0, k_abs − a.size() + 1)`,
/// `j_max = min(k_abs + 1, b.size())`, and `bbc` denotes the
/// `accum_mul_q120_bc` + `accum_to_q120b` product.
///
/// Output limbs `min_size..res.size()` are zeroed.
/// `_tmp` is unused (accumulators live on the stack).
#[allow(clippy::too_many_arguments)]
pub fn ntt120_cnv_apply_dft<R, A, B, BE>(
    module: &impl NttModuleHandle,
    res: &mut R,
    res_offset: usize,
    res_col: usize,
    a: &A,
    a_col: usize,
    b: &B,
    b_col: usize,
    _tmp: &mut [u8],
) where
    BE: Backend<ScalarPrep = Q120bScalar>,
    R: VecZnxDftToMut<BE>,
    A: CnvPVecLToRef<BE>,
    B: CnvPVecRToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: CnvPVecL<&[u8], BE> = a.to_ref();
    let b: CnvPVecR<&[u8], BE> = b.to_ref();

    let meta = module.get_bbc_meta();
    let n = res.n();
    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();

    let bound = a_size + b_size - 1;
    let min_size = res_size.min(bound);
    let offset = res_offset.min(bound);

    for k in 0..min_size {
        let k_abs = k + offset;
        let j_min = k_abs.saturating_sub(a_size - 1);
        let j_max = (k_abs + 1).min(b_size);

        let res_u64: &mut [u64] = cast_slice_mut(res.at_mut(res_col, k));

        for n_i in 0..n {
            let mut s = [0u64; 8];
            for j in j_min..j_max {
                // SAFETY: a.at and b.at return slices of length n, and 8*n_i+8 <= 8*n.
                let ai: &[u32; 8] =
                    unsafe { &*(cast_slice::<_, u32>(a.at(a_col, k_abs - j))[8 * n_i..].as_ptr() as *const [u32; 8]) };
                let bi: &[u32; 8] = unsafe { &*(cast_slice::<_, u32>(b.at(b_col, j))[8 * n_i..].as_ptr() as *const [u32; 8]) };
                accum_mul_q120_bc(&mut s, ai, bi);
            }
            let mut r4 = [0u64; 4];
            accum_to_q120b::<Primes30>(&mut r4, &s, meta);
            res_u64[4 * n_i..4 * n_i + 4].copy_from_slice(&r4);
        }
    }

    for j in min_size..res_size {
        for x in cast_slice_mut::<_, u64>(res.at_mut(res_col, j)).iter_mut() {
            *x = 0;
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// By-const apply  (VecZnx × &[i64] → VecZnxBig, coefficient domain)
// ──────────────────────────────────────────────────────────────────────────────

/// Scratch bytes required by [`ntt120_cnv_by_const_apply`].
///
/// Returns 0: the function uses `i128` stack accumulators.
pub fn ntt120_cnv_by_const_apply_tmp_bytes(_res_size: usize, _a_size: usize, _b_size: usize) -> usize {
    0
}

/// Coefficient-domain negacyclic convolution: `res[k] = Σ a[k_abs−j] * b[j]`.
///
/// Unlike [`ntt120_cnv_apply_dft`], this function operates entirely in
/// the **coefficient domain** (no NTT).  Each output limb is computed as
/// an `i128` inner product, suitable for accumulation into a
/// [`VecZnxBig`] with `ScalarBig = i128`.
///
/// For each output limb `k ∈ [0, min_size)` and ring coefficient `n_i`:
///
/// ```text
/// res[res_col, k, n_i] = Σ_{j=j_min}^{j_max-1}  a[a_col, k_abs−j, n_i]  ×  b[j]
/// ```
///
/// where `k_abs = k + res_offset`.
/// Output limbs `min_size..res.size()` are zeroed.
/// `_tmp` is unused.
#[allow(clippy::too_many_arguments)]
pub fn ntt120_cnv_by_const_apply<R, A, BE>(
    res: &mut R,
    res_offset: usize,
    res_col: usize,
    a: &A,
    a_col: usize,
    b: &[i64],
    _tmp: &mut [u8],
) where
    BE: Backend<ScalarPrep = Q120bScalar, ScalarBig = i128>,
    R: VecZnxBigToMut<BE>,
    A: VecZnxToRef,
{
    let mut res: VecZnxBig<&mut [u8], BE> = res.to_mut();
    let a: VecZnx<&[u8]> = a.to_ref();
    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.len();

    let bound = a_size + b_size - 1;
    let min_size = res_size.min(bound);
    let offset = res_offset.min(bound);

    for k in 0..min_size {
        let k_abs = k + offset;
        let j_min = k_abs.saturating_sub(a_size - 1);
        let j_max = (k_abs + 1).min(b_size);
        let res_limb: &mut [i128] = res.at_mut(res_col, k);
        for (n_i, r) in res_limb.iter_mut().enumerate() {
            let mut acc: i128 = 0;
            for (j, &b_j) in b.iter().enumerate().take(j_max).skip(j_min) {
                acc += a.at(a_col, k_abs - j)[n_i] as i128 * b_j as i128;
            }
            *r = acc;
        }
    }

    for j in min_size..res_size {
        for x in res.at_mut(res_col, j).iter_mut() {
            *x = 0i128;
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Pairwise apply DFT
// ──────────────────────────────────────────────────────────────────────────────

/// Scratch bytes required by [`ntt120_cnv_pairwise_apply_dft`].
///
/// Returns 0: dual accumulators are kept on the stack.
pub fn ntt120_cnv_pairwise_apply_dft_tmp_bytes(_res_size: usize, _a_size: usize, _b_size: usize) -> usize {
    0
}

/// Compute the sum of two DFT-domain convolutions:
/// `res = a[:,col_i] ⊙ b[:,col_i]  +  a[:,col_j] ⊙ b[:,col_j]`.
///
/// When `col_i == col_j` this delegates to [`ntt120_cnv_apply_dft`].
///
/// Otherwise, for each output limb `k` and NTT coefficient `n_i`, two
/// independent `accum_mul_q120_bc` accumulations are run in parallel.
/// Their q120b results are added with the lazy-reduction bound
/// `Q_SHIFTED[k] = Q[k] << 33` to keep residues in range:
///
/// ```text
/// res[res_col, k, n_i] = (r0[crt] % Q_SHIFTED[crt]) + (r1[crt] % Q_SHIFTED[crt])
/// ```
///
/// Output limbs `min_size..res.size()` are zeroed.
/// `_tmp` is unused.
#[allow(clippy::too_many_arguments)]
pub fn ntt120_cnv_pairwise_apply_dft<R, A, B, BE>(
    module: &impl NttModuleHandle,
    res: &mut R,
    res_offset: usize,
    res_col: usize,
    a: &A,
    b: &B,
    col_i: usize,
    col_j: usize,
    _tmp: &mut [u8],
) where
    BE: Backend<ScalarPrep = Q120bScalar>,
    R: VecZnxDftToMut<BE>,
    A: CnvPVecLToRef<BE>,
    B: CnvPVecRToRef<BE>,
{
    if col_i == col_j {
        ntt120_cnv_apply_dft(module, res, res_offset, res_col, a, col_i, b, col_j, &mut []);
        return;
    }

    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: CnvPVecL<&[u8], BE> = a.to_ref();
    let b: CnvPVecR<&[u8], BE> = b.to_ref();

    let meta = module.get_bbc_meta();
    let n = res.n();
    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();

    let bound = a_size + b_size - 1;
    let min_size = res_size.min(bound);
    let offset = res_offset.min(bound);

    for k in 0..min_size {
        let k_abs = k + offset;
        let j_min = k_abs.saturating_sub(a_size - 1);
        let j_max = (k_abs + 1).min(b_size);

        let res_u64: &mut [u64] = cast_slice_mut(res.at_mut(res_col, k));

        for n_i in 0..n {
            // Compute (a[col_i] + a[col_j]) ⊙ (b[col_i] + b[col_j]):
            // sum operands first, then one bbc product per j.
            //
            // q120b values (a side) can be full 64-bit after NTT butterflies.
            // Reconstruct each u64 residue from its two u32 halves, reduce mod Q_k,
            // then form a_sum with hi = 0 (sum < 2*Q_k < 2^31 fits in one u32).
            //
            // q120c values (b side) come from c_from_b_ref so each entry < Q_k < 2^30.
            // The u32 sums stay below 2*Q_k < 2^31 and do not overflow.
            let mut s = [0u64; 8];
            for j in j_min..j_max {
                // SAFETY: at() slices have length n, and 8*n_i+8 <= 8*n.
                let ai: &[u32; 8] =
                    unsafe { &*(cast_slice::<_, u32>(a.at(col_i, k_abs - j))[8 * n_i..].as_ptr() as *const [u32; 8]) };
                let aj: &[u32; 8] =
                    unsafe { &*(cast_slice::<_, u32>(a.at(col_j, k_abs - j))[8 * n_i..].as_ptr() as *const [u32; 8]) };
                let bi: &[u32; 8] = unsafe { &*(cast_slice::<_, u32>(b.at(col_i, j))[8 * n_i..].as_ptr() as *const [u32; 8]) };
                let bj: &[u32; 8] = unsafe { &*(cast_slice::<_, u32>(b.at(col_j, j))[8 * n_i..].as_ptr() as *const [u32; 8]) };
                let mut a_sum = [0u32; 8];
                let mut b_sum = [0u32; 8];
                for k in 0..4 {
                    let q = Primes30::Q[k] as u64;
                    // Reconstruct the full u64 residue from the two u32 halves.
                    let ai_k = (ai[2 * k] as u64) | ((ai[2 * k + 1] as u64) << 32);
                    let aj_k = (aj[2 * k] as u64) | ((aj[2 * k + 1] as u64) << 32);
                    // Reduce mod Q_k so the sum fits in a u32 (hi stays 0).
                    a_sum[2 * k] = ((ai_k % q) + (aj_k % q)) as u32;
                    b_sum[2 * k] = bi[2 * k] + bj[2 * k];
                    b_sum[2 * k + 1] = bi[2 * k + 1] + bj[2 * k + 1];
                }
                accum_mul_q120_bc(&mut s, &a_sum, &b_sum);
            }
            let mut r = [0u64; 4];
            accum_to_q120b::<Primes30>(&mut r, &s, meta);
            res_u64[4 * n_i..4 * n_i + 4].copy_from_slice(&r);
        }
    }

    for j in min_size..res_size {
        for x in cast_slice_mut::<_, u64>(res.at_mut(res_col, j)).iter_mut() {
            *x = 0;
        }
    }
}
