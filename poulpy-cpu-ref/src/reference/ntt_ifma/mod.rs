//! IFMA NTT reference implementation.
//!
//! This module provides a 3-prime CRT NTT backend using ~42-bit primes
//! designed for hardware acceleration via AVX512-IFMA52 instructions.
//! The composite modulus `Q = Q₀·Q₁·Q₂ ≈ 2^126` matches the existing
//! NTT120 backend's modulus size but uses 3 larger primes instead of 4
//! smaller ones, yielding a 25% reduction in per-coefficient work.
//!
//! # Representation
//!
//! Ring elements are stored in CRT form with three residues per coefficient.
//! Each coefficient occupies 4 × u64 (32 bytes) — three active residues
//! plus one padding lane for SIMD alignment — reusing the same
//! [`Q120bScalar`](super::ntt120::types::Q120bScalar) type as NTT120.
//!
//! The Harvey butterfly replaces the split-precomputed multiplication used
//! in NTT120: instead of packing two 32-bit halves into a u64 and using
//! `_mm256_mul_epu32`, twiddle factors store `(ω, ⌊ω·2^52/q⌋)` pairs
//! consumed by `_mm256_madd52lo/hi_epu64`.
//!
//! # Submodules
//!
//! - [`primes`]: [`PrimeSetIfma`] trait and [`Primes42`] implementation.
//! - [`types`]: Scalar type aliases and lazy-reduction constants.
//! - [`ntt`]: NTT precomputation tables and reference execution.
//! - [`arithmetic`]: Element-wise CRT conversions (i64 ↔ CRT, CRT → prepared).
//! - [`mat_vec`]: Lazy-accumulation matrix–vector product metadata and kernels.
//!
//! # Trait overview
//!
//! | Trait | Description |
//! |-------|-------------|
//! | [`NttIfmaDFTExecute`] | Forward or inverse NTT execution |
//! | [`NttIfmaFromZnx64`] | Load `i64` coefficients into 3-prime CRT format |
//! | [`NttIfmaToZnx128`] | CRT-reconstruct from 3-prime CRT to `i128` |
//! | [`NttIfmaAdd`] | Component-wise addition |
//! | [`NttIfmaAddAssign`] | In-place component-wise addition |
//! | [`NttIfmaSub`] | Component-wise subtraction |
//! | [`NttIfmaSubAssign`] | In-place component-wise subtraction |
//! | [`NttIfmaSubNegateAssign`] | In-place swap-subtract: `res = a - res` |
//! | [`NttIfmaNegate`] | Component-wise negation |
//! | [`NttIfmaNegateAssign`] | In-place negation |
//! | [`NttIfmaZero`] | Zero a CRT vector |
//! | [`NttIfmaCopy`] | Copy a CRT vector |
//! | [`NttIfmaMulBbc`] | Pointwise product: b × c → b |
//! | [`NttIfmaCFromB`] | Convert b → c (Harvey-prepared form) |
//! | [`NttIfmaMulBbc1ColX2`] | x2-block 1-column bbc product |
//! | [`NttIfmaMulBbc2ColsX2`] | x2-block 2-column bbc product |
//! | [`NttIfmaExtract1BlkContiguous`] | Extract one x2-block from contiguous array |

pub mod arithmetic;
pub mod convolution;
pub mod mat_vec;
pub mod ntt;
pub mod primes;
pub mod svp;
pub mod types;
pub mod vec_znx_dft;
pub mod vmp;

pub use arithmetic::*;
pub use convolution::*;
pub use mat_vec::*;
pub use ntt::*;
pub use primes::*;
pub use types::*;
pub use vec_znx_dft::*;
pub use vmp::*;

// ──────────────────────────────────────────────────────────────────────────────
// NTT-domain operation traits (3-prime IFMA variant)
// ──────────────────────────────────────────────────────────────────────────────

/// Execute a forward or inverse NTT using a precomputed table.
pub trait NttIfmaDFTExecute<Table> {
    fn ntt_ifma_dft_execute(table: &Table, data: &mut [u64]);
}

/// Load a polynomial from i64 coefficients into 3-prime CRT format.
///
/// `res` has length `4 * a.len()` (3 active residues + 1 padding per coefficient).
pub trait NttIfmaFromZnx64 {
    fn ntt_ifma_from_znx64(res: &mut [u64], a: &[i64]);

    fn ntt_ifma_from_znx64_masked(res: &mut [u64], a: &[i64], mask: i64) {
        self::arithmetic::b_ifma_from_znx64_masked_ref(a.len(), res, a, mask)
    }
}

/// Recover `i128` coefficients from 3-prime CRT format via Garner's algorithm.
pub trait NttIfmaToZnx128 {
    fn ntt_ifma_to_znx128(res: &mut [i128], divisor_is_n: usize, a: &[u64]);
}

/// Component-wise addition of two CRT vectors.
pub trait NttIfmaAdd {
    fn ntt_ifma_add(res: &mut [u64], a: &[u64], b: &[u64]);
}

/// In-place component-wise addition.
pub trait NttIfmaAddAssign {
    fn ntt_ifma_add_assign(res: &mut [u64], a: &[u64]);
}

/// Component-wise subtraction.
pub trait NttIfmaSub {
    fn ntt_ifma_sub(res: &mut [u64], a: &[u64], b: &[u64]);
}

/// In-place component-wise subtraction.
pub trait NttIfmaSubAssign {
    fn ntt_ifma_sub_assign(res: &mut [u64], a: &[u64]);
}

/// In-place swap-subtract: `res = a - res`.
pub trait NttIfmaSubNegateAssign {
    fn ntt_ifma_sub_negate_assign(res: &mut [u64], a: &[u64]);
}

/// Component-wise negation.
pub trait NttIfmaNegate {
    fn ntt_ifma_negate(res: &mut [u64], a: &[u64]);
}

/// In-place negation.
pub trait NttIfmaNegateAssign {
    fn ntt_ifma_negate_assign(res: &mut [u64]);
}

/// Zero a CRT vector.
pub trait NttIfmaZero {
    fn ntt_ifma_zero(res: &mut [u64]);
}

/// Copy a CRT vector.
pub trait NttIfmaCopy {
    fn ntt_ifma_copy(res: &mut [u64], a: &[u64]);
}

/// Pointwise product: b × c → b (overwrite).
///
/// `ntt_coeff` is in b format (as u32 view), `prepared` is in Harvey-prepared c format.
pub trait NttIfmaMulBbc {
    fn ntt_ifma_mul_bbc(meta: &BbcIfmaMeta<Primes42>, ell: usize, res: &mut [u64], ntt_coeff: &[u32], prepared: &[u32]);
}

/// Convert b → c (Harvey-prepared form).
pub trait NttIfmaCFromB {
    fn ntt_ifma_c_from_b(n: usize, res: &mut [u32], a: &[u64]);
}

/// x2-block 1-column bbc product.
pub trait NttIfmaMulBbc1ColX2 {
    fn ntt_ifma_mul_bbc_1col_x2(meta: &BbcIfmaMeta<Primes42>, ell: usize, res: &mut [u64], a: &[u32], b: &[u32]);
}

/// x2-block 2-column bbc product.
pub trait NttIfmaMulBbc2ColsX2 {
    fn ntt_ifma_mul_bbc_2cols_x2(meta: &BbcIfmaMeta<Primes42>, ell: usize, res: &mut [u64], a: &[u32], b: &[u32]);
}

/// Extract one x2-block from contiguous array.
pub trait NttIfmaExtract1BlkContiguous {
    fn ntt_ifma_extract_1blk_contiguous(n: usize, row_max: usize, blk: usize, dst: &mut [u64], src: &[u64]);
}
