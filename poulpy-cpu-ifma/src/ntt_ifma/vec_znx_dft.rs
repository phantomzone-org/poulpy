//! NTT-domain SIMD helpers for [`NTTIfma`](crate::NTTIfma).
//!
//! This module contains:
//!
//! - AVX512-IFMA/VL SIMD Garner CRT reconstruction helpers used by the consume path.
//! - The `vec_znx_idft_apply_consume` entry point called by the `hal_impl` macro.

use bytemuck::cast_slice_mut;
use poulpy_cpu_ref::reference::ntt_ifma::{
    ntt::NttIfmaTableInv,
    primes::{PrimeSetIfma, Primes40},
    vec_znx_dft::NttIfmaModuleHandle,
};
use poulpy_hal::layouts::{Data, Module, VecZnxBig, VecZnxDft, VecZnxDftToMut, ZnxInfos, ZnxViewMut};

use super::ntt_ifma_avx512::{cond_sub_2q_si256, harvey_modmul_si256, intt_ifma_avx512};

use core::arch::x86_64::{
    __m256i, _mm256_add_epi64, _mm256_loadu_si256, _mm256_permute2x128_si256, _mm256_set_epi64x, _mm256_set1_epi64x,
    _mm256_storeu_si256, _mm256_sub_epi64, _mm256_unpackhi_epi64, _mm256_unpacklo_epi64,
};

// ──────────────────────────────────────────────────────────────────────────────
// In-place 3-prime CRT -> i128 compaction helper (Garner's algorithm)
// ──────────────────────────────────────────────────────────────────────────────

const Q: [u64; 3] = Primes40::Q;
const INV01: u64 = Primes40::CRT_CST[0];
const INV012: u64 = Primes40::CRT_CST[1];
const Q0: u64 = Q[0];
const Q1: u64 = Q[1];
const Q2: u64 = Q[2];
const Q01: u128 = Q0 as u128 * Q1 as u128;
const BIG_Q: u128 = Q01 * Q2 as u128;
const HALF_BIG_Q: u128 = BIG_Q / 2;

// Harvey quotient for Garner step 2: floor(INV01 * 2^52 / Q1)
const INV01_QUOT: u64 = ((INV01 as u128 * (1u128 << 52)) / Q1 as u128) as u64;
// Harvey quotient for Garner step 3: floor(INV012 * 2^52 / Q2)
const INV012_QUOT: u64 = ((INV012 as u128 * (1u128 << 52)) / Q2 as u128) as u64;
// Q0 mod Q2 and its Harvey quotient (for computing v1*Q0 mod Q2)
const Q0_MOD_Q2: u64 = Q0 % Q2;
const Q0_MOD_Q2_QUOT: u64 = ((Q0_MOD_Q2 as u128 * (1u128 << 52)) / Q2 as u128) as u64;

/// Harvey scalar modular multiply: `(a * omega) mod q`, result in `[0, q)`.
///
/// Input: `a ∈ [0, q)`, `omega ∈ [0, q)`.
/// `omega_quot = floor(omega * 2^52 / q)`.
#[inline(always)]
fn harvey_modmul_scalar(a: u64, omega: u64, omega_quot: u64, q: u64) -> u64 {
    let qhat = ((a as u128 * omega_quot as u128) >> 52) as u64;
    let product_lo = (a as u128 * omega as u128) as u64;
    let qhat_times_q = (qhat as u128 * q as u128) as u64;
    let mut r = product_lo.wrapping_sub(qhat_times_q);
    if (r as i64) < 0 {
        r = r.wrapping_add(q);
    }
    if r >= q { r - q } else { r }
}

/// Conditional subtract: if x >= q, return x - q.
#[inline(always)]
fn cond_sub_scalar(x: u64, q: u64) -> u64 {
    if x >= q { x - q } else { x }
}

/// SIMD-assisted single-coefficient Garner CRT reconstruction.
///
/// Loads one `__m256i` from `src` (3 residues in `[0, 2q)` + padding), reduces
/// to `[0, q)` in SIMD, then performs scalar Harvey Garner reconstruction.
/// Returns the reconstructed `i128` in symmetric representation `(-Q/2, Q/2]`.
///
/// # Safety
///
/// - `src` must be valid for reading 4 × u64 (one `__m256i`).
/// - Caller must ensure AVX512-VL support.
#[target_feature(enable = "avx512vl")]
pub(crate) unsafe fn garner_crt_single(src: *const u64, q_vec: __m256i) -> i128 {
    unsafe {
        // Load and reduce [0, 2q) -> [0, q) in SIMD
        let xv = _mm256_loadu_si256(src as *const __m256i);
        let reduced = cond_sub_2q_si256(xv, q_vec);

        // Extract reduced residues to scalar registers
        let mut lanes = [0u64; 4];
        _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, reduced);
        let (r0, r1, r2) = (lanes[0], lanes[1], lanes[2]);

        garner_from_residues(r0, r1, r2)
    }
}

/// Scalar Garner CRT reconstruction from 3 reduced residues.
///
/// Input: `r0 ∈ [0, Q0)`, `r1 ∈ [0, Q1)`, `r2 ∈ [0, Q2)`.
/// Output: reconstructed `i128` in symmetric representation `(-Q/2, Q/2]`.
#[inline(always)]
fn garner_from_residues(r0: u64, r1: u64, r2: u64) -> i128 {
    // Garner step 1: v0 = r0
    let v0 = r0;

    // Garner step 2: v1 = ((r1 - v0 mod Q1) * INV01) mod Q1
    let v0_mod_q1 = cond_sub_scalar(v0, Q1);
    let diff1 = cond_sub_scalar(r1 + Q1 - v0_mod_q1, Q1);
    let v1 = harvey_modmul_scalar(diff1, INV01, INV01_QUOT, Q1);

    // Garner step 3: v2 = ((r2 - (v0 + v1*Q0) mod Q2) * INV012) mod Q2
    let v0_mod_q2 = cond_sub_scalar(v0, Q2);
    let v1q0_mod_q2 = harvey_modmul_scalar(v1, Q0_MOD_Q2, Q0_MOD_Q2_QUOT, Q2);
    let partial = cond_sub_scalar(v0_mod_q2 + v1q0_mod_q2, Q2);
    let diff2 = cond_sub_scalar(r2 + Q2 - partial, Q2);
    let v2 = harvey_modmul_scalar(diff2, INV012, INV012_QUOT, Q2);

    // Reconstruct: result = v0 + v1*Q0 + v2*Q0*Q1
    let result_u128 = v0 as u128 + v1 as u128 * Q0 as u128 + v2 as u128 * Q01;

    // Symmetric lift to (-Q/2, Q/2].
    if result_u128 > HALF_BIG_Q {
        result_u128 as i128 - BIG_Q as i128
    } else {
        result_u128 as i128
    }
}

/// Vectorized Garner CRT reconstruction for 4 coefficients in parallel.
///
/// Takes 4 AoS `__m256i` values (each `[r0, r1, r2, 0]` in `[0, q)`),
/// transposes to SoA layout, performs all Garner steps in SIMD, and
/// writes 4 `i128` results.
///
/// # Safety
///
/// - `dst` must be valid for writing 4 × i128 (64 bytes).
/// - All input vectors must have residues in `[0, q)` (already reduced).
/// - Caller must ensure AVX512-IFMA and AVX512-VL support.
#[target_feature(enable = "avx512ifma,avx512vl")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn garner_4coeffs_simd(
    a0: __m256i,
    a1: __m256i,
    a2: __m256i,
    a3: __m256i,
    dst: *mut i128,
    q1_bcast: __m256i,
    q2_bcast: __m256i,
    q2x2_bcast: __m256i,
    inv01_bcast: __m256i,
    inv01_quot_bcast: __m256i,
    inv012_bcast: __m256i,
    inv012_quot_bcast: __m256i,
    q0modq2_bcast: __m256i,
    q0modq2_quot_bcast: __m256i,
) {
    unsafe {
        // ── Transpose AoS → SoA ──────────────────────────────────────────
        // Input:  a0=[a.r0, a.r1, a.r2, 0], a1=[b.r0, b.r1, b.r2, 0], ...
        // Output: vec_r0=[a.r0, b.r0, c.r0, d.r0], vec_r1=[...], vec_r2=[...]
        let ab_lo = _mm256_unpacklo_epi64(a0, a1); // [a.r0, b.r0, a.r2, b.r2]
        let ab_hi = _mm256_unpackhi_epi64(a0, a1); // [a.r1, b.r1, 0, 0]
        let cd_lo = _mm256_unpacklo_epi64(a2, a3); // [c.r0, d.r0, c.r2, d.r2]
        let cd_hi = _mm256_unpackhi_epi64(a2, a3); // [c.r1, d.r1, 0, 0]

        let vec_r0 = _mm256_permute2x128_si256::<0x20>(ab_lo, cd_lo); // [a.r0, b.r0, c.r0, d.r0]
        let vec_r1 = _mm256_permute2x128_si256::<0x20>(ab_hi, cd_hi); // [a.r1, b.r1, c.r1, d.r1]
        let vec_r2 = _mm256_permute2x128_si256::<0x31>(ab_lo, cd_lo); // [a.r2, b.r2, c.r2, d.r2]

        // ── Garner step 1: V0 = R0 ──────────────────────────────────────
        let vec_v0 = vec_r0;

        // ── Garner step 2: V1 = ((R1 - V0 mod Q1) * INV01) mod Q1 ──────
        // v0 mod Q1: V0 < Q0, Q0 > Q1, one cond_sub suffices
        let v0_mod_q1 = cond_sub_2q_si256(vec_v0, q1_bcast);
        // diff1 = (R1 + Q1 - v0_mod_q1) cond_sub Q1
        let diff1_raw = _mm256_sub_epi64(_mm256_add_epi64(vec_r1, q1_bcast), v0_mod_q1);
        let diff1 = cond_sub_2q_si256(diff1_raw, q1_bcast);
        // v1 = diff1 * INV01 mod Q1 via Harvey; result in [0, 2q)
        let vec_v1_lazy = harvey_modmul_si256(diff1, inv01_bcast, inv01_quot_bcast, q1_bcast);
        // Reduce to [0, Q1)
        let vec_v1 = cond_sub_2q_si256(vec_v1_lazy, q1_bcast);

        // ── Garner step 3: V2 = ((R2 - (V0 + V1*Q0) mod Q2) * INV012) mod Q2 ──
        // v0 mod Q2: V0 < Q0, Q0 > Q2, one cond_sub suffices
        let v0_mod_q2 = cond_sub_2q_si256(vec_v0, q2_bcast);
        // v1*Q0 mod Q2: V1 < Q1 < 2*Q2, so V1 is a valid [0, 2q) input for Harvey with modulus Q2
        let v1q0_lazy = harvey_modmul_si256(vec_v1, q0modq2_bcast, q0modq2_quot_bcast, q2_bcast);
        // v1q0_lazy in [0, 2*Q2)
        // partial = (v0_mod_q2 + v1q0_lazy): v0_mod_q2 < Q2, v1q0_lazy < 2*Q2, sum < 3*Q2
        let partial_raw = _mm256_add_epi64(v0_mod_q2, v1q0_lazy);
        // Two conditional subtracts: [0, 3*Q2) -> [0, Q2)
        let partial = cond_sub_2q_si256(cond_sub_2q_si256(partial_raw, q2x2_bcast), q2_bcast);
        // diff2 = (R2 + Q2 - partial) cond_sub Q2
        let diff2_raw = _mm256_sub_epi64(_mm256_add_epi64(vec_r2, q2_bcast), partial);
        let diff2 = cond_sub_2q_si256(diff2_raw, q2_bcast);
        // v2 = diff2 * INV012 mod Q2 via Harvey; result in [0, 2q)
        let vec_v2_lazy = harvey_modmul_si256(diff2, inv012_bcast, inv012_quot_bcast, q2_bcast);
        let vec_v2 = cond_sub_2q_si256(vec_v2_lazy, q2_bcast);

        // ── Extract scalars and compose u128 results ─────────────────────
        let mut v0s = [0u64; 4];
        let mut v1s = [0u64; 4];
        let mut v2s = [0u64; 4];
        _mm256_storeu_si256(v0s.as_mut_ptr() as *mut __m256i, vec_v0);
        _mm256_storeu_si256(v1s.as_mut_ptr() as *mut __m256i, vec_v1);
        _mm256_storeu_si256(v2s.as_mut_ptr() as *mut __m256i, vec_v2);

        for lane in 0..4 {
            let result_u128 = v0s[lane] as u128 + v1s[lane] as u128 * Q0 as u128 + v2s[lane] as u128 * Q01;
            let val: i128 = if result_u128 > HALF_BIG_Q {
                result_u128 as i128 - BIG_Q as i128
            } else {
                result_u128 as i128
            };
            dst.add(lane).write_unaligned(val);
        }
    }
}

/// Vectorized CRT reconstruction: 3-prime IFMA b-format to i128.
///
/// Processes coefficients in batches of 4 using SIMD Garner reconstruction.
/// Falls back to single-coefficient path for the tail.
///
/// Input residues must be in `[0, 2q)` (b-format after iNTT).
///
/// # Safety
///
/// - `a` must contain at least `4 * nn` u64 values.
/// - `res` must have room for at least `nn` i128 values.
/// - Caller must ensure AVX512-IFMA and AVX512-VL support.
#[target_feature(enable = "avx512ifma,avx512vl")]
pub(crate) unsafe fn simd_b_ifma_to_znx128(nn: usize, res: &mut [i128], a: &[u64]) {
    unsafe {
        // Per-prime Q vector for AoS cond_sub
        let q_vec = _mm256_set_epi64x(0, Q2 as i64, Q1 as i64, Q0 as i64);
        // Broadcast constants for SoA Garner
        let q1_bcast = _mm256_set1_epi64x(Q1 as i64);
        let q2_bcast = _mm256_set1_epi64x(Q2 as i64);
        let q2x2_bcast = _mm256_set1_epi64x((2 * Q2) as i64);
        let inv01_bcast = _mm256_set1_epi64x(INV01 as i64);
        let inv01_quot_bcast = _mm256_set1_epi64x(INV01_QUOT as i64);
        let inv012_bcast = _mm256_set1_epi64x(INV012 as i64);
        let inv012_quot_bcast = _mm256_set1_epi64x(INV012_QUOT as i64);
        let q0modq2_bcast = _mm256_set1_epi64x(Q0_MOD_Q2 as i64);
        let q0modq2_quot_bcast = _mm256_set1_epi64x(Q0_MOD_Q2_QUOT as i64);

        let a_ptr = a.as_ptr() as *const __m256i;
        let dst = res.as_mut_ptr();

        // Main loop: 4 coefficients at a time
        let mut c = 0usize;
        while c + 4 <= nn {
            let a0 = cond_sub_2q_si256(_mm256_loadu_si256(a_ptr.add(c)), q_vec);
            let a1 = cond_sub_2q_si256(_mm256_loadu_si256(a_ptr.add(c + 1)), q_vec);
            let a2 = cond_sub_2q_si256(_mm256_loadu_si256(a_ptr.add(c + 2)), q_vec);
            let a3 = cond_sub_2q_si256(_mm256_loadu_si256(a_ptr.add(c + 3)), q_vec);

            garner_4coeffs_simd(
                a0,
                a1,
                a2,
                a3,
                dst.add(c),
                q1_bcast,
                q2_bcast,
                q2x2_bcast,
                inv01_bcast,
                inv01_quot_bcast,
                inv012_bcast,
                inv012_quot_bcast,
                q0modq2_bcast,
                q0modq2_quot_bcast,
            );
            c += 4;
        }

        // Tail: remaining coefficients (0-3)
        while c < nn {
            res[c] = garner_crt_single(a.as_ptr().add(4 * c), q_vec);
            c += 1;
        }
    }
}

/// In-place CRT-compact all NTT blocks from Q120b (32 bytes/coeff) to i128 (16 bytes/coeff).
///
/// For each block `k` in `0..n_blocks`, in order:
///
/// 1. Applies the inverse NTT to the 3-prime CRT block in-place.
/// 2. Uses vectorized Garner reconstruction (4 coefficients at a time via SoA
///    transpose) with Harvey modular multiply to convert residues to i128.
///
/// # Ordering invariant
///
/// Blocks are processed in order `k = 0, 1, ..., n_blocks-1`.  For `k >= 1` the
/// destination `[16nk, 16n(k+1))` never overlaps the source `[32nk, 32n(k+1))`.
/// For `k = 0` all three residues are read before the i128 is written.
///
/// # Safety
///
/// - `u64_ptr` must be valid for reads and writes of at least `4 * n * n_blocks` u64 values.
/// - The backing allocation must be at least 16-byte aligned (guaranteed by `DEFAULTALIGN = 64`).
/// - No other references to the same memory may be live during this call.
#[target_feature(enable = "avx512ifma,avx512vl")]
unsafe fn compact_all_blocks(n: usize, n_blocks: usize, u64_ptr: *mut u64, table: &NttIfmaTableInv<Primes40>) {
    unsafe {
        // Per-prime Q vector for AoS cond_sub: [Q0, Q1, Q2, 0]
        let q_vec = _mm256_set_epi64x(0, Q2 as i64, Q1 as i64, Q0 as i64);
        // Broadcast constants for SoA Garner (loaded once)
        let q1_bcast = _mm256_set1_epi64x(Q1 as i64);
        let q2_bcast = _mm256_set1_epi64x(Q2 as i64);
        let q2x2_bcast = _mm256_set1_epi64x((2 * Q2) as i64);
        let inv01_bcast = _mm256_set1_epi64x(INV01 as i64);
        let inv01_quot_bcast = _mm256_set1_epi64x(INV01_QUOT as i64);
        let inv012_bcast = _mm256_set1_epi64x(INV012 as i64);
        let inv012_quot_bcast = _mm256_set1_epi64x(INV012_QUOT as i64);
        let q0modq2_bcast = _mm256_set1_epi64x(Q0_MOD_Q2 as i64);
        let q0modq2_quot_bcast = _mm256_set1_epi64x(Q0_MOD_Q2_QUOT as i64);

        for k in 0..n_blocks {
            let src_start = 4 * n * k;
            let dst_start = 2 * n * k;

            // Step 1: inverse NTT in-place.
            {
                let blk = std::slice::from_raw_parts_mut(u64_ptr.add(src_start), 4 * n);
                intt_ifma_avx512::<Primes40>(table, blk);
            }

            // Step 2: Garner CRT-compact 4n u64s → n i128s.
            let src_base = u64_ptr.add(src_start) as *const __m256i;
            let dst_base = u64_ptr.add(dst_start) as *mut i128;

            // Main loop: 4 coefficients at a time via SoA Garner
            let mut c = 0usize;
            while c + 4 <= n {
                // Load 4 AoS coefficients and reduce [0, 2q) -> [0, q)
                let a0 = cond_sub_2q_si256(_mm256_loadu_si256(src_base.add(c)), q_vec);
                let a1 = cond_sub_2q_si256(_mm256_loadu_si256(src_base.add(c + 1)), q_vec);
                let a2 = cond_sub_2q_si256(_mm256_loadu_si256(src_base.add(c + 2)), q_vec);
                let a3 = cond_sub_2q_si256(_mm256_loadu_si256(src_base.add(c + 3)), q_vec);

                garner_4coeffs_simd(
                    a0,
                    a1,
                    a2,
                    a3,
                    dst_base.add(c),
                    q1_bcast,
                    q2_bcast,
                    q2x2_bcast,
                    inv01_bcast,
                    inv01_quot_bcast,
                    inv012_bcast,
                    inv012_quot_bcast,
                    q0modq2_bcast,
                    q0modq2_quot_bcast,
                );
                c += 4;
            }

            // Tail: remaining 0-3 coefficients via single-coefficient path
            while c < n {
                let val = garner_crt_single(u64_ptr.add(src_start + 4 * c), q_vec);
                dst_base.add(c).write_unaligned(val);
                c += 1;
            }
        }
    }
}

/// AVX512-accelerated `vec_znx_idft_apply_consume` for [`NTTIfma`](crate::NTTIfma).
///
/// Converts the DFT-domain `VecZnxDft` into a `VecZnxBig` by applying inverse NTT
/// and in-place CRT compaction (q120b 32 bytes/coeff → i128 16 bytes/coeff) for
/// each block, then reinterpreting the buffer.
pub(crate) fn vec_znx_idft_apply_consume<D: Data>(
    module: &Module<crate::NTTIfma>,
    mut a: VecZnxDft<D, crate::NTTIfma>,
) -> VecZnxBig<D, crate::NTTIfma>
where
    VecZnxDft<D, crate::NTTIfma>: VecZnxDftToMut<crate::NTTIfma>,
{
    let table = module.get_intt_ifma_table();

    let (n, n_blocks, u64_ptr) = {
        let mut a_mut: VecZnxDft<&mut [u8], crate::NTTIfma> = a.to_mut();
        let n = a_mut.n();
        let n_blocks = a_mut.cols() * a_mut.size();
        let ptr: *mut u64 = {
            let s = a_mut.raw_mut();
            cast_slice_mut::<_, u64>(s).as_mut_ptr()
        };
        (n, n_blocks, ptr)
    };

    unsafe { compact_all_blocks(n, n_blocks, u64_ptr, table) };
    a.into_big()
}
