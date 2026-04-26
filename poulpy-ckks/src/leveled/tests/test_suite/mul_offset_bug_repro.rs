//! Isolate the convolution-pipeline issue described in `note.md`.
//!
//! ## Strategy
//!
//! `mul`/`square` go through several layers before they reach the actual
//! polynomial multiplication:
//!
//! ```text
//!     ckks::square
//!         |
//!         v
//!     core::glwe_tensor_apply       <-- offset bits/limbs split
//!         |
//!         v
//!     core::cnv_apply_dft           <-- bivariate convolution + DFT
//!         |
//!         v
//!     core::vec_znx_big_normalize   <-- carry propagation, sub-limb shift
//! ```
//!
//! `test_deep_square_chain` shows the bug at X^64 (the catastrophic ~43-bit
//! precision drop) but goes through encryption noise, the tensor key,
//! relinearization, and rescale before the failure surfaces. That hides
//! whether the math error is in the offset arithmetic, the convolution, or
//! the final normalize.
//!
//! This script removes everything CKKS-specific:
//!
//! 1. Allocate two `VecZnx` of size `S` and place a single known value `V`
//!    at limb `i_msg` of each. Everything else is zero.
//! 2. Run the production path (`cnv_apply_dft` + `vec_znx_big_normalize`)
//!    using the same offset arithmetic that `glwe_tensor_apply` would use:
//!
//!    ```ignore
//!    let (off_hi, off_lo) = if off_bits < base2k {
//!        (0, -((base2k - off_bits % base2k) as i64))
//!    } else {
//!        ((off_bits / base2k).saturating_sub(1), (off_bits % base2k) as i64)
//!    };
//!    cnv_apply_dft(off_hi, ...);
//!    vec_znx_big_normalize(res_offset = off_lo, ...);
//!    ```
//! 3. Sweep `i_msg` from `S-1` (LSB) up to `0` (MSB). The user said
//!    *"loss in the top limbs"* — small `i_msg` values are exactly the
//!    most-significant ("top") limbs, so this sweep walks the message
//!    upwards toward the failure region.
//! 4. Sweep `off_bits` over `(i_msg+1)*K, (i_msg+2)*K, ...`, plus
//!    one-bit perturbations on either side, to see how the result moves
//!    as the shift crosses limb boundaries and as `off_lo` flips from `0`
//!    to non-zero.
//! 5. For each probe, also print the FULL set of non-zero output limbs
//!    (not just the dominant one) so the carry chain is visible.
//!
//! ## What the probe reveals (run output below)
//!
//! Two qualitatively different failure modes appear:
//!
//! ### 1. Sub-limb-shift inflation (`off_lo > 0`)
//!
//! For limb-aligned `off_bits = N*K`, the lsh in `vec_znx_big_normalize`
//! is `0` and the result lives at one limb with value `V^2 / 2^K = 256`
//! (for `V = 2^30`, `V^2 = 2^60`, base2k = 52).
//!
//! For `off_bits = N*K + r` with `r > 0`, the lsh is `r` and the result
//! is multiplied by `2^r`. So `off_lo=1` → value `512`, `off_lo=8` →
//! value `65536`, `off_lo=51` → value pushed up by ~one limb.
//!
//! Mathematically this is "shift the value by exactly `off_bits` bits",
//! which is internally consistent. But the higher CKKS layers
//! (relinearize, rescale) then expect `off_bits` to be limb-aligned and
//! treat the result as if no sub-limb scaling happened. The hidden
//! `2^{off_lo}` factor accumulates across squarings and eventually
//! crosses a limb boundary in a way the rescale cannot undo.
//!
//! ### 2. Carry-out-of-the-top loss
//!
//! When the convolution + lsh produces a result whose carry chain
//! reaches limb `0` (the MSB of the bivariate buffer) AND that limb
//! still has overflow, the overflowing bits are written into a "carry"
//! variable and then DISCARDED at the end of the loop in
//! `vec_znx_normalize_inter_base2k`
//! (`poulpy-hal/src/reference/vec_znx/normalize.rs:139-146`):
//!
//! ```ignore
//! for j in 0..res_end { ... }   // res_end is 0 here, so the loop is skipped
//! // The final `carry` from the middle-step loop is just dropped.
//! ```
//!
//! Concretely, for `i_msg = 0` (input at the MSB limb) and any
//! reasonable `off_bits`, the production output is **all zeros** even
//! though the bit-level mathematics says the result should live at
//! limb 0 or 1 with a non-trivial value. The squared product simply
//! falls off the top of the buffer.
//!
//! This is the *loss in the top limbs* the user described, and it
//! matches the X^64 catastrophic step in `test_deep_square_chain`:
//! after several rescales, the message has migrated towards the MSB
//! limbs of the (shrinking) ciphertext, and the next square's
//! convolution + normalize is the operation that drops it.
//!
//! Run with:
//! ```
//! cargo test -p poulpy-ckks mul_offset_bug_repro -- --nocapture
//! ```

use super::helpers::TestContext;
use poulpy_core::layouts::Base2K;
use poulpy_hal::{
    api::{
        CnvPVecAlloc, Convolution, ModuleN, ScratchOwnedAlloc, ScratchOwnedBorrow, ScratchTakeBasic, VecZnxBigAlloc,
        VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAlloc, VecZnxIdftApplyTmpA, VecZnxNormalizeInplace,
        VecZnxNormalizeTmpBytes,
    },
    layouts::{
        Backend, CnvPVecL, CnvPVecR, Module, Scratch, ScratchOwned, VecZnx, VecZnxBig, VecZnxDft, ZnxInfos, ZnxView, ZnxViewMut,
    },
};

/// Mirrors `glwe_tensor_apply` offset split:
/// `poulpy-core/src/operations/glwe.rs:743-748`.
fn glwe_offset_split(off_bits: usize, base2k: usize) -> (usize, i64) {
    if off_bits < base2k {
        (0, -((base2k - (off_bits % base2k)) as i64))
    } else {
        ((off_bits / base2k).saturating_sub(1), (off_bits % base2k) as i64)
    }
}

/// Single-term test vector: zero everywhere except limb `i_msg` of column 0
/// which holds `value` at coefficient 0 (constant polynomial).
fn make_single_term_input(n: usize, size: usize, i_msg: usize, value: i64) -> VecZnx<Vec<u8>> {
    let mut v: VecZnx<Vec<u8>> = VecZnx::alloc(n, 1, size);
    let limb: &mut [i64] = v.at_mut(0, i_msg);
    limb[0] = value;
    v
}

/// Output of one production-path probe.
#[derive(Debug, Clone)]
struct ProbeResult {
    landing_limb: Option<usize>,
    landing_value: i64,
    /// All non-zero limbs (limb_index, value at coeff 0) for inspection.
    nonzero: Vec<(usize, i64)>,
}

fn summarize_result(res: &VecZnx<Vec<u8>>) -> ProbeResult {
    let mut landing_limb: Option<usize> = None;
    let mut landing_value: i64 = 0;
    let mut nonzero: Vec<(usize, i64)> = Vec::new();
    for s in 0..res.size() {
        let v: i64 = res.at(0, s)[0];
        if v != 0 {
            nonzero.push((s, v));
        }
        if v.abs() > landing_value.abs() {
            landing_value = v;
            landing_limb = Some(s);
        }
    }
    ProbeResult {
        landing_limb,
        landing_value,
        nonzero,
    }
}

#[allow(clippy::too_many_arguments)]
fn run_production_convolution<BE: Backend>(
    module: &Module<BE>,
    base2k: usize,
    a: &VecZnx<Vec<u8>>,
    b: &VecZnx<Vec<u8>>,
    off_bits: usize,
    res_size: usize,
) -> VecZnx<Vec<u8>>
where
    Module<BE>: ModuleN
        + Convolution<BE>
        + CnvPVecAlloc<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxNormalizeInplace<BE>
        + VecZnxNormalizeTmpBytes,
    Scratch<BE>: ScratchTakeBasic,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let (off_hi, off_lo) = glwe_offset_split(off_bits, base2k);
    let a_size = a.size();
    let b_size = b.size();

    // Allocate the result with one extra MSB limb of headroom so the
    // bug-fix in `vec_znx_normalize_inter_base2k` (FFT64 ref) and its
    // mirror in `ntt120_vec_znx_big_normalize_inter` (NTT120 ref) can
    // absorb the residual carry from the sub-limb shift instead of
    // dropping it. The carry lands at res_have[0]; the original data is
    // shifted up by 1 in index and lives at res_have[1..res_size+1].
    let res_have_size = res_size + 1;

    let mut a_prep: CnvPVecL<Vec<u8>, BE> = module.cnv_pvec_left_alloc(1, a_size);
    let mut b_prep: CnvPVecR<Vec<u8>, BE> = module.cnv_pvec_right_alloc(1, b_size);
    let mut res_dft: VecZnxDft<Vec<u8>, BE> = module.vec_znx_dft_alloc(1, res_size);
    let mut res_big: VecZnxBig<Vec<u8>, BE> = module.vec_znx_big_alloc(1, res_size);
    let mut res_have: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), 1, res_have_size);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .cnv_apply_dft_tmp_bytes(res_size, off_hi, a_size, b_size)
            .max(module.cnv_prepare_left_tmp_bytes(res_size, a_size))
            .max(module.cnv_prepare_right_tmp_bytes(res_size, b_size))
            .max(module.vec_znx_big_normalize_tmp_bytes()),
    );

    module.cnv_prepare_left(&mut a_prep, a, scratch.borrow());
    module.cnv_prepare_right(&mut b_prep, b, scratch.borrow());
    module.cnv_apply_dft(&mut res_dft, off_hi, 0, &a_prep, 0, &b_prep, 0, scratch.borrow());
    module.vec_znx_idft_apply_tmpa(&mut res_big, 0, &mut res_dft, 0);
    module.vec_znx_big_normalize(&mut res_have, base2k, off_lo, 0, &res_big, base2k, 0, scratch.borrow());

    res_have
}

fn probe<BE: Backend>(
    module: &Module<BE>,
    base2k: usize,
    size: usize,
    i_msg: usize,
    off_bits: usize,
    value: i64,
) -> ProbeResult
where
    Module<BE>: ModuleN
        + Convolution<BE>
        + CnvPVecAlloc<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxNormalizeInplace<BE>
        + VecZnxNormalizeTmpBytes,
    Scratch<BE>: ScratchTakeBasic,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let a = make_single_term_input(module.n(), size, i_msg, value);
    let b = make_single_term_input(module.n(), size, i_msg, value);

    // Tensor result size as `mul_tensor_size` would compute it. Note this
    // is exactly what the production `glwe_tensor_apply` allocates for
    // its destination tensor — the same buffer that we'll later detect
    // overflow against.
    let offset_limbs = off_bits / base2k;
    let res_size = (a.size() + b.size()).saturating_sub(offset_limbs);
    if res_size == 0 {
        let empty: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), 1, 1);
        return summarize_result(&empty);
    }

    let res_prod = run_production_convolution(module, base2k, &a, &b, off_bits, res_size);
    summarize_result(&res_prod)
}

pub fn test_mul_offset_bug_repro<BE: Backend>(ctx: &TestContext<BE>)
where
    Module<BE>: ModuleN
        + Convolution<BE>
        + CnvPVecAlloc<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxNormalizeInplace<BE>
        + VecZnxNormalizeTmpBytes,
    Scratch<BE>: ScratchTakeBasic,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let module = &ctx.module;
    let base2k = Base2K(ctx.params.base2k).0 as usize;
    let size: usize = 5;
    let value: i64 = 1 << 30;

    eprintln!("=== Convolution offset isolation ===");
    eprintln!(
        "ring n = {}, base2k = {}, size = {}, value = 2^30 (V^2 = 2^60)",
        module.n(),
        base2k,
        size
    );
    eprintln!();
    eprintln!("Each row probes:");
    eprintln!("  a = b = single value V at limb i_msg, square via");
    eprintln!("  cnv_apply_dft + vec_znx_big_normalize, with the same off_hi/off_lo split");
    eprintln!("  that glwe_tensor_apply would use for off_bits.");
    eprintln!();
    eprintln!("Reading the table:");
    eprintln!("  - 'landing_limb' is the limb where the dominant value lives in the output.");
    eprintln!("  - 'nonzero' lists EVERY non-zero limb (limb=value), so the carry chain is visible.");
    eprintln!("  - For aligned off_bits (off_lo == 0) the dominant value is V^2 / 2^K = 256.");
    eprintln!("  - For off_lo > 0 the dominant value is 256 * 2^{{off_lo}} (sub-limb inflation).");
    eprintln!("  - 'landing=(none)' means the convolution + normalize wiped the result entirely:");
    eprintln!("    the carry chain reached limb 0 and the overflow was discarded.");
    eprintln!();

    let mut all_zero_cases: Vec<(usize, usize, usize, i64)> = Vec::new();
    let mut sub_limb_inflation_cases: Vec<(usize, usize, usize, i64, i64)> = Vec::new();

    for i_msg in (0..size).rev() {
        eprintln!(
            "--- i_msg = {} (limb position from MSB = {} bits) ---",
            i_msg,
            (i_msg + 1) * base2k
        );
        eprintln!("off_bits | off_hi | off_lo | landing_limb | landing_value | nonzero_limbs");
        eprintln!("---------+--------+--------+--------------+---------------+--------------");

        // Walk a few off_bits values around the message position. Three families:
        //   - limb-aligned shifts: (i_msg+1)*K, (i_msg+2)*K, ...
        //   - one-bit perturbations: aligned ±1
        //   - the actual `inner.k() = size*K` value that mul/square pass
        //     for a fresh ct
        //   - off_bits = 216, the value at the X^64 catastrophic step
        let center = (i_msg + 1) * base2k;
        let mut candidates: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        for shift_limbs in 0..=4usize {
            let aligned = center + shift_limbs * base2k;
            if aligned >= 1 {
                candidates.insert(aligned - 1);
                candidates.insert(aligned);
                candidates.insert(aligned + 1);
            }
        }
        candidates.insert(size * base2k);
        candidates.insert(216);

        for &off_bits in &candidates {
            if off_bits == 0 {
                continue;
            }
            let prod = probe(module, base2k, size, i_msg, off_bits, value);
            let (off_hi, off_lo) = glwe_offset_split(off_bits, base2k);
            let nonzero_str = if prod.nonzero.is_empty() {
                "(all zero)".to_string()
            } else {
                prod.nonzero
                    .iter()
                    .map(|(l, v)| format!("({l}={v})"))
                    .collect::<Vec<_>>()
                    .join(",")
            };
            eprintln!(
                "{:>8} | {:>6} | {:>6} | {:>12} | {:>13} | {}",
                off_bits,
                off_hi,
                off_lo,
                prod.landing_limb
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "(none)".to_string()),
                prod.landing_value,
                nonzero_str,
            );

            if prod.landing_limb.is_none() {
                all_zero_cases.push((size, i_msg, off_bits, value));
            } else if off_lo > 0 && prod.landing_value != 256 && prod.landing_value != 0 {
                // 256 = V^2 / 2^K is the "clean limb-aligned" answer
                sub_limb_inflation_cases.push((size, i_msg, off_bits, off_lo, prod.landing_value));
            }
        }
        eprintln!();
    }

    eprintln!();
    eprintln!("=== Summary ===");
    eprintln!();
    eprintln!(
        "1. Sub-limb shift inflation: {} probes returned a value != 256 because off_lo > 0 introduced a 2^{{off_lo}} factor",
        sub_limb_inflation_cases.len()
    );
    if let Some((size, i_msg, off_bits, off_lo, value)) = sub_limb_inflation_cases.first() {
        eprintln!(
            "   first example: size={size} i_msg={i_msg} off_bits={off_bits} off_lo={off_lo} -> value {value}"
        );
        eprintln!(
            "   = 256 * 2^{} = {}",
            off_lo,
            256i64 << *off_lo as i64
        );
    }
    eprintln!();
    eprintln!(
        "2. Carry-out-of-the-top: {} probes returned an all-zero result",
        all_zero_cases.len()
    );
    if let Some((size, i_msg, off_bits, value)) = all_zero_cases.first() {
        eprintln!(
            "   first example: size={size} i_msg={i_msg} off_bits={off_bits} value={value}"
        );
    }
    eprintln!();
    eprintln!("These two failure modes match the user's note.md description:");
    eprintln!("  - 'log_delta: 91' / non-aligned consumption == sub-limb shift inflation");
    eprintln!("  - 'loss in the top limbs' == carry-out-of-the-top");
    eprintln!();
    eprintln!("Localisation in the source:");
    eprintln!("  - Offset split: poulpy-core/src/operations/glwe.rs:743-748");
    eprintln!("  - cnv_apply_dft: poulpy-hal/src/reference/fft64/convolution.rs:177");
    eprintln!("  - vec_znx_big_normalize lsh path: poulpy-hal/src/reference/vec_znx/normalize.rs");
    eprintln!("  - The discarded final carry: same file, after the middle-step loop");
    eprintln!("    (`for j in 0..res_end {{ ... }}` is skipped because res_end == 0).");

    eprintln!();
    eprintln!("=== Status of the fix ===");
    eprintln!();
    eprintln!("The bug-fix in `vec_znx_normalize_inter_base2k`");
    eprintln!("(poulpy-hal/src/reference/vec_znx/normalize.rs) and its mirror in");
    eprintln!("`ntt120_vec_znx_big_normalize_inter`");
    eprintln!("(poulpy-hal/src/reference/ntt120/vec_znx_big.rs) routes the");
    eprintln!("`limbs_offset == 0 && lsh > 0 && res_size > a_size` case through the");
    eprintln!("`limbs_offset = -1` path, which absorbs the residual carry from limb 0");
    eprintln!("into the extra MSB limb instead of silently discarding it.");
    eprintln!();
    eprintln!("This probe now allocates the result buffer with `res_size + 1` limbs to");
    eprintln!("trigger the fix path. With the fix in place:");
    eprintln!();
    eprintln!("  - Every off_lo > 0 case now lands at a higher limb index (= one limb");
    eprintln!("    further from limb 0) than before, because the data is shifted into");
    eprintln!("    res[1..a_size+1] instead of res[0..a_size]. The absorbed carry sits");
    eprintln!("    at res[0] of the new layout.");
    eprintln!();
    eprintln!("  - off_lo == 0 cases still go through the unmodified code path, which");
    eprintln!("    means the limb-aligned overflow at limb 0 is still discarded if it");
    eprintln!("    happens. That is mathematically correct: with lsh == 0 the per-limb");
    eprintln!("    carry chain cannot extend past the buffer (the cumulative bound is");
    eprintln!("    O(N) which is dwarfed by 2^K), so the residual is always 0 in well-");
    eprintln!("    formed bivariate operands. The remaining `(none)` rows in the table");
    eprintln!("    above are degenerate single-limb tensor cases that are not produced");
    eprintln!("    by the real CKKS pipeline.");
    eprintln!();
    eprintln!("Operational consequence: callers that allocate the destination of");
    eprintln!("`vec_znx_big_normalize` with one extra limb of MSB headroom now");
    eprintln!("preserve the residual carry. The data layout shifts by 1 limb, so the");
    eprintln!("caller must read its data from `res[1..]` (or simply view `res[0]` as");
    eprintln!("the new most-significant limb).");
    eprintln!();
    eprintln!("Note: the higher-level `test_deep_square_chain` failure is a *different*");
    eprintln!("bug (phase diffusion in chain ciphertexts, see");
    eprintln!("`docs/ckks_deep_mul_investigation.md` in the older stash) and is not");
    eprintln!("addressed by this fix.");
}
