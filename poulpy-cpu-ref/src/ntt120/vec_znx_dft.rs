//! NTT-domain ring element vector operations for [`NTT120Ref`](crate::NTT120Ref).
//!
//! Implements the `VecZnxDft*` and `VecZnxIdft*` OEP traits. `VecZnxDft` stores
//! ring element vectors in the NTT domain (`ScalarPrep = Q120bScalar`), where
//! polynomial multiplication reduces to component-wise lazy-modular multiplication
//! over four CRT residues.
//!
//! Operations include:
//!
//! - **Allocation**: byte-size calculation, heap allocation, construction from raw bytes.
//! - **Forward NTT**: integer-domain `VecZnx` → NTT-domain `VecZnxDft`, with
//!   configurable step/offset for partial transforms.
//! - **Inverse NTT**: `VecZnxDft` → `VecZnxBig` (large-coefficient), with variants that
//!   consume, borrow, or use the input as temporary storage.
//! - **NTT-domain arithmetic**: add, sub, negate, scaled-add, copy, zero.
//!
//! The IDFT-consume path performs in-place CRT compaction from the Q120b layout
//! (32 bytes/coefficient) to the i128 layout (16 bytes/coefficient), enabling
//! zero-copy conversion of an owned `VecZnxDft` into a `VecZnxBig`.

use std::mem::size_of;

use bytemuck::cast_slice_mut;
use poulpy_hal::{
    api::TakeSlice,
    layouts::{
        Data, Module, Scratch, VecZnxBig, VecZnxBigToMut, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef, VecZnxToRef, ZnxInfos,
        ZnxViewMut,
    },
    oep::{
        TakeSliceImpl, VecZnxDftAddImpl, VecZnxDftAddInplaceImpl, VecZnxDftAddScaledInplaceImpl, VecZnxDftApplyImpl,
        VecZnxDftCopyImpl, VecZnxDftSubImpl, VecZnxDftSubInplaceImpl, VecZnxDftSubNegateInplaceImpl, VecZnxDftZeroImpl,
        VecZnxIdftApplyConsumeImpl, VecZnxIdftApplyImpl, VecZnxIdftApplyTmpAImpl, VecZnxIdftApplyTmpBytesImpl,
    },
    reference::ntt120::{
        ntt::{NttTableInv, intt_ref},
        primes::{PrimeSet, Primes30},
        vec_znx_dft::{
            NttModuleHandle, ntt120_vec_znx_dft_add, ntt120_vec_znx_dft_add_inplace, ntt120_vec_znx_dft_add_scaled_inplace,
            ntt120_vec_znx_dft_apply, ntt120_vec_znx_dft_copy, ntt120_vec_znx_dft_sub, ntt120_vec_znx_dft_sub_inplace,
            ntt120_vec_znx_dft_sub_negate_inplace, ntt120_vec_znx_dft_zero, ntt120_vec_znx_idft_apply,
            ntt120_vec_znx_idft_apply_tmp_bytes, ntt120_vec_znx_idft_apply_tmpa,
        },
    },
};

use crate::NTT120Ref;

// ──────────────────────────────────────────────────────────────────────────────
// In-place Q120b → i128 compaction helper
// ──────────────────────────────────────────────────────────────────────────────

/// Barrett reduction: `x < 2^61`, `q < 2^30`, `mu = floor(2^61 / q)` → `x mod q` in `[0, q)`.
///
/// Two conditional subtracts correct the quotient approximation error (≤ 2).
#[inline(always)]
fn barrett_u61(x: u64, q: u64, mu: u64) -> u64 {
    let q_approx = ((x as u128 * mu as u128) >> 61) as u64;
    let r = x - q_approx * q;
    let r = if r >= q { r - q } else { r };
    if r >= q { r - q } else { r }
}

/// Fused q120b reduce + CRT multiply: `(x mod Q) * CRT_CST mod Q` in one Barrett pass.
///
/// Instead of two separate Barrett reductions (`reduce_q120b` then `barrett(x * CRT)`),
/// this splits `x = x_hi * 2^32 + x_lo_hi * 2^16 + x_lo_lo` and uses precomputed combined
/// constants so the full computation collapses to a single Barrett pass.
///
/// # Bounds
/// `tmp < Q^2 + 2 * 2^16 * Q < 2^60 + 2^47 < 2^61` ✓ — one Barrett pass suffices.
#[inline(always)]
fn reduce_q120b_crt(x: u64, q: u64, mu: u64, pow32_crt: u64, pow16_crt: u64, crt: u64) -> u64 {
    let x_hi = x >> 32;
    let x_hi_r = if x_hi >= q { x_hi - q } else { x_hi };
    let x_lo = x & 0xFFFF_FFFF;
    let x_lo_hi = x_lo >> 16;
    let x_lo_lo = x_lo & 0xFFFF;
    let tmp = x_hi_r
        .wrapping_mul(pow32_crt)
        .wrapping_add(x_lo_hi.wrapping_mul(pow16_crt))
        .wrapping_add(x_lo_lo.wrapping_mul(crt));
    barrett_u61(tmp, q, mu)
}

/// In-place CRT-compact all NTT blocks from Q120b (32 bytes/coeff) to i128 (16 bytes/coeff).
///
/// For each block `k` in `0..n_blocks`, in order:
///
/// 1. Applies the inverse NTT to the Q120b block in-place.
/// 2. For each coefficient, uses scalar Barrett reduction (no division) to compute
///    `t[j] = (x[j] % Q[j] * CRT[j]) % Q[j]` for each prime `j`, then accumulates
///    the CRT weighted sum in u128 and reduces with bounded conditional subtractions.
///
/// # Ordering invariant
///
/// Blocks are processed in order `k = 0, 1, ..., n_blocks-1`.  For `k ≥ 1` the
/// destination `[16nk, 16n(k+1))` never overlaps the source `[32nk, 32n(k+1))`.
/// For `k = 0` all four residues are read before the i128 is written.
///
/// # Safety
///
/// - `u64_ptr` must be valid for reads and writes of at least `4 * n * n_blocks` u64 values.
/// - The backing allocation must be at least 16-byte aligned (guaranteed by `DEFAULTALIGN = 64`).
/// - No other references to the same memory may be live during this call.
unsafe fn compact_all_blocks(n: usize, n_blocks: usize, u64_ptr: *mut u64, table: &NttTableInv<Primes30>) {
    // Barrett constants: mu[k] = floor(2^61 / Q[k]).
    let q_u64: [u64; 4] = Primes30::Q.map(|qi| qi as u64);
    let mu: [u64; 4] = q_u64.map(|qi| (1u64 << 61) / qi);
    let crt: [u64; 4] = Primes30::CRT_CST.map(|c| c as u64);

    // Fused CRT constants: pow32_crt[k] = (2^32 mod Q[k]) * CRT[k] mod Q[k],
    // pow16_crt[k] = 2^16 * CRT[k] mod Q[k]  (2^16 < Q for all Primes30).
    let pow32_crt: [u64; 4] = std::array::from_fn(|k| {
        let pow32 = ((1u128 << 32) % q_u64[k] as u128) as u64;
        barrett_u61(pow32 * crt[k], q_u64[k], mu[k])
    });
    let pow16_crt: [u64; 4] = std::array::from_fn(|k| barrett_u61((1u64 << 16) * crt[k], q_u64[k], mu[k]));

    // CRT reconstruction constants in u128 (no i128 sign issues).
    let q: [u128; 4] = q_u64.map(|qi| qi as u128);
    let total_q: u128 = q[0] * q[1] * q[2] * q[3];
    let qm: [u128; 4] = [q[1] * q[2] * q[3], q[0] * q[2] * q[3], q[0] * q[1] * q[3], q[0] * q[1] * q[2]];
    let half_q: u128 = total_q.div_ceil(2);
    // Table-based modular reduction: precomputed once, reused across all blocks.
    let total_q_mult: [u128; 4] = [0, total_q, total_q * 2, total_q * 3];

    for k in 0..n_blocks {
        let src_start = 4 * n * k; // u64 index: start of DFT block k
        let dst_start = 2 * n * k; // u64 index: start of Big block k

        // Step 1: inverse NTT in-place.
        {
            let blk: &mut [u64] = unsafe { std::slice::from_raw_parts_mut(u64_ptr.add(src_start), 4 * n) };
            intt_ref::<Primes30>(table, blk);
        }

        // Step 2: CRT-compact 4n u64s → n i128s.
        for c in 0..n {
            // Read all four residues before any write (critical for k=0, c=0).
            let (x0, x1, x2, x3) = unsafe {
                (
                    *u64_ptr.add(src_start + 4 * c),
                    *u64_ptr.add(src_start + 4 * c + 1),
                    *u64_ptr.add(src_start + 4 * c + 2),
                    *u64_ptr.add(src_start + 4 * c + 3),
                )
            };

            // Fused: t[k] = (x[k] * CRT[k]) mod Q[k] in one Barrett pass (no div).
            let t0 = reduce_q120b_crt(x0, q_u64[0], mu[0], pow32_crt[0], pow16_crt[0], crt[0]);
            let t1 = reduce_q120b_crt(x1, q_u64[1], mu[1], pow32_crt[1], pow16_crt[1], crt[1]);
            let t2 = reduce_q120b_crt(x2, q_u64[2], mu[2], pow32_crt[2], pow16_crt[2], crt[2]);
            let t3 = reduce_q120b_crt(x3, q_u64[3], mu[3], pow32_crt[3], pow16_crt[3], crt[3]);

            // CRT weighted sum in u128 (t[k] < Q[k] ≤ 2^30, qm[k] < 2^90).
            let mut v: u128 = t0 as u128 * qm[0] + t1 as u128 * qm[1] + t2 as u128 * qm[2] + t3 as u128 * qm[3];

            // Table-based reduction: q_approx = floor(v / 2^120) ∈ {0,1,2,3}.
            let q_approx = (v >> 120) as usize;
            v -= total_q_mult[q_approx]; // unconditional subtract (never underflows)
            if v >= total_q {
                v -= total_q; // at most 1 correction (proved: q_real - q_approx ≤ 1)
            }

            // Symmetric lift to (-total_q/2, total_q/2].
            let val: i128 = if v >= half_q { v as i128 - total_q as i128 } else { v as i128 };

            unsafe { (u64_ptr.add(dst_start + 2 * c) as *mut i128).write_unaligned(val) };
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Allocation
// ──────────────────────────────────────────────────────────────────────────────

// ──────────────────────────────────────────────────────────────────────────────
// Inverse NTT
// ──────────────────────────────────────────────────────────────────────────────

unsafe impl VecZnxIdftApplyTmpBytesImpl<Self> for NTT120Ref {
    fn vec_znx_idft_apply_tmp_bytes_impl(module: &Module<Self>) -> usize {
        ntt120_vec_znx_idft_apply_tmp_bytes(module.n())
    }
}

unsafe impl VecZnxIdftApplyImpl<Self> for NTT120Ref
where
    Self: TakeSliceImpl<Self>,
{
    fn vec_znx_idft_apply_impl<R, A>(
        module: &Module<Self>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<Self>,
    ) where
        R: VecZnxBigToMut<Self>,
        A: VecZnxDftToRef<Self>,
    {
        let (tmp, _) = scratch.take_slice(ntt120_vec_znx_idft_apply_tmp_bytes(module.n()) / size_of::<u64>());
        ntt120_vec_znx_idft_apply::<R, A, Self>(module, res, res_col, a, a_col, tmp);
    }
}

unsafe impl VecZnxIdftApplyTmpAImpl<Self> for NTT120Ref {
    fn vec_znx_idft_apply_tmpa_impl<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &mut A, a_col: usize)
    where
        R: VecZnxBigToMut<Self>,
        A: VecZnxDftToMut<Self>,
    {
        ntt120_vec_znx_idft_apply_tmpa::<R, A, Self>(module, res, res_col, a, a_col);
    }
}

unsafe impl VecZnxIdftApplyConsumeImpl<Self> for NTT120Ref {
    fn vec_znx_idft_apply_consume_impl<D: Data>(
        module: &Module<NTT120Ref>,
        mut a: VecZnxDft<D, NTT120Ref>,
    ) -> VecZnxBig<D, NTT120Ref>
    where
        VecZnxDft<D, NTT120Ref>: VecZnxDftToMut<NTT120Ref>,
    {
        let table = module.get_intt_table();

        // Obtain a mutable view, extract geometry and raw pointer, then release the borrow.
        let (n, n_blocks, u64_ptr) = {
            let mut a_mut: VecZnxDft<&mut [u8], NTT120Ref> = a.to_mut();
            let n = a_mut.n();
            let n_blocks = a_mut.cols() * a_mut.size();
            // Obtain a raw u64 pointer from the flat scalar slice; the borrow ends
            // at the closing brace so no &mut reference is live during compact_all_blocks.
            let ptr: *mut u64 = {
                let s = a_mut.raw_mut(); // &mut [Q120bScalar]
                cast_slice_mut::<_, u64>(s).as_mut_ptr()
            };
            (n, n_blocks, ptr)
        }; // a_mut (and the &mut borrow of a) dropped here

        // In-place: apply iNTT per block and CRT-compact Q120b → i128.
        // After this, the first n*cols*size i128 values are at bytes [0, 16*n*cols*size).
        // SAFETY:
        //   - u64_ptr came from `a_mut.raw_mut()` which covers 4*n*cols*size u64s.
        //   - No other references to `a`'s data exist after the block above.
        //   - DEFAULTALIGN = 64 guarantees 64-byte alignment; all i128 writes are
        //     at multiples of 16 bytes (safe for write_unaligned on all platforms).
        unsafe { compact_all_blocks(n, n_blocks, u64_ptr, table) };

        // Reinterpret the (now compacted) buffer as VecZnxBig<D, NTT120Ref>.
        // The first n*cols*size i128s are at the correct offsets for VecZnxBig layout.
        a.into_big()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Forward NTT
// ──────────────────────────────────────────────────────────────────────────────

unsafe impl VecZnxDftApplyImpl<Self> for NTT120Ref {
    fn vec_znx_dft_apply_impl<R, A>(
        module: &Module<Self>,
        step: usize,
        offset: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
    ) where
        R: VecZnxDftToMut<Self>,
        A: VecZnxToRef,
    {
        ntt120_vec_znx_dft_apply::<R, A, Self>(module, step, offset, res, res_col, a, a_col);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// NTT-domain arithmetic
// ──────────────────────────────────────────────────────────────────────────────

unsafe impl VecZnxDftAddImpl<Self> for NTT120Ref {
    fn vec_znx_dft_add_impl<R, A, D>(
        _module: &Module<Self>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &D,
        b_col: usize,
    ) where
        R: VecZnxDftToMut<Self>,
        A: VecZnxDftToRef<Self>,
        D: VecZnxDftToRef<Self>,
    {
        ntt120_vec_znx_dft_add::<R, A, D, Self>(res, res_col, a, a_col, b, b_col);
    }
}

unsafe impl VecZnxDftAddScaledInplaceImpl<Self> for NTT120Ref {
    fn vec_znx_dft_add_scaled_inplace_impl<R, A>(
        _module: &Module<Self>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        a_scale: i64,
    ) where
        R: VecZnxDftToMut<Self>,
        A: VecZnxDftToRef<Self>,
    {
        ntt120_vec_znx_dft_add_scaled_inplace::<R, A, Self>(res, res_col, a, a_col, a_scale);
    }
}

unsafe impl VecZnxDftAddInplaceImpl<Self> for NTT120Ref {
    fn vec_znx_dft_add_inplace_impl<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<Self>,
        A: VecZnxDftToRef<Self>,
    {
        ntt120_vec_znx_dft_add_inplace::<R, A, Self>(res, res_col, a, a_col);
    }
}

unsafe impl VecZnxDftSubImpl<Self> for NTT120Ref {
    fn vec_znx_dft_sub_impl<R, A, D>(
        _module: &Module<Self>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &D,
        b_col: usize,
    ) where
        R: VecZnxDftToMut<Self>,
        A: VecZnxDftToRef<Self>,
        D: VecZnxDftToRef<Self>,
    {
        ntt120_vec_znx_dft_sub::<R, A, D, Self>(res, res_col, a, a_col, b, b_col);
    }
}

unsafe impl VecZnxDftSubInplaceImpl<Self> for NTT120Ref {
    fn vec_znx_dft_sub_inplace_impl<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<Self>,
        A: VecZnxDftToRef<Self>,
    {
        ntt120_vec_znx_dft_sub_inplace::<R, A, Self>(res, res_col, a, a_col);
    }
}

unsafe impl VecZnxDftSubNegateInplaceImpl<Self> for NTT120Ref {
    fn vec_znx_dft_sub_negate_inplace_impl<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<Self>,
        A: VecZnxDftToRef<Self>,
    {
        ntt120_vec_znx_dft_sub_negate_inplace::<R, A, Self>(res, res_col, a, a_col);
    }
}

unsafe impl VecZnxDftCopyImpl<Self> for NTT120Ref {
    fn vec_znx_dft_copy_impl<R, A>(
        _module: &Module<Self>,
        step: usize,
        offset: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
    ) where
        R: VecZnxDftToMut<Self>,
        A: VecZnxDftToRef<Self>,
    {
        ntt120_vec_znx_dft_copy::<R, A, Self>(step, offset, res, res_col, a, a_col);
    }
}

unsafe impl VecZnxDftZeroImpl<Self> for NTT120Ref {
    fn vec_znx_dft_zero_impl<R>(_module: &Module<Self>, res: &mut R, res_col: usize)
    where
        R: VecZnxDftToMut<Self>,
    {
        ntt120_vec_znx_dft_zero::<R, Self>(res, res_col);
    }
}
