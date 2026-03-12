//! NTT-domain ring element vector operations for [`NTT120Avx`](super::NTT120Avx).
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
use core::arch::x86_64::__m256i;
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
        ntt::NttTableInv,
        primes::Primes30,
        vec_znx_dft::{
            NttModuleHandle, ntt120_vec_znx_dft_add, ntt120_vec_znx_dft_add_inplace, ntt120_vec_znx_dft_add_scaled_inplace,
            ntt120_vec_znx_dft_apply, ntt120_vec_znx_dft_copy, ntt120_vec_znx_dft_sub, ntt120_vec_znx_dft_sub_inplace,
            ntt120_vec_znx_dft_sub_negate_inplace, ntt120_vec_znx_dft_zero, ntt120_vec_znx_idft_apply,
            ntt120_vec_znx_idft_apply_tmp_bytes, ntt120_vec_znx_idft_apply_tmpa,
        },
    },
};

use super::NTT120Avx;

// ──────────────────────────────────────────────────────────────────────────────
// In-place Q120b → i128 compaction helper
// ──────────────────────────────────────────────────────────────────────────────

/// AVX2-accelerated in-place CRT-compact: Q120b (32 bytes/coeff) → i128 (16 bytes/coeff).
///
/// For each block `k` in `0..n_blocks`, in order:
///
/// 1. Applies the inverse NTT (AVX2) to the Q120b block in-place.
/// 2. For each coefficient, uses [`reduce_b_and_apply_crt`] to compute
///    `t[j] = (x[j] * CRT[j]) mod Q[j]` for all four prime lanes in a single Barrett pass,
///    then accumulates the CRT weighted sum in scalar u128 with bounded conditional subtracts.
///
/// # Ordering invariant
///
/// Blocks are processed in order `k = 0, 1, ..., n_blocks-1`.  For `k ≥ 1` the
/// destination `[16nk, 16n(k+1))` never overlaps the source `[32nk, 32n(k+1))`.
/// For `k = 0` each AVX load (32 bytes) precedes the i128 write (16 bytes).
///
/// # Safety
///
/// - `u64_ptr` must be valid for reads and writes of at least `4 * n * n_blocks` u64 values.
/// - Caller must guarantee AVX2 support (ensured by `Module::<NTT120Avx>::new()`).
/// - No other references to the same memory may be live during this call.
#[target_feature(enable = "avx2")]
unsafe fn compact_all_blocks(n: usize, n_blocks: usize, u64_ptr: *mut u64, table: &NttTableInv<Primes30>) {
    use super::arithmetic_avx::{
        BARRETT_MU, CRT_VEC, POW16_CRT, POW32_CRT, Q_VEC, QM_HI, QM_LO, QM_MID, TOTAL_Q, TOTAL_Q_MULT, crt_accumulate_avx2,
        reduce_b_and_apply_crt,
    };
    use super::ntt::intt_avx2;
    use core::arch::x86_64::_mm256_loadu_si256;

    let half_q: u128 = TOTAL_Q.div_ceil(2);

    // Load all AVX2 constants once before the block loop.
    let q_avx = unsafe { _mm256_loadu_si256(Q_VEC.as_ptr() as *const __m256i) };
    let mu_avx = unsafe { _mm256_loadu_si256(BARRETT_MU.as_ptr() as *const __m256i) };
    let pow32_crt_avx = unsafe { _mm256_loadu_si256(POW32_CRT.as_ptr() as *const __m256i) };
    let pow16_crt_avx = unsafe { _mm256_loadu_si256(POW16_CRT.as_ptr() as *const __m256i) };
    let crt_avx = unsafe { _mm256_loadu_si256(CRT_VEC.as_ptr() as *const __m256i) };
    let qm_hi_avx = unsafe { _mm256_loadu_si256(QM_HI.as_ptr() as *const __m256i) };
    let qm_mid_avx = unsafe { _mm256_loadu_si256(QM_MID.as_ptr() as *const __m256i) };
    let qm_lo_avx = unsafe { _mm256_loadu_si256(QM_LO.as_ptr() as *const __m256i) };

    for k in 0..n_blocks {
        let src_start = 4 * n * k; // u64 index: start of DFT block k
        let dst_start = 2 * n * k; // u64 index: start of Big block k

        // Step 1: inverse NTT in-place (AVX2).
        {
            let blk: &mut [u64] = unsafe { std::slice::from_raw_parts_mut(u64_ptr.add(src_start), 4 * n) };
            unsafe { intt_avx2::<Primes30>(table, blk) };
        }

        // Step 2: CRT-compact 4n u64s → n i128s.
        for c in 0..n {
            // Load all four residues for coefficient c as one __m256i (32 bytes).
            // This read precedes any write, so the k=0 overlap is safe.
            let xv: __m256i = unsafe { _mm256_loadu_si256(u64_ptr.add(src_start + 4 * c) as *const __m256i) };

            // Fused AVX2: t[j] = (x[j] * CRT[j]) mod Q[j] in one Barrett pass.
            let t = unsafe { reduce_b_and_apply_crt(xv, q_avx, mu_avx, pow32_crt_avx, pow16_crt_avx, crt_avx) };

            // Vectorized CRT accumulation: v = Σ t[k] * qm[k] (no store-to-stack round-trip).
            let mut v = unsafe { crt_accumulate_avx2(t, qm_hi_avx, qm_mid_avx, qm_lo_avx) };

            // Table-based modular reduction: q_approx = floor(v / 2^120) ∈ {0,1,2,3}.
            let q_approx = (v >> 120) as usize;
            v -= TOTAL_Q_MULT[q_approx]; // unconditional subtract (never underflows)
            if v >= TOTAL_Q {
                v -= TOTAL_Q; // at most 1 correction (proved: q_real - q_approx ≤ 1)
            }

            // Symmetric lift to (-total_q/2, total_q/2].
            let val: i128 = if v >= half_q { v as i128 - TOTAL_Q as i128 } else { v as i128 };

            // Write i128 (16 bytes) to the compacted destination.
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

unsafe impl VecZnxIdftApplyTmpBytesImpl<Self> for NTT120Avx {
    fn vec_znx_idft_apply_tmp_bytes_impl(module: &Module<Self>) -> usize {
        ntt120_vec_znx_idft_apply_tmp_bytes(module.n())
    }
}

unsafe impl VecZnxIdftApplyImpl<Self> for NTT120Avx
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

unsafe impl VecZnxIdftApplyTmpAImpl<Self> for NTT120Avx {
    fn vec_znx_idft_apply_tmpa_impl<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &mut A, a_col: usize)
    where
        R: VecZnxBigToMut<Self>,
        A: VecZnxDftToMut<Self>,
    {
        ntt120_vec_znx_idft_apply_tmpa::<R, A, Self>(module, res, res_col, a, a_col);
    }
}

unsafe impl VecZnxIdftApplyConsumeImpl<Self> for NTT120Avx {
    fn vec_znx_idft_apply_consume_impl<D: Data>(
        module: &Module<NTT120Avx>,
        mut a: VecZnxDft<D, NTT120Avx>,
    ) -> VecZnxBig<D, NTT120Avx>
    where
        VecZnxDft<D, NTT120Avx>: VecZnxDftToMut<NTT120Avx>,
    {
        let table = module.get_intt_table();

        // Obtain a mutable view, extract geometry and raw pointer, then release the borrow.
        let (n, n_blocks, u64_ptr) = {
            let mut a_mut: VecZnxDft<&mut [u8], NTT120Avx> = a.to_mut();
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

        // Reinterpret the (now compacted) buffer as VecZnxBig<D, NTT120Avx>.
        // The first n*cols*size i128s are at the correct offsets for VecZnxBig layout.
        a.into_big()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Forward NTT
// ──────────────────────────────────────────────────────────────────────────────

unsafe impl VecZnxDftApplyImpl<Self> for NTT120Avx {
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

unsafe impl VecZnxDftAddImpl<Self> for NTT120Avx {
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

unsafe impl VecZnxDftAddScaledInplaceImpl<Self> for NTT120Avx {
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

unsafe impl VecZnxDftAddInplaceImpl<Self> for NTT120Avx {
    fn vec_znx_dft_add_inplace_impl<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<Self>,
        A: VecZnxDftToRef<Self>,
    {
        ntt120_vec_znx_dft_add_inplace::<R, A, Self>(res, res_col, a, a_col);
    }
}

unsafe impl VecZnxDftSubImpl<Self> for NTT120Avx {
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

unsafe impl VecZnxDftSubInplaceImpl<Self> for NTT120Avx {
    fn vec_znx_dft_sub_inplace_impl<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<Self>,
        A: VecZnxDftToRef<Self>,
    {
        ntt120_vec_znx_dft_sub_inplace::<R, A, Self>(res, res_col, a, a_col);
    }
}

unsafe impl VecZnxDftSubNegateInplaceImpl<Self> for NTT120Avx {
    fn vec_znx_dft_sub_negate_inplace_impl<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<Self>,
        A: VecZnxDftToRef<Self>,
    {
        ntt120_vec_znx_dft_sub_negate_inplace::<R, A, Self>(res, res_col, a, a_col);
    }
}

unsafe impl VecZnxDftCopyImpl<Self> for NTT120Avx {
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

unsafe impl VecZnxDftZeroImpl<Self> for NTT120Avx {
    fn vec_znx_dft_zero_impl<R>(_module: &Module<Self>, res: &mut R, res_col: usize)
    where
        R: VecZnxDftToMut<Self>,
    {
        ntt120_vec_znx_dft_zero::<R, Self>(res, res_col);
    }
}
