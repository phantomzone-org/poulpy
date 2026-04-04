//! NTT-domain ring element vector operations for [`NTTIfmaRef`](crate::NTTIfmaRef).
//!
//! Implements the `VecZnxDft*` and `VecZnxIdft*` OEP traits. `VecZnxDft` stores
//! ring element vectors in the NTT domain (`ScalarPrep = Q120bScalar`), where
//! polynomial multiplication reduces to component-wise lazy-modular multiplication
//! over three CRT residues (with one padding lane).
//!
//! Operations include:
//!
//! - **Forward NTT**: integer-domain `VecZnx` -> NTT-domain `VecZnxDft`, using
//!   the IFMA 3-prime backend.
//! - **Inverse NTT**: `VecZnxDft` -> `VecZnxBig` (large-coefficient), with variants
//!   that consume, borrow, or use the input as temporary storage.
//! - **NTT-domain arithmetic**: add, sub, negate, scaled-add, copy, zero -- delegated
//!   to the ntt120 generic functions since `NTTIfmaRef` implements the same
//!   `Ntt{Add,Sub,...}` traits.

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
    reference::ntt_ifma::{
        ntt::NttIfmaTableInv,
        primes::{PrimeSetIfma, Primes40},
        vec_znx_dft::{
            NttIfmaModuleHandle, ntt_ifma_vec_znx_dft_apply, ntt_ifma_vec_znx_idft_apply, ntt_ifma_vec_znx_idft_apply_tmp_bytes,
            ntt_ifma_vec_znx_idft_apply_tmpa,
        },
    },
    reference::ntt120::vec_znx_dft::{
        ntt120_vec_znx_dft_add, ntt120_vec_znx_dft_add_inplace, ntt120_vec_znx_dft_add_scaled_inplace, ntt120_vec_znx_dft_copy,
        ntt120_vec_znx_dft_sub, ntt120_vec_znx_dft_sub_inplace, ntt120_vec_znx_dft_sub_negate_inplace, ntt120_vec_znx_dft_zero,
    },
};

use crate::NTTIfmaRef;

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

/// In-place CRT-compact all NTT blocks from Q120b (32 bytes/coeff) to i128 (16 bytes/coeff).
///
/// For each block `k` in `0..n_blocks`, in order:
///
/// 1. Applies the inverse NTT to the 3-prime CRT block in-place.
/// 2. For each coefficient, uses Garner's algorithm to reconstruct the i128 value
///    from the three CRT residues.
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
unsafe fn compact_all_blocks(n: usize, n_blocks: usize, u64_ptr: *mut u64, table: &NttIfmaTableInv<Primes40>) {
    use poulpy_hal::reference::ntt_ifma::ntt::intt_ifma_ref;

    for k in 0..n_blocks {
        let src_start = 4 * n * k; // u64 index: start of DFT block k
        let dst_start = 2 * n * k; // u64 index: start of Big block k

        // Step 1: inverse NTT in-place.
        {
            let blk: &mut [u64] = unsafe { std::slice::from_raw_parts_mut(u64_ptr.add(src_start), 4 * n) };
            intt_ifma_ref::<Primes40>(table, blk);
        }

        // Step 2: Garner CRT-compact 4n u64s -> n i128s.
        for c in 0..n {
            // Read all three residues (+ padding) before any write (critical for k=0, c=0).
            let (r0, r1, r2) = unsafe {
                (
                    *u64_ptr.add(src_start + 4 * c),
                    *u64_ptr.add(src_start + 4 * c + 1),
                    *u64_ptr.add(src_start + 4 * c + 2),
                )
            };

            // Reduce residues mod their respective primes
            let r0 = r0 % Q0;
            let r1 = r1 % Q1;
            let r2 = r2 % Q2;

            // Garner step 1: v0 = r0
            let v0 = r0 as u128;

            // Garner step 2: v1 = ((r1 - v0) * INV01) mod Q1
            let r1_mod = r1 as u128;
            let v0_mod = v0 % Q1 as u128;
            let diff1 = (r1_mod + Q1 as u128 - v0_mod) % Q1 as u128;
            let v1 = ((diff1 * INV01 as u128) % Q1 as u128) as u64;

            // Garner step 3: v2 = ((r2 - v0 - v1*Q0) * INV012) mod Q2
            let partial = (v0 + v1 as u128 * Q0 as u128) % Q2 as u128;
            let r2_mod = r2 as u128;
            let diff2 = (r2_mod + Q2 as u128 - partial) % Q2 as u128;
            let v2 = ((diff2 * INV012 as u128) % Q2 as u128) as u64;

            // Reconstruct: result = v0 + v1*Q0 + v2*Q0*Q1
            let result_u128 = v0 + v1 as u128 * Q0 as u128 + v2 as u128 * Q01;

            // Symmetric lift to (-Q/2, Q/2].
            let val: i128 = if result_u128 > BIG_Q / 2 {
                result_u128 as i128 - BIG_Q as i128
            } else {
                result_u128 as i128
            };

            unsafe { (u64_ptr.add(dst_start + 2 * c) as *mut i128).write_unaligned(val) };
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Inverse NTT
// ──────────────────────────────────────────────────────────────────────────────

unsafe impl VecZnxIdftApplyTmpBytesImpl<Self> for NTTIfmaRef {
    fn vec_znx_idft_apply_tmp_bytes_impl(module: &Module<Self>) -> usize {
        ntt_ifma_vec_znx_idft_apply_tmp_bytes(module.n())
    }
}

unsafe impl VecZnxIdftApplyImpl<Self> for NTTIfmaRef
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
        let (tmp, _) = scratch.take_slice(ntt_ifma_vec_znx_idft_apply_tmp_bytes(module.n()) / size_of::<u64>());
        ntt_ifma_vec_znx_idft_apply::<R, A, Self>(module, res, res_col, a, a_col, tmp);
    }
}

unsafe impl VecZnxIdftApplyTmpAImpl<Self> for NTTIfmaRef {
    fn vec_znx_idft_apply_tmpa_impl<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &mut A, a_col: usize)
    where
        R: VecZnxBigToMut<Self>,
        A: VecZnxDftToMut<Self>,
    {
        ntt_ifma_vec_znx_idft_apply_tmpa::<R, A, Self>(module, res, res_col, a, a_col);
    }
}

unsafe impl VecZnxIdftApplyConsumeImpl<Self> for NTTIfmaRef {
    fn vec_znx_idft_apply_consume_impl<D: Data>(
        module: &Module<NTTIfmaRef>,
        mut a: VecZnxDft<D, NTTIfmaRef>,
    ) -> VecZnxBig<D, NTTIfmaRef>
    where
        VecZnxDft<D, NTTIfmaRef>: VecZnxDftToMut<NTTIfmaRef>,
    {
        let table = module.get_intt_ifma_table();

        // Obtain a mutable view, extract geometry and raw pointer, then release the borrow.
        let (n, n_blocks, u64_ptr) = {
            let mut a_mut: VecZnxDft<&mut [u8], NTTIfmaRef> = a.to_mut();
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

        // In-place: apply iNTT per block and CRT-compact Q120b -> i128.
        // After this, the first n*cols*size i128 values are at bytes [0, 16*n*cols*size).
        // SAFETY:
        //   - u64_ptr came from `a_mut.raw_mut()` which covers 4*n*cols*size u64s.
        //   - No other references to `a`'s data exist after the block above.
        //   - DEFAULTALIGN = 64 guarantees 64-byte alignment; all i128 writes are
        //     at multiples of 16 bytes (safe for write_unaligned on all platforms).
        unsafe { compact_all_blocks(n, n_blocks, u64_ptr, table) };

        // Reinterpret the (now compacted) buffer as VecZnxBig<D, NTTIfmaRef>.
        // The first n*cols*size i128s are at the correct offsets for VecZnxBig layout.
        a.into_big()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Forward NTT
// ──────────────────────────────────────────────────────────────────────────────

unsafe impl VecZnxDftApplyImpl<Self> for NTTIfmaRef {
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
        ntt_ifma_vec_znx_dft_apply::<R, A, Self>(module, step, offset, res, res_col, a, a_col);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// NTT-domain arithmetic (reused from ntt120 generic functions)
// ──────────────────────────────────────────────────────────────────────────────

unsafe impl VecZnxDftAddImpl<Self> for NTTIfmaRef {
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

unsafe impl VecZnxDftAddScaledInplaceImpl<Self> for NTTIfmaRef {
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

unsafe impl VecZnxDftAddInplaceImpl<Self> for NTTIfmaRef {
    fn vec_znx_dft_add_inplace_impl<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<Self>,
        A: VecZnxDftToRef<Self>,
    {
        ntt120_vec_znx_dft_add_inplace::<R, A, Self>(res, res_col, a, a_col);
    }
}

unsafe impl VecZnxDftSubImpl<Self> for NTTIfmaRef {
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

unsafe impl VecZnxDftSubInplaceImpl<Self> for NTTIfmaRef {
    fn vec_znx_dft_sub_inplace_impl<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<Self>,
        A: VecZnxDftToRef<Self>,
    {
        ntt120_vec_znx_dft_sub_inplace::<R, A, Self>(res, res_col, a, a_col);
    }
}

unsafe impl VecZnxDftSubNegateInplaceImpl<Self> for NTTIfmaRef {
    fn vec_znx_dft_sub_negate_inplace_impl<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<Self>,
        A: VecZnxDftToRef<Self>,
    {
        ntt120_vec_znx_dft_sub_negate_inplace::<R, A, Self>(res, res_col, a, a_col);
    }
}

unsafe impl VecZnxDftCopyImpl<Self> for NTTIfmaRef {
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

unsafe impl VecZnxDftZeroImpl<Self> for NTTIfmaRef {
    fn vec_znx_dft_zero_impl<R>(_module: &Module<Self>, res: &mut R, res_col: usize)
    where
        R: VecZnxDftToMut<Self>,
    {
        ntt120_vec_znx_dft_zero::<R, Self>(res, res_col);
    }
}
