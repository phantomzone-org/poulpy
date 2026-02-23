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
        Backend, Data, Module, Scratch, VecZnxBig, VecZnxBigToMut, VecZnxDft, VecZnxDftOwned, VecZnxDftToMut, VecZnxDftToRef,
        VecZnxToRef, ZnxInfos, ZnxViewMut,
    },
    oep::{
        TakeSliceImpl, VecZnxDftAddImpl, VecZnxDftAddInplaceImpl, VecZnxDftAddScaledInplaceImpl, VecZnxDftAllocBytesImpl,
        VecZnxDftAllocImpl, VecZnxDftApplyImpl, VecZnxDftCopyImpl, VecZnxDftFromBytesImpl, VecZnxDftSubImpl,
        VecZnxDftSubInplaceImpl, VecZnxDftSubNegateInplaceImpl, VecZnxDftZeroImpl, VecZnxIdftApplyConsumeImpl,
        VecZnxIdftApplyImpl, VecZnxIdftApplyTmpAImpl, VecZnxIdftApplyTmpBytesImpl,
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

/// In-place CRT-compact all NTT blocks from Q120b (32 bytes/coeff) to i128 (16 bytes/coeff).
///
/// For each block `k` in `0..n_blocks`, in order:
///
/// 1. Applies the inverse NTT to the Q120b block in-place (4n u64 values).
/// 2. CRT-reconstructs each coefficient from 4 CRT residues to one `i128` and
///    writes it to the destination offset (`2*n*k` in u64 units = `16*n*k` in bytes).
///
/// # Ordering invariant
///
/// Blocks must be processed in order `k = 0, 1, ..., n_blocks-1`. For `k ≥ 1`
/// the destination range `[16nk, 16n(k+1))` never overlaps the source range
/// `[32nk, 32n(k+1))`.  For `k = 0` all four residues of each coefficient are
/// read into locals before the i128 is written.
///
/// # Safety
///
/// - `u64_ptr` must be valid for reads and writes of at least `4 * n * n_blocks` u64 values.
/// - The backing allocation must be at least 16-byte aligned (guaranteed by `DEFAULTALIGN = 64`).
/// - No other references to the same memory may be live during this call.
unsafe fn compact_all_blocks(n: usize, n_blocks: usize, u64_ptr: *mut u64, table: &NttTableInv<Primes30>) {
    // Precompute CRT reconstruction constants once for all blocks.
    let q: [i128; 4] = Primes30::Q.map(|qi| qi as i128);
    let total_q: i128 = q[0] * q[1] * q[2] * q[3];
    let qm: [i128; 4] = [q[1] * q[2] * q[3], q[0] * q[2] * q[3], q[0] * q[1] * q[3], q[0] * q[1] * q[2]];
    let crt: [i128; 4] = Primes30::CRT_CST.map(|c| c as i128);
    let half: i128 = (total_q + 1) / 2;

    for k in 0..n_blocks {
        let src_start = 4 * n * k; // index into u64 array: start of DFT block k
        let dst_start = 2 * n * k; // index into u64 array: start of Big block k

        // Step 1: inverse NTT in-place on the Q120b block.
        {
            let blk: &mut [u64] = unsafe { std::slice::from_raw_parts_mut(u64_ptr.add(src_start), 4 * n) };
            intt_ref::<Primes30>(table, blk);
        } // mutable borrow ends here

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

            // CRT reconstruction (matches b_to_znx128_ref).
            // v = Σ_k  ((x_k % Q[k]) * CRT_CST[k]) % Q[k]  *  QM[k]
            let mut v: i128 = 0;
            v += (x0 % Primes30::Q[0] as u64) as i128 * crt[0] % q[0] * qm[0];
            v += (x1 % Primes30::Q[1] as u64) as i128 * crt[1] % q[1] * qm[1];
            v += (x2 % Primes30::Q[2] as u64) as i128 * crt[2] % q[2] * qm[2];
            v += (x3 % Primes30::Q[3] as u64) as i128 * crt[3] % q[3] * qm[3];
            v %= total_q;
            let val: i128 = if v >= half { v - total_q } else { v };

            // Write i128 to destination offset (16-byte aligned by DEFAULTALIGN=64).
            unsafe { (u64_ptr.add(dst_start + 2 * c) as *mut i128).write_unaligned(val) };
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Allocation
// ──────────────────────────────────────────────────────────────────────────────

unsafe impl VecZnxDftFromBytesImpl<Self> for NTT120Ref {
    fn vec_znx_dft_from_bytes_impl(n: usize, cols: usize, size: usize, bytes: Vec<u8>) -> VecZnxDftOwned<Self> {
        VecZnxDft::<Vec<u8>, Self>::from_bytes(n, cols, size, bytes)
    }
}

unsafe impl VecZnxDftAllocBytesImpl<Self> for NTT120Ref {
    fn vec_znx_dft_bytes_of_impl(n: usize, cols: usize, size: usize) -> usize {
        Self::layout_prep_word_count() * n * cols * size * size_of::<<NTT120Ref as Backend>::ScalarPrep>()
    }
}

unsafe impl VecZnxDftAllocImpl<Self> for NTT120Ref {
    fn vec_znx_dft_alloc_impl(n: usize, cols: usize, size: usize) -> VecZnxDftOwned<Self> {
        VecZnxDftOwned::alloc(n, cols, size)
    }
}

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
