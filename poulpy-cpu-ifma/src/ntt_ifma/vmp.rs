//! Vector-matrix product AVX512 kernels for [`NTTIfma`](crate::NTTIfma).
//!
//! This module contains AVX512-IFMA SIMD kernels for vector-matrix product
//! (VMP) operations in the IFMA NTT layout. These kernels override the generic
//! IFMA reference path with a backend-local 4-column tiled layout and a direct
//! row-strided apply kernel.

use bytemuck::{cast_slice, cast_slice_mut};
use core::arch::x86_64::{
    __m256i, __m512i, _mm512_add_epi64, _mm512_loadu_si512, _mm512_storeu_si512, _mm512_stream_si512, _mm_sfence,
};
use std::mem::size_of;

use poulpy_cpu_ref::reference::ntt_ifma::{
    NttIfmaCFromB, NttIfmaDFTExecute, NttIfmaExtract1BlkContiguous, NttIfmaFromZnx64, NttIfmaMulBbc1ColX2, NttIfmaMulBbc2ColsX2,
    mat_vec::BbcIfmaMeta, primes::Primes40, types::Q_SHIFTED_IFMA, vec_znx_dft::NttIfmaModuleHandle,
};
use poulpy_hal::layouts::{
    DataView, DataViewMut, MatZnx, MatZnxToRef, Module, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef, VmpPMat, VmpPMatToMut,
    VmpPMatToRef, ZnxInfos, ZnxView, ZnxViewMut,
};

use super::ntt_ifma_avx512::cond_sub_2q_si512;

// ──────────────────────────────────────────────────────────────────────────────
// SIMD save helpers
// ──────────────────────────────────────────────────────────────────────────────

fn q2_shifted_vec_512() -> __m512i {
    let q = Q_SHIFTED_IFMA;
    let q2_512 = [q[0], q[1], q[2], q[3], q[0], q[1], q[2], q[3]];
    unsafe { _mm512_loadu_si512(q2_512.as_ptr() as *const __m512i) }
}

/// Non-temporal write: bypass the cache so the result lines do not evict
/// matrix data the kernel still needs. `dst.as_mut_ptr().add(8 * blk)` is
/// 64-byte aligned because `VecZnxDft` storage is `DEFAULTALIGN`-aligned and
/// `8 * blk * size_of::<u64>() = 64 * blk` keeps that alignment. The
/// `_mm_sfence` to make these stores globally ordered is issued once at the
/// end of the apply loop, not per call.
#[target_feature(enable = "avx512f")]
unsafe fn save_blk_overwrite(_n: usize, blk: usize, dst: &mut [u64], src: &[u64]) {
    let off = 8 * blk;
    let dst_ptr = unsafe { dst.as_mut_ptr().add(off) as *mut __m512i };
    let src_ptr = src.as_ptr() as *const __m512i;
    unsafe {
        _mm512_stream_si512(dst_ptr, _mm512_loadu_si512(src_ptr));
    }
}

#[target_feature(enable = "avx512f,avx512vl")]
unsafe fn save_blk_add_simd(_n: usize, blk: usize, dst: &mut [u64], src: &[u64]) {
    unsafe {
        let q2_512 = q2_shifted_vec_512();
        let off = 8 * blk;
        let dst_ptr = dst.as_mut_ptr().add(off) as *mut __m256i;
        let src_ptr = src.as_ptr() as *const __m256i;

        let dst_ptr_512 = dst_ptr as *mut __m512i;
        let src_ptr_512 = src_ptr as *const __m512i;
        let d = _mm512_loadu_si512(dst_ptr_512 as *const __m512i);
        let s = _mm512_loadu_si512(src_ptr_512);
        let d_red = cond_sub_2q_si512(d, q2_512);
        _mm512_storeu_si512(dst_ptr_512, _mm512_add_epi64(d_red, s));
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// IFMA-local VMP prepare
// ──────────────────────────────────────────────────────────────────────────────

pub(crate) fn vmp_prepare_tmp_bytes_ifma(n: usize) -> usize {
    8 * n * size_of::<u64>()
}

pub(crate) fn vmp_apply_tmp_bytes_ifma(a_size: usize, b_rows: usize, b_cols_in: usize) -> usize {
    let row_max = a_size.min(b_rows) * b_cols_in;
    // 16 u64 for the 2-col kernel output + 8 * row_max u64 for the per-block
    // packed `a`. The 24x figure was the legacy estimate that included a
    // per-call matrix staging buffer that no longer exists.
    (16 + 8 * row_max) * size_of::<u64>()
}

pub(crate) fn vmp_prepare_ifma<R, A>(module: &Module<crate::NTTIfma>, res: &mut R, a: &A, tmp: &mut [u64])
where
    R: VmpPMatToMut<crate::NTTIfma>,
    A: MatZnxToRef,
{
    let mut res: VmpPMat<&mut [u8], crate::NTTIfma> = res.to_mut();
    let a: MatZnx<&[u8]> = a.to_ref();
    let n = res.n();
    let nrows = a.cols_in() * a.rows();
    let ncols = a.cols_out() * a.size();
    let n_blks = n / 2;
    let block_stride = nrows * ncols * 16;

    let (tmp_b, tmp_c_u64) = tmp.split_at_mut(4 * n);
    let tmp_c: &mut [u32] = cast_slice_mut(tmp_c_u64);
    let mat_i64: &[i64] = a.raw();
    let pmat_u32: &mut [u32] = cast_slice_mut(res.data_mut());

    // 2-col tile layout: tiles of up to 2 consecutive output columns, row-major
    // within each tile. Per (block, tile, row) the two columns are adjacent in
    // memory, which exactly matches the [col0_pair, col1_pair] input layout of
    // [`vec_mat2cols_product_x2_bbc_ifma`] — the 2-col apply path can read
    // straight from the tile with no staging copy. Only the last tile can be
    // partial (width 1) when `ncols` is odd.
    for row_i in 0..nrows {
        for col_i in 0..ncols {
            let pos = n * (row_i * ncols + col_i);
            crate::NTTIfma::ntt_ifma_from_znx64(tmp_b, &mat_i64[pos..pos + n]);
            crate::NTTIfma::ntt_ifma_dft_execute(module.get_ntt_ifma_table(), tmp_b);
            crate::NTTIfma::ntt_ifma_c_from_b(n, tmp_c, tmp_b);

            let tile = col_i / 2;
            let tile_width = (ncols - 2 * tile).min(2);
            let dst_base = tile * (nrows * 32) + row_i * (tile_width * 16) + (col_i % 2) * 16;

            for blk_j in 0..n_blks {
                let pmat_off = blk_j * block_stride + dst_base;
                pmat_u32[pmat_off..pmat_off + 16].copy_from_slice(&tmp_c[16 * blk_j..16 * blk_j + 16]);
            }
        }
    }
}

#[inline(always)]
fn tile_base_u32(nrows: usize, tile: usize) -> usize {
    tile * nrows * 32
}

#[inline(always)]
fn tile_width_for(ncols: usize, tile: usize) -> usize {
    (ncols - 2 * tile).min(2)
}

// ──────────────────────────────────────────────────────────────────────────────
// IFMA-local VMP apply
// ──────────────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512vl")]
unsafe fn vmp_apply_core_2col_simd<const OVERWRITE: bool>(
    n: usize,
    res_u64: &mut [u64],
    a_u64: &[u64],
    pmat_u32: &[u32],
    limb_offset: usize,
    nrows: usize,
    ncols: usize,
    meta: &BbcIfmaMeta<Primes40>,
    tmp: &mut [u64],
) {
    if n < 2 {
        return;
    }

    let n_blks = n / 2;
    let a_size = a_u64.len() / (4 * n);
    let res_size = res_u64.len() / (4 * n);
    let row_max = nrows.min(a_size);
    let col_max = ncols.min(res_size + limb_offset);

    if limb_offset >= col_max {
        if OVERWRITE {
            res_u64.fill(0);
        }
        return;
    }

    // Scratch: 16 u64 for the 2-col kernel output + 8 * row_max u64 for the
    // per-block packed `a`. No staging buffer — the SIMD kernels read tiles
    // directly thanks to the 2-col matrix layout.
    let (kernel_output, extracted_blk_u64) = tmp.split_at_mut(16);
    let extracted_blk = &mut extracted_blk_u64[..8 * row_max];
    let block_stride = nrows * ncols * 16;

    for blk_j in 0..n_blks {
        let mat_blk_u32 = &pmat_u32[blk_j * block_stride..];
        crate::NTTIfma::ntt_ifma_extract_1blk_contiguous(n, row_max, blk_j, extracted_blk, a_u64);
        let extracted_u32: &[u32] = cast_slice(extracted_blk);

        let start_col = limb_offset;
        let mut col_pmat = start_col;
        let mut col_res = 0usize;

        // Odd start: current col sits in the second slot of a 2-col tile. Call
        // the 2-col kernel on that tile and keep only the second result lane.
        if !start_col.is_multiple_of(2) && col_pmat < col_max {
            let tile = (col_pmat - 1) / 2;
            let tile_base = tile_base_u32(nrows, tile);
            debug_assert_eq!(tile_width_for(ncols, tile), 2);
            let y = &mat_blk_u32[tile_base..];
            crate::NTTIfma::ntt_ifma_mul_bbc_2cols_x2(meta, row_max, &mut kernel_output[..16], extracted_u32, y);
            let base = col_res * 4 * n;
            if OVERWRITE {
                unsafe { save_blk_overwrite(n, blk_j, &mut res_u64[base..], &kernel_output[8..16]) };
            } else {
                unsafe { save_blk_add_simd(n, blk_j, &mut res_u64[base..], &kernel_output[8..16]) };
            }
            col_pmat += 1;
            col_res += 1;
        }

        while col_pmat + 1 < col_max {
            let tile = col_pmat / 2;
            debug_assert_eq!(tile_width_for(ncols, tile), 2);
            let tile_base = tile_base_u32(nrows, tile);
            let y = &mat_blk_u32[tile_base..];
            crate::NTTIfma::ntt_ifma_mul_bbc_2cols_x2(meta, row_max, &mut kernel_output[..16], extracted_u32, y);
            let base0 = col_res * 4 * n;
            let base1 = (col_res + 1) * 4 * n;
            if OVERWRITE {
                unsafe {
                    save_blk_overwrite(n, blk_j, &mut res_u64[base0..], &kernel_output[0..8]);
                    save_blk_overwrite(n, blk_j, &mut res_u64[base1..], &kernel_output[8..16]);
                }
            } else {
                unsafe {
                    save_blk_add_simd(n, blk_j, &mut res_u64[base0..], &kernel_output[0..8]);
                    save_blk_add_simd(n, blk_j, &mut res_u64[base1..], &kernel_output[8..16]);
                }
            }
            col_pmat += 2;
            col_res += 2;
        }

        // 1-col tail when ncols is odd: sits alone in a width-1 tile (or in
        // slot 0 of an incomplete pair at the right edge).
        if col_pmat < col_max {
            let tile = col_pmat / 2;
            let tile_base = tile_base_u32(nrows, tile);
            let y = &mat_blk_u32[tile_base..];
            crate::NTTIfma::ntt_ifma_mul_bbc_1col_x2(meta, row_max, &mut kernel_output[0..8], extracted_u32, y);
            let base = col_res * 4 * n;
            if OVERWRITE {
                unsafe { save_blk_overwrite(n, blk_j, &mut res_u64[base..], &kernel_output[0..8]) };
            } else {
                unsafe { save_blk_add_simd(n, blk_j, &mut res_u64[base..], &kernel_output[0..8]) };
            }
        }
    }

    if OVERWRITE {
        let active_cols = col_max.saturating_sub(limb_offset);
        for col in active_cols..res_size {
            res_u64[col * 4 * n..(col + 1) * 4 * n].fill(0);
        }
        // Order the non-temporal stores from `save_blk_overwrite` against any
        // subsequent loads of the result buffer.
        _mm_sfence();
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Public IFMA hooks
// ──────────────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub(crate) fn vmp_apply_dft_to_dft_ifma<R, A, C>(
    module: &Module<crate::NTTIfma>,
    res: &mut R,
    a: &A,
    pmat: &C,
    limb_offset: usize,
    tmp: &mut [u64],
) where
    R: VecZnxDftToMut<crate::NTTIfma>,
    A: VecZnxDftToRef<crate::NTTIfma>,
    C: VmpPMatToRef<crate::NTTIfma>,
{
    let mut res_ref: VecZnxDft<&mut [u8], crate::NTTIfma> = res.to_mut();
    let a_ref: VecZnxDft<&[u8], crate::NTTIfma> = a.to_ref();
    let pmat_ref: VmpPMat<&[u8], crate::NTTIfma> = pmat.to_ref();

    let n = res_ref.n();
    let res_size = res_ref.size();
    let nrows = pmat_ref.rows() * pmat_ref.cols_in();
    let ncols = pmat_ref.cols_out() * pmat_ref.size();
    let limb_offset = limb_offset * pmat_ref.cols_out();
    let _ = res_size;

    let res_u64: &mut [u64] = cast_slice_mut(res_ref.raw_mut());
    let a_u64: &[u64] = cast_slice(a_ref.raw());
    let pmat_u32: &[u32] = cast_slice(pmat_ref.data().as_ref());

    unsafe {
        vmp_apply_core_2col_simd::<true>(
            n,
            res_u64,
            a_u64,
            pmat_u32,
            limb_offset,
            nrows,
            ncols,
            module.get_bbc_ifma_meta(),
            tmp,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn vmp_apply_dft_to_dft_accumulate_ifma<R, A, C>(
    module: &Module<crate::NTTIfma>,
    res: &mut R,
    a: &A,
    pmat: &C,
    limb_offset: usize,
    tmp: &mut [u64],
) where
    R: VecZnxDftToMut<crate::NTTIfma>,
    A: VecZnxDftToRef<crate::NTTIfma>,
    C: VmpPMatToRef<crate::NTTIfma>,
{
    let mut res_ref: VecZnxDft<&mut [u8], crate::NTTIfma> = res.to_mut();
    let a_ref: VecZnxDft<&[u8], crate::NTTIfma> = a.to_ref();
    let pmat_ref: VmpPMat<&[u8], crate::NTTIfma> = pmat.to_ref();

    let n = res_ref.n();
    let res_size = res_ref.size();
    let nrows = pmat_ref.rows() * pmat_ref.cols_in();
    let ncols = pmat_ref.cols_out() * pmat_ref.size();
    let limb_offset = limb_offset * pmat_ref.cols_out();
    let _ = res_size;

    let res_u64: &mut [u64] = cast_slice_mut(res_ref.raw_mut());
    let a_u64: &[u64] = cast_slice(a_ref.raw());
    let pmat_u32: &[u32] = cast_slice(pmat_ref.data().as_ref());

    unsafe {
        vmp_apply_core_2col_simd::<false>(
            n,
            res_u64,
            a_u64,
            pmat_u32,
            limb_offset,
            nrows,
            ncols,
            module.get_bbc_ifma_meta(),
            tmp,
        );
    }
}
