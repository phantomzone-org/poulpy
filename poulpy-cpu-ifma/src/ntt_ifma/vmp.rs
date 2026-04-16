//! Vector-matrix product AVX512 kernels for [`NTTIfma`](crate::NTTIfma).
//!
//! This module contains AVX512-IFMA SIMD kernels for vector-matrix product
//! (VMP) operations in the IFMA NTT layout. These kernels override the generic
//! IFMA reference path with a backend-local 4-column tiled layout and a direct
//! row-strided apply kernel.

use bytemuck::{cast_slice, cast_slice_mut};
use core::arch::x86_64::{__m256i, __m512i, _mm512_add_epi64, _mm512_loadu_si512, _mm512_storeu_si512};
use std::mem::size_of;

use poulpy_cpu_ref::reference::ntt_ifma::{
    NttIfmaCFromB, NttIfmaDFTExecute, NttIfmaExtract1BlkContiguous, NttIfmaFromZnx64, NttIfmaMulBbc1ColX2, NttIfmaMulBbc2ColsX2,
    mat_vec::BbcIfmaMeta, primes::Primes40, types::Q_SHIFTED_IFMA, vec_znx_dft::NttIfmaModuleHandle,
};
use poulpy_hal::layouts::{
    DataViewMut, MatZnx, MatZnxToRef, Module, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef, VmpPMat, VmpPMatToMut, VmpPMatToRef,
    ZnxInfos, ZnxView, ZnxViewMut,
};

use super::{mat_vec_ifma::vec_mat4cols_product_x2_strided_ifma, ntt_ifma_avx512::cond_sub_2q_si512};

// ──────────────────────────────────────────────────────────────────────────────
// SIMD save helpers
// ──────────────────────────────────────────────────────────────────────────────

fn q2_shifted_vec_512() -> __m512i {
    let q = Q_SHIFTED_IFMA;
    let q2_512 = [q[0], q[1], q[2], q[3], q[0], q[1], q[2], q[3]];
    unsafe { _mm512_loadu_si512(q2_512.as_ptr() as *const __m512i) }
}

#[target_feature(enable = "avx512f")]
unsafe fn save_blk_overwrite(_n: usize, blk: usize, dst: &mut [u64], src: &[u64]) {
    let off = 8 * blk;
    let dst_ptr = unsafe { dst.as_mut_ptr().add(off) as *mut __m512i };
    let src_ptr = src.as_ptr() as *const __m512i;
    unsafe {
        _mm512_storeu_si512(dst_ptr, _mm512_loadu_si512(src_ptr));
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
    (16 + 24 * row_max) * size_of::<u64>()
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

    for row_i in 0..nrows {
        for col_i in 0..ncols {
            let pos = n * (row_i * ncols + col_i);
            crate::NTTIfma::ntt_ifma_from_znx64(tmp_b, &mat_i64[pos..pos + n]);
            crate::NTTIfma::ntt_ifma_dft_execute(module.get_ntt_ifma_table(), tmp_b);
            crate::NTTIfma::ntt_ifma_c_from_b(n, tmp_c, tmp_b);

            let tile = col_i / 4;
            let tile_width = (ncols - 4 * tile).min(4);
            let dst_base = tile * (nrows * 64) + row_i * (tile_width * 16) + (col_i % 4) * 16;

            for blk_j in 0..n_blks {
                let pmat_off = blk_j * block_stride + dst_base;
                pmat_u32[pmat_off..pmat_off + 16].copy_from_slice(&tmp_c[16 * blk_j..16 * blk_j + 16]);
            }
        }
    }
}

#[inline(always)]
fn tile_base_u32(nrows: usize, tile: usize) -> usize {
    tile * nrows * 64
}

#[inline(always)]
fn prefer_4col_strided_path(n: usize, row_max: usize, active_cols: usize) -> bool {
    const EXTRACT_COPY_CROSSOVER_BYTES: usize = 8 * 1024 * 1024;
    active_cols >= 4 && (n / 2) * row_max * 64 >= EXTRACT_COPY_CROSSOVER_BYTES
}

#[inline(always)]
fn copy_one_col_from_tiled(mat_blk_u32: &[u32], nrows: usize, ncols: usize, col: usize, dst: &mut [u32]) {
    let tile = col / 4;
    let tile_width = (ncols - 4 * tile).min(4);
    let col_in_tile = col % 4;
    let tile_base = tile_base_u32(nrows, tile);

    for row in 0..nrows {
        let src_off = tile_base + row * tile_width * 16 + col_in_tile * 16;
        let dst_off = row * 16;
        dst[dst_off..dst_off + 16].copy_from_slice(&mat_blk_u32[src_off..src_off + 16]);
    }
}

#[inline(always)]
fn copy_two_cols_from_tiled(mat_blk_u32: &[u32], nrows: usize, ncols: usize, col0: usize, col1: usize, dst: &mut [u32]) {
    let tile0 = col0 / 4;
    let tile1 = col1 / 4;
    let tile_width0 = (ncols - 4 * tile0).min(4);
    let tile_width1 = (ncols - 4 * tile1).min(4);
    let col0_in_tile = col0 % 4;
    let col1_in_tile = col1 % 4;
    let tile_base0 = tile_base_u32(nrows, tile0);
    let tile_base1 = tile_base_u32(nrows, tile1);

    for row in 0..nrows {
        let src0 = tile_base0 + row * tile_width0 * 16 + col0_in_tile * 16;
        let src1 = tile_base1 + row * tile_width1 * 16 + col1_in_tile * 16;
        let dst_off = row * 32;
        dst[dst_off..dst_off + 16].copy_from_slice(&mat_blk_u32[src0..src0 + 16]);
        dst[dst_off + 16..dst_off + 32].copy_from_slice(&mat_blk_u32[src1..src1 + 16]);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// IFMA-local VMP apply
// ──────────────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512vl")]
unsafe fn vmp_apply_core_simd<const OVERWRITE: bool>(
    n: usize,
    res_u64: &mut [u64],
    a_u64: &[u64],
    pmat_u32: &[u32],
    limb_offset: usize,
    nrows: usize,
    ncols: usize,
    meta: &BbcIfmaMeta<poulpy_cpu_ref::reference::ntt_ifma::primes::Primes40>,
    _tmp: &mut [u64],
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

    let block_stride = nrows * ncols * 16;
    let full_tiles = ncols / 4;

    for blk_j in 0..n_blks {
        let mat_blk_u32 = &pmat_u32[blk_j * block_stride..];
        let mut col_pmat = limb_offset;
        let mut col_res = 0usize;

        while col_pmat < col_max {
            let tile = col_pmat / 4;
            let tile_width = if tile < full_tiles { 4 } else { ncols - 4 * tile };
            let start_in_tile = col_pmat % 4;
            let cols_this_tile = (tile_width - start_in_tile).min(col_max - col_pmat);
            let tile_base = tile_base_u32(nrows, tile);
            let mut tile_output = [0u64; 32];

            unsafe {
                vec_mat4cols_product_x2_strided_ifma(
                    meta,
                    row_max,
                    n,
                    blk_j,
                    &mut tile_output[..8 * cols_this_tile],
                    a_u64,
                    &mat_blk_u32[tile_base..],
                    tile_width,
                    start_in_tile,
                    cols_this_tile,
                );
            }

            for tile_col in 0..cols_this_tile {
                let base = (col_res + tile_col) * 4 * n;
                let src = &tile_output[8 * tile_col..8 * (tile_col + 1)];
                if OVERWRITE {
                    unsafe { save_blk_overwrite(n, blk_j, &mut res_u64[base..], src) };
                } else {
                    unsafe { save_blk_add_simd(n, blk_j, &mut res_u64[base..], src) };
                }
            }

            col_pmat += cols_this_tile;
            col_res += cols_this_tile;
        }
    }

    if OVERWRITE {
        let active_cols = col_max.saturating_sub(limb_offset);
        for col in active_cols..res_size {
            res_u64[col * 4 * n..(col + 1) * 4 * n].fill(0);
        }
    }
}

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

    let (mat2cols_output, tmp) = tmp.split_at_mut(16);
    let (extracted_blk, staged_cols_u64) = tmp.split_at_mut(8 * row_max);
    let staged_cols_u32: &mut [u32] = cast_slice_mut(staged_cols_u64);
    let block_stride = nrows * ncols * 16;

    for blk_j in 0..n_blks {
        let mat_blk_u32 = &pmat_u32[blk_j * block_stride..];
        crate::NTTIfma::ntt_ifma_extract_1blk_contiguous(n, row_max, blk_j, extracted_blk, a_u64);
        let extracted_u32: &[u32] = cast_slice(extracted_blk);

        let start_col = limb_offset;
        let mut col_pmat = start_col;
        let mut col_res = 0usize;

        if !start_col.is_multiple_of(2) && col_pmat < col_max {
            copy_two_cols_from_tiled(
                mat_blk_u32,
                row_max,
                ncols,
                col_pmat - 1,
                col_pmat,
                &mut staged_cols_u32[..32 * row_max],
            );
            crate::NTTIfma::ntt_ifma_mul_bbc_2cols_x2(
                meta,
                row_max,
                mat2cols_output,
                extracted_u32,
                &staged_cols_u32[..32 * row_max],
            );
            let base = col_res * 4 * n;
            if OVERWRITE {
                unsafe { save_blk_overwrite(n, blk_j, &mut res_u64[base..], &mat2cols_output[8..16]) };
            } else {
                unsafe { save_blk_add_simd(n, blk_j, &mut res_u64[base..], &mat2cols_output[8..16]) };
            }
            col_pmat += 1;
            col_res += 1;
        }

        while col_pmat + 1 < col_max {
            copy_two_cols_from_tiled(
                mat_blk_u32,
                row_max,
                ncols,
                col_pmat,
                col_pmat + 1,
                &mut staged_cols_u32[..32 * row_max],
            );
            crate::NTTIfma::ntt_ifma_mul_bbc_2cols_x2(
                meta,
                row_max,
                mat2cols_output,
                extracted_u32,
                &staged_cols_u32[..32 * row_max],
            );
            let base0 = col_res * 4 * n;
            let base1 = (col_res + 1) * 4 * n;
            if OVERWRITE {
                unsafe {
                    save_blk_overwrite(n, blk_j, &mut res_u64[base0..], &mat2cols_output[0..8]);
                    save_blk_overwrite(n, blk_j, &mut res_u64[base1..], &mat2cols_output[8..16]);
                }
            } else {
                unsafe {
                    save_blk_add_simd(n, blk_j, &mut res_u64[base0..], &mat2cols_output[0..8]);
                    save_blk_add_simd(n, blk_j, &mut res_u64[base1..], &mat2cols_output[8..16]);
                }
            }
            col_pmat += 2;
            col_res += 2;
        }

        if col_pmat < col_max {
            copy_one_col_from_tiled(mat_blk_u32, row_max, ncols, col_pmat, &mut staged_cols_u32[..16 * row_max]);
            crate::NTTIfma::ntt_ifma_mul_bbc_1col_x2(
                meta,
                row_max,
                &mut mat2cols_output[0..8],
                extracted_u32,
                &staged_cols_u32[..16 * row_max],
            );
            let base = col_res * 4 * n;
            if OVERWRITE {
                unsafe { save_blk_overwrite(n, blk_j, &mut res_u64[base..], &mat2cols_output[0..8]) };
            } else {
                unsafe { save_blk_add_simd(n, blk_j, &mut res_u64[base..], &mat2cols_output[0..8]) };
            }
        }
    }

    if OVERWRITE {
        let active_cols = col_max.saturating_sub(limb_offset);
        for col in active_cols..res_size {
            res_u64[col * 4 * n..(col + 1) * 4 * n].fill(0);
        }
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
    let row_max = nrows.min(a_ref.size());
    let col_max = ncols.min(res_size + limb_offset);
    let active_cols = col_max.saturating_sub(limb_offset);

    let res_u64: &mut [u64] = cast_slice_mut(res_ref.raw_mut());
    let a_u64: &[u64] = cast_slice(a_ref.raw());
    let pmat_u32: &[u32] = cast_slice(pmat_ref.raw());

    unsafe {
        if prefer_4col_strided_path(n, row_max, active_cols) {
            vmp_apply_core_simd::<true>(
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
        } else {
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
    let row_max = nrows.min(a_ref.size());
    let col_max = ncols.min(res_size + limb_offset);
    let active_cols = col_max.saturating_sub(limb_offset);

    let res_u64: &mut [u64] = cast_slice_mut(res_ref.raw_mut());
    let a_u64: &[u64] = cast_slice(a_ref.raw());
    let pmat_u32: &[u32] = cast_slice(pmat_ref.raw());

    unsafe {
        if prefer_4col_strided_path(n, row_max, active_cols) {
            vmp_apply_core_simd::<false>(
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
        } else {
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
}
