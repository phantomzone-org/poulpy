//! VMP operations for the IFMA backend.
//!
//! Mirrors [`crate::reference::ntt120::vmp`] with IFMA trait bounds.

use bytemuck::{cast_slice, cast_slice_mut};
use std::mem::size_of;

use crate::{
    layouts::{
        Backend, DataViewMut, MatZnx, MatZnxToRef, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef, VmpPMat, VmpPMatToMut,
        VmpPMatToRef, ZnxInfos, ZnxView, ZnxViewMut,
    },
    reference::ntt120::types::Q120bScalar,
};

use super::{
    NttIfmaCFromB, NttIfmaDFTExecute, NttIfmaExtract1BlkContiguous, NttIfmaFromZnx64, NttIfmaMulBbc1ColX2, NttIfmaMulBbc2ColsX2,
    mat_vec::BbcIfmaMeta, ntt::NttIfmaTable, primes::Primes42, types::Q_SHIFTED_IFMA, vec_znx_dft::NttIfmaModuleHandle,
};

pub fn ntt_ifma_vmp_prepare_tmp_bytes(n: usize) -> usize {
    4 * n * size_of::<u64>()
}

/// Encode a polynomial matrix into prepared format.
pub fn ntt_ifma_vmp_prepare<R, A, BE>(module: &impl NttIfmaModuleHandle, res: &mut R, a: &A, tmp: &mut [u64])
where
    BE: Backend<ScalarPrep = Q120bScalar> + NttIfmaDFTExecute<NttIfmaTable<Primes42>> + NttIfmaFromZnx64 + NttIfmaCFromB,
    R: VmpPMatToMut<BE>,
    A: MatZnxToRef,
{
    let mut res: VmpPMat<&mut [u8], BE> = res.to_mut();
    let a: MatZnx<&[u8]> = a.to_ref();
    let n = res.n();

    let nrows: usize = a.cols_in() * a.rows();
    let ncols: usize = a.cols_out() * a.size();
    let n_blks: usize = n / 2;
    let offset: usize = nrows * ncols * 16;

    let mat_i64: &[i64] = a.raw();
    let pmat_u32: &mut [u32] = cast_slice_mut(res.data_mut());

    for row_i in 0..nrows {
        for col_i in 0..ncols {
            let pos = n * (row_i * ncols + col_i);

            BE::ntt_ifma_from_znx64(tmp, &mat_i64[pos..pos + n]);
            BE::ntt_ifma_dft_execute(module.get_ntt_ifma_table(), tmp);

            let tmp_q120c: Vec<u32> = {
                let mut v = vec![0u32; 8 * n];
                BE::ntt_ifma_c_from_b(n, &mut v, tmp);
                v
            };

            let dst_base: usize = if col_i == ncols - 1 && !ncols.is_multiple_of(2) {
                col_i * nrows * 16 + row_i * 16
            } else {
                (col_i / 2) * (nrows * 32) + row_i * 32 + (col_i % 2) * 16
            };

            for blk_j in 0..n_blks {
                let pmat_off = dst_base + blk_j * offset;
                pmat_u32[pmat_off..pmat_off + 16].copy_from_slice(&tmp_q120c[16 * blk_j..16 * blk_j + 16]);
            }
        }
    }
}

pub fn ntt_ifma_vmp_apply_dft_to_dft_tmp_bytes(a_size: usize, b_rows: usize, b_cols_in: usize) -> usize {
    let row_max = a_size.min(b_rows) * b_cols_in;
    (16 + 8 * row_max) * size_of::<u64>()
}

#[inline(always)]
fn save_blk_overwrite(_n: usize, blk: usize, dst: &mut [u64], src: &[u64]) {
    dst[8 * blk..8 * blk + 8].copy_from_slice(&src[..8]);
}

#[inline(always)]
fn save_blk_add(_n: usize, blk: usize, dst: &mut [u64], src: &[u64]) {
    for i in 0..8 {
        let k = i % 4;
        let qs = if k < 3 { Q_SHIFTED_IFMA[k] } else { 1 };
        dst[8 * blk + i] = if qs > 0 { dst[8 * blk + i] % qs + src[i] % qs } else { 0 };
    }
}

#[allow(clippy::too_many_arguments)]
fn vmp_apply_core<const OVERWRITE: bool, BE>(
    n: usize,
    res_u64: &mut [u64],
    a_u64: &[u64],
    pmat_u32: &[u32],
    limb_offset: usize,
    nrows: usize,
    ncols: usize,
    meta: &BbcIfmaMeta<Primes42>,
    tmp: &mut [u64],
) where
    BE: NttIfmaExtract1BlkContiguous + NttIfmaMulBbc1ColX2 + NttIfmaMulBbc2ColsX2,
{
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

    let (mat2cols_output, extracted_blk) = tmp.split_at_mut(16);
    let offset = nrows * ncols * 16;

    for blk_j in 0..n_blks {
        let mat_blk_u32 = &pmat_u32[blk_j * offset..];
        BE::ntt_ifma_extract_1blk_contiguous(n, row_max, blk_j, extracted_blk, a_u64);
        let extracted_u32: &[u32] = cast_slice(extracted_blk);

        // Process paired columns
        let start_col = limb_offset;
        let mut col_pmat = start_col;
        let mut col_res = 0usize;

        // Handle odd start
        if !start_col.is_multiple_of(2) && col_pmat < col_max {
            let col_offset = (col_pmat - 1) * (nrows * 16);
            BE::ntt_ifma_mul_bbc_2cols_x2(meta, row_max, mat2cols_output, extracted_u32, &mat_blk_u32[col_offset..]);
            let base = col_res * 4 * n;
            if OVERWRITE {
                save_blk_overwrite(n, blk_j, &mut res_u64[base..], &mat2cols_output[8..16]);
            } else {
                save_blk_add(n, blk_j, &mut res_u64[base..], &mat2cols_output[8..16]);
            }
            col_pmat += 1;
            col_res += 1;
        }

        // Process paired columns
        while col_pmat + 1 < col_max {
            let col_offset = col_pmat * (nrows * 16);
            BE::ntt_ifma_mul_bbc_2cols_x2(meta, row_max, mat2cols_output, extracted_u32, &mat_blk_u32[col_offset..]);
            let base0 = col_res * 4 * n;
            let base1 = (col_res + 1) * 4 * n;
            if OVERWRITE {
                save_blk_overwrite(n, blk_j, &mut res_u64[base0..], &mat2cols_output[0..8]);
                save_blk_overwrite(n, blk_j, &mut res_u64[base1..], &mat2cols_output[8..16]);
            } else {
                save_blk_add(n, blk_j, &mut res_u64[base0..], &mat2cols_output[0..8]);
                save_blk_add(n, blk_j, &mut res_u64[base1..], &mat2cols_output[8..16]);
            }
            col_pmat += 2;
            col_res += 2;
        }

        // Handle last odd column
        if col_pmat < col_max {
            let col_offset = col_pmat * (nrows * 16);
            BE::ntt_ifma_mul_bbc_1col_x2(
                meta,
                row_max,
                &mut mat2cols_output[0..8],
                extracted_u32,
                &mat_blk_u32[col_offset..],
            );
            let base = col_res * 4 * n;
            if OVERWRITE {
                save_blk_overwrite(n, blk_j, &mut res_u64[base..], &mat2cols_output[0..8]);
            } else {
                save_blk_add(n, blk_j, &mut res_u64[base..], &mat2cols_output[0..8]);
            }
        }
    }

    // Zero remaining output columns in overwrite mode
    if OVERWRITE {
        let active_cols = col_max.saturating_sub(limb_offset);
        for col in active_cols..res_size {
            res_u64[col * 4 * n..(col + 1) * 4 * n].fill(0);
        }
    }
}

/// NTT-domain vector-matrix product: `res = a · pmat`.
pub fn ntt_ifma_vmp_apply_dft_to_dft<R, A, C, BE>(
    module: &impl NttIfmaModuleHandle,
    res: &mut R,
    a: &A,
    pmat: &C,
    limb_offset: usize,
    tmp: &mut [u64],
) where
    BE: Backend<ScalarPrep = Q120bScalar> + NttIfmaExtract1BlkContiguous + NttIfmaMulBbc1ColX2 + NttIfmaMulBbc2ColsX2,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
    C: VmpPMatToRef<BE>,
{
    let mut res_ref: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a_ref: VecZnxDft<&[u8], BE> = a.to_ref();
    let pmat_ref: VmpPMat<&[u8], BE> = pmat.to_ref();

    let n = res_ref.n();
    let nrows = pmat_ref.rows() * pmat_ref.cols_in();
    let ncols = pmat_ref.cols_out() * pmat_ref.size();

    let res_u64: &mut [u64] = cast_slice_mut(res_ref.raw_mut());
    let a_u64: &[u64] = cast_slice(a_ref.raw());
    let pmat_u32: &[u32] = cast_slice(pmat_ref.raw());

    let meta = module.get_bbc_ifma_meta();

    vmp_apply_core::<true, BE>(
        n,
        res_u64,
        a_u64,
        pmat_u32,
        limb_offset * pmat_ref.cols_out(),
        nrows,
        ncols,
        meta,
        tmp,
    );
}

/// Zero a VmpPMat.
pub fn ntt_ifma_vmp_zero<R, BE>(res: &mut R)
where
    BE: Backend<ScalarPrep = Q120bScalar>,
    R: VmpPMatToMut<BE>,
{
    res.to_mut().data_mut().as_mut().fill(0);
}
