//! Bivariate convolution operations for the IFMA backend.
//!
//! Mirrors [`crate::reference::ntt120::convolution`] but uses the 3-prime
//! IFMA NTT tables and pointwise arithmetic.

use bytemuck::cast_slice_mut;
use std::mem::size_of;

use crate::{
    layouts::{
        Backend, CnvPVecL, CnvPVecLToMut, CnvPVecLToRef, CnvPVecR, CnvPVecRToMut, CnvPVecRToRef, VecZnx, VecZnxBig,
        VecZnxBigToMut, VecZnxDft, VecZnxDftToMut, VecZnxToRef, ZnxInfos, ZnxView, ZnxViewMut,
    },
    reference::ntt120::types::Q120bScalar,
};

use super::{
    NttIfmaCFromB, NttIfmaDFTExecute, NttIfmaFromZnx64,
    ntt::NttIfmaTable,
    primes::{PrimeSetIfma, Primes40},
    vec_znx_dft::NttIfmaModuleHandle,
};

const Q: [u64; 3] = <Primes40 as PrimeSetIfma>::Q;

#[inline(always)]
fn mul_accumulate_ifma(acc: &mut [u128; 3], a: &[Q120bScalar], a_col: usize, b: &[Q120bScalar], b_col: usize, n_i: usize) {
    let a_lane = a[a_col + n_i].0;
    let b_lane = b[b_col + n_i].0;
    for k in 0..3 {
        acc[k] += (a_lane[k] % Q[k]) as u128 * (b_lane[k] % Q[k]) as u128;
    }
}

pub fn ntt_ifma_cnv_prepare_left_tmp_bytes(_n: usize) -> usize {
    0
}

pub fn ntt_ifma_cnv_prepare_left<R, A, BE>(module: &impl NttIfmaModuleHandle, res: &mut R, a: &A, _mask: i64, _tmp: &mut [u8])
where
    BE: Backend<ScalarPrep = Q120bScalar> + NttIfmaFromZnx64 + NttIfmaDFTExecute<NttIfmaTable<Primes40>>,
    R: CnvPVecLToMut<BE>,
    A: VecZnxToRef,
{
    let mut res: CnvPVecL<&mut [u8], BE> = res.to_mut();
    let a: VecZnx<&[u8]> = a.to_ref();
    let table = module.get_ntt_ifma_table();
    let cols = res.cols();
    let res_size = res.size();
    let min_size = res_size.min(a.size());

    for col in 0..cols {
        for j in 0..min_size {
            let res_u64: &mut [u64] = cast_slice_mut(res.at_mut(col, j));
            BE::ntt_ifma_from_znx64(res_u64, a.at(col, j));
            BE::ntt_ifma_dft_execute(table, res_u64);
        }
        for j in min_size..res_size {
            cast_slice_mut::<_, u64>(res.at_mut(col, j)).fill(0);
        }
    }
}

pub fn ntt_ifma_cnv_prepare_right_tmp_bytes(n: usize) -> usize {
    4 * n * size_of::<u64>()
}

pub fn ntt_ifma_cnv_prepare_right<R, A, BE>(module: &impl NttIfmaModuleHandle, res: &mut R, a: &A, _mask: i64, tmp: &mut [u64])
where
    BE: Backend<ScalarPrep = Q120bScalar> + NttIfmaFromZnx64 + NttIfmaDFTExecute<NttIfmaTable<Primes40>> + NttIfmaCFromB,
    R: CnvPVecRToMut<BE>,
    A: VecZnxToRef,
{
    let mut res: CnvPVecR<&mut [u8], BE> = res.to_mut();
    let a: VecZnx<&[u8]> = a.to_ref();
    let n = res.n();
    let table = module.get_ntt_ifma_table();
    let cols = res.cols();
    let res_size = res.size();
    let min_size = res_size.min(a.size());

    for col in 0..cols {
        for j in 0..min_size {
            BE::ntt_ifma_from_znx64(tmp, a.at(col, j));
            BE::ntt_ifma_dft_execute(table, tmp);
            let res_u32: &mut [u32] = cast_slice_mut(res.at_mut(col, j));
            BE::ntt_ifma_c_from_b(n, res_u32, tmp);
        }
        for j in min_size..res_size {
            cast_slice_mut::<_, u64>(res.at_mut(col, j)).fill(0);
        }
    }
}

pub fn ntt_ifma_cnv_apply_dft_tmp_bytes(_res_size: usize, _a_size: usize, _b_size: usize) -> usize {
    0
}

#[allow(clippy::too_many_arguments)]
pub fn ntt_ifma_cnv_apply_dft<R, A, B, BE>(
    _module: &impl NttIfmaModuleHandle,
    res: &mut R,
    res_offset: usize,
    res_col: usize,
    a: &A,
    a_col: usize,
    b: &B,
    b_col: usize,
    _tmp: &mut [u8],
) where
    BE: Backend<ScalarPrep = Q120bScalar>,
    R: VecZnxDftToMut<BE>,
    A: CnvPVecLToRef<BE>,
    B: CnvPVecRToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: CnvPVecL<&[u8], BE> = a.to_ref();
    let b: CnvPVecR<&[u8], BE> = b.to_ref();

    let n = res.n();
    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();

    let bound = a_size + b_size - 1;
    let min_size = res_size.min(bound);
    let offset = res_offset.min(bound);

    for k in 0..min_size {
        let k_abs = k + offset;
        let j_min = k_abs.saturating_sub(a_size - 1);
        let j_max = (k_abs + 1).min(b_size);

        let res_u64: &mut [u64] = cast_slice_mut(res.at_mut(res_col, k));

        for n_i in 0..n {
            let mut acc = [0u128; 3];
            for j in j_min..j_max {
                mul_accumulate_ifma(&mut acc, a.at(a_col, k_abs - j), 0, b.at(b_col, j), 0, n_i);
            }
            let coeff_base = 4 * n_i;
            for p in 0..3 {
                res_u64[coeff_base + p] = (acc[p] % Q[p] as u128) as u64;
            }
            res_u64[coeff_base + 3] = 0;
        }
    }

    for j in min_size..res_size {
        cast_slice_mut::<_, u64>(res.at_mut(res_col, j)).fill(0);
    }
}

pub fn ntt_ifma_cnv_by_const_apply_tmp_bytes(_res_size: usize, _a_size: usize, _b_size: usize) -> usize {
    0
}

#[allow(clippy::too_many_arguments)]
pub fn ntt_ifma_cnv_by_const_apply<R, A, BE>(
    res: &mut R,
    res_offset: usize,
    res_col: usize,
    a: &A,
    a_col: usize,
    b: &[i64],
    _tmp: &mut [u8],
) where
    BE: Backend<ScalarPrep = Q120bScalar, ScalarBig = i128>,
    R: VecZnxBigToMut<BE>,
    A: VecZnxToRef,
{
    let mut res: VecZnxBig<&mut [u8], BE> = res.to_mut();
    let a: VecZnx<&[u8]> = a.to_ref();
    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.len();

    let bound = a_size + b_size - 1;
    let min_size = res_size.min(bound);
    let offset = res_offset.min(bound);

    for k in 0..min_size {
        let k_abs = k + offset;
        let j_min = k_abs.saturating_sub(a_size - 1);
        let j_max = (k_abs + 1).min(b_size);
        let res_limb: &mut [i128] = res.at_mut(res_col, k);
        for (n_i, r) in res_limb.iter_mut().enumerate() {
            let mut acc: i128 = 0;
            for (j, &b_j) in b.iter().enumerate().take(j_max).skip(j_min) {
                acc += a.at(a_col, k_abs - j)[n_i] as i128 * b_j as i128;
            }
            *r = acc;
        }
    }

    for j in min_size..res_size {
        res.at_mut(res_col, j).fill(0i128);
    }
}

pub fn ntt_ifma_cnv_pairwise_apply_dft_tmp_bytes(_res_size: usize, _a_size: usize, _b_size: usize) -> usize {
    0
}

#[allow(clippy::too_many_arguments)]
pub fn ntt_ifma_cnv_pairwise_apply_dft<R, A, B, BE>(
    _module: &impl NttIfmaModuleHandle,
    res: &mut R,
    res_offset: usize,
    res_col: usize,
    a: &A,
    b: &B,
    col_i: usize,
    col_j: usize,
    _tmp: &mut [u8],
) where
    BE: Backend<ScalarPrep = Q120bScalar>,
    R: VecZnxDftToMut<BE>,
    A: CnvPVecLToRef<BE>,
    B: CnvPVecRToRef<BE>,
{
    if col_i == col_j {
        let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
        let a: CnvPVecL<&[u8], BE> = a.to_ref();
        let b: CnvPVecR<&[u8], BE> = b.to_ref();
        let n = res.n();
        let res_size = res.size();
        let a_size = a.size();
        let b_size = b.size();
        let bound = a_size + b_size - 1;
        let min_size = res_size.min(bound);
        let offset = res_offset.min(bound);

        for k in 0..min_size {
            let k_abs = k + offset;
            let j_min = k_abs.saturating_sub(a_size - 1);
            let j_max = (k_abs + 1).min(b_size);
            let res_u64: &mut [u64] = cast_slice_mut(res.at_mut(res_col, k));

            for n_i in 0..n {
                let mut acc = [0u128; 3];
                for j in j_min..j_max {
                    mul_accumulate_ifma(&mut acc, a.at(col_i, k_abs - j), 0, b.at(col_j, j), 0, n_i);
                }
                let coeff_base = 4 * n_i;
                for p in 0..3 {
                    res_u64[coeff_base + p] = (acc[p] % Q[p] as u128) as u64;
                }
                res_u64[coeff_base + 3] = 0;
            }
        }

        for j in min_size..res_size {
            cast_slice_mut::<_, u64>(res.at_mut(res_col, j)).fill(0);
        }
        return;
    }

    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: CnvPVecL<&[u8], BE> = a.to_ref();
    let b: CnvPVecR<&[u8], BE> = b.to_ref();

    let n = res.n();
    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();

    let bound = a_size + b_size - 1;
    let min_size = res_size.min(bound);
    let offset = res_offset.min(bound);

    for k in 0..min_size {
        let k_abs = k + offset;
        let j_min = k_abs.saturating_sub(a_size - 1);
        let j_max = (k_abs + 1).min(b_size);

        let res_u64: &mut [u64] = cast_slice_mut(res.at_mut(res_col, k));

        for n_i in 0..n {
            let mut acc = [0u128; 3];
            for j in j_min..j_max {
                let ai = a.at(col_i, k_abs - j)[n_i];
                let aj = a.at(col_j, k_abs - j)[n_i];
                let bi = b.at(col_i, j)[n_i];
                let bj = b.at(col_j, j)[n_i];
                for p in 0..3 {
                    let a_sum = (ai.0[p] % Q[p] + aj.0[p] % Q[p]) % Q[p];
                    let b_sum = (bi.0[p] % Q[p] + bj.0[p] % Q[p]) % Q[p];
                    acc[p] += a_sum as u128 * b_sum as u128;
                }
            }
            let coeff_base = 4 * n_i;
            for p in 0..3 {
                res_u64[coeff_base + p] = (acc[p] % Q[p] as u128) as u64;
            }
            res_u64[coeff_base + 3] = 0;
        }
    }

    for j in min_size..res_size {
        cast_slice_mut::<_, u64>(res.at_mut(res_col, j)).fill(0);
    }
}

pub fn ntt_ifma_cnv_prepare_self_tmp_bytes(n: usize) -> usize {
    ntt_ifma_cnv_prepare_left_tmp_bytes(n).max(ntt_ifma_cnv_prepare_right_tmp_bytes(n))
}

#[allow(clippy::too_many_arguments)]
pub fn ntt_ifma_cnv_prepare_self<L, R, A, BE>(
    module: &impl NttIfmaModuleHandle,
    left: &mut L,
    right: &mut R,
    a: &A,
    mask: i64,
    tmp: &mut [u8],
) where
    BE: Backend<ScalarPrep = Q120bScalar> + NttIfmaFromZnx64 + NttIfmaDFTExecute<NttIfmaTable<Primes40>> + NttIfmaCFromB,
    L: CnvPVecLToMut<BE>,
    R: CnvPVecRToMut<BE>,
    A: VecZnxToRef + ZnxInfos,
{
    // Prepare left side.
    ntt_ifma_cnv_prepare_left::<L, A, BE>(module, left, a, mask, tmp);
    // Prepare right side reusing the same source.
    let right_bytes = ntt_ifma_cnv_prepare_right_tmp_bytes(a.n());
    let (prefix, _) = tmp.split_at_mut(right_bytes);
    let tmp_u64: &mut [u64] = bytemuck::cast_slice_mut(prefix);
    ntt_ifma_cnv_prepare_right::<R, A, BE>(module, right, a, mask, tmp_u64);
}
