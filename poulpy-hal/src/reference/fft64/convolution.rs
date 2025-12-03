use crate::{
    layouts::{
        Backend, CnvPVecL, CnvPVecLToMut, CnvPVecLToRef, CnvPVecR, CnvPVecRToMut, CnvPVecRToRef, VecZnx, VecZnxDft,
        VecZnxDftToMut, VecZnxToRef, ZnxInfos, ZnxView, ZnxViewMut, ZnxZero,
    },
    reference::fft64::{
        reim::{ReimAdd, ReimCopy, ReimDFTExecute, ReimFFTTable, ReimFromZnx, ReimZero},
        reim4::{
            Reim4Convolution, Reim4Convolution1Coeff, Reim4Convolution2Coeffs, Reim4Extract1BlkContiguous,
            Reim4Save1BlkContiguous,
        },
        vec_znx_dft::vec_znx_dft_apply,
    },
};

pub fn convolution_prepare_left<R, A, T, BE>(table: &ReimFFTTable<f64>, res: &mut R, a: &A, tmp: &mut T)
where
    BE: Backend<ScalarPrep = f64>
        + ReimZero
        + Reim4Extract1BlkContiguous
        + ReimDFTExecute<ReimFFTTable<f64>, f64>
        + ReimFromZnx
        + ReimZero,
    R: CnvPVecLToMut<BE> + ZnxInfos + ZnxViewMut<Scalar = BE::ScalarPrep>,
    A: VecZnxToRef,
    T: VecZnxDftToMut<BE>,
{
    convolution_prepare(table, res, a, tmp)
}

pub fn convolution_prepare_right<R, A, T, BE>(table: &ReimFFTTable<f64>, res: &mut R, a: &A, tmp: &mut T)
where
    BE: Backend<ScalarPrep = f64>
        + ReimZero
        + Reim4Extract1BlkContiguous
        + ReimDFTExecute<ReimFFTTable<f64>, f64>
        + ReimFromZnx
        + ReimZero,
    R: CnvPVecRToMut<BE> + ZnxInfos + ZnxViewMut<Scalar = BE::ScalarPrep>,
    A: VecZnxToRef,
    T: VecZnxDftToMut<BE>,
{
    convolution_prepare(table, res, a, tmp)
}

fn convolution_prepare<R, A, T, BE>(table: &ReimFFTTable<f64>, res: &mut R, a: &A, tmp: &mut T)
where
    BE: Backend<ScalarPrep = f64>
        + ReimZero
        + Reim4Extract1BlkContiguous
        + ReimDFTExecute<ReimFFTTable<f64>, f64>
        + ReimFromZnx
        + ReimZero,
    R: ZnxInfos + ZnxViewMut<Scalar = BE::ScalarPrep>,
    A: VecZnxToRef,
    T: VecZnxDftToMut<BE>,
{
    let a: &VecZnx<&[u8]> = &a.to_ref();
    let tmp: &mut VecZnxDft<&mut [u8], BE> = &mut tmp.to_mut();

    let cols: usize = res.cols();
    assert_eq!(a.cols(), cols, "a.cols():{} != res.cols():{cols}", a.cols());

    let res_size: usize = res.size();
    let min_size: usize = res_size.min(a.size());

    let m: usize = a.n() >> 1;

    let n: usize = table.m() << 1;

    let res_raw: &mut [f64] = res.raw_mut();

    for i in 0..cols {
        vec_znx_dft_apply(table, 1, 0, tmp, 0, a, i);

        let tmp_raw: &[f64] = tmp.raw();
        let res_col: &mut [f64] = &mut res_raw[i * n * res_size..];

        for blk_i in 0..m / 4 {
            BE::reim4_extract_1blk_contiguous(m, min_size, blk_i, &mut res_col[blk_i * res_size * 8..], tmp_raw);
            BE::reim_zero(&mut res_col[blk_i * res_size * 8 + min_size * 8..(blk_i + 1) * res_size * 8]);
        }
    }
}

pub fn convolution_apply_dft_tmp_bytes(res_size: usize, a_size: usize, b_size: usize) -> usize {
    let min_size: usize = res_size.min(a_size + b_size - 1);
    size_of::<f64>() * 8 * min_size
}

#[allow(clippy::too_many_arguments)]
pub fn convolution_apply_dft<R, A, B, BE>(
    res: &mut R,
    res_offset: usize,
    res_col: usize,
    a: &A,
    a_col: usize,
    b: &B,
    b_col: usize,
    tmp: &mut [f64],
) where
    BE: Backend<ScalarPrep = f64> + Reim4Save1BlkContiguous + Reim4Convolution1Coeff + Reim4Convolution2Coeffs,
    R: VecZnxDftToMut<BE>,
    A: CnvPVecLToRef<BE>,
    B: CnvPVecRToRef<BE>,
{
    let res: &mut VecZnxDft<&mut [u8], BE> = &mut res.to_mut();
    let a: &CnvPVecL<&[u8], BE> = &a.to_ref();
    let b: &CnvPVecR<&[u8], BE> = &b.to_ref();

    let n: usize = res.n();
    assert_eq!(a.n(), n);
    assert_eq!(b.n(), n);
    let m: usize = n >> 1;

    let res_size: usize = res.size();
    let a_size: usize = a.size();
    let b_size: usize = b.size();

    let bound: usize = a_size + b_size - 1;
    let min_size: usize = res_size.min(bound);
    let offset: usize = res_offset.min(bound);

    let dst: &mut [f64] = res.raw_mut();
    let a_raw: &[f64] = a.raw();
    let b_raw: &[f64] = b.raw();

    let mut a_idx: usize = a_col * n * a_size;
    let mut b_idx: usize = b_col * n * b_size;
    let a_offset: usize = a_size * 8;
    let b_offset: usize = b_size * 8;
    for blk_i in 0..m / 4 {
        BE::reim4_convolution(tmp, min_size, offset, &a_raw[a_idx..], a_size, &b_raw[b_idx..], b_size);
        BE::reim4_save_1blk_contiguous(m, min_size, blk_i, dst, tmp);
        a_idx += a_offset;
        b_idx += b_offset;
    }

    for j in min_size..res_size {
        res.zero_at(res_col, j);
    }
}

pub fn convolution_pairwise_apply_dft_tmp_bytes(res_size: usize, a_size: usize, b_size: usize) -> usize {
    convolution_apply_dft_tmp_bytes(res_size, a_size, b_size) + (a_size + b_size) * size_of::<f64>() * 8
}

#[allow(clippy::too_many_arguments)]
pub fn convolution_pairwise_apply_dft<R, A, B, BE>(
    res: &mut R,
    res_offset: usize,
    res_col: usize,
    a: &A,
    b: &B,
    col_i: usize,
    col_j: usize,
    tmp: &mut [f64],
) where
    BE: Backend<ScalarPrep = f64>
        + ReimZero
        + ReimAdd
        + ReimCopy
        + Reim4Save1BlkContiguous
        + Reim4Convolution1Coeff
        + Reim4Convolution2Coeffs,
    R: VecZnxDftToMut<BE>,
    A: CnvPVecLToRef<BE>,
    B: CnvPVecRToRef<BE>,
{
    if col_i == col_j {
        convolution_apply_dft(res, res_offset, res_col, a, col_i, b, col_j, tmp);
        return;
    }

    let res: &mut VecZnxDft<&mut [u8], BE> = &mut res.to_mut();
    let a: &CnvPVecL<&[u8], BE> = &a.to_ref();
    let b: &CnvPVecR<&[u8], BE> = &b.to_ref();

    let n: usize = res.n();
    let m: usize = n >> 1;

    assert_eq!(a.n(), n);
    assert_eq!(b.n(), n);

    let res_size: usize = res.size();
    let a_size: usize = a.size();
    let b_size: usize = b.size();

    assert_eq!(
        tmp.len(),
        convolution_pairwise_apply_dft_tmp_bytes(res_size, a_size, b_size) / size_of::<f64>()
    );

    let bound: usize = a_size + b_size - 1;
    let min_size: usize = res_size.min(bound);
    let offset: usize = res_offset.min(bound);

    let res_raw: &mut [f64] = res.raw_mut();
    let a_raw: &[f64] = a.raw();
    let b_raw: &[f64] = b.raw();

    let a_row_size: usize = a_size * 8;
    let b_row_size: usize = b_size * 8;

    let mut a0_idx: usize = col_i * n * a_size;
    let mut a1_idx: usize = col_j * n * a_size;
    let mut b0_idx: usize = col_i * n * b_size;
    let mut b1_idx: usize = col_j * n * b_size;

    let (tmp_a, tmp) = tmp.split_at_mut(a_row_size);
    let (tmp_b, tmp_res) = tmp.split_at_mut(b_row_size);

    for blk_i in 0..m / 4 {
        let a0: &[f64] = &a_raw[a0_idx..];
        let a1: &[f64] = &a_raw[a1_idx..];
        let b0: &[f64] = &b_raw[b0_idx..];
        let b1: &[f64] = &b_raw[b1_idx..];

        BE::reim_add(tmp_a, &a0[..a_row_size], &a1[..a_row_size]);
        BE::reim_add(tmp_b, &b0[..b_row_size], &b1[..b_row_size]);

        BE::reim4_convolution(tmp_res, min_size, offset, tmp_a, a_size, tmp_b, b_size);
        BE::reim4_save_1blk_contiguous(m, min_size, blk_i, res_raw, tmp_res);

        a0_idx += a_row_size;
        a1_idx += a_row_size;
        b0_idx += b_row_size;
        b1_idx += b_row_size;
    }

    for j in min_size..res_size {
        res.zero_at(res_col, j);
    }
}
