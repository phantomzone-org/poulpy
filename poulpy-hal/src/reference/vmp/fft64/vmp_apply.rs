use std::hint::black_box;

use crate::{
    api::{
        ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxDftAlloc, VmpApplyDft, VmpApplyDftTmpBytes, VmpApplyDftToDft,
        VmpApplyDftToDftTmpBytes, VmpPMatAlloc,
    },
    cast_mut,
    layouts::{DataViewMut, Module, ScratchOwned, VecZnx, VecZnxToRef, ZnxViewMut},
    oep::VecZnxDftAllocBytesImpl,
    reference::{
        reim::{ReimArithmetic, ReimArithmeticRef, ReimConv, ReimConvRef, ReimDFTExecute, ReimFFTRef, ReimFFTTable},
        reim4::{Reim4Blk, Reim4BlkRef},
        vec_znx_dft::fft64::vec_znx_dft_apply,
    },
    source::Source,
};
use criterion::{BenchmarkId, Criterion};
use rand::RngCore;

use crate::layouts::{Backend, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef, VmpPMat, VmpPMatToRef, ZnxInfos};

pub fn vmp_apply_dft_tmp_bytes(n: usize, a_size: usize, pmat_rows: usize, pmat_cols_in: usize) -> usize {
    let row_max: usize = (a_size).min(pmat_rows);
    (16 + (n + 8) * row_max * pmat_cols_in) * size_of::<f64>()
}

pub fn vmp_apply_dft<R, A, M, BE, REIM, CONV, REIM4, FFT>(
    table: &ReimFFTTable<f64>,
    res: &mut R,
    a: &A,
    pmat: &M,
    tmp_bytes: &mut [f64],
) where
    BE: Backend<ScalarPrep = f64> + VecZnxDftAllocBytesImpl<BE>,
    R: VecZnxDftToMut<BE>,
    A: VecZnxToRef,
    M: VmpPMatToRef<BE>,
    REIM: ReimArithmetic,
    CONV: ReimConv,
    REIM4: Reim4Blk,
    FFT: ReimDFTExecute<ReimFFTTable<f64>, f64>,
{
    let a: VecZnx<&[u8]> = a.to_ref();
    let pmat: VmpPMat<&[u8], BE> = pmat.to_ref();

    let n: usize = a.n();
    let cols: usize = pmat.cols_in();
    let size: usize = a.size().min(pmat.rows());

    #[cfg(debug_assertions)]
    {
        assert!(tmp_bytes.len() >= vmp_apply_dft_tmp_bytes(n, size, pmat.rows(), cols));
        assert!(a.cols() <= cols);
    }

    let (data, tmp_bytes) = tmp_bytes.split_at_mut(BE::vec_znx_dft_alloc_bytes_impl(n, cols, size));

    let mut a_dft: VecZnxDft<&mut [u8], BE> = VecZnxDft::from_data(cast_mut(data), n, cols, size);

    let offset: usize = cols - a.cols();
    for j in 0..cols {
        vec_znx_dft_apply::<_, _, _, REIM, CONV, FFT>(table, 1, 0, &mut a_dft, j, &a, offset + j);
    }

    vmp_apply_dft_to_dft::<_, _, _, _, REIM, REIM4>(res, &a_dft, &pmat, tmp_bytes);
}

pub fn vmp_apply_dft_to_dft_tmp_bytes(a_size: usize, pmat_rows: usize, pmat_cols_in: usize) -> usize {
    let row_max: usize = (a_size).min(pmat_rows);
    (16 + 8 * row_max * pmat_cols_in) * size_of::<f64>()
}

pub fn vmp_apply_dft_to_dft<R, A, M, BE, REIM, REIM4>(res: &mut R, a: &A, pmat: &M, tmp_bytes: &mut [f64])
where
    BE: Backend<ScalarPrep = f64>,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
    M: VmpPMatToRef<BE>,
    REIM: ReimArithmetic,
    REIM4: Reim4Blk,
{
    use crate::layouts::{ZnxView, ZnxViewMut};

    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: VecZnxDft<&[u8], BE> = a.to_ref();
    let pmat: VmpPMat<&[u8], BE> = pmat.to_ref();

    #[cfg(debug_assertions)]
    {
        assert_eq!(res.n(), pmat.n());
        assert_eq!(a.n(), pmat.n());
        assert_eq!(res.cols(), pmat.cols_out());
        assert_eq!(a.cols(), pmat.cols_in());
    }

    let n: usize = res.n();
    let nrows: usize = pmat.cols_in() * pmat.rows();
    let ncols: usize = pmat.cols_out() * pmat.size();

    let pmat_raw: &[f64] = pmat.raw();
    let a_raw: &[f64] = a.raw();
    let res_raw: &mut [f64] = res.raw_mut();

    vmp_apply_dft_to_dft_core::<REIM, REIM4>(n, res_raw, a_raw, pmat_raw, nrows, ncols, tmp_bytes)
}

fn vmp_apply_dft_to_dft_core<REIM, REIM4>(
    n: usize,
    res: &mut [f64],
    a: &[f64],
    pmat: &[f64],
    nrows: usize,
    ncols: usize,
    tmp_bytes: &mut [f64],
) where
    REIM: ReimArithmetic,
    REIM4: Reim4Blk,
{
    #[cfg(debug_assertions)]
    {
        assert!(n >= 8);
        assert!(n.is_power_of_two());
        assert_eq!(pmat.len(), n * nrows * ncols);
        assert!(res.len() & (n - 1) == 0);
        assert!(a.len() & (n - 1) == 0);
    }

    let a_size: usize = a.len() / n;
    let res_size: usize = res.len() / n;

    let m: usize = n >> 1;

    let (mat2cols_output, extracted_blk) = tmp_bytes.split_at_mut(16);

    let row_max: usize = nrows.min(a_size);
    let col_max: usize = ncols.min(res_size);

    for blk_i in 0..(m >> 2) {
        let mat_blk_start: &[f64] = &pmat[blk_i * (8 * nrows * ncols)..];

        REIM4::reim4_extract_1blk_from_reim(m, row_max, blk_i, extracted_blk, a);

        for col_i in (0..col_max - 1).step_by(2) {
            let col_offset: usize = col_i * (8 * nrows);

            REIM4::reim4_vec_mat2cols_product(
                row_max,
                mat2cols_output,
                extracted_blk,
                &mat_blk_start[col_offset..],
            );
            REIM4::reim4_save_2blk_to_reim(m, blk_i, &mut res[col_i * n..], mat2cols_output)
        }

        if !col_max.is_multiple_of(2) {
            let last_col: usize = col_max - 1;
            let col_offset: usize = last_col * (8 * nrows);
            if ncols == col_max {
                REIM4::reim4_vec_mat1col_product(
                    row_max,
                    mat2cols_output,
                    extracted_blk,
                    &mat_blk_start[col_offset..],
                );
            } else {
                REIM4::reim4_vec_mat2cols_product(
                    row_max,
                    mat2cols_output,
                    extracted_blk,
                    &mat_blk_start[col_offset..],
                );
            }
            REIM4::reim4_save_1blk_to_reim(m, blk_i, &mut res[last_col * n..], mat2cols_output);
        }
    }

    REIM::reim_zero(&mut res[col_max * n..]);
}

pub fn test_vmp_apply_dft<B>()
where
    B: Backend<ScalarPrep = f64> + VecZnxDftAllocBytesImpl<B>,
    Module<B>: ModuleNew<B> + VmpApplyDftTmpBytes + VmpApplyDft<B> + VmpPMatAlloc<B> + VecZnxDftAlloc<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let log_n: i32 = 5;
    let n: usize = 1 << log_n;

    let module: Module<B> = Module::<B>::new(n as u64);
    let a_size: usize = 5;
    let mat_size: usize = 6;
    let res_size: usize = a_size;

    let mut source: Source = Source::new([0u8; 32]);
    let table: ReimFFTTable<f64> = ReimFFTTable::new(n >> 1);

    [1, 2].iter().for_each(|cols_in| {
        [1, 2].iter().for_each(|cols_out| {
            let a_cols: usize = *cols_in;
            let res_cols: usize = *cols_out;

            let mat_rows: usize = a_size;
            let mat_cols_in: usize = a_cols;
            let mat_cols_out: usize = res_cols;

            let mut tmp_bytes: Vec<f64> =
                vec![0f64; vmp_apply_dft_tmp_bytes(n, a_size, mat_rows, mat_cols_in) / size_of::<f64>()];
            println!(
                "{}",
                module.vmp_apply_dft_tmp_bytes(
                    res_size,
                    a_size,
                    mat_rows,
                    mat_cols_in,
                    mat_cols_out,
                    mat_size,
                )
            );

            let mut scratch = ScratchOwned::alloc(module.vmp_apply_dft_tmp_bytes(
                res_size,
                a_size,
                mat_rows,
                mat_cols_in,
                mat_cols_out,
                mat_size,
            ));

            let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, a_cols, a_size);

            (0..a_cols).for_each(|i| {
                a.at_mut(i, a_size - 1)[i + 1] = 1;
            });

            let mut pmat: VmpPMat<Vec<u8>, B> = module.vmp_pmat_alloc(mat_rows, mat_cols_in, mat_cols_out, mat_size);

            pmat.raw_mut()
                .iter_mut()
                .for_each(|x| *x = source.next_f64(-1., 1.));

            let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, a_cols, a_size);
            a.raw_mut()
                .iter_mut()
                .for_each(|x| *x = source.next_i32() as i64);

            let mut res_0: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(mat_cols_out, mat_size);
            let mut res_1: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(mat_cols_out, mat_size);

            source.fill_bytes(res_0.data_mut());
            source.fill_bytes(res_1.data_mut());

            module.vmp_apply_dft(&mut res_1, &a, &pmat, scratch.borrow());
            vmp_apply_dft::<_, _, _, _, ReimArithmeticRef, ReimConvRef, Reim4BlkRef, ReimFFTRef>(
                &table,
                &mut res_0,
                &a,
                &pmat,
                &mut tmp_bytes,
            );
        });
    });
}

pub fn bench_vmp_apply_dft<B>(c: &mut Criterion, label: &str)
where
    B: Backend<ScalarPrep = f64> + VecZnxDftAllocBytesImpl<B>,
    Module<B>: ModuleNew<B> + VmpApplyDftTmpBytes + VmpApplyDft<B> + VmpPMatAlloc<B> + VecZnxDftAlloc<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vmp_apply_dft::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(params: [usize; 5]) -> impl FnMut()
    where
        B: Backend<ScalarPrep = f64> + VecZnxDftAllocBytesImpl<B>,
        Module<B>: ModuleNew<B> + VmpApplyDftTmpBytes + VmpApplyDft<B> + VmpPMatAlloc<B> + VecZnxDftAlloc<B>,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << params[0]);

        let rows: usize = params[1];
        let cols_in: usize = params[2];
        let cols_out: usize = params[3];
        let size: usize = params[4];

        let mut source: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(1 << 20);

        let mut res: VecZnxDft<Vec<u8>, _> = module.vec_znx_dft_alloc(cols_out, size);
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols_in, size);
        let mut pmat: VmpPMat<Vec<u8>, B> = module.vmp_pmat_alloc(rows, cols_in, cols_out, size);

        source.fill_bytes(pmat.data_mut());
        source.fill_bytes(res.data_mut());
        source.fill_bytes(a.data_mut());

        move || {
            module.vmp_apply_dft(&mut res, &a, &pmat, scratch.borrow());
            black_box(());
        }
    }

    for params in [
        [10, 2, 1, 2, 3],
        [11, 4, 1, 2, 5],
        [12, 7, 1, 2, 8],
        [13, 15, 1, 2, 16],
        [14, 31, 1, 2, 32],
    ] {
        let id = BenchmarkId::from_parameter(format!(
            "{}x({}x{})x({}x{})",
            1 << params[0],
            params[1],
            params[2],
            params[3],
            params[4]
        ));
        let mut runner = runner::<B>(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn test_vmp_apply_dft_to_dft<B>()
where
    B: Backend<ScalarPrep = f64>,
    Module<B>: ModuleNew<B> + VmpApplyDftToDftTmpBytes + VmpApplyDftToDft<B> + VmpPMatAlloc<B> + VecZnxDftAlloc<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let log_n: i32 = 5;
    let n: usize = 1 << log_n;

    let module: Module<B> = Module::<B>::new(n as u64);
    let a_size: usize = 5;
    let mat_size: usize = 6;
    let res_size: usize = a_size;

    let mut source: Source = Source::new([0u8; 32]);

    [1, 2].iter().for_each(|cols_in| {
        [1, 2].iter().for_each(|cols_out| {
            let a_cols: usize = *cols_in;
            let res_cols: usize = *cols_out;

            let mat_rows: usize = a_size;
            let mat_cols_in: usize = a_cols;
            let mat_cols_out: usize = res_cols;

            let mut tmp_bytes: Vec<f64> =
                vec![0f64; vmp_apply_dft_to_dft_tmp_bytes(a_size, mat_rows, mat_cols_in) / size_of::<f64>()];

            let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.vmp_apply_dft_to_dft_tmp_bytes(
                res_size,
                a_size,
                mat_rows,
                mat_cols_in,
                mat_cols_out,
                mat_size,
            ));

            let mut pmat: VmpPMat<Vec<u8>, B> = module.vmp_pmat_alloc(mat_rows, mat_cols_in, mat_cols_out, mat_size);

            pmat.raw_mut()
                .iter_mut()
                .for_each(|x| *x = source.next_f64(-1., 1.));

            let mut a: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(mat_cols_in, mat_size);

            a.raw_mut()
                .iter_mut()
                .for_each(|x| *x = source.next_f64(-1., 1.));

            let mut res_0: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(mat_cols_out, mat_size);
            let mut res_1: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(mat_cols_out, mat_size);

            source.fill_bytes(res_0.data_mut());
            source.fill_bytes(res_1.data_mut());

            module.vmp_apply_dft_to_dft(&mut res_1, &a, &pmat, scratch.borrow());
            vmp_apply_dft_to_dft::<_, _, _, _, ReimArithmeticRef, Reim4BlkRef>(&mut res_0, &a, &pmat, &mut tmp_bytes);
        });
    });
}

pub fn bench_vmp_apply_dft_to_dft<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: ModuleNew<B> + VecZnxDftAlloc<B> + VmpPMatAlloc<B> + VmpApplyDftToDft<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vmp_apply_dft_to_dft::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 5]) -> impl FnMut()
    where
        Module<B>: ModuleNew<B> + VecZnxDftAlloc<B> + VmpPMatAlloc<B> + VmpApplyDftToDft<B>,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << params[0]);

        let rows: usize = params[1];
        let cols_in: usize = params[2];
        let cols_out: usize = params[3];
        let size: usize = params[4];

        let mut source: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(1 << 20);

        let mut res: VecZnxDft<Vec<u8>, _> = module.vec_znx_dft_alloc(cols_out, size);
        let mut a: VecZnxDft<Vec<u8>, _> = module.vec_znx_dft_alloc(cols_in, size);

        let mut pmat: VmpPMat<Vec<u8>, B> = module.vmp_pmat_alloc(rows, cols_in, cols_out, size);

        source.fill_bytes(pmat.data_mut());
        source.fill_bytes(res.data_mut());
        source.fill_bytes(a.data_mut());

        move || {
            module.vmp_apply_dft_to_dft(&mut res, &a, &pmat, scratch.borrow());
            black_box(());
        }
    }

    for params in [
        [10, 2, 1, 2, 3],
        [11, 4, 1, 2, 5],
        [12, 7, 1, 2, 8],
        [13, 15, 1, 2, 16],
        [14, 31, 1, 2, 32],
    ] {
        let id = BenchmarkId::from_parameter(format!(
            "{}x({}x{})x({}x{})",
            1 << params[0],
            params[1],
            params[2],
            params[3],
            params[4]
        ));
        let mut runner = runner::<B>(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}
