use std::hint::black_box;

use crate::{
    api::{
        ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxDftAlloc, VmpApplyDft, VmpApplyDftTmpBytes, VmpApplyDftToDft,
        VmpApplyDftToDftTmpBytes, VmpPMatAlloc,
    },
    cast_mut,
    layouts::{DataViewMut, Module, ScratchOwned, VecZnx, VecZnxToRef},
    oep::VecZnxDftAllocBytesImpl,
    reference::fft64::{
        reim::{ReimDFTExecute, ReimFFTTable, ReimFromZnx, ReimZero},
        reim4::{Reim4Extract1Blk, Reim4Mat1ColProd, Reim4Mat2Cols2ndColProd, Reim4Mat2ColsProd, Reim4Save1Blk, Reim4Save2Blks},
        vec_znx_dft::vec_znx_dft_apply,
    },
    source::Source,
};
use criterion::{BenchmarkId, Criterion};
use rand::RngCore;

use crate::layouts::{Backend, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef, VmpPMat, VmpPMatToRef, ZnxInfos};

pub fn vmp_apply_dft_tmp_bytes(n: usize, a_size: usize, prows: usize, pcols_in: usize) -> usize {
    let row_max: usize = (a_size).min(prows);
    (16 + (n + 8) * row_max * pcols_in) * size_of::<f64>()
}

pub fn vmp_apply_dft<R, A, M, BE>(table: &ReimFFTTable<f64>, res: &mut R, a: &A, pmat: &M, tmp_bytes: &mut [f64])
where
    BE: Backend<ScalarPrep = f64>
        + VecZnxDftAllocBytesImpl<BE>
        + ReimDFTExecute<ReimFFTTable<f64>, f64>
        + ReimZero
        + Reim4Extract1Blk
        + Reim4Mat1ColProd
        + Reim4Mat2Cols2ndColProd
        + Reim4Mat2ColsProd
        + Reim4Save2Blks
        + Reim4Save1Blk
        + ReimFromZnx,
    R: VecZnxDftToMut<BE>,
    A: VecZnxToRef,
    M: VmpPMatToRef<BE>,
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
        vec_znx_dft_apply(table, 1, 0, &mut a_dft, j, &a, offset + j);
    }

    vmp_apply_dft_to_dft(res, &a_dft, &pmat, tmp_bytes);
}

pub fn vmp_apply_dft_to_dft_tmp_bytes(a_size: usize, prows: usize, pcols_in: usize) -> usize {
    let row_max: usize = (a_size).min(prows);
    (16 + 8 * row_max * pcols_in) * size_of::<f64>()
}

pub fn vmp_apply_dft_to_dft<R, A, M, BE>(res: &mut R, a: &A, pmat: &M, tmp_bytes: &mut [f64])
where
    BE: Backend<ScalarPrep = f64>
        + ReimZero
        + Reim4Extract1Blk
        + Reim4Mat1ColProd
        + Reim4Mat2Cols2ndColProd
        + Reim4Mat2ColsProd
        + Reim4Save2Blks
        + Reim4Save1Blk,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
    M: VmpPMatToRef<BE>,
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

    vmp_apply_dft_to_dft_core::<true, BE>(n, res_raw, a_raw, pmat_raw, 0, nrows, ncols, tmp_bytes)
}

pub fn vmp_apply_dft_to_dft_add<R, A, M, BE>(res: &mut R, a: &A, pmat: &M, limb_offset: usize, tmp_bytes: &mut [f64])
where
    BE: Backend<ScalarPrep = f64>
        + ReimZero
        + Reim4Extract1Blk
        + Reim4Mat1ColProd
        + Reim4Mat2Cols2ndColProd
        + Reim4Mat2ColsProd
        + Reim4Save2Blks
        + Reim4Save1Blk,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
    M: VmpPMatToRef<BE>,
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

    vmp_apply_dft_to_dft_core::<false, BE>(
        n,
        res_raw,
        a_raw,
        pmat_raw,
        limb_offset,
        nrows,
        ncols,
        tmp_bytes,
    )
}

#[allow(clippy::too_many_arguments)]
fn vmp_apply_dft_to_dft_core<const OVERWRITE: bool, REIM>(
    n: usize,
    res: &mut [f64],
    a: &[f64],
    pmat: &[f64],
    limb_offset: usize,
    nrows: usize,
    ncols: usize,
    tmp_bytes: &mut [f64],
) where
    REIM: ReimZero
        + Reim4Extract1Blk
        + Reim4Mat1ColProd
        + Reim4Mat2Cols2ndColProd
        + Reim4Mat2ColsProd
        + Reim4Save2Blks
        + Reim4Save1Blk,
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

    if limb_offset >= col_max {
        if OVERWRITE {
            REIM::reim_zero(res);
        }
        return;
    }

    for blk_i in 0..(m >> 2) {
        let mat_blk_start: &[f64] = &pmat[blk_i * (8 * nrows * ncols)..];

        REIM::reim4_extract_1blk(m, row_max, blk_i, extracted_blk, a);

        if limb_offset.is_multiple_of(2) {
            for (col_res, col_pmat) in (0..).step_by(2).zip((limb_offset..col_max - 1).step_by(2)) {
                let col_offset: usize = col_pmat * (8 * nrows);
                REIM::reim4_mat2cols_prod(
                    row_max,
                    mat2cols_output,
                    extracted_blk,
                    &mat_blk_start[col_offset..],
                );
                REIM::reim4_save_2blks::<OVERWRITE>(m, blk_i, &mut res[col_res * n..], mat2cols_output);
            }
        } else {
            let col_offset: usize = (limb_offset - 1) * (8 * nrows);
            REIM::reim4_mat2cols_2ndcol_prod(
                row_max,
                mat2cols_output,
                extracted_blk,
                &mat_blk_start[col_offset..],
            );

            REIM::reim4_save_1blk::<OVERWRITE>(m, blk_i, res, mat2cols_output);

            for (col_res, col_pmat) in (1..)
                .step_by(2)
                .zip((limb_offset + 1..col_max - 1).step_by(2))
            {
                let col_offset: usize = col_pmat * (8 * nrows);
                REIM::reim4_mat2cols_prod(
                    row_max,
                    mat2cols_output,
                    extracted_blk,
                    &mat_blk_start[col_offset..],
                );
                REIM::reim4_save_2blks::<OVERWRITE>(m, blk_i, &mut res[col_res * n..], mat2cols_output);
            }
        }

        if !col_max.is_multiple_of(2) {
            let last_col: usize = col_max - 1;
            let col_offset: usize = last_col * (8 * nrows);

            if last_col >= limb_offset {
                if ncols == col_max {
                    REIM::reim4_mat1col_prod(
                        row_max,
                        mat2cols_output,
                        extracted_blk,
                        &mat_blk_start[col_offset..],
                    );
                } else {
                    REIM::reim4_mat2cols_prod(
                        row_max,
                        mat2cols_output,
                        extracted_blk,
                        &mat_blk_start[col_offset..],
                    );
                }
                REIM::reim4_save_1blk::<OVERWRITE>(
                    m,
                    blk_i,
                    &mut res[(last_col - limb_offset) * n..],
                    mat2cols_output,
                );
            }
        }
    }

    REIM::reim_zero(&mut res[col_max * n..]);
}

pub fn bench_vmp_apply_dft<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: ModuleNew<B> + VmpApplyDftTmpBytes + VmpApplyDft<B> + VmpPMatAlloc<B> + VecZnxDftAlloc<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vmp_apply_dft::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 5]) -> impl FnMut()
    where
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

pub fn bench_vmp_apply_dft_to_dft<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: ModuleNew<B> + VecZnxDftAlloc<B> + VmpPMatAlloc<B> + VmpApplyDftToDft<B> + VmpApplyDftToDftTmpBytes,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vmp_apply_dft_to_dft::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 5]) -> impl FnMut()
    where
        Module<B>: ModuleNew<B> + VecZnxDftAlloc<B> + VmpPMatAlloc<B> + VmpApplyDftToDft<B> + VmpApplyDftToDftTmpBytes,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << params[0]);

        let rows: usize = params[1];
        let cols_in: usize = params[2];
        let cols_out: usize = params[3];
        let size: usize = params[4];

        let mut source: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<B> =
            ScratchOwned::alloc(module.vmp_apply_dft_to_dft_tmp_bytes(size, size, rows, cols_in, cols_out, size));

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
