use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};
use rand::RngCore;

use crate::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VmpPMatAlloc, VmpPrepare, VmpPrepareTmpBytes},
    layouts::{
        Backend, DataViewMut, MatZnx, MatZnxToRef, Module, ScratchOwned, VmpPMat, VmpPMatToMut, ZnxInfos, ZnxView, ZnxViewMut,
    },
    reference::{
        reim::{ReimConv, ReimConvRef, ReimDFTExecute, ReimFFTRef, ReimFFTTable, as_arr},
        reim4::{Reim4Blk, Reim4BlkRef},
        vec_znx_dft::fft64::assert_approx_eq_slice,
    },
    source::Source,
};

pub fn vmp_prepare_tmp_bytes(n: usize) -> usize {
    n * size_of::<i64>()
}

pub fn vmp_prepare<R, A, BE, BLK, CONV, FFT>(table: &ReimFFTTable<f64>, pmat: &mut R, mat: &A, tmp: &mut [f64])
where
    BE: Backend<ScalarPrep = f64>,
    R: VmpPMatToMut<BE>,
    A: MatZnxToRef,
    BLK: Reim4Blk,
    CONV: ReimConv,
    FFT: ReimDFTExecute<ReimFFTTable<f64>, f64>,
{
    let mut res: crate::layouts::VmpPMat<&mut [u8], BE> = pmat.to_mut();
    let a: MatZnx<&[u8]> = mat.to_ref();

    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), res.n());
        assert_eq!(
            res.cols_in(),
            a.cols_in(),
            "res.cols_in: {} != a.cols_in: {}",
            res.cols_in(),
            a.cols_in()
        );
        assert_eq!(
            res.rows(),
            a.rows(),
            "res.rows: {} != a.rows: {}",
            res.rows(),
            a.rows()
        );
        assert_eq!(
            res.cols_out(),
            a.cols_out(),
            "res.cols_out: {} != a.cols_out: {}",
            res.cols_out(),
            a.cols_out()
        );
        assert_eq!(
            res.size(),
            a.size(),
            "res.size: {} != a.size: {}",
            res.size(),
            a.size()
        );
    }

    let nrows: usize = a.cols_in() * a.rows();
    let ncols: usize = a.cols_out() * a.size();
    vmp_prepare_core::<BLK, CONV, FFT>(table, res.raw_mut(), a.raw(), nrows, ncols, tmp);
}

pub(crate) fn vmp_prepare_core<BLK, CONV, FFT>(
    table: &ReimFFTTable<f64>,
    pmat: &mut [f64],
    mat: &[i64],
    nrows: usize,
    ncols: usize,
    tmp: &mut [f64],
) where
    BLK: Reim4Blk,
    CONV: ReimConv,
    FFT: ReimDFTExecute<ReimFFTTable<f64>, f64>,
{
    let m: usize = table.m();
    let n: usize = m << 1;

    #[cfg(debug_assertions)]
    {
        assert!(n >= 8);
        assert_eq!(mat.len(), n * nrows * ncols);
        assert_eq!(pmat.len(), n * nrows * ncols);
        assert_eq!(tmp.len(), vmp_prepare_tmp_bytes(n) / size_of::<i64>())
    }

    let offset: usize = nrows * ncols * 8;

    for row_i in 0..nrows {
        for col_i in 0..ncols {
            let pos: usize = n * (row_i * ncols + col_i);

            CONV::reim_from_znx_i64(tmp, &mat[pos..pos + n]);

            FFT::reim_dft_execute(table, tmp);

            let dst: &mut [f64] = if col_i == (ncols - 1) && !ncols.is_multiple_of(2) {
                &mut pmat[col_i * nrows * 8 + row_i * 8..]
            } else {
                &mut pmat[(col_i / 2) * (2 * nrows * 8) + row_i * 2 * 8 + (col_i % 2) * 8..]
            };

            for blk_i in 0..m >> 2 {
                BLK::reim4_extract_1blk_from_reim(m, 1, blk_i, &mut dst[blk_i * offset..], tmp);
            }
        }
    }
}

pub fn test_vmp_prepare<B>()
where
    B: Backend<ScalarPrep = f64>,
    Module<B>: ModuleNew<B> + VmpPrepare<B> + VmpPMatAlloc<B> + VmpPrepareTmpBytes,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let log_n: i32 = 5;
    let n: usize = 1 << log_n;

    let module: Module<B> = Module::<B>::new(n as u64);
    let mut source: Source = Source::new([0u8; 32]);

    let table: ReimFFTTable<f64> = ReimFFTTable::<f64>::new(n >> 1);

    [1, 2].iter().for_each(|cols_in| {
        [1, 2].iter().for_each(|cols_out| {
            let mat_rows: usize = 5;
            let mat_cols_in: usize = *cols_in;
            let mat_cols_out: usize = *cols_out;
            let mat_size: usize = 6;

            let mut tmp_bytes: Vec<f64> = vec![0f64; vmp_prepare_tmp_bytes(module.n()) / size_of::<f64>()];

            let mut scratch: ScratchOwned<B> =
                ScratchOwned::alloc(module.vmp_prepare_tmp_bytes(mat_rows, mat_cols_in, mat_cols_out, mat_size));

            let mut mat: MatZnx<Vec<u8>> = MatZnx::alloc(n, mat_rows, mat_cols_in, mat_cols_out, mat_size);
            mat.raw_mut()
                .iter_mut()
                .for_each(|x| *x = source.next_i32() as i64);

            let mut pmat_0: VmpPMat<Vec<u8>, B> = module.vmp_pmat_alloc(mat_rows, mat_cols_in, mat_cols_out, mat_size);
            let mut pmat_1: VmpPMat<Vec<u8>, B> = module.vmp_pmat_alloc(mat_rows, mat_cols_in, mat_cols_out, mat_size);

            source.fill_bytes(pmat_0.data_mut());
            source.fill_bytes(pmat_1.data_mut());

            module.vmp_prepare(&mut pmat_0, &mat, scratch.borrow());
            vmp_prepare::<_, _, _, Reim4BlkRef, ReimConvRef, ReimFFTRef>(&table, &mut pmat_1, &mat, &mut tmp_bytes);

            assert_approx_eq_slice(pmat_0.raw(), pmat_1.raw(), 1e-10);
        });
    });
}

pub fn bench_vmp_prepare<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: ModuleNew<B> + VmpPMatAlloc<B> + VmpPrepare<B> + VmpPrepareTmpBytes,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vmp_prepare::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 5]) -> impl FnMut()
    where
        Module<B>: ModuleNew<B> + VmpPMatAlloc<B> + VmpPrepare<B> + VmpPrepareTmpBytes,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << params[0]);

        let rows: usize = params[1];
        let cols_in: usize = params[2];
        let cols_out: usize = params[3];
        let size: usize = params[4];

        let mut source: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.vmp_prepare_tmp_bytes(rows, cols_in, cols_out, size));

        let mut mat: MatZnx<Vec<u8>> = MatZnx::alloc(module.n(), rows, cols_in, cols_out, size);
        let mut pmat: VmpPMat<Vec<u8>, B> = module.vmp_pmat_alloc(rows, cols_in, cols_out, size);

        source.fill_bytes(mat.data_mut());
        source.fill_bytes(pmat.data_mut());

        move || {
            module.vmp_prepare(&mut pmat, &mat, scratch.borrow());
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
