use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};
use rand::RngCore;

use crate::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VmpPMatAlloc, VmpPrepare, VmpPrepareTmpBytes},
    layouts::{
        Backend, DataViewMut, MatZnx, MatZnxToRef, Module, ScratchOwned, VmpPMat, VmpPMatToMut, ZnxInfos, ZnxView, ZnxViewMut,
    },
    reference::fft64::{
        reim::{ReimDFTExecute, ReimFFTTable, ReimFromZnx},
        reim4::Reim4Extract1Blk,
    },
    source::Source,
};

pub fn vmp_prepare_tmp_bytes(n: usize) -> usize {
    n * size_of::<i64>()
}

pub fn vmp_prepare<R, A, BE>(table: &ReimFFTTable<f64>, pmat: &mut R, mat: &A, tmp: &mut [f64])
where
    BE: Backend<ScalarPrep = f64> + ReimDFTExecute<ReimFFTTable<f64>, f64> + ReimFromZnx + Reim4Extract1Blk,
    R: VmpPMatToMut<BE>,
    A: MatZnxToRef,
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
    vmp_prepare_core::<BE>(table, res.raw_mut(), a.raw(), nrows, ncols, tmp);
}

pub(crate) fn vmp_prepare_core<REIM>(
    table: &ReimFFTTable<f64>,
    pmat: &mut [f64],
    mat: &[i64],
    nrows: usize,
    ncols: usize,
    tmp: &mut [f64],
) where
    REIM: ReimDFTExecute<ReimFFTTable<f64>, f64> + ReimFromZnx + Reim4Extract1Blk,
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

            REIM::reim_from_znx(tmp, &mat[pos..pos + n]);
            REIM::reim_dft_execute(table, tmp);

            let dst: &mut [f64] = if col_i == (ncols - 1) && !ncols.is_multiple_of(2) {
                &mut pmat[col_i * nrows * 8 + row_i * 8..]
            } else {
                &mut pmat[(col_i / 2) * (nrows * 16) + row_i * 16 + (col_i % 2) * 8..]
            };

            for blk_i in 0..m >> 2 {
                REIM::reim4_extract_1blk(m, 1, blk_i, &mut dst[blk_i * offset..], tmp);
            }
        }
    }
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
