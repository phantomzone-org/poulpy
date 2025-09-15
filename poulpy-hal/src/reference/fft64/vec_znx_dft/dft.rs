use std::hint::black_box;

use bytemuck::cast_slice_mut;
use criterion::{BenchmarkId, Criterion};
use rand::RngCore;

use crate::{
    api::{
        ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxBigAlloc, VecZnxDftAlloc, VecZnxDftApply, VecZnxIdftApply,
        VecZnxIdftApplyTmpA, VecZnxIdftApplyTmpBytes,
    },
    layouts::{
        Backend, Data, DataViewMut, Module, ScratchOwned, VecZnx, VecZnxBig, VecZnxBigToMut, VecZnxDft, VecZnxDftToMut,
        VecZnxDftToRef, VecZnxToRef, ZnxInfos, ZnxView, ZnxViewMut,
    },
    reference::{
        fft64::reim::{
            ReimCopy, ReimDFTExecute, ReimFFTTable, ReimFromZnx, ReimIFFTTable, ReimToZnx, ReimToZnxInplace, ReimZero,
        },
        znx::ZnxZero,
    },
    source::Source,
};

pub fn vec_znx_dft_apply<R, A, BE>(
    table: &ReimFFTTable<f64>,
    step: usize,
    offset: usize,
    res: &mut R,
    res_col: usize,
    a: &A,
    a_col: usize,
) where
    BE: Backend<ScalarPrep = f64> + ReimDFTExecute<ReimFFTTable<f64>, f64> + ReimFromZnx + ReimZero,
    R: VecZnxDftToMut<BE>,
    A: VecZnxToRef,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: VecZnx<&[u8]> = a.to_ref();

    #[cfg(debug_assertions)]
    {
        assert!(step > 0);
        assert_eq!(table.m() << 1, res.n());
        assert_eq!(a.n(), res.n());
    }

    let a_size: usize = a.size();
    let res_size: usize = res.size();

    let steps: usize = a_size.div_ceil(step);
    let min_steps: usize = res_size.min(steps);

    for j in 0..min_steps {
        let limb = offset + j * step;
        if limb < a_size {
            BE::reim_from_znx(res.at_mut(res_col, j), a.at(a_col, limb));
            BE::reim_dft_execute(table, res.at_mut(res_col, j));
        }
    }

    (min_steps..res.size()).for_each(|j| {
        BE::reim_zero(res.at_mut(res_col, j));
    });
}

pub fn vec_znx_idft_apply<R, A, BE>(table: &ReimIFFTTable<f64>, res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarPrep = f64, ScalarBig = i64>
        + ReimDFTExecute<ReimIFFTTable<f64>, f64>
        + ReimCopy
        + ReimToZnxInplace
        + ZnxZero,
    R: VecZnxBigToMut<BE>,
    A: VecZnxDftToRef<BE>,
{
    let mut res: VecZnxBig<&mut [u8], BE> = res.to_mut();
    let a: VecZnxDft<&[u8], BE> = a.to_ref();

    #[cfg(debug_assertions)]
    {
        assert_eq!(table.m() << 1, res.n());
        assert_eq!(a.n(), res.n());
    }

    let res_size: usize = res.size();
    let min_size: usize = res_size.min(a.size());

    let divisor: f64 = table.m() as f64;

    for j in 0..min_size {
        let res_slice_f64: &mut [f64] = cast_slice_mut(res.at_mut(res_col, j));
        BE::reim_copy(res_slice_f64, a.at(a_col, j));
        BE::reim_dft_execute(table, res_slice_f64);
        BE::reim_to_znx_inplace(res_slice_f64, divisor);
    }

    for j in min_size..res_size {
        BE::znx_zero(res.at_mut(res_col, j));
    }
}

pub fn vec_znx_idft_apply_tmpa<R, A, BE>(table: &ReimIFFTTable<f64>, res: &mut R, res_col: usize, a: &mut A, a_col: usize)
where
    BE: Backend<ScalarPrep = f64, ScalarBig = i64> + ReimDFTExecute<ReimIFFTTable<f64>, f64> + ReimToZnx + ZnxZero,
    R: VecZnxBigToMut<BE>,
    A: VecZnxDftToMut<BE>,
{
    let mut res: VecZnxBig<&mut [u8], BE> = res.to_mut();
    let mut a: VecZnxDft<&mut [u8], BE> = a.to_mut();

    #[cfg(debug_assertions)]
    {
        assert_eq!(table.m() << 1, res.n());
        assert_eq!(a.n(), res.n());
    }

    let res_size = res.size();
    let min_size: usize = res_size.min(a.size());

    let divisor: f64 = table.m() as f64;

    for j in 0..min_size {
        BE::reim_dft_execute(table, a.at_mut(a_col, j));
        BE::reim_to_znx(res.at_mut(res_col, j), divisor, a.at(a_col, j));
    }

    for j in min_size..res_size {
        BE::znx_zero(res.at_mut(res_col, j));
    }
}

pub fn vec_znx_idft_apply_consume<D: Data, BE>(table: &ReimIFFTTable<f64>, mut res: VecZnxDft<D, BE>) -> VecZnxBig<D, BE>
where
    BE: Backend<ScalarPrep = f64, ScalarBig = i64> + ReimDFTExecute<ReimIFFTTable<f64>, f64> + ReimToZnxInplace,
    VecZnxDft<D, BE>: VecZnxDftToMut<BE>,
{
    {
        let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(table.m() << 1, res.n());
        }

        let divisor: f64 = table.m() as f64;

        for i in 0..res.cols() {
            for j in 0..res.size() {
                BE::reim_dft_execute(table, res.at_mut(i, j));
                BE::reim_to_znx_inplace(res.at_mut(i, j), divisor);
            }
        }
    }

    res.into_big()
}

pub fn bench_vec_znx_dft_apply<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxDftApply<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
{
    let group_name: String = format!("vec_znx_dft_apply::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxDftApply<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
    {
        let n: usize = params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut res: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(cols, size);
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);
        source.fill_bytes(res.data_mut());
        source.fill_bytes(a.data_mut());

        move || {
            for i in 0..cols {
                module.vec_znx_dft_apply(1, 0, &mut res, i, &a, i);
            }
            black_box(());
        }
    }

    for params in [[10, 2, 2], [11, 2, 4], [12, 2, 8], [13, 2, 16], [14, 2, 32]] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << params[0], params[1], params[2],));
        let mut runner = runner::<B>(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vec_znx_idft_apply<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxIdftApply<B> + ModuleNew<B> + VecZnxIdftApplyTmpBytes + VecZnxDftAlloc<B> + VecZnxBigAlloc<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vec_znx_idft_apply::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxIdftApply<B> + ModuleNew<B> + VecZnxIdftApplyTmpBytes + VecZnxDftAlloc<B> + VecZnxBigAlloc<B>,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let n: usize = params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut res: VecZnxBig<Vec<u8>, B> = module.vec_znx_big_alloc(cols, size);
        let mut a: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(cols, size);
        source.fill_bytes(res.data_mut());
        source.fill_bytes(a.data_mut());

        let mut scratch = ScratchOwned::alloc(module.vec_znx_idft_apply_tmp_bytes());

        move || {
            for i in 0..cols {
                module.vec_znx_idft_apply(&mut res, i, &a, i, scratch.borrow());
            }
            black_box(());
        }
    }

    for params in [[10, 2, 2], [11, 2, 4], [12, 2, 8], [13, 2, 16], [14, 2, 32]] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << params[0], params[1], params[2],));
        let mut runner = runner::<B>(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vec_znx_idft_apply_tmpa<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxIdftApplyTmpA<B> + ModuleNew<B> + VecZnxDftAlloc<B> + VecZnxBigAlloc<B>,
{
    let group_name: String = format!("vec_znx_idft_apply_tmpa::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxIdftApplyTmpA<B> + ModuleNew<B> + VecZnxDftAlloc<B> + VecZnxBigAlloc<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << params[0]);

        let cols: usize = params[1];
        let size: usize = params[2];

        let mut source: Source = Source::new([0u8; 32]);

        let mut res: VecZnxBig<Vec<u8>, B> = module.vec_znx_big_alloc(cols, size);
        let mut a: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(cols, size);
        source.fill_bytes(res.data_mut());
        source.fill_bytes(a.data_mut());

        move || {
            for i in 0..cols {
                module.vec_znx_idft_apply_tmpa(&mut res, i, &mut a, i);
            }
            black_box(());
        }
    }

    for params in [[10, 2, 2], [11, 2, 4], [12, 2, 8], [13, 2, 16], [14, 2, 32]] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << params[0], params[1], params[2],));
        let mut runner = runner::<B>(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}
