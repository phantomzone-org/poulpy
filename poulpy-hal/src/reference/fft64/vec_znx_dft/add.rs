use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};
use rand::RngCore;

use crate::{
    api::{ModuleNew, VecZnxDftAdd, VecZnxDftAddInplace, VecZnxDftAlloc},
    layouts::{Backend, DataViewMut, Module, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef, ZnxInfos, ZnxView, ZnxViewMut},
    reference::fft64::reim::{ReimAdd, ReimAddInplace, ReimCopy, ReimZero},
    source::Source,
};

pub fn vec_znx_dft_add<R, A, B, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend<ScalarPrep = f64> + ReimAdd + ReimCopy + ReimZero,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
    B: VecZnxDftToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: VecZnxDft<&[u8], BE> = a.to_ref();
    let b: VecZnxDft<&[u8], BE> = b.to_ref();

    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), res.n());
        assert_eq!(b.n(), res.n());
    }

    let res_size: usize = res.size();
    let a_size: usize = a.size();
    let b_size: usize = b.size();

    if a_size <= b_size {
        let sum_size: usize = a_size.min(res_size);
        let cpy_size: usize = b_size.min(res_size);

        for j in 0..sum_size {
            BE::reim_add(res.at_mut(res_col, j), a.at(a_col, j), b.at(b_col, j));
        }

        for j in sum_size..cpy_size {
            BE::reim_copy(res.at_mut(res_col, j), b.at(b_col, j));
        }

        for j in cpy_size..res_size {
            BE::reim_zero(res.at_mut(res_col, j));
        }
    } else {
        let sum_size: usize = b_size.min(res_size);
        let cpy_size: usize = a_size.min(res_size);

        for j in 0..sum_size {
            BE::reim_add(res.at_mut(res_col, j), a.at(a_col, j), b.at(b_col, j));
        }

        for j in sum_size..cpy_size {
            BE::reim_copy(res.at_mut(res_col, j), a.at(a_col, j));
        }

        for j in cpy_size..res_size {
            BE::reim_zero(res.at_mut(res_col, j));
        }
    }
}

pub fn vec_znx_dft_add_inplace<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarPrep = f64> + ReimAddInplace,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
{
    let a: VecZnxDft<&[u8], BE> = a.to_ref();
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();

    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), res.n());
    }

    let res_size: usize = res.size();
    let a_size: usize = a.size();

    let sum_size: usize = a_size.min(res_size);

    for j in 0..sum_size {
        BE::reim_add_inplace(res.at_mut(res_col, j), a.at(a_col, j));
    }
}

pub fn bench_vec_znx_dft_add<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxDftAdd<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
{
    let group_name: String = format!("vec_znx_dft_add::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxDftAdd<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
    {
        let n: usize = params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(cols, size);
        let mut b: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(cols, size);
        let mut c: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(cols, size);

        source.fill_bytes(a.data_mut());
        source.fill_bytes(b.data_mut());
        source.fill_bytes(c.data_mut());

        move || {
            for i in 0..cols {
                module.vec_znx_dft_add(&mut c, i, &a, i, &b, i);
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

pub fn bench_vec_znx_dft_add_inplace<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxDftAddInplace<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
{
    let group_name: String = format!("vec_znx_dft_add_inplace::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxDftAddInplace<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
    {
        let n: usize = params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(cols, size);
        let mut c: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(cols, size);

        // Fill a with random i64
        source.fill_bytes(a.data_mut());
        source.fill_bytes(c.data_mut());

        move || {
            for i in 0..cols {
                module.vec_znx_dft_add_inplace(&mut c, i, &a, i);
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
