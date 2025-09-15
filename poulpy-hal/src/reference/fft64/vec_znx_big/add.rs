use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};
use rand::RngCore;

use crate::{
    api::{ModuleNew, VecZnxBigAdd, VecZnxBigAddInplace, VecZnxBigAddSmall, VecZnxBigAddSmallInplace, VecZnxBigAlloc},
    layouts::{Backend, DataViewMut, Module, VecZnx, VecZnxBig, VecZnxBigToMut, VecZnxBigToRef, VecZnxToRef},
    reference::{
        vec_znx::{vec_znx_add, vec_znx_add_inplace},
        znx::{ZnxAdd, ZnxAddInplace, ZnxCopy, ZnxZero},
    },
    source::Source,
};

pub fn vec_znx_big_add<R, A, B, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend<ScalarBig = i64> + ZnxAdd + ZnxCopy + ZnxZero,
    R: VecZnxBigToMut<BE>,
    A: VecZnxBigToRef<BE>,
    B: VecZnxBigToRef<BE>,
{
    let res: VecZnxBig<&mut [u8], BE> = res.to_mut();
    let a: VecZnxBig<&[u8], BE> = a.to_ref();
    let b: VecZnxBig<&[u8], BE> = b.to_ref();

    let mut res_vznx: VecZnx<&mut [u8]> = VecZnx {
        data: res.data,
        n: res.n,
        cols: res.cols,
        size: res.size,
        max_size: res.max_size,
    };

    let a_vznx: VecZnx<&[u8]> = VecZnx {
        data: a.data,
        n: a.n,
        cols: a.cols,
        size: a.size,
        max_size: a.max_size,
    };

    let b_vznx: VecZnx<&[u8]> = VecZnx {
        data: b.data,
        n: b.n,
        cols: b.cols,
        size: b.size,
        max_size: b.max_size,
    };

    vec_znx_add::<_, _, _, BE>(&mut res_vznx, res_col, &a_vznx, a_col, &b_vznx, b_col);
}

pub fn vec_znx_big_add_inplace<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarBig = i64> + ZnxAddInplace,
    R: VecZnxBigToMut<BE>,
    A: VecZnxBigToRef<BE>,
{
    let res: VecZnxBig<&mut [u8], BE> = res.to_mut();
    let a: VecZnxBig<&[u8], BE> = a.to_ref();

    let mut res_vznx: VecZnx<&mut [u8]> = VecZnx {
        data: res.data,
        n: res.n,
        cols: res.cols,
        size: res.size,
        max_size: res.max_size,
    };

    let a_vznx: VecZnx<&[u8]> = VecZnx {
        data: a.data,
        n: a.n,
        cols: a.cols,
        size: a.size,
        max_size: a.max_size,
    };

    vec_znx_add_inplace::<_, _, BE>(&mut res_vznx, res_col, &a_vznx, a_col);
}

pub fn vec_znx_big_add_small<R, A, B, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend<ScalarBig = i64> + ZnxAdd + ZnxCopy + ZnxZero,
    R: VecZnxBigToMut<BE>,
    A: VecZnxBigToRef<BE>,
    B: VecZnxToRef,
{
    let res: VecZnxBig<&mut [u8], BE> = res.to_mut();
    let a: VecZnxBig<&[u8], BE> = a.to_ref();

    let mut res_vznx: VecZnx<&mut [u8]> = VecZnx {
        data: res.data,
        n: res.n,
        cols: res.cols,
        size: res.size,
        max_size: res.max_size,
    };

    let a_vznx: VecZnx<&[u8]> = VecZnx {
        data: a.data,
        n: a.n,
        cols: a.cols,
        size: a.size,
        max_size: a.max_size,
    };

    vec_znx_add::<_, _, _, BE>(&mut res_vznx, res_col, &a_vznx, a_col, b, b_col);
}

pub fn vec_znx_big_add_small_inplace<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarBig = i64> + ZnxAddInplace,
    R: VecZnxBigToMut<BE>,
    A: VecZnxToRef,
{
    let res: VecZnxBig<&mut [u8], BE> = res.to_mut();
    let mut res_vznx: VecZnx<&mut [u8]> = VecZnx {
        data: res.data,
        n: res.n,
        cols: res.cols,
        size: res.size,
        max_size: res.max_size,
    };

    vec_znx_add_inplace::<_, _, BE>(&mut res_vznx, res_col, a, a_col);
}

pub fn bench_vec_znx_big_add<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxBigAdd<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
{
    let group_name: String = format!("vec_znx_big_add::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxBigAdd<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
    {
        let n: usize = params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxBig<Vec<u8>, B> = module.vec_znx_big_alloc(cols, size);
        let mut b: VecZnxBig<Vec<u8>, B> = module.vec_znx_big_alloc(cols, size);
        let mut c: VecZnxBig<Vec<u8>, B> = module.vec_znx_big_alloc(cols, size);

        // Fill a with random bytes
        source.fill_bytes(a.data_mut());
        source.fill_bytes(b.data_mut());
        source.fill_bytes(c.data_mut());

        move || {
            for i in 0..cols {
                module.vec_znx_big_add(&mut c, i, &a, i, &b, i);
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

pub fn bench_vec_znx_big_add_inplace<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxBigAddInplace<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
{
    let group_name: String = format!("vec_znx_big_add_inplace::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxBigAddInplace<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
    {
        let n: usize = params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxBig<Vec<u8>, B> = module.vec_znx_big_alloc(cols, size);
        let mut c: VecZnxBig<Vec<u8>, B> = module.vec_znx_big_alloc(cols, size);

        // Fill a with random i64
        source.fill_bytes(a.data_mut());
        source.fill_bytes(c.data_mut());

        move || {
            for i in 0..cols {
                module.vec_znx_big_add_inplace(&mut c, i, &a, i);
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

pub fn bench_vec_znx_big_add_small<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxBigAddSmall<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
{
    let group_name: String = format!("vec_znx_big_add_small::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxBigAddSmall<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
    {
        let n: usize = params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxBig<Vec<u8>, B> = module.vec_znx_big_alloc(cols, size);
        let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);
        let mut c: VecZnxBig<Vec<u8>, B> = module.vec_znx_big_alloc(cols, size);

        // Fill a with random i64
        source.fill_bytes(a.data_mut());
        source.fill_bytes(b.data_mut());
        source.fill_bytes(c.data_mut());

        move || {
            for i in 0..cols {
                module.vec_znx_big_add_small(&mut c, i, &a, i, &b, i);
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

pub fn bench_vec_znx_big_add_small_inplace<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxBigAddSmallInplace<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
{
    let group_name: String = format!("vec_znx_big_add_small_inplace::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxBigAddSmallInplace<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
    {
        let n: usize = params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);
        let mut c: VecZnxBig<Vec<u8>, B> = module.vec_znx_big_alloc(cols, size);

        // Fill a with random i64
        source.fill_bytes(a.data_mut());
        source.fill_bytes(c.data_mut());

        move || {
            for i in 0..cols {
                module.vec_znx_big_add_small_inplace(&mut c, i, &a, i);
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
