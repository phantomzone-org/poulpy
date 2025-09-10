use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};

use crate::{
    api::{ModuleNew, VecZnxBigAdd, VecZnxBigAddInplace, VecZnxBigAddSmall, VecZnxBigAddSmallInplace},
    layouts::{
        Backend, FillUniform, Module, VecZnx, VecZnxBig, VecZnxBigToMut, VecZnxBigToRef, VecZnxToRef, ZnxView, ZnxViewMut,
    },
    oep::VecZnxBigAllocBytesImpl,
    reference::{
        vec_znx::{vec_znx_add, vec_znx_add_inplace},
        znx::{ZnxArithmetic, ZnxArithmeticRef},
    },
    source::Source,
};

pub fn vec_znx_big_add<R, A, B, BE, ZNXARI>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend<ScalarBig = i64>,
    R: VecZnxBigToMut<BE>,
    A: VecZnxBigToRef<BE>,
    B: VecZnxBigToRef<BE>,
    ZNXARI: ZnxArithmetic,
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

    vec_znx_add::<_, _, _, ZNXARI>(&mut res_vznx, res_col, &a_vznx, a_col, &b_vznx, b_col);
}

pub fn vec_znx_big_add_inplace<R, A, BE, ZNXARI>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarBig = i64>,
    R: VecZnxBigToMut<BE>,
    A: VecZnxBigToRef<BE>,
    ZNXARI: ZnxArithmetic,
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

    vec_znx_add_inplace::<_, _, ZNXARI>(&mut res_vznx, res_col, &a_vznx, a_col);
}

pub fn vec_znx_big_add_small<R, A, B, BE, ZNXARI>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend<ScalarBig = i64>,
    R: VecZnxBigToMut<BE>,
    A: VecZnxBigToRef<BE>,
    B: VecZnxToRef,
    ZNXARI: ZnxArithmetic,
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

    vec_znx_add::<_, _, _, ZNXARI>(&mut res_vznx, res_col, &a_vznx, a_col, b, b_col);
}

pub fn vec_znx_big_add_small_inplace<R, A, BE, ZNXARI>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarBig = i64>,
    R: VecZnxBigToMut<BE>,
    A: VecZnxToRef,
    ZNXARI: ZnxArithmetic,
{
    let res: VecZnxBig<&mut [u8], BE> = res.to_mut();
    let mut res_vznx: VecZnx<&mut [u8]> = VecZnx {
        data: res.data,
        n: res.n,
        cols: res.cols,
        size: res.size,
        max_size: res.max_size,
    };

    vec_znx_add_inplace::<_, _, ZNXARI>(&mut res_vznx, res_col, a, a_col);
}

pub fn test_vec_znx_big_add<B>(module: &Module<B>)
where
    Module<B>: VecZnxBigAdd<B>,
    B: Backend<ScalarBig = i64> + VecZnxBigAllocBytesImpl<B>,
{
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 6, 11] {
        let mut a: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, a_size);
        a.raw_mut()
            .iter_mut()
            .for_each(|x| *x = source.next_i32() as i64);

        for b_size in [1, 2, 6, 11] {
            let mut b: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, b_size);
            b.raw_mut()
                .iter_mut()
                .for_each(|x| *x = source.next_i32() as i64);

            for res_size in [1, 2, 6, 11] {
                let mut res_0: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, res_size);
                let mut res_1: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, res_size);

                // Set d to garbage
                res_0.fill_uniform(&mut source);
                res_1.fill_uniform(&mut source);

                // Reference
                for i in 0..cols {
                    vec_znx_big_add::<_, _, _, _, ZnxArithmeticRef>(&mut res_0, i, &a, i, &b, i);
                    module.vec_znx_big_add(&mut res_1, i, &a, i, &b, i);
                }

                assert_eq!(res_0.raw(), res_1.raw());
            }
        }
    }
}

pub fn test_vec_znx_big_add_inplace<B>(module: &Module<B>)
where
    Module<B>: VecZnxBigAddInplace<B>,
    B: Backend<ScalarBig = i64> + VecZnxBigAllocBytesImpl<B>,
{
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 6, 11] {
        let mut a: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, a_size);
        a.raw_mut()
            .iter_mut()
            .for_each(|x| *x = source.next_i32() as i64);

        for res_size in [1, 2, 6, 11] {
            let mut res_0: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, res_size);
            let mut res_1: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, res_size);

            res_0
                .raw_mut()
                .iter_mut()
                .for_each(|x| *x = source.next_i32() as i64);

            res_1.raw_mut().copy_from_slice(res_0.raw());

            for i in 0..cols {
                vec_znx_big_add_inplace::<_, _, _, ZnxArithmeticRef>(&mut res_0, i, &a, i);
                module.vec_znx_big_add_inplace(&mut res_1, i, &a, i);
            }

            assert_eq!(res_0.raw(), res_1.raw());
        }
    }
}

pub fn test_vec_znx_big_add_small<B>(module: &Module<B>)
where
    Module<B>: VecZnxBigAddSmall<B>,
    B: Backend<ScalarBig = i64> + VecZnxBigAllocBytesImpl<B>,
{
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 6, 11] {
        let mut a: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, a_size);
        a.raw_mut()
            .iter_mut()
            .for_each(|x| *x = source.next_i32() as i64);

        for b_size in [1, 2, 6, 11] {
            let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, b_size);
            b.raw_mut()
                .iter_mut()
                .for_each(|x| *x = source.next_i32() as i64);

            for res_size in [1, 2, 6, 11] {
                let mut res_0: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, res_size);
                let mut res_1: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, res_size);

                // Set d to garbage
                res_0.fill_uniform(&mut source);
                res_1.fill_uniform(&mut source);

                // Reference
                for i in 0..cols {
                    vec_znx_big_add_small::<_, _, _, _, ZnxArithmeticRef>(&mut res_0, i, &a, i, &b, i);
                    module.vec_znx_big_add_small(&mut res_1, i, &a, i, &b, i);
                }

                assert_eq!(res_0.raw(), res_1.raw());
            }
        }
    }
}

pub fn test_vec_znx_big_add_small_inplace<B>(module: &Module<B>)
where
    Module<B>: VecZnxBigAddSmallInplace<B>,
    B: Backend<ScalarBig = i64> + VecZnxBigAllocBytesImpl<B>,
{
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 6, 11] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, a_size);
        a.raw_mut()
            .iter_mut()
            .for_each(|x| *x = source.next_i32() as i64);

        for res_size in [1, 2, 6, 11] {
            let mut res_0: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, res_size);
            let mut res_1: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, res_size);

            res_0
                .raw_mut()
                .iter_mut()
                .for_each(|x| *x = source.next_i32() as i64);

            res_1.raw_mut().copy_from_slice(res_0.raw());

            for i in 0..cols {
                vec_znx_big_add_small_inplace::<_, _, _, ZnxArithmeticRef>(&mut res_0, i, &a, i);
                module.vec_znx_big_add_small_inplace(&mut res_1, i, &a, i);
            }

            assert_eq!(res_0.raw(), res_1.raw());
        }
    }
}

pub fn bench_vec_znx_big_add<B>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxBigAdd<B> + ModuleNew<B>,
    B: Backend + VecZnxBigAllocBytesImpl<B>,
{
    let group_name: String = format!("vec_znx_big_add::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxBigAdd<B> + ModuleNew<B>,
        B: Backend + VecZnxBigAllocBytesImpl<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << params[0]);

        let cols: usize = params[1];
        let size: usize = params[2];

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, size);
        let mut b: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, size);
        let mut c: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, size);

        // Fill a with random i64
        a.fill_uniform(&mut source);
        b.fill_uniform(&mut source);
        c.fill_uniform(&mut source);

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

pub fn bench_vec_znx_big_add_inplace<B>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxBigAddInplace<B> + ModuleNew<B>,
    B: Backend + VecZnxBigAllocBytesImpl<B>,
{
    let group_name: String = format!("vec_znx_big_add_inplace::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxBigAddInplace<B> + ModuleNew<B>,
        B: Backend + VecZnxBigAllocBytesImpl<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << params[0]);

        let cols: usize = params[1];
        let size: usize = params[2];

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, size);
        let mut c: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, size);

        // Fill a with random i64
        a.fill_uniform(&mut source);
        c.fill_uniform(&mut source);

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

pub fn bench_vec_znx_big_add_small<B>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxBigAddSmall<B> + ModuleNew<B>,
    B: Backend + VecZnxBigAllocBytesImpl<B>,
{
    let group_name: String = format!("vec_znx_big_add_small::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxBigAddSmall<B> + ModuleNew<B>,
        B: Backend + VecZnxBigAllocBytesImpl<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << params[0]);

        let cols: usize = params[1];
        let size: usize = params[2];

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, size);
        let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);
        let mut c: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, size);

        // Fill a with random i64
        a.fill_uniform(&mut source);
        b.fill_uniform(&mut source);
        c.fill_uniform(&mut source);

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

pub fn bench_vec_znx_big_add_small_inplace<B>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxBigAddSmallInplace<B> + ModuleNew<B>,
    B: Backend + VecZnxBigAllocBytesImpl<B>,
{
    let group_name: String = format!("vec_znx_big_add_small_inplace::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxBigAddSmallInplace<B> + ModuleNew<B>,
        B: Backend + VecZnxBigAllocBytesImpl<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << params[0]);

        let cols: usize = params[1];
        let size: usize = params[2];

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);
        let mut c: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, size);

        // Fill a with random i64
        a.fill_uniform(&mut source);
        c.fill_uniform(&mut source);

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
