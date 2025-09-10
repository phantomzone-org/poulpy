use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};

use crate::{
    api::{ModuleNew, VecZnxNegate, VecZnxNegateInplace},
    layouts::{Backend, FillUniform, Module, VecZnx, VecZnxToMut, VecZnxToRef, ZnxInfos, ZnxView, ZnxViewMut},
    reference::znx::{ZnxArithmetic, ZnxArithmeticRef},
    source::Source,
};

pub fn vec_znx_negate<R, A, ZNXARI>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    R: VecZnxToMut,
    A: VecZnxToRef,
    ZNXARI: ZnxArithmetic,
{
    let a: VecZnx<&[u8]> = a.to_ref();
    let mut res: VecZnx<&mut [u8]> = res.to_mut();

    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), res.n());
    }

    let min_size: usize = res.size().min(a.size());

    for j in 0..min_size {
        ZNXARI::znx_negate(res.at_mut(res_col, j), a.at(a_col, j));
    }

    for j in min_size..res.size() {
        ZNXARI::znx_zero(res.at_mut(res_col, j));
    }
}

pub fn vec_znx_negate_inplace<R, ZNXARI>(res: &mut R, res_col: usize)
where
    R: VecZnxToMut,
    ZNXARI: ZnxArithmetic,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();
    for j in 0..res.size() {
        ZNXARI::znx_negate_inplace(res.at_mut(res_col, j));
    }
}

pub fn test_vec_znx_negate<B: Backend>(module: &Module<B>)
where
    Module<B>: VecZnxNegate,
{
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 6, 11] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, a_size);
        a.raw_mut()
            .iter_mut()
            .for_each(|x| *x = source.next_i32() as i64);

        for res_size in [1, 2, 6, 11] {
            let mut res_0: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, res_size);
            let mut res_1: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, res_size);

            res_0
                .raw_mut()
                .iter_mut()
                .for_each(|x| *x = source.next_i32() as i64);

            res_1.raw_mut().copy_from_slice(res_0.raw());

            for i in 0..cols {
                vec_znx_negate::<_, _, ZnxArithmeticRef>(&mut res_0, i, &a, i);
                module.vec_znx_negate(&mut res_1, i, &a, i);
            }

            assert_eq!(res_0.raw(), res_1.raw());
        }
    }
}

pub fn test_vec_znx_negate_inplace<B: Backend>(module: &Module<B>)
where
    Module<B>: VecZnxNegateInplace,
{
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for res_size in [1, 2, 6, 11] {
        let mut res_0: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, res_size);
        let mut res_1: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, res_size);

        res_0
            .raw_mut()
            .iter_mut()
            .for_each(|x| *x = source.next_i32() as i64);

        res_1.raw_mut().copy_from_slice(res_0.raw());

        for i in 0..cols {
            vec_znx_negate_inplace::<_, ZnxArithmeticRef>(&mut res_0, i);
            module.vec_znx_negate_inplace(&mut res_1, i);
        }

        assert_eq!(res_0.raw(), res_1.raw());
    }
}

pub fn bench_vec_znx_negate<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxNegate + ModuleNew<B>,
{
    let group_name: String = format!("vec_znx_negate::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxNegate + ModuleNew<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << params[0]);

        let cols: usize = params[1];
        let size: usize = params[2];

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);
        let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);

        // Fill a with random i64
        a.fill_uniform(&mut source);
        b.fill_uniform(&mut source);

        move || {
            for i in 0..cols {
                module.vec_znx_negate(&mut b, i, &a, i);
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

pub fn bench_vec_znx_negate_inplace<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxNegateInplace + ModuleNew<B>,
{
    let group_name: String = format!("vec_znx_negate_inplace::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxNegateInplace + ModuleNew<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << params[0]);

        let cols: usize = params[1];
        let size: usize = params[2];

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);

        // Fill a with random i64
        a.fill_uniform(&mut source);
        move || {
            for i in 0..cols {
                module.vec_znx_negate_inplace(&mut a, i);
            }
            black_box(());
        }
    }

    for params in [[10, 2, 2], [11, 2, 4], [12, 2, 8], [13, 2, 16], [14, 2, 32]] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << params[0], params[1], params[2]));
        let mut runner = runner::<B>(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}
