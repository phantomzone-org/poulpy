use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};

use crate::{
    api::{ModuleNew, VecZnxBigNegate, VecZnxBigNegateInplace},
    layouts::{Backend, FillUniform, Module, VecZnx, VecZnxBig, VecZnxBigToMut, VecZnxBigToRef, ZnxView, ZnxViewMut},
    oep::VecZnxBigAllocBytesImpl,
    reference::vec_znx::{vec_znx_negate_inplace_ref, vec_znx_negate_ref},
    source::Source,
};

pub fn vec_znx_big_negate_ref<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarBig = i64>,
    R: VecZnxBigToMut<BE>,
    A: VecZnxBigToRef<BE>,
{
    let res: VecZnxBig<&mut [u8], _> = res.to_mut();
    let a: VecZnxBig<&[u8], _> = a.to_ref();

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

    vec_znx_negate_ref(&mut res_vznx, res_col, &a_vznx, a_col);
}

pub fn vec_znx_big_negate_inplace_ref<R, BE>(res: &mut R, res_col: usize)
where
    BE: Backend<ScalarBig = i64>,
    R: VecZnxBigToMut<BE>,
{
    let res: VecZnxBig<&mut [u8], _> = res.to_mut();

    let mut res_vznx: VecZnx<&mut [u8]> = VecZnx {
        data: res.data,
        n: res.n,
        cols: res.cols,
        size: res.size,
        max_size: res.max_size,
    };

    vec_znx_negate_inplace_ref(&mut res_vznx, res_col);
}

pub fn test_vec_znx_big_negate<B>(module: &Module<B>)
where
    Module<B>: VecZnxBigNegate<B>,
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
                vec_znx_big_negate_ref(&mut res_0, i, &a, i);
                module.vec_znx_big_negate(&mut res_1, i, &a, i);
            }

            assert_eq!(res_0.raw(), res_1.raw());
        }
    }
}

pub fn test_vec_znx_big_negate_inplace<B>(module: &Module<B>)
where
    Module<B>: VecZnxBigNegateInplace<B>,
    B: Backend<ScalarBig = i64> + VecZnxBigAllocBytesImpl<B>,
{
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for res_size in [1, 2, 6, 11] {
        let mut res_0: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, res_size);
        let mut res_1: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, res_size);

        res_0
            .raw_mut()
            .iter_mut()
            .for_each(|x| *x = source.next_i32() as i64);

        res_1.raw_mut().copy_from_slice(res_0.raw());

        for i in 0..cols {
            vec_znx_big_negate_inplace_ref(&mut res_0, i);
            module.vec_znx_big_negate_inplace(&mut res_1, i);
        }

        assert_eq!(res_0.raw(), res_1.raw());
    }
}

pub fn bench_vec_znx_big_negate<B>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxBigNegate<B> + ModuleNew<B>,
    B: Backend<ScalarBig = i64> + VecZnxBigAllocBytesImpl<B>,
{
    let group_name: String = format!("vec_znx_big_negate::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxBigNegate<B> + ModuleNew<B>,
        B: Backend<ScalarBig = i64> + VecZnxBigAllocBytesImpl<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << params[0]);

        let cols: usize = params[1];
        let size: usize = params[2];

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, size);
        let mut b: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, size);

        // Fill a with random i64
        a.fill_uniform(&mut source);
        b.fill_uniform(&mut source);

        move || {
            for i in 0..cols {
                module.vec_znx_big_negate(&mut b, i, &a, i);
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

pub fn bench_vec_znx_big_negate_inplace<B>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxBigNegateInplace<B> + ModuleNew<B>,
    B: Backend<ScalarBig = i64> + VecZnxBigAllocBytesImpl<B>,
{
    let group_name: String = format!("vec_znx_negate_big_inplace::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxBigNegateInplace<B> + ModuleNew<B>,
        B: Backend<ScalarBig = i64> + VecZnxBigAllocBytesImpl<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << params[0]);

        let cols: usize = params[1];
        let size: usize = params[2];

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, size);

        // Fill a with random i64
        a.fill_uniform(&mut source);

        move || {
            for i in 0..cols {
                module.vec_znx_big_negate_inplace(&mut a, i);
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
