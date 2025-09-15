use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};
use rand::RngCore;

use crate::{
    api::{ModuleNew, VecZnxBigAlloc, VecZnxBigNegate, VecZnxBigNegateInplace},
    layouts::{Backend, DataViewMut, Module, VecZnx, VecZnxBig, VecZnxBigToMut, VecZnxBigToRef},
    reference::{
        vec_znx::{vec_znx_negate, vec_znx_negate_inplace},
        znx::{ZnxNegate, ZnxNegateInplace, ZnxZero},
    },
    source::Source,
};

pub fn vec_znx_big_negate<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarBig = i64> + ZnxNegate + ZnxZero,
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

    vec_znx_negate::<_, _, BE>(&mut res_vznx, res_col, &a_vznx, a_col);
}

pub fn vec_znx_big_negate_inplace<R, BE>(res: &mut R, res_col: usize)
where
    BE: Backend<ScalarBig = i64> + ZnxNegateInplace,
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

    vec_znx_negate_inplace::<_, BE>(&mut res_vznx, res_col);
}

pub fn bench_vec_znx_big_negate<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxBigNegate<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
{
    let group_name: String = format!("vec_znx_big_negate::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxBigNegate<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << params[0]);

        let cols: usize = params[1];
        let size: usize = params[2];

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxBig<Vec<u8>, B> = module.vec_znx_big_alloc(cols, size);
        let mut b: VecZnxBig<Vec<u8>, B> = module.vec_znx_big_alloc(cols, size);

        // Fill a with random i64
        source.fill_bytes(a.data_mut());
        source.fill_bytes(b.data_mut());

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

pub fn bench_vec_znx_big_negate_inplace<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxBigNegateInplace<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
{
    let group_name: String = format!("vec_znx_negate_big_inplace::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxBigNegateInplace<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << params[0]);

        let cols: usize = params[1];
        let size: usize = params[2];

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxBig<Vec<u8>, B> = module.vec_znx_big_alloc(cols, size);

        // Fill a with random i64
        source.fill_bytes(a.data_mut());

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
