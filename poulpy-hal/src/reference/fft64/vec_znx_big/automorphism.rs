use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};
use rand::RngCore;

use crate::{
    api::{
        ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxBigAlloc, VecZnxBigAutomorphism, VecZnxBigAutomorphismInplace,
        VecZnxBigAutomorphismInplaceTmpBytes,
    },
    layouts::{Backend, DataViewMut, Module, ScratchOwned, VecZnx, VecZnxBig, VecZnxBigToMut, VecZnxBigToRef},
    reference::{
        vec_znx::{vec_znx_automorphism, vec_znx_automorphism_inplace},
        znx::{ZnxAutomorphism, ZnxCopy, ZnxZero},
    },
    source::Source,
};

pub fn vec_znx_big_automorphism_inplace_tmp_bytes(n: usize) -> usize {
    n * size_of::<i64>()
}

pub fn vec_znx_big_automorphism<R, A, BE>(p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarBig = i64> + ZnxAutomorphism + ZnxZero,
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

    vec_znx_automorphism::<_, _, BE>(p, &mut res_vznx, res_col, &a_vznx, a_col);
}

pub fn vec_znx_big_automorphism_inplace<R, BE>(p: i64, res: &mut R, res_col: usize, tmp: &mut [i64])
where
    BE: Backend<ScalarBig = i64> + ZnxAutomorphism + ZnxCopy,
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

    vec_znx_automorphism_inplace::<_, BE>(p, &mut res_vznx, res_col, tmp);
}

pub fn bench_vec_znx_big_automorphism<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxBigAutomorphism<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
{
    let group_name: String = format!("vec_znx_big_automorphism::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxBigAutomorphism<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
    {
        let n: usize = params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxBig<Vec<u8>, B> = module.vec_znx_big_alloc(cols, size);
        let mut res: VecZnxBig<Vec<u8>, B> = module.vec_znx_big_alloc(cols, size);

        // Fill a with random i64
        source.fill_bytes(a.data_mut());
        source.fill_bytes(res.data_mut());

        move || {
            for i in 0..cols {
                module.vec_znx_big_automorphism(-7, &mut res, i, &a, i);
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

pub fn bench_vec_znx_automorphism_inplace<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxBigAutomorphismInplace<B> + VecZnxBigAutomorphismInplaceTmpBytes + ModuleNew<B> + VecZnxBigAlloc<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vec_znx_automorphism_inplace::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxBigAutomorphismInplace<B> + ModuleNew<B> + VecZnxBigAutomorphismInplaceTmpBytes + VecZnxBigAlloc<B>,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let n: usize = params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut res: VecZnxBig<Vec<u8>, B> = module.vec_znx_big_alloc(cols, size);

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.vec_znx_big_automorphism_inplace_tmp_bytes());

        // Fill a with random i64
        source.fill_bytes(res.data_mut());

        move || {
            for i in 0..cols {
                module.vec_znx_big_automorphism_inplace(-7, &mut res, i, scratch.borrow());
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
