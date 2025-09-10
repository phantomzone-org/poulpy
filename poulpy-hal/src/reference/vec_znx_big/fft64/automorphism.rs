use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};

use crate::{
    api::{
        ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxBigAutomorphism, VecZnxBigAutomorphismInplace,
        VecZnxBigAutomorphismInplaceTmpBytes,
    },
    layouts::{
        Backend, FillUniform, Module, ScratchOwned, VecZnx, VecZnxBig, VecZnxBigToMut, VecZnxBigToRef, ZnxView, ZnxViewMut,
    },
    oep::VecZnxBigAllocBytesImpl,
    reference::{
        vec_znx::{vec_znx_automorphism, vec_znx_automorphism_inplace},
        znx::{ZnxArithmetic, ZnxArithmeticRef},
    },
    source::Source,
};

pub fn vec_znx_big_automorphism_inplace_tmp_bytes(n: usize) -> usize {
    n * size_of::<i64>()
}

pub fn vec_znx_big_automorphism<R, A, BE, ZNXARI>(p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarBig = i64>,
    R: VecZnxBigToMut<BE>,
    A: VecZnxBigToRef<BE>,
    ZNXARI: ZnxArithmetic,
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

    vec_znx_automorphism::<_, _, ZNXARI>(p, &mut res_vznx, res_col, &a_vznx, a_col);
}

pub fn vec_znx_big_automorphism_inplace<R, BE, ZNXARI>(p: i64, res: &mut R, res_col: usize, tmp: &mut [i64])
where
    BE: Backend<ScalarBig = i64>,
    R: VecZnxBigToMut<BE>,
    ZNXARI: ZnxArithmetic,
{
    let res: VecZnxBig<&mut [u8], _> = res.to_mut();

    let mut res_vznx: VecZnx<&mut [u8]> = VecZnx {
        data: res.data,
        n: res.n,
        cols: res.cols,
        size: res.size,
        max_size: res.max_size,
    };

    vec_znx_automorphism_inplace::<_, ZNXARI>(p, &mut res_vznx, res_col, tmp);
}

pub fn test_vec_znx_big_automorphism<B>(module: &Module<B>)
where
    Module<B>: VecZnxBigAutomorphism<B>,
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

            let p = -7;

            for i in 0..cols {
                vec_znx_big_automorphism::<_, _, _, ZnxArithmeticRef>(p, &mut res_0, i, &a, i);
                module.vec_znx_big_automorphism(p, &mut res_1, i, &a, i);
            }

            assert_eq!(res_0.raw(), res_1.raw());

            for i in 0..cols {
                vec_znx_big_automorphism::<_, _, _, ZnxArithmeticRef>(-p, &mut res_0, i, &a, i);
                module.vec_znx_big_automorphism(-p, &mut res_1, i, &a, i);
            }

            assert_eq!(res_0.raw(), res_1.raw());
        }
    }
}

pub fn test_vec_znx_big_automorphism_inplace<B>(module: &Module<B>)
where
    Module<B>: VecZnxBigAutomorphismInplace<B> + VecZnxBigAutomorphismInplaceTmpBytes,
    B: Backend<ScalarBig = i64> + VecZnxBigAllocBytesImpl<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.vec_znx_big_automorphism_inplace_tmp_bytes());

    let mut tmp: Vec<i64> = vec![0i64; module.n()];

    for res_size in [1, 2, 6, 11] {
        let mut res_0: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, res_size);
        let mut res_1: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, res_size);

        res_0
            .raw_mut()
            .iter_mut()
            .for_each(|x| *x = source.next_i32() as i64);

        res_1.raw_mut().copy_from_slice(res_0.raw());

        let p = -7;

        for i in 0..cols {
            vec_znx_big_automorphism_inplace::<_, _, ZnxArithmeticRef>(p, &mut res_0, i, &mut tmp);
            module.vec_znx_big_automorphism_inplace(p, &mut res_1, i, scratch.borrow());
        }

        assert_eq!(res_0.raw(), res_1.raw());

        for i in 0..cols {
            vec_znx_big_automorphism_inplace::<_, _, ZnxArithmeticRef>(-p, &mut res_0, i, &mut tmp);
            module.vec_znx_big_automorphism_inplace(-p, &mut res_1, i, scratch.borrow());
        }

        assert_eq!(res_0.raw(), res_1.raw());
    }
}

pub fn bench_vec_znx_big_automorphism<B>(c: &mut Criterion, label: &str)
where
    B: Backend<ScalarBig = i64> + VecZnxBigAllocBytesImpl<B>,
    Module<B>: VecZnxBigAutomorphism<B> + ModuleNew<B>,
{
    let group_name: String = format!("vec_znx_big_automorphism::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(params: [usize; 3]) -> impl FnMut()
    where
        B: Backend<ScalarBig = i64> + VecZnxBigAllocBytesImpl<B>,
        Module<B>: VecZnxBigAutomorphism<B> + ModuleNew<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << params[0]);

        let cols: usize = params[1];
        let size: usize = params[2];

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, size);
        let mut res: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, size);

        // Fill a with random i64
        a.fill_uniform(&mut source);
        res.fill_uniform(&mut source);

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

pub fn bench_vec_znx_automorphism_inplace<B>(c: &mut Criterion, label: &str)
where
    B: Backend<ScalarBig = i64> + VecZnxBigAllocBytesImpl<B>,
    Module<B>: VecZnxBigAutomorphismInplace<B> + VecZnxBigAutomorphismInplaceTmpBytes + ModuleNew<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vec_znx_automorphism_inplace::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(params: [usize; 3]) -> impl FnMut()
    where
        B: Backend<ScalarBig = i64> + VecZnxBigAllocBytesImpl<B>,
        Module<B>: VecZnxBigAutomorphismInplace<B> + ModuleNew<B> + VecZnxBigAutomorphismInplaceTmpBytes,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << params[0]);

        let cols: usize = params[1];
        let size: usize = params[2];

        let mut source: Source = Source::new([0u8; 32]);

        let mut res: VecZnxBig<Vec<u8>, B> = VecZnxBig::alloc(module.n(), cols, size);

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.vec_znx_big_automorphism_inplace_tmp_bytes());

        // Fill a with random i64
        res.fill_uniform(&mut source);

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
