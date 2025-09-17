use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};

use crate::{
    api::{
        ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxMulXpMinusOne, VecZnxMulXpMinusOneInplace,
        VecZnxMulXpMinusOneInplaceTmpBytes,
    },
    layouts::{Backend, FillUniform, Module, ScratchOwned, VecZnx, VecZnxToMut, VecZnxToRef, ZnxInfos, ZnxView, ZnxViewMut},
    reference::{
        vec_znx::{vec_znx_rotate, vec_znx_sub_ab_inplace},
        znx::{ZnxNegate, ZnxRotate, ZnxSubABInplace, ZnxSubBAInplace, ZnxZero},
    },
    source::Source,
};

pub fn vec_znx_mul_xp_minus_one_inplace_tmp_bytes(n: usize) -> usize {
    n * size_of::<i64>()
}

pub fn vec_znx_mul_xp_minus_one<R, A, ZNXARI>(p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    R: VecZnxToMut,
    A: VecZnxToRef,
    ZNXARI: ZnxRotate + ZnxZero + ZnxSubABInplace,
{
    vec_znx_rotate::<_, _, ZNXARI>(p, res, res_col, a, a_col);
    vec_znx_sub_ab_inplace::<_, _, ZNXARI>(res, res_col, a, a_col);
}

pub fn vec_znx_mul_xp_minus_one_inplace<R, ZNXARI>(p: i64, res: &mut R, res_col: usize, tmp: &mut [i64])
where
    R: VecZnxToMut,
    ZNXARI: ZnxRotate + ZnxNegate + ZnxSubBAInplace,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.n(), tmp.len());
    }
    for j in 0..res.size() {
        ZNXARI::znx_rotate(p, tmp, res.at(res_col, j));
        ZNXARI::znx_sub_ba_inplace(res.at_mut(res_col, j), tmp);
    }
}

pub fn bench_vec_znx_mul_xp_minus_one<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxMulXpMinusOne + ModuleNew<B>,
{
    let group_name: String = format!("vec_znx_mul_xp_minus_one::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxMulXpMinusOne + ModuleNew<B>,
    {
        let n: usize = 1 << params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);
        let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);

        // Fill a with random i64
        a.fill_uniform(50, &mut source);
        res.fill_uniform(50, &mut source);

        move || {
            for i in 0..cols {
                module.vec_znx_mul_xp_minus_one(-7, &mut res, i, &a, i);
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

pub fn bench_vec_znx_mul_xp_minus_one_inplace<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxMulXpMinusOneInplace<B> + VecZnxMulXpMinusOneInplaceTmpBytes + ModuleNew<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vec_znx_mul_xp_minus_one_inplace::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxMulXpMinusOneInplace<B> + ModuleNew<B> + VecZnxMulXpMinusOneInplaceTmpBytes,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let n: usize = 1 << params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);

        let mut scratch = ScratchOwned::alloc(module.vec_znx_mul_xp_minus_one_inplace_tmp_bytes());

        // Fill a with random i64
        res.fill_uniform(50, &mut source);

        move || {
            for i in 0..cols {
                module.vec_znx_mul_xp_minus_one_inplace(-7, &mut res, i, scratch.borrow());
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
