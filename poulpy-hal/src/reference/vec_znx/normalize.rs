use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};

use crate::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes},
    layouts::{Backend, FillUniform, Module, ScratchOwned, VecZnx, VecZnxToMut, VecZnxToRef, ZnxInfos, ZnxView, ZnxViewMut},
    reference::znx::{ZnxArithmetic, ZnxArithmeticRef, ZnxNormalize, ZnxNormalizeRef},
    source::Source,
};

pub fn vec_znx_normalize_tmp_bytes(n: usize) -> usize {
    n * size_of::<i64>()
}

pub fn vec_znx_normalize<R, A, ZNXARI, ZNXNORM>(basek: usize, res: &mut R, res_col: usize, a: &A, a_col: usize, carry: &mut [i64])
where
    R: VecZnxToMut,
    A: VecZnxToRef,
    ZNXARI: ZnxArithmetic,
    ZNXNORM: ZnxNormalize,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();
    let a: VecZnx<&[u8]> = a.to_ref();

    #[cfg(debug_assertions)]
    {
        assert!(carry.len() >= res.n());
    }

    let res_size: usize = res.size();
    let a_size = a.size();

    if a_size > res_size {
        for j in (res_size..a_size).rev() {
            if j == a_size - 1 {
                ZNXNORM::znx_normalize_first_step_carry_only(basek, 0, a.at(a_col, j), carry);
            } else {
                ZNXNORM::znx_normalize_middle_step_carry_only(basek, 0, a.at(a_col, j), carry);
            }
        }

        for j in (1..res_size).rev() {
            ZNXNORM::znx_normalize_middle_step(basek, 0, res.at_mut(res_col, j), a.at(a_col, j), carry);
        }

        ZNXNORM::znx_normalize_final_step(basek, 0, res.at_mut(res_col, 0), a.at(a_col, 0), carry);
    } else {
        for j in (0..a_size).rev() {
            if j == a_size - 1 {
                ZNXNORM::znx_normalize_first_step(basek, 0, res.at_mut(res_col, j), a.at(a_col, j), carry);
            } else if j == 0 {
                ZNXNORM::znx_normalize_final_step(basek, 0, res.at_mut(res_col, j), a.at(a_col, j), carry);
            } else {
                ZNXNORM::znx_normalize_middle_step(basek, 0, res.at_mut(res_col, j), a.at(a_col, j), carry);
            }
        }

        for j in a_size..res_size {
            ZNXARI::znx_zero(res.at_mut(res_col, j));
        }
    }
}

pub fn vec_znx_normalize_inplace<R: VecZnxToMut, ZNXNORM: ZnxNormalize>(
    basek: usize,
    res: &mut R,
    res_col: usize,
    carry: &mut [i64],
) {
    let mut res: VecZnx<&mut [u8]> = res.to_mut();

    #[cfg(debug_assertions)]
    {
        assert!(carry.len() >= res.n());
    }

    let res_size: usize = res.size();

    for j in (0..res_size).rev() {
        if j == res_size - 1 {
            ZNXNORM::znx_normalize_first_step_inplace(basek, 0, res.at_mut(res_col, j), carry);
        } else if j == 0 {
            ZNXNORM::znx_normalize_final_step_inplace(basek, 0, res.at_mut(res_col, j), carry);
        } else {
            ZNXNORM::znx_normalize_middle_step_inplace(basek, 0, res.at_mut(res_col, j), carry);
        }
    }
}

pub fn test_vec_znx_normalize<B: Backend>(module: &Module<B>)
where
    Module<B>: VecZnxNormalize<B> + VecZnxNormalizeTmpBytes,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;
    let basek: usize = 12;

    let mut carry: Vec<i64> = vec![0i64; module.n()];

    let mut scratch = ScratchOwned::alloc(module.vec_znx_normalize_tmp_bytes());

    for a_size in [1, 2, 6, 11] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, a_size);
        a.raw_mut()
            .iter_mut()
            .for_each(|x| *x = source.next_i32() as i64);

        for res_size in [1, 2, 6, 11] {
            let mut res_0: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, res_size);
            let mut res_1: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, res_size);

            // Set d to garbage
            res_0.fill_uniform(&mut source);
            res_1.fill_uniform(&mut source);

            // Reference
            for i in 0..cols {
                vec_znx_normalize::<_, _, ZnxArithmeticRef, ZnxNormalizeRef>(basek, &mut res_0, i, &a, i, &mut carry);
                module.vec_znx_normalize(basek, &mut res_1, i, &a, i, scratch.borrow());
            }

            assert_eq!(res_0.raw(), res_1.raw());
        }
    }
}

pub fn test_vec_znx_normalize_inplace<B: Backend>(module: &Module<B>)
where
    Module<B>: VecZnxNormalizeInplace<B> + VecZnxNormalizeTmpBytes,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;
    let basek: usize = 12;

    let mut carry: Vec<i64> = vec![0i64; module.n()];

    let mut scratch = ScratchOwned::alloc(module.vec_znx_normalize_tmp_bytes());

    for res_size in [1, 2, 6, 11] {
        let mut res_0: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, res_size);
        let mut res_1: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, res_size);

        res_0
            .raw_mut()
            .iter_mut()
            .for_each(|x| *x = source.next_i32() as i64);
        res_1.raw_mut().copy_from_slice(res_0.raw());

        // Reference
        for i in 0..cols {
            vec_znx_normalize_inplace::<_, ZnxNormalizeRef>(basek, &mut res_0, i, &mut carry);
            module.vec_znx_normalize_inplace(basek, &mut res_1, i, scratch.borrow());
        }

        assert_eq!(res_0.raw(), res_1.raw());
    }
}

pub fn bench_vec_znx_normalize<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxNormalize<B> + ModuleNew<B> + VecZnxNormalizeTmpBytes,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vec_znx_normalize::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxNormalize<B> + ModuleNew<B> + VecZnxNormalizeTmpBytes,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << params[0]);

        let cols: usize = params[1];
        let size: usize = params[2];

        let basek = 50;

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);
        let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);

        // Fill a with random i64
        a.fill_uniform(&mut source);
        res.fill_uniform(&mut source);

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.vec_znx_normalize_tmp_bytes());

        move || {
            for i in 0..cols {
                module.vec_znx_normalize(basek, &mut res, i, &a, i, scratch.borrow());
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

pub fn bench_vec_znx_normalize_inplace<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxNormalizeInplace<B> + ModuleNew<B> + VecZnxNormalizeTmpBytes,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vec_znx_normalize_inplace::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxNormalizeInplace<B> + ModuleNew<B> + VecZnxNormalizeTmpBytes,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << params[0]);

        let cols: usize = params[1];
        let size: usize = params[2];

        let basek = 50;

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);

        // Fill a with random i64
        a.fill_uniform(&mut source);

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.vec_znx_normalize_tmp_bytes());

        move || {
            for i in 0..cols {
                module.vec_znx_normalize_inplace(basek, &mut a, i, scratch.borrow());
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
