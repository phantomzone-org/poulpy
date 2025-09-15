use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};

use crate::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes},
    layouts::{Backend, FillUniform, Module, ScratchOwned, VecZnx, VecZnxToMut, VecZnxToRef, ZnxInfos, ZnxView, ZnxViewMut},
    reference::znx::{
        ZnxNormalizeFinalStep, ZnxNormalizeFinalStepInplace, ZnxNormalizeFirstStep, ZnxNormalizeFirstStepCarryOnly,
        ZnxNormalizeFirstStepInplace, ZnxNormalizeMiddleStep, ZnxNormalizeMiddleStepCarryOnly, ZnxNormalizeMiddleStepInplace,
        ZnxZero,
    },
    source::Source,
};

pub fn vec_znx_normalize_tmp_bytes(n: usize) -> usize {
    n * size_of::<i64>()
}

pub fn vec_znx_normalize<R, A, ZNXARI>(basek: usize, res: &mut R, res_col: usize, a: &A, a_col: usize, carry: &mut [i64])
where
    R: VecZnxToMut,
    A: VecZnxToRef,
    ZNXARI: ZnxZero
        + ZnxNormalizeFirstStepCarryOnly
        + ZnxNormalizeMiddleStepCarryOnly
        + ZnxNormalizeMiddleStep
        + ZnxNormalizeFinalStep
        + ZnxNormalizeFirstStep,
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
                ZNXARI::znx_normalize_first_step_carry_only(basek, 0, a.at(a_col, j), carry);
            } else {
                ZNXARI::znx_normalize_middle_step_carry_only(basek, 0, a.at(a_col, j), carry);
            }
        }

        for j in (1..res_size).rev() {
            ZNXARI::znx_normalize_middle_step(basek, 0, res.at_mut(res_col, j), a.at(a_col, j), carry);
        }

        ZNXARI::znx_normalize_final_step(basek, 0, res.at_mut(res_col, 0), a.at(a_col, 0), carry);
    } else {
        for j in (0..a_size).rev() {
            if j == a_size - 1 {
                ZNXARI::znx_normalize_first_step(basek, 0, res.at_mut(res_col, j), a.at(a_col, j), carry);
            } else if j == 0 {
                ZNXARI::znx_normalize_final_step(basek, 0, res.at_mut(res_col, j), a.at(a_col, j), carry);
            } else {
                ZNXARI::znx_normalize_middle_step(basek, 0, res.at_mut(res_col, j), a.at(a_col, j), carry);
            }
        }

        for j in a_size..res_size {
            ZNXARI::znx_zero(res.at_mut(res_col, j));
        }
    }
}

pub fn vec_znx_normalize_inplace<R: VecZnxToMut, ZNXARI>(basek: usize, res: &mut R, res_col: usize, carry: &mut [i64])
where
    ZNXARI: ZnxNormalizeFirstStepInplace + ZnxNormalizeMiddleStepInplace + ZnxNormalizeFinalStepInplace,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();

    #[cfg(debug_assertions)]
    {
        assert!(carry.len() >= res.n());
    }

    let res_size: usize = res.size();

    for j in (0..res_size).rev() {
        if j == res_size - 1 {
            ZNXARI::znx_normalize_first_step_inplace(basek, 0, res.at_mut(res_col, j), carry);
        } else if j == 0 {
            ZNXARI::znx_normalize_final_step_inplace(basek, 0, res.at_mut(res_col, j), carry);
        } else {
            ZNXARI::znx_normalize_middle_step_inplace(basek, 0, res.at_mut(res_col, j), carry);
        }
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
        let n: usize = 1 << params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let basek: usize = 50;

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);
        let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);

        // Fill a with random i64
        a.fill_uniform(50, &mut source);
        res.fill_uniform(50, &mut source);

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
        let n: usize = 1 << params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let basek: usize = 50;

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);

        // Fill a with random i64
        a.fill_uniform(50, &mut source);

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
