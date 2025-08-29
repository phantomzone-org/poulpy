use std::hint::black_box;

use crate::{
    api::{
        ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAutomorphism, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes,
    },
    layouts::{Backend, FillUniform, Module, ScratchOwned, VecZnx},
    oep::{
        ModuleNewImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, VecZnxAutomorphismImpl, VecZnxNormalizeInplaceImpl,
        VecZnxNormalizeTmpBytesImpl,
    },
    source::Source,
};
use criterion::{BenchmarkId, Criterion};

pub fn bench_vec_znx_normalize_inplace<B: Backend>(c: &mut Criterion, label: &str)
where
    B: ModuleNewImpl<B>
        + VecZnxNormalizeInplaceImpl<B>
        + ScratchOwnedAllocImpl<B>
        + ScratchOwnedBorrowImpl<B>
        + VecZnxNormalizeTmpBytesImpl<B>,
{
    let group_name: String = format!("vec_znx_normalize::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxNormalizeInplace<B> + ModuleNew<B> + VecZnxNormalizeTmpBytes,
        B: ScratchOwnedAllocImpl<B> + ScratchOwnedBorrowImpl<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << params[0]);

        let basek: usize = 12;

        let cols: usize = params[1];
        let size: usize = params[2];

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);
        // let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);

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

    for params in [[10, 2, 7], [11, 2, 7], [12, 2, 7], [13, 2, 7], [14, 2, 7]] {
        let id = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << params[0], params[1], params[2],));
        let mut runner = runner::<B>(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vec_znx_automorphism<B: Backend>(c: &mut Criterion, label: &str)
where
    B: ModuleNewImpl<B> + VecZnxAutomorphismImpl<B>,
{
    let group_name: String = format!("vec_znx_automorphism::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxAutomorphism + ModuleNew<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << params[0]);

        let cols: usize = params[1];
        let size: usize = params[2];

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);
        let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);

        // Fill a with random i64
        a.fill_uniform(&mut source);

        move || {
            for i in 0..cols {
                module.vec_znx_automorphism(-5, &mut b, i, &a, i);
            }
            black_box(());
        }
    }

    for params in [[10, 2, 7], [11, 2, 7], [12, 2, 7], [13, 2, 7], [14, 2, 7]] {
        let id = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << params[0], params[1], params[2],));
        let mut runner = runner::<B>(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}
