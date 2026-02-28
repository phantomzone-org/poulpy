use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};
use rand::Rng;

use crate::{
    api::{
        ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxBigAlloc, VecZnxDftAdd, VecZnxDftAddInplace, VecZnxDftAlloc,
        VecZnxDftApply, VecZnxDftSub, VecZnxDftSubInplace, VecZnxDftSubNegateInplace, VecZnxIdftApply, VecZnxIdftApplyTmpA,
        VecZnxIdftApplyTmpBytes,
    },
    layouts::{Backend, DataViewMut, Module, ScratchOwned, VecZnx, VecZnxBig, VecZnxDft},
    source::Source,
};

pub fn bench_vec_znx_dft_add<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxDftAdd<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
{
    let group_name: String = format!("vec_znx_dft_add::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxDftAdd<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
    {
        let n: usize = params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(cols, size);
        let mut b: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(cols, size);
        let mut c: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(cols, size);

        source.fill_bytes(a.data_mut());
        source.fill_bytes(b.data_mut());
        source.fill_bytes(c.data_mut());

        move || {
            for i in 0..cols {
                module.vec_znx_dft_add(&mut c, i, &a, i, &b, i);
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

pub fn bench_vec_znx_dft_add_inplace<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxDftAddInplace<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
{
    let group_name: String = format!("vec_znx_dft_add_inplace::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxDftAddInplace<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
    {
        let n: usize = params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(cols, size);
        let mut c: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(cols, size);

        // Fill a with random i64
        source.fill_bytes(a.data_mut());
        source.fill_bytes(c.data_mut());

        move || {
            for i in 0..cols {
                module.vec_znx_dft_add_inplace(&mut c, i, &a, i);
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

pub fn bench_vec_znx_dft_apply<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxDftApply<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
{
    let group_name: String = format!("vec_znx_dft_apply::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxDftApply<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
    {
        let n: usize = params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut res: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(cols, size);
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);
        source.fill_bytes(res.data_mut());
        source.fill_bytes(a.data_mut());

        move || {
            for i in 0..cols {
                module.vec_znx_dft_apply(1, 0, &mut res, i, &a, i);
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

pub fn bench_vec_znx_idft_apply<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxIdftApply<B> + ModuleNew<B> + VecZnxIdftApplyTmpBytes + VecZnxDftAlloc<B> + VecZnxBigAlloc<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vec_znx_idft_apply::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxIdftApply<B> + ModuleNew<B> + VecZnxIdftApplyTmpBytes + VecZnxDftAlloc<B> + VecZnxBigAlloc<B>,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let n: usize = params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut res: VecZnxBig<Vec<u8>, B> = module.vec_znx_big_alloc(cols, size);
        let mut a: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(cols, size);
        source.fill_bytes(res.data_mut());
        source.fill_bytes(a.data_mut());

        let mut scratch = ScratchOwned::alloc(module.vec_znx_idft_apply_tmp_bytes());

        move || {
            for i in 0..cols {
                module.vec_znx_idft_apply(&mut res, i, &a, i, scratch.borrow());
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

pub fn bench_vec_znx_idft_apply_tmpa<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxIdftApplyTmpA<B> + ModuleNew<B> + VecZnxDftAlloc<B> + VecZnxBigAlloc<B>,
{
    let group_name: String = format!("vec_znx_idft_apply_tmpa::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxIdftApplyTmpA<B> + ModuleNew<B> + VecZnxDftAlloc<B> + VecZnxBigAlloc<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << params[0]);

        let cols: usize = params[1];
        let size: usize = params[2];

        let mut source: Source = Source::new([0u8; 32]);

        let mut res: VecZnxBig<Vec<u8>, B> = module.vec_znx_big_alloc(cols, size);
        let mut a: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(cols, size);
        source.fill_bytes(res.data_mut());
        source.fill_bytes(a.data_mut());

        move || {
            for i in 0..cols {
                module.vec_znx_idft_apply_tmpa(&mut res, i, &mut a, i);
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

pub fn bench_vec_znx_dft_sub<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxDftSub<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
{
    let group_name: String = format!("vec_znx_dft_sub::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxDftSub<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
    {
        let n: usize = params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(cols, size);
        let mut b: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(cols, size);
        let mut c: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(cols, size);

        source.fill_bytes(a.data_mut());
        source.fill_bytes(b.data_mut());
        source.fill_bytes(c.data_mut());

        move || {
            for i in 0..cols {
                module.vec_znx_dft_sub(&mut c, i, &a, i, &b, i);
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

pub fn bench_vec_znx_dft_sub_inplace<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxDftSubInplace<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
{
    let group_name: String = format!("vec_znx_dft_sub_inplace::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxDftSubInplace<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
    {
        let n: usize = params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(cols, size);
        let mut c: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(cols, size);

        // Fill a with random i64
        source.fill_bytes(a.data_mut());
        source.fill_bytes(c.data_mut());

        move || {
            for i in 0..cols {
                module.vec_znx_dft_sub_inplace(&mut c, i, &a, i);
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

pub fn bench_vec_znx_dft_sub_negate_inplace<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxDftSubNegateInplace<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
{
    let group_name: String = format!("vec_znx_dft_sub_negate_inplace::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxDftSubNegateInplace<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
    {
        let n: usize = params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(cols, size);
        let mut c: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(cols, size);

        // Fill a with random i64
        source.fill_bytes(a.data_mut());
        source.fill_bytes(c.data_mut());

        move || {
            for i in 0..cols {
                module.vec_znx_dft_sub_negate_inplace(&mut c, i, &a, i);
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
