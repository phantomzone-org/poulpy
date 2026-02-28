use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};
use rand::Rng;

use crate::{
    api::{
        ModuleNew, SvpApplyDft, SvpApplyDftToDft, SvpApplyDftToDftAdd, SvpApplyDftToDftInplace, SvpPPolAlloc, SvpPrepare,
        VecZnxDftAlloc,
    },
    layouts::{Backend, DataViewMut, FillUniform, Module, ScalarZnx, SvpPPol, VecZnx, VecZnxDft},
    source::Source,
};

pub fn bench_svp_prepare<B>(c: &mut Criterion, label: &str)
where
    Module<B>: SvpPrepare<B> + SvpPPolAlloc<B> + ModuleNew<B>,
    B: Backend,
{
    let group_name: String = format!("svp_prepare::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(log_n: usize) -> impl FnMut()
    where
        Module<B>: SvpPrepare<B> + SvpPPolAlloc<B> + ModuleNew<B>,
        B: Backend,
    {
        let module: Module<B> = Module::<B>::new(1 << log_n);

        let cols: usize = 2;

        let mut svp: SvpPPol<Vec<u8>, B> = module.svp_ppol_alloc(cols);
        let mut a: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(module.n(), cols);
        let mut source = Source::new([0u8; 32]);
        a.fill_uniform(50, &mut source);

        move || {
            module.svp_prepare(&mut svp, 0, &a, 0);
            black_box(());
        }
    }

    for log_n in [10, 11, 12, 13, 14] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}", 1 << log_n));
        let mut runner = runner::<B>(log_n);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_svp_apply_dft<B>(c: &mut Criterion, label: &str)
where
    Module<B>: SvpApplyDft<B> + SvpPPolAlloc<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
    B: Backend,
{
    let group_name: String = format!("svp_apply_dft::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: SvpApplyDft<B> + SvpPPolAlloc<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
        B: Backend,
    {
        let n: usize = 1 << params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut svp: SvpPPol<Vec<u8>, B> = module.svp_ppol_alloc(cols);
        let mut res: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(cols, size);
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);

        let mut source = Source::new([0u8; 32]);

        source.fill_bytes(svp.data_mut());
        source.fill_bytes(res.data_mut());
        source.fill_bytes(a.data_mut());

        move || {
            for j in 0..cols {
                module.svp_apply_dft(&mut res, j, &svp, j, &a, j);
            }
            black_box(());
        }
    }

    for params in [[10, 2, 2], [11, 2, 4], [12, 2, 7], [13, 2, 15], [14, 2, 31]] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << params[0], params[1], params[2]));
        let mut runner = runner::<B>(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_svp_apply_dft_to_dft<B>(c: &mut Criterion, label: &str)
where
    Module<B>: SvpApplyDftToDft<B> + SvpPPolAlloc<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
    B: Backend,
{
    let group_name: String = format!("svp_apply_dft_to_dft::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: SvpApplyDftToDft<B> + SvpPPolAlloc<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
        B: Backend,
    {
        let n: usize = 1 << params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut svp: SvpPPol<Vec<u8>, B> = module.svp_ppol_alloc(cols);
        let mut res: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(cols, size);
        let mut a: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(cols, size);

        let mut source = Source::new([0u8; 32]);

        source.fill_bytes(svp.data_mut());
        source.fill_bytes(res.data_mut());
        source.fill_bytes(a.data_mut());

        move || {
            for j in 0..cols {
                module.svp_apply_dft_to_dft(&mut res, j, &svp, j, &a, j);
            }
            black_box(());
        }
    }

    for params in [[10, 2, 2], [11, 2, 4], [12, 2, 7], [13, 2, 15], [14, 2, 31]] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << params[0], params[1], params[2]));
        let mut runner = runner::<B>(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_svp_apply_dft_to_dft_add<B>(c: &mut Criterion, label: &str)
where
    Module<B>: SvpApplyDftToDftAdd<B> + SvpPPolAlloc<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
    B: Backend,
{
    let group_name: String = format!("svp_apply_dft_to_dft_add::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: SvpApplyDftToDftAdd<B> + SvpPPolAlloc<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
        B: Backend,
    {
        let n: usize = 1 << params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut svp: SvpPPol<Vec<u8>, B> = module.svp_ppol_alloc(cols);
        let mut res: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(cols, size);
        let mut a: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(cols, size);

        let mut source = Source::new([0u8; 32]);

        source.fill_bytes(svp.data_mut());
        source.fill_bytes(res.data_mut());
        source.fill_bytes(a.data_mut());

        move || {
            for j in 0..cols {
                module.svp_apply_dft_to_dft_add(&mut res, j, &svp, j, &a, j);
            }
            black_box(());
        }
    }

    for params in [[10, 2, 2], [11, 2, 4], [12, 2, 7], [13, 2, 15], [14, 2, 31]] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << params[0], params[1], params[2]));
        let mut runner = runner::<B>(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_svp_apply_dft_to_dft_inplace<B>(c: &mut Criterion, label: &str)
where
    Module<B>: SvpApplyDftToDftInplace<B> + SvpPPolAlloc<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
    B: Backend,
{
    let group_name: String = format!("svp_apply_dft_to_dft_inplace::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: SvpApplyDftToDftInplace<B> + SvpPPolAlloc<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
        B: Backend,
    {
        let n: usize = 1 << params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut svp: SvpPPol<Vec<u8>, B> = module.svp_ppol_alloc(cols);
        let mut res: VecZnxDft<Vec<u8>, B> = module.vec_znx_dft_alloc(cols, size);

        let mut source = Source::new([0u8; 32]);

        source.fill_bytes(svp.data_mut());
        source.fill_bytes(res.data_mut());

        move || {
            for j in 0..cols {
                module.svp_apply_dft_to_dft_inplace(&mut res, j, &svp, j);
            }
            black_box(());
        }
    }

    for params in [[10, 2, 2], [11, 2, 4], [12, 2, 7], [13, 2, 15], [14, 2, 31]] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << params[0], params[1], params[2]));
        let mut runner = runner::<B>(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}
