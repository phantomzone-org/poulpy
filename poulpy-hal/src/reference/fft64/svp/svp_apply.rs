use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};
use rand::RngCore;

use crate::{
    api::{ModuleNew, SvpApplyDft, SvpApplyDftToDft, SvpApplyDftToDftAdd, SvpApplyDftToDftInplace, SvpPPolAlloc, VecZnxDftAlloc},
    layouts::{
        Backend, DataViewMut, Module, SvpPPol, SvpPPolToRef, VecZnx, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef, VecZnxToRef,
        ZnxInfos, ZnxView, ZnxViewMut,
    },
    reference::fft64::reim::{ReimAddMul, ReimDFTExecute, ReimFFTTable, ReimFromZnx, ReimMul, ReimMulInplace, ReimZero},
    source::Source,
};

pub fn svp_apply_dft<R, A, B, BE>(
    table: &ReimFFTTable<f64>,
    res: &mut R,
    res_col: usize,
    a: &A,
    a_col: usize,
    b: &B,
    b_col: usize,
) where
    BE: Backend<ScalarPrep = f64> + ReimDFTExecute<ReimFFTTable<f64>, f64> + ReimZero + ReimFromZnx + ReimMulInplace,
    R: VecZnxDftToMut<BE>,
    A: SvpPPolToRef<BE>,
    B: VecZnxToRef,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: SvpPPol<&[u8], BE> = a.to_ref();
    let b: VecZnx<&[u8]> = b.to_ref();

    let res_size: usize = res.size();
    let b_size: usize = b.size();
    let min_size: usize = res_size.min(b_size);

    let ppol: &[f64] = a.at(a_col, 0);
    for j in 0..min_size {
        let out: &mut [f64] = res.at_mut(res_col, j);
        BE::reim_from_znx(out, b.at(b_col, j));
        BE::reim_dft_execute(table, out);
        BE::reim_mul_inplace(out, ppol);
    }

    for j in min_size..res_size {
        BE::reim_zero(res.at_mut(res_col, j));
    }
}

pub fn svp_apply_dft_to_dft<R, A, B, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend<ScalarPrep = f64> + ReimMul + ReimZero,
    R: VecZnxDftToMut<BE>,
    A: SvpPPolToRef<BE>,
    B: VecZnxDftToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: SvpPPol<&[u8], BE> = a.to_ref();
    let b: VecZnxDft<&[u8], BE> = b.to_ref();

    let res_size: usize = res.size();
    let b_size: usize = b.size();
    let min_size: usize = res_size.min(b_size);

    let ppol: &[f64] = a.at(a_col, 0);
    for j in 0..min_size {
        BE::reim_mul(res.at_mut(res_col, j), ppol, b.at(b_col, j));
    }

    for j in min_size..res_size {
        BE::reim_zero(res.at_mut(res_col, j));
    }
}

pub fn svp_apply_dft_to_dft_add<R, A, B, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend<ScalarPrep = f64> + ReimAddMul + ReimZero,
    R: VecZnxDftToMut<BE>,
    A: SvpPPolToRef<BE>,
    B: VecZnxDftToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: SvpPPol<&[u8], BE> = a.to_ref();
    let b: VecZnxDft<&[u8], BE> = b.to_ref();

    let res_size: usize = res.size();
    let b_size: usize = b.size();
    let min_size: usize = res_size.min(b_size);

    let ppol: &[f64] = a.at(a_col, 0);
    for j in 0..min_size {
        BE::reim_addmul(res.at_mut(res_col, j), ppol, b.at(b_col, j));
    }

    for j in min_size..res_size {
        BE::reim_zero(res.at_mut(res_col, j));
    }
}

pub fn svp_apply_dft_to_dft_inplace<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarPrep = f64> + ReimMulInplace,
    R: VecZnxDftToMut<BE>,
    A: SvpPPolToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: SvpPPol<&[u8], BE> = a.to_ref();

    let ppol: &[f64] = a.at(a_col, 0);
    for j in 0..res.size() {
        BE::reim_mul_inplace(res.at_mut(res_col, j), ppol);
    }
}

pub fn bench_svp_apply_dft<B>(c: &mut Criterion, label: &str)
where
    Module<B>: SvpApplyDft<B> + SvpPPolAlloc<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
    B: Backend<ScalarPrep = f64>,
{
    let group_name: String = format!("svp_apply_dft::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: SvpApplyDft<B> + SvpPPolAlloc<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
        B: Backend<ScalarPrep = f64>,
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
    B: Backend<ScalarPrep = f64>,
{
    let group_name: String = format!("svp_apply_dft_to_dft::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: SvpApplyDftToDft<B> + SvpPPolAlloc<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
        B: Backend<ScalarPrep = f64>,
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
    B: Backend<ScalarPrep = f64>,
{
    let group_name: String = format!("svp_apply_dft_to_dft_add::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: SvpApplyDftToDftAdd<B> + SvpPPolAlloc<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
        B: Backend<ScalarPrep = f64>,
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
    B: Backend<ScalarPrep = f64>,
{
    let group_name: String = format!("svp_apply_dft_to_dft_inplace::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: SvpApplyDftToDftInplace<B> + SvpPPolAlloc<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
        B: Backend<ScalarPrep = f64>,
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
