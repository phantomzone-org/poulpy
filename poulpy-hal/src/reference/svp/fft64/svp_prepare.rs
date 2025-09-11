use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};

use crate::{
    api::{ModuleNew, SvpPPolAlloc, SvpPrepare},
    layouts::{Backend, FillUniform, Module, ScalarZnx, ScalarZnxToRef, SvpPPol, SvpPPolToMut, ZnxView, ZnxViewMut},
    reference::{
        reim::{ReimConv, ReimConvRef, ReimDFTExecute, ReimFFTRef, ReimFFTTable},
        vec_znx_dft::fft64::assert_approx_eq_slice,
    },
    source::Source,
};

pub fn svp_prepare<R, A, BE, CONV, FFT>(table: &ReimFFTTable<f64>, res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarPrep = f64>,
    R: SvpPPolToMut<BE>,
    A: ScalarZnxToRef,
    CONV: ReimConv,
    FFT: ReimDFTExecute<ReimFFTTable<f64>, f64>,
{
    let mut res: SvpPPol<&mut [u8], BE> = res.to_mut();
    let a: ScalarZnx<&[u8]> = a.to_ref();
    CONV::reim_from_znx_i64(res.at_mut(res_col, 0), a.at(a_col, 0));
    FFT::reim_dft_execute(table, res.at_mut(res_col, 0));
}

pub fn test_svp_prepare<B>(module: &Module<B>)
where
    Module<B>: SvpPrepare<B> + SvpPPolAlloc<B>,
    B: Backend<ScalarPrep = f64>,
{
    let cols: usize = 2;

    let mut svp_0: SvpPPol<Vec<u8>, B> = module.svp_ppol_alloc(cols);
    let mut svp_1: SvpPPol<Vec<u8>, B> = module.svp_ppol_alloc(cols);

    let mut a: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(module.n(), cols);

    let mut source = Source::new([0u8; 32]);

    a.raw_mut().iter_mut().for_each(|x| {
        *x = source.next_i32() as i64;
    });

    let table: ReimFFTTable<f64> = ReimFFTTable::new(module.n() >> 1);

    for i in 0..cols {
        module.svp_prepare(&mut svp_0, i, &a, i);
        svp_prepare::<_, _, _, ReimConvRef, ReimFFTRef>(&table, &mut svp_1, i, &a, i);
    }

    assert_approx_eq_slice(svp_0.raw(), svp_1.raw(), 1e-10);
}

pub fn bench_svp_prepare<B>(c: &mut Criterion, label: &str)
where
    Module<B>: SvpPrepare<B> + SvpPPolAlloc<B> + ModuleNew<B>,
    B: Backend<ScalarPrep = f64>,
{
    let group_name: String = format!("svp_prepare::{}", label);

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(log_n: usize) -> impl FnMut()
    where
        Module<B>: SvpPrepare<B> + SvpPPolAlloc<B> + ModuleNew<B>,
        B: Backend<ScalarPrep = f64>,
    {
        let module: Module<B> = Module::<B>::new(1 << log_n);

        let cols: usize = 2;

        let mut svp: SvpPPol<Vec<u8>, B> = module.svp_ppol_alloc(cols);
        let mut a: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(module.n(), cols);
        let mut source = Source::new([0u8; 32]);
        a.fill_uniform(&mut source);

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
