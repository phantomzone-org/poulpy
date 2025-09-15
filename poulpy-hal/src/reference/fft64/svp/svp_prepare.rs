use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};

use crate::{
    api::{ModuleNew, SvpPPolAlloc, SvpPrepare},
    layouts::{Backend, FillUniform, Module, ScalarZnx, ScalarZnxToRef, SvpPPol, SvpPPolToMut, ZnxView, ZnxViewMut},
    reference::fft64::reim::{ReimDFTExecute, ReimFFTTable, ReimFromZnx},
    source::Source,
};

pub fn svp_prepare<R, A, BE>(table: &ReimFFTTable<f64>, res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarPrep = f64> + ReimDFTExecute<ReimFFTTable<f64>, f64> + ReimFromZnx,
    R: SvpPPolToMut<BE>,
    A: ScalarZnxToRef,
{
    let mut res: SvpPPol<&mut [u8], BE> = res.to_mut();
    let a: ScalarZnx<&[u8]> = a.to_ref();
    BE::reim_from_znx(res.at_mut(res_col, 0), a.at(a_col, 0));
    BE::reim_dft_execute(table, res.at_mut(res_col, 0));
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
