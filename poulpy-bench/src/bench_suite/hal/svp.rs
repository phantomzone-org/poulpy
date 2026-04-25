use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};
use rand::Rng;

use poulpy_hal::{
    api::{ModuleNew, SvpApplyDft, SvpApplyDftToDft, SvpApplyDftToDftInplace, SvpPPolAlloc, SvpPrepare, VecZnxDftAlloc},
    layouts::{
        Backend, DataViewMut, FillUniform, Module, ScalarZnx, ScalarZnxToBackendRef, SvpPPol, SvpPPolToBackendMut,
        SvpPPolToBackendRef, VecZnx, VecZnxDft, VecZnxDftToBackendMut, VecZnxToBackendRef,
    },
    source::Source,
};

pub fn bench_svp_prepare<B>(params: &crate::params::SvpPrepareParams, c: &mut Criterion, label: &str)
where
    Module<B>: SvpPrepare<B> + SvpPPolAlloc<B> + ModuleNew<B>,
    B: Backend,
    B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
{
    let group_name: String = format!("svp_prepare::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(log_n: usize) -> impl FnMut()
    where
        Module<B>: SvpPrepare<B> + SvpPPolAlloc<B> + ModuleNew<B>,
        B: Backend,
        B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
    {
        let module: Module<B> = Module::<B>::new(1 << log_n);

        let cols: usize = 2;

        let mut svp: SvpPPol<B::OwnedBuf, B> = module.svp_ppol_alloc(cols);
        let mut a: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(module.n(), cols);
        let mut source = Source::new([0u8; 32]);
        a.fill_uniform(50, &mut source);
        let a_backend = ScalarZnx::from_data(B::from_host_bytes(a.to_ref().data), a.n, a.cols);

        move || {
            module.svp_prepare(
                &mut svp.to_backend_mut(),
                0,
                &<ScalarZnx<B::OwnedBuf> as ScalarZnxToBackendRef<B>>::to_backend_ref(&a_backend),
                0,
            );
            black_box(());
        }
    }

    for &log_n in &params.log_n {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}", 1 << log_n));
        let mut runner = runner::<B>(log_n);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_svp_apply_dft<B>(params: &crate::params::HalSweepParams, c: &mut Criterion, label: &str)
where
    Module<B>: SvpApplyDft<B> + SvpPPolAlloc<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
    B: Backend<OwnedBuf = Vec<u8>>,
    B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
{
    let group_name: String = format!("svp_apply_dft::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(sweep: [usize; 3]) -> impl FnMut()
    where
        Module<B>: SvpApplyDft<B> + SvpPPolAlloc<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
        B: Backend<OwnedBuf = Vec<u8>>,
        B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
    {
        let n: usize = 1 << sweep[0];
        let cols: usize = sweep[1];
        let size: usize = sweep[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut svp: SvpPPol<B::OwnedBuf, B> = module.svp_ppol_alloc(cols);
        let mut res: VecZnxDft<B::OwnedBuf, B> = module.vec_znx_dft_alloc(cols, size);
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);

        let mut source = Source::new([0u8; 32]);

        source.fill_bytes(svp.data_mut().as_mut());
        source.fill_bytes(res.data_mut().as_mut());
        source.fill_bytes(a.data_mut().as_mut());

        move || {
            let svp = svp.to_backend_ref();
            let a = <VecZnx<Vec<u8>> as VecZnxToBackendRef<B>>::to_backend_ref(&a);
            let mut res = <VecZnxDft<B::OwnedBuf, B> as VecZnxDftToBackendMut<B>>::to_backend_mut(&mut res);
            for j in 0..cols {
                module.svp_apply_dft(&mut res, j, &svp, j, &a, j);
            }
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << sweep[0], sweep[1], sweep[2]));
        let mut runner = runner::<B>(*sweep);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_svp_apply_dft_to_dft<B>(params: &crate::params::HalSweepParams, c: &mut Criterion, label: &str)
where
    Module<B>: SvpApplyDftToDft<B> + SvpPPolAlloc<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
    B: Backend,
    B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
{
    let group_name: String = format!("svp_apply_dft_to_dft::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(sweep: [usize; 3]) -> impl FnMut()
    where
        Module<B>: SvpApplyDftToDft<B> + SvpPPolAlloc<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
        B: Backend,
        B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
    {
        let n: usize = 1 << sweep[0];
        let cols: usize = sweep[1];
        let size: usize = sweep[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut svp: SvpPPol<B::OwnedBuf, B> = module.svp_ppol_alloc(cols);
        let mut res: VecZnxDft<B::OwnedBuf, B> = module.vec_znx_dft_alloc(cols, size);
        let mut a: VecZnxDft<B::OwnedBuf, B> = module.vec_znx_dft_alloc(cols, size);

        let mut source = Source::new([0u8; 32]);

        source.fill_bytes(svp.data_mut().as_mut());
        source.fill_bytes(res.data_mut().as_mut());
        source.fill_bytes(a.data_mut().as_mut());

        move || {
            let svp = svp.to_backend_ref();
            for j in 0..cols {
                module.svp_apply_dft_to_dft(&mut res, j, &svp, j, &a, j);
            }
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << sweep[0], sweep[1], sweep[2]));
        let mut runner = runner::<B>(*sweep);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_svp_apply_dft_to_dft_assign<B>(params: &crate::params::HalSweepParams, c: &mut Criterion, label: &str)
where
    Module<B>: SvpApplyDftToDftAssign<B> + SvpPPolAlloc<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
    B: Backend,
    B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
{
    let group_name: String = format!("svp_apply_dft_to_dft_assign::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(sweep: [usize; 3]) -> impl FnMut()
    where
        Module<B>: SvpApplyDftToDftAssign<B> + SvpPPolAlloc<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
        B: Backend,
        B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
    {
        let n: usize = 1 << sweep[0];
        let cols: usize = sweep[1];
        let size: usize = sweep[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut svp: SvpPPol<B::OwnedBuf, B> = module.svp_ppol_alloc(cols);
        let mut res: VecZnxDft<B::OwnedBuf, B> = module.vec_znx_dft_alloc(cols, size);

        let mut source = Source::new([0u8; 32]);

        source.fill_bytes(svp.data_mut().as_mut());
        source.fill_bytes(res.data_mut().as_mut());

        move || {
            let svp = svp.to_backend_ref();
            for j in 0..cols {
                module.svp_apply_dft_to_dft_assign(&mut res, j, &svp, j);
            }
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << sweep[0], sweep[1], sweep[2]));
        let mut runner = runner::<B>(*sweep);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}
