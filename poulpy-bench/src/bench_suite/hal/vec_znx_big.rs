use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};
use rand::Rng;

use poulpy_hal::{
    api::{
        ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxBigAddAssign, VecZnxBigAddInto, VecZnxBigAddSmallAssign,
        VecZnxBigAddSmallIntoBackend, VecZnxBigAlloc, VecZnxBigAutomorphism, VecZnxBigAutomorphismInplace,
        VecZnxBigAutomorphismInplaceTmpBytes, VecZnxBigNegate, VecZnxBigNegateInplace, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxBigSub, VecZnxBigSubInplace, VecZnxBigSubNegateInplace, VecZnxBigSubSmallABackend,
        VecZnxBigSubSmallBBackend,
    },
    layouts::{
        Backend, DataViewMut, Module, ScratchOwned, VecZnx, VecZnxBig, VecZnxBigToBackendMut, VecZnxBigToBackendRef,
        VecZnxToBackendMut, VecZnxToBackendRef,
    },
    source::Source,
};

pub fn bench_vec_znx_big_add_into<B: Backend>(params: &crate::params::HalSweepParams, c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxBigAddInto<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
    B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
{
    let group_name: String = format!("vec_znx_big_add_into::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(sweep: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxBigAddInto<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
        B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
    {
        let n: usize = 1 << sweep[0];
        let cols: usize = sweep[1];
        let size: usize = sweep[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxBig<B::OwnedBuf, B> = module.vec_znx_big_alloc(cols, size);
        let mut b: VecZnxBig<B::OwnedBuf, B> = module.vec_znx_big_alloc(cols, size);
        let mut c: VecZnxBig<B::OwnedBuf, B> = module.vec_znx_big_alloc(cols, size);

        // Fill a with random bytes
        source.fill_bytes(a.data_mut().as_mut());
        source.fill_bytes(b.data_mut().as_mut());
        source.fill_bytes(c.data_mut().as_mut());

        move || {
            let a = <VecZnxBig<B::OwnedBuf, B> as VecZnxBigToBackendRef<B>>::to_backend_ref(&a);
            let b = <VecZnxBig<B::OwnedBuf, B> as VecZnxBigToBackendRef<B>>::to_backend_ref(&b);
            let mut c = <VecZnxBig<B::OwnedBuf, B> as VecZnxBigToBackendMut<B>>::to_backend_mut(&mut c);
            for i in 0..cols {
                module.vec_znx_big_add_into(&mut c, i, &a, i, &b, i);
            }
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << sweep[0], sweep[1], sweep[2],));
        let mut runner = runner::<B>(*sweep);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vec_znx_big_add_assign<B: Backend>(params: &crate::params::HalSweepParams, c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxBigAddAssign<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
    B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
{
    let group_name: String = format!("vec_znx_big_add_assign::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(sweep: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxBigAddAssign<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
        B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
    {
        let n: usize = 1 << sweep[0];
        let cols: usize = sweep[1];
        let size: usize = sweep[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxBig<B::OwnedBuf, B> = module.vec_znx_big_alloc(cols, size);
        let mut c: VecZnxBig<B::OwnedBuf, B> = module.vec_znx_big_alloc(cols, size);

        // Fill a with random i64
        source.fill_bytes(a.data_mut().as_mut());
        source.fill_bytes(c.data_mut().as_mut());

        move || {
            let a = <VecZnxBig<B::OwnedBuf, B> as VecZnxBigToBackendRef<B>>::to_backend_ref(&a);
            let mut c = <VecZnxBig<B::OwnedBuf, B> as VecZnxBigToBackendMut<B>>::to_backend_mut(&mut c);
            for i in 0..cols {
                module.vec_znx_big_add_assign(&mut c, i, &a, i);
            }
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << sweep[0], sweep[1], sweep[2],));
        let mut runner = runner::<B>(*sweep);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vec_znx_big_add_small_into<B: Backend<OwnedBuf = Vec<u8>>>(
    params: &crate::params::HalSweepParams,
    c: &mut Criterion,
    label: &str,
)
where
    Module<B>: VecZnxBigAddSmallIntoBackend<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
    B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
{
    let group_name: String = format!("vec_znx_big_add_small_into_backend::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend<OwnedBuf = Vec<u8>>>(sweep: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxBigAddSmallIntoBackend<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
        B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
    {
        let n: usize = 1 << sweep[0];
        let cols: usize = sweep[1];
        let size: usize = sweep[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxBig<B::OwnedBuf, B> = module.vec_znx_big_alloc(cols, size);
        let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);
        let mut c: VecZnxBig<B::OwnedBuf, B> = module.vec_znx_big_alloc(cols, size);

        // Fill a with random i64
        source.fill_bytes(a.data_mut().as_mut());
        source.fill_bytes(b.data_mut().as_mut());
        source.fill_bytes(c.data_mut().as_mut());

        move || {
            let a = <VecZnxBig<B::OwnedBuf, B> as VecZnxBigToBackendRef<B>>::to_backend_ref(&a);
            let b = <VecZnx<Vec<u8>> as VecZnxToBackendRef<B>>::to_backend_ref(&b);
            let mut c = <VecZnxBig<B::OwnedBuf, B> as VecZnxBigToBackendMut<B>>::to_backend_mut(&mut c);
            for i in 0..cols {
                module.vec_znx_big_add_small_into_backend(&mut c, i, &a, i, &b, i);
            }
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << sweep[0], sweep[1], sweep[2],));
        let mut runner = runner::<B>(*sweep);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vec_znx_big_add_small_assign<B: Backend<OwnedBuf = Vec<u8>>>(
    params: &crate::params::HalSweepParams,
    c: &mut Criterion,
    label: &str,
) where
    Module<B>: VecZnxBigAddSmallAssign<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
    B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
{
    let group_name: String = format!("vec_znx_big_add_small_assign::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend<OwnedBuf = Vec<u8>>>(sweep: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxBigAddSmallAssign<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
        B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
    {
        let n: usize = 1 << sweep[0];
        let cols: usize = sweep[1];
        let size: usize = sweep[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);
        let mut c: VecZnxBig<B::OwnedBuf, B> = module.vec_znx_big_alloc(cols, size);

        // Fill a with random i64
        source.fill_bytes(a.data_mut().as_mut());
        source.fill_bytes(c.data_mut().as_mut());

        move || {
            let a = <VecZnx<Vec<u8>> as VecZnxToBackendRef<B>>::to_backend_ref(&a);
            let mut c = <VecZnxBig<B::OwnedBuf, B> as VecZnxBigToBackendMut<B>>::to_backend_mut(&mut c);
            for i in 0..cols {
                module.vec_znx_big_add_small_assign(&mut c, i, &a, i);
            }
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << sweep[0], sweep[1], sweep[2],));
        let mut runner = runner::<B>(*sweep);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vec_znx_big_automorphism<B: Backend>(params: &crate::params::HalSweepParams, c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxBigAutomorphism<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
    B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
{
    let group_name: String = format!("vec_znx_big_automorphism::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(sweep: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxBigAutomorphism<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
        B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
    {
        let n: usize = 1 << sweep[0];
        let cols: usize = sweep[1];
        let size: usize = sweep[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxBig<B::OwnedBuf, B> = module.vec_znx_big_alloc(cols, size);
        let mut res: VecZnxBig<B::OwnedBuf, B> = module.vec_znx_big_alloc(cols, size);

        // Fill a with random i64
        source.fill_bytes(a.data_mut().as_mut());
        source.fill_bytes(res.data_mut().as_mut());

        move || {
            let a = <VecZnxBig<B::OwnedBuf, B> as VecZnxBigToBackendRef<B>>::to_backend_ref(&a);
            let mut res = <VecZnxBig<B::OwnedBuf, B> as VecZnxBigToBackendMut<B>>::to_backend_mut(&mut res);
            for i in 0..cols {
                module.vec_znx_big_automorphism(-7, &mut res, i, &a, i);
            }
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << sweep[0], sweep[1], sweep[2],));
        let mut runner = runner::<B>(*sweep);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vec_znx_automorphism_assign<B: Backend>(params: &crate::params::HalSweepParams, c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxBigAutomorphismAssign<B> + VecZnxBigAutomorphismAssignTmpBytes + ModuleNew<B> + VecZnxBigAlloc<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
{
    let group_name: String = format!("vec_znx_automorphism_assign::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(sweep: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxBigAutomorphismAssign<B> + ModuleNew<B> + VecZnxBigAutomorphismAssignTmpBytes + VecZnxBigAlloc<B>,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
        B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
    {
        let n: usize = 1 << sweep[0];
        let cols: usize = sweep[1];
        let size: usize = sweep[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut res: VecZnxBig<B::OwnedBuf, B> = module.vec_znx_big_alloc(cols, size);

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.vec_znx_big_automorphism_assign_tmp_bytes());

        // Fill a with random i64
        source.fill_bytes(res.data_mut().as_mut());

        move || {
            let mut res = <VecZnxBig<B::OwnedBuf, B> as VecZnxBigToBackendMut<B>>::to_backend_mut(&mut res);
            for i in 0..cols {
                module.vec_znx_big_automorphism_inplace(-7, &mut res, i, &mut scratch.borrow());
            }
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << sweep[0], sweep[1], sweep[2],));
        let mut runner = runner::<B>(*sweep);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vec_znx_big_negate<B: Backend>(params: &crate::params::HalSweepParams, c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxBigNegate<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
    B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
{
    let group_name: String = format!("vec_znx_big_negate::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(sweep: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxBigNegate<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
        B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
    {
        let module: Module<B> = Module::<B>::new(1 << sweep[0]);

        let cols: usize = sweep[1];
        let size: usize = sweep[2];

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxBig<B::OwnedBuf, B> = module.vec_znx_big_alloc(cols, size);
        let mut b: VecZnxBig<B::OwnedBuf, B> = module.vec_znx_big_alloc(cols, size);

        // Fill a with random i64
        source.fill_bytes(a.data_mut().as_mut());
        source.fill_bytes(b.data_mut().as_mut());

        move || {
            let a = <VecZnxBig<B::OwnedBuf, B> as VecZnxBigToBackendRef<B>>::to_backend_ref(&a);
            let mut b = <VecZnxBig<B::OwnedBuf, B> as VecZnxBigToBackendMut<B>>::to_backend_mut(&mut b);
            for i in 0..cols {
                module.vec_znx_big_negate(&mut b, i, &a, i);
            }
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << sweep[0], sweep[1], sweep[2],));
        let mut runner = runner::<B>(*sweep);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vec_znx_big_negate_assign<B: Backend>(params: &crate::params::HalSweepParams, c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxBigNegateInplace<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
    B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
{
    let group_name: String = format!("vec_znx_negate_big_assign::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(sweep: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxBigNegateInplace<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
        B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
    {
        let module: Module<B> = Module::<B>::new(1 << sweep[0]);

        let cols: usize = sweep[1];
        let size: usize = sweep[2];

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxBig<B::OwnedBuf, B> = module.vec_znx_big_alloc(cols, size);

        // Fill a with random i64
        source.fill_bytes(a.data_mut().as_mut());

        move || {
            let mut a = <VecZnxBig<B::OwnedBuf, B> as VecZnxBigToBackendMut<B>>::to_backend_mut(&mut a);
            for i in 0..cols {
                module.vec_znx_big_negate_assign(&mut a, i);
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

pub fn bench_vec_znx_normalize<B: Backend<OwnedBuf = Vec<u8>>>(
    params: &crate::params::HalSweepParams,
    c: &mut Criterion,
    label: &str,
) where
    Module<B>: VecZnxBigNormalize<B> + ModuleNew<B> + VecZnxBigNormalizeTmpBytes + VecZnxBigAlloc<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
{
    let group_name: String = format!("vec_znx_big_normalize::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend<OwnedBuf = Vec<u8>>>(sweep: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxBigNormalize<B> + ModuleNew<B> + VecZnxBigNormalizeTmpBytes + VecZnxBigAlloc<B>,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
        B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
    {
        let n: usize = 1 << sweep[0];
        let cols: usize = sweep[1];
        let size: usize = sweep[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let base2k: usize = 50;

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxBig<B::OwnedBuf, B> = module.vec_znx_big_alloc(cols, size);
        let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);

        // Fill a with random i64
        source.fill_bytes(a.data_mut().as_mut());
        source.fill_bytes(res.data_mut().as_mut());

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.vec_znx_big_normalize_tmp_bytes());

        move || {
            let a = <VecZnxBig<B::OwnedBuf, B> as VecZnxBigToBackendRef<B>>::to_backend_ref(&a);
            let mut res = <VecZnx<Vec<u8>> as VecZnxToBackendMut<B>>::to_backend_mut(&mut res);
            for i in 0..cols {
                module.vec_znx_big_normalize(&mut res, base2k, 0, i, &a, base2k, i, &mut scratch.borrow());
            }
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << sweep[0], sweep[1], sweep[2],));
        let mut runner = runner::<B>(*sweep);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vec_znx_normalize_add_assign<B: Backend>(params: &crate::params::HalSweepParams, c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxBigNormalize<B> + ModuleNew<B> + VecZnxBigNormalizeTmpBytes + VecZnxBigAlloc<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vec_znx_big_normalize_add_assign::{label}");
    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(sweep: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxBigNormalize<B> + ModuleNew<B> + VecZnxBigNormalizeTmpBytes + VecZnxBigAlloc<B>,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let n: usize = 1 << sweep[0];
        let cols: usize = sweep[1];
        let size: usize = sweep[2];

        let module: Module<B> = Module::<B>::new(n as u64);
        let base2k: usize = 50;
        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxBig<DeviceBuf<B>, B> = module.vec_znx_big_alloc(cols, size);
        let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);

        source.fill_bytes(a.data_mut().as_mut());
        source.fill_bytes(res.data_mut().as_mut());

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.vec_znx_big_normalize_tmp_bytes());

        move || {
            for i in 0..cols {
                module.vec_znx_big_normalize_add_assign(&mut res, base2k, 0, i, &a, base2k, i, scratch.borrow());
            }
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << sweep[0], sweep[1], sweep[2],));
        let mut runner = runner::<B>(*sweep);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vec_znx_normalize_sub_assign<B: Backend>(params: &crate::params::HalSweepParams, c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxBigNormalize<B> + ModuleNew<B> + VecZnxBigNormalizeTmpBytes + VecZnxBigAlloc<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vec_znx_big_normalize_sub_assign::{label}");
    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(sweep: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxBigNormalize<B> + ModuleNew<B> + VecZnxBigNormalizeTmpBytes + VecZnxBigAlloc<B>,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let n: usize = 1 << sweep[0];
        let cols: usize = sweep[1];
        let size: usize = sweep[2];

        let module: Module<B> = Module::<B>::new(n as u64);
        let base2k: usize = 50;
        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxBig<DeviceBuf<B>, B> = module.vec_znx_big_alloc(cols, size);
        let mut res: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);

        source.fill_bytes(a.data_mut().as_mut());
        source.fill_bytes(res.data_mut().as_mut());

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.vec_znx_big_normalize_tmp_bytes());

        move || {
            for i in 0..cols {
                module.vec_znx_big_normalize_sub_assign(&mut res, base2k, 0, i, &a, base2k, i, scratch.borrow());
            }
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << sweep[0], sweep[1], sweep[2],));
        let mut runner = runner::<B>(*sweep);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vec_znx_normalize_add_assign_compare<B: Backend>(
    params: &crate::params::HalSweepParams,
    c: &mut Criterion,
    label: &str,
) where
    Module<B>: VecZnxBigNormalize<B> + ModuleNew<B> + VecZnxBigNormalizeTmpBytes + VecZnxBigAlloc<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vec_znx_big_normalize_add_assign_compare::{label}");
    let mut group = c.benchmark_group(group_name);

    for sweep in &params.sweeps {
        let n: usize = 1 << sweep[0];
        let cols: usize = sweep[1];
        let size: usize = sweep[2];
        let module: Module<B> = Module::<B>::new(n as u64);
        let base2k: usize = 50;
        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxBig<DeviceBuf<B>, B> = module.vec_znx_big_alloc(cols, size);
        let mut res_fused: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);
        let mut res_fallback: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);
        let mut tmp: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), 1, size);

        source.fill_bytes(a.data_mut().as_mut());
        source.fill_bytes(res_fused.data_mut().as_mut());
        res_fallback.data_mut().copy_from_slice(res_fused.data());

        let mut scratch_fused: ScratchOwned<B> = ScratchOwned::alloc(module.vec_znx_big_normalize_tmp_bytes());
        let mut scratch_fallback: ScratchOwned<B> = ScratchOwned::alloc(module.vec_znx_big_normalize_tmp_bytes());

        let id = format!("{}x({}x{})", n, cols, size);

        group.bench_with_input(BenchmarkId::new("fused", &id), &(), |b, _| {
            b.iter(|| {
                for i in 0..cols {
                    module.vec_znx_big_normalize_add_assign(&mut res_fused, base2k, 0, i, &a, base2k, i, scratch_fused.borrow());
                }
                black_box(());
            })
        });

        group.bench_with_input(BenchmarkId::new("fallback", &id), &(), |b, _| {
            b.iter(|| {
                for i in 0..cols {
                    module.vec_znx_big_normalize(&mut tmp, base2k, 0, 0, &a, base2k, i, scratch_fallback.borrow());
                    for j in 0..size {
                        for (ri, ti) in res_fallback.at_mut(i, j).iter_mut().zip(tmp.at(0, j).iter()) {
                            *ri = ri.wrapping_add(*ti);
                        }
                    }
                }
                black_box(());
            })
        });
    }

    group.finish();
}

pub fn bench_vec_znx_normalize_sub_assign_compare<B: Backend>(
    params: &crate::params::HalSweepParams,
    c: &mut Criterion,
    label: &str,
) where
    Module<B>: VecZnxBigNormalize<B> + ModuleNew<B> + VecZnxBigNormalizeTmpBytes + VecZnxBigAlloc<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vec_znx_big_normalize_sub_assign_compare::{label}");
    let mut group = c.benchmark_group(group_name);

    for sweep in &params.sweeps {
        let n: usize = 1 << sweep[0];
        let cols: usize = sweep[1];
        let size: usize = sweep[2];
        let module: Module<B> = Module::<B>::new(n as u64);
        let base2k: usize = 50;
        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxBig<DeviceBuf<B>, B> = module.vec_znx_big_alloc(cols, size);
        let mut res_fused: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);
        let mut res_fallback: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);
        let mut tmp: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), 1, size);

        source.fill_bytes(a.data_mut().as_mut());
        source.fill_bytes(res_fused.data_mut().as_mut());
        res_fallback.data_mut().copy_from_slice(res_fused.data());

        let mut scratch_fused: ScratchOwned<B> = ScratchOwned::alloc(module.vec_znx_big_normalize_tmp_bytes());
        let mut scratch_fallback: ScratchOwned<B> = ScratchOwned::alloc(module.vec_znx_big_normalize_tmp_bytes());

        let id = format!("{}x({}x{})", n, cols, size);

        group.bench_with_input(BenchmarkId::new("fused", &id), &(), |b, _| {
            b.iter(|| {
                for i in 0..cols {
                    module.vec_znx_big_normalize_sub_assign(&mut res_fused, base2k, 0, i, &a, base2k, i, scratch_fused.borrow());
                }
                black_box(());
            })
        });

        group.bench_with_input(BenchmarkId::new("fallback", &id), &(), |b, _| {
            b.iter(|| {
                for i in 0..cols {
                    module.vec_znx_big_normalize(&mut tmp, base2k, 0, 0, &a, base2k, i, scratch_fallback.borrow());
                    for j in 0..size {
                        for (ri, ti) in res_fallback.at_mut(i, j).iter_mut().zip(tmp.at(0, j).iter()) {
                            *ri = ri.wrapping_sub(*ti);
                        }
                    }
                }
                black_box(());
            })
        });
    }

    group.finish();
}

pub fn bench_vec_znx_big_sub<B: Backend>(params: &crate::params::HalSweepParams, c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxBigSub<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
    B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
{
    let group_name: String = format!("vec_znx_big_sub::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(sweep: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxBigSub<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
        B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
    {
        let module: Module<B> = Module::<B>::new(1 << sweep[0]);

        let cols: usize = sweep[1];
        let size: usize = sweep[2];

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxBig<B::OwnedBuf, B> = module.vec_znx_big_alloc(cols, size);
        let mut b: VecZnxBig<B::OwnedBuf, B> = module.vec_znx_big_alloc(cols, size);
        let mut c: VecZnxBig<B::OwnedBuf, B> = module.vec_znx_big_alloc(cols, size);

        // Fill a with random bytes
        source.fill_bytes(a.data_mut().as_mut());
        source.fill_bytes(b.data_mut().as_mut());
        source.fill_bytes(c.data_mut().as_mut());

        move || {
            let a = <VecZnxBig<B::OwnedBuf, B> as VecZnxBigToBackendRef<B>>::to_backend_ref(&a);
            let b = <VecZnxBig<B::OwnedBuf, B> as VecZnxBigToBackendRef<B>>::to_backend_ref(&b);
            let mut c = <VecZnxBig<B::OwnedBuf, B> as VecZnxBigToBackendMut<B>>::to_backend_mut(&mut c);
            for i in 0..cols {
                module.vec_znx_big_sub(&mut c, i, &a, i, &b, i);
            }
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << sweep[0], sweep[1], sweep[2],));
        let mut runner = runner::<B>(*sweep);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vec_znx_big_sub_assign<B: Backend>(params: &crate::params::HalSweepParams, c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxBigSubInplace<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
    B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
{
    let group_name: String = format!("vec_znx_big_sub_assign::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(sweep: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxBigSubInplace<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
        B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
    {
        let module: Module<B> = Module::<B>::new(1 << sweep[0]);

        let cols: usize = sweep[1];
        let size: usize = sweep[2];

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxBig<B::OwnedBuf, B> = module.vec_znx_big_alloc(cols, size);
        let mut c: VecZnxBig<B::OwnedBuf, B> = module.vec_znx_big_alloc(cols, size);

        // Fill a with random bytes
        source.fill_bytes(a.data_mut().as_mut());
        source.fill_bytes(c.data_mut().as_mut());

        move || {
            let a = <VecZnxBig<B::OwnedBuf, B> as VecZnxBigToBackendRef<B>>::to_backend_ref(&a);
            let mut c = <VecZnxBig<B::OwnedBuf, B> as VecZnxBigToBackendMut<B>>::to_backend_mut(&mut c);
            for i in 0..cols {
                module.vec_znx_big_sub_assign(&mut c, i, &a, i);
            }
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << sweep[0], sweep[1], sweep[2],));
        let mut runner = runner::<B>(*sweep);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vec_znx_big_sub_negate_assign<B: Backend>(params: &crate::params::HalSweepParams, c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxBigSubNegateInplace<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
    B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
{
    let group_name: String = format!("vec_znx_big_sub_assign::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(sweep: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxBigSubNegateInplace<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
        B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
    {
        let module: Module<B> = Module::<B>::new(1 << sweep[0]);

        let cols: usize = sweep[1];
        let size: usize = sweep[2];

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxBig<B::OwnedBuf, B> = module.vec_znx_big_alloc(cols, size);
        let mut c: VecZnxBig<B::OwnedBuf, B> = module.vec_znx_big_alloc(cols, size);

        // Fill a with random bytes
        source.fill_bytes(a.data_mut().as_mut());
        source.fill_bytes(c.data_mut().as_mut());

        move || {
            let a = <VecZnxBig<B::OwnedBuf, B> as VecZnxBigToBackendRef<B>>::to_backend_ref(&a);
            let mut c = <VecZnxBig<B::OwnedBuf, B> as VecZnxBigToBackendMut<B>>::to_backend_mut(&mut c);
            for i in 0..cols {
                module.vec_znx_big_sub_negate_assign(&mut c, i, &a, i);
            }
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << sweep[0], sweep[1], sweep[2],));
        let mut runner = runner::<B>(*sweep);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vec_znx_big_sub_small_a<B: Backend<OwnedBuf = Vec<u8>>>(
    params: &crate::params::HalSweepParams,
    c: &mut Criterion,
    label: &str,
)
where
    Module<B>: VecZnxBigSubSmallABackend<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
    B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
{
    let group_name: String = format!("vec_znx_big_sub_small_a_backend::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend<OwnedBuf = Vec<u8>>>(sweep: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxBigSubSmallABackend<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
        B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
    {
        let module: Module<B> = Module::<B>::new(1 << sweep[0]);

        let cols: usize = sweep[1];
        let size: usize = sweep[2];

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);
        let mut b: VecZnxBig<B::OwnedBuf, B> = module.vec_znx_big_alloc(cols, size);
        let mut c: VecZnxBig<B::OwnedBuf, B> = module.vec_znx_big_alloc(cols, size);

        // Fill a with random bytes
        source.fill_bytes(a.data_mut().as_mut());
        source.fill_bytes(b.data_mut().as_mut());
        source.fill_bytes(c.data_mut().as_mut());

        move || {
            let a = <VecZnx<Vec<u8>> as VecZnxToBackendRef<B>>::to_backend_ref(&a);
            let b = <VecZnxBig<B::OwnedBuf, B> as VecZnxBigToBackendRef<B>>::to_backend_ref(&b);
            let mut c = <VecZnxBig<B::OwnedBuf, B> as VecZnxBigToBackendMut<B>>::to_backend_mut(&mut c);
            for i in 0..cols {
                module.vec_znx_big_sub_small_a_backend(&mut c, i, &a, i, &b, i);
            }
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << sweep[0], sweep[1], sweep[2],));
        let mut runner = runner::<B>(*sweep);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vec_znx_big_sub_small_b<B: Backend<OwnedBuf = Vec<u8>>>(
    params: &crate::params::HalSweepParams,
    c: &mut Criterion,
    label: &str,
)
where
    Module<B>: VecZnxBigSubSmallBBackend<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
    B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
{
    let group_name: String = format!("vec_znx_big_sub_small_b_backend::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend<OwnedBuf = Vec<u8>>>(sweep: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxBigSubSmallBBackend<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
        B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
    {
        let module: Module<B> = Module::<B>::new(1 << sweep[0]);

        let cols: usize = sweep[1];
        let size: usize = sweep[2];

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxBig<B::OwnedBuf, B> = module.vec_znx_big_alloc(cols, size);
        let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);
        let mut c: VecZnxBig<B::OwnedBuf, B> = module.vec_znx_big_alloc(cols, size);

        // Fill a with random bytes
        source.fill_bytes(a.data_mut().as_mut());
        source.fill_bytes(b.data_mut().as_mut());
        source.fill_bytes(c.data_mut().as_mut());

        move || {
            let a = <VecZnxBig<B::OwnedBuf, B> as VecZnxBigToBackendRef<B>>::to_backend_ref(&a);
            let b = <VecZnx<Vec<u8>> as VecZnxToBackendRef<B>>::to_backend_ref(&b);
            let mut c = <VecZnxBig<B::OwnedBuf, B> as VecZnxBigToBackendMut<B>>::to_backend_mut(&mut c);
            for i in 0..cols {
                module.vec_znx_big_sub_small_b_backend(&mut c, i, &a, i, &b, i);
            }
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << sweep[0], sweep[1], sweep[2],));
        let mut runner = runner::<B>(*sweep);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}
