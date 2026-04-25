use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};
use rand::Rng;

use poulpy_hal::{
    api::{
        ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxDftAlloc, VmpApplyDft, VmpApplyDftTmpBytes, VmpApplyDftToDft,
        VmpApplyDftToDftTmpBytes, VmpPMatAlloc, VmpPrepare, VmpPrepareTmpBytes,
    },
    layouts::{
        Backend, DataViewMut, MatZnx, MatZnxBackendRef, MatZnxToBackendRef, Module, ScratchOwned, VecZnx, VecZnxDft,
        VecZnxDftToBackendRef, VecZnxToBackendRef, VmpPMat, VmpPMatBackendMut, VmpPMatToBackendMut, VmpPMatToBackendRef,
    },
    source::Source,
};

pub fn bench_vmp_prepare<B>(params: &crate::params::VmpSweepParams, c: &mut Criterion, label: &str)
where
    Module<B>: ModuleNew<B> + VmpPMatAlloc<B> + VmpPrepare<B> + VmpPrepareTmpBytes,
    B: Backend<OwnedBuf = Vec<u8>>,
    B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vmp_prepare::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(sweep: [usize; 5]) -> impl FnMut()
    where
        Module<B>: ModuleNew<B> + VmpPMatAlloc<B> + VmpPrepare<B> + VmpPrepareTmpBytes,
        B: Backend<OwnedBuf = Vec<u8>>,
        B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << sweep[0]);

        let rows: usize = sweep[1];
        let cols_in: usize = sweep[2];
        let cols_out: usize = sweep[3];
        let size: usize = sweep[4];

        let mut source: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.vmp_prepare_tmp_bytes(rows, cols_in, cols_out, size));

        let mut mat: MatZnx<Vec<u8>> = MatZnx::alloc(module.n(), rows, cols_in, cols_out, size);
        let mut pmat: VmpPMat<B::OwnedBuf, B> = module.vmp_pmat_alloc(rows, cols_in, cols_out, size);

        source.fill_bytes(mat.data_mut());
        source.fill_bytes(pmat.data_mut().as_mut());

        move || {
            let mut pmat_backend: VmpPMatBackendMut<'_, B> = pmat.to_backend_mut();
            let mat_backend: MatZnxBackendRef<'_, B> = <MatZnx<Vec<u8>> as MatZnxToBackendRef<B>>::to_backend_ref(&mat);
            module.vmp_prepare(&mut pmat_backend, &mat_backend, &mut scratch.borrow());
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let id = BenchmarkId::from_parameter(format!(
            "{}x({}x{})x({}x{})",
            1 << sweep[0],
            sweep[2],
            sweep[1],
            sweep[3],
            sweep[4]
        ));
        let mut runner = runner::<B>(*sweep);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vmp_apply_dft<B: Backend<OwnedBuf = Vec<u8>>>(params: &crate::params::VmpSweepParams, c: &mut Criterion, label: &str)
where
    Module<B>: ModuleNew<B> + VmpApplyDftTmpBytes + VmpApplyDft<B> + VmpPMatAlloc<B> + VecZnxDftAlloc<B>,
    B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vmp_apply_dft::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend<OwnedBuf = Vec<u8>>>(sweep: [usize; 5]) -> impl FnMut()
    where
        Module<B>: ModuleNew<B> + VmpApplyDftTmpBytes + VmpApplyDft<B> + VmpPMatAlloc<B> + VecZnxDftAlloc<B>,
        B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << sweep[0]);

        let rows: usize = sweep[1];
        let cols_in: usize = sweep[2];
        let cols_out: usize = sweep[3];
        let size: usize = sweep[4];

        let mut source: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(1 << 20);

        let mut res: VecZnxDft<B::OwnedBuf, B> = module.vec_znx_dft_alloc(cols_out, size);
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols_in, size);
        let mut pmat: VmpPMat<B::OwnedBuf, B> = module.vmp_pmat_alloc(rows, cols_in, cols_out, size);

        source.fill_bytes(pmat.data_mut().as_mut());
        source.fill_bytes(res.data_mut().as_mut());
        source.fill_bytes(a.data_mut().as_mut());

        move || {
            let pmat = pmat.to_backend_ref();
            let a = <VecZnx<Vec<u8>> as VecZnxToBackendRef<B>>::to_backend_ref(&a);
            module.vmp_apply_dft(&mut res, &a, &pmat, &mut scratch.borrow());
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let id = BenchmarkId::from_parameter(format!(
            "{}x({}x{})x({}x{})",
            1 << sweep[0],
            sweep[2],
            sweep[1],
            sweep[3],
            sweep[4]
        ));
        let mut runner = runner::<B>(*sweep);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vmp_apply_dft_to_dft<B: Backend>(params: &crate::params::VmpSweepParams, c: &mut Criterion, label: &str)
where
    Module<B>: ModuleNew<B> + VecZnxDftAlloc<B> + VmpPMatAlloc<B> + VmpApplyDftToDft<B> + VmpApplyDftToDftTmpBytes,
    B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vmp_apply_dft_to_dft::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(sweep: [usize; 5]) -> impl FnMut()
    where
        Module<B>: ModuleNew<B> + VecZnxDftAlloc<B> + VmpPMatAlloc<B> + VmpApplyDftToDft<B> + VmpApplyDftToDftTmpBytes,
        B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << sweep[0]);

        let rows: usize = sweep[1];
        let cols_in: usize = sweep[2];
        let cols_out: usize = sweep[3];
        let size: usize = sweep[4];

        let mut source: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<B> =
            ScratchOwned::alloc(module.vmp_apply_dft_to_dft_tmp_bytes(size, size, rows, cols_in, cols_out, size));

        let mut res: VecZnxDft<B::OwnedBuf, B> = module.vec_znx_dft_alloc(cols_out, size);
        let mut a: VecZnxDft<B::OwnedBuf, B> = module.vec_znx_dft_alloc(cols_in, size);

        let mut pmat: VmpPMat<B::OwnedBuf, B> = module.vmp_pmat_alloc(rows, cols_in, cols_out, size);

        source.fill_bytes(pmat.data_mut().as_mut());
        source.fill_bytes(res.data_mut().as_mut());
        source.fill_bytes(a.data_mut().as_mut());

        move || {
            let pmat = pmat.to_backend_ref();
            module.vmp_apply_dft_to_dft(&mut res, &a.to_backend_ref(), &pmat, 0, &mut scratch.borrow());
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let id = BenchmarkId::from_parameter(format!(
            "{}x({}x{})x({}x{})",
            1 << sweep[0], // n
            sweep[2],      // cols_in
            sweep[1],      // size_in (=rows)
            sweep[3],      // cols_out
            sweep[4]       // size_out
        ));
        let mut runner = runner::<B>(*sweep);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}
