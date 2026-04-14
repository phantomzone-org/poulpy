use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};
use rand::Rng;

use poulpy_hal::{
    api::{
        ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxDftAlloc, VmpApplyDft, VmpApplyDftTmpBytes, VmpApplyDftToDft,
        VmpApplyDftToDftTmpBytes, VmpPMatAlloc, VmpPrepare, VmpPrepareTmpBytes,
    },
    layouts::{Backend, DataViewMut, DeviceBuf, MatZnx, Module, ScratchOwned, VecZnx, VecZnxDft, VmpPMat},
    source::Source,
};

pub fn bench_vmp_prepare<B: Backend>(params: &crate::params::VmpSweepParams, c: &mut Criterion, label: &str)
where
    Module<B>: ModuleNew<B> + VmpPMatAlloc<B> + VmpPrepare<B> + VmpPrepareTmpBytes,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vmp_prepare::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(sweep: [usize; 5]) -> impl FnMut()
    where
        Module<B>: ModuleNew<B> + VmpPMatAlloc<B> + VmpPrepare<B> + VmpPrepareTmpBytes,
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
        let mut pmat: VmpPMat<DeviceBuf<B>, B> = module.vmp_pmat_alloc(rows, cols_in, cols_out, size);

        source.fill_bytes(mat.data_mut());
        source.fill_bytes(pmat.data_mut().as_mut());

        move || {
            module.vmp_prepare(&mut pmat, &mat, scratch.borrow());
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

pub fn bench_vmp_apply_dft<B: Backend>(params: &crate::params::VmpSweepParams, c: &mut Criterion, label: &str)
where
    Module<B>: ModuleNew<B> + VmpApplyDftTmpBytes + VmpApplyDft<B> + VmpPMatAlloc<B> + VecZnxDftAlloc<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vmp_apply_dft::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(sweep: [usize; 5]) -> impl FnMut()
    where
        Module<B>: ModuleNew<B> + VmpApplyDftTmpBytes + VmpApplyDft<B> + VmpPMatAlloc<B> + VecZnxDftAlloc<B>,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << sweep[0]);

        let rows: usize = sweep[1];
        let cols_in: usize = sweep[2];
        let cols_out: usize = sweep[3];
        let size: usize = sweep[4];

        let mut source: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(1 << 20);

        let mut res: VecZnxDft<DeviceBuf<B>, B> = module.vec_znx_dft_alloc(cols_out, size);
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols_in, size);
        let mut pmat: VmpPMat<DeviceBuf<B>, B> = module.vmp_pmat_alloc(rows, cols_in, cols_out, size);

        source.fill_bytes(pmat.data_mut().as_mut());
        source.fill_bytes(res.data_mut().as_mut());
        source.fill_bytes(a.data_mut().as_mut());

        move || {
            module.vmp_apply_dft(&mut res, &a, &pmat, scratch.borrow());
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
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vmp_apply_dft_to_dft::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(sweep: [usize; 5]) -> impl FnMut()
    where
        Module<B>: ModuleNew<B> + VecZnxDftAlloc<B> + VmpPMatAlloc<B> + VmpApplyDftToDft<B> + VmpApplyDftToDftTmpBytes,
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

        let mut res: VecZnxDft<DeviceBuf<B>, B> = module.vec_znx_dft_alloc(cols_out, size);
        let mut a: VecZnxDft<DeviceBuf<B>, B> = module.vec_znx_dft_alloc(cols_in, size);

        let mut pmat: VmpPMat<DeviceBuf<B>, B> = module.vmp_pmat_alloc(rows, cols_in, cols_out, size);

        source.fill_bytes(pmat.data_mut().as_mut());
        source.fill_bytes(res.data_mut().as_mut());
        source.fill_bytes(a.data_mut().as_mut());

        move || {
            module.vmp_apply_dft_to_dft(&mut res, &a, &pmat, 0, scratch.borrow());
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
