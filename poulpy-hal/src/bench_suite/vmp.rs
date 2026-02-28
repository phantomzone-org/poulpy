use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};
use rand::Rng;

use crate::{
    api::{
        ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxDftAlloc, VmpApplyDft, VmpApplyDftTmpBytes, VmpApplyDftToDft,
        VmpApplyDftToDftAdd, VmpApplyDftToDftAddTmpBytes, VmpApplyDftToDftTmpBytes, VmpPMatAlloc, VmpPrepare, VmpPrepareTmpBytes,
    },
    layouts::{Backend, DataViewMut, MatZnx, Module, ScratchOwned, VecZnx, VecZnxDft, VmpPMat},
    source::Source,
};

pub fn bench_vmp_prepare<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: ModuleNew<B> + VmpPMatAlloc<B> + VmpPrepare<B> + VmpPrepareTmpBytes,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vmp_prepare::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 5]) -> impl FnMut()
    where
        Module<B>: ModuleNew<B> + VmpPMatAlloc<B> + VmpPrepare<B> + VmpPrepareTmpBytes,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << params[0]);

        let rows: usize = params[1];
        let cols_in: usize = params[2];
        let cols_out: usize = params[3];
        let size: usize = params[4];

        let mut source: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.vmp_prepare_tmp_bytes(rows, cols_in, cols_out, size));

        let mut mat: MatZnx<Vec<u8>> = MatZnx::alloc(module.n(), rows, cols_in, cols_out, size);
        let mut pmat: VmpPMat<Vec<u8>, B> = module.vmp_pmat_alloc(rows, cols_in, cols_out, size);

        source.fill_bytes(mat.data_mut());
        source.fill_bytes(pmat.data_mut());

        move || {
            module.vmp_prepare(&mut pmat, &mat, scratch.borrow());
            black_box(());
        }
    }

    for params in [
        [10, 2, 1, 2, 3],
        [11, 4, 1, 2, 5],
        [12, 7, 1, 2, 8],
        [13, 15, 1, 2, 16],
        [14, 31, 1, 2, 32],
    ] {
        let id = BenchmarkId::from_parameter(format!(
            "{}x({}x{})x({}x{})",
            1 << params[0],
            params[2],
            params[1],
            params[3],
            params[4]
        ));
        let mut runner = runner::<B>(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vmp_apply_dft<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: ModuleNew<B> + VmpApplyDftTmpBytes + VmpApplyDft<B> + VmpPMatAlloc<B> + VecZnxDftAlloc<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vmp_apply_dft::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 5]) -> impl FnMut()
    where
        Module<B>: ModuleNew<B> + VmpApplyDftTmpBytes + VmpApplyDft<B> + VmpPMatAlloc<B> + VecZnxDftAlloc<B>,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << params[0]);

        let rows: usize = params[1];
        let cols_in: usize = params[2];
        let cols_out: usize = params[3];
        let size: usize = params[4];

        let mut source: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(1 << 20);

        let mut res: VecZnxDft<Vec<u8>, _> = module.vec_znx_dft_alloc(cols_out, size);
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols_in, size);
        let mut pmat: VmpPMat<Vec<u8>, B> = module.vmp_pmat_alloc(rows, cols_in, cols_out, size);

        source.fill_bytes(pmat.data_mut());
        source.fill_bytes(res.data_mut());
        source.fill_bytes(a.data_mut());

        move || {
            module.vmp_apply_dft(&mut res, &a, &pmat, scratch.borrow());
            black_box(());
        }
    }

    for params in [
        [10, 2, 1, 2, 3],
        [11, 4, 1, 2, 5],
        [12, 7, 1, 2, 8],
        [13, 15, 1, 2, 16],
        [14, 31, 1, 2, 32],
    ] {
        let id = BenchmarkId::from_parameter(format!(
            "{}x({}x{})x({}x{})",
            1 << params[0],
            params[2],
            params[1],
            params[3],
            params[4]
        ));
        let mut runner = runner::<B>(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vmp_apply_dft_to_dft<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: ModuleNew<B> + VecZnxDftAlloc<B> + VmpPMatAlloc<B> + VmpApplyDftToDft<B> + VmpApplyDftToDftTmpBytes,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vmp_apply_dft_to_dft::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 5]) -> impl FnMut()
    where
        Module<B>: ModuleNew<B> + VecZnxDftAlloc<B> + VmpPMatAlloc<B> + VmpApplyDftToDft<B> + VmpApplyDftToDftTmpBytes,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << params[0]);

        let rows: usize = params[1];
        let cols_in: usize = params[2];
        let cols_out: usize = params[3];
        let size: usize = params[4];

        let mut source: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<B> =
            ScratchOwned::alloc(module.vmp_apply_dft_to_dft_tmp_bytes(size, size, rows, cols_in, cols_out, size));

        let mut res: VecZnxDft<Vec<u8>, _> = module.vec_znx_dft_alloc(cols_out, size);
        let mut a: VecZnxDft<Vec<u8>, _> = module.vec_znx_dft_alloc(cols_in, size);

        let mut pmat: VmpPMat<Vec<u8>, B> = module.vmp_pmat_alloc(rows, cols_in, cols_out, size);

        source.fill_bytes(pmat.data_mut());
        source.fill_bytes(res.data_mut());
        source.fill_bytes(a.data_mut());

        move || {
            module.vmp_apply_dft_to_dft(&mut res, &a, &pmat, scratch.borrow());
            black_box(());
        }
    }

    for params in [
        [10, 2, 1, 2, 3],
        [11, 4, 1, 2, 5],
        [12, 7, 1, 2, 8],
        [13, 15, 1, 2, 16],
        [14, 31, 1, 2, 32],
    ] {
        let id = BenchmarkId::from_parameter(format!(
            "{}x({}x{})x({}x{})",
            1 << params[0], // n
            params[2],      // cols_in
            params[1],      // size_in (=rows)
            params[3],      // cols_out
            params[4]       // size_out
        ));
        let mut runner = runner::<B>(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vmp_apply_dft_to_dft_add<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: ModuleNew<B> + VecZnxDftAlloc<B> + VmpPMatAlloc<B> + VmpApplyDftToDftAdd<B> + VmpApplyDftToDftAddTmpBytes,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vmp_apply_dft_to_dft_add::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 5]) -> impl FnMut()
    where
        Module<B>: ModuleNew<B> + VecZnxDftAlloc<B> + VmpPMatAlloc<B> + VmpApplyDftToDftAdd<B> + VmpApplyDftToDftAddTmpBytes,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << params[0]);

        let rows: usize = params[1];
        let cols_in: usize = params[2];
        let cols_out: usize = params[3];
        let size: usize = params[4];

        let mut source: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<B> =
            ScratchOwned::alloc(module.vmp_apply_dft_to_dft_add_tmp_bytes(size, size, rows, cols_in, cols_out, size));

        let mut res: VecZnxDft<Vec<u8>, _> = module.vec_znx_dft_alloc(cols_out, size);
        let mut a: VecZnxDft<Vec<u8>, _> = module.vec_znx_dft_alloc(cols_in, size);

        let mut pmat: VmpPMat<Vec<u8>, B> = module.vmp_pmat_alloc(rows, cols_in, cols_out, size);

        source.fill_bytes(pmat.data_mut());
        source.fill_bytes(res.data_mut());
        source.fill_bytes(a.data_mut());

        move || {
            module.vmp_apply_dft_to_dft_add(&mut res, &a, &pmat, 1, scratch.borrow());
            black_box(());
        }
    }

    for params in [
        [10, 2, 1, 2, 3],
        [11, 4, 1, 2, 5],
        [12, 7, 1, 2, 8],
        [13, 15, 1, 2, 16],
        [14, 31, 1, 2, 32],
    ] {
        let id = BenchmarkId::from_parameter(format!(
            "{}x({}x{})x({}x{})",
            1 << params[0],
            params[2],
            params[1],
            params[3],
            params[4]
        ));
        let mut runner = runner::<B>(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}
