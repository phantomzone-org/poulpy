use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};

use crate::{
    api::{CnvPVecAlloc, Convolution, ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxBigAlloc, VecZnxDftAlloc},
    layouts::{Backend, CnvPVecL, CnvPVecR, FillUniform, Module, ScratchOwned, VecZnx, VecZnxBig},
    source::Source,
};

pub fn bench_cnv_prepare_left<BE: Backend>(c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE> + Convolution<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let group_name: String = format!("cnv_prepare_left::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<BE: Backend>(n: usize, size: usize) -> impl FnMut()
    where
        Module<BE>: ModuleNew<BE> + Convolution<BE> + CnvPVecAlloc<BE>,
        ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    {
        let mut source: Source = Source::new([0u8; 32]);

        let base2k: usize = 12;

        let c_size: usize = size + size - 1;

        let module: Module<BE> = Module::<BE>::new(n as u64);

        let mut a_prep: CnvPVecL<Vec<u8>, BE> = module.cnv_pvec_left_alloc(1, size);

        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), 1, size);

        a.fill_uniform(base2k, &mut source);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.cnv_prepare_left_tmp_bytes(c_size, size));

        move || {
            module.cnv_prepare_left(&mut a_prep, &a, scratch.borrow());
            black_box(());
        }
    }

    for params in [[10, 1], [11, 2], [12, 4], [13, 8], [14, 16], [15, 32], [16, 64]] {
        let log_n: usize = params[0];
        let size: usize = params[1];
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x{}", 1 << log_n, size));
        let mut runner = runner(1 << log_n, size);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_cnv_prepare_right<BE: Backend>(c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE> + Convolution<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let group_name: String = format!("cnv_prepare_right::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<BE: Backend>(n: usize, size: usize) -> impl FnMut()
    where
        Module<BE>: ModuleNew<BE> + Convolution<BE> + CnvPVecAlloc<BE>,
        ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    {
        let mut source: Source = Source::new([0u8; 32]);

        let base2k: usize = 12;

        let c_size: usize = size + size - 1;

        let module: Module<BE> = Module::<BE>::new(n as u64);

        let mut a_prep: CnvPVecR<Vec<u8>, BE> = module.cnv_pvec_right_alloc(1, size);

        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), 1, size);

        a.fill_uniform(base2k, &mut source);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.cnv_prepare_right_tmp_bytes(c_size, size));

        move || {
            module.cnv_prepare_right(&mut a_prep, &a, scratch.borrow());
            black_box(());
        }
    }

    for params in [[10, 1], [11, 2], [12, 4], [13, 8], [14, 16], [15, 32], [16, 64]] {
        let log_n: usize = params[0];
        let size: usize = params[1];
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x{}", 1 << log_n, size));
        let mut runner = runner(1 << log_n, size);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_cnv_apply_dft<BE: Backend>(c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE> + Convolution<BE> + VecZnxDftAlloc<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let group_name: String = format!("cnv_apply_dft::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<BE: Backend>(n: usize, size: usize) -> impl FnMut()
    where
        Module<BE>: ModuleNew<BE> + Convolution<BE> + VecZnxDftAlloc<BE> + CnvPVecAlloc<BE>,
        ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    {
        let mut source: Source = Source::new([0u8; 32]);

        let base2k: usize = 12;

        let c_size: usize = size + size - 1;

        let module: Module<BE> = Module::<BE>::new(n as u64);

        let mut a_prep: CnvPVecL<Vec<u8>, BE> = module.cnv_pvec_left_alloc(1, size);
        let mut b_prep: CnvPVecR<Vec<u8>, BE> = module.cnv_pvec_right_alloc(1, size);

        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), 1, size);
        let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), 1, size);
        let mut c_dft = module.vec_znx_dft_alloc(1, c_size);

        a.fill_uniform(base2k, &mut source);
        b.fill_uniform(base2k, &mut source);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            module
                .cnv_apply_dft_tmp_bytes(c_size, 0, size, size)
                .max(module.cnv_prepare_left_tmp_bytes(c_size, size))
                .max(module.cnv_prepare_right_tmp_bytes(c_size, size)),
        );
        module.cnv_prepare_left(&mut a_prep, &a, scratch.borrow());
        module.cnv_prepare_right(&mut b_prep, &b, scratch.borrow());
        move || {
            module.cnv_apply_dft(&mut c_dft, 0, 0, &a_prep, 0, &b_prep, 0, scratch.borrow());
            black_box(());
        }
    }

    for params in [[10, 1], [11, 2], [12, 4], [13, 8], [14, 16], [15, 32], [16, 64]] {
        let log_n: usize = params[0];
        let size: usize = params[1];
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x{}", 1 << log_n, size));
        let mut runner = runner(1 << log_n, size);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_cnv_pairwise_apply_dft<BE: Backend>(c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE> + Convolution<BE> + VecZnxDftAlloc<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let group_name: String = format!("cnv_pairwise_apply_dft::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<BE: Backend>(n: usize, size: usize) -> impl FnMut()
    where
        Module<BE>: ModuleNew<BE> + Convolution<BE> + VecZnxDftAlloc<BE> + CnvPVecAlloc<BE>,
        ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    {
        let mut source: Source = Source::new([0u8; 32]);

        let base2k: usize = 12;

        let module: Module<BE> = Module::<BE>::new(n as u64);

        let cols = 2;
        let c_size: usize = size + size - 1;

        let mut a_prep: CnvPVecL<Vec<u8>, BE> = module.cnv_pvec_left_alloc(cols, size);
        let mut b_prep: CnvPVecR<Vec<u8>, BE> = module.cnv_pvec_right_alloc(cols, size);

        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);
        let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);
        let mut c_dft = module.vec_znx_dft_alloc(1, c_size);

        a.fill_uniform(base2k, &mut source);
        b.fill_uniform(base2k, &mut source);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            module
                .cnv_pairwise_apply_dft_tmp_bytes(c_size, 0, size, size)
                .max(module.cnv_prepare_left_tmp_bytes(c_size, size))
                .max(module.cnv_prepare_right_tmp_bytes(c_size, size)),
        );
        module.cnv_prepare_left(&mut a_prep, &a, scratch.borrow());
        module.cnv_prepare_right(&mut b_prep, &b, scratch.borrow());
        move || {
            module.cnv_pairwise_apply_dft(&mut c_dft, 0, 0, &a_prep, &b_prep, 0, 1, scratch.borrow());
            black_box(());
        }
    }

    for params in [[10, 1], [11, 2], [12, 4], [13, 8], [14, 16], [15, 32], [16, 64]] {
        let log_n: usize = params[0];
        let size: usize = params[1];
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x{}", 1 << log_n, size));
        let mut runner = runner(1 << log_n, size);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_cnv_by_const_apply<BE: Backend>(c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE> + Convolution<BE> + VecZnxBigAlloc<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let group_name: String = format!("cnv_by_const::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<BE: Backend>(n: usize, size: usize) -> impl FnMut()
    where
        Module<BE>: ModuleNew<BE> + Convolution<BE> + VecZnxBigAlloc<BE> + CnvPVecAlloc<BE>,
        ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    {
        let mut source: Source = Source::new([0u8; 32]);

        let base2k: usize = 12;

        let module: Module<BE> = Module::<BE>::new(n as u64);

        let cols = 2;
        let c_size: usize = size + size - 1;

        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);
        let mut c_big: VecZnxBig<Vec<u8>, BE> = module.vec_znx_big_alloc(1, c_size);

        a.fill_uniform(base2k, &mut source);
        let mut b = vec![0i64; size];
        for x in &mut b {
            *x = source.next_i64();
        }

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.cnv_by_const_apply_tmp_bytes(c_size, 0, size, size));
        move || {
            module.cnv_by_const_apply(&mut c_big, 0, 0, &a, 0, &b, scratch.borrow());
            black_box(());
        }
    }

    for params in [[10, 1], [11, 2], [12, 4], [13, 8], [14, 16], [15, 32], [16, 64]] {
        let log_n: usize = params[0];
        let size: usize = params[1];
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x{}", 1 << log_n, size));
        let mut runner = runner(1 << log_n, size);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}
