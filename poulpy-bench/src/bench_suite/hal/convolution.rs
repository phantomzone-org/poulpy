use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};

use poulpy_hal::{
    api::{CnvPVecAlloc, Convolution, ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxBigAlloc, VecZnxDftAlloc},
    layouts::{
        Backend, CnvPVecL, CnvPVecLToBackendRef, CnvPVecR, CnvPVecRToBackendRef, FillUniform, Module, ScratchOwned, VecZnx,
        VecZnxBig, VecZnxToRef, vec_znx_backend_ref_from_ref,
    },
    source::Source,
};

pub fn bench_cnv_prepare_left<BE>(params: &crate::params::CnvSweepParams, c: &mut Criterion, label: &str)
where
    BE: Backend + 'static,
    Module<BE>: ModuleNew<BE> + Convolution<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    BE::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
    for<'x> BE: Backend<BufRef<'x> = &'x [u8]>,
{
    let group_name: String = format!("cnv_prepare_left::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<BE>(n: usize, size: usize) -> impl FnMut()
    where
        BE: Backend + 'static,
        Module<BE>: ModuleNew<BE> + Convolution<BE> + CnvPVecAlloc<BE>,
        ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
        BE::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
        for<'x> BE: Backend<BufRef<'x> = &'x [u8]>,
    {
        let mut source: Source = Source::new([0u8; 32]);

        let base2k: usize = 12;

        let c_size: usize = size + size - 1;

        let module: Module<BE> = Module::<BE>::new(n as u64);

        let mut a_prep: CnvPVecL<BE::OwnedBuf, BE> = module.cnv_pvec_left_alloc(1, size);

        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), 1, size);

        a.fill_uniform(base2k, &mut source);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.cnv_prepare_left_tmp_bytes(c_size, size));

        move || {
            module.cnv_prepare_left(
                &mut a_prep,
                &vec_znx_backend_ref_from_ref::<BE>(&a.to_ref()),
                !0i64,
                &mut scratch.borrow(),
            );
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let log_n: usize = sweep[0];
        let size: usize = sweep[1];
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x{}", 1 << log_n, size));
        let mut runner = runner(1 << log_n, size);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_cnv_prepare_right<BE>(params: &crate::params::CnvSweepParams, c: &mut Criterion, label: &str)
where
    BE: Backend + 'static,
    Module<BE>: ModuleNew<BE> + Convolution<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    BE::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
    for<'x> BE: Backend<BufRef<'x> = &'x [u8]>,
{
    let group_name: String = format!("cnv_prepare_right::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<BE>(n: usize, size: usize) -> impl FnMut()
    where
        BE: Backend + 'static,
        Module<BE>: ModuleNew<BE> + Convolution<BE> + CnvPVecAlloc<BE>,
        ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
        BE::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
        for<'x> BE: Backend<BufRef<'x> = &'x [u8]>,
    {
        let mut source: Source = Source::new([0u8; 32]);

        let base2k: usize = 12;

        let c_size: usize = size + size - 1;

        let module: Module<BE> = Module::<BE>::new(n as u64);

        let mut a_prep: CnvPVecR<BE::OwnedBuf, BE> = module.cnv_pvec_right_alloc(1, size);

        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), 1, size);

        a.fill_uniform(base2k, &mut source);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.cnv_prepare_right_tmp_bytes(c_size, size));

        move || {
            module.cnv_prepare_right(
                &mut a_prep,
                &vec_znx_backend_ref_from_ref::<BE>(&a.to_ref()),
                !0i64,
                &mut scratch.borrow(),
            );
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let log_n: usize = sweep[0];
        let size: usize = sweep[1];
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x{}", 1 << log_n, size));
        let mut runner = runner(1 << log_n, size);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_cnv_apply_dft<BE>(params: &crate::params::CnvSweepParams, c: &mut Criterion, label: &str)
where
    BE: Backend + 'static,
    Module<BE>: ModuleNew<BE> + Convolution<BE> + VecZnxDftAlloc<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    BE::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
    for<'x> BE: Backend<BufRef<'x> = &'x [u8]>,
{
    let group_name: String = format!("cnv_apply_dft::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<BE>(n: usize, size: usize) -> impl FnMut()
    where
        BE: Backend + 'static,
        Module<BE>: ModuleNew<BE> + Convolution<BE> + VecZnxDftAlloc<BE> + CnvPVecAlloc<BE>,
        ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
        BE::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
        for<'x> BE: Backend<BufRef<'x> = &'x [u8]>,
    {
        let mut source: Source = Source::new([0u8; 32]);

        let base2k: usize = 12;

        let c_size: usize = size + size - 1;

        let module: Module<BE> = Module::<BE>::new(n as u64);

        let mut a_prep: CnvPVecL<BE::OwnedBuf, BE> = module.cnv_pvec_left_alloc(1, size);
        let mut b_prep: CnvPVecR<BE::OwnedBuf, BE> = module.cnv_pvec_right_alloc(1, size);

        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), 1, size);
        let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), 1, size);
        let mut c_dft = module.vec_znx_dft_alloc(1, c_size);

        a.fill_uniform(base2k, &mut source);
        b.fill_uniform(base2k, &mut source);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            module
                .cnv_apply_dft_tmp_bytes(0, c_size, size, size)
                .max(module.cnv_prepare_left_tmp_bytes(c_size, size))
                .max(module.cnv_prepare_right_tmp_bytes(c_size, size)),
        );
        module.cnv_prepare_left(
            &mut a_prep,
            &vec_znx_backend_ref_from_ref::<BE>(&a.to_ref()),
            !0i64,
            &mut scratch.borrow(),
        );
        module.cnv_prepare_right(
            &mut b_prep,
            &vec_znx_backend_ref_from_ref::<BE>(&b.to_ref()),
            !0i64,
            &mut scratch.borrow(),
        );
        move || {
            module.cnv_apply_dft(
                0,
                &mut c_dft,
                0,
                &a_prep.to_backend_ref(),
                0,
                &b_prep.to_backend_ref(),
                0,
                &mut scratch.borrow(),
            );
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let log_n: usize = sweep[0];
        let size: usize = sweep[1];
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x{}", 1 << log_n, size));
        let mut runner = runner(1 << log_n, size);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_cnv_pairwise_apply_dft<BE>(params: &crate::params::CnvSweepParams, c: &mut Criterion, label: &str)
where
    BE: Backend + 'static,
    Module<BE>: ModuleNew<BE> + Convolution<BE> + VecZnxDftAlloc<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    BE::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
    for<'x> BE: Backend<BufRef<'x> = &'x [u8]>,
{
    let group_name: String = format!("cnv_pairwise_apply_dft::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<BE>(n: usize, size: usize) -> impl FnMut()
    where
        BE: Backend + 'static,
        Module<BE>: ModuleNew<BE> + Convolution<BE> + VecZnxDftAlloc<BE> + CnvPVecAlloc<BE>,
        ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
        BE::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
        for<'x> BE: Backend<BufRef<'x> = &'x [u8]>,
    {
        let mut source: Source = Source::new([0u8; 32]);

        let base2k: usize = 12;

        let module: Module<BE> = Module::<BE>::new(n as u64);

        let cols = 2;
        let c_size: usize = size + size - 1;

        let mut a_prep: CnvPVecL<BE::OwnedBuf, BE> = module.cnv_pvec_left_alloc(cols, size);
        let mut b_prep: CnvPVecR<BE::OwnedBuf, BE> = module.cnv_pvec_right_alloc(cols, size);

        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);
        let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);
        let mut c_dft = module.vec_znx_dft_alloc(1, c_size);

        a.fill_uniform(base2k, &mut source);
        b.fill_uniform(base2k, &mut source);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            module
                .cnv_pairwise_apply_dft_tmp_bytes(0, c_size, size, size)
                .max(module.cnv_prepare_left_tmp_bytes(c_size, size))
                .max(module.cnv_prepare_right_tmp_bytes(c_size, size)),
        );
        module.cnv_prepare_left(
            &mut a_prep,
            &vec_znx_backend_ref_from_ref::<BE>(&a.to_ref()),
            !0i64,
            &mut scratch.borrow(),
        );
        module.cnv_prepare_right(
            &mut b_prep,
            &vec_znx_backend_ref_from_ref::<BE>(&b.to_ref()),
            !0i64,
            &mut scratch.borrow(),
        );
        move || {
            module.cnv_pairwise_apply_dft(
                0,
                &mut c_dft,
                0,
                &a_prep.to_backend_ref(),
                &b_prep.to_backend_ref(),
                0,
                1,
                &mut scratch.borrow(),
            );
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let log_n: usize = sweep[0];
        let size: usize = sweep[1];
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x{}", 1 << log_n, size));
        let mut runner = runner(1 << log_n, size);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_cnv_by_const_apply<BE>(params: &crate::params::CnvSweepParams, c: &mut Criterion, label: &str)
where
    BE: Backend + 'static,
    Module<BE>: ModuleNew<BE> + Convolution<BE> + VecZnxBigAlloc<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    BE::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
    for<'x> BE: Backend<BufRef<'x> = &'x [u8]>,
{
    let group_name: String = format!("cnv_by_const::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<BE>(n: usize, size: usize) -> impl FnMut()
    where
        BE: Backend + 'static,
        Module<BE>: ModuleNew<BE> + Convolution<BE> + VecZnxBigAlloc<BE> + CnvPVecAlloc<BE>,
        ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
        BE::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
        for<'x> BE: Backend<BufRef<'x> = &'x [u8]>,
    {
        let mut source: Source = Source::new([0u8; 32]);

        let base2k: usize = 12;

        let module: Module<BE> = Module::<BE>::new(n as u64);

        let cols = 2;
        let c_size: usize = size + size - 1;

        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);
        let mut c_big: VecZnxBig<BE::OwnedBuf, BE> = module.vec_znx_big_alloc(1, c_size);

        a.fill_uniform(base2k, &mut source);
        let mut b = vec![0i64; size];
        for x in &mut b {
            *x = source.next_i64();
        }

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.cnv_by_const_apply_tmp_bytes(0, c_size, size, size));
        move || {
            module.cnv_by_const_apply(
                0,
                &mut c_big,
                0,
                &vec_znx_backend_ref_from_ref::<BE>(&a.to_ref()),
                0,
                &b,
                &mut scratch.borrow(),
            );
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let log_n: usize = sweep[0];
        let size: usize = sweep[1];
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x{}", 1 << log_n, size));
        let mut runner = runner(1 << log_n, size);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}
