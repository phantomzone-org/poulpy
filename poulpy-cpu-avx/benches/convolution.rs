use criterion::{Criterion, criterion_group, criterion_main};

//#[cfg(not(all(feature = "enable-avx", target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
// fn bench_convolution_v2_avx2_fma(_c: &mut Criterion) {
//    eprintln!("Skipping: AVX IFft benchmark requires x86_64 + AVX2 + FMA");
//}

//#[cfg(all(feature = "enable-avx", target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
pub fn bench_convolution_avx2_fma(c: &mut Criterion) {
    use criterion::BenchmarkId;
    use poulpy_cpu_avx::FFT64Avx;
    use poulpy_hal::{
        api::{CnvPVecAlloc, Convolution, ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxDftAlloc},
        layouts::{CnvPVecL, CnvPVecR, FillUniform, Module, ScratchOwned, VecZnx},
        source::Source,
    };
    use std::hint::black_box;

    let group_name: String = "convolution_avx".to_string();

    let mut group = c.benchmark_group(group_name);

    fn runner(n: usize) -> impl FnMut() {
        let mut source: Source = Source::new([0u8; 32]);

        let base2k: usize = 12;

        let module: Module<FFT64Avx> = Module::<FFT64Avx>::new(n as u64);

        let a_size: usize = 15;
        let b_size: usize = 15;
        let c_size: usize = a_size + b_size - 1;

        let mut a_prep: CnvPVecL<Vec<u8>, FFT64Avx> = module.cnv_pvec_left_alloc(1, a_size);
        let mut b_prep: CnvPVecR<Vec<u8>, FFT64Avx> = module.cnv_pvec_right_alloc(1, b_size);

        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), 1, a_size);
        let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), 1, a_size);
        let mut c_dft = module.vec_znx_dft_alloc(1, c_size);

        a.fill_uniform(base2k, &mut source);
        b.fill_uniform(base2k, &mut source);

        let mut scratch: ScratchOwned<FFT64Avx> = ScratchOwned::alloc(
            module
                .cnv_apply_dft_tmp_bytes(c_size, 0, a_size, b_size)
                .max(module.cnv_prepare_left_tmp_bytes(c_size, a_size))
                .max(module.cnv_prepare_right_tmp_bytes(c_size, b_size)),
        );

        move || {
            module.cnv_prepare_left(&mut a_prep, &a, scratch.borrow());
            module.cnv_prepare_right(&mut b_prep, &b, scratch.borrow());
            module.cnv_apply_dft(&mut c_dft, 0, 0, &a_prep, 0, &b_prep, 0, scratch.borrow());
            black_box(());
        }
    }

    for log_n in [12] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("n: {}", 1 << log_n));
        let mut runner = runner(1 << log_n);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

criterion_group!(benches, bench_convolution_avx2_fma);
criterion_main!(benches);
