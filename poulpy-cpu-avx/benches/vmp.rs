use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(not(all(
    feature = "enable-avx",
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
)))]
fn bench_vmp_apply_dft_to_dft_cpu_avx_fft64(_c: &mut Criterion) {
    eprintln!("Skipping: AVX IFft benchmark requires x86_64 + AVX2 + FMA");
}

#[cfg(all(
    feature = "enable-avx",
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
fn bench_vmp_apply_dft_to_dft_cpu_avx_fft64(c: &mut Criterion) {
    use poulpy_cpu_avx::FFT64Avx;
    poulpy_hal::bench_suite::vmp::bench_vmp_apply_dft_to_dft::<FFT64Avx>(c, "FFT64Avx");
}

criterion_group!(benches_x86, bench_vmp_apply_dft_to_dft_cpu_avx_fft64,);
criterion_main!(benches_x86);
