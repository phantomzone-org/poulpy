use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(not(all(feature = "enable-avx", target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
fn bench_cnv_prepare_left_cpu_avx_fft64(_c: &mut Criterion) {
    eprintln!("Skipping: AVX IFft benchmark requires x86_64 + AVX2 + FMA");
}

#[cfg(all(feature = "enable-avx", target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
fn bench_cnv_prepare_left_cpu_avx_fft64(c: &mut Criterion) {
    use poulpy_cpu_avx::FFT64Avx;
    poulpy_hal::bench_suite::convolution::bench_cnv_prepare_left::<FFT64Avx>(c, "cpu_avx::fft64");
}

#[cfg(not(all(feature = "enable-avx", target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
fn bench_cnv_prepare_right_cpu_avx_fft64(_c: &mut Criterion) {
    eprintln!("Skipping: AVX IFft benchmark requires x86_64 + AVX2 + FMA");
}

#[cfg(all(feature = "enable-avx", target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
fn bench_cnv_prepare_right_cpu_avx_fft64(c: &mut Criterion) {
    use poulpy_cpu_avx::FFT64Avx;
    poulpy_hal::bench_suite::convolution::bench_cnv_prepare_right::<FFT64Avx>(c, "cpu_avx::fft64");
}

#[cfg(not(all(feature = "enable-avx", target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
fn bench_bench_cnv_apply_dft_cpu_avx_fft64(_c: &mut Criterion) {
    eprintln!("Skipping: AVX IFft benchmark requires x86_64 + AVX2 + FMA");
}

#[cfg(all(feature = "enable-avx", target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
fn bench_bench_cnv_apply_dft_cpu_avx_fft64(c: &mut Criterion) {
    use poulpy_cpu_avx::FFT64Avx;
    poulpy_hal::bench_suite::convolution::bench_cnv_apply_dft::<FFT64Avx>(c, "cpu_avx::fft64");
}

#[cfg(not(all(feature = "enable-avx", target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
fn bench_bench_bench_cnv_pairwise_apply_dft_cpu_avx_fft64(_c: &mut Criterion) {
    eprintln!("Skipping: AVX IFft benchmark requires x86_64 + AVX2 + FMA");
}

#[cfg(all(feature = "enable-avx", target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
fn bench_bench_bench_cnv_pairwise_apply_dft_cpu_avx_fft64(c: &mut Criterion) {
    use poulpy_cpu_avx::FFT64Avx;
    poulpy_hal::bench_suite::convolution::bench_cnv_pairwise_apply_dft::<FFT64Avx>(c, "cpu_avx::fft64");
}

criterion_group!(
    benches,
    bench_cnv_prepare_left_cpu_avx_fft64,
    bench_cnv_prepare_right_cpu_avx_fft64,
    bench_bench_cnv_apply_dft_cpu_avx_fft64,
    bench_bench_bench_cnv_pairwise_apply_dft_cpu_avx_fft64
);
criterion_main!(benches);
