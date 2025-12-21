use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(not(all(
    feature = "enable-avx",
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
)))]
fn bench_vec_znx_add_cpu_avx_fft64(_c: &mut Criterion) {
    eprintln!("Skipping: AVX IFft benchmark requires x86_64 + AVX2 + FMA");
}

#[cfg(all(
    feature = "enable-avx",
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
fn bench_vec_znx_add_cpu_avx_fft64(c: &mut Criterion) {
    use poulpy_cpu_avx::FFT64Avx;
    poulpy_hal::reference::vec_znx::bench_vec_znx_add::<FFT64Avx>(c, "FFT64Avx");
}

#[cfg(not(all(
    feature = "enable-avx",
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
)))]
fn bench_vec_znx_normalize_inplace_cpu_avx_fft64(_c: &mut Criterion) {
    eprintln!("Skipping: AVX IFft benchmark requires x86_64 + AVX2 + FMA");
}

#[cfg(all(
    feature = "enable-avx",
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
fn bench_vec_znx_normalize_inplace_cpu_avx_fft64(c: &mut Criterion) {
    use poulpy_cpu_avx::FFT64Avx;
    poulpy_hal::reference::vec_znx::bench_vec_znx_normalize_inplace::<FFT64Avx>(c, "FFT64Avx");
}

#[cfg(not(all(
    feature = "enable-avx",
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
)))]
fn bench_vec_znx_automorphism_cpu_avx_fft64(_c: &mut Criterion) {
    eprintln!("Skipping: AVX IFft benchmark requires x86_64 + AVX2 + FMA");
}

#[cfg(all(
    feature = "enable-avx",
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
fn bench_vec_znx_automorphism_cpu_avx_fft64(c: &mut Criterion) {
    use poulpy_cpu_avx::FFT64Avx;
    poulpy_hal::reference::vec_znx::bench_vec_znx_automorphism::<FFT64Avx>(c, "FFT64Avx");
}

criterion_group!(
    benches,
    bench_vec_znx_add_cpu_avx_fft64,
    bench_vec_znx_normalize_inplace_cpu_avx_fft64,
    bench_vec_znx_automorphism_cpu_avx_fft64,
);
criterion_main!(benches);
