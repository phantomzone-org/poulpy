use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
pub use poulpy_cpu_avx::FFT64Avx as BackendImpl;

#[cfg(not(all(feature = "enable-avx", target_arch = "x86_64")))]
pub use poulpy_cpu_ref::FFT64Ref as BackendImpl;

fn bench_svp_prepare(c: &mut Criterion) {
    poulpy_hal::bench_suite::svp::bench_svp_prepare::<BackendImpl>(c, "fft64");
}
fn bench_svp_apply_dft(c: &mut Criterion) {
    poulpy_hal::bench_suite::svp::bench_svp_apply_dft::<BackendImpl>(c, "fft64");
}
fn bench_svp_apply_dft_to_dft(c: &mut Criterion) {
    poulpy_hal::bench_suite::svp::bench_svp_apply_dft_to_dft::<BackendImpl>(c, "fft64");
}
fn bench_svp_apply_dft_to_dft_inplace(c: &mut Criterion) {
    poulpy_hal::bench_suite::svp::bench_svp_apply_dft_to_dft_inplace::<BackendImpl>(c, "fft64");
}

criterion_group!(
    benches,
    bench_svp_prepare,
    bench_svp_apply_dft,
    bench_svp_apply_dft_to_dft,
    bench_svp_apply_dft_to_dft_inplace,
);
criterion_main!(benches);
