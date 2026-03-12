use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
pub use poulpy_cpu_avx::FFT64Avx as BackendImpl;

#[cfg(not(all(feature = "enable-avx", target_arch = "x86_64")))]
pub use poulpy_cpu_ref::FFT64Ref as BackendImpl;

fn bench_cnv_prepare_left(c: &mut Criterion) {
    poulpy_hal::bench_suite::convolution::bench_cnv_prepare_left::<BackendImpl>(c, "fft64");
}
fn bench_cnv_prepare_right(c: &mut Criterion) {
    poulpy_hal::bench_suite::convolution::bench_cnv_prepare_right::<BackendImpl>(c, "fft64");
}
fn bench_cnv_apply_dft(c: &mut Criterion) {
    poulpy_hal::bench_suite::convolution::bench_cnv_apply_dft::<BackendImpl>(c, "fft64");
}
fn bench_cnv_pairwise_apply_dft(c: &mut Criterion) {
    poulpy_hal::bench_suite::convolution::bench_cnv_pairwise_apply_dft::<BackendImpl>(c, "fft64");
}
fn bench_cnv_by_const_apply(c: &mut Criterion) {
    poulpy_hal::bench_suite::convolution::bench_cnv_by_const_apply::<BackendImpl>(c, "fft64");
}

criterion_group!(
    benches,
    bench_cnv_prepare_left,
    bench_cnv_prepare_right,
    bench_cnv_apply_dft,
    bench_cnv_pairwise_apply_dft,
    bench_cnv_by_const_apply,
);
criterion_main!(benches);
