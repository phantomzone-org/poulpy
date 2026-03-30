use criterion::{Criterion, criterion_group, criterion_main};

fn bench_svp_prepare(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_hal::bench_suite::svp::bench_svp_prepare; c);
}
fn bench_svp_apply_dft(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_hal::bench_suite::svp::bench_svp_apply_dft; c);
}
fn bench_svp_apply_dft_to_dft(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_hal::bench_suite::svp::bench_svp_apply_dft_to_dft; c);
}
fn bench_svp_apply_dft_to_dft_inplace(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_hal::bench_suite::svp::bench_svp_apply_dft_to_dft_inplace; c);
}

criterion_group!(
    benches,
    bench_svp_prepare,
    bench_svp_apply_dft,
    bench_svp_apply_dft_to_dft,
    bench_svp_apply_dft_to_dft_inplace,
);
criterion_main!(benches);
