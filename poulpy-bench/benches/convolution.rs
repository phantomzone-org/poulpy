use criterion::{Criterion, criterion_group, criterion_main};

fn bench_cnv_prepare_left(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_hal::bench_suite::convolution::bench_cnv_prepare_left; c);
}
fn bench_cnv_prepare_right(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_hal::bench_suite::convolution::bench_cnv_prepare_right; c);
}
fn bench_cnv_apply_dft(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_hal::bench_suite::convolution::bench_cnv_apply_dft; c);
}
fn bench_cnv_pairwise_apply_dft(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_hal::bench_suite::convolution::bench_cnv_pairwise_apply_dft; c);
}
fn bench_cnv_by_const_apply(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_hal::bench_suite::convolution::bench_cnv_by_const_apply; c);
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
