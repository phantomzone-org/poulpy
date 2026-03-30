use criterion::{Criterion, criterion_group, criterion_main};

fn bench_vmp_prepare(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_hal::bench_suite::vmp::bench_vmp_prepare; c);
}
fn bench_vmp_apply_dft(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_hal::bench_suite::vmp::bench_vmp_apply_dft; c);
}
fn bench_vmp_apply_dft_to_dft(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_hal::bench_suite::vmp::bench_vmp_apply_dft_to_dft; c);
}

criterion_group!(benches, bench_vmp_prepare, bench_vmp_apply_dft, bench_vmp_apply_dft_to_dft,);
criterion_main!(benches);
