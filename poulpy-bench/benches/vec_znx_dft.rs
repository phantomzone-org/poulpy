use criterion::{Criterion, criterion_group, criterion_main};

fn bench_vec_znx_dft_add(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_hal::bench_suite::vec_znx_dft::bench_vec_znx_dft_add; c);
}
fn bench_vec_znx_dft_add_inplace(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_hal::bench_suite::vec_znx_dft::bench_vec_znx_dft_add_inplace; c);
}
fn bench_vec_znx_dft_apply(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_hal::bench_suite::vec_znx_dft::bench_vec_znx_dft_apply; c);
}
fn bench_vec_znx_idft_apply(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_hal::bench_suite::vec_znx_dft::bench_vec_znx_idft_apply; c);
}
fn bench_vec_znx_idft_apply_tmpa(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_hal::bench_suite::vec_znx_dft::bench_vec_znx_idft_apply_tmpa; c);
}
fn bench_vec_znx_dft_sub(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_hal::bench_suite::vec_znx_dft::bench_vec_znx_dft_sub; c);
}
fn bench_vec_znx_dft_sub_inplace(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_hal::bench_suite::vec_znx_dft::bench_vec_znx_dft_sub_inplace; c);
}
fn bench_vec_znx_dft_sub_negate_inplace(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_hal::bench_suite::vec_znx_dft::bench_vec_znx_dft_sub_negate_inplace; c);
}

criterion_group!(
    benches,
    bench_vec_znx_dft_add,
    bench_vec_znx_dft_add_inplace,
    bench_vec_znx_dft_apply,
    bench_vec_znx_idft_apply,
    bench_vec_znx_idft_apply_tmpa,
    bench_vec_znx_dft_sub,
    bench_vec_znx_dft_sub_inplace,
    bench_vec_znx_dft_sub_negate_inplace,
);
criterion_main!(benches);
