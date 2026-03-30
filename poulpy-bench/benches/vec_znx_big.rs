use criterion::{Criterion, criterion_group, criterion_main};

fn bench_vec_znx_big_add(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_hal::bench_suite::vec_znx_big::bench_vec_znx_big_add; c);
}
fn bench_vec_znx_big_add_inplace(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_hal::bench_suite::vec_znx_big::bench_vec_znx_big_add_inplace; c);
}
fn bench_vec_znx_big_add_small(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_hal::bench_suite::vec_znx_big::bench_vec_znx_big_add_small; c);
}
fn bench_vec_znx_big_add_small_inplace(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_hal::bench_suite::vec_znx_big::bench_vec_znx_big_add_small_inplace; c);
}
fn bench_vec_znx_big_automorphism(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_hal::bench_suite::vec_znx_big::bench_vec_znx_big_automorphism; c);
}
fn bench_vec_znx_automorphism_inplace(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_hal::bench_suite::vec_znx_big::bench_vec_znx_automorphism_inplace; c);
}
fn bench_vec_znx_big_negate(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_hal::bench_suite::vec_znx_big::bench_vec_znx_big_negate; c);
}
fn bench_vec_znx_big_negate_inplace(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_hal::bench_suite::vec_znx_big::bench_vec_znx_big_negate_inplace; c);
}
fn bench_vec_znx_normalize(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_hal::bench_suite::vec_znx_big::bench_vec_znx_normalize; c);
}
fn bench_vec_znx_big_sub(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_hal::bench_suite::vec_znx_big::bench_vec_znx_big_sub; c);
}
fn bench_vec_znx_big_sub_inplace(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_hal::bench_suite::vec_znx_big::bench_vec_znx_big_sub_inplace; c);
}
fn bench_vec_znx_big_sub_negate_inplace(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_hal::bench_suite::vec_znx_big::bench_vec_znx_big_sub_negate_inplace; c);
}
fn bench_vec_znx_big_sub_small_a(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_hal::bench_suite::vec_znx_big::bench_vec_znx_big_sub_small_a; c);
}
fn bench_vec_znx_big_sub_small_b(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_hal::bench_suite::vec_znx_big::bench_vec_znx_big_sub_small_b; c);
}

criterion_group!(
    benches,
    bench_vec_znx_big_add,
    bench_vec_znx_big_add_inplace,
    bench_vec_znx_big_add_small,
    bench_vec_znx_big_add_small_inplace,
    bench_vec_znx_big_automorphism,
    bench_vec_znx_automorphism_inplace,
    bench_vec_znx_big_negate,
    bench_vec_znx_big_negate_inplace,
    bench_vec_znx_normalize,
    bench_vec_znx_big_sub,
    bench_vec_znx_big_sub_inplace,
    bench_vec_znx_big_sub_negate_inplace,
    bench_vec_znx_big_sub_small_a,
    bench_vec_znx_big_sub_small_b,
);
criterion_main!(benches);
