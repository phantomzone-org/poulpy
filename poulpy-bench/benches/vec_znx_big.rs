use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
pub use poulpy_cpu_avx::FFT64Avx as BackendImpl;

#[cfg(not(all(feature = "enable-avx", target_arch = "x86_64")))]
pub use poulpy_cpu_ref::FFT64Ref as BackendImpl;

fn bench_vec_znx_big_add(c: &mut Criterion) {
    poulpy_hal::bench_suite::vec_znx_big::bench_vec_znx_big_add::<BackendImpl>(c, "fft64");
}
fn bench_vec_znx_big_add_inplace(c: &mut Criterion) {
    poulpy_hal::bench_suite::vec_znx_big::bench_vec_znx_big_add_inplace::<BackendImpl>(c, "fft64");
}
fn bench_vec_znx_big_add_small(c: &mut Criterion) {
    poulpy_hal::bench_suite::vec_znx_big::bench_vec_znx_big_add_small::<BackendImpl>(c, "fft64");
}
fn bench_vec_znx_big_add_small_inplace(c: &mut Criterion) {
    poulpy_hal::bench_suite::vec_znx_big::bench_vec_znx_big_add_small_inplace::<BackendImpl>(c, "fft64");
}
fn bench_vec_znx_big_automorphism(c: &mut Criterion) {
    poulpy_hal::bench_suite::vec_znx_big::bench_vec_znx_big_automorphism::<BackendImpl>(c, "fft64");
}
fn bench_vec_znx_automorphism_inplace(c: &mut Criterion) {
    poulpy_hal::bench_suite::vec_znx_big::bench_vec_znx_automorphism_inplace::<BackendImpl>(c, "fft64");
}
fn bench_vec_znx_big_negate(c: &mut Criterion) {
    poulpy_hal::bench_suite::vec_znx_big::bench_vec_znx_big_negate::<BackendImpl>(c, "fft64");
}
fn bench_vec_znx_big_negate_inplace(c: &mut Criterion) {
    poulpy_hal::bench_suite::vec_znx_big::bench_vec_znx_big_negate_inplace::<BackendImpl>(c, "fft64");
}
fn bench_vec_znx_normalize(c: &mut Criterion) {
    poulpy_hal::bench_suite::vec_znx_big::bench_vec_znx_normalize::<BackendImpl>(c, "fft64");
}
fn bench_vec_znx_big_sub(c: &mut Criterion) {
    poulpy_hal::bench_suite::vec_znx_big::bench_vec_znx_big_sub::<BackendImpl>(c, "fft64");
}
fn bench_vec_znx_big_sub_inplace(c: &mut Criterion) {
    poulpy_hal::bench_suite::vec_znx_big::bench_vec_znx_big_sub_inplace::<BackendImpl>(c, "fft64");
}
fn bench_vec_znx_big_sub_negate_inplace(c: &mut Criterion) {
    poulpy_hal::bench_suite::vec_znx_big::bench_vec_znx_big_sub_negate_inplace::<BackendImpl>(c, "fft64");
}
fn bench_vec_znx_big_sub_small_a(c: &mut Criterion) {
    poulpy_hal::bench_suite::vec_znx_big::bench_vec_znx_big_sub_small_a::<BackendImpl>(c, "fft64");
}
fn bench_vec_znx_big_sub_small_b(c: &mut Criterion) {
    poulpy_hal::bench_suite::vec_znx_big::bench_vec_znx_big_sub_small_b::<BackendImpl>(c, "fft64");
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
