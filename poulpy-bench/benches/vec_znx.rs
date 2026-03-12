use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
pub use poulpy_cpu_avx::FFT64Avx as BackendImpl;

#[cfg(not(all(feature = "enable-avx", target_arch = "x86_64")))]
pub use poulpy_cpu_ref::FFT64Ref as BackendImpl;

fn bench_vec_znx_add(c: &mut Criterion) {
    poulpy_hal::bench_suite::vec_znx::bench_vec_znx_add::<BackendImpl>(c, "fft64");
}
fn bench_vec_znx_add_inplace(c: &mut Criterion) {
    poulpy_hal::bench_suite::vec_znx::bench_vec_znx_add_inplace::<BackendImpl>(c, "fft64");
}
fn bench_vec_znx_sub(c: &mut Criterion) {
    poulpy_hal::bench_suite::vec_znx::bench_vec_znx_sub::<BackendImpl>(c, "fft64");
}
fn bench_vec_znx_sub_inplace(c: &mut Criterion) {
    poulpy_hal::bench_suite::vec_znx::bench_vec_znx_sub_inplace::<BackendImpl>(c, "fft64");
}
fn bench_vec_znx_sub_negate_inplace(c: &mut Criterion) {
    poulpy_hal::bench_suite::vec_znx::bench_vec_znx_sub_negate_inplace::<BackendImpl>(c, "fft64");
}
fn bench_vec_znx_negate(c: &mut Criterion) {
    poulpy_hal::bench_suite::vec_znx::bench_vec_znx_negate::<BackendImpl>(c, "fft64");
}
fn bench_vec_znx_negate_inplace(c: &mut Criterion) {
    poulpy_hal::bench_suite::vec_znx::bench_vec_znx_negate_inplace::<BackendImpl>(c, "fft64");
}
fn bench_vec_znx_normalize(c: &mut Criterion) {
    poulpy_hal::bench_suite::vec_znx::bench_vec_znx_normalize::<BackendImpl>(c, "fft64");
}
fn bench_vec_znx_normalize_inplace(c: &mut Criterion) {
    poulpy_hal::bench_suite::vec_znx::bench_vec_znx_normalize_inplace::<BackendImpl>(c, "fft64");
}
fn bench_vec_znx_rotate(c: &mut Criterion) {
    poulpy_hal::bench_suite::vec_znx::bench_vec_znx_rotate::<BackendImpl>(c, "fft64");
}
fn bench_vec_znx_rotate_inplace(c: &mut Criterion) {
    poulpy_hal::bench_suite::vec_znx::bench_vec_znx_rotate_inplace::<BackendImpl>(c, "fft64");
}
fn bench_vec_znx_automorphism(c: &mut Criterion) {
    poulpy_hal::bench_suite::vec_znx::bench_vec_znx_automorphism::<BackendImpl>(c, "fft64");
}
fn bench_vec_znx_automorphism_inplace(c: &mut Criterion) {
    poulpy_hal::bench_suite::vec_znx::bench_vec_znx_automorphism_inplace::<BackendImpl>(c, "fft64");
}
fn bench_vec_znx_lsh(c: &mut Criterion) {
    poulpy_hal::bench_suite::vec_znx::bench_vec_znx_lsh::<BackendImpl>(c, "fft64");
}
fn bench_vec_znx_lsh_inplace(c: &mut Criterion) {
    poulpy_hal::bench_suite::vec_znx::bench_vec_znx_lsh_inplace::<BackendImpl>(c, "fft64");
}
fn bench_vec_znx_rsh(c: &mut Criterion) {
    poulpy_hal::bench_suite::vec_znx::bench_vec_znx_rsh::<BackendImpl>(c, "fft64");
}
fn bench_vec_znx_rsh_inplace(c: &mut Criterion) {
    poulpy_hal::bench_suite::vec_znx::bench_vec_znx_rsh_inplace::<BackendImpl>(c, "fft64");
}
fn bench_vec_znx_mul_xp_minus_one(c: &mut Criterion) {
    poulpy_hal::bench_suite::vec_znx::bench_vec_znx_mul_xp_minus_one::<BackendImpl>(c, "fft64");
}
fn bench_vec_znx_mul_xp_minus_one_inplace(c: &mut Criterion) {
    poulpy_hal::bench_suite::vec_znx::bench_vec_znx_mul_xp_minus_one_inplace::<BackendImpl>(c, "fft64");
}

criterion_group!(
    benches,
    bench_vec_znx_add,
    bench_vec_znx_add_inplace,
    bench_vec_znx_sub,
    bench_vec_znx_sub_inplace,
    bench_vec_znx_sub_negate_inplace,
    bench_vec_znx_negate,
    bench_vec_znx_negate_inplace,
    bench_vec_znx_normalize,
    bench_vec_znx_normalize_inplace,
    bench_vec_znx_rotate,
    bench_vec_znx_rotate_inplace,
    bench_vec_znx_automorphism,
    bench_vec_znx_automorphism_inplace,
    bench_vec_znx_lsh,
    bench_vec_znx_lsh_inplace,
    bench_vec_znx_rsh,
    bench_vec_znx_rsh_inplace,
    bench_vec_znx_mul_xp_minus_one,
    bench_vec_znx_mul_xp_minus_one_inplace,
);
criterion_main!(benches);
