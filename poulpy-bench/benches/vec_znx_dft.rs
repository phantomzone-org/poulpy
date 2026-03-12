use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
pub use poulpy_cpu_avx::FFT64Avx as BackendImpl;

#[cfg(not(all(feature = "enable-avx", target_arch = "x86_64")))]
pub use poulpy_cpu_ref::FFT64Ref as BackendImpl;

fn bench_vec_znx_dft_add(c: &mut Criterion) {
    poulpy_hal::bench_suite::vec_znx_dft::bench_vec_znx_dft_add::<BackendImpl>(c, "fft64");
}
fn bench_vec_znx_dft_add_inplace(c: &mut Criterion) {
    poulpy_hal::bench_suite::vec_znx_dft::bench_vec_znx_dft_add_inplace::<BackendImpl>(c, "fft64");
}
fn bench_vec_znx_dft_apply(c: &mut Criterion) {
    poulpy_hal::bench_suite::vec_znx_dft::bench_vec_znx_dft_apply::<BackendImpl>(c, "fft64");
}
fn bench_vec_znx_idft_apply(c: &mut Criterion) {
    poulpy_hal::bench_suite::vec_znx_dft::bench_vec_znx_idft_apply::<BackendImpl>(c, "fft64");
}
fn bench_vec_znx_idft_apply_tmpa(c: &mut Criterion) {
    poulpy_hal::bench_suite::vec_znx_dft::bench_vec_znx_idft_apply_tmpa::<BackendImpl>(c, "fft64");
}
fn bench_vec_znx_dft_sub(c: &mut Criterion) {
    poulpy_hal::bench_suite::vec_znx_dft::bench_vec_znx_dft_sub::<BackendImpl>(c, "fft64");
}
fn bench_vec_znx_dft_sub_inplace(c: &mut Criterion) {
    poulpy_hal::bench_suite::vec_znx_dft::bench_vec_znx_dft_sub_inplace::<BackendImpl>(c, "fft64");
}
fn bench_vec_znx_dft_sub_negate_inplace(c: &mut Criterion) {
    poulpy_hal::bench_suite::vec_znx_dft::bench_vec_znx_dft_sub_negate_inplace::<BackendImpl>(c, "fft64");
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
