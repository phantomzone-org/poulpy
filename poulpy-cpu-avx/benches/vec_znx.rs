#![cfg(target_arch = "x86_64")]
// poulpy-backend/benches/vec_znx_add.rs
use criterion::{Criterion, criterion_group, criterion_main};
use poulpy_cpu_avx::FFT64Avx;
use poulpy_hal::reference::vec_znx::{bench_vec_znx_add, bench_vec_znx_automorphism, bench_vec_znx_normalize_inplace};

#[allow(dead_code)]
fn bench_vec_znx_add_cpu_avx_fft64(c: &mut Criterion) {
    bench_vec_znx_add::<FFT64Avx>(c, "FFT64Avx");
}

#[allow(dead_code)]
fn bench_vec_znx_normalize_inplace_cpu_avx_fft64(c: &mut Criterion) {
    bench_vec_znx_normalize_inplace::<FFT64Avx>(c, "FFT64Avx");
}

fn bench_vec_znx_automorphism_cpu_avx_fft64(c: &mut Criterion) {
    bench_vec_znx_automorphism::<FFT64Avx>(c, "FFT64Avx");
}

criterion_group!(
    benches,
    bench_vec_znx_add_cpu_avx_fft64,
    bench_vec_znx_normalize_inplace_cpu_avx_fft64,
    bench_vec_znx_automorphism_cpu_avx_fft64,
);
criterion_main!(benches);
