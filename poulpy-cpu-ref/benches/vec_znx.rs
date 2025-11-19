// poulpy-backend/benches/vec_znx_add.rs
use criterion::{Criterion, criterion_group, criterion_main};
use poulpy_cpu_ref::FFT64Ref;
use poulpy_hal::reference::vec_znx::{bench_vec_znx_add, bench_vec_znx_automorphism, bench_vec_znx_normalize_inplace};

#[allow(dead_code)]
fn bench_vec_znx_add_cpu_ref_fft64(c: &mut Criterion) {
    bench_vec_znx_add::<FFT64Ref>(c, "cpu_spqlios::fft64");
}

#[allow(dead_code)]
fn bench_vec_znx_normalize_inplace_cpu_ref_fft64(c: &mut Criterion) {
    bench_vec_znx_normalize_inplace::<FFT64Ref>(c, "cpu_ref::fft64");
}

fn bench_vec_znx_automorphism_cpu_ref_fft64(c: &mut Criterion) {
    bench_vec_znx_automorphism::<FFT64Ref>(c, "cpu_ref::fft64");
}

criterion_group!(
    benches,
    // bench_vec_znx_add_cpu_spqlios_fft64,
    // bench_vec_znx_add_cpu_ref_fft64,
    // bench_vec_znx_normalize_inplace_cpu_ref_fft64,
    // bench_vec_znx_normalize_inplace_cpu_spqlios_fft64,
    bench_vec_znx_automorphism_cpu_ref_fft64,
);
criterion_main!(benches);
