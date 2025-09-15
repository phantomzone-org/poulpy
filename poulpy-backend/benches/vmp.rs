// poulpy-backend/benches/vec_znx_add.rs
use criterion::{Criterion, criterion_group, criterion_main};
use poulpy_backend::{FFT64Avx, FFT64Ref, FFT64Spqlios};
use poulpy_hal::bench_suite::vmp::bench_vmp_apply_dft_to_dft;

fn bench_vmp_apply_dft_to_dft_cpu_spqlios_fft64(c: &mut Criterion) {
    bench_vmp_apply_dft_to_dft::<FFT64Spqlios>(c, "cpu_spqlios::fft64");
}

fn bench_vmp_apply_dft_to_dft_cpu_ref_fft64(c: &mut Criterion) {
    bench_vmp_apply_dft_to_dft::<FFT64Ref>(c, "cpu_ref::fft64");
}

fn bench_vmp_apply_dft_to_dft_cpu_avx_fft64(c: &mut Criterion) {
    bench_vmp_apply_dft_to_dft::<FFT64Avx>(c, "cpu_avx::fft64");
}

criterion_group!(
    benches,
    bench_vmp_apply_dft_to_dft_cpu_spqlios_fft64,
    bench_vmp_apply_dft_to_dft_cpu_ref_fft64,
    bench_vmp_apply_dft_to_dft_cpu_avx_fft64,
);
criterion_main!(benches);
