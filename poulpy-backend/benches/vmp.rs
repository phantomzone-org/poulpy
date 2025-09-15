// poulpy-backend/benches/vec_znx_add.rs
use criterion::{Criterion, criterion_group, criterion_main};
use poulpy_backend::{cpu_fft64_ref, cpu_spqlios};
use poulpy_hal::reference::fft64::vmp::bench_vmp_apply_dft_to_dft;

fn bench_vmp_apply_dft_to_dft_cpu_spqlios_fft64(c: &mut Criterion) {
    bench_vmp_apply_dft_to_dft::<cpu_spqlios::FFT64Spqlios>(c, "cpu_spqlios::fft64");
}

fn bench_vmp_apply_dft_to_dft_cpu_ref_fft64(c: &mut Criterion) {
    bench_vmp_apply_dft_to_dft::<cpu_fft64_ref::FFT64Ref>(c, "cpu_ref::fft64");
}

criterion_group!(
    benches,
    bench_vmp_apply_dft_to_dft_cpu_spqlios_fft64,
    bench_vmp_apply_dft_to_dft_cpu_ref_fft64
);
criterion_main!(benches);
