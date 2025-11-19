// poulpy-backend/benches/vec_znx_add.rs
use criterion::{Criterion, criterion_group, criterion_main};
use poulpy_cpu_avx::FFT64Avx;
use poulpy_hal::bench_suite::vmp::bench_vmp_apply_dft_to_dft;

fn bench_vmp_apply_dft_to_dft_cpu_avx_fft64(c: &mut Criterion) {
    bench_vmp_apply_dft_to_dft::<FFT64Avx>(c, "FFT64Avx");
}

criterion_group!(benches_x86, bench_vmp_apply_dft_to_dft_cpu_avx_fft64,);
criterion_main!(benches_x86);
