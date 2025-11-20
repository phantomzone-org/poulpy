// poulpy-backend/benches/vec_znx_add.rs
use criterion::{Criterion, criterion_group, criterion_main};
#[cfg(target_arch = "x86_64")]
use poulpy_cpu_avx::FFT64Avx;
#[cfg(not(target_arch = "x86_64"))]
use poulpy_cpu_ref::FFT64Ref;

use poulpy_hal::bench_suite::vmp::bench_vmp_apply_dft_to_dft;

#[cfg(target_arch = "x86_64")]
fn bench_vmp_apply_dft_to_dft_cpu_avx_fft64(c: &mut Criterion) {
    bench_vmp_apply_dft_to_dft::<FFT64Avx>(c, "FFT64Avx");
}
#[cfg(not(target_arch = "x86_64"))]
fn bench_vmp_apply_dft_to_dft_cpu_ref_fft64(c: &mut Criterion) {
    bench_vmp_apply_dft_to_dft::<FFT64Ref>(c, "FFT64Ref");
}

#[cfg(target_arch = "x86_64")]
criterion_group!(benches_x86, bench_vmp_apply_dft_to_dft_cpu_avx_fft64,);
#[cfg(not(target_arch = "x86_64"))]
criterion_group!(benches_ref, bench_vmp_apply_dft_to_dft_cpu_ref_fft64,);

#[cfg(target_arch = "x86_64")]
criterion_main!(benches_x86);
#[cfg(not(target_arch = "x86_64"))]
criterion_main!(benches_ref);
