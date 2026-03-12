use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
pub use poulpy_cpu_avx::FFT64Avx as BackendImpl;

#[cfg(not(all(feature = "enable-avx", target_arch = "x86_64")))]
pub use poulpy_cpu_ref::FFT64Ref as BackendImpl;

fn bench_vmp_prepare(c: &mut Criterion) {
    poulpy_hal::bench_suite::vmp::bench_vmp_prepare::<BackendImpl>(c, "fft64");
}
fn bench_vmp_apply_dft(c: &mut Criterion) {
    poulpy_hal::bench_suite::vmp::bench_vmp_apply_dft::<BackendImpl>(c, "fft64");
}
fn bench_vmp_apply_dft_to_dft(c: &mut Criterion) {
    poulpy_hal::bench_suite::vmp::bench_vmp_apply_dft_to_dft::<BackendImpl>(c, "fft64");
}

criterion_group!(benches, bench_vmp_prepare, bench_vmp_apply_dft, bench_vmp_apply_dft_to_dft,);
criterion_main!(benches);
