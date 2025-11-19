// poulpy-backend/benches/vec_znx_add.rs
use criterion::{Criterion, criterion_group, criterion_main};
use poulpy_cpu_ref::FFT64Ref;
use poulpy_hal::bench_suite::vmp::bench_vmp_apply_dft_to_dft;

fn bench_vmp_apply_dft_to_dft_cpu_ref_fft64(c: &mut Criterion) {
    bench_vmp_apply_dft_to_dft::<FFT64Ref>(c, "cpu_ref::fft64");
}

criterion_group!(benches, bench_vmp_apply_dft_to_dft_cpu_ref_fft64,);
criterion_main!(benches);
