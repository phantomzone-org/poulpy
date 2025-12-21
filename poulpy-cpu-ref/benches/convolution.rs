use criterion::{Criterion, criterion_group, criterion_main};
use poulpy_cpu_ref::FFT64Ref;
use poulpy_hal::bench_suite::convolution::{
    bench_cnv_apply_dft, bench_cnv_by_const_apply, bench_cnv_pairwise_apply_dft, bench_cnv_prepare_left, bench_cnv_prepare_right,
};

fn bench_cnv_prepare_left_cpu_ref_fft64(c: &mut Criterion) {
    bench_cnv_prepare_left::<FFT64Ref>(c, "cpu_ref::fft64");
}

fn bench_cnv_prepare_right_cpu_ref_fft64(c: &mut Criterion) {
    bench_cnv_prepare_right::<FFT64Ref>(c, "cpu_ref::fft64");
}

fn bench_bench_cnv_apply_dft_cpu_ref_fft64(c: &mut Criterion) {
    bench_cnv_apply_dft::<FFT64Ref>(c, "cpu_ref::fft64");
}

fn bench_bench_bench_cnv_pairwise_apply_dft_cpu_ref_fft64(c: &mut Criterion) {
    bench_cnv_pairwise_apply_dft::<FFT64Ref>(c, "cpu_ref::fft64");
}

fn bench_cnv_by_const_apply_cpu_ref_fft64(c: &mut Criterion) {
    bench_cnv_by_const_apply::<FFT64Ref>(c, "cpu_ref::fft64");
}

criterion_group!(
    benches,
    bench_cnv_prepare_left_cpu_ref_fft64,
    bench_cnv_prepare_right_cpu_ref_fft64,
    bench_bench_cnv_apply_dft_cpu_ref_fft64,
    bench_bench_bench_cnv_pairwise_apply_dft_cpu_ref_fft64,
    bench_cnv_by_const_apply_cpu_ref_fft64,
);
criterion_main!(benches);
