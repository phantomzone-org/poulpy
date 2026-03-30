use criterion::{Criterion, criterion_group, criterion_main};
use poulpy_core::layouts::GLWELayout;

const INFOS: GLWELayout = GLWELayout {
    n: poulpy_core::layouts::Degree(1 << 12),
    base2k: poulpy_core::layouts::Base2K(18),
    k: poulpy_core::layouts::TorusPrecision(54),
    rank: poulpy_core::layouts::Rank(1),
};

fn bench_glwe_add(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_core::bench_suite::operations::bench_glwe_add, &INFOS; c);
}
fn bench_glwe_add_inplace(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_core::bench_suite::operations::bench_glwe_add_inplace, &INFOS; c);
}
fn bench_glwe_sub(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_core::bench_suite::operations::bench_glwe_sub, &INFOS; c);
}
fn bench_glwe_sub_inplace(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_core::bench_suite::operations::bench_glwe_sub_inplace, &INFOS; c);
}
fn bench_glwe_normalize(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_core::bench_suite::operations::bench_glwe_normalize, &INFOS; c);
}
fn bench_glwe_normalize_inplace(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_core::bench_suite::operations::bench_glwe_normalize_inplace, &INFOS; c);
}
fn bench_glwe_mul_plain(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_core::bench_suite::operations::bench_glwe_mul_plain, &INFOS; c);
}
fn bench_glwe_mul_plain_inplace(c: &mut Criterion) {
    poulpy_bench::for_each_fft_backend!(poulpy_core::bench_suite::operations::bench_glwe_mul_plain_inplace, &INFOS; c);
}

criterion_group!(
    benches,
    bench_glwe_add,
    bench_glwe_add_inplace,
    bench_glwe_sub,
    bench_glwe_sub_inplace,
    bench_glwe_normalize,
    bench_glwe_normalize_inplace,
    bench_glwe_mul_plain,
    bench_glwe_mul_plain_inplace,
);
criterion_main!(benches);
