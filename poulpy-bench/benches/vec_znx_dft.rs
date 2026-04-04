use criterion::{Criterion, criterion_group, criterion_main};

fn bench_vec_znx_dft_add(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx_dft::bench_vec_znx_dft_add, &poulpy_bench::params::BenchParams::get().hal; c);
}
fn bench_vec_znx_dft_add_inplace(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx_dft::bench_vec_znx_dft_add_inplace, &poulpy_bench::params::BenchParams::get().hal; c);
}
fn bench_vec_znx_dft_apply(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx_dft::bench_vec_znx_dft_apply, &poulpy_bench::params::BenchParams::get().hal; c);
}
fn bench_vec_znx_idft_apply(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx_dft::bench_vec_znx_idft_apply, &poulpy_bench::params::BenchParams::get().hal; c);
}
fn bench_vec_znx_idft_apply_tmpa(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx_dft::bench_vec_znx_idft_apply_tmpa, &poulpy_bench::params::BenchParams::get().hal; c);
}
fn bench_vec_znx_dft_sub(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx_dft::bench_vec_znx_dft_sub, &poulpy_bench::params::BenchParams::get().hal; c);
}
fn bench_vec_znx_dft_sub_inplace(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx_dft::bench_vec_znx_dft_sub_inplace, &poulpy_bench::params::BenchParams::get().hal; c);
}
fn bench_vec_znx_dft_sub_negate_inplace(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx_dft::bench_vec_znx_dft_sub_negate_inplace, &poulpy_bench::params::BenchParams::get().hal; c);
}

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = bench_vec_znx_dft_add,
    bench_vec_znx_dft_add_inplace,
    bench_vec_znx_dft_apply,
    bench_vec_znx_idft_apply,
    bench_vec_znx_idft_apply_tmpa,
    bench_vec_znx_dft_sub,
    bench_vec_znx_dft_sub_inplace,
    bench_vec_znx_dft_sub_negate_inplace
}
criterion_main!(benches);
