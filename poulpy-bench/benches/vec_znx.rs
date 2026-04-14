use criterion::{Criterion, criterion_group, criterion_main};

fn bench_vec_znx_add_into(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx::bench_vec_znx_add_into; c);
}
fn bench_vec_znx_add_assign(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx::bench_vec_znx_add_assign; c);
}
fn bench_vec_znx_sub(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx::bench_vec_znx_sub; c);
}
fn bench_vec_znx_sub_inplace(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx::bench_vec_znx_sub_inplace; c);
}
fn bench_vec_znx_sub_negate_inplace(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx::bench_vec_znx_sub_negate_inplace; c);
}
fn bench_vec_znx_negate(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx::bench_vec_znx_negate; c);
}
fn bench_vec_znx_negate_inplace(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx::bench_vec_znx_negate_inplace; c);
}
fn bench_vec_znx_normalize(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx::bench_vec_znx_normalize; c);
}
fn bench_vec_znx_normalize_inplace(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx::bench_vec_znx_normalize_inplace; c);
}
fn bench_vec_znx_rotate(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx::bench_vec_znx_rotate; c);
}
fn bench_vec_znx_rotate_inplace(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx::bench_vec_znx_rotate_inplace; c);
}
fn bench_vec_znx_automorphism(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx::bench_vec_znx_automorphism; c);
}
fn bench_vec_znx_automorphism_inplace(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx::bench_vec_znx_automorphism_inplace; c);
}
fn bench_vec_znx_lsh(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx::bench_vec_znx_lsh; c);
}
fn bench_vec_znx_lsh_inplace(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx::bench_vec_znx_lsh_inplace; c);
}
fn bench_vec_znx_rsh(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx::bench_vec_znx_rsh; c);
}
fn bench_vec_znx_rsh_inplace(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx::bench_vec_znx_rsh_inplace; c);
}
fn bench_vec_znx_mul_xp_minus_one(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx::bench_vec_znx_mul_xp_minus_one; c);
}
fn bench_vec_znx_mul_xp_minus_one_inplace(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx::bench_vec_znx_mul_xp_minus_one_inplace; c);
}

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = bench_vec_znx_add_into,
    bench_vec_znx_add_assign,
    bench_vec_znx_sub,
    bench_vec_znx_sub_inplace,
    bench_vec_znx_sub_negate_inplace,
    bench_vec_znx_negate,
    bench_vec_znx_negate_inplace,
    bench_vec_znx_normalize,
    bench_vec_znx_normalize_inplace,
    bench_vec_znx_rotate,
    bench_vec_znx_rotate_inplace,
    bench_vec_znx_automorphism,
    bench_vec_znx_automorphism_inplace,
    bench_vec_znx_lsh,
    bench_vec_znx_lsh_inplace,
    bench_vec_znx_rsh,
    bench_vec_znx_rsh_inplace,
    bench_vec_znx_mul_xp_minus_one,
    bench_vec_znx_mul_xp_minus_one_inplace
}
criterion_main!(benches);
