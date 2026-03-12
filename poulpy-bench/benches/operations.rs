use criterion::{Criterion, criterion_group, criterion_main};
use poulpy_core::layouts::GLWELayout;

#[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
pub use poulpy_cpu_avx::FFT64Avx as BackendImpl;

#[cfg(not(all(feature = "enable-avx", target_arch = "x86_64")))]
pub use poulpy_cpu_ref::FFT64Ref as BackendImpl;

const INFOS: GLWELayout = GLWELayout {
    n: poulpy_core::layouts::Degree(1 << 12),
    base2k: poulpy_core::layouts::Base2K(18),
    k: poulpy_core::layouts::TorusPrecision(54),
    rank: poulpy_core::layouts::Rank(1),
};

fn bench_glwe_add(c: &mut Criterion) {
    poulpy_core::bench_suite::operations::bench_glwe_add::<BackendImpl, _>(&INFOS, c, "fft64");
}

fn bench_glwe_add_inplace(c: &mut Criterion) {
    poulpy_core::bench_suite::operations::bench_glwe_add_inplace::<BackendImpl, _>(&INFOS, c, "fft64");
}

fn bench_glwe_sub(c: &mut Criterion) {
    poulpy_core::bench_suite::operations::bench_glwe_sub::<BackendImpl, _>(&INFOS, c, "fft64");
}

fn bench_glwe_sub_inplace(c: &mut Criterion) {
    poulpy_core::bench_suite::operations::bench_glwe_sub_inplace::<BackendImpl, _>(&INFOS, c, "fft64");
}

fn bench_glwe_normalize(c: &mut Criterion) {
    poulpy_core::bench_suite::operations::bench_glwe_normalize::<BackendImpl, _>(&INFOS, c, "fft64");
}

fn bench_glwe_normalize_inplace(c: &mut Criterion) {
    poulpy_core::bench_suite::operations::bench_glwe_normalize_inplace::<BackendImpl, _>(&INFOS, c, "fft64");
}

fn bench_glwe_mul_plain(c: &mut Criterion) {
    poulpy_core::bench_suite::operations::bench_glwe_mul_plain::<BackendImpl, _>(&INFOS, c, "fft64");
}

fn bench_glwe_mul_plain_inplace(c: &mut Criterion) {
    poulpy_core::bench_suite::operations::bench_glwe_mul_plain_inplace::<BackendImpl, _>(&INFOS, c, "fft64");
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
