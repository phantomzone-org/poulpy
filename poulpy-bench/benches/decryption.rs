use criterion::{Criterion, criterion_group, criterion_main};
use poulpy_core::layouts::GLWELayout;

fn bench_glwe_decrypt(c: &mut Criterion) {
    let infos = GLWELayout {
        n: (1_u32 << 12).into(),
        base2k: 18_u32.into(),
        k: 54_u32.into(),
        rank: 1_u32.into(),
    };
    poulpy_bench::for_each_fft_backend!(
        poulpy_core::bench_suite::decryption::bench_glwe_decrypt,
        &infos;
        c
    );
}

criterion_group!(benches, bench_glwe_decrypt,);
criterion_main!(benches);
