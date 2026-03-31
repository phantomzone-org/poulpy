use criterion::{Criterion, criterion_group, criterion_main};
use poulpy_core::layouts::GLWELayout;

fn bench_glwe_decrypt(c: &mut Criterion) {
    let p = &poulpy_bench::params::BenchParams::get().core;
    let infos = GLWELayout {
        n: p.n.into(),
        base2k: p.base2k.into(),
        k: p.k.into(),
        rank: p.rank.into(),
    };
    poulpy_bench::for_each_backend!(
        poulpy_bench::bench_suite::core::decryption::bench_glwe_decrypt,
        &infos;
        c
    );
}

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = bench_glwe_decrypt
}
criterion_main!(benches);
