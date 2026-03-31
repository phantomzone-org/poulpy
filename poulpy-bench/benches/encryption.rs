use criterion::{Criterion, criterion_group, criterion_main};
use poulpy_core::layouts::{Dnum, Dsize, GGSWLayout, GLWEAutomorphismKeyLayout, GLWELayout};

fn bench_glwe_encrypt_sk(c: &mut Criterion) {
    let p = &poulpy_bench::params::BenchParams::get().core;
    let n = p.n;
    let base2k = p.base2k;
    let k = p.k;
    let infos = GLWELayout {
        n: n.into(),
        base2k: base2k.into(),
        k: k.into(),
        rank: p.rank.into(),
    };
    poulpy_bench::for_each_backend!(
        poulpy_bench::bench_suite::core::encryption::bench_glwe_encrypt_sk,
        &infos;
        c
    );
}

fn bench_ggsw_encrypt_sk(c: &mut Criterion) {
    let p = &poulpy_bench::params::BenchParams::get().core;
    let n = p.n;
    let base2k = p.base2k;
    let k = p.k;
    let dsize = Dsize(p.dsize);
    let dnum = Dnum(p.dnum());
    let infos = GGSWLayout {
        n: n.into(),
        base2k: base2k.into(),
        k: k.into(),
        rank: p.rank.into(),
        dnum,
        dsize,
    };
    poulpy_bench::for_each_backend!(
        poulpy_bench::bench_suite::core::encryption::bench_ggsw_encrypt_sk,
        &infos;
        c
    );
}

fn bench_glwe_automorphism_key_encrypt_sk(c: &mut Criterion) {
    let p = &poulpy_bench::params::BenchParams::get().core;
    let n = p.n;
    let base2k = p.base2k;
    let k_atk = p.k;
    let dsize = Dsize(p.dsize);
    let dnum = Dnum(p.dnum());

    let atk_infos = GLWEAutomorphismKeyLayout {
        n: n.into(),
        base2k: base2k.into(),
        k: k_atk.into(),
        rank: p.rank.into(),
        dnum,
        dsize,
    };

    poulpy_bench::for_each_backend!(
        poulpy_bench::bench_suite::core::encryption::bench_glwe_automorphism_key_encrypt_sk,
        &atk_infos, 3;
        c
    );
}

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = bench_glwe_encrypt_sk,
    bench_ggsw_encrypt_sk,
    bench_glwe_automorphism_key_encrypt_sk
}
criterion_main!(benches);
