use criterion::{Criterion, criterion_group, criterion_main};
use poulpy_core::layouts::{Base2K, Degree, Dnum, Dsize, GLWELayout, GLWESwitchingKeyLayout, Rank, TorusPrecision};

fn bench_glwe_keyswitch(c: &mut Criterion) {
    let n: u32 = 1 << 16;
    let base2k: u32 = 52;
    let dsize: u32 = 6;
    let k: u32 = 1800 - dsize * base2k;
    let dnum: u32 = k.div_ceil(dsize * base2k);
    let k_ksk: u32 = k + dsize * base2k;

    let glwe_in = GLWELayout {
        n: Degree(n),
        base2k: Base2K(base2k),
        k: TorusPrecision(k),
        rank: Rank(1),
    };
    let glwe_out = GLWELayout {
        n: Degree(n),
        base2k: Base2K(base2k),
        k: TorusPrecision(k),
        rank: Rank(1),
    };
    let gglwe = GLWESwitchingKeyLayout {
        n: Degree(n),
        base2k: Base2K(base2k),
        k: TorusPrecision(k_ksk),
        rank_in: Rank(1),
        rank_out: Rank(1),
        dnum: Dnum(dnum),
        dsize: Dsize(dsize),
    };

    poulpy_bench::for_each_ntt_backend!(
        poulpy_core::bench_suite::keyswitch::bench_glwe_keyswitch,
        &glwe_in, &glwe_out, &gglwe;
        c
    );
}

criterion_group!(benches, bench_glwe_keyswitch);
criterion_main!(benches);
