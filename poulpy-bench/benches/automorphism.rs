use criterion::{Criterion, criterion_group, criterion_main};
use poulpy_core::layouts::{Dnum, Dsize, GLWEAutomorphismKeyLayout, GLWELayout};

fn bench_glwe_automorphism(c: &mut Criterion) {
    let n: u32 = 1 << 12;
    let base2k: u32 = 18;
    let k_ct: u32 = 54;
    let k_atk: u32 = 54;
    let dsize = Dsize(1);
    let dnum = Dnum(k_ct.div_ceil(dsize.0 * base2k));

    let glwe_infos = GLWELayout {
        n: n.into(),
        base2k: base2k.into(),
        k: k_ct.into(),
        rank: 1_u32.into(),
    };
    let atk_infos = GLWEAutomorphismKeyLayout {
        n: n.into(),
        base2k: base2k.into(),
        k: k_atk.into(),
        rank: 1_u32.into(),
        dnum,
        dsize,
    };

    poulpy_bench::for_each_fft_backend!(
        poulpy_core::bench_suite::automorphism::bench_glwe_automorphism,
        &glwe_infos, &atk_infos, 3;
        c
    );
}

criterion_group!(benches, bench_glwe_automorphism,);
criterion_main!(benches);
