use criterion::{Criterion, criterion_group, criterion_main};
use poulpy_core::layouts::{Dnum, Dsize, GGSWLayout, GLWEAutomorphismKeyLayout, GLWELayout};

fn bench_glwe_encrypt_sk(c: &mut Criterion) {
    let infos = GLWELayout {
        n: (1_u32 << 12).into(),
        base2k: 18_u32.into(),
        k: 54_u32.into(),
        rank: 1_u32.into(),
    };
    poulpy_bench::for_each_fft_backend!(
        poulpy_core::bench_suite::encryption::bench_glwe_encrypt_sk,
        &infos;
        c
    );
}

fn bench_ggsw_encrypt_sk(c: &mut Criterion) {
    let n: u32 = 1 << 11;
    let base2k: u32 = 22;
    let k: u32 = 44;
    let dsize = Dsize(1);
    let dnum = Dnum(2);
    let infos = GGSWLayout {
        n: n.into(),
        base2k: base2k.into(),
        k: k.into(),
        rank: 1_u32.into(),
        dnum,
        dsize,
    };
    poulpy_bench::for_each_fft_backend!(
        poulpy_core::bench_suite::encryption::bench_ggsw_encrypt_sk,
        &infos;
        c
    );
}

fn bench_glwe_automorphism_key_encrypt_sk(c: &mut Criterion) {
    let n: u32 = 1 << 12;
    let base2k: u32 = 18;
    let k_atk: u32 = 54;
    let dsize = Dsize(1);
    let dnum = Dnum(k_atk.div_ceil(dsize.0 * base2k));

    let atk_infos = GLWEAutomorphismKeyLayout {
        n: n.into(),
        base2k: base2k.into(),
        k: k_atk.into(),
        rank: 1_u32.into(),
        dnum,
        dsize,
    };

    poulpy_bench::for_each_fft_backend!(
        poulpy_core::bench_suite::encryption::bench_glwe_automorphism_key_encrypt_sk,
        &atk_infos, 3;
        c
    );
}

criterion_group!(
    benches,
    bench_glwe_encrypt_sk,
    bench_ggsw_encrypt_sk,
    bench_glwe_automorphism_key_encrypt_sk,
);
criterion_main!(benches);
