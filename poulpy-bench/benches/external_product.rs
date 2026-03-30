use criterion::{Criterion, criterion_group, criterion_main};
use poulpy_core::layouts::{Dnum, Dsize, GGSWLayout, GLWELayout};

fn bench_glwe_external_product(c: &mut Criterion) {
    let n: u32 = 1 << 11;
    let base2k: u32 = 22;
    let dsize = Dsize(1);
    let dnum = Dnum(1);

    let glwe_infos = GLWELayout {
        n: n.into(),
        base2k: base2k.into(),
        k: 44_u32.into(),
        rank: 1_u32.into(),
    };
    let ggsw_infos = GGSWLayout {
        n: n.into(),
        base2k: base2k.into(),
        k: 54_u32.into(),
        rank: 1_u32.into(),
        dnum,
        dsize,
    };
    poulpy_bench::for_each_fft_backend!(
        poulpy_core::bench_suite::external_product::bench_glwe_external_product,
        &glwe_infos, &ggsw_infos;
        c
    );
}

fn bench_glwe_external_product_inplace(c: &mut Criterion) {
    let n: u32 = 1 << 12;
    let base2k: u32 = 18;
    let k: u32 = 54;
    let dsize = Dsize(1);
    let dnum = Dnum(k.div_ceil(dsize.0 * base2k));

    let infos = GGSWLayout {
        n: n.into(),
        base2k: base2k.into(),
        k: k.into(),
        rank: 1_u32.into(),
        dnum,
        dsize,
    };
    poulpy_bench::for_each_fft_backend!(
        poulpy_core::bench_suite::external_product::bench_glwe_external_product_inplace,
        &infos;
        c
    );
}

criterion_group!(benches, bench_glwe_external_product, bench_glwe_external_product_inplace,);
criterion_main!(benches);
