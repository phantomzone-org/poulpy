use poulpy_core::layouts::{GLWELayout, GLWESwitchingKey, GLWESwitchingKeyLayout};

use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
pub use poulpy_cpu_avx::NTT120Avx as BackendImpl;

#[cfg(not(all(feature = "enable-avx", target_arch = "x86_64")))]
pub use poulpy_cpu_ref::NTT120Ref as BackendImpl;

fn bench_keyswitch_glwe(c: &mut Criterion) {
    let n_in: usize = 1 << 16;
    let n_out: usize = 1 << 16;
    let rank_in: usize = 1;
    let rank_out: usize = 1;
    let base2k_in: usize = 52;
    let base2k_out: usize = 52;
    let base2k_ksk: usize = 52;

    let dsize: usize = 6;

    let k_in: usize = 1800 - dsize * base2k_in;
    let k_out: usize = 1800 - dsize * base2k_in;

    let dnum: usize = k_in.div_ceil(dsize * base2k_ksk);

    let k_ksk: usize = k_in + dsize * base2k_ksk;

    let glwe_in: GLWELayout = GLWELayout {
        n: n_in.into(),
        base2k: base2k_in.into(),
        k: k_in.into(),
        rank: rank_in.into(),
    };

    let glwe_out: GLWELayout = GLWELayout {
        n: n_out.into(),
        base2k: base2k_out.into(),
        k: k_out.into(),
        rank: rank_out.into(),
    };

    let gglwe: GLWESwitchingKeyLayout = GLWESwitchingKeyLayout {
        n: n_in.into(),
        base2k: base2k_ksk.into(),
        k: k_ksk.into(),
        rank_in: rank_in.into(),
        rank_out: rank_out.into(),
        dnum: dnum.into(),
        dsize: dsize.into(),
    };

    println!("gglwe_bytes: {}", GLWESwitchingKey::bytes_of_from_infos(&gglwe));

    poulpy_core::bench_suite::keyswitch::gglwe::bench_keyswitch_glwe::<BackendImpl, _, _, _>(
        &glwe_in, &glwe_out, &gglwe, c, "ntt120",
    );
}

criterion_group!(benches, bench_keyswitch_glwe);
criterion_main!(benches);
