use poulpy_core::layouts::{
    Base2K, Degree, Dnum, Dsize, GLWE, GLWEAutomorphismKey, GLWEAutomorphismKeyLayout, GLWELayout, GLWESecret, GLWESwitchingKey,
    GLWESwitchingKeyLayout, GLWESwitchingKeyPrepared, Rank, TorusPrecision,
    prepared::{GLWEAutomorphismKeyPrepared, GLWESecretPrepared},
};
use std::{hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use poulpy_cpu_ref::FFT64Ref;
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Module, ScratchOwned},
    source::Source,
};

fn bench_keyswitch_glwe_fft64(c: &mut Criterion) {
    let mut group = c.benchmark_group("keyswitch_glwe_fft64");

    struct Params {
        log_n: usize,
        base2k: Base2K,
        k_ct_in: TorusPrecision,
        k_ct_out: TorusPrecision,
        k_ksk: TorusPrecision,
        dsize: Dsize,
        rank: Rank,
    }

    fn runner(p: Params) -> impl FnMut() {
        let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << p.log_n);

        let n: Degree = Degree(module.n() as u32);
        let base2k: Base2K = p.base2k;
        let k_glwe_in: TorusPrecision = p.k_ct_in;
        let k_glwe_out: TorusPrecision = p.k_ct_out;
        let k_gglwe: TorusPrecision = p.k_ksk;
        let rank: Rank = p.rank;
        let dsize: Dsize = p.dsize;

        let dnum: Dnum = p.k_ct_in.div_ceil(p.base2k.0 * dsize.0).into();

        let gglwe_atk_layout: GLWEAutomorphismKeyLayout = GLWEAutomorphismKeyLayout {
            n,
            base2k,
            k: k_gglwe,
            dnum,
            rank,
            dsize,
        };

        let glwe_in_layout: GLWELayout = GLWELayout {
            n,
            base2k,
            k: k_glwe_in,
            rank,
        };

        let glwe_out_layout: GLWELayout = GLWELayout {
            n,
            base2k,
            k: k_glwe_out,
            rank,
        };

        let mut ksk: GLWEAutomorphismKey<Vec<u8>> = GLWEAutomorphismKey::alloc_from_infos(&gglwe_atk_layout);
        let mut ct_in: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_in_layout);
        let mut ct_out: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_out_layout);

        let mut scratch: ScratchOwned<FFT64Ref> = ScratchOwned::alloc(
            GLWESwitchingKey::encrypt_sk_tmp_bytes(&module, &gglwe_atk_layout)
                | GLWE::encrypt_sk_tmp_bytes(&module, &glwe_in_layout)
                | GLWE::keyswitch_tmp_bytes(
                    &module,
                    &glwe_out_layout,
                    &glwe_in_layout,
                    &gglwe_atk_layout,
                ),
        );

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&glwe_in_layout);
        sk_in.fill_ternary_prob(0.5, &mut source_xs);

        let mut sk_in_dft: GLWESecretPrepared<Vec<u8>, FFT64Ref> = GLWESecretPrepared::alloc(&module, rank);
        sk_in_dft.prepare(&module, &sk_in);

        ksk.encrypt_sk(
            &module,
            -1,
            &sk_in,
            &mut source_xa,
            &mut source_xe,
            scratch.borrow(),
        );

        ct_in.encrypt_zero_sk(
            &module,
            &sk_in_dft,
            &mut source_xa,
            &mut source_xe,
            scratch.borrow(),
        );

        let mut ksk_prepared: GLWEAutomorphismKeyPrepared<Vec<u8>, _> =
            GLWEAutomorphismKeyPrepared::alloc_from_infos(&module, &ksk);
        ksk_prepared.prepare(&module, &ksk, scratch.borrow());

        move || {
            ct_out.automorphism(&module, &ct_in, &ksk_prepared, scratch.borrow());
            black_box(());
        }
    }

    let base2k: usize = 19;
    let dsize = 1;

    let params_set: Vec<Params> = vec![Params {
        log_n: 15,
        base2k: base2k.into(),
        k_ct_in: (874 - dsize * base2k).into(),
        k_ct_out: (874 - dsize * base2k).into(),
        k_ksk: 874_u32.into(),
        dsize: 1_u32.into(),
        rank: 1_u32.into(),
    }];

    for params in params_set {
        let id = BenchmarkId::new("KEYSWITCH_GLWE_FFT64", "");
        let mut runner = runner(params);
        group.sample_size(500);
        group.measurement_time(Duration::from_secs(40));
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

fn bench_keyswitch_glwe_inplace_fft64(c: &mut Criterion) {
    let mut group = c.benchmark_group("keyswitch_glwe_inplace_fft64");

    struct Params {
        log_n: usize,
        base2k: Base2K,
        k_ct: TorusPrecision,
        k_ksk: TorusPrecision,
        rank: Rank,
    }

    fn runner(p: Params) -> impl FnMut() {
        let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << p.log_n);

        let n: Degree = Degree(module.n() as u32);
        let base2k: Base2K = p.base2k;
        let k_ct: TorusPrecision = p.k_ct;
        let k_ksk: TorusPrecision = p.k_ksk;
        let rank: Rank = p.rank;
        let dsize: Dsize = Dsize(1);

        let dnum: Dnum = p.k_ct.div_ceil(p.base2k).into();

        let gglwe_layout: GLWESwitchingKeyLayout = GLWESwitchingKeyLayout {
            n,
            base2k,
            k: k_ksk,
            dnum,
            dsize,
            rank_in: rank,
            rank_out: rank,
        };

        let glwe_layout: GLWELayout = GLWELayout {
            n,
            base2k,
            k: k_ct,
            rank,
        };

        let mut ksk: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc_from_infos(&gglwe_layout);
        let mut ct: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_layout);

        let mut scratch: ScratchOwned<FFT64Ref> = ScratchOwned::alloc(
            GLWESwitchingKey::encrypt_sk_tmp_bytes(&module, &gglwe_layout)
                | GLWE::encrypt_sk_tmp_bytes(&module, &glwe_layout)
                | GLWE::keyswitch_tmp_bytes(&module, &glwe_layout, &glwe_layout, &gglwe_layout),
        );

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&glwe_layout);
        sk_in.fill_ternary_prob(0.5, &mut source_xs);

        let mut sk_in_dft: GLWESecretPrepared<Vec<u8>, FFT64Ref> = GLWESecretPrepared::alloc(&module, rank);
        sk_in_dft.prepare(&module, &sk_in);

        let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&glwe_layout);
        sk_out.fill_ternary_prob(0.5, &mut source_xs);

        ksk.encrypt_sk(
            &module,
            &sk_in,
            &sk_out,
            &mut source_xa,
            &mut source_xe,
            scratch.borrow(),
        );

        ct.encrypt_zero_sk(
            &module,
            &sk_in_dft,
            &mut source_xa,
            &mut source_xe,
            scratch.borrow(),
        );

        let mut ksk_prepared: GLWESwitchingKeyPrepared<Vec<u8>, _> = GLWESwitchingKeyPrepared::alloc_from_infos(&module, &ksk);
        ksk_prepared.prepare(&module, &ksk, scratch.borrow());

        move || {
            ct.keyswitch_inplace(&module, &ksk_prepared, scratch.borrow());
            black_box(());
        }
    }

    let params_set: Vec<Params> = vec![Params {
        log_n: 9,
        base2k: 18_u32.into(),
        k_ct: 27_u32.into(),
        k_ksk: 27_u32.into(),
        rank: 1_u32.into(),
    }];

    for params in params_set {
        let id = BenchmarkId::new("KEYSWITCH_GLWE_INPLACE_FFT64", "");
        let mut runner = runner(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_keyswitch_glwe_fft64,
    bench_keyswitch_glwe_inplace_fft64
);
criterion_main!(benches);
