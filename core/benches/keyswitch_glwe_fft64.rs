use backend::{FFT64, Module, ScratchOwned};
use core::{FourierGLWESecret, GLWEAutomorphismKey, GLWECiphertext, GLWESecret, GLWESwitchingKey, Infos};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use sampling::source::Source;
use std::{hint::black_box, time::Duration};

fn bench_keyswitch_glwe_fft64(c: &mut Criterion) {
    let mut group = c.benchmark_group("keyswitch_glwe_fft64");

    struct Params {
        log_n: usize,
        basek: usize,
        k_ct_in: usize,
        k_ct_out: usize,
        k_ksk: usize,
        digits: usize,
        rank_in: usize,
        rank_out: usize,
    }

    fn runner(p: Params) -> impl FnMut() {
        let module: Module<FFT64> = Module::<FFT64>::new(1 << p.log_n);

        let basek: usize = p.basek;
        let k_rlwe_in: usize = p.k_ct_in;
        let k_rlwe_out: usize = p.k_ct_out;
        let k_grlwe: usize = p.k_ksk;
        let rank_in: usize = p.rank_in;
        let rank_out: usize = p.rank_out;
        let digits: usize = p.digits;

        let rows: usize = (p.k_ct_in + (p.basek * digits) - 1) / (p.basek * digits);
        let sigma: f64 = 3.2;

        let mut ksk: GLWEAutomorphismKey<Vec<u8>, FFT64> =
            GLWEAutomorphismKey::alloc(&module, basek, k_grlwe, rows, digits, rank_out);
        let mut ct_in: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, k_rlwe_in, rank_in);
        let mut ct_out: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, k_rlwe_out, rank_out);

        let mut scratch = ScratchOwned::new(
            GLWESwitchingKey::encrypt_sk_scratch_space(&module, basek, ksk.k(), rank_out)
                | GLWECiphertext::encrypt_sk_scratch_space(&module, basek, ct_in.k())
                | GLWECiphertext::keyswitch_scratch_space(
                    &module,
                    basek,
                    ct_out.k(),
                    ct_in.k(),
                    ksk.k(),
                    digits,
                    rank_in,
                    rank_out,
                ),
        );

        let mut source_xs = Source::new([0u8; 32]);
        let mut source_xe = Source::new([0u8; 32]);
        let mut source_xa = Source::new([0u8; 32]);

        let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc(&module, rank_in);
        sk_in.fill_ternary_prob(0.5, &mut source_xs);
        let sk_in_dft: FourierGLWESecret<Vec<u8>, FFT64> = FourierGLWESecret::from(&module, &sk_in);

        let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc(&module, rank_out);
        sk_out.fill_ternary_prob(0.5, &mut source_xs);

        ksk.generate_from_sk(
            &module,
            -1,
            &sk_in,
            &mut source_xa,
            &mut source_xe,
            sigma,
            scratch.borrow(),
        );

        ct_in.encrypt_zero_sk(
            &module,
            &sk_in_dft,
            &mut source_xa,
            &mut source_xe,
            sigma,
            scratch.borrow(),
        );

        move || {
            black_box(ct_out.automorphism(&module, &ct_in, &ksk, scratch.borrow()));
        }
    }

    let digits: usize = 1;
    let basek: usize = 19;

    let params_set: Vec<Params> = vec![Params {
        log_n: 15,
        basek: basek,
        k_ct_in: 874 - digits * basek,
        k_ct_out: 874 - digits * basek,
        k_ksk: 874,
        digits: digits,
        rank_in: 1,
        rank_out: 1,
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
        basek: usize,
        k_ct: usize,
        k_ksk: usize,
        rank: usize,
    }

    fn runner(p: Params) -> impl FnMut() {
        let module: Module<FFT64> = Module::<FFT64>::new(1 << p.log_n);

        let basek: usize = p.basek;
        let k_ct: usize = p.k_ct;
        let k_ksk: usize = p.k_ksk;
        let rank: usize = p.rank;
        let digits: usize = 1;

        let rows: usize = (p.k_ct + p.basek - 1) / p.basek;
        let sigma: f64 = 3.2;

        let mut ksk: GLWESwitchingKey<Vec<u8>, FFT64> = GLWESwitchingKey::alloc(&module, basek, k_ksk, rows, digits, rank, rank);
        let mut ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, k_ct, rank);

        let mut scratch = ScratchOwned::new(
            GLWESwitchingKey::encrypt_sk_scratch_space(&module, basek, ksk.k(), rank)
                | GLWECiphertext::encrypt_sk_scratch_space(&module, basek, ct.k())
                | GLWECiphertext::keyswitch_inplace_scratch_space(&module, basek, ct.k(), ksk.k(), digits, rank),
        );

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc(&module, rank);
        sk_in.fill_ternary_prob(0.5, &mut source_xs);
        let sk_in_dft: FourierGLWESecret<Vec<u8>, FFT64> = FourierGLWESecret::from(&module, &sk_in);

        let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc(&module, rank);
        sk_out.fill_ternary_prob(0.5, &mut source_xs);
        let sk_out_dft: FourierGLWESecret<Vec<u8>, FFT64> = FourierGLWESecret::from(&module, &sk_out);

        ksk.generate_from_sk(
            &module,
            &sk_in,
            &sk_out_dft,
            &mut source_xa,
            &mut source_xe,
            sigma,
            scratch.borrow(),
        );

        ct.encrypt_zero_sk(
            &module,
            &sk_in_dft,
            &mut source_xa,
            &mut source_xe,
            sigma,
            scratch.borrow(),
        );

        move || {
            black_box(ct.keyswitch_inplace(&module, &ksk, scratch.borrow()));
        }
    }

    let params_set: Vec<Params> = vec![Params {
        log_n: 9,
        basek: 18,
        k_ct: 27,
        k_ksk: 27,
        rank: 1,
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
