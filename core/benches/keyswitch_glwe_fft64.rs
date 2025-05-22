use backend::{FFT64, Module, ScratchOwned};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use core::{
    elem::Infos,
    glwe_ciphertext::GLWECiphertext,
    keys::{SecretKey, SecretKeyFourier},
    keyswitch_key::GLWESwitchingKey,
};
use sampling::source::Source;

fn bench_keyswitch_glwe_fft64(c: &mut Criterion) {
    let mut group = c.benchmark_group("keyswitch_glwe_fft64");

    struct Params {
        log_n: usize,
        basek: usize,
        k_ct_in: usize,
        k_ct_out: usize,
        k_ksk: usize,
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

        let rows: usize = (p.k_ct_in + p.basek - 1) / p.basek;
        let sigma: f64 = 3.2;

        let mut ksk: GLWESwitchingKey<Vec<u8>, FFT64> = GLWESwitchingKey::alloc(&module, basek, k_grlwe, rows, rank_in, rank_out);
        let mut ct_in: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, k_rlwe_in, rank_in);
        let mut ct_out: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, k_rlwe_out, rank_out);

        let mut scratch = ScratchOwned::new(
            GLWESwitchingKey::encrypt_sk_scratch_space(&module, rank_out, ksk.size())
                | GLWECiphertext::encrypt_sk_scratch_space(&module, ct_in.size())
                | GLWECiphertext::keyswitch_scratch_space(
                    &module,
                    ct_out.size(),
                    ct_in.size(),
                    ksk.size(),
                    rank_in,
                    rank_out,
                ),
        );

        let mut source_xs = Source::new([0u8; 32]);
        let mut source_xe = Source::new([0u8; 32]);
        let mut source_xa = Source::new([0u8; 32]);

        let mut sk_in: SecretKey<Vec<u8>> = SecretKey::alloc(&module, rank_in);
        sk_in.fill_ternary_prob(0.5, &mut source_xs);
        let mut sk_in_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::alloc(&module, rank_in);
        sk_in_dft.dft(&module, &sk_in);

        let mut sk_out: SecretKey<Vec<u8>> = SecretKey::alloc(&module, rank_out);
        sk_out.fill_ternary_prob(0.5, &mut source_xs);
        let mut sk_out_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::alloc(&module, rank_out);
        sk_out_dft.dft(&module, &sk_out);

        ksk.encrypt_sk(
            &module,
            &sk_in,
            &sk_out_dft,
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
            ct_out.keyswitch(
                black_box(&module),
                black_box(&ct_in),
                black_box(&ksk),
                black_box(scratch.borrow()),
            );
        }
    }

    let params_set: Vec<Params> = vec![Params {
        log_n: 16,
        basek: 50,
        k_ct_in: 1250,
        k_ct_out: 1250,
        k_ksk: 1250 + 66,
        rank_in: 1,
        rank_out: 1,
    }];

    for params in params_set {
        let id = BenchmarkId::new("KEYSWITCH_GLWE_FFT64", "");
        let mut runner = runner(params);
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

        let rows: usize = (p.k_ct + p.basek - 1) / p.basek;
        let sigma: f64 = 3.2;

        let mut ksk: GLWESwitchingKey<Vec<u8>, FFT64> = GLWESwitchingKey::alloc(&module, basek, k_ksk, rows, rank, rank);
        let mut ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, k_ct, rank);

        let mut scratch = ScratchOwned::new(
            GLWESwitchingKey::encrypt_sk_scratch_space(&module, rank, ksk.size())
                | GLWECiphertext::encrypt_sk_scratch_space(&module, ct.size())
                | GLWECiphertext::keyswitch_inplace_scratch_space(&module, ct.size(), ksk.size(), rank),
        );

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        let mut sk_in: SecretKey<Vec<u8>> = SecretKey::alloc(&module, rank);
        sk_in.fill_ternary_prob(0.5, &mut source_xs);
        let mut sk_in_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::alloc(&module, rank);
        sk_in_dft.dft(&module, &sk_in);

        let mut sk_out: SecretKey<Vec<u8>> = SecretKey::alloc(&module, rank);
        sk_out.fill_ternary_prob(0.5, &mut source_xs);
        let mut sk_out_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::alloc(&module, rank);
        sk_out_dft.dft(&module, &sk_out);

        ksk.encrypt_sk(
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
            ct.keyswitch_inplace(
                black_box(&module),
                black_box(&ksk),
                black_box(scratch.borrow()),
            );
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
