use backend::{FFT64, Module, ScalarZnx, ScalarZnxAlloc, ScratchOwned};
use core::{GGSWCiphertext, GLWECiphertext, GLWESecret, Infos};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use sampling::source::Source;
use std::hint::black_box;

fn bench_external_product_glwe_fft64(c: &mut Criterion) {
    let mut group = c.benchmark_group("external_product_glwe_fft64");

    struct Params {
        log_n: usize,
        basek: usize,
        k_ct_in: usize,
        k_ct_out: usize,
        k_ggsw: usize,
        rank: usize,
    }

    fn runner(p: Params) -> impl FnMut() {
        let module: Module<FFT64> = Module::<FFT64>::new(1 << p.log_n);

        let basek: usize = p.basek;
        let k_ct_in: usize = p.k_ct_in;
        let k_ct_out: usize = p.k_ct_out;
        let k_ggsw: usize = p.k_ggsw;
        let rank: usize = p.rank;
        let digits: usize = 1;

        let rows: usize = (p.k_ct_in + p.basek - 1) / p.basek;
        let sigma: f64 = 3.2;

        let mut ct_ggsw: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::alloc(&module, basek, k_ggsw, rows, digits, rank);
        let mut ct_glwe_in: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, k_ct_in, rank);
        let mut ct_glwe_out: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, k_ct_out, rank);
        let pt_rgsw: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);

        let mut scratch = ScratchOwned::new(
            GGSWCiphertext::encrypt_sk_scratch_space(&module, basek, ct_ggsw.k(), rank)
                | GLWECiphertext::encrypt_sk_scratch_space(&module, basek, ct_glwe_in.k())
                | GLWECiphertext::external_product_scratch_space(
                    &module,
                    basek,
                    ct_glwe_out.k(),
                    ct_glwe_in.k(),
                    ct_ggsw.k(),
                    digits,
                    rank,
                ),
        );

        let mut source_xs = Source::new([0u8; 32]);
        let mut source_xe = Source::new([0u8; 32]);
        let mut source_xa = Source::new([0u8; 32]);

        let mut sk: GLWESecret<Vec<u8>, FFT64> = GLWESecret::alloc(&module, rank);
        sk.fill_ternary_prob(&module, 0.5, &mut source_xs);

        ct_ggsw.encrypt_sk(
            &module,
            &pt_rgsw,
            &sk,
            &mut source_xa,
            &mut source_xe,
            sigma,
            scratch.borrow(),
        );

        ct_glwe_in.encrypt_zero_sk(
            &module,
            &sk,
            &mut source_xa,
            &mut source_xe,
            sigma,
            scratch.borrow(),
        );

        move || {
            ct_glwe_out.external_product(
                black_box(&module),
                black_box(&ct_glwe_in),
                black_box(&ct_ggsw),
                black_box(scratch.borrow()),
            );
        }
    }

    let params_set: Vec<Params> = vec![Params {
        log_n: 10,
        basek: 7,
        k_ct_in: 27,
        k_ct_out: 27,
        k_ggsw: 27,
        rank: 1,
    }];

    for params in params_set {
        let id = BenchmarkId::new("EXTERNAL_PRODUCT_GLWE_FFT64", "");
        let mut runner = runner(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

fn bench_external_product_glwe_inplace_fft64(c: &mut Criterion) {
    let mut group = c.benchmark_group("external_product_glwe_inplace_fft64");

    struct Params {
        log_n: usize,
        basek: usize,
        k_ct: usize,
        k_ggsw: usize,
        rank: usize,
    }

    fn runner(p: Params) -> impl FnMut() {
        let module: Module<FFT64> = Module::<FFT64>::new(1 << p.log_n);

        let basek: usize = p.basek;
        let k_glwe: usize = p.k_ct;
        let k_ggsw: usize = p.k_ggsw;
        let rank: usize = p.rank;
        let digits: usize = 1;

        let rows: usize = (p.k_ct + p.basek - 1) / p.basek;
        let sigma: f64 = 3.2;

        let mut ct_ggsw: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::alloc(&module, basek, k_ggsw, rows, digits, rank);
        let mut ct_glwe: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&module, basek, k_glwe, rank);
        let pt_rgsw: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);

        let mut scratch = ScratchOwned::new(
            GGSWCiphertext::encrypt_sk_scratch_space(&module, basek, ct_ggsw.k(), rank)
                | GLWECiphertext::encrypt_sk_scratch_space(&module, basek, ct_glwe.k())
                | GLWECiphertext::external_product_inplace_scratch_space(&module, basek, ct_glwe.k(), ct_ggsw.k(), digits, rank),
        );

        let mut source_xs = Source::new([0u8; 32]);
        let mut source_xe = Source::new([0u8; 32]);
        let mut source_xa = Source::new([0u8; 32]);

        let mut sk: GLWESecret<Vec<u8>, FFT64> = GLWESecret::alloc(&module, rank);
        sk.fill_ternary_prob(&module, 0.5, &mut source_xs);

        ct_ggsw.encrypt_sk(
            &module,
            &pt_rgsw,
            &sk,
            &mut source_xa,
            &mut source_xe,
            sigma,
            scratch.borrow(),
        );

        ct_glwe.encrypt_zero_sk(
            &module,
            &sk,
            &mut source_xa,
            &mut source_xe,
            sigma,
            scratch.borrow(),
        );

        move || {
            let scratch_borrow = scratch.borrow();
            (0..687).for_each(|_| {
                ct_glwe.external_product_inplace(
                    black_box(&module),
                    black_box(&ct_ggsw),
                    black_box(scratch_borrow),
                );
            });
        }
    }

    let params_set: Vec<Params> = vec![Params {
        log_n: 12,
        basek: 18,
        k_ct: 54,
        k_ggsw: 54,
        rank: 1,
    }];

    for params in params_set {
        let id = BenchmarkId::new("EXTERNAL_PRODUCT_GLWE_INPLACE_FFT64", "");
        let mut runner = runner(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_external_product_glwe_fft64,
    bench_external_product_glwe_inplace_fft64
);
criterion_main!(benches);
