use poulpy_core::layouts::{
    GGSWCiphertext, GLWECiphertext, GLWESecret, Infos,
    prepared::{GGSWCiphertextPrepared, GLWESecretPrepared, PrepareAlloc},
};
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use poulpy_backend::cpu_spqlios::FFT64Spqlios;
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Module, ScalarZnx, ScratchOwned},
    source::Source,
};

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
        let module: Module<FFT64Spqlios> = Module::<FFT64Spqlios>::new(1 << p.log_n);

        let n: usize = module.n();
        let basek: usize = p.basek;
        let k_ct_in: usize = p.k_ct_in;
        let k_ct_out: usize = p.k_ct_out;
        let k_ggsw: usize = p.k_ggsw;
        let rank: usize = p.rank;
        let digits: usize = 1;

        let rows: usize = 1; //(p.k_ct_in.div_ceil(p.basek);

        let mut ct_ggsw: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(n, basek, k_ggsw, rows, digits, rank);
        let mut ct_glwe_in: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(n, basek, k_ct_in, rank);
        let mut ct_glwe_out: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(n, basek, k_ct_out, rank);
        let pt_rgsw: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);

        let mut scratch: ScratchOwned<FFT64Spqlios> = ScratchOwned::alloc(
            GGSWCiphertext::encrypt_sk_scratch_space(&module, basek, ct_ggsw.k(), rank)
                | GLWECiphertext::encrypt_sk_scratch_space(&module, basek, ct_glwe_in.k())
                | GLWECiphertext::external_product_scratch_space(
                    &module,
                    basek,
                    ct_glwe_out.k(),
                    basek,
                    ct_glwe_in.k(),
                    basek,
                    ct_ggsw.k(),
                    digits,
                    rank,
                ),
        );

        let mut source_xs = Source::new([0u8; 32]);
        let mut source_xe = Source::new([0u8; 32]);
        let mut source_xa = Source::new([0u8; 32]);

        let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
        sk.fill_ternary_prob(0.5, &mut source_xs);
        let sk_dft: GLWESecretPrepared<Vec<u8>, FFT64Spqlios> = sk.prepare_alloc(&module, scratch.borrow());

        ct_ggsw.encrypt_sk(
            &module,
            &pt_rgsw,
            &sk_dft,
            &mut source_xa,
            &mut source_xe,
            scratch.borrow(),
        );

        ct_glwe_in.encrypt_zero_sk(
            &module,
            &sk_dft,
            &mut source_xa,
            &mut source_xe,
            scratch.borrow(),
        );

        let ggsw_prepared: GGSWCiphertextPrepared<Vec<u8>, FFT64Spqlios> = ct_ggsw.prepare_alloc(&module, scratch.borrow());

        move || {
            ct_glwe_out.external_product(&module, &ct_glwe_in, &ggsw_prepared, scratch.borrow());
            black_box(());
        }
    }

    let params_set: Vec<Params> = vec![Params {
        log_n: 11,
        basek: 22,
        k_ct_in: 44,
        k_ct_out: 44,
        k_ggsw: 54,
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
        let module: Module<FFT64Spqlios> = Module::<FFT64Spqlios>::new(1 << p.log_n);

        let n = module.n();
        let basek: usize = p.basek;
        let k_glwe: usize = p.k_ct;
        let k_ggsw: usize = p.k_ggsw;
        let rank: usize = p.rank;
        let digits: usize = 1;

        let rows: usize = p.k_ct.div_ceil(p.basek);

        let mut ct_ggsw: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(n, basek, k_ggsw, rows, digits, rank);
        let mut ct_glwe: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(n, basek, k_glwe, rank);
        let pt_rgsw: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);

        let mut scratch: ScratchOwned<FFT64Spqlios> = ScratchOwned::alloc(
            GGSWCiphertext::encrypt_sk_scratch_space(&module, basek, ct_ggsw.k(), rank)
                | GLWECiphertext::encrypt_sk_scratch_space(&module, basek, ct_glwe.k())
                | GLWECiphertext::external_product_inplace_scratch_space(
                    &module,
                    basek,
                    ct_glwe.k(),
                    basek,
                    ct_ggsw.k(),
                    digits,
                    rank,
                ),
        );

        let mut source_xs = Source::new([0u8; 32]);
        let mut source_xe = Source::new([0u8; 32]);
        let mut source_xa = Source::new([0u8; 32]);

        let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(n, rank);
        sk.fill_ternary_prob(0.5, &mut source_xs);
        let sk_dft: GLWESecretPrepared<Vec<u8>, FFT64Spqlios> = sk.prepare_alloc(&module, scratch.borrow());

        ct_ggsw.encrypt_sk(
            &module,
            &pt_rgsw,
            &sk_dft,
            &mut source_xa,
            &mut source_xe,
            scratch.borrow(),
        );

        ct_glwe.encrypt_zero_sk(
            &module,
            &sk_dft,
            &mut source_xa,
            &mut source_xe,
            scratch.borrow(),
        );

        let ggsw_prepared: GGSWCiphertextPrepared<Vec<u8>, FFT64Spqlios> = ct_ggsw.prepare_alloc(&module, scratch.borrow());

        move || {
            let scratch_borrow = scratch.borrow();
            ct_glwe.external_product_inplace(&module, &ggsw_prepared, scratch_borrow);
            black_box(());
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
