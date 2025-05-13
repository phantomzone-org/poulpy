use base2k::{FFT64, Module, ScalarZnxAlloc, ScratchOwned};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use rlwe::{
    elem::Infos,
    encryption::EncryptSkScratchSpace,
    external_product::{
        ExternalProduct, ExternalProductInplace, ExternalProductInplaceScratchSpace, ExternalProductScratchSpace,
    },
    ggsw_ciphertext::GGSWCiphertext,
    glwe::GLWECiphertext,
    keys::{SecretKey, SecretKeyFourier},
};
use sampling::source::Source;

fn bench_external_product_glwe_fft64(c: &mut Criterion) {
    let mut group = c.benchmark_group("external_product_glwe_fft64");

    struct Params {
        log_n: usize,
        basek: usize,
        k_rlwe_in: usize,
        k_rlwe_out: usize,
        k_rgsw: usize,
    }

    fn runner(p: Params) -> impl FnMut() {
        let module: Module<FFT64> = Module::<FFT64>::new(1 << p.log_n);

        let basek: usize = p.basek;
        let k_rlwe_in: usize = p.k_rlwe_in;
        let k_rlwe_out: usize = p.k_rlwe_out;
        let k_rgsw: usize = p.k_rgsw;

        let rows: usize = (p.k_rlwe_in + p.basek - 1) / p.basek;
        let sigma: f64 = 3.2;
        let bound: f64 = sigma * 6.0;

        let mut ct_rgsw: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::new(&module, basek, k_rgsw, rows);
        let mut ct_rlwe_in: GLWECiphertext<Vec<u8>> = GLWECiphertext::new(&module, basek, k_rlwe_in);
        let mut ct_rlwe_out: GLWECiphertext<Vec<u8>> = GLWECiphertext::new(&module, basek, k_rlwe_out);
        let pt_rgsw: base2k::ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);

        let mut scratch = ScratchOwned::new(
            GGSWCiphertext::encrypt_sk_scratch_space(&module, ct_rgsw.size())
                | GLWECiphertext::encrypt_sk_scratch_space(&module, ct_rlwe_in.size())
                | GLWECiphertext::external_product_scratch_space(
                    &module,
                    ct_rlwe_out.size(),
                    ct_rlwe_in.size(),
                    ct_rgsw.size(),
                ),
        );

        let mut source_xs = Source::new([0u8; 32]);
        let mut source_xe = Source::new([0u8; 32]);
        let mut source_xa = Source::new([0u8; 32]);

        let mut sk: SecretKey<Vec<u8>> = SecretKey::new(&module);
        sk.fill_ternary_prob(0.5, &mut source_xs);
        let mut sk_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::new(&module);
        sk_dft.dft(&module, &sk);

        ct_rgsw.encrypt_sk(
            &module,
            &pt_rgsw,
            &sk_dft,
            &mut source_xa,
            &mut source_xe,
            sigma,
            bound,
            scratch.borrow(),
        );

        ct_rlwe_in.encrypt_zero_sk(
            &module,
            &sk_dft,
            &mut source_xa,
            &mut source_xe,
            sigma,
            bound,
            scratch.borrow(),
        );

        move || {
            ct_rlwe_out.external_product(
                black_box(&module),
                black_box(&ct_rlwe_in),
                black_box(&ct_rgsw),
                black_box(scratch.borrow()),
            );
        }
    }

    let params_set: Vec<Params> = vec![Params {
        log_n: 10,
        basek: 7,
        k_rlwe_in: 27,
        k_rlwe_out: 27,
        k_rgsw: 27,
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
        k_rlwe: usize,
        k_rgsw: usize,
    }

    fn runner(p: Params) -> impl FnMut() {
        let module: Module<FFT64> = Module::<FFT64>::new(1 << p.log_n);

        let basek: usize = p.basek;
        let k_rlwe: usize = p.k_rlwe;
        let k_rgsw: usize = p.k_rgsw;

        let rows: usize = (p.k_rlwe + p.basek - 1) / p.basek;
        let sigma: f64 = 3.2;
        let bound: f64 = sigma * 6.0;

        let mut ct_rgsw: GGSWCiphertext<Vec<u8>, FFT64> = GGSWCiphertext::new(&module, basek, k_rgsw, rows);
        let mut ct_rlwe: GLWECiphertext<Vec<u8>> = GLWECiphertext::new(&module, basek, k_rlwe);
        let pt_rgsw: base2k::ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);

        let mut scratch = ScratchOwned::new(
            GGSWCiphertext::encrypt_sk_scratch_space(&module, ct_rgsw.size())
                | GLWECiphertext::encrypt_sk_scratch_space(&module, ct_rlwe.size())
                | GLWECiphertext::external_product_inplace_scratch_space(&module, ct_rlwe.size(), ct_rgsw.size()),
        );

        let mut source_xs = Source::new([0u8; 32]);
        let mut source_xe = Source::new([0u8; 32]);
        let mut source_xa = Source::new([0u8; 32]);

        let mut sk: SecretKey<Vec<u8>> = SecretKey::new(&module);
        sk.fill_ternary_prob(0.5, &mut source_xs);
        let mut sk_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::new(&module);
        sk_dft.dft(&module, &sk);

        ct_rgsw.encrypt_sk(
            &module,
            &pt_rgsw,
            &sk_dft,
            &mut source_xa,
            &mut source_xe,
            sigma,
            bound,
            scratch.borrow(),
        );

        ct_rlwe.encrypt_zero_sk(
            &module,
            &sk_dft,
            &mut source_xa,
            &mut source_xe,
            sigma,
            bound,
            scratch.borrow(),
        );

        move || {
            let scratch_borrow = scratch.borrow();
            (0..1374).for_each(|i| {
                ct_rlwe.external_product_inplace(
                    black_box(&module),
                    black_box(&ct_rgsw),
                    black_box(scratch_borrow),
                );
            });
        }
    }

    let params_set: Vec<Params> = vec![Params {
        log_n: 9,
        basek: 18,
        k_rlwe: 27,
        k_rgsw: 27,
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
