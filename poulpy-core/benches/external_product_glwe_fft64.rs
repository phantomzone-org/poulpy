use poulpy_core::layouts::{
    Base2K, Degree, Dnum, Dsize, GGSW, GGSWLayout, GLWE, GLWELayout, GLWESecret, Rank, TorusPrecision,
    prepared::{GGSWPrepared, GLWESecretPrepared},
};
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

#[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
pub use poulpy_cpu_avx::FFT64Avx as BackendImpl;

#[cfg(not(all(feature = "enable-avx", target_arch = "x86_64")))]
pub use poulpy_cpu_ref::FFT64Ref as BackendImpl;

use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Module, ScalarZnx, ScratchOwned},
    source::Source,
};

fn bench_external_product_glwe_fft64(c: &mut Criterion) {
    let mut group = c.benchmark_group("external_product_glwe_fft64");

    struct Params {
        log_n: usize,
        base2k: Base2K,
        k_ct_in: TorusPrecision,
        k_ct_out: TorusPrecision,
        k_ggsw: TorusPrecision,
        rank: Rank,
    }

    fn runner(p: Params) -> impl FnMut() {
        let module: Module<BackendImpl> = Module::<BackendImpl>::new(1 << p.log_n);

        let n: Degree = Degree(module.n() as u32);
        let base2k: Base2K = p.base2k;
        let k_ct_in: TorusPrecision = p.k_ct_in;
        let k_ct_out: TorusPrecision = p.k_ct_out;
        let k_ggsw: TorusPrecision = p.k_ggsw;
        let rank: Rank = p.rank;
        let dsize: Dsize = Dsize(1);

        let dnum: Dnum = Dnum(1); //(p.k_ct_in.div_ceil(p.base2k);

        let ggsw_layout: GGSWLayout = GGSWLayout {
            n,
            base2k,
            k: k_ggsw,
            dnum,
            dsize,
            rank,
        };

        let glwe_out_layout: GLWELayout = GLWELayout {
            n,
            base2k,
            k: k_ct_out,
            rank,
        };

        let glwe_in_layout: GLWELayout = GLWELayout {
            n,
            base2k,
            k: k_ct_in,
            rank,
        };

        let mut ct_ggsw: GGSW<Vec<u8>> = GGSW::alloc_from_infos(&ggsw_layout);
        let mut ct_glwe_in: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_in_layout);
        let mut ct_glwe_out: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_out_layout);
        let pt_rgsw: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n.into(), 1);

        let mut scratch: ScratchOwned<BackendImpl> = ScratchOwned::alloc(
            GGSW::encrypt_sk_tmp_bytes(&module, &ggsw_layout)
                | GLWE::encrypt_sk_tmp_bytes(&module, &glwe_in_layout)
                | GLWE::external_product_tmp_bytes(&module, &glwe_out_layout, &glwe_in_layout, &ggsw_layout),
        );

        let mut source_xs = Source::new([0u8; 32]);
        let mut source_xe = Source::new([0u8; 32]);
        let mut source_xa = Source::new([0u8; 32]);

        let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&glwe_in_layout);
        sk.fill_ternary_prob(0.5, &mut source_xs);

        let mut sk_dft: GLWESecretPrepared<Vec<u8>, BackendImpl> = GLWESecretPrepared::alloc(&module, rank);
        sk_dft.prepare(&module, &sk);

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

        let mut ggsw_prepared: GGSWPrepared<Vec<u8>, BackendImpl> = GGSWPrepared::alloc_from_infos(&module, &ct_ggsw);
        ggsw_prepared.prepare(&module, &ct_ggsw, scratch.borrow());

        move || {
            ct_glwe_out.external_product(&module, &ct_glwe_in, &ggsw_prepared, scratch.borrow());
            black_box(());
        }
    }

    let params_set: Vec<Params> = vec![Params {
        log_n: 11,
        base2k: 22_u32.into(),
        k_ct_in: 44_u32.into(),
        k_ct_out: 44_u32.into(),
        k_ggsw: 54_u32.into(),
        rank: 1_u32.into(),
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
        base2k: Base2K,
        k_ct: TorusPrecision,
        k_ggsw: TorusPrecision,
        rank: Rank,
    }

    fn runner(p: Params) -> impl FnMut() {
        let module: Module<BackendImpl> = Module::<BackendImpl>::new(1 << p.log_n);

        let n: Degree = Degree(module.n() as u32);
        let base2k: Base2K = p.base2k;
        let k_glwe: TorusPrecision = p.k_ct;
        let k_ggsw: TorusPrecision = p.k_ggsw;
        let rank: Rank = p.rank;
        let dsize: Dsize = Dsize(1);

        let dnum: Dnum = p.k_ct.div_ceil(p.base2k).into();

        let ggsw_layout: GGSWLayout = GGSWLayout {
            n,
            base2k,
            k: k_ggsw,
            dnum,
            dsize,
            rank,
        };

        let glwe_layout: GLWELayout = GLWELayout {
            n,
            base2k,
            k: k_glwe,
            rank,
        };

        let mut ct_ggsw: GGSW<Vec<u8>> = GGSW::alloc_from_infos(&ggsw_layout);
        let mut ct_glwe: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_layout);
        let pt_rgsw: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n.into(), 1);

        let mut scratch: ScratchOwned<BackendImpl> = ScratchOwned::alloc(
            GGSW::encrypt_sk_tmp_bytes(&module, &ggsw_layout)
                | GLWE::encrypt_sk_tmp_bytes(&module, &glwe_layout)
                | GLWE::external_product_tmp_bytes(&module, &glwe_layout, &glwe_layout, &ggsw_layout),
        );

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&glwe_layout);
        sk.fill_ternary_prob(0.5, &mut source_xs);

        let mut sk_dft: GLWESecretPrepared<Vec<u8>, BackendImpl> = GLWESecretPrepared::alloc(&module, rank);
        sk_dft.prepare(&module, &sk);

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

        let mut ggsw_prepared: GGSWPrepared<Vec<u8>, BackendImpl> = GGSWPrepared::alloc_from_infos(&module, &ct_ggsw);
        ggsw_prepared.prepare(&module, &ct_ggsw, scratch.borrow());
        move || {
            let scratch_borrow = scratch.borrow();
            ct_glwe.external_product_inplace(&module, &ggsw_prepared, scratch_borrow);
            black_box(());
        }
    }

    let params_set: Vec<Params> = vec![Params {
        log_n: 12,
        base2k: 18_u32.into(),
        k_ct: 54_u32.into(),
        k_ggsw: 54_u32.into(),
        rank: 1_u32.into(),
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
