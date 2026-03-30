use poulpy_core::{
    GGSWEncryptSk, GLWEEncryptSk, GLWEExternalProduct, ScratchTakeCore,
    layouts::{
        GGSW, GGSWInfos, GLWE, GLWEInfos, GLWESecret, GLWESecretPreparedFactory,
        prepared::{GGSWPrepared, GGSWPreparedFactory, GLWESecretPrepared},
    },
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, ScalarZnx, Scratch, ScratchOwned},
    source::Source,
};
use std::{hint::black_box};

use criterion::Criterion;

pub fn bench_glwe_external_product<BE: Backend>(glwe_infos: &impl GLWEInfos, ggsw_infos: &impl GGSWInfos, c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE>
        + GLWEExternalProduct<BE>
        + GGSWEncryptSk<BE>
        + GGSWPreparedFactory<BE>
        + GLWEEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let n: usize = ggsw_infos.n().into();
    let module: Module<BE> = Module::<BE>::new(n as u64);

    let mut source_xs = Source::new([0u8; 32]);
    let mut source_xa = Source::new([1u8; 32]);
    let mut source_xe = Source::new([2u8; 32]);

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(ggsw_infos);
    sk.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc(&module, ggsw_infos.rank());
    sk_prepared.prepare(&module, &sk);

    let pt = ScalarZnx::alloc(n, 1);
    let mut ct_ggsw: GGSW<Vec<u8>> = GGSW::alloc_from_infos(ggsw_infos);
    let mut ct_glwe_in: GLWE<Vec<u8>> = GLWE::alloc_from_infos(glwe_infos);
    let mut ct_glwe_out: GLWE<Vec<u8>> = GLWE::alloc_from_infos(glwe_infos);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        GGSW::encrypt_sk_tmp_bytes(&module, ggsw_infos)
            | GLWE::encrypt_sk_tmp_bytes(&module, glwe_infos)
            | GLWE::external_product_tmp_bytes(&module, glwe_infos, glwe_infos, ggsw_infos),
    );

    ct_ggsw.encrypt_sk(&module, &pt, &sk_prepared, &mut source_xa, &mut source_xe, scratch.borrow());
    ct_glwe_in.encrypt_zero_sk(&module, &sk_prepared, &mut source_xa, &mut source_xe, scratch.borrow());

    let mut ggsw_prepared: GGSWPrepared<Vec<u8>, BE> = GGSWPrepared::alloc_from_infos(&module, &ct_ggsw);
    ggsw_prepared.prepare(&module, &ct_ggsw, scratch.borrow());

    let group_name = format!("glwe_external_product::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            ct_glwe_out.external_product(&module, &ct_glwe_in, &ggsw_prepared, scratch.borrow());
            black_box(());
        })
    });
    group.finish();
}

pub fn bench_glwe_external_product_inplace<BE: Backend>(infos: &(impl GGSWInfos + GLWEInfos), c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE>
        + GLWEExternalProduct<BE>
        + GGSWEncryptSk<BE>
        + GGSWPreparedFactory<BE>
        + GLWEEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let n: usize = infos.n().into();
    let module: Module<BE> = Module::<BE>::new(n as u64);

    let mut source_xs = Source::new([0u8; 32]);
    let mut source_xa = Source::new([1u8; 32]);
    let mut source_xe = Source::new([2u8; 32]);

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(infos);
    sk.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc(&module, infos.rank());
    sk_prepared.prepare(&module, &sk);

    let pt = ScalarZnx::alloc(n, 1);
    let mut ct_ggsw: GGSW<Vec<u8>> = GGSW::alloc_from_infos(infos);
    let mut ct_glwe: GLWE<Vec<u8>> = GLWE::alloc_from_infos(infos);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        GGSW::encrypt_sk_tmp_bytes(&module, infos)
            | GLWE::encrypt_sk_tmp_bytes(&module, infos)
            | GLWE::external_product_tmp_bytes(&module, infos, infos, infos),
    );

    ct_ggsw.encrypt_sk(&module, &pt, &sk_prepared, &mut source_xa, &mut source_xe, scratch.borrow());
    ct_glwe.encrypt_zero_sk(&module, &sk_prepared, &mut source_xa, &mut source_xe, scratch.borrow());

    let mut ggsw_prepared: GGSWPrepared<Vec<u8>, BE> = GGSWPrepared::alloc_from_infos(&module, &ct_ggsw);
    ggsw_prepared.prepare(&module, &ct_ggsw, scratch.borrow());

    let group_name = format!("glwe_external_product_inplace::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            ct_glwe.external_product_inplace(&module, &ggsw_prepared, scratch.borrow());
            black_box(());
        })
    });
    group.finish();
}
