use poulpy_core::{
    DEFAULT_BOUND_XE, DEFAULT_SIGMA_XE, GGSWEncryptSk, GLWEAutomorphismKeyEncryptSk, GLWEEncryptSk, ScratchTakeCore,
    layouts::{
        GGLWEInfos, GGSW, GGSWInfos, GLWEAutomorphismKey, GLWEInfos, GLWESecret, GLWESecretPreparedFactory, LWEInfos,
        prepared::GLWESecretPrepared,
    },
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, NoiseInfos, ScalarZnx, Scratch, ScratchOwned},
    source::Source,
};
use std::hint::black_box;

use criterion::Criterion;

pub fn bench_glwe_encrypt_sk<BE: Backend>(infos: &impl GLWEInfos, c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE> + GLWEEncryptSk<BE> + GLWESecretPreparedFactory<BE>,
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

    let mut ct = poulpy_core::layouts::GLWE::alloc_from_infos(infos);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(poulpy_core::layouts::GLWE::encrypt_sk_tmp_bytes(&module, infos));

    let enc_infos = NoiseInfos::new(infos.max_k().as_usize(), DEFAULT_SIGMA_XE, DEFAULT_BOUND_XE).unwrap();

    let group_name = format!("glwe_encrypt_sk::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            ct.encrypt_zero_sk(&module, &sk_prepared, &enc_infos, &mut source_xe, &mut source_xa, scratch.borrow());
            black_box(());
        })
    });
    group.finish();
}

pub fn bench_ggsw_encrypt_sk<BE: Backend>(infos: &impl GGSWInfos, c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE> + GGSWEncryptSk<BE> + GLWESecretPreparedFactory<BE>,
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
    let mut ct = GGSW::alloc_from_infos(infos);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(GGSW::encrypt_sk_tmp_bytes(&module, infos));

    let enc_infos = NoiseInfos::new(infos.max_k().as_usize(), DEFAULT_SIGMA_XE, DEFAULT_BOUND_XE).unwrap();

    let group_name = format!("ggsw_encrypt_sk::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            ct.encrypt_sk(&module, &pt, &sk_prepared, &enc_infos, &mut source_xe, &mut source_xa, scratch.borrow());
            black_box(());
        })
    });
    group.finish();
}

pub fn bench_glwe_automorphism_key_encrypt_sk<BE: Backend>(atk_infos: &impl GGLWEInfos, p: i64, c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE> + GLWEAutomorphismKeyEncryptSk<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let n: usize = atk_infos.n().into();
    let module: Module<BE> = Module::<BE>::new(n as u64);

    let mut source_xs = Source::new([0u8; 32]);
    let mut source_xa = Source::new([1u8; 32]);
    let mut source_xe = Source::new([2u8; 32]);

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(atk_infos);
    sk.fill_ternary_prob(0.5, &mut source_xs);

    let mut atk: GLWEAutomorphismKey<Vec<u8>> = GLWEAutomorphismKey::alloc_from_infos(atk_infos);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(GLWEAutomorphismKey::encrypt_sk_tmp_bytes(&module, atk_infos));

    let enc_infos = NoiseInfos::new(atk_infos.max_k().as_usize(), DEFAULT_SIGMA_XE, DEFAULT_BOUND_XE).unwrap();

    let group_name = format!("glwe_automorphism_key_encrypt_sk::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            atk.encrypt_sk(&module, p, &sk, &enc_infos, &mut source_xe, &mut source_xa, scratch.borrow());
            black_box(());
        })
    });
    group.finish();
}
