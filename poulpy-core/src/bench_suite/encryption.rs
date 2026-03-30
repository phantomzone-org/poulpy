use crate::{
    GGSWEncryptSk, GLWEAutomorphismKeyEncryptSk, GLWEEncryptSk, ScratchTakeCore,
    layouts::{
        GGLWEInfos, GGSW, GGSWInfos, GLWEAutomorphismKey, GLWEInfos, GLWESecret, GLWESecretPreparedFactory,
        prepared::GLWESecretPrepared,
    },
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, ScalarZnx, Scratch, ScratchOwned},
    source::Source,
};
use std::{hint::black_box, time::Duration};

use criterion::Criterion;

pub fn bench_glwe_encrypt_sk<BE: Backend, A>(infos: &A, c: &mut Criterion, label: &str)
where
    A: GLWEInfos,
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

    let mut ct = crate::layouts::GLWE::alloc_from_infos(infos);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(crate::layouts::GLWE::encrypt_sk_tmp_bytes(&module, infos));

    let group_name = format!("glwe_encrypt_sk::{label}");
    let mut group = c.benchmark_group(group_name);
    group.measurement_time(Duration::from_secs(10));
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            ct.encrypt_zero_sk(&module, &sk_prepared, &mut source_xa, &mut source_xe, scratch.borrow());
            black_box(());
        })
    });
    group.finish();
}

pub fn bench_ggsw_encrypt_sk<BE: Backend, A>(infos: &A, c: &mut Criterion, label: &str)
where
    A: GGSWInfos,
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

    let group_name = format!("ggsw_encrypt_sk::{label}");
    let mut group = c.benchmark_group(group_name);
    group.measurement_time(Duration::from_secs(10));
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            ct.encrypt_sk(&module, &pt, &sk_prepared, &mut source_xa, &mut source_xe, scratch.borrow());
            black_box(());
        })
    });
    group.finish();
}

pub fn bench_glwe_automorphism_key_encrypt_sk<BE: Backend, A>(atk_infos: &A, p: i64, c: &mut Criterion, label: &str)
where
    A: GGLWEInfos,
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

    let group_name = format!("glwe_automorphism_key_encrypt_sk::{label}");
    let mut group = c.benchmark_group(group_name);
    group.measurement_time(Duration::from_secs(40));
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            atk.encrypt_sk(&module, p, &sk, &mut source_xa, &mut source_xe, scratch.borrow());
            black_box(());
        })
    });
    group.finish();
}
