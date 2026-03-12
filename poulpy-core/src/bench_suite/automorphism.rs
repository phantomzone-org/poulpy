use crate::{
    GLWEAutomorphism, GLWEAutomorphismKeyEncryptSk, GLWEEncryptSk, ScratchTakeCore,
    layouts::{
        GGLWEInfos, GLWE, GLWEAutomorphismKey, GLWEInfos, GLWESecret,
        GLWESecretPreparedFactory,
        prepared::{GLWEAutomorphismKeyPrepared, GLWEAutomorphismKeyPreparedFactory, GLWESecretPrepared},
    },
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, Scratch, ScratchOwned},
    source::Source,
};
use std::{hint::black_box, time::Duration};

use criterion::Criterion;

/// Benchmarks the GLWE automorphism operation.
///
/// `glwe_infos` describes the input/output GLWE layout.
/// `atk_infos` describes the automorphism key layout.
/// `p` is the Galois element (e.g. 3 for X -> X^3).
pub fn bench_glwe_automorphism<BE: Backend, A, B>(
    glwe_infos: &A,
    atk_infos: &B,
    p: i64,
    c: &mut Criterion,
    label: &str,
) where
    A: GLWEInfos,
    B: GGLWEInfos,
    Module<BE>: ModuleNew<BE>
        + GLWEAutomorphism<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>,
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

    let mut sk_prepared: GLWESecretPrepared<Vec<u8>, BE> =
        GLWESecretPrepared::alloc(&module, atk_infos.rank_out());
    sk_prepared.prepare(&module, &sk);

    let mut atk: GLWEAutomorphismKey<Vec<u8>> = GLWEAutomorphismKey::alloc_from_infos(atk_infos);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        GLWEAutomorphismKey::encrypt_sk_tmp_bytes(&module, atk_infos)
            | GLWE::encrypt_sk_tmp_bytes(&module, glwe_infos)
            | GLWE::automorphism_tmp_bytes(&module, glwe_infos, glwe_infos, atk_infos),
    );

    atk.encrypt_sk(&module, p, &sk, &mut source_xa, &mut source_xe, scratch.borrow());

    let mut atk_prepared: GLWEAutomorphismKeyPrepared<Vec<u8>, BE> =
        GLWEAutomorphismKeyPrepared::alloc_from_infos(&module, &atk);
    module.prepare_glwe_automorphism_key(&mut atk_prepared, &atk, scratch.borrow());

    let mut ct_in: GLWE<Vec<u8>> = GLWE::alloc_from_infos(glwe_infos);
    let mut ct_out: GLWE<Vec<u8>> = GLWE::alloc_from_infos(glwe_infos);

    ct_in.encrypt_zero_sk(&module, &sk_prepared, &mut source_xa, &mut source_xe, scratch.borrow());

    let group_name = format!("glwe_automorphism::{label}");
    let mut group = c.benchmark_group(group_name);
    group.measurement_time(Duration::from_secs(40));
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            ct_out.automorphism(&module, &ct_in, &atk_prepared, scratch.borrow());
            black_box(());
        })
    });
    group.finish();
}
