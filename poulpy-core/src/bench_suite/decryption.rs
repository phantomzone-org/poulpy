use crate::{
    GLWEDecrypt, GLWEEncryptSk, ScratchTakeCore,
    layouts::{
        GLWE, GLWEInfos, GLWEPlaintext, GLWESecret, GLWESecretPreparedFactory,
        prepared::GLWESecretPrepared,
    },
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, Scratch, ScratchOwned},
    source::Source,
};
use std::{hint::black_box, time::Duration};

use criterion::Criterion;

pub fn bench_glwe_decrypt<BE: Backend, A>(infos: &A, c: &mut Criterion, label: &str)
where
    A: GLWEInfos,
    Module<BE>: ModuleNew<BE>
        + GLWEDecrypt<BE>
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

    let mut sk_prepared: GLWESecretPrepared<Vec<u8>, BE> =
        GLWESecretPrepared::alloc(&module, infos.rank());
    sk_prepared.prepare(&module, &sk);

    let mut ct: GLWE<Vec<u8>> = GLWE::alloc_from_infos(infos);
    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(infos);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        GLWE::encrypt_sk_tmp_bytes(&module, infos) | GLWE::decrypt_tmp_bytes(&module, infos),
    );

    ct.encrypt_zero_sk(&module, &sk_prepared, &mut source_xa, &mut source_xe, scratch.borrow());

    let group_name = format!("glwe_decrypt::{label}");
    let mut group = c.benchmark_group(group_name);
    group.measurement_time(Duration::from_secs(10));
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            ct.decrypt(&module, &mut pt, &sk_prepared, scratch.borrow());
            black_box(());
        })
    });
    group.finish();
}
