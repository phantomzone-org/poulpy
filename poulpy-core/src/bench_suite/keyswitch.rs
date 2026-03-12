use crate::{
    GLWEEncryptSk, GLWEKeyswitch, GLWESwitchingKeyEncryptSk, ScratchTakeCore,
    layouts::{
        GGLWEInfos, GLWE, GLWEInfos, GLWESecret, GLWESecretPreparedFactory, GLWESwitchingKey,
        GLWESwitchingKeyPrepared, GLWESwitchingKeyPreparedFactory, prepared::GLWESecretPrepared,
    },
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, Scratch, ScratchOwned},
    source::Source,
};

use std::{hint::black_box, time::Duration};

use criterion::Criterion;

pub fn bench_glwe_keyswitch<BE: Backend, A, B, C>(
    glwe_in: &A,
    glwe_out: &B,
    gglwe: &C,
    c: &mut Criterion,
    label: &str,
) where
    A: GLWEInfos,
    B: GLWEInfos,
    C: GGLWEInfos,
    Module<BE>: ModuleNew<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GLWEEncryptSk<BE>
        + GLWEKeyswitch<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESwitchingKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let n: usize = gglwe.n().into();
    let module: Module<BE> = Module::<BE>::new(n as u64);

    let mut ksk: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc_from_infos(gglwe);
    let mut ct_in: GLWE<Vec<u8>> = GLWE::alloc_from_infos(glwe_in);
    let mut ct_out: GLWE<Vec<u8>> = GLWE::alloc_from_infos(glwe_out);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        GLWESwitchingKey::encrypt_sk_tmp_bytes(&module, gglwe)
            | GLWE::encrypt_sk_tmp_bytes(&module, glwe_in)
            | GLWE::keyswitch_tmp_bytes(&module, glwe_out, glwe_in, gglwe),
    );

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([1u8; 32]);
    let mut source_xa: Source = Source::new([2u8; 32]);

    let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(glwe_in);
    sk_in.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_in_prepared: GLWESecretPrepared<Vec<u8>, BE> =
        GLWESecretPrepared::alloc(&module, glwe_in.rank());
    sk_in_prepared.prepare(&module, &sk_in);

    let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(glwe_out);
    sk_out.fill_ternary_prob(0.5, &mut source_xs);

    ksk.encrypt_sk(&module, &sk_in, &sk_out, &mut source_xa, &mut source_xe, scratch.borrow());

    ct_in.encrypt_zero_sk(&module, &sk_in_prepared, &mut source_xa, &mut source_xe, scratch.borrow());

    let mut ksk_prepared: GLWESwitchingKeyPrepared<Vec<u8>, BE> =
        GLWESwitchingKeyPrepared::alloc_from_infos(&module, &ksk);
    ksk_prepared.prepare(&module, &ksk, scratch.borrow());

    let group_name = format!("glwe_keyswitch::{label}");
    let mut group = c.benchmark_group(group_name);
    group.measurement_time(Duration::from_secs(40));
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            ct_out.keyswitch(&module, &ct_in, &ksk_prepared, scratch.borrow());
            black_box(());
        })
    });
    group.finish();
}
