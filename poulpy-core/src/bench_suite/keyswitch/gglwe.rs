use crate::{
    GLWEEncryptSk, GLWEKeyswitch, GLWESwitchingKeyEncryptSk, ScratchTakeCore,
    layouts::{
        Degree, GGLWEInfos, GLWE, GLWEInfos, GLWESecret, GLWESecretPreparedFactory, GLWESwitchingKey, GLWESwitchingKeyPrepared,
        GLWESwitchingKeyPreparedFactory, prepared::GLWESecretPrepared,
    },
};
use poulpy_hal::{
    api::{ModuleNew, ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, Scratch, ScratchOwned},
    source::Source,
};

use std::{hint::black_box, time::Duration};

use criterion::Criterion;

pub fn bench_keyswitch_glwe<BE: Backend, A, B, C>(glwe_in: &A, glwe_out: &B, gglwe: &C, c: &mut Criterion, label: &str)
where
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
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    assert_eq!(glwe_in.n(), gglwe.n());
    assert_eq!(glwe_out.n(), gglwe.n());
    assert_eq!(glwe_in.rank(), gglwe.rank_in());
    assert_eq!(glwe_out.rank(), gglwe.rank_out());

    let group_name: String = format!("keyswitch_glwe::{label}");

    let mut group = c.benchmark_group(group_name);

    let n: usize = gglwe.n().into();
    let cols_in_glwe: usize = 1 + glwe_in.rank().as_usize();
    let cols_gglwe: usize = cols_in_glwe - 1;
    let cols_out_glwe: usize = 1 + glwe_out.rank().as_usize();
    let glwe_size_in: usize = glwe_in.size();
    let glwe_size_out: usize = glwe_out.size();
    let gglwe_size: usize = gglwe.size();
    let dnum: usize = gglwe.dnum().as_usize();

    let module: Module<BE> = Module::<BE>::new(n as u64);

    let n: Degree = Degree(module.n() as u32);

    let mut ksk: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc_from_infos(gglwe);
    let mut ct_in: GLWE<Vec<u8>> = GLWE::alloc_from_infos(glwe_in);
    let mut ct_out: GLWE<Vec<u8>> = GLWE::alloc_from_infos(glwe_out);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        GLWESwitchingKey::encrypt_sk_tmp_bytes(&module, gglwe)
            | GLWE::encrypt_sk_tmp_bytes(&module, glwe_in)
            | GLWE::keyswitch_tmp_bytes(&module, glwe_out, glwe_in, gglwe),
    );

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(glwe_in);
    sk_in.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_in_dft: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc(&module, glwe_in.rank());
    sk_in_dft.prepare(&module, &sk_in);

    let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(glwe_in);
    sk_out.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_out_dft: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc(&module, glwe_out.rank());
    sk_out_dft.prepare(&module, &sk_out);

    ksk.encrypt_sk(&module, &sk_in, &sk_out, &mut source_xa, &mut source_xe, scratch.borrow());

    ct_in.encrypt_zero_sk(&module, &sk_in_dft, &mut source_xa, &mut source_xe, scratch.borrow());

    let mut ksk_prepared: GLWESwitchingKeyPrepared<Vec<u8>, _> = GLWESwitchingKeyPrepared::alloc_from_infos(&module, &ksk);
    ksk_prepared.prepare(&module, &ksk, scratch.borrow());

    group.measurement_time(Duration::from_secs(40));
    group.bench_with_input(
        format!("({n} x {glwe_size_in} x {cols_in_glwe}) x ({n} x {gglwe_size} x {dnum} x {cols_gglwe} x {cols_out_glwe}) -> ({n} x {glwe_size_out} x {cols_in_glwe})"),
        &(),
        |b, _| {
            b.iter(|| {
                ct_out.keyswitch(&module, &ct_in, &ksk_prepared, scratch.borrow());
                black_box(());
            })
        },
    );

    group.finish();
}
