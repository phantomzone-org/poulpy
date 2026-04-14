use poulpy_core::{
    DEFAULT_BOUND_XE, DEFAULT_SIGMA_XE, GLWEEncryptSk, GLWEKeyswitch, GLWESwitchingKeyEncryptSk, ScratchTakeCore,
    layouts::{
        GGLWEInfos, GLWE, GLWEInfos, GLWESecret, GLWESecretPreparedFactory, GLWESwitchingKey, GLWESwitchingKeyPrepared,
        GLWESwitchingKeyPreparedFactory, prepared::GLWESecretPrepared,
    },
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, DeviceBuf, Module, NoiseInfos, Scratch, ScratchOwned},
    source::Source,
};

use std::hint::black_box;

use criterion::Criterion;

pub fn bench_glwe_keyswitch<BE: Backend>(
    glwe_in: &impl GLWEInfos,
    glwe_out: &impl GLWEInfos,
    gglwe: &impl GGLWEInfos,
    c: &mut Criterion,
    label: &str,
) where
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
        module.glwe_switching_key_encrypt_sk_tmp_bytes(gglwe)
            | module.glwe_encrypt_sk_tmp_bytes(glwe_in)
            | module.glwe_keyswitch_tmp_bytes(glwe_out, glwe_in, gglwe),
    );

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([1u8; 32]);
    let mut source_xa: Source = Source::new([2u8; 32]);

    let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(glwe_in);
    sk_in.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_in_prepared: GLWESecretPrepared<DeviceBuf<BE>, BE> = module.alloc_glwe_secret_prepared(glwe_in.rank());
    module.prepare_glwe_secret(&mut sk_in_prepared, &sk_in);

    let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(glwe_out);
    sk_out.fill_ternary_prob(0.5, &mut source_xs);

    let ksk_enc_infos = NoiseInfos::new(gglwe.max_k().as_usize(), DEFAULT_SIGMA_XE, DEFAULT_BOUND_XE).unwrap();
    let glwe_enc_infos = NoiseInfos::new(glwe_in.max_k().as_usize(), DEFAULT_SIGMA_XE, DEFAULT_BOUND_XE).unwrap();

    module.glwe_switching_key_encrypt_sk(
        &mut ksk,
        &sk_in,
        &sk_out,
        &ksk_enc_infos,
        &mut source_xe,
        &mut source_xa,
        scratch.borrow(),
    );

    module.glwe_encrypt_zero_sk(
        &mut ct_in,
        &sk_in_prepared,
        &glwe_enc_infos,
        &mut source_xe,
        &mut source_xa,
        scratch.borrow(),
    );

    let mut ksk_prepared: GLWESwitchingKeyPrepared<DeviceBuf<BE>, BE> = module.alloc_glwe_switching_key_prepared_from_infos(&ksk);
    module.prepare_glwe_switching(&mut ksk_prepared, &ksk, scratch.borrow());

    let group_name = format!("glwe_keyswitch::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            module.glwe_keyswitch(&mut ct_out, &ct_in, &ksk_prepared, scratch.borrow());
            black_box(());
        })
    });
    group.finish();
}
