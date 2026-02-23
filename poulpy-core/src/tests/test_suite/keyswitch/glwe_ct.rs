use poulpy_hal::{
    api::{ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxFillUniform},
    layouts::{Backend, Module, Scratch, ScratchOwned},
    source::Source,
};

use crate::{
    GLWEEncryptSk, GLWEKeyswitch, GLWENoise, GLWENormalize, GLWESwitchingKeyEncryptSk, ScratchTakeCore,
    encryption::SIGMA,
    layouts::{
        GLWE, GLWELayout, GLWEPlaintext, GLWESecret, GLWESecretPreparedFactory, GLWESwitchingKey, GLWESwitchingKeyLayout,
        GLWESwitchingKeyPreparedFactory, LWEInfos,
        prepared::{GLWESecretPrepared, GLWESwitchingKeyPrepared},
    },
    var_noise_gglwe_product_v2,
};

#[allow(clippy::too_many_arguments)]
pub fn test_glwe_keyswitch<BE: Backend>(module: &Module<BE>)
where
    Module<BE>: VecZnxFillUniform
        + GLWESwitchingKeyEncryptSk<BE>
        + GLWEEncryptSk<BE>
        + GLWEKeyswitch<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESwitchingKeyPreparedFactory<BE>
        + GLWENoise<BE>
        + GLWENormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let in_base2k: usize = 17;
    let key_base2k: usize = 13;
    let out_base2k: usize = 15;
    let k_in: usize = 102;
    let max_dsize: usize = k_in.div_ceil(key_base2k);

    for rank_in in 1_usize..3 {
        for rank_out in 1_usize..3 {
            for dsize in 1_usize..max_dsize + 1 {
                let k_ksk: usize = k_in + key_base2k * dsize;
                let k_out: usize = k_ksk; // better capture noise

                let n: usize = module.n();
                let dnum: usize = k_in.div_ceil(key_base2k * dsize);

                let glwe_in_infos: GLWELayout = GLWELayout {
                    n: n.into(),
                    base2k: in_base2k.into(),
                    k: k_in.into(),
                    rank: rank_in.into(),
                };

                let glwe_out_infos: GLWELayout = GLWELayout {
                    n: n.into(),
                    base2k: out_base2k.into(),
                    k: k_out.into(),
                    rank: rank_out.into(),
                };

                let ksk: GLWESwitchingKeyLayout = GLWESwitchingKeyLayout {
                    n: n.into(),
                    base2k: key_base2k.into(),
                    k: k_ksk.into(),
                    dnum: dnum.into(),
                    dsize: dsize.into(),
                    rank_in: rank_in.into(),
                    rank_out: rank_out.into(),
                };

                let mut ksk: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc_from_infos(&ksk);
                let mut glwe_in: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_in_infos);
                let mut glwe_out: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_out_infos);
                let mut pt_in: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_in_infos);
                let mut pt_out: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_out_infos);

                let mut source_xs: Source = Source::new([0u8; 32]);
                let mut source_xe: Source = Source::new([0u8; 32]);
                let mut source_xa: Source = Source::new([0u8; 32]);

                module.vec_znx_fill_uniform(pt_in.base2k().into(), &mut pt_in.data, 0, &mut source_xa);

                let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                    GLWESwitchingKey::encrypt_sk_tmp_bytes(module, &ksk)
                        | GLWE::encrypt_sk_tmp_bytes(module, &glwe_in_infos)
                        | GLWE::keyswitch_tmp_bytes(module, &glwe_out_infos, &glwe_in_infos, &ksk),
                );

                let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), rank_in.into());
                sk_in.fill_ternary_prob(0.5, &mut source_xs);

                let mut sk_in_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc(module, rank_in.into());
                sk_in_prepared.prepare(module, &sk_in);

                let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), rank_out.into());
                sk_out.fill_ternary_prob(0.5, &mut source_xs);

                let mut sk_out_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc(module, rank_out.into());
                sk_out_prepared.prepare(module, &sk_out);

                ksk.encrypt_sk(module, &sk_in, &sk_out, &mut source_xa, &mut source_xe, scratch.borrow());

                glwe_in.encrypt_sk(
                    module,
                    &pt_in,
                    &sk_in_prepared,
                    &mut source_xa,
                    &mut source_xe,
                    scratch.borrow(),
                );

                let mut ksk_prepared: GLWESwitchingKeyPrepared<Vec<u8>, BE> =
                    GLWESwitchingKeyPrepared::alloc_from_infos(module, &ksk);
                ksk_prepared.prepare(module, &ksk, scratch.borrow());

                glwe_out.keyswitch(module, &glwe_in, &ksk_prepared, scratch.borrow());

                let noise_max: f64 = var_noise_gglwe_product_v2(
                    module.n() as f64,
                    k_ksk,
                    dnum,
                    dsize,
                    key_base2k,
                    0.5,
                    0.5,
                    0f64,
                    SIGMA * SIGMA,
                    0f64,
                    rank_in as f64,
                )
                .sqrt()
                .log2()
                    + 1.0;

                module.glwe_normalize(&mut pt_out, &pt_in, scratch.borrow());

                let noise_have = glwe_out
                    .noise(module, &pt_out, &sk_out_prepared, scratch.borrow())
                    .std()
                    .log2();

                assert!(noise_have <= noise_max, "noise_have: {noise_have} > noise_max: {noise_max}");
            }
        }
    }
}

pub fn test_glwe_keyswitch_inplace<BE: Backend>(module: &Module<BE>)
where
    Module<BE>: VecZnxFillUniform
        + GLWESwitchingKeyEncryptSk<BE>
        + GLWEEncryptSk<BE>
        + GLWEKeyswitch<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESwitchingKeyPreparedFactory<BE>
        + GLWENoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let out_base2k: usize = 13;
    let key_base2k: usize = 13;

    let k_out: usize = 102;
    let max_dsize: usize = k_out.div_ceil(key_base2k);

    for rank in 1_usize..3 {
        for dsize in 1..max_dsize + 1 {
            let k_ksk: usize = k_out + key_base2k * dsize;

            let n: usize = module.n();
            let dnum: usize = k_out.div_ceil(key_base2k * dsize);
            let glwe_out_infos: GLWELayout = GLWELayout {
                n: n.into(),
                base2k: out_base2k.into(),
                k: k_out.into(),
                rank: rank.into(),
            };

            let ksk_infos: GLWESwitchingKeyLayout = GLWESwitchingKeyLayout {
                n: n.into(),
                base2k: key_base2k.into(),
                k: k_ksk.into(),
                dnum: dnum.into(),
                dsize: dsize.into(),
                rank_in: rank.into(),
                rank_out: rank.into(),
            };

            let mut ksk: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc_from_infos(&ksk_infos);
            let mut glwe_out: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_out_infos);
            let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_out_infos);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            module.vec_znx_fill_uniform(pt_want.base2k().into(), &mut pt_want.data, 0, &mut source_xa);

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                GLWESwitchingKey::encrypt_sk_tmp_bytes(module, &ksk_infos)
                    | GLWE::encrypt_sk_tmp_bytes(module, &glwe_out_infos)
                    | GLWE::keyswitch_tmp_bytes(module, &glwe_out_infos, &glwe_out_infos, &ksk_infos),
            );

            let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), rank.into());
            sk_in.fill_ternary_prob(0.5, &mut source_xs);

            let mut sk_in_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc(module, rank.into());
            sk_in_prepared.prepare(module, &sk_in);

            let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), rank.into());
            sk_out.fill_ternary_prob(0.5, &mut source_xs);

            let mut sk_out_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc(module, rank.into());
            sk_out_prepared.prepare(module, &sk_out);

            ksk.encrypt_sk(module, &sk_in, &sk_out, &mut source_xa, &mut source_xe, scratch.borrow());

            glwe_out.encrypt_sk(
                module,
                &pt_want,
                &sk_in_prepared,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            let mut ksk_prepared: GLWESwitchingKeyPrepared<Vec<u8>, BE> =
                GLWESwitchingKeyPrepared::alloc_from_infos(module, &ksk);
            ksk_prepared.prepare(module, &ksk, scratch.borrow());

            glwe_out.keyswitch_inplace(module, &ksk_prepared, scratch.borrow());

            let noise_max: f64 = var_noise_gglwe_product_v2(
                module.n() as f64,
                k_ksk,
                dnum,
                dsize,
                key_base2k,
                0.5,
                0.5,
                0f64,
                SIGMA * SIGMA,
                0f64,
                rank as f64,
            )
            .sqrt()
            .log2()
                + 1.0;

            let noise_have = glwe_out
                .noise(module, &pt_want, &sk_out_prepared, scratch.borrow())
                .std()
                .log2();

            assert!(noise_have <= noise_max, "noise_have: {noise_have} > noise_max: {noise_max}");
        }
    }
}
