use poulpy_hal::{
    api::{ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, Scratch, ScratchOwned},
    source::Source,
};

use crate::{
    GGLWEKeyswitch, GGLWENoise, GLWESwitchingKeyEncryptSk, ScratchTakeCore,
    encryption::SIGMA,
    layouts::{
        GGLWEInfos, GLWESecret, GLWESecretPreparedFactory, GLWESwitchingKey, GLWESwitchingKeyLayout,
        GLWESwitchingKeyPreparedFactory,
        prepared::{GLWESecretPrepared, GLWESwitchingKeyPrepared},
    },
    noise::log2_std_noise_gglwe_product,
    var_noise_gglwe_product_v2,
};

pub fn test_gglwe_switching_key_keyswitch<BE: Backend>(module: &Module<BE>)
where
    Module<BE>: GLWESwitchingKeyEncryptSk<BE>
        + GGLWEKeyswitch<BE>
        + GLWESwitchingKeyPreparedFactory<BE>
        + GLWESecretPreparedFactory<BE>
        + GGLWENoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k_in: usize = 17;
    let base2k_key: usize = 13;
    let base2k_out: usize = base2k_in; // MUST BE SAME
    let k_in: usize = 102;
    let max_dsize: usize = k_in.div_ceil(base2k_key);

    for rank_in_s0s1 in 1_usize..2 {
        for rank_out_s0s1 in 1_usize..3 {
            for rank_out_s1s2 in 1_usize..3 {
                for dsize in 1_usize..max_dsize + 1 {
                    let k_ksk: usize = k_in + base2k_key * dsize;
                    let k_out: usize = k_ksk; // Better capture noise.

                    let n: usize = module.n();
                    let dsize_in: usize = 1;
                    let dnum_in: usize = k_in / base2k_in;
                    let dnum_ksk: usize = k_in.div_ceil(base2k_key * dsize);

                    let gglwe_s0s1_infos: GLWESwitchingKeyLayout = GLWESwitchingKeyLayout {
                        n: n.into(),
                        base2k: base2k_in.into(),
                        k: k_in.into(),
                        dnum: dnum_in.into(),
                        dsize: dsize_in.into(),
                        rank_in: rank_in_s0s1.into(),
                        rank_out: rank_out_s0s1.into(),
                    };

                    let gglwe_s1s2_infos: GLWESwitchingKeyLayout = GLWESwitchingKeyLayout {
                        n: n.into(),
                        base2k: base2k_key.into(),
                        k: k_ksk.into(),
                        dnum: dnum_ksk.into(),
                        dsize: dsize.into(),
                        rank_in: rank_out_s0s1.into(),
                        rank_out: rank_out_s1s2.into(),
                    };

                    let gglwe_s0s2_infos: GLWESwitchingKeyLayout = GLWESwitchingKeyLayout {
                        n: n.into(),
                        base2k: base2k_out.into(),
                        k: k_out.into(),
                        dnum: dnum_in.into(),
                        dsize: dsize_in.into(),
                        rank_in: rank_in_s0s1.into(),
                        rank_out: rank_out_s1s2.into(),
                    };

                    let mut gglwe_s0s1: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc_from_infos(&gglwe_s0s1_infos);
                    let mut gglwe_s1s2: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc_from_infos(&gglwe_s1s2_infos);
                    let mut gglwe_s0s2: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc_from_infos(&gglwe_s0s2_infos);

                    let mut source_xs: Source = Source::new([0u8; 32]);
                    let mut source_xe: Source = Source::new([0u8; 32]);
                    let mut source_xa: Source = Source::new([0u8; 32]);

                    let mut scratch_enc: ScratchOwned<BE> = ScratchOwned::alloc(
                        GLWESwitchingKey::encrypt_sk_tmp_bytes(module, &gglwe_s0s1_infos)
                            | GLWESwitchingKey::encrypt_sk_tmp_bytes(module, &gglwe_s1s2_infos)
                            | GLWESwitchingKey::encrypt_sk_tmp_bytes(module, &gglwe_s0s2_infos),
                    );
                    let mut scratch_apply: ScratchOwned<BE> = ScratchOwned::alloc(GLWESwitchingKey::keyswitch_tmp_bytes(
                        module,
                        &gglwe_s0s2_infos,
                        &gglwe_s0s1_infos,
                        &gglwe_s1s2_infos,
                    ));

                    let mut sk0: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), rank_in_s0s1.into());
                    sk0.fill_ternary_prob(0.5, &mut source_xs);

                    let mut sk1: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), rank_out_s0s1.into());
                    sk1.fill_ternary_prob(0.5, &mut source_xs);

                    let mut sk2: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), rank_out_s1s2.into());
                    sk2.fill_ternary_prob(0.5, &mut source_xs);

                    let mut sk2_prepared: GLWESecretPrepared<Vec<u8>, BE> =
                        GLWESecretPrepared::alloc(module, rank_out_s1s2.into());
                    sk2_prepared.prepare(module, &sk2);

                    // gglwe_{s1}(s0) = s0 -> s1
                    gglwe_s0s1.encrypt_sk(module, &sk0, &sk1, &mut source_xa, &mut source_xe, scratch_enc.borrow());

                    // gglwe_{s2}(s1) -> s1 -> s2
                    gglwe_s1s2.encrypt_sk(module, &sk1, &sk2, &mut source_xa, &mut source_xe, scratch_enc.borrow());

                    let mut gglwe_s1s2_prepared: GLWESwitchingKeyPrepared<Vec<u8>, BE> =
                        GLWESwitchingKeyPrepared::alloc_from_infos(module, &gglwe_s1s2);
                    gglwe_s1s2_prepared.prepare(module, &gglwe_s1s2, scratch_apply.borrow());

                    // gglwe_{s1}(s0) (x) gglwe_{s2}(s1) = gglwe_{s2}(s0)
                    gglwe_s0s2.keyswitch(module, &gglwe_s0s1, &gglwe_s1s2_prepared, scratch_apply.borrow());

                    let max_noise: f64 = var_noise_gglwe_product_v2(
                        module.n() as f64,
                        k_ksk,
                        dnum_ksk,
                        dsize,
                        base2k_key,
                        0.5,
                        0.5,
                        0f64,
                        SIGMA * SIGMA,
                        0f64,
                        rank_out_s0s1 as f64,
                    )
                    .sqrt()
                    .log2();

                    for row in 0..gglwe_s0s2.dnum().as_usize() {
                        for col in 0..gglwe_s0s2.rank_in().as_usize() {
                            assert!(
                                gglwe_s0s2
                                    .key
                                    .noise(module, row, col, &sk0.data, &sk2_prepared, scratch_apply.borrow())
                                    .std()
                                    .log2()
                                    <= max_noise + 0.5
                            )
                        }
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn test_gglwe_switching_key_keyswitch_inplace<BE: Backend>(module: &Module<BE>)
where
    Module<BE>: GLWESwitchingKeyEncryptSk<BE>
        + GGLWEKeyswitch<BE>
        + GLWESecretPreparedFactory<BE>
        + GGLWENoise<BE>
        + GLWESwitchingKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k_out: usize = 17;
    let base2k_key: usize = 13;
    let k_out: usize = 102;
    let max_dsize: usize = k_out.div_ceil(base2k_key);

    for rank_in in 1_usize..3 {
        for rank_out in 1_usize..3 {
            for dsize in 1_usize..max_dsize + 1 {
                let k_ksk: usize = k_out + base2k_key * dsize;

                let n: usize = module.n();
                let dsize_in: usize = 1;

                let dnum_in: usize = k_out / base2k_out;
                let dnum_ksk: usize = k_out.div_ceil(base2k_key * dsize);

                let gglwe_s0s1_infos: GLWESwitchingKeyLayout = GLWESwitchingKeyLayout {
                    n: n.into(),
                    base2k: base2k_out.into(),
                    k: k_out.into(),
                    dnum: dnum_in.into(),
                    dsize: dsize_in.into(),
                    rank_in: rank_in.into(),
                    rank_out: rank_out.into(),
                };

                let gglwe_s1s2_infos: GLWESwitchingKeyLayout = GLWESwitchingKeyLayout {
                    n: n.into(),
                    base2k: base2k_key.into(),
                    k: k_ksk.into(),
                    dnum: dnum_ksk.into(),
                    dsize: dsize.into(),
                    rank_in: rank_out.into(),
                    rank_out: rank_out.into(),
                };

                let mut gglwe_s0s1: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc_from_infos(&gglwe_s0s1_infos);
                let mut gglwe_s1s2: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc_from_infos(&gglwe_s1s2_infos);

                let mut source_xs: Source = Source::new([0u8; 32]);
                let mut source_xe: Source = Source::new([0u8; 32]);
                let mut source_xa: Source = Source::new([0u8; 32]);

                let mut scratch_enc: ScratchOwned<BE> = ScratchOwned::alloc(
                    GLWESwitchingKey::encrypt_sk_tmp_bytes(module, &gglwe_s0s1_infos)
                        | GLWESwitchingKey::encrypt_sk_tmp_bytes(module, &gglwe_s1s2_infos),
                );
                let mut scratch_apply: ScratchOwned<BE> = ScratchOwned::alloc(GLWESwitchingKey::keyswitch_tmp_bytes(
                    module,
                    &gglwe_s0s1_infos,
                    &gglwe_s0s1_infos,
                    &gglwe_s1s2_infos,
                ));

                let var_xs: f64 = 0.5;

                let mut sk0: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), rank_in.into());
                sk0.fill_ternary_prob(var_xs, &mut source_xs);

                let mut sk1: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), rank_out.into());
                sk1.fill_ternary_prob(var_xs, &mut source_xs);

                let mut sk2: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), rank_out.into());
                sk2.fill_ternary_prob(var_xs, &mut source_xs);

                let mut sk2_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc(module, rank_out.into());
                sk2_prepared.prepare(module, &sk2);

                // gglwe_{s1}(s0) = s0 -> s1
                gglwe_s0s1.encrypt_sk(module, &sk0, &sk1, &mut source_xa, &mut source_xe, scratch_enc.borrow());

                // gglwe_{s2}(s1) -> s1 -> s2
                gglwe_s1s2.encrypt_sk(module, &sk1, &sk2, &mut source_xa, &mut source_xe, scratch_enc.borrow());

                let mut gglwe_s1s2_prepared: GLWESwitchingKeyPrepared<Vec<u8>, BE> =
                    GLWESwitchingKeyPrepared::alloc_from_infos(module, &gglwe_s1s2);
                gglwe_s1s2_prepared.prepare(module, &gglwe_s1s2, scratch_apply.borrow());

                // gglwe_{s1}(s0) (x) gglwe_{s2}(s1) = gglwe_{s2}(s0)
                gglwe_s0s1.keyswitch_inplace(module, &gglwe_s1s2_prepared, scratch_apply.borrow());

                let gglwe_s0s2: GLWESwitchingKey<Vec<u8>> = gglwe_s0s1;

                let max_noise: f64 = log2_std_noise_gglwe_product(
                    n as f64,
                    base2k_key * dsize,
                    var_xs,
                    var_xs,
                    0f64,
                    SIGMA * SIGMA,
                    0f64,
                    rank_out as f64,
                    k_out,
                    k_ksk,
                );

                for row in 0..gglwe_s0s2.dnum().as_usize() {
                    for col in 0..gglwe_s0s2.rank_in().as_usize() {
                        assert!(
                            gglwe_s0s2
                                .key
                                .noise(module, row, col, &sk0.data, &sk2_prepared, scratch_apply.borrow())
                                .std()
                                .log2()
                                <= max_noise + 0.5
                        )
                    }
                }
            }
        }
    }
}
