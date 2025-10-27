use poulpy_hal::{
    api::{ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, ScalarZnx, Scratch, ScratchOwned},
    source::Source,
};

use crate::{
    GGLWEToGGSWKeyEncryptSk, GGSWEncryptSk, GGSWKeyswitch, GGSWNoise, GLWESwitchingKeyEncryptSk, ScratchTakeCore,
    encryption::SIGMA,
    layouts::{
        GGLWEToGGSWKey, GGLWEToGGSWKeyPrepared, GGLWEToGGSWKeyPreparedFactory, GGSW, GGSWLayout, GLWESecret,
        GLWESecretPreparedFactory, GLWESwitchingKey, GLWESwitchingKeyLayout, GLWESwitchingKeyPreparedFactory,
        GLWETensorKeyLayout,
        prepared::{GLWESecretPrepared, GLWESwitchingKeyPrepared},
    },
    noise::noise_ggsw_keyswitch,
};

#[allow(clippy::too_many_arguments)]
pub fn test_ggsw_keyswitch<BE: Backend>(module: &Module<BE>)
where
    Module<BE>: GGSWEncryptSk<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GGLWEToGGSWKeyEncryptSk<BE>
        + GGSWKeyswitch<BE>
        + GLWESecretPreparedFactory<BE>
        + GGLWEToGGSWKeyPreparedFactory<BE>
        + GLWESwitchingKeyPreparedFactory<BE>
        + GGSWNoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k: usize = 12;
    let k_in: usize = 54;
    let dsize: usize = k_in.div_ceil(base2k);
    for rank in 1_usize..3 {
        for di in 1..dsize + 1 {
            let k_ksk: usize = k_in + base2k * di;
            let k_tsk: usize = k_ksk;
            let k_out: usize = k_ksk; // Better capture noise.

            let n: usize = module.n();
            let dnum: usize = k_in.div_ceil(di * base2k);

            let dsize_in: usize = 1;

            let ggsw_in_infos: GGSWLayout = GGSWLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_in.into(),
                dnum: dnum.into(),
                dsize: dsize_in.into(),
                rank: rank.into(),
            };

            let ggsw_out_infos: GGSWLayout = GGSWLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_out.into(),
                dnum: dnum.into(),
                dsize: dsize_in.into(),
                rank: rank.into(),
            };

            let tsk_infos: GLWETensorKeyLayout = GLWETensorKeyLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_tsk.into(),
                dnum: dnum.into(),
                dsize: di.into(),
                rank: rank.into(),
            };

            let ksk_apply_infos: GLWESwitchingKeyLayout = GLWESwitchingKeyLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_ksk.into(),
                dnum: dnum.into(),
                dsize: di.into(),
                rank_in: rank.into(),
                rank_out: rank.into(),
            };

            let mut ggsw_in: GGSW<Vec<u8>> = GGSW::alloc_from_infos(&ggsw_in_infos);
            let mut ggsw_out: GGSW<Vec<u8>> = GGSW::alloc_from_infos(&ggsw_out_infos);
            let mut tsk: GGLWEToGGSWKey<Vec<u8>> = GGLWEToGGSWKey::alloc_from_infos(&tsk_infos);
            let mut ksk: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc_from_infos(&ksk_apply_infos);
            let mut pt_scalar: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                GGSW::encrypt_sk_tmp_bytes(module, &ggsw_in_infos)
                    | GLWESwitchingKey::encrypt_sk_tmp_bytes(module, &ksk_apply_infos)
                    | GGLWEToGGSWKey::encrypt_sk_tmp_bytes(module, &tsk_infos)
                    | GGSW::keyswitch_tmp_bytes(
                        module,
                        &ggsw_out_infos,
                        &ggsw_in_infos,
                        &ksk_apply_infos,
                        &tsk_infos,
                    ),
            );

            let var_xs: f64 = 0.5;

            let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), rank.into());
            sk_in.fill_ternary_prob(var_xs, &mut source_xs);

            let mut sk_in_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc(module, rank.into());
            sk_in_prepared.prepare(module, &sk_in);

            let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), rank.into());
            sk_out.fill_ternary_prob(var_xs, &mut source_xs);

            let mut sk_out_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc(module, rank.into());
            sk_out_prepared.prepare(module, &sk_out);

            ksk.encrypt_sk(
                module,
                &sk_in,
                &sk_out,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );
            tsk.encrypt_sk(
                module,
                &sk_out,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            pt_scalar.fill_ternary_hw(0, n, &mut source_xs);

            ggsw_in.encrypt_sk(
                module,
                &pt_scalar,
                &sk_in_prepared,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            let mut ksk_prepared: GLWESwitchingKeyPrepared<Vec<u8>, BE> =
                GLWESwitchingKeyPrepared::alloc_from_infos(module, &ksk);
            ksk_prepared.prepare(module, &ksk, scratch.borrow());

            let mut tsk_prepared: GGLWEToGGSWKeyPrepared<Vec<u8>, BE> = GGLWEToGGSWKeyPrepared::alloc_from_infos(module, &tsk);
            tsk_prepared.prepare(module, &tsk, scratch.borrow());

            ggsw_out.keyswitch(
                module,
                &ggsw_in,
                &ksk_prepared,
                &tsk_prepared,
                scratch.borrow(),
            );

            let max_noise = |col_j: usize| -> f64 {
                noise_ggsw_keyswitch(
                    n as f64,
                    base2k * di,
                    col_j,
                    var_xs,
                    0f64,
                    SIGMA * SIGMA,
                    0f64,
                    rank as f64,
                    k_in,
                    k_ksk,
                    k_tsk,
                ) + 0.5
            };

            ggsw_out.assert_noise(module, &sk_out_prepared, &pt_scalar, &max_noise);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn test_ggsw_keyswitch_inplace<BE: Backend>(module: &Module<BE>)
where
    Module<BE>: GGSWEncryptSk<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GGLWEToGGSWKeyEncryptSk<BE>
        + GGSWKeyswitch<BE>
        + GLWESecretPreparedFactory<BE>
        + GGLWEToGGSWKeyPreparedFactory<BE>
        + GLWESwitchingKeyPreparedFactory<BE>
        + GGSWNoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k: usize = 12;
    let k_out: usize = 54;
    let dsize: usize = k_out.div_ceil(base2k);
    for rank in 1_usize..3 {
        for di in 1..dsize + 1 {
            let k_ksk: usize = k_out + base2k * di;
            let k_tsk: usize = k_ksk;

            let n: usize = module.n();
            let dnum: usize = k_out.div_ceil(di * base2k);

            let dsize_in: usize = 1;

            let ggsw_out_infos: GGSWLayout = GGSWLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_out.into(),
                dnum: dnum.into(),
                dsize: dsize_in.into(),
                rank: rank.into(),
            };

            let tsk_infos: GLWETensorKeyLayout = GLWETensorKeyLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_tsk.into(),
                dnum: dnum.into(),
                dsize: di.into(),
                rank: rank.into(),
            };

            let ksk_apply_infos: GLWESwitchingKeyLayout = GLWESwitchingKeyLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_ksk.into(),
                dnum: dnum.into(),
                dsize: di.into(),
                rank_in: rank.into(),
                rank_out: rank.into(),
            };

            let mut ggsw_out: GGSW<Vec<u8>> = GGSW::alloc_from_infos(&ggsw_out_infos);
            let mut tsk: GGLWEToGGSWKey<Vec<u8>> = GGLWEToGGSWKey::alloc_from_infos(&tsk_infos);
            let mut ksk: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc_from_infos(&ksk_apply_infos);
            let mut pt_scalar: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                GGSW::encrypt_sk_tmp_bytes(module, &ggsw_out_infos)
                    | GLWESwitchingKey::encrypt_sk_tmp_bytes(module, &ksk_apply_infos)
                    | GGLWEToGGSWKey::encrypt_sk_tmp_bytes(module, &tsk_infos)
                    | GGSW::keyswitch_tmp_bytes(
                        module,
                        &ggsw_out_infos,
                        &ggsw_out_infos,
                        &ksk_apply_infos,
                        &tsk_infos,
                    ),
            );

            let var_xs: f64 = 0.5;

            let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), rank.into());
            sk_in.fill_ternary_prob(var_xs, &mut source_xs);

            let mut sk_in_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc(module, rank.into());
            sk_in_prepared.prepare(module, &sk_in);

            let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), rank.into());
            sk_out.fill_ternary_prob(var_xs, &mut source_xs);

            let mut sk_out_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc(module, rank.into());
            sk_out_prepared.prepare(module, &sk_out);

            ksk.encrypt_sk(
                module,
                &sk_in,
                &sk_out,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );
            tsk.encrypt_sk(
                module,
                &sk_out,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            pt_scalar.fill_ternary_hw(0, n, &mut source_xs);

            ggsw_out.encrypt_sk(
                module,
                &pt_scalar,
                &sk_in_prepared,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            let mut ksk_prepared: GLWESwitchingKeyPrepared<Vec<u8>, BE> =
                GLWESwitchingKeyPrepared::alloc_from_infos(module, &ksk);
            ksk_prepared.prepare(module, &ksk, scratch.borrow());

            let mut tsk_prepared: GGLWEToGGSWKeyPrepared<Vec<u8>, BE> = GGLWEToGGSWKeyPrepared::alloc_from_infos(module, &tsk);
            tsk_prepared.prepare(module, &tsk, scratch.borrow());

            ggsw_out.keyswitch_inplace(module, &ksk_prepared, &tsk_prepared, scratch.borrow());

            let max_noise = |col_j: usize| -> f64 {
                noise_ggsw_keyswitch(
                    n as f64,
                    base2k * di,
                    col_j,
                    var_xs,
                    0f64,
                    SIGMA * SIGMA,
                    0f64,
                    rank as f64,
                    k_out,
                    k_ksk,
                    k_tsk,
                ) + 0.5
            };

            ggsw_out.assert_noise(module, &sk_out_prepared, &pt_scalar, &max_noise);
        }
    }
}
