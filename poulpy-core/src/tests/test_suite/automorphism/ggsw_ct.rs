use poulpy_hal::{
    api::{ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAutomorphismInplace},
    layouts::{Backend, Module, ScalarZnx, Scratch, ScratchOwned},
    source::Source,
};

use crate::{
    GGLWEToGGSWKeyEncryptSk, GGSWAutomorphism, GGSWEncryptSk, GGSWNoise, GLWEAutomorphismKeyEncryptSk, ScratchTakeCore,
    encryption::SIGMA,
    layouts::{
        GGLWEToGGSWKey, GGLWEToGGSWKeyLayout, GGLWEToGGSWKeyPreparedFactory, GGSW, GGSWLayout, GLWEAutomorphismKey,
        GLWEAutomorphismKeyPreparedFactory, GLWESecret, GLWESecretPreparedFactory,
        prepared::{GGLWEToGGSWKeyPrepared, GLWEAutomorphismKeyPrepared, GLWESecretPrepared},
    },
    noise::noise_ggsw_keyswitch,
};

pub fn test_ggsw_automorphism<BE: Backend>(module: &Module<BE>)
where
    Module<BE>: GGSWEncryptSk<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GGSWAutomorphism<BE>
        + GGLWEToGGSWKeyPreparedFactory<BE>
        + GGLWEToGGSWKeyEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxAutomorphismInplace<BE>
        + GGSWNoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k_in: usize = 17;
    let base2k_key: usize = 13;
    let base2k_out: usize = base2k_in; // MUST BE SAME
    let k_in: usize = 102;
    let max_dsize: usize = k_in.div_ceil(base2k_key);

    let p: i64 = -5;
    for rank in 1_usize..3 {
        for dsize in 1..max_dsize + 1 {
            let k_ksk: usize = k_in + base2k_key * dsize;
            let k_tsk: usize = k_ksk;
            let k_out: usize = k_ksk; // Better capture noise.

            let n: usize = module.n();
            let dnum_in: usize = k_in / base2k_in;
            let dnum_ksk: usize = k_in.div_ceil(base2k_key * dsize);

            let dsize_in: usize = 1;

            let ggsw_in_layout: GGSWLayout = GGSWLayout {
                n: n.into(),
                base2k: base2k_in.into(),
                k: k_in.into(),
                dnum: dnum_in.into(),
                dsize: dsize_in.into(),
                rank: rank.into(),
            };

            let ggsw_out_layout: GGSWLayout = GGSWLayout {
                n: n.into(),
                base2k: base2k_out.into(),
                k: k_out.into(),
                dnum: dnum_in.into(),
                dsize: dsize_in.into(),
                rank: rank.into(),
            };

            let tsk_layout: GGLWEToGGSWKeyLayout = GGLWEToGGSWKeyLayout {
                n: n.into(),
                base2k: base2k_key.into(),
                k: k_tsk.into(),
                dnum: dnum_ksk.into(),
                dsize: dsize.into(),
                rank: rank.into(),
            };

            let auto_key_layout: GGLWEToGGSWKeyLayout = GGLWEToGGSWKeyLayout {
                n: n.into(),
                base2k: base2k_key.into(),
                k: k_ksk.into(),
                dnum: dnum_ksk.into(),
                dsize: dsize.into(),
                rank: rank.into(),
            };

            let mut ct_in: GGSW<Vec<u8>> = GGSW::alloc_from_infos(&ggsw_in_layout);
            let mut ct_out: GGSW<Vec<u8>> = GGSW::alloc_from_infos(&ggsw_out_layout);
            let mut tsk: GGLWEToGGSWKey<Vec<u8>> = GGLWEToGGSWKey::alloc_from_infos(&tsk_layout);
            let mut auto_key: GLWEAutomorphismKey<Vec<u8>> = GLWEAutomorphismKey::alloc_from_infos(&auto_key_layout);
            let mut pt_scalar: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                GGSW::encrypt_sk_tmp_bytes(module, &ct_in)
                    | GLWEAutomorphismKey::encrypt_sk_tmp_bytes(module, &auto_key)
                    | GGLWEToGGSWKey::encrypt_sk_tmp_bytes(module, &tsk)
                    | GGSW::automorphism_tmp_bytes(module, &ct_out, &ct_in, &auto_key, &tsk),
            );

            let var_xs: f64 = 0.5;

            let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&ct_out);
            sk.fill_ternary_prob(var_xs, &mut source_xs);

            let mut sk_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc_from_infos(module, &sk);
            sk_prepared.prepare(module, &sk);

            auto_key.encrypt_sk(
                module,
                p,
                &sk,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );
            tsk.encrypt_sk(
                module,
                &sk,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            pt_scalar.fill_ternary_hw(0, n, &mut source_xs);

            ct_in.encrypt_sk(
                module,
                &pt_scalar,
                &sk_prepared,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            let mut auto_key_prepared: GLWEAutomorphismKeyPrepared<Vec<u8>, BE> =
                GLWEAutomorphismKeyPrepared::alloc_from_infos(module, &auto_key_layout);
            auto_key_prepared.prepare(module, &auto_key, scratch.borrow());

            let mut tsk_prepared: GGLWEToGGSWKeyPrepared<Vec<u8>, BE> = GGLWEToGGSWKeyPrepared::alloc_from_infos(module, &tsk);
            tsk_prepared.prepare(module, &tsk, scratch.borrow());

            ct_out.automorphism(
                module,
                &ct_in,
                &auto_key_prepared,
                &tsk_prepared,
                scratch.borrow(),
            );

            module.vec_znx_automorphism_inplace(p, &mut pt_scalar.as_vec_znx_mut(), 0, scratch.borrow());

            let max_noise = |col_j: usize| -> f64 {
                noise_ggsw_keyswitch(
                    n as f64,
                    base2k_key * dsize,
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

            ct_out.assert_noise(module, &sk_prepared, &pt_scalar, &max_noise);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn test_ggsw_automorphism_inplace<BE: Backend>(module: &Module<BE>)
where
    Module<BE>: GGSWEncryptSk<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GGSWAutomorphism<BE>
        + GGLWEToGGSWKeyPreparedFactory<BE>
        + GGLWEToGGSWKeyEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxAutomorphismInplace<BE>
        + GGSWNoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k_out: usize = 17;
    let base2k_key: usize = 13;
    let k_out: usize = 102;
    let max_dsize: usize = k_out.div_ceil(base2k_key);

    let p: i64 = -1;
    for rank in 1_usize..3 {
        for dsize in 1..max_dsize + 1 {
            let k_ksk: usize = k_out + base2k_key * dsize;
            let k_tsk: usize = k_ksk;

            let n: usize = module.n();
            let dnum_in: usize = k_out / base2k_out;
            let dnum_ksk: usize = k_out.div_ceil(base2k_key * dsize);
            let dsize_in: usize = 1;

            let ggsw_out_layout: GGSWLayout = GGSWLayout {
                n: n.into(),
                base2k: base2k_out.into(),
                k: k_out.into(),
                dnum: dnum_in.into(),
                dsize: dsize_in.into(),
                rank: rank.into(),
            };

            let tsk_layout: GGLWEToGGSWKeyLayout = GGLWEToGGSWKeyLayout {
                n: n.into(),
                base2k: base2k_key.into(),
                k: k_tsk.into(),
                dnum: dnum_ksk.into(),
                dsize: dsize.into(),
                rank: rank.into(),
            };

            let auto_key_layout: GGLWEToGGSWKeyLayout = GGLWEToGGSWKeyLayout {
                n: n.into(),
                base2k: base2k_key.into(),
                k: k_ksk.into(),
                dnum: dnum_ksk.into(),
                dsize: dsize.into(),
                rank: rank.into(),
            };

            let mut ct: GGSW<Vec<u8>> = GGSW::alloc_from_infos(&ggsw_out_layout);
            let mut tsk: GGLWEToGGSWKey<Vec<u8>> = GGLWEToGGSWKey::alloc_from_infos(&tsk_layout);
            let mut auto_key: GLWEAutomorphismKey<Vec<u8>> = GLWEAutomorphismKey::alloc_from_infos(&auto_key_layout);
            let mut pt_scalar: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                GGSW::encrypt_sk_tmp_bytes(module, &ct)
                    | GLWEAutomorphismKey::encrypt_sk_tmp_bytes(module, &auto_key)
                    | GGLWEToGGSWKey::encrypt_sk_tmp_bytes(module, &tsk)
                    | GGSW::automorphism_tmp_bytes(module, &ct, &ct, &auto_key, &tsk),
            );

            let var_xs: f64 = 0.5;

            let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&ct);
            sk.fill_ternary_prob(var_xs, &mut source_xs);

            let mut sk_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc_from_infos(module, &sk);
            sk_prepared.prepare(module, &sk);

            auto_key.encrypt_sk(
                module,
                p,
                &sk,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );
            tsk.encrypt_sk(
                module,
                &sk,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            pt_scalar.fill_ternary_hw(0, n, &mut source_xs);

            ct.encrypt_sk(
                module,
                &pt_scalar,
                &sk_prepared,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            let mut auto_key_prepared: GLWEAutomorphismKeyPrepared<Vec<u8>, BE> =
                GLWEAutomorphismKeyPrepared::alloc_from_infos(module, &auto_key_layout);
            auto_key_prepared.prepare(module, &auto_key, scratch.borrow());

            let mut tsk_prepared: GGLWEToGGSWKeyPrepared<Vec<u8>, BE> = GGLWEToGGSWKeyPrepared::alloc_from_infos(module, &tsk);
            tsk_prepared.prepare(module, &tsk, scratch.borrow());

            ct.automorphism_inplace(module, &auto_key_prepared, &tsk_prepared, scratch.borrow());

            module.vec_znx_automorphism_inplace(p, &mut pt_scalar.as_vec_znx_mut(), 0, scratch.borrow());

            let max_noise = |col_j: usize| -> f64 {
                noise_ggsw_keyswitch(
                    n as f64,
                    base2k_key * dsize,
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

            ct.assert_noise(module, &sk_prepared, &pt_scalar, &max_noise);
        }
    }
}
