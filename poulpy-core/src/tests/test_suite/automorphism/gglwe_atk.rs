use poulpy_hal::{
    api::{ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAutomorphism, VecZnxSubScalarInplace},
    layouts::{Backend, GaloisElement, Module, Scratch, ScratchOwned},
    source::Source,
};

use crate::{
    GLWEAutomorphismKeyAutomorphism, GLWEAutomorphismKeyEncryptSk, GLWEDecrypt, ScratchTakeCore,
    encryption::SIGMA,
    layouts::{
        GGLWEInfos, GLWEAutomorphismKey, GLWEAutomorphismKeyLayout, GLWEAutomorphismKeyPreparedFactory, GLWEPlaintext,
        GLWESecret, GLWESecretPreparedFactory,
        prepared::{GLWEAutomorphismKeyPrepared, GLWESecretPrepared},
    },
    noise::log2_std_noise_gglwe_product,
};

#[allow(clippy::too_many_arguments)]
pub fn test_gglwe_automorphism_key_automorphism<BE: Backend>(module: &Module<BE>)
where
    Module<BE>: GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWEAutomorphismKeyAutomorphism<BE>
        + VecZnxAutomorphism
        + GaloisElement
        + VecZnxSubScalarInplace
        + GLWESecretPreparedFactory<BE>
        + GLWEDecrypt<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k: usize = 12;
    let k_in: usize = 60;
    let k_out: usize = 40;
    let dsize: usize = k_in.div_ceil(base2k);
    let p0: i64 = -1;
    let p1: i64 = -5;
    for rank in 1_usize..3 {
        for di in 1..dsize + 1 {
            let k_apply: usize = (dsize + di) * base2k;

            let n: usize = module.n();
            let dsize_in: usize = 1;

            let dnum_in: usize = k_in / (base2k * di);
            let dnum_out: usize = k_out / (base2k * di);
            let dnum_apply: usize = k_in.div_ceil(base2k * di);

            let auto_key_in_infos: GLWEAutomorphismKeyLayout = GLWEAutomorphismKeyLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_in.into(),
                dnum: dnum_in.into(),
                dsize: dsize_in.into(),
                rank: rank.into(),
            };

            let auto_key_out_infos: GLWEAutomorphismKeyLayout = GLWEAutomorphismKeyLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_out.into(),
                dnum: dnum_out.into(),
                dsize: dsize_in.into(),
                rank: rank.into(),
            };

            let auto_key_apply_infos: GLWEAutomorphismKeyLayout = GLWEAutomorphismKeyLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_apply.into(),
                dnum: dnum_apply.into(),
                dsize: di.into(),
                rank: rank.into(),
            };

            let mut auto_key_in: GLWEAutomorphismKey<Vec<u8>> = GLWEAutomorphismKey::alloc_from_infos(&auto_key_in_infos);
            let mut auto_key_out: GLWEAutomorphismKey<Vec<u8>> = GLWEAutomorphismKey::alloc_from_infos(&auto_key_out_infos);
            let mut auto_key_apply: GLWEAutomorphismKey<Vec<u8>> = GLWEAutomorphismKey::alloc_from_infos(&auto_key_apply_infos);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                GLWEAutomorphismKey::encrypt_sk_tmp_bytes(module, &auto_key_in_infos)
                    | GLWEAutomorphismKey::encrypt_sk_tmp_bytes(module, &auto_key_apply_infos)
                    | GLWEAutomorphismKey::automorphism_tmp_bytes(
                        module,
                        &auto_key_out_infos,
                        &auto_key_in_infos,
                        &auto_key_apply_infos,
                    ),
            );

            let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&auto_key_in);
            sk.fill_ternary_prob(0.5, &mut source_xs);

            // gglwe_{s1}(s0) = s0 -> s1
            auto_key_in.encrypt_sk(
                module,
                p0,
                &sk,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            // gglwe_{s2}(s1) -> s1 -> s2
            auto_key_apply.encrypt_sk(
                module,
                p1,
                &sk,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            let mut auto_key_apply_prepared: GLWEAutomorphismKeyPrepared<Vec<u8>, BE> =
                GLWEAutomorphismKeyPrepared::alloc_from_infos(module, &auto_key_apply_infos);

            auto_key_apply_prepared.prepare(module, &auto_key_apply, scratch.borrow());

            // gglwe_{s1}(s0) (x) gglwe_{s2}(s1) = gglwe_{s2}(s0)
            auto_key_out.automorphism(
                module,
                &auto_key_in,
                &auto_key_apply_prepared,
                scratch.borrow(),
            );

            let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&auto_key_out_infos);

            let mut sk_auto: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&auto_key_out_infos);
            sk_auto.fill_zero(); // Necessary to avoid panic of unfilled sk
            for i in 0..rank {
                module.vec_znx_automorphism(
                    module.galois_element_inv(p0 * p1),
                    &mut sk_auto.data.as_vec_znx_mut(),
                    i,
                    &sk.data.as_vec_znx(),
                    i,
                );
            }

            let mut sk_auto_dft: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc_from_infos(module, &sk_auto);
            sk_auto_dft.prepare(module, &sk_auto);

            (0..auto_key_out.rank_in().into()).for_each(|col_i| {
                (0..auto_key_out.dnum().into()).for_each(|row_i| {
                    auto_key_out
                        .at(row_i, col_i)
                        .decrypt(module, &mut pt, &sk_auto_dft, scratch.borrow());

                    module.vec_znx_sub_scalar_inplace(
                        &mut pt.data,
                        0,
                        (dsize_in - 1) + row_i * dsize_in,
                        &sk.data,
                        col_i,
                    );

                    let noise_have: f64 = pt.data.stats(base2k, 0).std().log2();
                    let noise_want: f64 = log2_std_noise_gglwe_product(
                        n as f64,
                        base2k * di,
                        0.5,
                        0.5,
                        0f64,
                        SIGMA * SIGMA,
                        0f64,
                        rank as f64,
                        k_out,
                        k_apply,
                    );

                    assert!(
                        noise_have < noise_want + 0.5,
                        "{noise_have} {}",
                        noise_want + 0.5
                    );
                });
            });
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn test_gglwe_automorphism_key_automorphism_inplace<BE: Backend>(module: &Module<BE>)
where
    Module<BE>: GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWEAutomorphismKeyAutomorphism<BE>
        + VecZnxAutomorphism
        + GaloisElement
        + VecZnxSubScalarInplace
        + GLWESecretPreparedFactory<BE>
        + GLWEDecrypt<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k: usize = 12;
    let k_in: usize = 60;
    let dsize: usize = k_in.div_ceil(base2k);
    let p0: i64 = -1;
    let p1: i64 = -5;
    for rank in 1_usize..3 {
        for di in 1..dsize + 1 {
            let k_apply: usize = (dsize + di) * base2k;

            let n: usize = module.n();
            let dsize_in: usize = 1;

            let dnum_in: usize = k_in / (base2k * di);
            let dnum_apply: usize = k_in.div_ceil(base2k * di);

            let auto_key_layout: GLWEAutomorphismKeyLayout = GLWEAutomorphismKeyLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_in.into(),
                dnum: dnum_in.into(),
                dsize: dsize_in.into(),
                rank: rank.into(),
            };

            let auto_key_apply_layout: GLWEAutomorphismKeyLayout = GLWEAutomorphismKeyLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_apply.into(),
                dnum: dnum_apply.into(),
                dsize: di.into(),
                rank: rank.into(),
            };

            let mut auto_key: GLWEAutomorphismKey<Vec<u8>> = GLWEAutomorphismKey::alloc_from_infos(&auto_key_layout);
            let mut auto_key_apply: GLWEAutomorphismKey<Vec<u8>> = GLWEAutomorphismKey::alloc_from_infos(&auto_key_apply_layout);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                GLWEAutomorphismKey::encrypt_sk_tmp_bytes(module, &auto_key)
                    | GLWEAutomorphismKey::encrypt_sk_tmp_bytes(module, &auto_key_apply)
                    | GLWEAutomorphismKey::automorphism_tmp_bytes(module, &auto_key, &auto_key, &auto_key_apply),
            );

            let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&auto_key);
            sk.fill_ternary_prob(0.5, &mut source_xs);

            // gglwe_{s1}(s0) = s0 -> s1
            auto_key.encrypt_sk(
                module,
                p0,
                &sk,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            // gglwe_{s2}(s1) -> s1 -> s2
            auto_key_apply.encrypt_sk(
                module,
                p1,
                &sk,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            let mut auto_key_apply_prepared: GLWEAutomorphismKeyPrepared<Vec<u8>, BE> =
                GLWEAutomorphismKeyPrepared::alloc_from_infos(module, &auto_key_apply_layout);

            auto_key_apply_prepared.prepare(module, &auto_key_apply, scratch.borrow());

            // gglwe_{s1}(s0) (x) gglwe_{s2}(s1) = gglwe_{s2}(s0)
            auto_key.automorphism_inplace(module, &auto_key_apply_prepared, scratch.borrow());

            let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&auto_key);

            let mut sk_auto: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&auto_key);
            sk_auto.fill_zero(); // Necessary to avoid panic of unfilled sk

            for i in 0..rank {
                module.vec_znx_automorphism(
                    module.galois_element_inv(p0 * p1),
                    &mut sk_auto.data.as_vec_znx_mut(),
                    i,
                    &sk.data.as_vec_znx(),
                    i,
                );
            }

            let mut sk_auto_dft: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc_from_infos(module, &sk_auto);
            sk_auto_dft.prepare(module, &sk_auto);

            (0..auto_key.rank_in().into()).for_each(|col_i| {
                (0..auto_key.dnum().into()).for_each(|row_i| {
                    auto_key
                        .at(row_i, col_i)
                        .decrypt(module, &mut pt, &sk_auto_dft, scratch.borrow());
                    module.vec_znx_sub_scalar_inplace(
                        &mut pt.data,
                        0,
                        (dsize_in - 1) + row_i * dsize_in,
                        &sk.data,
                        col_i,
                    );

                    let noise_have: f64 = pt.data.stats(base2k, 0).std().log2();
                    let noise_want: f64 = log2_std_noise_gglwe_product(
                        n as f64,
                        base2k * di,
                        0.5,
                        0.5,
                        0f64,
                        SIGMA * SIGMA,
                        0f64,
                        rank as f64,
                        k_in,
                        k_apply,
                    );

                    assert!(
                        noise_have < noise_want + 0.5,
                        "{noise_have} {}",
                        noise_want + 0.5
                    );
                });
            });
        }
    }
}
