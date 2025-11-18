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
        GLWESecret, GLWESecretPreparedFactory, LWEInfos,
        prepared::{GLWEAutomorphismKeyPrepared, GLWESecretPrepared},
    },
    var_noise_gglwe_product_v2,
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
    let base2k_in: usize = 17;
    let base2k_key: usize = 13;
    let base2k_out: usize = base2k_in; // MUST BE SAME
    let k_in: usize = 102;
    let max_dsize: usize = k_in.div_ceil(base2k_key);
    let p0: i64 = -1;
    let p1: i64 = -5;
    for rank in 1_usize..3 {
        for dsize in 1..max_dsize + 1 {
            let k_ksk: usize = k_in + base2k_key * dsize;
            let k_out: usize = k_ksk; // Better capture noise.

            let n: usize = module.n();
            let dsize_in: usize = 1;

            let dnum_in: usize = k_in / base2k_in;
            let dnum_ksk: usize = k_in.div_ceil(base2k_key * dsize);

            let auto_key_in_infos: GLWEAutomorphismKeyLayout = GLWEAutomorphismKeyLayout {
                n: n.into(),
                base2k: base2k_in.into(),
                k: k_in.into(),
                dnum: dnum_in.into(),
                dsize: dsize_in.into(),
                rank: rank.into(),
            };

            let auto_key_out_infos: GLWEAutomorphismKeyLayout = GLWEAutomorphismKeyLayout {
                n: n.into(),
                base2k: base2k_out.into(),
                k: k_out.into(),
                dnum: dnum_in.into(),
                dsize: dsize_in.into(),
                rank: rank.into(),
            };

            let auto_key_apply_infos: GLWEAutomorphismKeyLayout = GLWEAutomorphismKeyLayout {
                n: n.into(),
                base2k: base2k_key.into(),
                k: k_ksk.into(),
                dnum: dnum_ksk.into(),
                dsize: dsize.into(),
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
                    .max(GLWEAutomorphismKey::encrypt_sk_tmp_bytes(
                        module,
                        &auto_key_apply_infos,
                    ))
                    .max(GLWEAutomorphismKey::automorphism_tmp_bytes(
                        module,
                        &auto_key_out_infos,
                        &auto_key_in_infos,
                        &auto_key_apply_infos,
                    )),
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

            let mut pt_out: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&auto_key_out_infos);

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

            for col_i in 0..auto_key_out.rank_in().into() {
                for row_i in 0..auto_key_out.dnum().into() {
                    auto_key_out
                        .at(row_i, col_i)
                        .decrypt(module, &mut pt_out, &sk_auto_dft, scratch.borrow());

                    module.vec_znx_sub_scalar_inplace(
                        &mut pt_out.data,
                        0,
                        (dsize_in - 1) + row_i * dsize_in,
                        &sk.data,
                        col_i,
                    );

                    let noise_have: f64 = pt_out.data.stats(pt_out.base2k().into(), 0).std().log2();
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
                        rank as f64,
                    )
                    .sqrt()
                    .log2();

                    assert!(
                        noise_have < max_noise + 0.5,
                        "{noise_have} {}",
                        max_noise + 0.5
                    );
                }
            }
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
    let base2k_out: usize = 17;
    let base2k_key: usize = 13;
    let k_out: usize = 102;
    let max_dsize: usize = k_out.div_ceil(base2k_key);

    let p0: i64 = -1;
    let p1: i64 = -5;
    for rank in 1_usize..3 {
        for dsize in 1..max_dsize + 1 {
            let k_ksk: usize = k_out + base2k_key * dsize;

            let n: usize = module.n();
            let dsize_in: usize = 1;

            let dnum_in: usize = k_out / base2k_out;
            let dnum_ksk: usize = k_out.div_ceil(base2k_key * dsize);

            let auto_key_layout: GLWEAutomorphismKeyLayout = GLWEAutomorphismKeyLayout {
                n: n.into(),
                base2k: base2k_out.into(),
                k: k_out.into(),
                dnum: dnum_in.into(),
                dsize: dsize_in.into(),
                rank: rank.into(),
            };

            let auto_key_apply_layout: GLWEAutomorphismKeyLayout = GLWEAutomorphismKeyLayout {
                n: n.into(),
                base2k: base2k_key.into(),
                k: k_ksk.into(),
                dnum: dnum_ksk.into(),
                dsize: dsize.into(),
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

                    let noise_have: f64 = pt.data.stats(pt.base2k().into(), 0).std().log2();
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
                        rank as f64,
                    )
                    .sqrt()
                    .log2();

                    assert!(
                        noise_have < max_noise + 0.5,
                        "{noise_have} {}",
                        max_noise + 0.5
                    );
                });
            });
        }
    }
}
