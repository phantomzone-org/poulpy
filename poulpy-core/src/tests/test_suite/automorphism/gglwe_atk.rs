use poulpy_hal::{
    api::{ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAutomorphism, VecZnxSubScalarInplace},
    layouts::{Backend, GaloisElement, Module, Scratch, ScratchOwned},
    source::Source,
    test_suite::TestParams,
};

use crate::{
    GGLWENoise, GLWEAutomorphismKeyAutomorphism, GLWEAutomorphismKeyEncryptSk, ScratchTakeCore,
    encryption::SIGMA,
    layouts::{
        GGLWEInfos, GLWEAutomorphismKey, GLWEAutomorphismKeyLayout, GLWEAutomorphismKeyPreparedFactory, GLWEInfos, GLWESecret,
        GLWESecretPreparedFactory,
        prepared::{GLWEAutomorphismKeyPrepared, GLWESecretPrepared},
    },
    var_noise_gglwe_product_v2,
};

#[allow(clippy::too_many_arguments)]
pub fn test_gglwe_automorphism_key_automorphism<BE: Backend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWEAutomorphismKeyAutomorphism<BE>
        + VecZnxAutomorphism
        + GaloisElement
        + VecZnxSubScalarInplace
        + GLWESecretPreparedFactory<BE>
        + GGLWENoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k: usize = params.base2k;
    let in_base2k: usize = base2k - 1;
    let key_base2k: usize = base2k;
    let out_base2k: usize = in_base2k; // MUST BE SAME
    let k_in: usize = 4 * in_base2k + 1;
    let max_dsize: usize = k_in.div_ceil(key_base2k);
    let p0: i64 = -1;
    let p1: i64 = -5;
    for rank in 1_usize..3 {
        for dsize in 1..max_dsize + 1 {
            let k_ksk: usize = k_in + key_base2k * dsize;
            let k_out: usize = k_ksk; // Better capture noise.

            let n: usize = module.n();
            let dsize_in: usize = 1;

            let dnum_in: usize = k_in / in_base2k;
            let dnum_ksk: usize = k_in.div_ceil(key_base2k * dsize);

            let auto_key_in_infos: GLWEAutomorphismKeyLayout = GLWEAutomorphismKeyLayout {
                n: n.into(),
                base2k: in_base2k.into(),
                k: k_in.into(),
                dnum: dnum_in.into(),
                dsize: dsize_in.into(),
                rank: rank.into(),
            };

            let auto_key_out_infos: GLWEAutomorphismKeyLayout = GLWEAutomorphismKeyLayout {
                n: n.into(),
                base2k: out_base2k.into(),
                k: k_out.into(),
                dnum: dnum_in.into(),
                dsize: dsize_in.into(),
                rank: rank.into(),
            };

            let auto_key_apply_infos: GLWEAutomorphismKeyLayout = GLWEAutomorphismKeyLayout {
                n: n.into(),
                base2k: key_base2k.into(),
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
                    .max(GLWEAutomorphismKey::encrypt_sk_tmp_bytes(module, &auto_key_apply_infos))
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
            auto_key_in.encrypt_sk(module, p0, &sk, &mut source_xa, &mut source_xe, scratch.borrow());

            // gglwe_{s2}(s1) -> s1 -> s2
            auto_key_apply.encrypt_sk(module, p1, &sk, &mut source_xa, &mut source_xe, scratch.borrow());

            let mut auto_key_apply_prepared: GLWEAutomorphismKeyPrepared<Vec<u8>, BE> =
                GLWEAutomorphismKeyPrepared::alloc_from_infos(module, &auto_key_apply_infos);

            auto_key_apply_prepared.prepare(module, &auto_key_apply, scratch.borrow());

            // gglwe_{s1}(s0) (x) gglwe_{s2}(s1) = gglwe_{s2}(s0)
            auto_key_out.automorphism(module, &auto_key_in, &auto_key_apply_prepared, scratch.borrow());

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

            let max_noise: f64 = var_noise_gglwe_product_v2(
                module.n() as f64,
                k_ksk,
                dnum_ksk,
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
            .log2();

            for row in 0..auto_key_out.dnum().as_usize() {
                for col in 0..auto_key_out.rank().as_usize() {
                    let noise_have = auto_key_out
                        .key
                        .noise(module, row, col, &sk.data, &sk_auto_dft, scratch.borrow())
                        .std()
                        .log2();

                    assert!(noise_have < max_noise + 0.5, "{noise_have} > {}", max_noise + 0.5);
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn test_gglwe_automorphism_key_automorphism_inplace<BE: Backend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWEAutomorphismKeyAutomorphism<BE>
        + VecZnxAutomorphism
        + GaloisElement
        + VecZnxSubScalarInplace
        + GLWESecretPreparedFactory<BE>
        + GGLWENoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k: usize = params.base2k;
    let out_base2k: usize = base2k - 1;
    let key_base2k: usize = base2k;
    let k_out: usize = 4 * out_base2k + 1;
    let max_dsize: usize = k_out.div_ceil(key_base2k);

    let p0: i64 = -1;
    let p1: i64 = -5;
    for rank in 1_usize..3 {
        for dsize in 1..max_dsize + 1 {
            let k_ksk: usize = k_out + key_base2k * dsize;

            let n: usize = module.n();
            let dsize_in: usize = 1;

            let dnum_in: usize = k_out / out_base2k;
            let dnum_ksk: usize = k_out.div_ceil(key_base2k * dsize);

            let auto_key_layout: GLWEAutomorphismKeyLayout = GLWEAutomorphismKeyLayout {
                n: n.into(),
                base2k: out_base2k.into(),
                k: k_out.into(),
                dnum: dnum_in.into(),
                dsize: dsize_in.into(),
                rank: rank.into(),
            };

            let auto_key_apply_layout: GLWEAutomorphismKeyLayout = GLWEAutomorphismKeyLayout {
                n: n.into(),
                base2k: key_base2k.into(),
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
            auto_key.encrypt_sk(module, p0, &sk, &mut source_xa, &mut source_xe, scratch.borrow());

            // gglwe_{s2}(s1) -> s1 -> s2
            auto_key_apply.encrypt_sk(module, p1, &sk, &mut source_xa, &mut source_xe, scratch.borrow());

            let mut auto_key_apply_prepared: GLWEAutomorphismKeyPrepared<Vec<u8>, BE> =
                GLWEAutomorphismKeyPrepared::alloc_from_infos(module, &auto_key_apply_layout);

            auto_key_apply_prepared.prepare(module, &auto_key_apply, scratch.borrow());

            // gglwe_{s1}(s0) (x) gglwe_{s2}(s1) = gglwe_{s2}(s0)
            auto_key.automorphism_inplace(module, &auto_key_apply_prepared, scratch.borrow());

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

            let max_noise: f64 = var_noise_gglwe_product_v2(
                module.n() as f64,
                k_ksk,
                dnum_ksk,
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
            .log2();

            for row in 0..auto_key.dnum().as_usize() {
                for col in 0..auto_key.rank().as_usize() {
                    let noise_have = auto_key
                        .key
                        .noise(module, row, col, &sk.data, &sk_auto_dft, scratch.borrow())
                        .std()
                        .log2();

                    assert!(noise_have < max_noise + 0.5, "{noise_have} {}", max_noise + 0.5);
                }
            }
        }
    }
}
