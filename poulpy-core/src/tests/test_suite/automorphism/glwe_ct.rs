use poulpy_hal::{
    api::{ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAutomorphismInplace, VecZnxFillUniform},
    layouts::{Backend, Module, Scratch, ScratchOwned},
    source::Source,
};

use crate::{
    GLWEAutomorphism, GLWEAutomorphismKeyEncryptSk, GLWEDecrypt, GLWEEncryptSk, GLWENoise, GLWENormalize, ScratchTakeCore,
    encryption::SIGMA,
    layouts::{
        GLWE, GLWEAutomorphismKey, GLWEAutomorphismKeyLayout, GLWEAutomorphismKeyPreparedFactory, GLWELayout, GLWEPlaintext,
        GLWESecret, GLWESecretPreparedFactory,
        prepared::{GLWEAutomorphismKeyPrepared, GLWESecretPrepared},
    },
    var_noise_gglwe_product_v2,
};

pub fn test_glwe_automorphism<BE: Backend>(module: &Module<BE>)
where
    Module<BE>: GLWEEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxFillUniform
        + GLWEDecrypt<BE>
        + GLWEAutomorphism<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWENoise<BE>
        + VecZnxAutomorphismInplace<BE>
        + GLWENormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k_in: usize = 17;
    let base2k_key: usize = 13;
    let base2k_out: usize = 15;
    let k_in: usize = 102;
    let max_dsize: usize = k_in.div_ceil(base2k_key);
    let p: i64 = -5;
    for rank in 1_usize..3 {
        for dsize in 1..max_dsize + 1 {
            let k_ksk: usize = k_in + base2k_key * dsize;
            let k_out: usize = k_ksk; // Better capture noise.

            let n: usize = module.n();
            let dnum: usize = k_in.div_ceil(base2k_key * dsize);

            let ct_in_infos: GLWELayout = GLWELayout {
                n: n.into(),
                base2k: base2k_in.into(),
                k: k_in.into(),
                rank: rank.into(),
            };

            let ct_out_infos: GLWELayout = GLWELayout {
                n: n.into(),
                base2k: base2k_out.into(),
                k: k_out.into(),
                rank: rank.into(),
            };

            let autokey_infos: GLWEAutomorphismKeyLayout = GLWEAutomorphismKeyLayout {
                n: n.into(),
                base2k: base2k_key.into(),
                k: k_out.into(),
                rank: rank.into(),
                dnum: dnum.into(),
                dsize: dsize.into(),
            };

            let mut autokey: GLWEAutomorphismKey<Vec<u8>> = GLWEAutomorphismKey::alloc_from_infos(&autokey_infos);
            let mut ct_in: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&ct_in_infos);
            let mut ct_out: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&ct_out_infos);
            let mut pt_in: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&ct_in_infos);
            let mut pt_out: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&ct_out_infos);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            module.vec_znx_fill_uniform(base2k_in, &mut pt_in.data, 0, &mut source_xa);

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                GLWEAutomorphismKey::encrypt_sk_tmp_bytes(module, &autokey)
                    | GLWE::decrypt_tmp_bytes(module, &ct_out)
                    | GLWE::encrypt_sk_tmp_bytes(module, &ct_in)
                    | GLWE::automorphism_tmp_bytes(module, &ct_out, &ct_in, &autokey),
            );

            let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&ct_out);
            sk.fill_ternary_prob(0.5, &mut source_xs);

            let mut sk_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc_from_infos(module, &sk);
            sk_prepared.prepare(module, &sk);

            autokey.encrypt_sk(
                module,
                p,
                &sk,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            ct_in.encrypt_sk(
                module,
                &pt_in,
                &sk_prepared,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            let mut autokey_prepared: GLWEAutomorphismKeyPrepared<Vec<u8>, BE> =
                GLWEAutomorphismKeyPrepared::alloc_from_infos(module, &autokey_infos);
            autokey_prepared.prepare(module, &autokey, scratch.borrow());

            ct_out.automorphism(module, &ct_in, &autokey_prepared, scratch.borrow());

            let max_noise: f64 = var_noise_gglwe_product_v2(
                module.n() as f64,
                k_ksk,
                dnum,
                max_dsize,
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

            module.glwe_normalize(&mut pt_out, &pt_in, scratch.borrow());
            module.vec_znx_automorphism_inplace(p, &mut pt_out.data, 0, scratch.borrow());

            ct_out.assert_noise(module, &sk_prepared, &pt_out, max_noise + 1.0);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn test_glwe_automorphism_inplace<BE: Backend>(module: &Module<BE>)
where
    Module<BE>: GLWEEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxFillUniform
        + GLWEDecrypt<BE>
        + GLWEAutomorphism<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWENoise<BE>
        + VecZnxAutomorphismInplace<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k_out: usize = 17;
    let base2k_key: usize = 13;
    let k_out: usize = 102;
    let max_dsize: usize = k_out.div_ceil(base2k_key);

    let p = -5;
    for rank in 1_usize..3 {
        for dsize in 1..max_dsize + 1 {
            let k_ksk: usize = k_out + base2k_key * dsize;

            let n: usize = module.n();
            let dnum: usize = k_out.div_ceil(base2k_key * dsize);

            let ct_out_infos: GLWELayout = GLWELayout {
                n: n.into(),
                base2k: base2k_out.into(),
                k: k_out.into(),
                rank: rank.into(),
            };

            let autokey_infos: GLWEAutomorphismKeyLayout = GLWEAutomorphismKeyLayout {
                n: n.into(),
                base2k: base2k_key.into(),
                k: k_ksk.into(),
                rank: rank.into(),
                dnum: dnum.into(),
                dsize: dsize.into(),
            };

            let mut autokey: GLWEAutomorphismKey<Vec<u8>> = GLWEAutomorphismKey::alloc_from_infos(&autokey_infos);
            let mut ct: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&ct_out_infos);
            let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&ct_out_infos);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            module.vec_znx_fill_uniform(base2k_out, &mut pt_want.data, 0, &mut source_xa);

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                GLWEAutomorphismKey::encrypt_sk_tmp_bytes(module, &autokey)
                    | GLWE::decrypt_tmp_bytes(module, &ct)
                    | GLWE::encrypt_sk_tmp_bytes(module, &ct)
                    | GLWE::automorphism_tmp_bytes(module, &ct, &ct, &autokey),
            );

            let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&ct);
            sk.fill_ternary_prob(0.5, &mut source_xs);

            let mut sk_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc_from_infos(module, &sk);
            sk_prepared.prepare(module, &sk);

            autokey.encrypt_sk(
                module,
                p,
                &sk,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            ct.encrypt_sk(
                module,
                &pt_want,
                &sk_prepared,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            let mut autokey_prepared: GLWEAutomorphismKeyPrepared<Vec<u8>, BE> =
                GLWEAutomorphismKeyPrepared::alloc_from_infos(module, &autokey);
            autokey_prepared.prepare(module, &autokey, scratch.borrow());

            ct.automorphism_inplace(module, &autokey_prepared, scratch.borrow());

            let max_noise: f64 = var_noise_gglwe_product_v2(
                module.n() as f64,
                k_ksk,
                dnum,
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

            module.vec_znx_automorphism_inplace(p, &mut pt_want.data, 0, scratch.borrow());

            ct.assert_noise(module, &sk_prepared, &pt_want, max_noise + 1.0);
        }
    }
}
