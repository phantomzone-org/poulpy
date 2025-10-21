use poulpy_hal::{
    api::{ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAutomorphismInplace, VecZnxFillUniform},
    layouts::{Backend, Module, Scratch, ScratchOwned},
    source::Source,
};

use crate::{
    AutomorphismKeyEncryptSk, GLWEAutomorphism, GLWEDecrypt, GLWEEncryptSk, GLWENoise, ScratchTakeCore,
    encryption::SIGMA,
    layouts::{
        AutomorphismKey, AutomorphismKeyLayout, GLWE, GLWEAutomorphismKeyPreparedApi, GLWELayout, GLWEPlaintext, GLWESecret,
        GLWESecretPreparedApi,
        prepared::{GLWEAutomorphismKeyPrepared, GLWESecretPrepared},
    },
    noise::log2_std_noise_gglwe_product,
};

pub fn test_glwe_automorphism<BE: Backend>(module: &Module<BE>)
where
    Module<BE>: GLWEEncryptSk<BE>
        + GLWESecretPreparedApi<BE>
        + VecZnxFillUniform
        + GLWEDecrypt<BE>
        + GLWEAutomorphism<BE>
        + AutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedApi<BE>
        + GLWENoise<BE>
        + VecZnxAutomorphismInplace<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k: usize = 12;
    let k_in: usize = 60;
    let dsize: usize = k_in.div_ceil(base2k);
    let p: i64 = -5;
    for rank in 1_usize..3 {
        for di in 1..dsize + 1 {
            let k_ksk: usize = k_in + base2k * di;
            let k_out: usize = k_ksk; // Better capture noise.

            let n: usize = module.n();
            let dnum: usize = k_in.div_ceil(base2k * dsize);

            let ct_in_infos: GLWELayout = GLWELayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_in.into(),
                rank: rank.into(),
            };

            let ct_out_infos: GLWELayout = GLWELayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_out.into(),
                rank: rank.into(),
            };

            let autokey_infos: AutomorphismKeyLayout = AutomorphismKeyLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_out.into(),
                rank: rank.into(),
                dnum: dnum.into(),
                dsize: di.into(),
            };

            let mut autokey: AutomorphismKey<Vec<u8>> = AutomorphismKey::alloc_from_infos(&autokey_infos);
            let mut ct_in: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&ct_in_infos);
            let mut ct_out: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&ct_out_infos);
            let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&ct_out_infos);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            module.vec_znx_fill_uniform(base2k, &mut pt_want.data, 0, &mut source_xa);

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                AutomorphismKey::encrypt_sk_tmp_bytes(module, &autokey)
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
                &pt_want,
                &sk_prepared,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            let mut autokey_prepared: GLWEAutomorphismKeyPrepared<Vec<u8>, BE> =
                GLWEAutomorphismKeyPrepared::alloc_from_infos(module, &autokey_infos);
            autokey_prepared.prepare(module, &autokey, scratch.borrow());

            ct_out.automorphism(module, &ct_in, &autokey_prepared, scratch.borrow());

            let max_noise: f64 = log2_std_noise_gglwe_product(
                module.n() as f64,
                base2k * dsize,
                0.5,
                0.5,
                0f64,
                SIGMA * SIGMA,
                0f64,
                rank as f64,
                k_in,
                k_ksk,
            );

            module.vec_znx_automorphism_inplace(p, &mut pt_want.data, 0, scratch.borrow());

            ct_out.assert_noise(module, &sk_prepared, &pt_want, max_noise + 1.0);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn test_glwe_automorphism_inplace<BE: Backend>(module: &Module<BE>)
where
    Module<BE>: GLWEEncryptSk<BE>
        + GLWESecretPreparedApi<BE>
        + VecZnxFillUniform
        + GLWEDecrypt<BE>
        + GLWEAutomorphism<BE>
        + AutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedApi<BE>
        + GLWENoise<BE>
        + VecZnxAutomorphismInplace<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k: usize = 12;
    let k_out: usize = 60;
    let dsize: usize = k_out.div_ceil(base2k);
    let p = -5;
    for rank in 1_usize..3 {
        for di in 1..dsize + 1 {
            let k_ksk: usize = k_out + base2k * di;

            let n: usize = module.n();
            let dnum: usize = k_out.div_ceil(base2k * dsize);

            let ct_out_infos: GLWELayout = GLWELayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_out.into(),
                rank: rank.into(),
            };

            let autokey_infos: AutomorphismKeyLayout = AutomorphismKeyLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k_ksk.into(),
                rank: rank.into(),
                dnum: dnum.into(),
                dsize: di.into(),
            };

            let mut autokey: AutomorphismKey<Vec<u8>> = AutomorphismKey::alloc_from_infos(&autokey_infos);
            let mut ct: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&ct_out_infos);
            let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&ct_out_infos);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            module.vec_znx_fill_uniform(base2k, &mut pt_want.data, 0, &mut source_xa);

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                AutomorphismKey::encrypt_sk_tmp_bytes(module, &autokey)
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

            let max_noise: f64 = log2_std_noise_gglwe_product(
                module.n() as f64,
                base2k * dsize,
                0.5,
                0.5,
                0f64,
                SIGMA * SIGMA,
                0f64,
                rank as f64,
                k_out,
                k_ksk,
            );

            module.vec_znx_automorphism_inplace(p, &mut pt_want.data, 0, scratch.borrow());

            ct.assert_noise(module, &sk_prepared, &pt_want, max_noise + 1.0);
        }
    }
}
