use std::collections::HashMap;

use poulpy_hal::{
    api::{ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxFillUniform, VecZnxNormalizeInplace, VecZnxSubInplace},
    layouts::{Backend, Module, Scratch, ScratchOwned, ZnxView, ZnxViewMut},
    source::Source,
};

use crate::{
    AutomorphismKeyEncryptSk, GLWEDecrypt, GLWEEncryptSk, ScratchTakeCore,
    encryption::SIGMA,
    glwe_trace::GLWETrace,
    layouts::{
        AutomorphismKey, AutomorphismKeyLayout, AutomorphismKeyPrepare, AutomorphismKeyPreparedAlloc, GLWE, GLWELayout,
        GLWEPlaintext, GLWESecret, GLWESecretPrepare, GLWESecretPreparedAlloc, LWEInfos,
        prepared::{AutomorphismKeyPrepared, GLWESecretPrepared},
    },
    noise::var_noise_gglwe_product,
};

pub fn test_glwe_trace_inplace<BE: Backend>(module: &Module<BE>)
where
    Module<BE>: GLWETrace<BE>
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + AutomorphismKeyEncryptSk<BE>
        + AutomorphismKeyPrepare<BE>
        + AutomorphismKeyPreparedAlloc<BE>
        + VecZnxFillUniform
        + GLWESecretPrepare<BE>
        + GLWESecretPreparedAlloc<BE>
        + VecZnxSubInplace
        + VecZnxNormalizeInplace<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k: usize = 8;
    let k: usize = 54;

    for rank in 1_usize..3 {
        let n: usize = module.n();
        let k_autokey: usize = k + base2k;

        let dsize: usize = 1;
        let dnum: usize = k.div_ceil(base2k * dsize);

        let glwe_out_infos: GLWELayout = GLWELayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k.into(),
            rank: rank.into(),
        };

        let key_infos: AutomorphismKeyLayout = AutomorphismKeyLayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k_autokey.into(),
            rank: rank.into(),
            dsize: dsize.into(),
            dnum: dnum.into(),
        };

        let mut glwe_out: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_out_infos);
        let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_out_infos);
        let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_out_infos);

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            GLWE::encrypt_sk_tmp_bytes(module, &glwe_out_infos)
                | GLWE::decrypt_tmp_bytes(module, &glwe_out_infos)
                | AutomorphismKey::encrypt_sk_tmp_bytes(module, &key_infos)
                | GLWE::trace_tmp_bytes(module, &glwe_out_infos, &glwe_out_infos, &key_infos),
        );

        let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&glwe_out_infos);
        sk.fill_ternary_prob(0.5, &mut source_xs);

        let mut sk_dft: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc_from_infos(module, &sk);
        sk_dft.prepare(module, &sk);

        let mut data_want: Vec<i64> = vec![0i64; n];

        data_want
            .iter_mut()
            .for_each(|x| *x = source_xa.next_i64() & 0xFF);

        module.vec_znx_fill_uniform(base2k, &mut pt_have.data, 0, &mut source_xa);

        glwe_out.encrypt_sk(
            module,
            &pt_have,
            &sk_dft,
            &mut source_xa,
            &mut source_xe,
            scratch.borrow(),
        );

        let mut auto_keys: HashMap<i64, AutomorphismKeyPrepared<Vec<u8>, BE>> = HashMap::new();
        let gal_els: Vec<i64> = GLWE::trace_galois_elements(module);
        let mut tmp: AutomorphismKey<Vec<u8>> = AutomorphismKey::alloc_from_infos(&key_infos);
        gal_els.iter().for_each(|gal_el| {
            tmp.encrypt_sk(
                module,
                *gal_el,
                &sk,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );
            let mut atk_prepared: AutomorphismKeyPrepared<Vec<u8>, BE> = AutomorphismKeyPrepared::alloc_from_infos(module, &tmp);
            atk_prepared.prepare(module, &tmp, scratch.borrow());
            auto_keys.insert(*gal_el, atk_prepared);
        });

        glwe_out.trace_inplace(module, 0, 5, &auto_keys, scratch.borrow());
        glwe_out.trace_inplace(module, 5, module.log_n(), &auto_keys, scratch.borrow());

        (0..pt_want.size()).for_each(|i| pt_want.data.at_mut(0, i)[0] = pt_have.data.at(0, i)[0]);

        glwe_out.decrypt(module, &mut pt_have, &sk_dft, scratch.borrow());

        module.vec_znx_sub_inplace(&mut pt_want.data, 0, &pt_have.data, 0);
        module.vec_znx_normalize_inplace(base2k, &mut pt_want.data, 0, scratch.borrow());

        let noise_have: f64 = pt_want.std().log2();

        let mut noise_want: f64 = var_noise_gglwe_product(
            n as f64,
            base2k,
            0.5,
            0.5,
            1.0 / 12.0,
            SIGMA * SIGMA,
            0.0,
            rank as f64,
            k,
            k_autokey,
        );
        noise_want += SIGMA * SIGMA * (-2.0 * (k) as f64).exp2();
        noise_want += n as f64 * 1.0 / 12.0 * 0.5 * rank as f64 * (-2.0 * (k) as f64).exp2();
        noise_want = noise_want.sqrt().log2();

        assert!(
            (noise_have - noise_want).abs() < 1.0,
            "{noise_have} > {noise_want}"
        );
    }
}
